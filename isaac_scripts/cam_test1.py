
"""
cam_test1.py — Isaac Sim world: PX4 Iris drone + Owl camera + red block.

Red block has physics. When intercept_static.py publishes to /interceptor/hit,
the block turns green and falls to the ground. Press Stop then Play in Isaac
Sim to reset the target back to red at its original position.

Referenced from: world_static_target.py (hit detection, color change, gravity toggle, reset)
Referenced from: Seabird's init_scene.py (ROS2CameraGraph, ROS2Backend, rclpy init guard)

Run:
    cd ~/interceptor/isaac_scripts
    isaac_run cam_test1.py
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import omni.timeline
import omni.usd
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.nucleus import get_assets_root_path
from pxr import UsdGeom, Gf, UsdShade, Sdf, UsdPhysics, PhysxSchema
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend

import rclpy
from std_msgs.msg import Bool

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

TARGET_PRIM_PATH = "/red_block_01"
TARGET_POSITION = (15.0, 15.0, 8.0)
TARGET_SCALE = (2.0, 2.0, 2.0)
COLOR_RED = Gf.Vec3f(1.0, 0.0, 0.0)
COLOR_GREEN = Gf.Vec3f(0.0, 1.0, 0.0)


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Initialize rclpy
        try:
            rclpy.init()
            print("[init] rclpy initialized")
        except RuntimeError:
            print("[init] rclpy already initialized — reusing")

        # ROS2 node for subscribing to hit signal
        self.ros_node = rclpy.create_node("world_hit_listener")
        self.hit_received = False
        self.ros_node.create_subscription(
            Bool, "/interceptor/hit", self._on_hit, 10
        )
        print("[ros2] Subscribing to /interceptor/hit")

        # Camera graph
        cam_graph = ROS2CameraGraph(
            camera_prim_path="body/owl/camera",
            config={
                "resolution": [1920, 1080],
                "types": ["rgb", "depth", "camera_info"],
                "namespace": "",
                "topic": "/owl",
                "tf_frame_id": "owl_camera",
            }
        )

        # PX4 backend
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )

        config_multirotor = MultirotorConfig()
        config_multirotor.backends = [
            PX4MavlinkBackend(mavlink_config),
            ROS2Backend(vehicle_id=0, num_rotors=4),
        ]
        config_multirotor.graphs = [cam_graph]

        Multirotor(
            "/World/quadrotor",
            ROBOTS["Iris"],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Attach Owl camera
        self.attach_owl_camera(
            body_path="/World/quadrotor/body",
            owl_name="owl",
            translation=(0.12, 0.0, 0.03),
            rotation_xyz_deg=(0.0, 0.0, 0.0),
        )

        # Fix camera orientation
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/quadrotor/body/owl/camera")
        if cam_prim.IsValid():
            xf = UsdGeom.Xformable(cam_prim)
            xf.ClearXformOpOrder()
            cam_prim.RemoveProperty("xformOp:translate")
            cam_prim.RemoveProperty("xformOp:orient")
            cam_prim.RemoveProperty("xformOp:rotateXYZ")
            cam_prim.RemoveProperty("xformOp:scale")
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 270.0))
            print("[INFO] Camera transform set to (90, 0, 270)")

        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(5.0)
        cam.GetHorizontalApertureAttr().Set(10.0)
        cam.GetVerticalApertureAttr().Set(7.5)
        cam.GetClippingRangeAttr().Set((0.1, 200.0))
        print("[INFO] Camera lens set to wide-angle FOV (~90)")

        # Add red block WITH PHYSICS
        self.add_red_block(
            prim_path=TARGET_PRIM_PATH,
            translation=TARGET_POSITION,
            scale=TARGET_SCALE,
        )

        self.verify_camera_path("/World/quadrotor/body/owl/camera")

        self.world.reset()

        # State tracking
        self.stop_sim = False
        self._was_playing = False

        print(f"\n[world] Target at {TARGET_POSITION}")
        print("[world] Waiting for /interceptor/hit to trigger green + fall")
        print("[world] Press STOP then PLAY in Isaac Sim to reset target\n")

    def _on_hit(self, msg):
        """Called when intercept_static.py declares interception."""
        if msg.data and not self.hit_received:
            self.hit_received = True
            print("\n[world] *** HIT SIGNAL RECEIVED ***")

    def _process_hit(self):
        """Turn target green and enable gravity so it falls."""
        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
        if not target_prim.IsValid():
            return

        # Change color to green
        shader_path = f"{TARGET_PRIM_PATH}/Looks/RedMaterial/Shader"
        shader_prim = stage.GetPrimAtPath(shader_path)
        if shader_prim.IsValid():
            shader = UsdShade.Shader(shader_prim)
            shader.GetInput("diffuseColor").Set(COLOR_GREEN)
            print("[world] Target color → GREEN")

        # Enable gravity so block falls
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(target_prim)
        if physx_rb:
            physx_rb.GetDisableGravityAttr().Set(False)
            print("[world] Target gravity → ON (falling)")

    def _reset_target(self):
        """Reset target to red, original position, gravity off."""
        stage = omni.usd.get_context().get_stage()
        target_prim = stage.GetPrimAtPath(TARGET_PRIM_PATH)
        if not target_prim.IsValid():
            return

        # Reset position
        xform = UsdGeom.Xformable(target_prim)
        ops = xform.GetOrderedXformOps()
        if ops:
            ops[0].Set(Gf.Vec3d(*TARGET_POSITION))

        # Reset color to red
        shader_path = f"{TARGET_PRIM_PATH}/Looks/RedMaterial/Shader"
        shader_prim = stage.GetPrimAtPath(shader_path)
        if shader_prim.IsValid():
            shader = UsdShade.Shader(shader_prim)
            shader.GetInput("diffuseColor").Set(COLOR_RED)

        # Disable gravity
        physx_rb = PhysxSchema.PhysxRigidBodyAPI(target_prim)
        if physx_rb:
            physx_rb.GetDisableGravityAttr().Set(True)

        self.hit_received = False
        print("[world] Target RESET — red, original position, gravity off")

    def attach_owl_camera(
        self,
        body_path: str,
        owl_name: str = "owl",
        translation=(0.12, 0.0, 0.03),
        rotation_xyz_deg=(0.0, 0.0, 0.0),
    ):
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not find Isaac Sim assets root.")

        owl_usd = assets_root + "/Isaac/Sensors/LeopardImaging/Owl/owl.usd"
        owl_prim_path = f"{body_path}/{owl_name}"
        add_reference_to_stage(usd_path=owl_usd, prim_path=owl_prim_path)

        stage = omni.usd.get_context().get_stage()
        owl_prim = stage.GetPrimAtPath(owl_prim_path)
        if not owl_prim.IsValid():
            raise RuntimeError(f"Failed to create Owl camera at {owl_prim_path}")

        xform = UsdGeom.Xformable(owl_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz_deg))

        print(f"[INFO] Owl camera attached at: {owl_prim_path}")
        for child in owl_prim.GetChildren():
            print(f"[INFO] Owl child prim: {child.GetPath()}")

    def add_red_block(
        self,
        prim_path: str = "/red_block_01",
        translation=(2.0, 0.0, 0.5),
        scale=(0.3, 0.3, 0.3),
    ):
        """
        Create red block WITH physics — RigidBody, Collision, Mass.
        Gravity starts OFF. Enabled when hit is received so block falls.
        Referenced from: world_static_target.py add_red_block()
        """
        stage = omni.usd.get_context().get_stage()

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))

        # Red material
        material_path = f"{prim_path}/Looks/RedMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0,0.0,0.0))


        shader.CreateInput("emissiveColor" , Sdf.ValueTypeNames.Color3f).Set((1.0,0.0,0.0))

        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

        # Physics — referenced from world_static_target.py
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass_api.CreateMassAttr().Set(0.5)

        rb = UsdPhysics.RigidBodyAPI(cube.GetPrim())
        rb.CreateRigidBodyEnabledAttr().Set(True)

        # Gravity OFF initially — turns ON when hit
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim())
        physx_rb.CreateDisableGravityAttr().Set(True)

        print(f"[INFO] Red block with physics at: {prim_path}")

    def verify_camera_path(self, camera_path: str):
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        if camera_prim.IsValid():
            print(f"[INFO] Camera prim confirmed at: {camera_path}")
        else:
            raise RuntimeError(f"[ERROR] Camera prim not found at: {camera_path}")

    def run(self):
        self.timeline.play()
        self._was_playing = True

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

            # Process ROS2 callbacks (check for hit signal)
            rclpy.spin_once(self.ros_node, timeout_sec=0)

            # Check play/stop transitions for reset
            is_playing = self.timeline.is_playing()
            if self._was_playing and not is_playing:
                # User pressed STOP — reset will happen on next PLAY
                pass
            if not self._was_playing and is_playing:
                # User pressed PLAY after STOP — reset target
                self._reset_target()
            self._was_playing = is_playing

            # Process hit if received
            if self.hit_received and is_playing:
                self._process_hit()
                self.hit_received = False  # only process once
                self._hit_processed = True

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        self.ros_node.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()