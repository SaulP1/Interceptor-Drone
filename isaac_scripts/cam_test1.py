


"""
cam_test1.py — Isaac Sim world: PX4 Iris drone + Owl camera + red block.

Spawns one PX4-backed Iris drone with an Owl camera attached to the body,
adds a red block target to the scene, and publishes camera frames to ROS2.

Changes from original:
  - Added ROS2CameraGraph so the Owl camera publishes to ROS2 topics
  - Added ROS2Backend so drone state is published to ROS2
  - Camera topics: /owl/rgb, /owl/depth, /owl/camera_info

Referenced from: Seabird's init_scene.py
  - ROS2CameraGraph setup pattern (creates render product + ROS2 publishers)
  - ROS2Backend for drone state publishing
  - rclpy initialization guard for Script Editor re-runs
  Adapted: different camera (Owl vs ZED), different topic names,
  different scene (Gridroom vs marina), no buoys.

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
from pxr import UsdGeom, Gf, UsdShade, Sdf
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# === NEW: Import ROS2 camera graph and backend ===
# ROS2CameraGraph: takes a camera prim and publishes its images to ROS2 topics
# ROS2Backend: publishes drone state (pose, velocity, IMU) to ROS2 topics
# Referenced from: Seabird's init_scene.py uses both of these
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Load environment
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # === NEW: Initialize rclpy for ROS2Backend ===
        # ROS2Backend creates a ROS2 node internally, which needs rclpy.
        # Guard against double-init if this script is re-run.
        # Referenced from: Seabird's init_scene.py rclpy init guard
        import rclpy
        try:
            rclpy.init()
            print("[init] rclpy initialized")
        except RuntimeError:
            print("[init] rclpy already initialized — reusing")

        # === NEW: Create ROS2 camera graph ===
        # This tells Isaac Sim: "render this camera prim and publish
        # the images as ROS2 topics." Without this, the camera exists
        # visually but nothing sends its frames over ROS2.
        #
        # camera_prim_path is RELATIVE to the vehicle prim — Pegasus
        # prepends the vehicle path automatically.
        #
        # Referenced from: Seabird's init_scene.py ROS2CameraGraph setup
        cam_graph = ROS2CameraGraph(
            camera_prim_path="body/owl/camera",
            config={
                "resolution": [640, 480],
                "types": ["rgb", "depth", "camera_info"],
                "namespace": "",
                "topic": "/owl",
                "tf_frame_id": "owl_camera",
            }
        )

        # PX4 backend config
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )

        # === UPDATED: Add ROS2Backend + camera graph to drone config ===
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

        # Attach Owl camera under the drone body
        self.attach_owl_camera(
            body_path="/World/quadrotor/body",
            owl_name="owl",
            translation=(0.12, 0.0, 0.03),
            rotation_xyz_deg=(0.0, 0.0,0.0),
        )

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
            print("[INFO] Camera transform reset to match Owl parent")
        else:
            print("[ERROR] Could not find camera prim to reset transform")

        cam = UsdGeom.Camera(cam_prim)
        cam.GetFocalLengthAttr().Set(5.0)
        cam.GetHorizontalApertureAttr().Set(10.0)
        cam.GetVerticalApertureAttr().Set(7.5)
        cam.GetClippingRangeAttr().Set((0.1, 200.0))
        print("[INFO] Camera lens set to wide-angle FOV (~90°)")

        # Add the red block to the scene
        self.add_red_block(
            prim_path="/red_block_01",
            translation=(2.0, 0.0, 0.5),
            scale=(0.3, 0.3, 0.3),
        )

        # Verify that the camera path exists
        self.verify_camera_path("/World/quadrotor/body/owl/camera")

        # Initialize articulations/sensors
        self.world.reset()

        self.stop_sim = False

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

        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*translation))

        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(*rotation_xyz_deg))

        print(f"[INFO] Owl camera attached at: {owl_prim_path}")

        for child in owl_prim.GetChildren():
            print(f"[INFO] Owl child prim: {child.GetPath()}")

    def add_red_block(
        self,
        prim_path: str = "/red_block_01",
        translation=(-0.07, -0.15, 0.0235),
        scale=(0.025, 0.025, 0.025),
    ):
        stage = omni.usd.get_context().get_stage()

        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))

        material_path = f"{prim_path}/Looks/RedMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.0, 0.0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

        print(f"[INFO] Red block created at: {prim_path}")

    def verify_camera_path(self, camera_path: str):
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)

        if camera_prim.IsValid():
            print(f"[INFO] Camera prim confirmed at: {camera_path}")
        else:
            raise RuntimeError(f"[ERROR] Camera prim not found at: {camera_path}")

    def run(self):
        self.timeline.play()

        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()



