"""
world_moving_target.py — Isaac Sim world: PX4 Iris drone + moving red block.

Same as cam_test1.py but the red block moves back and forth along Y axis.
Block starts at (2.0, 0.0, 0.5) and oscillates Y from -2.0 to 2.0.
Full cycle (left to right to left) takes ~8 seconds (4 seconds each way).

Also publishes the target's position and velocity to ROS2 topics so the
autonomous flight script knows where the target is and where it's heading.

Referenced from: cam_test1.py (base scene setup)
Referenced from: Seabird's init_scene.py (ROS2Backend, ROS2CameraGraph)

Run:
    cd ~/interceptor/isaac_scripts
    isaac_run world_moving_target.py
"""

import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import math
import time
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
from pegasus.simulator.logic.graphs.ros2_camera_graph import ROS2CameraGraph
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend

# ROS2 for publishing target state
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped


# ═══════════════════════════════════════════════════════════════
# TARGET MOTION PARAMETERS
# ═══════════════════════════════════════════════════════════════

TARGET_X = 2.0          # fixed X position (in front of drone)
TARGET_Z = 0.5          # fixed height
TARGET_Y_MIN = -2.0     # left extent
TARGET_Y_MAX = 2.0      # right extent
TARGET_PERIOD_S = 8.0   # full cycle time (left-right-left)

# Derived: speed = total_distance / half_period = 4.0 / 4.0 = 1.0 m/s
TARGET_Y_RANGE = TARGET_Y_MAX - TARGET_Y_MIN
TARGET_SPEED = TARGET_Y_RANGE / (TARGET_PERIOD_S / 2.0)


class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()

        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Load environment
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Initialize rclpy
        try:
            rclpy.init()
            print("[init] rclpy initialized")
        except RuntimeError:
            print("[init] rclpy already initialized — reusing")

        # Create ROS2 node for publishing target state
        self.ros_node = rclpy.create_node("moving_target_publisher")
        self.target_pose_pub = self.ros_node.create_publisher(
            PoseStamped, "/interceptor/target_pose", 10
        )
        self.target_vel_pub = self.ros_node.create_publisher(
            TwistStamped, "/interceptor/target_velocity", 10
        )
        print("[ros2] Publishing target state to /interceptor/target_pose")
        print("[ros2] Publishing target velocity to /interceptor/target_velocity")

        # Camera graph
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

        # PX4 backend
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": self.pg.px4_path,
                "px4_vehicle_model": self.pg.px4_default_airframe,
            }
        )

        # Drone config
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

        # Attach Owl camera (same as cam_test1.py)
        self.attach_owl_camera(
            body_path="/World/quadrotor/body",
            owl_name="owl",
            translation=(0.12, 0.0, 0.03),
            rotation_xyz_deg=(0.0, 0.0, 0.0),
        )

        # Override Owl camera transform (same fix as cam_test1.py)
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

        # Create the moving red block
        self.add_red_block(
            prim_path="/World/target_block",
            translation=(TARGET_X, 0.0, TARGET_Z),
            scale=(0.3, 0.3, 0.3),
        )

        # Store reference to the target prim for animation
        self.target_prim = stage.GetPrimAtPath("/World/target_block")
        self.target_xform = UsdGeom.Xformable(self.target_prim)

        # Initialize
        self.world.reset()
        self.stop_sim = False
        self.start_time = None

        print(f"\n[target] Moving red block:")
        print(f"  Position: X={TARGET_X}, Z={TARGET_Z}")
        print(f"  Y range: {TARGET_Y_MIN} to {TARGET_Y_MAX}")
        print(f"  Period: {TARGET_PERIOD_S}s (speed: {TARGET_SPEED:.1f} m/s)")

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

    def add_red_block(
        self,
        prim_path: str,
        translation=(2.0, 0.0, 0.5),
        scale=(0.3, 0.3, 0.3),
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
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(1.0, 0.0, 0.0)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        material.CreateSurfaceOutput().ConnectToSource(
            shader.ConnectableAPI(), "surface"
        )
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)

        print(f"[INFO] Red block created at: {prim_path}")

    def _update_target(self):
        """
        Move the red block along Y axis using a triangle wave.

        Triangle wave gives constant speed back and forth:
          - Linear motion from Y_MIN to Y_MAX in half the period
          - Linear motion from Y_MAX to Y_MIN in the other half
          - Speed is constant (no acceleration/deceleration)
        """
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time

        # Triangle wave: goes 0→1→0→1... over the period
        # fmod gives position in current cycle
        cycle_pos = math.fmod(elapsed, TARGET_PERIOD_S)
        half_period = TARGET_PERIOD_S / 2.0

        if cycle_pos < half_period:
            # Moving from Y_MIN to Y_MAX
            fraction = cycle_pos / half_period
            current_y = TARGET_Y_MIN + fraction * TARGET_Y_RANGE
            velocity_y = TARGET_SPEED
        else:
            # Moving from Y_MAX to Y_MIN
            fraction = (cycle_pos - half_period) / half_period
            current_y = TARGET_Y_MAX - fraction * TARGET_Y_RANGE
            velocity_y = -TARGET_SPEED

        # Update the USD prim position
        translate_op = self.target_xform.GetOrderedXformOps()[0]
        translate_op.Set(Gf.Vec3d(TARGET_X, current_y, TARGET_Z))

        # Publish position to ROS2
        pose_msg = PoseStamped()
        now = self.ros_node.get_clock().now().to_msg()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = TARGET_X
        pose_msg.pose.position.y = current_y
        pose_msg.pose.position.z = TARGET_Z
        pose_msg.pose.orientation.w = 1.0
        self.target_pose_pub.publish(pose_msg)

        # Publish velocity to ROS2
        vel_msg = TwistStamped()
        vel_msg.header.stamp = now
        vel_msg.header.frame_id = "world"
        vel_msg.twist.linear.x = 0.0
        vel_msg.twist.linear.y = velocity_y
        vel_msg.twist.linear.z = 0.0
        self.target_vel_pub.publish(vel_msg)

    def run(self):
        self.timeline.play()

        frame_count = 0
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)

            # Update target position every frame
            self._update_target()

            # Process ROS2 callbacks
            rclpy.spin_once(self.ros_node, timeout_sec=0)

            # Log periodically
            frame_count += 1
            if frame_count % 300 == 0:
                print(f"[target] Frame {frame_count} — target publishing")

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