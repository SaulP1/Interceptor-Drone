#!/usr/bin/env python3
"""
keyboard_fly.py — Fly the drone with WASD using MAVROS.

Publishes position setpoints to /mavros/setpoint_position/local.
Reads current position and adds offsets based on key presses.

Referenced from: Seabird's keyboard_controller.py
  - Same concept: keyboard input → velocity/position commands
  - Adapted: uses MAVROS (PoseStamped) instead of MAVSDK (VelocityBodyYawspeed)
  - Uses position offsets instead of velocity for simplicity with MAVROS

Controls:
    W / S     — forward / backward (X axis)
    A / D     — left / right (Y axis)
    R / F     — up / down (Z axis)
    T         — takeoff to 1.0m
    L         — land (returns to ground)
    Q         — quit

Requires: PX4 + MAVROS running. Will set OFFBOARD mode and arm automatically.

Run:
    source /opt/ros/humble/setup.bash
    python3 ~/interceptor/perception/keyboard_fly.py
"""

import sys
import os
import tty
import termios
import select
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode


class KeyboardFlyer(Node):
    def __init__(self):
        super().__init__("keyboard_flyer")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Current drone position (from MAVROS)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0

        # Target position (what we command)
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0

        # State
        self.current_mode = ""
        self.armed = False
        self.flying = False

        # Step size — how far each keypress moves (meters)
        self.step = 0.3

        # Subscribe to current pose
        self.create_subscription(
            PoseStamped, "/mavros/local_position/pose",
            self._on_pose, qos
        )

        # Subscribe to state
        self.create_subscription(
            State, "/mavros/state",
            self._on_state, qos
        )

        # Publisher for setpoints
        self.setpoint_pub = self.create_publisher(
            PoseStamped, "/mavros/setpoint_position/local", qos
        )

        # Service clients for arming and mode
        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")

        # Publish setpoints at 20Hz (must be > 2Hz for PX4 OFFBOARD)
        self.timer = self.create_timer(0.05, self._publish_setpoint)

        self.get_logger().info("Keyboard Flyer ready")

    def _on_pose(self, msg):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z

    def _on_state(self, msg):
        self.current_mode = msg.mode
        self.armed = msg.armed

    def _publish_setpoint(self):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)

    def set_offboard_and_arm(self):
        """Set OFFBOARD mode and arm the drone."""
        self.get_logger().info("Setting OFFBOARD mode...")
        mode_req = SetMode.Request()
        mode_req.custom_mode = "OFFBOARD"

        # Need to publish setpoints before switching to OFFBOARD
        time.sleep(2)

        future = self.mode_client.call_async(mode_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5)

        self.get_logger().info("Arming...")
        arm_req = CommandBool.Request()
        arm_req.value = True
        future = self.arm_client.call_async(arm_req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5)

    def takeoff(self, height=1.0):
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.target_z = height
        self.flying = True
        self.get_logger().info(f"Takeoff to {height}m")

    def land(self):
        self.target_x = self.current_x
        self.target_y = self.current_y
        self.target_z = 0.0
        self.flying = False
        self.get_logger().info("Landing...")


def get_key(timeout=0.05):
    """Non-blocking single character read from terminal."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if select.select([sys.stdin], [], [], timeout)[0]:
            return sys.stdin.read(1)
        return ""
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rclpy.init()
    node = KeyboardFlyer()

    # Spin ROS2 in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Wait for MAVROS connection
    print("Waiting for MAVROS connection...")
    while node.current_mode == "":
        time.sleep(0.5)
    print(f"Connected. Mode: {node.current_mode}, Armed: {node.armed}")

    print("\n=== INTERCEPTOR KEYBOARD CONTROLLER ===")
    print("  T = takeoff    L = land    Q = quit")
    print("  W/S = forward/backward")
    print("  A/D = left/right")
    print("  R/F = up/down")
    print("  Step size: 0.3m per keypress")
    print("========================================\n")

    try:
        while True:
            key = get_key()
            if not key:
                continue

            key = key.lower()

            if key == 'q':
                print("\nQuitting...")
                break

            elif key == 't':
                node.set_offboard_and_arm()
                node.takeoff(1.0)

            elif key == 'l':
                node.land()

            elif key == 'w':
                node.target_x += node.step
            elif key == 's':
                node.target_x -= node.step
            elif key == 'a':
                node.target_y += node.step
            elif key == 'd':
                node.target_y -= node.step
            elif key == 'r':
                node.target_z += node.step
            elif key == 'f':
                node.target_z -= node.step

            if key in ('w', 's', 'a', 'd', 'r', 'f', 't', 'l'):
                print(f"\r  Pos: ({node.current_x:.1f}, {node.current_y:.1f}, {node.current_z:.1f})"
                      f"  Target: ({node.target_x:.1f}, {node.target_y:.1f}, {node.target_z:.1f})"
                      f"  Mode: {node.current_mode}  Armed: {node.armed}    ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()