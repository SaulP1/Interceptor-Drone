
Overview
The interceptor drone uses its onboard camera to visually track a target and autonomously fly toward it. No world coordinates or GPS are used for tracking — the drone steers entirely based on what the camera sees.
How it works:

1.The camera captures frames and publishes them to ROS2
2.A detection module (HSV color detection) finds the target and computes its offset from the camera center
3.A depth image measures distance to the target
4.A PID controller converts the camera offset into yaw (turn), altitude, and forward velocity commands
5.Forward speed is determined by a physics-based stopping distance equation — the drone flies at the fastest speed it can safely decelerate from
6.The drone turns to face the target, matches its altitude, and flies forward until interception
Dependencies

Isaac Sim 5.1.0 — simulation environment
- Pegasus Simulator Framework — drone spawning and backends
- PX4 SITL — flight stack
- MAVROS — ROS2-to-MAVLink bridge
- ROS2 Humble — middleware
- QGroundControl — monitoring and arming
- OpenCV — image processing
- NumPy / SciPy — math operations
