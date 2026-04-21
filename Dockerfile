# Interceptor Project — Docker Container
# Contains: perception + flight control scripts (color detection mode)
# Communicates with: Isaac Sim, MAVROS, PX4 via ROS2 topics (--network host)
#
# Build:
#   cd ~/interceptor
#   docker build -t interceptor .
#
# Run:
#   docker run --rm --network host -v ~/interceptor/logs:/app/logs interceptor camera_viewer
#   docker run --rm --network host -v ~/interceptor/logs:/app/logs interceptor intercept_yaw
#   docker run --rm --network host interceptor keyboard_fly
#
# The -v flag mounts your local logs directory so debug frames are saved
# on the host machine, not inside the container.

FROM ros:humble

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-numpy \
    python3-opencv \
    python3-scipy \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 message packages needed by our scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-mavros-msgs \
    && rm -rf /var/lib/apt/lists/*

# Set up workspace
WORKDIR /app

# Copy config first (changes less often — better layer caching)
COPY config/interceptor_config.py /app/config/interceptor_config.py

# Copy perception scripts
COPY perception/camera_viewer.py /app/perception/camera_viewer.py
COPY perception/color_detector.py /app/perception/color_detector.py
COPY perception/intercept_static.py /app/perception/intercept_static.py
COPY perception/intercept_yaw.py /app/perception/intercept_yaw.py
COPY perception/keyboard_fly.py /app/perception/keyboard_fly.py

# Create logs directory inside container
RUN mkdir -p /app/logs/debug_frames

# Fix Python path — scripts import from config and perception directories
# using sys.path.insert with ~/interceptor/. Inside Docker we use /app/ instead.
ENV PYTHONPATH="/app/config:/app/perception:${PYTHONPATH}"

# Override the HOME-based paths in config at runtime
# The config uses os.path.expanduser("~") which resolves to /root in Docker.
# We create a symlink so ~/interceptor points to /app
RUN mkdir -p /root && ln -s /app /root/interceptor

# Source ROS2 in every shell
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc

# Entrypoint script that sources ROS2 and runs the requested node
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]