# Prepared RGB-D-pose frame for the Risk Scene Graph pipeline.
# This message is produced by RSG_pre_processor and consumed by the main worker node.

std_msgs/Header header

# Unique frame identifier generated from local session date + ROS data timestamp + sequence number.
string rsg_frame_id
string source
uint64 sequence

# Prepared sensor data.
sensor_msgs/Image rgb
sensor_msgs/Image depth_m
sensor_msgs/CameraInfo camera_info

# Optional camera IMU sample matched to the RGB timestamp.
bool has_camera_imu
sensor_msgs/Imu camera_imu
float64 rgb_imu_dt_sec

# Camera pose in the configured world/odom frame.
geometry_msgs/PoseStamped camera_pose

# Existing framework-compatible pose representation.
float64[3] tx
float64[9] rot_m

# Synchronization and validation metadata.
float32 invalid_depth_ratio
float64 rgb_depth_dt_sec
float64 rgb_camera_info_dt_sec
float64 rgb_odom_dt_sec
string metadata_json
