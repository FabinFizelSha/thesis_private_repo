# Hydra-ready RGB-D semantic input produced by rsg_object_detection.
# If a Hydra integration requires separate topics, the coordinator can also republish these fields separately.

std_msgs/Header header
string rsg_frame_id
string source
uint64 sequence

sensor_msgs/Image rgb
sensor_msgs/Image depth_m
sensor_msgs/CameraInfo camera_info
geometry_msgs/PoseStamped camera_pose

float64[3] tx
float64[9] rot_m

# Pixel-level labels from SAM/RAP classification.
sensor_msgs/Image semantic_labels
sensor_msgs/Image instance_labels

# JSON metadata used by Hydra adapters, fusion, and later risk annotation.
string label_table_json
string object_metadata_json
string unknown_candidates_json
string perception_metadata_json
string metadata_json

# Timing data for monitoring end-to-end delay from preprocessor input to Hydra-ready output.
float64 coordinator_delay_ms
float64 classifier_delay_ms
float64 total_delay_ms
