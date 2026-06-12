# Result produced by rsg_object_detection for one RSG frame.
# The coordinator combines this result with the original RsgFrame and forwards it to Hydra.

std_msgs/Header header
string rsg_frame_id
uint64 sequence

bool success
string status
string reason

# Pixel-level labels for Hydra input.
# Default encoding is 16UC1. The semantic image stores class IDs.
# The instance image stores per-object instance IDs.
sensor_msgs/Image semantic_labels
sensor_msgs/Image instance_labels

# JSON metadata. Kept as JSON to avoid locking the design before Hydra/fusion interfaces are final.
string label_table_json
string object_metadata_json
string unknown_candidates_json
string vlm_dispatch_json
string metadata_json

# Timing/debug data.
float64 input_age_ms
float64 classifier_delay_ms
float64 sam_delay_ms
float64 rap_delay_ms
float64 label_map_delay_ms
float64 metadata_delay_ms
float64 image_conversion_delay_ms
float64 result_message_build_delay_ms
float64 classifier_debug_record_delay_ms

uint32 num_masks
uint32 num_known
uint32 num_unknown
