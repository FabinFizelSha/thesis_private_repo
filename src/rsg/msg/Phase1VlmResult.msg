# Asynchronous VLM result for one unknown object candidate.
# This is intentionally separate from Phase1ClassificationResult so Hydra input does not wait for VLM.

std_msgs/Header header
string rsg_frame_id
uint64 sequence
string candidate_id
# Persistent physical-object ID assigned by the unknown-object tracker.
# Several frame-local candidate_id values may map to one unknown_track_id.
string unknown_track_id
string mask_id

bool success
string status
string reason

string predicted_label
float32 confidence
string backend
string model

float64 vlm_delay_ms
float64 total_age_ms

# JSON representation of the original unknown object candidate and any VLM response metadata.
string object_metadata_json
string vlm_metadata_json
