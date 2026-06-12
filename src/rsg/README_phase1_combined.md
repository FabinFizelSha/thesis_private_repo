# Phase 1 Combined Processor

This package includes an optimized single-process Phase 1 node:

```bash
ros2 launch rsg rsg_phase1.launch.py
```

The launch now starts:

```text
rsg_object_detection
```

This implements Option A:

```text
/rsg/preprocessed/frame
    -> internal FIFO frame queue
    -> SAM
    -> RAP
    -> unknown-object tracking
    -> Hydra-ready frame output
```

There is no ROS round trip from coordinator to classifier and back. The large
RGB-D frame stays inside one Python process, reducing DDS serialization and
Python executor scheduling overhead.

The previous two-node version is still available for comparison:

```bash
ros2 launch rsg rsg_phase1_two_node.launch.py
```

Important queues:

```text
1. Internal frame FIFO before SAM/RAP
   Config: phase1.coordinator.request_queue_size

2. VLM FIFO after RAP/unknown tracking
   Config: phase1.vlm.queue_size
```

Important debug CSVs:

```text
phase1_cordinator_hydra_latency.csv
phase1_classifier_phase_latency.csv
phase1_frame_fifo_queue.csv
phase1_unknown_tracks.csv
phase1_vlm_queue.csv
phase1_vlm_latency.csv
```

In combined mode:

```text
sent_to_classifier_delay_ms = internal FIFO wait before SAM/RAP starts
pipeline_wait_ms = remaining in-process overhead after accounting for FIFO wait, classifier time, and Hydra publish time
```
