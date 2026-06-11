# RSG preprocessor structure

The preprocessor was split from one large Python file into a small main node and helper modules.

- `rsg_preprocessing/RSG_pre_processor.py`: ROS 2 node, subscriptions, callbacks, publishing, status/timing flow.
- `rsg_preprocessing/helper_functions/config_loader.py`: YAML loading and all configurable switches.
- `rsg_preprocessing/helper_functions/image_converter.py`: RGB/depth conversion and invalid-depth ratio.
- `rsg_preprocessing/helper_functions/frame_validator.py`: rejection checks.
- `rsg_preprocessing/helper_functions/odom_buffer.py`: odometry buffering and timestamp lookup/interpolation.
- `rsg_preprocessing/helper_functions/imu_buffer.py`: optional camera IMU buffering and lookup.
- `rsg_preprocessing/helper_functions/transform_math.py`: matrix/quaternion/pose utilities.
- `rsg_preprocessing/helper_functions/timing_excel_recorder.py`: Excel debug output including rejected-frame rows.

All rejection checks can be controlled from `config/rsg_pipeline.yaml` under `preprocessing.validation`.
