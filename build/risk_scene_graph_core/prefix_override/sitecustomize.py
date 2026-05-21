import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/fabin/rsg_ros2_ws/install/risk_scene_graph_core'
