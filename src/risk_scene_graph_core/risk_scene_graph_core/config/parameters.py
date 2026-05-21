import numpy as np
import os

'''
# Camera intrinsics for the D455 (from RealSense Viewer)
fx= 386.4931640625
fy= 386.0564880371094
cx= 325.06463623046875
cy= 246.09620666503906
K_rgb = np.array([[fx,   0.0, cx],
                  [  0.0, fy, cy],
                  [  0.0,   0.0,   1.0]])
dist_coeffs_rgb = np.array([-0.055510714650154114, 0.06504587084054947, -0.00012300504022277892, 0.00034680962562561035, -0.021409034729003906],dtype=np.float64)
'''

# Camera intrinsics
# Note: If depth is aligned to RGB, set K_depth/dist_coeffs_depth equal to RGB.
fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6
K_rgb = np.array([[fx,   0.0, cx],
                  [  0.0, fy, cy],
                  [  0.0,   0.0,   1.0]])
dist_coeffs_rgb = np.zeros(5)  # [k1, k2, p1, p2, k3]

K_depth = K_rgb.copy()
dist_coeffs_depth = np.zeros(5) #always zero for realsene


# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.dirname(script_dir)
tmp_dir = os.path.join(main_dir, "data", "tmp")
masks_output_path = os.path.join(tmp_dir, "masks")
masks_detected_output_path = os.path.join(tmp_dir, "masks_detected")


# Dataset paths 
# depth_scale: Divides raw depth values to convert to meters
#   - TUM RGB-D dataset: Use 5000 (official depth scaling factor)
#   - RealSense D455: Use 1000 (Z16 format stores depth in millimeters)
depth_scale = 5000.0
img_size = np.array([480, 640])  # height first, width second
dataset_dir = os.path.join(main_dir,"data","dataset_test")
rgb_dir = os.path.join(dataset_dir,"rgb")
depth_dir = os.path.join(dataset_dir,"depth")
gt_file = os.path.join(dataset_dir,"groundtruth.txt")

vlm_input_format = 'json' #(json, yaml, toon)
appearance_threshold = 0 # VLM appearance threshold until they get deleted
margin = 1 # frustum margin (higher = larger frustum considered, 1 = exact)

# SAM segmentation parameters
sam_stability_threshold = 0.88  # Stability score threshold for SAM masks (higher = fewer masks)
sam_pred_iou_thresh = 0.96  # Predicted IoU threshold (higher = fewer masks, fewer hierarchical overlaps)
sam_padding = 0.2  # Padding around detected objects as % of bbox size
sam_min_segment_pixels = 500  # Minimum pixels for valid segment (filters tiny junk)

# Visual RAG parameters
vrap_distance_threshold = 0.3  # Distance threshold for Visual RAG matches (lower = stricter)