import numpy as np
from scipy.spatial.transform import Rotation as R
from risk_scene_graph_core.config.parameters import rgb_dir, depth_dir, gt_file,depth_scale
import os
import cv2
from typing import Optional

class DatasetLoader:
    def __init__(self, dataset_dir_override: Optional[str] = None):
        """Initialize dataset loader with an optional override path.

        Args:
            dataset_dir_override: Optional dataset root path.

        Returns:
            None
        """
        self.dataset_dir_override = dataset_dir_override
        

    def parse_groundtruth(self,gt_path):
        """Parse groundtruth file into pose dictionary.

        Args:
            gt_path: Path to groundtruth.txt.

        Returns:
            Mapping of timestamp to (translation, rotation matrix).
        """
        poses = {}
        with open(gt_path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                timestamp = parts[0]
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                # Convert quaternion to rotation matrix
                rotM = R.from_quat([qx, qy, qz, qw]).as_matrix()
                poses[timestamp] = (np.array([tx, ty, tz]), rotM)
        return poses


    def load_dataset(self):
        """Load RGB, depth, and pose data into a list of samples.

        Args:
            None

        Returns:
            List of dataset samples with images and poses.
        """

        if self.dataset_dir_override:
            base = self.dataset_dir_override
            rgb_path = os.path.join(base, "rgb")
            depth_path = os.path.join(base, "depth")
            gt_path = os.path.join(base, "groundtruth.txt")
        else:
            rgb_path = rgb_dir
            depth_path = depth_dir
            gt_path = gt_file

        rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith(".png")])
        depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(".png")])
        poses = self.parse_groundtruth(gt_path)

        dataset = []
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            timestamp = rgb_file.rsplit(".", 1)[0]
            rgb_image=cv2.imread(os.path.join(rgb_path, rgb_file))
            depth_image=cv2.imread(os.path.join(depth_path, depth_file),cv2.IMREAD_UNCHANGED)/depth_scale
            if timestamp in poses:
                tx, rotM = poses[timestamp]
                dataset.append({
                    "timestamp": timestamp,
                    "rgb_img": rgb_image,
                    "depth_img": depth_image,
                    "tx": tx,
                    "rotM": rotM
                })
        return dataset
    
