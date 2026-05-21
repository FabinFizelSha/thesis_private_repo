import numpy as np
import cv2
from risk_scene_graph_core.config.parameters import K_rgb

class PositionCalculator:
    def __init__(self):
        """Initialize the position calculator.

        Args:
            None

        Returns:
            None
        """
        pass

    def calculate_translated_position(self, tx, rotM, offset):
        """Compute translated position from an offset in robot frame.

        Args:
            tx: Translation vector.
            rotM: Rotation matrix.
            offset: Offset vector in local frame.

        Returns:
            New position vector.
        """
        rotated_offset = rotM @ offset
        new_position = tx + rotated_offset
        return new_position

    def pos_to_pixel(self, X, Y, Z, rotM, tx):
        """Project a 3D world point into pixel coordinates.

        Args:
            X: World X coordinate.
            Y: World Y coordinate.
            Z: World Z coordinate.
            rotM: Rotation matrix (camera to world).
            tx: Translation vector.

        Returns:
            Pixel coordinates as a NumPy array (x, y).
        """
        # world_point as column
        world_point = np.array([X, Y, Z], dtype=np.float64).reshape((3, 1))
        tx = tx.reshape((3, 1))

        # Convert world -> camera (camera_point in camera coords)
        # rotM maps camera -> world: world = rotM @ camera + tx
        # so camera = rotM.T @ (world - tx)
        camera_point = rotM.T @ (world_point - tx)

        # If point is behind camera it's invalid
        if camera_point[2, 0] <= 0:
            return np.array([-1, -1], dtype=int)

        # Project using OpenCV (camera_point already camera coords)
        camera_point_cv = camera_point.reshape((1, 1, 3)).astype(np.float32)
        image_points, _ = cv2.projectPoints(
            camera_point_cv,
            np.zeros((3, 1), dtype=np.float32),   # rvec
            np.zeros((3, 1), dtype=np.float32),   # tvec
            K_rgb.astype(np.float32),
            None
        )
        pixel = np.round(image_points[0, 0]).astype(int)
        return pixel

    def pixel_to_pos(self, px_x, px_y, depth_img, rotM, tx):
        """Back-project pixel coordinates to a 3D world position.

        Args:
            px_x: Pixel x coordinate.
            px_y: Pixel y coordinate.
            depth_img: Depth image in meters.
            rotM: Rotation matrix (camera to world).
            tx: Translation vector.

        Returns:
            3D world point as a NumPy array.
        """
        px_x_i = int(px_x)
        px_y_i = int(px_y)
        z = float(depth_img[px_y_i, px_x_i])
        if z <= 0 or np.isnan(z):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

        # Convert pixel to normalized camera coordinates (images already undistorted)
        fx = float(K_rgb[0, 0])
        fy = float(K_rgb[1, 1])
        cx = float(K_rgb[0, 2])
        cy = float(K_rgb[1, 2])
        x_norm = (float(px_x) - cx) / fx
        y_norm = (float(px_y) - cy) / fy
        # camera coordinates (multiply by depth)
        x_cam = x_norm * z
        y_cam = y_norm * z
        camera_point = np.array([x_cam, y_cam, z], dtype=np.float64).reshape((3, 1))

        # Convert camera -> world: world = rotM @ camera + tx
        tx = tx.reshape((3, 1))
        world_point = rotM @ camera_point + tx
        return world_point.flatten()


