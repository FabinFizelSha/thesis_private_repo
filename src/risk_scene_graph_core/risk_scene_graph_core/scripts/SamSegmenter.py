import torch
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import urllib.request


class SamSegmenter:
    # SAM checkpoint URL
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    def __init__(
        self,
        checkpoint_path=None,
        device=None,
        mask_threshold=0.88,
        padding=0.1,
        min_segment_pixels=500,
        points_per_side=16,
        pred_iou_thresh=0.96,
    ):
        """Initialize the SAM segmenter and mask generator.

        Args:
            checkpoint_path: Optional path to SAM checkpoint.
            device: Torch device override.
            mask_threshold: Stability score threshold.
            padding: Padding ratio for crops.
            min_segment_pixels: Minimum mask size in pixels.
            points_per_side: Grid size for sampling points.
            pred_iou_thresh: IOU threshold for mask predictions.

        Returns:
            None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_threshold = mask_threshold
        self.padding = padding
        self.min_segment_pixels = min_segment_pixels
        self.pred_iou_thresh = pred_iou_thresh

        # Get or download checkpoint
        if checkpoint_path is None:
            checkpoint_path = self._get_checkpoint()
        
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        
        # Create automatic mask generator with reduced mask count
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,  
            pred_iou_thresh=pred_iou_thresh,  
            stability_score_thresh=mask_threshold,
            min_mask_region_area=min_segment_pixels,
        )

    def _get_checkpoint(self):
        """Download checkpoint if not exists and return the path.

        Args:
            None

        Returns:
            Path to the checkpoint file.
        """
        cache_dir = os.path.expanduser("~/.cache/sam")
        os.makedirs(cache_dir, exist_ok=True)
        
        checkpoint_name = "sam_vit_h_4b8939.pth"
        checkpoint_path = os.path.join(cache_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"[SAM] Downloading vit_h checkpoint (~2.4GB)...")
            print(f"[SAM] This may take several minutes depending on your connection...")
            
            def progress_hook(block_num, block_size, total_size):
                """Report download progress for the checkpoint.

                Args:
                    block_num: Current block index.
                    block_size: Size of each block in bytes.
                    total_size: Total download size in bytes.

                Returns:
                    None
                """
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\r[SAM] Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='', flush=True)
            
            urllib.request.urlretrieve(
                self.CHECKPOINT_URL,
                checkpoint_path,
                reporthook=progress_hook
            )
            print(f"\n[SAM] Downloaded to {checkpoint_path}")
        
        return checkpoint_path

    def segment(self, image: Image.Image):
        """Generate object proposals from the input image.

        Args:
            image: PIL image.

        Returns:
            List of detection dicts with crop, bbox, mask, and score.
        """
        #Generate object proposals from the input image
        image_np = np.array(image)
        
        masks = self.mask_generator.generate(image_np)
        
        width, height = image.size
        results = []

        for mask_data in masks:
            segmentation = mask_data['segmentation']
            score = mask_data['stability_score']  
            bbox_xywh = mask_data['bbox'] 
            
            x, y, w, h = bbox_xywh
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Add padding
            pad_x = int(w * self.padding)
            pad_y = int(h * self.padding)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            
            crop = image.crop((x1, y1, x2, y2))

            results.append({
                "crop": crop,
                "bbox": (x1, y1, x2, y2),
                "mask": segmentation,
                "score": float(score),
            })

        return results
