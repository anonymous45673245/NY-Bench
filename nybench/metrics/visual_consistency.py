import torch
import lpips
import numpy as np
import cv2
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr

class VisualConsistencyMetric:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[VC] Loading LPIPS model on {self.device}...")
        # LPIPS (AlexNet)
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)

    def load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def to_tensor(self, img_np):
        # (H,W,C) -> (1,C,H,W) & Normalize [-1, 1]
        img = img_np.astype(np.float32) / 127.5 - 1.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return img.to(self.device)

    def calculate_scores(self, img_ref_path, img_curr_path, mask_fg_path=None):
        """
        Calculate S_restore and S_preserve.
        Automatically resizes img_curr to match img_ref shape.
        """
        # 1. Load Images
        img_ref = self.load_image(img_ref_path)
        img_curr = self.load_image(img_curr_path)

        # ---------------------------------------------------------
        # [FIX] Resize curr to match ref (Resolution Mismatch 해결)
        # ---------------------------------------------------------
        ref_h, ref_w = img_ref.shape[:2]
        curr_h, curr_w = img_curr.shape[:2]

        if (ref_h, ref_w) != (curr_h, curr_w):
            # cv2.resize takes (Width, Height)
            img_curr = cv2.resize(img_curr, (ref_w, ref_h), interpolation=cv2.INTER_AREA)

        # 2. Load Mask (Foreground)
        if mask_fg_path:
            mask = cv2.imread(str(mask_fg_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (ref_h, ref_w):
                    mask = cv2.resize(mask, (ref_w, ref_h), interpolation=cv2.INTER_NEAREST)
                mask_bg = 255 - mask
            else:
                mask_bg = np.ones((ref_h, ref_w), dtype=np.uint8) * 255
        else:
            mask_bg = np.ones((ref_h, ref_w), dtype=np.uint8) * 255

        # 3. S_restore (Background Consistency using LPIPS)
        mask_bg_3c = cv2.merge([mask_bg, mask_bg, mask_bg]) / 255.0
        
        # Apply mask
        img_ref_bg = img_ref * mask_bg_3c
        img_curr_bg = img_curr * mask_bg_3c

        t_ref_bg = self.to_tensor(img_ref_bg)
        t_curr_bg = self.to_tensor(img_curr_bg)

        with torch.no_grad():
            lpips_val = self.loss_fn(t_ref_bg, t_curr_bg).item()
        
        s_restore = max(0, 1 - lpips_val)

        # 4. S_preserve (Overall Quality using PSNR)
        try:
            # img_ref and img_curr are now guaranteed to be same shape
            psnr_val = psnr(img_ref, img_curr, data_range=255)
        except:
            psnr_val = 0
        
        s_preserve = min(max(psnr_val, 0), 30) / 30.0

        return {"S_restore": s_restore, "S_preserve": s_preserve}
