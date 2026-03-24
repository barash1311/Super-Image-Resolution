import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class RealisticDegradation:
    def __init__(self, upscale=4):
        self.upscale = upscale

    def gaussian_blur(self, img, kernel_size=11, sigma_range=(0.2, 1.2)):
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        # cv2.GaussianBlur expects img in [H, W, C]
        img_np = img.permute(1, 2, 0).numpy()
        img_blur = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        return torch.from_numpy(img_blur).permute(2, 0, 1)

    def bicubic_downscale(self, img, factor=4):
        # img is [C, H, W] tensor
        _, h, w = img.shape
        return F.interpolate(img.unsqueeze(0), size=(h // factor, w // factor), mode='bicubic', align_corners=False).squeeze(0)

    def gaussian_noise(self, img, sigma_range=(0, 0.05)):
        sigma = np.random.uniform(sigma_range[0], sigma_range[1])
        noise = torch.randn_like(img) * sigma
        return img + noise

    def haze_simulation(self, img, alpha_range=(0.85, 0.98), L=1.0):
        # I = J * t + A * (1 - t) where J is clean, A is atmospheric light, t is transmission
        # Reduced haze intensity for better color preservation
        if np.random.random() > 0.5: # Apply haze only 50% of the time
            alpha = np.random.uniform(alpha_range[0], alpha_range[1])
            return alpha * img + (1 - alpha) * L
        return img

    def __call__(self, hr_patch):
        # hr_patch: [C, H, W] tensor
        lr = self.gaussian_blur(hr_patch)
        lr = self.bicubic_downscale(lr, self.upscale)
        lr = self.gaussian_noise(lr)
        lr = self.haze_simulation(lr)
        return torch.clamp(lr, 0, 1)
