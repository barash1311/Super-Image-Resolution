import torch
import torch.nn.functional as F
import numpy as np

def calculate_psnr(img1, img2, crop_border=4):
    # img1 and img2: [C, H, W] tensors or [H, W, C] numpy arrays
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Convert to Y channel if requested (typically PSNR for SR is on Y)
    if img1.shape[2] == 3:
        img1 = 16 + (65.481 * img1[:, :, 0] + 128.553 * img1[:, :, 1] + 24.966 * img1[:, :, 2]) / 255.0
        img2 = 16 + (65.481 * img2[:, :, 0] + 128.553 * img2[:, :, 1] + 24.966 * img2[:, :, 2]) / 255.0

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim(img1, img2, crop_border=4):
    # Using a simple SSIM implementation or skimage
    # For research, skimage.metrics.structural_similarity is standard
    from skimage.metrics import structural_similarity as ssim
    
    if isinstance(img1, torch.Tensor):
        img1 = img1.permute(1, 2, 0).cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.permute(1, 2, 0).cpu().numpy()

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if img1.shape[2] == 3:
        img1 = 16 + (65.481 * img1[:, :, 0] + 128.553 * img1[:, :, 1] + 24.966 * img1[:, :, 2]) / 255.0
        img2 = 16 + (65.481 * img2[:, :, 0] + 128.553 * img2[:, :, 1] + 24.966 * img2[:, :, 2]) / 255.0

    return ssim(img1, img2, data_range=255)
