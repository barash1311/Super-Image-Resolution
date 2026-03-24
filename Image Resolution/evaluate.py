import torch
import yaml
import os
from models.hybrid_sr import HybridSRModel
from data.dataset import SRDataset
from utils.metrics import calculate_psnr, calculate_ssim
from utils.tile_inference import tiled_inference
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def evaluate():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = HybridSRModel(
        nf=config['model']['nf'],
        gc=config['model']['gc'],
        n_rrdb=config['model']['n_rrdb'],
        transformer_depth=config['model']['transformer_depth'],
        transformer_heads=config['model']['transformer_heads']
    ).to(device)
    
    # Load last checkpoint if exists
    checkpoint_path = os.path.join(config['train']['save_dir'], 'latest.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded checkpoint from", checkpoint_path)
    else:
        print("No checkpoint found, evaluating with random weights.")

    model.eval()

    # Dataset
    val_dataset = SRDataset(
        hr_dir=config['val']['hr_dir'],
        upscale=config['train']['upscale'],
        mode='val'
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    avg_psnr = 0
    avg_ssim = 0
    
    output_dir = os.path.join(config['train']['save_dir'], 'results')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (lr, hr) in enumerate(val_loader):
            lr, hr = lr.to(device), hr.to(device)
            
            # Use tiled inference for validation
            sr = tiled_inference(model, lr, tile_size=128, overlap=16, upscale=config['train']['upscale'])
            
            # Clip output
            sr = torch.clamp(sr, 0, 1)
            
            # Compute metrics (on CPU/Numpy)
            # Rescale to [0, 255] for standard metrics
            sr_np = (sr.squeeze(0) * 255.0)
            hr_np = (hr.squeeze(0) * 255.0)
            
            avg_psnr += calculate_psnr(sr_np, hr_np, crop_border=config['train']['upscale'])
            avg_ssim += calculate_ssim(sr_np, hr_np, crop_border=config['train']['upscale'])
            
            # Save output images
            save_image(sr, os.path.join(output_dir, f'result_{i}.png'))

    avg_psnr /= len(val_loader)
    avg_ssim /= len(val_loader)
    
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    evaluate()
