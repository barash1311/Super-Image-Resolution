import torch
import torch.nn as nn
import yaml
import os
import cv2
import numpy as np
from PIL import Image
from models.hybrid_sr import HybridSRModel
from utils.tile_inference import tiled_inference

class SRInference:
    def __init__(self, config_path='config.yaml', model_path=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = HybridSRModel(
            nf=self.config['model']['nf'],
            gc=self.config['model']['gc'],
            n_rrdb=self.config['model']['n_rrdb'],
            transformer_depth=self.config['model']['transformer_depth'],
            transformer_heads=self.config['model']['transformer_heads']
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'ema_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['ema_state_dict'])
                print(f"Loaded EMA model weights from {model_path}")
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model weights from {model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model from {model_path}")
        else:
            print("Warning: No model weights loaded. Using random initialization.")
            
        self.model.eval()

    def enhance(self, input_image):
        """
        input_image: PIL Image or path to image
        returns: PIL Image (enhanced)
        """
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert('RGB')
        
        img_tensor = torch.from_numpy(np.array(input_image)).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Use tiled inference to handle any image size
            sr_tensor = tiled_inference(
                self.model, 
                img_tensor, 
                tile_size=128, 
                overlap=16, 
                upscale=self.config['train']['upscale']
            )
            
            sr_tensor = torch.clamp(sr_tensor, 0, 1)
            sr_img = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            sr_img = (sr_img * 255.0).astype(np.uint8)
            
        return Image.fromarray(sr_img)

if __name__ == '__main__':
    # Simple CLI test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='enhanced.png')
    parser.add_argument('--model', type=str, default='experiments/hybrid_sr_v1/latest.pth')
    args = parser.parse_args()
    
    sr = SRInference(model_path=args.model)
    result = sr.enhance(args.input)
    result.save(args.output)
    print(f"Saved enhanced image to {args.output}")
