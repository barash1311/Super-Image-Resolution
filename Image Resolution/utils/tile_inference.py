import torch
import torch.nn.functional as F

def tiled_inference(model, img, tile_size=128, overlap=16, upscale=4):
    """
    Tiled inference to handle large images without OOM.
    """
    b, c, h, w = img.shape
    output_h = h * upscale
    output_w = w * upscale
    
    output = torch.zeros((b, c, output_h, output_w), device=img.device)
    weight = torch.zeros((b, c, output_h, output_w), device=img.device)
    
    stride = tile_size - overlap
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(y2 - tile_size, 0)
            x1 = max(x2 - tile_size, 0)
            
            tile = img[:, :, y1:y2, x1:x2]
            
            # Inference
            with torch.no_grad():
                out_tile = model(tile)
                
            # Accumulate
            oy1, oy2 = y1 * upscale, y2 * upscale
            ox1, ox2 = x1 * upscale, x2 * upscale
            
            output[:, :, oy1:oy2, ox1:ox2] += out_tile
            weight[:, :, oy1:oy2, ox1:ox2] += 1.0
            
    return output / weight
