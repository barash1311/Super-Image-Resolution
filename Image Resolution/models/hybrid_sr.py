import torch
import torch.nn as nn
import torch.nn.functional as F
from .rrdb import RRDB
from .attention import SEBlock, SpatialAttention
from .transformer import LightweightTransformer

class HybridSRModel(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, gc=32, n_rrdb=6, transformer_depth=2, transformer_heads=8, upscale=4):
        super(HybridSRModel, self).__init__()
        self.upscale = upscale
        
        # 1. Initial shallow feature extractor
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        
        # 2. Deep feature extraction with RRDB and SE blocks
        rrdb_blocks = []
        for _ in range(n_rrdb):
            rrdb_blocks.append(RRDB(nf, gc))
            rrdb_blocks.append(SEBlock(nf))
        self.rrdb_trunk = nn.Sequential(*rrdb_blocks)
        
        # 3. Spatial Attention
        self.spatial_attn = SpatialAttention()
        
        # 4. Lightweight Transformer in the bottleneck
        self.transformer = LightweightTransformer(dim=nf, depth=transformer_depth, num_heads=transformer_heads)
        
        # 5. Trunk Conv
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # 6. Upsampling (PixelShuffle)
        # nf -> nf*4 -> PS(2) -> nf -> nf*4 -> PS(2) -> nf
        self.upconv_x2_1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.upconv_x2_2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        
        # 7. Final reconstruction
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pixel_shuffle_blur_fix = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        # Global Skip Connection (Bicubic Upsample)
        x_upsampled = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        
        # Shallow feature
        fea_shallow = self.conv_first(x)
        
        # RRDB + SE Trunk
        fea = self.rrdb_trunk(fea_shallow)
        
        # Spatial Attention
        fea = self.spatial_attn(fea)
        
        # Transformer Bottleneck (Global Context)
        b, c, h, w = fea.shape
        fea_trans = fea.flatten(2).transpose(1, 2) # [B, H*W, C]
        fea_trans = self.transformer(fea_trans)
        fea_trans = fea_trans.transpose(1, 2).view(b, c, h, w)
        
        fea = fea + fea_trans
        fea = self.trunk_conv(fea)
        
        # Global Residual connection
        fea = fea + fea_shallow
        
        # Upsampling x4 (Two stages of PixelShuffle)
        out = self.lrelu(self.pixel_shuffle(self.upconv_x2_1(fea)))
        out = self.pixel_shuffle_blur_fix(out)
        out = self.lrelu(self.pixel_shuffle(self.upconv_x2_2(out)))
        
        # Final RGB reconstruction
        out = self.conv_last(out)
        
        # Add the upsampled input (Residual Learning)
        return out + x_upsampled
