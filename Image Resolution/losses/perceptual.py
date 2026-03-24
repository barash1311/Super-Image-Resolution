import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        # Use multiple layers for better detail preservation
        # relu1_2, relu2_2, relu3_4, relu4_4, relu5_4
        self.layer_indices = [3, 8, 17, 26, 35]
        self.vgg_layers = nn.ModuleList([nn.Sequential(*list(vgg.children())[:i+1]) for i in self.layer_indices])
        
        for layer in self.vgg_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        self.layer_weights = [0.1, 0.1, 1.0, 1.0, 1.0]
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        loss = 0
        for i, layer in enumerate(self.vgg_layers):
            x_feat = layer(x)
            y_feat = layer(y)
            loss += self.layer_weights[i] * self.criterion(x_feat, y_feat)
        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class LabColorLoss(nn.Module):
    def __init__(self):
        super(LabColorLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def rgb_to_lab(self, rgb):
        # Normalize RGB to [0, 1] if not already
        rgb = torch.clamp(rgb, 0, 1)
        
        # RGB to XYZ
        mask = (rgb > 0.04045).float()
        rgb = (((rgb + 0.055) / 1.055) ** 2.4) * mask + (rgb / 12.92) * (1 - mask)
        
        if not hasattr(self, 'xyz_matrix') or self.xyz_matrix.device != rgb.device:
            self.xyz_matrix = torch.tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]
            ], device=rgb.device).t()
        
        xyz = torch.matmul(rgb.permute(0, 2, 3, 1), self.xyz_matrix).permute(0, 3, 1, 2)
        
        # XYZ to Lab
        xyz[:, 0] /= 0.95047
        xyz[:, 1] /= 1.00000
        xyz[:, 2] /= 1.08883
        
        mask = (xyz > 0.008856).float()
        f_xyz = (xyz ** (1/3)) * mask + (7.787 * xyz + 16/116) * (1 - mask)
        
        L = 116 * f_xyz[:, 1:2, :, :] - 16
        a = 500 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])
        b = 200 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])
        
        return torch.cat([L, a, b], dim=1)

    def forward(self, x, y):
        x_lab = self.rgb_to_lab(x)
        y_lab = self.rgb_to_lab(y)
        return self.l1_loss(x_lab, y_lab)

class LaplacianEdgeLoss(nn.Module):
    def __init__(self):
        super(LaplacianEdgeLoss, self).__init__()
        # Use a more comprehensive 3x3 Laplacian kernel
        self.register_buffer('kernel', torch.tensor([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]], dtype=torch.float32))
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        # Convert to grayscale for edge detection
        x_gray = 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]
        y_gray = 0.299 * y[:, 0:1, :, :] + 0.587 * y[:, 1:2, :, :] + 0.114 * y[:, 2:3, :, :]
        
        x_edges = F.conv2d(x_gray, self.kernel, padding=1)
        y_edges = F.conv2d(y_gray, self.kernel, padding=1)
        
        return self.l1_loss(x_edges, y_edges)
