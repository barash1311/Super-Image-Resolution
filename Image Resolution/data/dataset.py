import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from .degradation import RealisticDegradation

class SRDataset(Dataset):
    def __init__(self, hr_dir, patch_size=128, upscale=4, mode='train'):
        super(SRDataset, self).__init__()
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.upscale = upscale
        self.mode = mode
        self.hr_filenames = [os.path.join(hr_dir, x) for x in os.listdir(hr_dir) if x.endswith(('.png', '.jpg', '.jpeg'))]
        self.degradation = RealisticDegradation(upscale=upscale)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, index):
        hr_img = Image.open(self.hr_filenames[index]).convert('RGB')
        
        if self.mode == 'train':
            # Random patch cropping
            w, h = hr_img.size
            x = random.randint(0, w - self.patch_size)
            y = random.randint(0, h - self.patch_size)
            hr_patch = hr_img.crop((x, y, x + self.patch_size, y + self.patch_size))
            
            # Augmentation
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_patch = hr_patch.transpose(Image.ROTATE_90)
                
            hr_tensor = self.to_tensor(hr_patch)
            lr_tensor = self.degradation(hr_tensor)
            
            return lr_tensor, hr_tensor
        else:
            hr_tensor = self.to_tensor(hr_img)
            lr_tensor = self.degradation(hr_tensor)
            return lr_tensor, hr_tensor
