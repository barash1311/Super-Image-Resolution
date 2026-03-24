import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
import yaml
import os
import argparse
from models.hybrid_sr import HybridSRModel
from data.dataset import SRDataset
from losses.perceptual import PerceptualLoss, TVLoss, LabColorLoss, LaplacianEdgeLoss
from utils.metrics import calculate_psnr, calculate_ssim
from utils.tile_inference import tiled_inference

def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    else:
        rank = -1
        world_size = 1
        gpu = 0
    
    if rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
        torch.cuda.set_device(gpu)
    return rank, world_size, gpu

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data.copy_(self.backup[name])
        self.backup = {}

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    rank, world_size, gpu = init_dist()
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Model
    model = HybridSRModel(
        nf=config['model']['nf'],
        gc=config['model']['gc'],
        n_rrdb=config['model']['n_rrdb'],
        transformer_depth=config['model']['transformer_depth'],
        transformer_heads=config['model']['transformer_heads']
    ).to(device)

    if rank != -1:
        model = DDP(model, device_ids=[gpu])
    
    # EMA
    ema = EMA(model, 0.999)
    ema.register()

    # Dataset
    train_dataset = SRDataset(
        hr_dir=config['train']['hr_dir'],
        patch_size=config['train']['patch_size'],
        upscale=config['train']['upscale'],
        mode='train'
    )
    
    sampler = DistributedSampler(train_dataset) if rank != -1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=(sampler is None),
        num_workers=config['train']['num_workers'],
        sampler=sampler,
        pin_memory=True
    )

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])
    
    # Loss functions
    criterion_l1 = nn.L1Loss().to(device)
    criterion_perceptual = PerceptualLoss().to(device)
    criterion_tv = TVLoss().to(device)
    criterion_lab = LabColorLoss().to(device)
    criterion_laplacian = LaplacianEdgeLoss().to(device)
    
    scaler = GradScaler()
    
    if rank <= 0:
        os.makedirs(config['train']['save_dir'], exist_ok=True)

    for epoch in range(config['train']['epochs']):
        if sampler:
            sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0
        for i, (lr, hr) in enumerate(train_loader):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()):
                sr = model(lr)
                
                l1_loss = criterion_l1(sr, hr)
                perc_loss = criterion_perceptual(sr, hr)
                tv_loss = criterion_tv(sr)
                lab_loss = criterion_lab(sr, hr)
                lap_loss = criterion_laplacian(sr, hr)
                
                total_loss = (
                    config['loss_weights']['l1'] * l1_loss +
                    config['loss_weights']['perceptual'] * perc_loss +
                    config['loss_weights']['tv'] * tv_loss +
                    config['loss_weights']['lab'] * lab_loss +
                    config['loss_weights']['laplacian'] * lap_loss
                )
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            epoch_loss += total_loss.item()
            
            if rank <= 0 and i % 100 == 0:
                print(f"Epoch [{epoch}/{config['train']['epochs']}] Batch [{i}/{len(train_loader)}] Loss: {total_loss.item():.4f}")

        scheduler.step()
        
        if rank <= 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if rank != -1 else model.state_dict(),
                'ema_state_dict': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': avg_loss
            }
            torch.save(save_dict, os.path.join(config['train']['save_dir'], 'latest.pth'))
            if (epoch + 1) % 10 == 0:
                 torch.save(save_dict, os.path.join(config['train']['save_dir'], f'model_epoch_{epoch+1}.pth'))

if __name__ == '__main__':
    train()
