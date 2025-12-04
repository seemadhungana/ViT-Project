import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import time
import math

def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed mode")
        torch.cuda.set_device(0)
        return 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    return rank, world_size, local_rank

def get_dataloaders(data_path, batch_size, num_workers, world_size, rank):
    """Create train and val dataloaders"""
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        transform=val_transform
    )
    
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler

def save_checkpoint(epoch, model, optimizer, scheduler, scaler, train_loss, val_loss, val_acc, output_dir, local_rank):
    """Save checkpoint - EVERY EPOCH"""
    if local_rank != 0:
        return
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    latest_path = os.path.join(output_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)
    
    print(f"✓ Saved checkpoint: epoch_{epoch:03d}.pth")
    
    csv_path = os.path.join(output_dir, 'training_metrics.csv')
    
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,val_loss,val_acc\n')
    
    with open(csv_path, 'a') as f:
        f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{val_acc:.4f}\n')

def train_one_epoch(model, train_loader, optimizer, scaler, epoch, local_rank, world_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    start_time = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.cuda(local_rank, non_blocking=True)
        targets = targets.cuda(local_rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0 and local_rank == 0:
            print(f"  Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    avg_loss = total_loss / num_batches
    avg_loss_tensor = torch.tensor(avg_loss).cuda(local_rank)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / world_size
    
    epoch_time = time.time() - start_time
    
    if local_rank == 0:
        print(f"✓ Epoch {epoch} training completed in {epoch_time/60:.2f} minutes. "
              f"Avg Train Loss: {avg_loss:.4f}")
    
    return avg_loss

def validate(model, val_loader, local_rank, world_size):
    """Validation"""
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)
            
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, targets)
            
            val_loss += loss.item()
            num_batches += 1
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = val_loss / num_batches
    
    metrics = torch.tensor([avg_loss, accuracy, total, correct]).cuda(local_rank)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    global_loss = metrics[0].item()
    global_total = int(metrics[2].item())
    global_correct = int(metrics[3].item())
    global_acc = 100. * global_correct / global_total
    
    if local_rank == 0:
        print(f"✓ Validation - Loss: {global_loss:.4f}, Accuracy: {global_acc:.2f}%")
    
    return global_loss, global_acc

class CosineAnnealingLRWithWarmup:
    """Cosine annealing scheduler with linear warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.eta_min + (base_lr - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
        
        self.current_epoch += 1
    
    def state_dict(self):
        return {
            'current_epoch': self.current_epoch,
            'base_lrs': self.base_lrs,
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'eta_min': self.eta_min
        }
    
    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.eta_min = state_dict['eta_min']

def main():
    parser = argparse.ArgumentParser(description='Train ViT-Base on ImageNet')
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNet data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs')
    args = parser.parse_args()
    
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("="*60)
        print("ViT-Base ImageNet Training (SGD)")
        print("="*60)
        print(f"World size: {world_size} GPUs")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"Total epochs: {args.epochs}")
        print(f"Optimizer: SGD")
        print(f"Learning rate: {args.lr}")
        print(f"Momentum: {args.momentum}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Warmup epochs: {args.warmup_epochs}")
        print(f"Output directory: {args.output_dir}")
        print("="*60)
        os.makedirs(args.output_dir, exist_ok=True)
    
    if world_size > 1:
        dist.barrier()
    
    if rank == 0:
        print("\nCreating ViT-Base model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
    model = model.cuda(local_rank)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created. Parameters: {total_params/1e6:.1f}M")
    
    # SGD optimizer with momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Cosine annealing with warmup
    scheduler = CosineAnnealingLRWithWarmup(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        eta_min=0
    )
    
    scaler = GradScaler()
    
    if rank == 0:
        print("\nLoading ImageNet data...")
    train_loader, val_loader, train_sampler = get_dataloaders(
        args.data_path, args.batch_size, args.num_workers, world_size, rank
    )
    
    if rank == 0:
        print(f"✓ Data loaded. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    start_epoch = 0
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_latest.pth')
    if os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"\n⟳ Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
        if world_size > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"✓ Resumed from epoch {checkpoint['epoch']}")
    
    if rank == 0:
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{args.epochs-1}")
            print(f"{'='*60}")
        
        train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch, local_rank, world_size)
        
        scheduler.step()
        
        val_loss, val_acc = validate(model, val_loader, local_rank, world_size)
        
        save_checkpoint(epoch, model, optimizer, scheduler, scaler, 
                       train_loss, val_loss, val_acc, args.output_dir, local_rank)
        
        if rank == 0:
            print(f"{'='*60}\n")
    
    if rank == 0:
        print("\n" + "="*60)
        print("✓ Training Complete!")
        print("="*60)
    
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
