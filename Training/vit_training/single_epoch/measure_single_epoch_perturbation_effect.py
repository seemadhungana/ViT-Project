#!/usr/bin/env python3
"""
Measure the immediate effect of single-epoch perturbations on ViT training.

For each perturbation type (gaussian, uniform_gray, label_shuffle, target_noise):
1. Load checkpoint from epoch (N-1)
2. Train ONLY epoch N with perturbation applied
3. Evaluate on validation set and compute RSA
4. Compute Δ loss and Δ RSA compared to baseline epoch N
5. Save results for plotting

This generates the data needed to compare perturbation types similar to the CLIP-HBA analysis.
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from torchvision import datasets, transforms
import timm
from torch.cuda.amp import autocast, GradScaler
import time
import math
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr
from PIL import Image

# =============================================================================
# Import perturbation transforms from existing script
# =============================================================================

class GaussianNoiseTransform:
    """Replaces image with pure Gaussian noise"""
    def __init__(self, base_transform, epsilon=0.1):
        self.base_transform = base_transform
        self.epsilon = epsilon

    def __call__(self, img):
        img = self.base_transform(img)
        noise = torch.randn_like(img) * self.epsilon
        return noise

class UniformGrayTransform:
    """Replaces image with uniform gray after normalization"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        img = self.base_transform(img)
        img = torch.zeros_like(img)
        return img

class ShuffledLabelsDataset(Dataset):
    """Wrapper that shuffles labels for the dataset"""
    def __init__(self, base_dataset, shuffle_seed=42):
        self.base_dataset = base_dataset
        self.num_samples = len(base_dataset)

        rng = np.random.RandomState(shuffle_seed)
        self.shuffled_indices = rng.permutation(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        _, shuffled_label = self.base_dataset[self.shuffled_indices[idx]]
        return img, shuffled_label

class TargetNoiseDataset(Dataset):
    """
    Wrapper that replaces true labels with random targets.
    This simulates 'target noise' similar to the CVPR paper.
    """
    def __init__(self, base_dataset, num_classes=1000, noise_seed=42):
        self.base_dataset = base_dataset
        self.num_classes = num_classes

        # Generate random targets for each sample
        rng = np.random.RandomState(noise_seed)
        self.random_targets = rng.randint(0, num_classes, len(base_dataset))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        random_target = self.random_targets[idx]
        return img, random_target

class THINGSInferenceDataset(Dataset):
    """Dataset for THINGS 48 validation images for RSA computation"""
    def __init__(self, csv_file, img_dir, rdm_path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.rdm_path = rdm_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return img_name, image

# =============================================================================
# Setup
# =============================================================================

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

# =============================================================================
# Dataloaders
# =============================================================================

def get_dataloaders(data_path, batch_size, num_workers, world_size, rank,
                   perturbation_type=None, epsilon=0.1, shuffle_seed=42):
    """
    Create train and val dataloaders with optional perturbation.

    Args:
        perturbation_type: None, 'gaussian', 'uniform_gray', 'label_shuffle', or 'target_noise'
        epsilon: Used only for gaussian noise
    """

    base_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Apply transform based on perturbation type
    if perturbation_type == 'gaussian':
        train_transform = GaussianNoiseTransform(base_train_transform, epsilon=epsilon)
    elif perturbation_type == 'uniform_gray':
        train_transform = UniformGrayTransform(base_train_transform)
    else:
        train_transform = base_train_transform

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

    # Apply label perturbations if requested
    if perturbation_type == 'label_shuffle':
        train_dataset = ShuffledLabelsDataset(train_dataset, shuffle_seed=shuffle_seed)
    elif perturbation_type == 'target_noise':
        train_dataset = TargetNoiseDataset(train_dataset, num_classes=1000, noise_seed=shuffle_seed)

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

# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(model, train_loader, optimizer, scaler, epoch, local_rank, world_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

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
            print(f"  [{batch_idx:4d}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_loss_tensor = torch.tensor(avg_loss).cuda(local_rank)
    dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss_tensor.item() / world_size

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

    return global_loss, global_acc

def compute_rsa_score(model, things_loader, rdm_path, local_rank, world_size):
    """Compute RSA score using THINGS 48 images"""
    model.eval()
    image_names = []
    embeddings = []

    with torch.no_grad():
        for img_names, images in things_loader:
            images = images.cuda(local_rank, non_blocking=True)

            if hasattr(model, 'module'):
                features = model.module.forward_features(images)
            else:
                features = model.forward_features(images)

            if hasattr(model, 'module'):
                if hasattr(model.module, 'global_pool') and model.module.global_pool == 'avg':
                    features = features[:, 1:].mean(dim=1)
                else:
                    features = features[:, 0]
            else:
                if hasattr(model, 'global_pool') and model.global_pool == 'avg':
                    features = features[:, 1:].mean(dim=1)
                else:
                    features = features[:, 0]

            image_names.extend(img_names)
            embeddings.extend(features.cpu().numpy())

    if world_size > 1:
        embeddings_tensor = torch.tensor(embeddings).cuda(local_rank)
        gathered_embeddings = [torch.zeros_like(embeddings_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_embeddings, embeddings_tensor)

        if local_rank == 0:
            all_embeddings = torch.cat(gathered_embeddings, dim=0).cpu().numpy()
            embeddings = all_embeddings[:48]
        else:
            return None, None
    else:
        embeddings = np.array(embeddings)

    if local_rank == 0:
        model_rdm = 1 - np.corrcoef(embeddings)
        np.fill_diagonal(model_rdm, 0)

        reference_rdm_dict = scipy.io.loadmat(rdm_path)
        reference_rdm = reference_rdm_dict['RDM48_triplet']

        upper_tri_indices = np.triu_indices_from(reference_rdm, k=1)
        reference_values = reference_rdm[upper_tri_indices]
        model_values = model_rdm[upper_tri_indices]

        rho, p_value = spearmanr(reference_values, model_values)

        return rho, p_value
    else:
        return None, None

# =============================================================================
# Scheduler
# =============================================================================

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
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * lr_scale
        else:
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

# =============================================================================
# Main measurement function
# =============================================================================

def measure_perturbation_effect(perturb_epoch, perturbation_type, baseline_checkpoint_dir,
                                baseline_metrics_csv, data_path, things_csv, things_img_dir,
                                things_rdm_path, epsilon, batch_size, lr, momentum, weight_decay,
                                warmup_epochs, total_epochs, num_workers, rank, world_size, local_rank):
    """
    Measure the immediate effect of a single-epoch perturbation.

    Returns:
        dict with keys: perturb_epoch, perturbation_type, baseline_loss, baseline_rsa,
                       perturbed_loss, perturbed_rsa, delta_loss, delta_rsa
    """

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"Measuring: {perturbation_type} @ epoch {perturb_epoch}")
        print(f"{'='*80}")

    # Load baseline metrics to get expected values
    baseline_df = pd.read_csv(baseline_metrics_csv)
    baseline_row = baseline_df[baseline_df['epoch'] == perturb_epoch]

    if baseline_row.empty:
        if rank == 0:
            print(f"⚠ No baseline data for epoch {perturb_epoch}")
        return None

    baseline_loss = baseline_row['val_loss'].values[0]
    baseline_rsa = baseline_row['rsa_score'].values[0]

    if rank == 0:
        print(f"Baseline @ epoch {perturb_epoch}: loss={baseline_loss:.4f}, RSA={baseline_rsa:.4f}")

    # Create THINGS dataloader for RSA
    things_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    things_dataset = THINGSInferenceDataset(
        csv_file=things_csv,
        img_dir=things_img_dir,
        rdm_path=things_rdm_path,
        transform=things_transform
    )

    things_sampler = DistributedSampler(
        things_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    things_loader = DataLoader(
        things_dataset,
        batch_size=8,
        sampler=things_sampler,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1000)
    model = model.cuda(local_rank)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Load checkpoint from (perturb_epoch - 1)
    checkpoint_path = os.path.join(baseline_checkpoint_dir, f'checkpoint_epoch_{perturb_epoch-1:03d}.pth')

    if not os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
    if world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create optimizer and scheduler
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    scheduler = CosineAnnealingLRWithWarmup(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=total_epochs,
        eta_min=0
    )

    scaler = GradScaler()

    # Load states
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Create perturbed dataloader
    train_loader, val_loader, train_sampler = get_dataloaders(
        data_path, batch_size, num_workers, world_size, rank,
        perturbation_type=perturbation_type,
        epsilon=epsilon,
        shuffle_seed=42
    )

    train_sampler.set_epoch(perturb_epoch)

    # Train ONE perturbed epoch
    if rank == 0:
        print(f"Training perturbed epoch {perturb_epoch}...")

    train_loss = train_one_epoch(model, train_loader, optimizer, scaler, perturb_epoch, local_rank, world_size)
    scheduler.step()

    # Evaluate
    if rank == 0:
        print(f"Evaluating...")

    val_loss, val_acc = validate(model, val_loader, local_rank, world_size)
    rsa_score, _ = compute_rsa_score(model, things_loader, things_rdm_path, local_rank, world_size)

    if rsa_score is None:
        rsa_score = 0.0

    # Compute deltas
    delta_loss = val_loss - baseline_loss
    delta_rsa = rsa_score - baseline_rsa

    if rank == 0:
        print(f"Perturbed: loss={val_loss:.4f}, RSA={rsa_score:.4f}")
        print(f"Δ loss={delta_loss:+.4f}, Δ RSA={delta_rsa:+.4f}")

    result = {
        'perturb_epoch': perturb_epoch,
        'perturbation_type': perturbation_type,
        'baseline_loss': baseline_loss,
        'baseline_rsa': baseline_rsa,
        'perturbed_loss': val_loss,
        'perturbed_rsa': rsa_score,
        'delta_loss': delta_loss,
        'delta_rsa': delta_rsa
    }

    return result

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Measure single-epoch perturbation effects on ViT')
    parser.add_argument('--baseline_checkpoint_dir', type=str, required=True,
                       help='Directory containing baseline checkpoints')
    parser.add_argument('--baseline_metrics_csv', type=str, required=True,
                       help='Path to baseline training_metrics.csv')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to ImageNet data')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='Output CSV file for results')

    # THINGS RSA arguments
    parser.add_argument('--things_csv', type=str, required=True,
                       help='Path to THINGS inference CSV file')
    parser.add_argument('--things_img_dir', type=str, required=True,
                       help='Directory containing THINGS images')
    parser.add_argument('--things_rdm_path', type=str, required=True,
                       help='Path to behavioral RDM .mat file')

    # Perturbation settings
    parser.add_argument('--perturbation_types', type=str, nargs='+',
                       default=['gaussian', 'uniform_gray', 'label_shuffle', 'target_noise'],
                       help='Perturbation types to test')
    parser.add_argument('--perturb_epochs', type=int, nargs='+',
                       default=[5, 10, 15, 16, 20, 25, 30, 35, 45, 70, 98],
                       help='Epochs to test perturbations at')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Perturbation strength for gaussian noise')

    # Training hyperparameters (must match baseline)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--total_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print("="*80)
        print("ViT Single-Epoch Perturbation Effect Measurement")
        print("="*80)
        print(f"Baseline checkpoint dir: {args.baseline_checkpoint_dir}")
        print(f"Baseline metrics CSV: {args.baseline_metrics_csv}")
        print(f"Perturbation types: {args.perturbation_types}")
        print(f"Perturbation epochs: {args.perturb_epochs}")
        print(f"Output CSV: {args.output_csv}")
        print("="*80)

    if world_size > 1:
        dist.barrier()

    # Collect all results
    all_results = []

    # Run measurements for all combinations
    for perturb_epoch in args.perturb_epochs:
        # Skip epoch 0 (no prior checkpoint)
        if perturb_epoch == 0:
            continue

        for perturbation_type in args.perturbation_types:
            result = measure_perturbation_effect(
                perturb_epoch=perturb_epoch,
                perturbation_type=perturbation_type,
                baseline_checkpoint_dir=args.baseline_checkpoint_dir,
                baseline_metrics_csv=args.baseline_metrics_csv,
                data_path=args.data_path,
                things_csv=args.things_csv,
                things_img_dir=args.things_img_dir,
                things_rdm_path=args.things_rdm_path,
                epsilon=args.epsilon,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                total_epochs=args.total_epochs,
                num_workers=args.num_workers,
                rank=rank,
                world_size=world_size,
                local_rank=local_rank
            )

            if result is not None:
                all_results.append(result)

    # Save results (only rank 0)
    if rank == 0:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"\n✓ Saved results to {args.output_csv}")
        print(f"\nResults summary:")
        print(results_df.to_string(index=False))

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
