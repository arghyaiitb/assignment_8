from __future__ import print_function
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.pyplot as plt
from model import ResNet34
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import os
from pathlib import Path
import sys
from datetime import datetime
import gradcam_utils
from torch.distributions.beta import Beta


# Create checkpoint directory
CHECKPOINT_DIR = Path('./checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create logging file
LOG_FILE = Path('./training_logs_v4.md')

# CIFAR-100 class names
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Initialize logger
sys.stdout = Logger(LOG_FILE)

# Write header to log file
print(f"\n{'='*80}")
print(f"# Training Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")
print("## Critical Changes Applied (Based on Analysis & ref_model)")
print("✅ **MaxPool Removed**: Preserves 32x32 spatial resolution through layer1")
print("✅ **Dropout Disabled**: Set to 0.0 in residual blocks (was 0.15)")
print("✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0")
print("✅ **Enhanced Augmentation**: Added PadIfNeeded+RandomCrop, GaussNoise")
print("✅ **Deeper Architecture**: Changed to [5,5,5,5] blocks (20 total, was 16)")
print("✅ **Increased FC Dropout**: 0.3 (was 0.25)")
print("✅ **Optimized LR Schedule**: max_lr=0.1, div_factor=25, final_div_factor=1000")
print("   └─ Fixes volatility issue, enables better convergence")
print("Expected Impact: +15-22% test accuracy improvement (TARGET: 75%+)")
print()


# Custom transform class for albumentations
class AlbumentationTransforms:
    def __init__(self, transforms_list):
        self.transforms = A.Compose(transforms_list)

    def __call__(self, img):
        img = np.array(img)
        return self.transforms(image=img)['image']

# Train Phase transformations with Albumentations (Enhanced with ref_model improvements)
train_transforms = AlbumentationTransforms([
    # IMPROVED: Pad and RandomCrop for better spatial variation (from ref_model)
    A.PadIfNeeded(min_height=40, min_width=40, border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
    A.RandomCrop(32, 32, p=1.0),
    
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=(0.0, 0.1), scale=(1.0-0.15, 1.0+0.15), rotate=(-15, 15), p=0.5),
    
    # IMPROVED: Better CoarseDropout + GaussNoise in OneOf (from ref_model)
    A.OneOf([
        A.CoarseDropout(
            num_holes_range=(1, 1),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            fill=tuple([int(x * 255) for x in [0.5071, 0.4867, 0.4408]]),
            p=1.0
        ),
        A.GaussNoise(std_range=(0.0125, 0.0278), p=1.0),
    ], p=0.3),
    
    A.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ToTensorV2()
])

# Test Phase transformations with torchvision (match reference)
test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

train = datasets.CIFAR100('./data', train=True, download=True, transform=train_transforms)
test = datasets.CIFAR100('./data', train=False, download=True, transform=test_transforms)


SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("cuDNN benchmark enabled for maximum speed")

# Optimized dataloader arguments for T4 GPU
train_dataloader_args = dict(
    shuffle=True, 
    batch_size=128,
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive between epochs
    prefetch_factor=4  # Increased prefetch for faster loading
) if cuda else dict(shuffle=True, batch_size=64)

test_dataloader_args = dict(
    shuffle=False, 
    batch_size=128,
    num_workers=4, 
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
) if cuda else dict(shuffle=False, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **train_dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **test_dataloader_args)


train_losses = []
test_losses = []
train_acc = []
test_acc = []
lrs = []


def save_and_plot_metrics(output_dir: Path):
    """Save metrics (CSV + NPZ) and generate overview plots (2x2 grid and individual).

    Files written under output_dir:
      - metrics.csv (epoch, train_loss, test_loss, train_acc, test_acc, lr)
      - metrics.npz (numpy arrays)
      - metrics_overview.png (2x2 grid)
      - train_loss.png, test_loss.png, train_acc.png, test_acc.png
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    epochs_axis = list(range(len(train_losses)))

    # Save CSV
    csv_path = output_dir / 'metrics.csv'
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,test_loss,train_acc,test_acc,lr\n')
        for i in range(len(epochs_axis)):
            tl = train_losses[i] if i < len(train_losses) else ''
            tsl = test_losses[i] if i < len(test_losses) else ''
            tra = train_acc[i] if i < len(train_acc) else ''
            tea = test_acc[i] if i < len(test_acc) else ''
            lr_i = lrs[i] if i < len(lrs) else ''
            f.write(f"{i},{tl},{tsl},{tra},{tea},{lr_i}\n")

    # Save NPZ for programmatic reuse
    np.savez(output_dir / 'metrics.npz',
             train_losses=np.array(train_losses, dtype=float),
             test_losses=np.array(test_losses, dtype=float),
             train_acc=np.array(train_acc, dtype=float),
             test_acc=np.array(test_acc, dtype=float),
             lrs=np.array(lrs, dtype=float))

    # Individual plots
    def _save_line(x, y, title, ylabel, path):
        plt.figure(figsize=(6, 4))
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        plt.close()

    _save_line(epochs_axis, train_losses, 'Training Loss', 'Loss', output_dir / 'train_loss.png')
    _save_line(epochs_axis, test_losses, 'Test Loss', 'Loss', output_dir / 'test_loss.png')
    _save_line(epochs_axis, train_acc, 'Training Accuracy', 'Accuracy (%)', output_dir / 'train_acc.png')
    _save_line(epochs_axis, test_acc, 'Test Accuracy', 'Accuracy (%)', output_dir / 'test_acc.png')

    # 2x2 overview
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs_axis, train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_axis, test_losses)
    axes[0, 1].set_title('Test Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs_axis, train_acc)
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_axis, test_acc)
    axes[1, 1].set_title('Test Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'metrics_overview.png', dpi=200)
    plt.close(fig)


def train_epoch(model, device, train_loader, optimizer, epoch, scaler=None, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    running_loss = 0.0
    
    # CutMix hyperparameters
    cutmix_prob = 0.3  # Apply CutMix with 30% probability
    cutmix_alpha = 1.0  # Beta distribution parameter
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Decide whether to apply CutMix for this batch
        use_cutmix = np.random.rand() < cutmix_prob
        
        if use_cutmix:
            # Sample lambda from Beta distribution
            lam = Beta(cutmix_alpha, cutmix_alpha).sample().item()
            batch_size = data.size(0)
            index = torch.randperm(batch_size).to(device)
            
            # Generate random bounding box
            H, W = data.size(2), data.size(3)
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            # Uniform random center point
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            # Bounding box coordinates
            x1 = np.clip(cx - cut_w // 2, 0, W)
            y1 = np.clip(cy - cut_h // 2, 0, H)
            x2 = np.clip(cx + cut_w // 2, 0, W)
            y2 = np.clip(cy + cut_h // 2, 0, H)
            
            # Apply CutMix: replace patch with shuffled batch
            data[:, :, y1:y2, x1:x2] = data[index, :, y1:y2, x1:x2]
            
            # Adjust lambda to actual area ratio
            lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
            
            # Mixed precision training with CutMix
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    # Mixed loss: NO label smoothing (CutMix already provides soft labels)
                    loss = lam * F.cross_entropy(output, target) + \
                           (1 - lam) * F.cross_entropy(output, target[index])
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                # Mixed loss: NO label smoothing (CutMix already provides soft labels)
                loss = lam * F.cross_entropy(output, target) + \
                       (1 - lam) * F.cross_entropy(output, target[index])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            # Standard training without CutMix
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = F.cross_entropy(output, target, label_smoothing=0.1)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = F.cross_entropy(output, target, label_smoothing=0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        
        # Step scheduler after each batch for OneCycleLR
        if scheduler is not None:
            scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        running_loss += loss.item()
        
        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    train_accuracy = 100 * correct / processed
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_acc.append(train_accuracy)
    return train_accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    return test_loss


def save_checkpoint(model, optimizer, epoch, test_loss, test_accuracy, is_best=False, checkpoint_dir=CHECKPOINT_DIR):
    """Save training checkpoint with model state, optimizer state, and training metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }
    
    # Save periodic checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save best model checkpoint
    if is_best:
        best_model_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_model_path)
        print(f"Best model saved: {best_model_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load training checkpoint to resume training."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    test_loss = checkpoint['test_loss']
    test_accuracy = checkpoint['test_accuracy']
    
    # Restore training history
    global train_losses, test_losses, train_acc, test_acc
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    train_acc = checkpoint['train_acc']
    test_acc = checkpoint['test_acc']
    
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"Resuming from epoch: {epoch}, Test Accuracy: {test_accuracy:.2f}%")
    
    return model, optimizer, epoch, test_loss, test_accuracy


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

print("\n## Model Configuration")
print("- Architecture: ResNet34-Enhanced (Modified for CIFAR-100)")
print("- Block Structure: [5, 5, 5, 5] = 20 residual blocks (was [3,4,6,3]=16)")
print("- MaxPool: REMOVED (preserves 32x32 spatial resolution)")
print("- Dropout in residual blocks: 0.0 (disabled)")
print("- FC Layer Dropout: 0.3 (increased from 0.25)")
print("- Number of classes: 100")
print()

model = ResNet34(num_classes=100).to(device)

# Use channels_last memory format for better T4 performance
if cuda:
    model = model.to(memory_format=torch.channels_last)
    print("Using channels_last memory format for optimal T4 performance")

summary(model, input_size=(3, 32, 32))

# Improved optimizer with better convergence and stronger weight decay
# Note: Initial LR will be set by OneCycleLR scheduler (max_lr/div_factor)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

# Initialize mixed precision scaler for faster training
scaler = torch.amp.GradScaler('cuda') if cuda else None
if scaler:
    print("Mixed precision training enabled (AMP) for faster training")

# OneCycleLR for better convergence - proven to work well for CIFAR
# IMPROVED: Better scheduler params for stability and convergence
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.2,  # Warmup for 20% of training (40 epochs)
    div_factor=25,  # Initial lr = 0.1/25 = 0.004 (was 0.01, more stable warmup)
    final_div_factor=1000,  # Final lr = 0.1/1000 = 0.0001 (was 0.001, better convergence)
    anneal_strategy='cos'
)

print("\n## Training Hyperparameters")
print(f"- Epochs: 200")
print(f"- Batch Size: 128")
print(f"- Optimizer: SGD (momentum=0.9, weight_decay=5e-4, nesterov=True)")
print(f"- Scheduler: OneCycleLR (max_lr=0.1, initial_lr=0.004, final_lr=0.0001)")
print(f"- Label Smoothing: 0.1")
print(f"- Gradient Clipping: max_norm=1.0")
print(f"- Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
print(f"- CutMix: Enabled (prob=0.3, alpha=1.0)")
print(f"- Augmentation: PadIfNeeded(40x40)→RandomCrop(32x32), HFlip, ShiftScaleRotate, CoarseDropout/GaussNoise")
print()

EPOCHS = 200
CHECKPOINT_INTERVAL = 5

best_test_accuracy = 0.0

print("\n## Training Progress\n")
print("| Epoch | Train Acc | Test Acc | Test Loss | LR | Status |")
print("|-------|-----------|----------|-----------|-----|--------|")

for epoch in range(EPOCHS):
    print(f"\n### EPOCH: {epoch}")

    train_accuracy = train_epoch(model, device, train_loader, optimizer, epoch, scaler, scheduler)
    test_loss = test(model, device, test_loader)

    # Print current LR for monitoring
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    print(f"Current LR: {current_lr:.6f}")
    
    # Calculate current test accuracy
    current_test_accuracy = test_acc[-1] if test_acc else 0.0
    
    # Display Train and Test Accuracy Side by Side
    print(f"\n{'='*70}")
    print(f"EPOCH: {epoch:3d} | Train Accuracy: {train_accuracy:6.2f}% | Test Accuracy: {current_test_accuracy:6.2f}%")
    print(f"{'='*70}\n")
    
    # Add row to markdown table
    status = ""
    
    # Save checkpoint logic
    is_best = current_test_accuracy > best_test_accuracy
    if is_best:
        best_test_accuracy = current_test_accuracy
        status = "🎯 **BEST**"
    
    # Add table row
    print(f"| {epoch:5d} | {train_accuracy:8.2f}% | {current_test_accuracy:7.2f}% | {test_loss:8.4f} | {current_lr:.6f} | {status} |")
    
    # Save best model checkpoint
    if is_best:
        save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=True)
    
    # Save periodic checkpoint
    if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=False)
    
    # Early stopping if we hit target accuracy
    if current_test_accuracy >= 75.0:
        print(f"\n{'='*70}")
        print(f"🎯 TARGET ACHIEVED! Test Accuracy: {current_test_accuracy:.2f}% >= 75%")
        print(f"{'='*70}\n")
        save_checkpoint(model, optimizer, epoch, test_loss, current_test_accuracy, is_best=True)
        break

print("\n" + "="*80)
print("# Training Completed!")
print(f"**Best Test Accuracy: {best_test_accuracy:.2f}%**")
print(f"Training Session Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80 + "\n")

# Save metrics and plots
metrics_dir = Path('./training_metrics')
print("Saving training metrics and plots to ./training_metrics ...")
save_and_plot_metrics(metrics_dir)
print("✅ Saved metrics: CSV, NPZ and plots.")

# Visualize GradCAM for the best model
print("\n## Generating GradCAM Visualizations")
print("Loading best model for visualization...\n")

try:
    # Load best model
    best_model_path = CHECKPOINT_DIR / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded best model (Test Acc: {checkpoint.get('test_accuracy', 'N/A'):.2f}%)")
        
        # Find target layer (last conv layer in layer4)
        target_layer = None
        for name, module in model.named_modules():
            if 'layer4' in name and isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            print("⚠️  Could not find target layer, using auto-detection")
        else:
            print(f"✅ Using target layer for GradCAM visualization")
        
        # Generate GradCAM visualizations and save outputs
        print("\nGenerating GradCAM visualizations for 6 sample images...")
        output_dir = Path('./gradcam_outputs')
        output_dir.mkdir(exist_ok=True)
        gradcam_utils.show_gradcam_batch(
            model=model,
            device=device,
            loader=test_loader,
            classes=CIFAR100_CLASSES,
            target_layer=target_layer,
            n=6,
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
            save_dir=str(output_dir),
            filename_prefix='gradcam',
            save_grid=True,
            save_individual=True,
            dpi=200,
            show=False
        )
        print("✅ GradCAM visualizations saved to ./gradcam_outputs/")
        print("\n**Note**: Close the matplotlib window to continue and save logs.")
    else:
        print(f"⚠️  Best model checkpoint not found at {best_model_path}")
        print("Skipping GradCAM visualization.")
        
except ImportError as e:
    print(f"⚠️  GradCAM visualization skipped: {e}")
    print("To enable GradCAM, install: pip install grad-cam")
except Exception as e:
    print(f"⚠️  GradCAM visualization failed: {e}")
    print("Continuing with training completion...")

print("\n" + "="*80)

# Close log file properly
if hasattr(sys.stdout, 'log'):
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
    print(f"Training logs saved to: {LOG_FILE}")
