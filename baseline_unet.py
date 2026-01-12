import os
import json
import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from utils import SegmentationAugmentation


# ==============================================================================
# Dataset
# ==============================================================================
class ChestXRaySegmentationDataset(Dataset):
    """Dataset for chest X-ray semantic segmentation"""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (512, 512),
        num_classes: int = 4
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_size = image_size
        self.num_classes = num_classes
        self.images = sorted([f for f in os.listdir(image_dir)])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        mask_name = img_name

        # Load image
        img_path = self.image_dir / img_name
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Load mask
        mask_path = self.mask_dir / mask_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Resize if needed
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size,
                               interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != self.image_size:
            mask = cv2.resize(mask, self.image_size,
                              interpolation=cv2.INTER_NEAREST)

        # Normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = np.clip(mask, 0, self.num_classes - 1)

        # Apply augmentation if transform is provided
        if self.transform:
            image, mask = self.transform(image, mask)

        # Convert to tensors
        if len(image.shape) == 2:
            image = image[np.newaxis, ...]  # Add channel dim
        image = torch.from_numpy(image).float()  # (1, H, W)
        mask = torch.from_numpy(mask).long().unsqueeze(0)  # (1, H, W)

        return image, mask


# ==============================================================================
# Standard U-Net Model (No Edge Enhancement)
# ==============================================================================
class UNetBlock(nn.Module):
    """Standard double convolution block"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class StandardUNet(nn.Module):
    """
    Standard U-Net implementation without any edge enhancement.
    - Encoder: 4 levels with max pooling
    - Decoder: 4 levels with transposed convolutions
    - Skip connections via concatenation
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        features: List[int] = [64, 128, 256, 512]
    ):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder path
        prev_channels = in_channels
        for feature in features:
            self.encoder.append(UNetBlock(prev_channels, feature))
            prev_channels = feature

        # Bottleneck
        self.bottleneck = UNetBlock(features[-1], features[-1] * 2)

        # Decoder path
        reversed_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feature in reversed_features:
            self.decoder.append(
                nn.ConvTranspose2d(prev_channels, feature,
                                   kernel_size=2, stride=2)
            )
            self.decoder.append(UNetBlock(feature * 2, feature))
            prev_channels = feature

        # Output layer
        self.output = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections.reverse()
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Upsample
            skip = skip_connections[i // 2]

            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i + 1](x)  # Conv block

        return self.output(x)


# ==============================================================================
# Loss Functions
# ==============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target.squeeze(
            1).long(), num_classes=num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred_flat = pred.reshape(-1)
        target_flat = target_one_hot.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice_coeff


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target.squeeze(1))
        return self.dice_weight * dice + self.ce_weight * ce


# ==============================================================================
# Metrics Calculation
# ==============================================================================
def calculate_metrics(pred: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1e-6) -> Dict[str, float]:
    """Calculate segmentation metrics: Dice, IoU, Pixel Accuracy, Sensitivity, Specificity"""
    pred_softmax = torch.softmax(pred, dim=1)
    pred_argmax = torch.argmax(pred_softmax, dim=1)
    target_squeezed = target.squeeze(1).long()

    # Pixel Accuracy
    correct = (pred_argmax == target_squeezed).long().sum()
    total = target_squeezed.numel()
    pixel_acc = (correct.float() / total).item()

    # Per-class metrics
    dice_sum, iou_sum, sens_sum, spec_sum = 0.0, 0.0, 0.0, 0.0
    for cls in range(num_classes):
        pred_cls = (pred_argmax == cls).float()
        target_cls = (target_squeezed == cls).float()

        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        tn = ((1 - pred_cls) * (1 - target_cls)).sum()

        dice_sum += (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
        iou_sum += (tp + smooth) / (tp + fp + fn + smooth)
        sens_sum += (tp + smooth) / (tp + fn + smooth)
        spec_sum += (tn + smooth) / (tn + fp + smooth)

    return {
        'dice': (dice_sum / num_classes).item(),
        'iou': (iou_sum / num_classes).item(),
        'pixel_acc': pixel_acc,
        'sensitivity': (sens_sum / num_classes).item(),
        'specificity': (spec_sum / num_classes).item()
    }


# ==============================================================================
# Trainer
# ==============================================================================
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 50,
        save_dir: str = "./checkpoints_baseline_unet"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.num_classes = 4

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'train_pixel_acc': [], 'val_pixel_acc': [],
            'train_sensitivity': [], 'val_sensitivity': [],
            'train_specificity': [], 'val_specificity': []
        }

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        metrics_sum = {'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0,
                       'sensitivity': 0.0, 'specificity': 0.0}
        num_batches = 0

        for data, target in tqdm(self.train_loader, desc="Training"):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_metrics = calculate_metrics(output, target, self.num_classes)
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
            num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        avg_metrics['loss'] = total_loss / num_batches
        return avg_metrics

    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        metrics_sum = {'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0,
                       'sensitivity': 0.0, 'specificity': 0.0}
        num_batches = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                batch_metrics = calculate_metrics(
                    output, target, self.num_classes)
                for k, v in batch_metrics.items():
                    metrics_sum[k] += v
                num_batches += 1

        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}
        avg_metrics['loss'] = total_loss / num_batches
        return avg_metrics

    def train(self) -> Dict[str, Any]:
        print(
            f"Starting Standard U-Net training for {self.num_epochs} epochs...")
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)

            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
                  f"IoU: {train_metrics['iou']:.4f}, Acc: {train_metrics['pixel_acc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}, Acc: {val_metrics['pixel_acc']:.4f}")

            # Update history
            for key in ['loss', 'dice', 'iou', 'pixel_acc', 'sensitivity', 'specificity']:
                self.history[f'train_{key}'].append(train_metrics[key])
                self.history[f'val_{key}'].append(val_metrics[key])

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_metrics': val_metrics
                }, self.save_dir / "best_model.pth")
                print(f"Saved best model (val_loss: {best_val_loss:.4f})")

        # Save final model and history
        torch.save({'model_state_dict': self.model.state_dict(), 'history': self.history},
                   self.save_dir / "final_model.pth")
        with open(self.save_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f)

        print("Training completed!")
        return self.history


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Dataset paths (adjust for your environment)
    # input_path = Path("/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt")

    # For local testing, use:
    input_path = Path("./data")

    if (input_path / "cxr").exists() and (input_path / "masks").exists():
        image_dir = input_path / "cxr"
        mask_dir = input_path / "masks"
    else:
        raise ValueError(f"Expected 'cxr' and 'masks' folders in {input_path}")

    print("=" * 60)
    print("Baseline 1: Standard U-Net (No Edge Enhancement)")
    print("=" * 60)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create augmentation for training
    train_augmentation = SegmentationAugmentation(image_size=(512, 512))

    # Create datasets (with augmentation for training, without for val/test)
    train_full_dataset = ChestXRaySegmentationDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        transform=train_augmentation,
        image_size=(512, 512),
        num_classes=4
    )
    val_test_dataset = ChestXRaySegmentationDataset(
        image_dir=str(image_dir),
        mask_dir=str(mask_dir),
        transform=None,  # No augmentation for validation/test
        image_size=(512, 512),
        num_classes=4
    )
    print(f"Total samples: {len(train_full_dataset)}")

    # Train/Val/Test split
    dataset_size = len(train_full_dataset)
    indices = list(range(dataset_size))
    random.seed(42)
    random.shuffle(indices)

    split_idx_1 = int(dataset_size * 0.70)
    split_idx_2 = int(dataset_size * 0.85)
    train_indices = indices[:split_idx_1]
    val_indices = indices[split_idx_1:split_idx_2]
    test_indices = indices[split_idx_2:]

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_test_dataset, val_indices)
    test_dataset = Subset(val_test_dataset, test_indices)

    print(
        f"Split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    # Data loaders
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = StandardUNet(in_channels=1, num_classes=4).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=50,
        save_dir="./checkpoints_baseline_unet"
    )
    history = trainer.train()

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    model.eval()
    test_metrics = {'dice': 0.0, 'iou': 0.0, 'pixel_acc': 0.0,
                    'sensitivity': 0.0, 'specificity': 0.0}
    num_batches = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            batch_metrics = calculate_metrics(output, target, 4)
            for k, v in batch_metrics.items():
                test_metrics[k] += v
            num_batches += 1

    for k in test_metrics:
        test_metrics[k] /= num_batches

    print("\nTest Results:")
    print(f"  Dice Coefficient: {test_metrics['dice']:.4f}")
    print(f"  IoU Score:        {test_metrics['iou']:.4f}")
    print(f"  Pixel Accuracy:   {test_metrics['pixel_acc']:.4f}")
    print(f"  Sensitivity:      {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity:      {test_metrics['specificity']:.4f}")
