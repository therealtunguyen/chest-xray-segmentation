import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from skimage.feature import hog, local_binary_pattern
import albumentations as A


# ==============================================================================
# Data Augmentation
# ==============================================================================
class SegmentationAugmentation:
    """Augmentation class for segmentation tasks using albumentations"""

    def __init__(
        self,
        rotation_limit: int = 15,
        flip_probability: float = 0.5,
        elastic_alpha: float = 100,
        elastic_sigma: float = 10,
        elastic_alpha_affine: float = 10,
        elastic_probability: float = 0.3,
        noise_var_limit: Tuple[float, float] = (1e-4, 1e-3),
        noise_probability: float = 0.2,
        brightness_limit: float = 0.1,
        contrast_limit: float = 0.1,
        brightness_contrast_probability: float = 0.3,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.augmentation = A.Compose([
            A.Rotate(limit=rotation_limit, p=flip_probability),
            A.HorizontalFlip(p=flip_probability),
            A.ElasticTransform(
                alpha=elastic_alpha,
                sigma=elastic_sigma,
                p=elastic_probability
            ),
            A.GaussNoise(p=noise_probability),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=brightness_contrast_probability
            ),
            A.Resize(height=image_size[0], width=image_size[1])
        ])

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation to image and mask"""
        augmented = self.augmentation(image=image, mask=mask)
        return augmented['image'], augmented['mask']


# ==============================================================================
# Feature Extraction Functions
# ==============================================================================
def extract_hog_features(image: np.ndarray, pixels_per_cell: Tuple[int, int] = (16, 16),
                         cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """Extract HOG features from an image patch"""
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        visualize=False,
        feature_vector=True
    )
    return features


def extract_lbp_features(image: np.ndarray, radius: int = 3, n_points: int = 24,
                         n_bins: int = 26) -> np.ndarray:
    """Extract LBP histogram features from an image patch"""
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                           range=(0, n_bins), density=True)
    return hist


def create_gabor_filters(num_orientations: int = 8, num_scales: int = 5) -> List[np.ndarray]:
    """Create a bank of Gabor filters"""
    filters = []
    for theta in range(num_orientations):
        theta_rad = theta * np.pi / num_orientations
        for scale in range(1, num_scales + 1):
            sigma = scale * 2
            lambd = sigma * 2
            kernel = cv2.getGaborKernel(
                ksize=(31, 31),
                sigma=sigma,
                theta=theta_rad,
                lambd=lambd,
                gamma=0.5,
                psi=0
            )
            filters.append(kernel)
    return filters


def extract_gabor_features(image: np.ndarray, gabor_filters: List[np.ndarray]) -> np.ndarray:
    """Extract Gabor filter responses (mean and std for each filter)"""
    features = []
    for kernel in gabor_filters:
        filtered = cv2.filter2D(image, cv2.CV_64F, kernel)
        features.append(filtered.mean())
        features.append(filtered.std())
    return np.array(features)


def extract_patch_features(patch: np.ndarray, gabor_filters: List[np.ndarray]) -> np.ndarray:
    """Extract all hand-crafted features from a patch"""
    # HOG features
    hog_feat = extract_hog_features(patch)

    # LBP features
    lbp_feat = extract_lbp_features(patch)

    # Gabor features
    gabor_feat = extract_gabor_features(patch, gabor_filters)

    # Intensity features
    intensity_feat = np.array([
        patch.mean(),
        patch.std(),
        patch.min(),
        patch.max()
    ])

    # Concatenate all features
    return np.concatenate([hog_feat, lbp_feat, gabor_feat, intensity_feat])


# ==============================================================================
# Dataset Loading
# ==============================================================================
def load_dataset(image_dir: Path, mask_dir: Path, image_size: Tuple[int, int] = (512, 512),
                 num_classes: int = 4) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load all images and masks"""
    images = []
    masks = []

    image_files = sorted(os.listdir(image_dir))

    for img_name in tqdm(image_files, desc="Loading dataset"):
        # Load image
        img_path = image_dir / img_name
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        # Load mask
        mask_path = mask_dir / img_name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            continue

        # Resize
        if image.shape[:2] != image_size:
            image = cv2.resize(image, image_size,
                               interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != image_size:
            mask = cv2.resize(mask, image_size,
                              interpolation=cv2.INTER_NEAREST)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        mask = np.clip(mask, 0, num_classes - 1)

        images.append(image)
        masks.append(mask)

    return images, masks


def extract_training_features(images: List[np.ndarray], masks: List[np.ndarray],
                              gabor_filters: List[np.ndarray],
                              patch_size: int = 32,
                              stride: int = 32,
                              max_samples_per_image: int = 100,
                              augmentation: SegmentationAugmentation = None,
                              num_augmentations: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from image patches for training.
    Uses patch-based approach for computational efficiency.
    Supports data augmentation for increased training data.
    """
    all_features = []
    all_labels = []

    for img, mask in tqdm(zip(images, masks), desc="Extracting features", total=len(images)):
        # Process original image and augmented versions
        images_to_process = [(img, mask)]

        # Add augmented versions if augmentation is provided
        if augmentation is not None:
            for _ in range(num_augmentations):
                aug_img, aug_mask = augmentation(img, mask)
                images_to_process.append((aug_img, aug_mask))

        for curr_img, curr_mask in images_to_process:
            h, w = curr_img.shape
            samples_collected = 0

            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    if samples_collected >= max_samples_per_image:
                        break

                    # Extract patch
                    patch = curr_img[y:y+patch_size, x:x+patch_size]
                    mask_patch = curr_mask[y:y+patch_size, x:x+patch_size]

                    # Get majority class in the patch as label
                    label = int(np.bincount(
                        mask_patch.flatten().astype(int)).argmax())

                    # Extract features
                    try:
                        features = extract_patch_features(patch, gabor_filters)
                        all_features.append(features)
                        all_labels.append(label)
                        samples_collected += 1
                    except Exception:
                        continue

                if samples_collected >= max_samples_per_image:
                    break

    return np.array(all_features), np.array(all_labels)


# ==============================================================================
# Metrics Calculation
# ==============================================================================
def calculate_metrics(pred: np.ndarray, target: np.ndarray, num_classes: int = 4) -> Dict[str, float]:
    """Calculate segmentation metrics"""
    # Pixel Accuracy
    pixel_acc = accuracy_score(target.flatten(), pred.flatten())

    # Per-class metrics
    dice_scores = []
    iou_scores = []
    sensitivities = []
    specificities = []

    smooth = 1e-6

    for cls in range(num_classes):
        pred_cls = (pred == cls).astype(float)
        target_cls = (target == cls).astype(float)

        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        tn = ((1 - pred_cls) * (1 - target_cls)).sum()

        dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        sensitivity = (tp + smooth) / (tp + fn + smooth)
        specificity = (tn + smooth) / (tn + fp + smooth)

        dice_scores.append(dice)
        iou_scores.append(iou)
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'pixel_acc': pixel_acc,
        'sensitivity': np.mean(sensitivities),
        'specificity': np.mean(specificities)
    }
