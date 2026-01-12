import json
import numpy as np
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import pickle
import warnings

from sklearn.ensemble import RandomForestClassifier

from utils import (
    SegmentationAugmentation,
    extract_patch_features,
    create_gabor_filters,
    load_dataset,
    extract_training_features,
    calculate_metrics
)


# ==============================================================================
# Prediction Function
# ==============================================================================
def predict_image(image: np.ndarray, rf_model: RandomForestClassifier,
                  gabor_filters: List[np.ndarray], patch_size: int = 32) -> np.ndarray:
    """Predict segmentation mask for a single image using sliding window"""
    h, w = image.shape
    pred_mask = np.zeros((h, w), dtype=np.int32)

    pad = patch_size // 2
    image_padded = np.pad(image, pad, mode='reflect')

    stride = patch_size // 2

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = image_padded[y:y+patch_size, x:x+patch_size]

            if patch.shape != (patch_size, patch_size):
                continue

            try:
                features = extract_patch_features(patch, gabor_filters)
                pred_class = rf_model.predict(features.reshape(1, -1))[0]

                y_end = min(y + stride, h)
                x_end = min(x + stride, w)
                pred_mask[y:y_end, x:x_end] = pred_class
            except:
                continue

    return pred_mask


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # Dataset paths
    # input_path = Path("/kaggle/input/jsrt-247-image-lung-segmentation-mask-dataset/content/jsrt")

    # For local testing:
    input_path = Path("./data")

    if (input_path / "cxr").exists() and (input_path / "masks").exists():
        image_dir = input_path / "cxr"
        mask_dir = input_path / "masks"
    else:
        raise ValueError(f"Expected 'cxr' and 'masks' folders in {input_path}")

    print("=" * 60)
    print("Baseline 3: Random Forest with Hand-crafted Features")
    print("  - 100 trees")
    print("  - Max depth: 20")
    print("  - Features: HOG, LBP, Gabor")
    print("=" * 60)

    # Load dataset
    images, masks = load_dataset(
        image_dir, mask_dir, image_size=(512, 512), num_classes=4)
    print(f"Total samples: {len(images)}")

    # Train/Val/Test split
    indices = list(range(len(images)))
    random.seed(42)
    random.shuffle(indices)

    split_idx_1 = int(len(images) * 0.70)
    split_idx_2 = int(len(images) * 0.85)

    train_indices = indices[:split_idx_1]
    val_indices = indices[split_idx_1:split_idx_2]
    test_indices = indices[split_idx_2:]

    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices]
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]

    print(
        f"Split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    # Create Gabor filter bank
    print("\nCreating Gabor filter bank...")
    gabor_filters = create_gabor_filters(num_orientations=8, num_scales=5)
    print(f"Created {len(gabor_filters)} Gabor filters")

    # Create augmentation for training
    print("\nCreating data augmentation...")
    augmentation = SegmentationAugmentation(image_size=(512, 512))
    print("Data augmentation enabled with: rotation, flip, elastic transform, noise, brightness/contrast")

    # Extract features for training (with augmentation)
    print("\nExtracting training features (with augmentation)...")
    X_train, y_train = extract_training_features(
        train_images, train_masks, gabor_filters,
        patch_size=32, stride=32, max_samples_per_image=100,
        augmentation=augmentation, num_augmentations=3
    )
    print(
        f"Training samples: {len(X_train)}, Feature dimension: {X_train.shape[1]}")

    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    print("  n_estimators: 100")
    print("  max_depth: 20")
    print("  class_weight: balanced")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=1
    )
    rf_model.fit(X_train, y_train)
    print("Random Forest training completed!")

    # Feature importance
    print("\nTop 10 Feature Importances:")
    feature_importance = rf_model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {feature_importance[idx]:.4f}")

    # Save model
    save_dir = Path("./output/random_forest/")
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "rf_model.pkl", 'wb') as f:
        pickle.dump({'model': rf_model, 'gabor_filters': gabor_filters}, f)
    print(f"\nModel saved to {save_dir / 'rf_model.pkl'}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    all_pred = []
    all_target = []

    for img, mask in tqdm(zip(test_images, test_masks), desc="Predicting", total=len(test_images)):
        pred_mask = predict_image(img, rf_model, gabor_filters, patch_size=32)
        all_pred.append(pred_mask)
        all_target.append(mask)

    # Calculate metrics
    all_pred = np.array(all_pred)
    all_target = np.array(all_target)

    test_metrics = calculate_metrics(all_pred, all_target, num_classes=4)

    print("\nTest Results:")
    print(f"  Dice Coefficient: {test_metrics['dice']:.4f}")
    print(f"  IoU Score:        {test_metrics['iou']:.4f}")
    print(f"  Pixel Accuracy:   {test_metrics['pixel_acc']:.4f}")
    print(f"  Sensitivity:      {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity:      {test_metrics['specificity']:.4f}")

    # Save results
    with open(save_dir / "test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {save_dir / 'test_results.json'}")

    # Per-class analysis
    print("\n" + "=" * 60)
    print("Per-Class Analysis")
    print("=" * 60)

    class_names = ["Background", "Lungs", "Heart", "Clavicles"]
    smooth = 1e-6

    for cls in range(4):
        pred_cls = (all_pred == cls).astype(float)
        target_cls = (all_target == cls).astype(float)

        tp = (pred_cls * target_cls).sum()
        fp = (pred_cls * (1 - target_cls)).sum()
        fn = ((1 - pred_cls) * target_cls).sum()
        tn = ((1 - pred_cls) * (1 - target_cls)).sum()

        dice = (2. * tp + smooth) / (2. * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        sens = (tp + smooth) / (tp + fn + smooth)
        spec = (tn + smooth) / (tn + fp + smooth)

        print(f"\n{class_names[cls]}:")
        print(f"  Dice: {dice:.4f}, IoU: {iou:.4f}")
        print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
