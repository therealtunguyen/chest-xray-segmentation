import json
import numpy as np
import random
from pathlib import Path
from typing import List
from tqdm import tqdm
import pickle
import warnings

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from utils import (
    SegmentationAugmentation,
    extract_patch_features,
    create_gabor_filters,
    load_dataset,
    extract_training_features,
    calculate_metrics,
    calculate_per_class_metrics
)


# ==============================================================================
# Prediction Function
# ==============================================================================
def predict_image(image: np.ndarray, svm_model: SVC, scaler: StandardScaler,
                  gabor_filters: List[np.ndarray], patch_size: int = 32) -> np.ndarray:
    """Predict segmentation mask for a single image using sliding window"""
    h, w = image.shape
    pred_mask = np.zeros((h, w), dtype=np.int32)
    vote_count = np.zeros((h, w), dtype=np.int32)

    # Pad image
    pad = patch_size // 2
    image_padded = np.pad(image, pad, mode='reflect')

    stride = patch_size // 2  # Overlapping predictions

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract patch from padded image
            patch = image_padded[y:y+patch_size, x:x+patch_size]

            if patch.shape != (patch_size, patch_size):
                continue

            # Extract features
            try:
                features = extract_patch_features(patch, gabor_filters)
                features_scaled = scaler.transform(features.reshape(1, -1))
                pred_class = svm_model.predict(features_scaled)[0]

                # Assign prediction to center region
                y_end = min(y + stride, h)
                x_end = min(x + stride, w)
                pred_mask[y:y_end, x:x_end] = pred_class
                vote_count[y:y_end, x:x_end] += 1
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
    print("Baseline 2: SVM for Binary Lung Segmentation")
    print("  - HOG (Histogram of Oriented Gradients)")
    print("  - LBP (Local Binary Pattern)")
    print("  - Gabor Filters")
    print("=" * 60)

    # Load dataset
    images, masks = load_dataset(
        image_dir, mask_dir, image_size=(512, 512), num_classes=2)
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

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM
    print("\nTraining SVM classifier...")
    print("  Kernel: RBF")
    print("  This may take a while...")

    svm_model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        verbose=True,
        max_iter=10000
    )
    svm_model.fit(X_train_scaled, y_train)
    print("SVM training completed!")

    # Save model
    save_dir = Path("./outputs/svm/")
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / "svm_model.pkl", 'wb') as f:
        pickle.dump({'model': svm_model, 'scaler': scaler,
                    'gabor_filters': gabor_filters}, f)
    print(f"Model saved to {save_dir / 'svm_model.pkl'}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

    all_pred = []
    all_target = []

    for img, mask in tqdm(zip(test_images, test_masks), desc="Predicting", total=len(test_images)):
        pred_mask = predict_image(
            img, svm_model, scaler, gabor_filters, patch_size=32)
        all_pred.append(pred_mask)
        all_target.append(mask)

    # Calculate metrics
    all_pred = np.array(all_pred)
    all_target = np.array(all_target)

    test_metrics = calculate_per_class_metrics(
        all_pred, all_target, num_classes=2)

    print("\nTest Results (Overall):")
    print(f"  Dice Coefficient: {test_metrics['overall']['dice']:.4f}")
    print(f"  IoU Score:        {test_metrics['overall']['iou']:.4f}")
    print(f"  Pixel Accuracy:   {test_metrics['overall']['pixel_acc']:.4f}")
    print(f"  Sensitivity:      {test_metrics['overall']['sensitivity']:.4f}")
    print(f"  Specificity:      {test_metrics['overall']['specificity']:.4f}")

    print("\nPer-Class Results:")
    for class_name, metrics in test_metrics['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  Dice: {metrics['dice']:.4f}, IoU: {metrics['iou']:.4f}")
        print(
            f"  Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f}")

    # Save results
    with open(save_dir / "test_results.json", 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nResults saved to {save_dir / 'test_results.json'}")
