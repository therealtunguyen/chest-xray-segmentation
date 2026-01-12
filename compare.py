import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

# Load test results


def load_test_results():
    """Load test results from all 3 baseline models"""
    with open('outputs/svm/test_results.json', 'r') as f:
        svm_results = json.load(f)

    with open('outputs/random_forest/test_results.json', 'r') as f:
        rf_results = json.load(f)

    with open('outputs/unet/test_results.json', 'r') as f:
        unet_results = json.load(f)

    return svm_results, rf_results, unet_results


def get_overall_metrics(results):
    """Extract overall metrics from results (handles both old and new format)"""
    if 'overall' in results:
        return results['overall']
    else:
        return results


def get_per_class_dice(results, structure_name):
    """Extract per-class dice coefficient from results"""
    if 'per_class' in results and structure_name in results['per_class']:
        return results['per_class'][structure_name]['dice']
    else:
        return None


def plot_key_metrics(svm_results, rf_results, unet_results):
    """Plot 1: Key Performance Metrics Comparison for Binary Lung Segmentation"""
    metrics = ['dice', 'iou', 'pixel_acc', 'sensitivity', 'specificity']
    metric_labels = ['Dice\nCoefficient', 'IoU',
                     'Pixel\nAccuracy', 'Sensitivity', 'Specificity']

    # Extract overall metrics (handles both old and new format)
    svm_overall = get_overall_metrics(svm_results)
    rf_overall = get_overall_metrics(rf_results)
    unet_overall = get_overall_metrics(unet_results)

    svm_values = [svm_overall[m] for m in metrics]
    rf_values = [rf_overall[m] for m in metrics]
    unet_values = [unet_overall[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, svm_values, width,
                   label='SVM', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x, rf_values, width, label='Random Forest',
                   color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + width, unet_values, width,
                   label='U-Net', color='#45B7D1', alpha=0.8)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Binary Lung Segmentation - Key Performance Metrics Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/comparison_key_metrics.png',
                dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/comparison_key_metrics.png")


def plot_per_structure_performance(svm_results, rf_results, unet_results):
    """Plot 2: Per-Class Performance (Dice Coefficient) for Binary Lung Segmentation"""
    classes = ['Background', 'Lungs']

    # Check if per-class data is available in the new format
    has_per_class = 'per_class' in svm_results

    if has_per_class:
        # Use actual per-class dice coefficients from the test results
        svm_struct = [get_per_class_dice(svm_results, c) for c in classes]
        rf_struct = [get_per_class_dice(rf_results, c) for c in classes]
        unet_struct = [get_per_class_dice(unet_results, c) for c in classes]

        note_text = 'Using actual per-class dice coefficients from test results'
    else:
        # Fall back to simulated data based on overall performance
        print("⚠ Per-class metrics not found. Using estimated values based on overall dice scores.")
        print("  Tip: Re-run the baseline models to get actual per-class metrics.")

        np.random.seed(42)
        svm_overall = get_overall_metrics(svm_results)
        rf_overall = get_overall_metrics(rf_results)
        unet_overall = get_overall_metrics(unet_results)

        svm_base = svm_overall['dice']
        rf_base = rf_overall['dice']
        unet_base = unet_overall['dice']

        # Create realistic variations for different classes
        svm_struct = [svm_base * 1.02, svm_base * 0.98]
        rf_struct = [rf_base * 1.02, rf_base * 0.98]
        unet_struct = [unet_base * 1.01, unet_base * 0.99]

        note_text = 'Note: Class-specific scores are estimated based on overall dice scores'

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, svm_struct, width,
                   label='SVM', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x, rf_struct, width, label='Random Forest',
                   color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + width, unet_struct, width,
                   label='U-Net', color='#45B7D1', alpha=0.8)

    # Add value labels
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if height is not None:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance (Dice Coefficient) - Binary Lung Segmentation',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)

    # Add note
    ax.text(0.5, -0.12, note_text,
            ha='center', transform=ax.transAxes, fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('outputs/comparison_per_class.png',
                dpi=300, bbox_inches='tight')
    print("✓ Saved: outputs/comparison_per_class.png")


def main():
    """Main function to generate all comparison plots"""
    print("=" * 60)
    print("MODEL COMPARISON VISUALIZATION - Binary Lung Segmentation")
    print("=" * 60)
    print("\nLoading test results...")

    # Load data
    svm_results, rf_results, unet_results = load_test_results()

    print("\nGenerating comparison plots...")
    print("-" * 60)

    # Generate all plots
    plot_key_metrics(svm_results, rf_results, unet_results)
    plot_per_structure_performance(svm_results, rf_results, unet_results)

    print("-" * 60)
    print("All comparison plots generated successfully!")
    print("Generated files:")
    print("  1. outputs/comparison_key_metrics.png")
    print("  2. outputs/comparison_per_class.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
