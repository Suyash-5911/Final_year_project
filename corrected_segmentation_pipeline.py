"""
CORRECTED Brain Tumor Segmentation Pipeline
============================================

This script provides an HONEST implementation that distinguishes:
- Brain extraction (what we can do) vs Tumor segmentation (what we need)
- Classification (what we have) vs Localization (what we need ground truth for)

Dataset Reality:
  ✓ Has: Classification labels (glioma, meningioma, pituitary)
  ✗ Lacks: Pixel-level tumor segmentation annotations

"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings


class BrainTumorPipeline:
    """
    Corrected pipeline that clearly separates:
    1. Brain extraction (automated)
    2. Tumor classification (possible with this dataset)
    3. Tumor segmentation (NOT possible without ground truth)
    """
    
    def __init__(self, dataset_dir: str = "data/brain_tumours"):
        self.dataset_dir = Path(dataset_dir)
        
    # ============================================================
    # BRAIN EXTRACTION (Skull Stripping)
    # ============================================================
    
    def extract_brain_region(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract brain region from MRI scan (remove background/skull)
        
        ⚠️ WARNING: This is BRAIN EXTRACTION, NOT tumor segmentation!
        
        Returns:
            brain_mask: Binary mask showing brain vs background
            brain_only: Image with background removed
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Otsu's thresholding - separates brain from background
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, brain_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply mask to image
        if len(image.shape) == 3:
            brain_only = cv2.bitwise_and(image, image, mask=brain_mask)
        else:
            brain_only = cv2.bitwise_and(gray, gray, mask=brain_mask)
        
        return brain_mask, brain_only
    
    # ============================================================
    # PSEUDO-SEGMENTATION (Heuristic - NOT ground truth)
    # ============================================================
    
    def generate_pseudo_mask(self, image: np.ndarray, method: str = 'intensity') -> np.ndarray:
        """
        Generate pseudo tumor mask using heuristics
        
        ⚠️ CRITICAL WARNING:
        - This is NOT a real tumor segmentation
        - This is NOT validated against expert annotations
        - DO NOT use for IoU/Dice evaluation
        - For VISUALIZATION PURPOSES ONLY
        
        Args:
            image: Input MRI scan
            method: 'intensity', 'edge', or 'region'
            
        Returns:
            pseudo_mask: Heuristic-based mask (NOT ground truth)
        """
        warnings.warn(
            "Pseudo-masks are heuristic approximations, NOT expert-validated tumor boundaries. "
            "Do not use for quantitative evaluation (IoU, Dice, etc.)",
            UserWarning
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # First extract brain region
        brain_mask, _ = self.extract_brain_region(image)
        
        if method == 'intensity':
            # Find brightest regions WITHIN brain
            # (Tumors often appear brighter, but not always)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Get top 10-20% brightest pixels within brain
            masked = cv2.bitwise_and(enhanced, enhanced, mask=brain_mask)
            threshold_value = np.percentile(masked[masked > 0], 80)
            _, pseudo_mask = cv2.threshold(masked, threshold_value, 255, cv2.THRESH_BINARY)
            
        elif method == 'edge':
            # Detect regions with high edge density (tumors have irregular boundaries)
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Keep only edges within brain
            pseudo_mask = cv2.bitwise_and(edges, edges, mask=brain_mask)
            
        elif method == 'region':
            # Region-based (similar to current implementation)
            pseudo_mask = brain_mask.copy()  # Placeholder
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        pseudo_mask = cv2.morphologyEx(pseudo_mask, cv2.MORPH_CLOSE, kernel)
        pseudo_mask = cv2.morphologyEx(pseudo_mask, cv2.MORPH_OPEN, kernel)
        
        return pseudo_mask
    
    # ============================================================
    # VISUALIZATION (Honest Labels)
    # ============================================================
    
    def visualize_pipeline(self, image_path: str, tumor_type: str, save_path: Optional[str] = None):
        """
        Create honest visualization showing what we can and cannot determine
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        brain_mask, brain_only = self.extract_brain_region(image)
        pseudo_mask = self.generate_pseudo_mask(image, method='intensity')
        
        # Calculate statistics
        brain_pct = 100 * (brain_mask > 127).sum() / brain_mask.size
        pseudo_pct = 100 * (pseudo_mask > 127).sum() / pseudo_mask.size
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Original, Brain Extraction, Brain Only
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('1. Original MRI Scan\\n(Classification: ' + tumor_type + ')', 
                             fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(brain_mask, cmap='gray')
        axes[0, 1].set_title(f'2. Brain Extraction Mask\\n({brain_pct:.1f}% of image)\\n' +
                             '⚠️ This is NOT tumor segmentation',
                             fontweight='bold', color='red')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(brain_only, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title('3. Brain Region Only\\n(Background removed)',
                             fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Pseudo-mask, Overlay, Legend
        axes[1, 0].imshow(pseudo_mask, cmap='hot')
        axes[1, 0].set_title(f'4. Pseudo-Tumor Mask\\n({pseudo_pct:.1f}% of image)\\n' +
                             '⚠️ Heuristic only, NOT validated',
                             fontweight='bold', color='orange')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = image_rgb.copy()
        red_overlay = np.zeros_like(image_rgb)
        red_overlay[pseudo_mask > 127] = [255, 100, 100]
        overlay = cv2.addWeighted(overlay, 0.7, red_overlay, 0.3, 0)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('5. Pseudo-Mask Overlay\\n(For visualization only)',
                             fontweight='bold')
        axes[1, 1].axis('off')
        
        # Legend/Explanation
        axes[1, 2].axis('off')
        legend_text = f"""
DATASET CAPABILITIES:

✓ AVAILABLE:
  • Classification label: {tumor_type}
  • Brain extraction (skull stripping)
  • Image preprocessing
  
✗ NOT AVAILABLE:
  • Pixel-level tumor annotations
  • Expert-validated tumor boundaries
  • Ground truth segmentation masks

WHAT MASKS SHOW:
  • Brain extraction: Entire brain region
  • Pseudo-mask: Heuristic approximation
  
⚠️ DO NOT USE FOR:
  • IoU/Dice evaluation
  • Segmentation model training
  • Ground truth validation

✓ CAN USE FOR:
  • Classification accuracy
  • Grad-CAM visualization
  • Qualitative analysis
        """
        axes[1, 2].text(0.1, 0.5, legend_text, fontsize=9, family='monospace',
                       verticalalignment='center')
        
        plt.suptitle('CORRECTED PIPELINE: Brain Tumor Analysis\\n' +
                     'Honest distinction between brain extraction and tumor segmentation',
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        
        plt.show()
        
    # ============================================================
    # DATASET EVALUATION (Honest Assessment)
    # ============================================================
    
    def evaluate_dataset_capabilities(self):
        """
        Provide honest assessment of what can and cannot be done with this dataset
        """
        print("=" * 80)
        print("DATASET CAPABILITY ASSESSMENT")
        print("=" * 80)
        
        capabilities = {
            "Classification Tasks": {
                "Tumor type classification": "✓ FULLY SUPPORTED",
                "Binary classification (tumor/no tumor)": "✓ FULLY SUPPORTED",
                "Multi-class accuracy metrics": "✓ FULLY SUPPORTED",
                "Transfer learning experiments": "✓ FULLY SUPPORTED",
            },
            "Explainability Tasks": {
                "Grad-CAM visualization": "✓ SUPPORTED (qualitative)",
                "Saliency maps": "✓ SUPPORTED (qualitative)",
                "Attention visualization": "✓ SUPPORTED (qualitative)",
            },
            "Localization Tasks": {
                "IoU calculation": "✗ NOT SUPPORTED (no ground truth)",
                "Dice score calculation": "✗ NOT SUPPORTED (no ground truth)",
                "Pointing game accuracy": "✗ NOT SUPPORTED (no ground truth)",
                "Energy ratio": "✗ NOT SUPPORTED (no ground truth)",
            },
            "Segmentation Tasks": {
                "U-Net training": "✗ NOT SUPPORTED (no pixel annotations)",
                "Segmentation model evaluation": "✗ NOT SUPPORTED (no pixel annotations)",
                "Boundary detection": "✗ NOT SUPPORTED (no pixel annotations)",
            }
        }
        
        for category, tasks in capabilities.items():
            print(f"\n{category}:")
            for task, status in tasks.items():
                print(f"  {status:30s} - {task}")
        
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print("""
For your project, you should:

1. ✓ Focus on CLASSIFICATION performance
   - Train CNN models (ResNet, VGG, EfficientNet)
   - Achieve high accuracy on test set
   - Compare different architectures

2. ✓ Use Grad-CAM for QUALITATIVE visualization
   - Show what regions model focuses on
   - Discuss whether activations align with expected tumor locations
   - Qualitative analysis in results section

3. ✗ Do NOT claim quantitative localization metrics
   - Do not calculate IoU/Dice without ground truth
   - Do not present pseudo-masks as validated
   - Be honest about dataset limitations

4. [OPTIONAL] Enhance with LGG Dataset
   - Download for glioma class only (~500 MB)
   - Get real segmentation masks
   - Perform localization evaluation on glioma subset
        """)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def main():
    """Demonstrate corrected pipeline"""
    
    pipeline = BrainTumorPipeline()
    
    # Show dataset capabilities
    pipeline.evaluate_dataset_capabilities()
    
    # Generate corrected visualizations
    print("\n" + "=" * 80)
    print("GENERATING CORRECTED VISUALIZATIONS")
    print("=" * 80)
    
    tumor_types = ['glioma', 'meningioma', 'pituitary']
    
    for tumor_type in tumor_types:
        img_dir = Path(f'data/brain_tumours/{tumor_type}/images')
        images = sorted(list(img_dir.glob('*.jpg')))
        
        if images:
            print(f"\nProcessing {tumor_type}...")
            save_path = f'corrected_visualization_{tumor_type}.png'
            pipeline.visualize_pipeline(str(images[0]), tumor_type, save_path)
    
    print("\n" + "=" * 80)
    print("✅ CORRECTED PIPELINE COMPLETE")
    print("=" * 80)
    print("""
Key Takeaways:
1. Brain extraction masks are NOT tumor segmentation masks
2. This dataset supports classification, not localization evaluation
3. Pseudo-masks can be used for visualization but not evaluation
4. Be honest about dataset limitations in your report

Check the generated 'corrected_visualization_*.png' files to see
the difference between brain extraction and tumor segmentation.
    """)


if __name__ == "__main__":
    main()
