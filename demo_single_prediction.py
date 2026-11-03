"""
Demo script for single image prediction with complete reliability assessment
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from src.config import *
from src.models import TumourClassifier
from src.data import Preprocessor
from src.explainability import ExplanationGenerator
from src.evaluation import LocalizationEvaluator, StabilityEvaluator, Perturbation
from src.reliability import ReliabilityAnalyzer


def visualize_results(
    image: np.ndarray,
    mask: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    prediction: str,
    confidence: float,
    reliability_score: float,
    flag: str,
    breakdown: dict,
    save_path: str = None
):
    """Visualize complete analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original MRI Scan')
    axes[0, 0].axis('off')
    
    # Ground truth mask
    axes[0, 1].imshow(mask, cmap='gray')
    axes[0, 1].set_title('Expert Tumour Mask')
    axes[0, 1].axis('off')
    
    # Grad-CAM heatmap
    axes[0, 2].imshow(cam, cmap='jet')
    axes[0, 2].set_title('Grad-CAM Heatmap')
    axes[0, 2].axis('off')
    
    # Overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Heatmap Overlay')
    axes[1, 0].axis('off')
    
    # Prediction info
    axes[1, 1].axis('off')
    info_text = f"""
PREDICTION RESULTS

Class: {prediction.upper()}
Confidence: {confidence:.1%}

RELIABILITY ASSESSMENT
Score: {reliability_score:.3f}
Flag: {flag}

EVIDENCE BREAKDOWN
• Confidence: {breakdown['confidence']:.3f}
• Localization: {breakdown['localization_score']:.3f}
• Stability: {breakdown['stability_score']:.3f}

Threshold: {breakdown['threshold']:.3f}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                    verticalalignment='center')
    
    # Reliability bar chart
    components = ['Confidence', 'Localization', 'Stability']
    scores = [
        breakdown['confidence'],
        breakdown['localization_score'],
        breakdown['stability_score']
    ]
    colors = ['#3b82f6', '#10b981', '#8b5cf6']
    
    axes[1, 2].barh(components, scores, color=colors, alpha=0.7)
    axes[1, 2].axvline(breakdown['threshold'], color='red', linestyle='--', 
                       label=f'Threshold ({breakdown["threshold"]:.2f})')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_xlabel('Score')
    axes[1, 2].set_title('Evidence Components')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.suptitle(
        f'Evidence-Aware Brain Tumour Classification - {flag}',
        fontsize=14,
        fontweight='bold',
        color='green' if flag == 'RELIABLE' else 'red'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    
    plt.show()


def predict_single_image(image_path: str, mask_path: str, model_path: str):
    """
    Complete prediction pipeline for a single image
    """
    print("=" * 60)
    print("SINGLE IMAGE PREDICTION WITH RELIABILITY ASSESSMENT")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = TumourClassifier(**MODEL_CONFIG)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load image and mask
    print(f"\nLoading image: {image_path}")
    image_pil = Image.open(image_path).convert('RGB')
    mask_pil = Image.open(mask_path).convert('L')
    
    # Preprocess
    preprocessor = Preprocessor(**PREPROCESS_CONFIG)
    image_tensor, mask_tensor = preprocessor.synchronized_augment(
        image_pil, mask_pil, apply_aug=False
    )
    image_batch = image_tensor.unsqueeze(0).to(device)
    
    # Original image for visualization
    image_np = np.array(image_pil.resize((224, 224)))
    mask_np = mask_tensor.cpu().numpy()[0]
    
    # Predict
    print("\n" + "=" * 60)
    print("STEP 1: CLASSIFICATION")
    print("=" * 60)
    class_name, confidence, probs = model.predict(image_batch)
    print(f"Predicted Class: {class_name.upper()}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Class Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {MODEL_CONFIG['class_names'][i]}: {prob:.1%}")
    
    # Generate explanation
    print("\n" + "=" * 60)
    print("STEP 2: EXPLANATION GENERATION")
    print("=" * 60)
    explainer = ExplanationGenerator(model, device=device)
    predicted_class = probs.argmax().item()
    cam = explainer.generate_cam(image_batch, predicted_class)
    overlay = explainer.overlay_on_image(image_np, cam, alpha=0.5)
    print("✓ Grad-CAM heatmap generated")
    
    # Localization evaluation
    print("\n" + "=" * 60)
    print("STEP 3: LOCALIZATION EVALUATION")
    print("=" * 60)
    loc_evaluator = LocalizationEvaluator(**LOCALIZATION_CONFIG)
    loc_metrics = loc_evaluator.evaluate_all(cam, mask_np)
    print(f"Localization Metrics:")
    print(f"  IoU: {loc_metrics['iou']:.3f}")
    print(f"  Dice: {loc_metrics['dice']:.3f}")
    print(f"  Pointing Game: {'HIT' if loc_metrics['pointing_game'] == 1.0 else 'MISS'}")
    print(f"  Energy Ratio: {loc_metrics['energy_ratio']:.3f}")
    
    # Stability evaluation
    print("\n" + "=" * 60)
    print("STEP 4: STABILITY EVALUATION")
    print("=" * 60)
    perturbation_engine = Perturbation(**PERTURBATION_CONFIG)
    stability_evaluator = StabilityEvaluator(**STABILITY_CONFIG)
    
    print(f"Generating {PERTURBATION_CONFIG['num_perturbations']} perturbations...")
    perturbed_images = perturbation_engine.generate_perturbation_suite(
        image_tensor,
        num_perturbations=PERTURBATION_CONFIG['num_perturbations']
    )
    
    print("Computing perturbed explanations...")
    perturbed_cams = []
    for p_img in perturbed_images:
        p_img_batch = p_img.unsqueeze(0).to(device)
        p_cam = explainer.generate_cam(p_img_batch, predicted_class)
        perturbed_cams.append(p_cam)
    
    stability_metrics = stability_evaluator.evaluate_stability(cam, perturbed_cams)
    print(f"Stability Metrics:")
    print(f"  Mean SSIM: {stability_metrics['mean_ssim']:.3f}")
    print(f"  Mean Spearman: {stability_metrics['mean_spearman']:.3f}")
    
    # Reliability assessment
    print("\n" + "=" * 60)
    print("STEP 5: RELIABILITY ASSESSMENT")
    print("=" * 60)
    reliability_analyzer = ReliabilityAnalyzer(**RELIABILITY_CONFIG)
    
    reliability_score, flag, breakdown = reliability_analyzer.assess_reliability(
        confidence, loc_metrics, stability_metrics
    )
    
    print(f"\n*** RELIABILITY SCORE: {reliability_score:.3f} ***")
    print(f"*** FLAG: {flag} ***")
    print(f"\nEvidence Breakdown:")
    print(f"  Confidence:    {breakdown['confidence']:.3f} (weight: {breakdown['weights']['confidence']:.2f})")
    print(f"  Localization:  {breakdown['localization_score']:.3f} (weight: {breakdown['weights']['localization']:.2f})")
    print(f"  Stability:     {breakdown['stability_score']:.3f} (weight: {breakdown['weights']['stability']:.2f})")
    print(f"  Threshold:     {breakdown['threshold']:.3f}")
    
    # Visualize
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION")
    print("=" * 60)
    
    save_path = RESULTS_DIR / f"prediction_{Path(image_path).stem}.png"
    visualize_results(
        image_np, mask_np, cam, overlay,
        class_name, confidence,
        reliability_score, flag, breakdown,
        save_path=str(save_path)
    )
    
    print("\n✓ Analysis complete!")
    
    return {
        'prediction': class_name,
        'confidence': confidence,
        'reliability_score': reliability_score,
        'flag': flag,
        'breakdown': breakdown
    }


def main():
    parser = argparse.ArgumentParser(description="Single image prediction demo")
    parser.add_argument('--image', type=str, required=True, help='Path to MRI image')
    parser.add_argument('--mask', type=str, required=True, help='Path to tumour mask')
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_classifier.pth',
        help='Path to trained model'
    )
    
    args = parser.parse_args()
    
    # Check files exist
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    if not Path(args.mask).exists():
        print(f"Error: Mask not found: {args.mask}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        print("Please train the model first using: python main.py --mode train")
        return
    
    # Run prediction
    result = predict_single_image(args.image, args.mask, args.model)
    
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Reliability: {result['flag']} (score: {result['reliability_score']:.3f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
