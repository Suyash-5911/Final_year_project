import os
from render_utils import render_svg_to_png_and_jpg
from build_part1 import SVG_DEFS

# -------------------------------------------------------------
# 9. Figure 4.7: UML Activity Diagram
# -------------------------------------------------------------
def build_fig_4_7():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1300 1050" width="1300" height="1050">
  <rect width="1300" height="1050" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1200" height="55" rx="8" fill="#1e293b" />
  <text x="650" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.7: UML Activity Diagram (Evidence-Aware Workflow)</text>
  
  <!-- Main Workflow Column (Centered) -->
  <!-- Initial Node -->
  <circle cx="650" cy="115" r="16" fill="#0f172a" />
  <text x="690" y="120" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">Initial State</text>
  
  <path d="M 650 131 L 650 165" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 1: Load MRI & Mask -->
  <g transform="translate(480, 165)">
    <rect width="340" height="48" rx="24" fill="#eff6ff" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)" />
    <text x="170" y="29" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Load MRI Slice &amp; Expert Tumour Mask</text>
  </g>
  
  <path d="M 650 213 L 650 245" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 2: Pre-process Image & Mask -->
  <g transform="translate(480, 245)">
    <rect width="340" height="48" rx="24" fill="#eff6ff" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)" />
    <text x="170" y="29" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Pre-process MRI &amp; Mask (Resize, Normalize)</text>
  </g>
  
  <path d="M 650 293 L 650 325" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 3: Classify Tumour -->
  <g transform="translate(470, 325)">
    <rect width="360" height="52" rx="26" fill="#eff6ff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <text x="180" y="24" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Perform CNN Inference</text>
    <text x="180" y="42" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Obtain Predicted Tumour Class &amp; Softmax Confidence (c)</text>
  </g>
  
  <path d="M 650 377 L 650 410" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 4: Generate Grad-CAM -->
  <g transform="translate(480, 410)">
    <rect width="340" height="48" rx="24" fill="#f0f9ff" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)" />
    <text x="170" y="29" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Generate Grad-CAM Visual Explanation (Heatmap h)</text>
  </g>
  
  <path d="M 650 458 L 650 490" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 5: Localization Evaluation -->
  <g transform="translate(450, 490)">
    <rect width="400" height="52" rx="26" fill="#ecfdf5" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <text x="200" y="24" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Compute Localization Agreement Against Mask</text>
    <text x="200" y="42" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Binarize Heatmap; Measure IoU, Dice, Pointing Game, Energy</text>
  </g>
  
  <path d="M 650 542 L 650 575" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 6: Input Perturbation & Re-explanation -->
  <g transform="translate(450, 575)">
    <rect width="400" height="52" rx="26" fill="#fdf4ff" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <text x="200" y="24" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Generate Perturbed Inputs &amp; Re-explain</text>
    <text x="200" y="42" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Apply Gaussian Noise, Blur, Shift; Regenerate K Heatmaps</text>
  </g>
  
  <path d="M 650 627 L 650 660" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 7: Stability Evaluation -->
  <g transform="translate(460, 660)">
    <rect width="380" height="50" rx="25" fill="#fdf4ff" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <text x="190" y="24" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Compute Explanation Stability Scores</text>
    <text x="190" y="41" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Calculate Mean SSIM and Spearman Rank Correlation (ρ)</text>
  </g>
  
  <path d="M 650 710 L 650 740" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Activity 8: Fuse Scores -->
  <g transform="translate(460, 740)">
    <rect width="380" height="52" rx="26" fill="#fdf2f8" stroke="#db2777" stroke-width="2.5" filter="url(#shadow)" />
    <text x="190" y="24" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Fuse Confidence, Localization &amp; Stability</text>
    <text x="190" y="42" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle">Synthesize Evidence Vector into Composite Reliability Score (r)</text>
  </g>
  
  <path d="M 650 792 L 650 820" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Decision Diamond -->
  <g transform="translate(605, 820)">
    <polygon points="45,0 90,30 45,60 0,30" fill="#fffbeb" stroke="#f59e0b" stroke-width="2.2" filter="url(#shadow)" />
    <text x="45" y="34" fill="#b45309" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Score &gt;= θ ?</text>
  </g>
  
  <!-- Decision Branches -->
  <!-- Yes Branch (Right) -->
  <path d="M 695 850 L 890 850" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />
  <rect x="715" y="828" width="135" height="18" rx="3" fill="#ecfdf5" stroke="#a7f3d0" />
  <text x="782" y="841" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">[Yes: Score ≥ Threshold]</text>
  
  <g transform="translate(890, 825)">
    <rect width="260" height="50" rx="25" fill="#f0fdf4" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <text x="130" y="23" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Accept Prediction</text>
    <text x="130" y="40" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Flag as "Reliable Assistance"</text>
  </g>
  
  <!-- No Branch (Left) -->
  <path d="M 605 850 L 410 850" stroke="#dc2626" stroke-width="2" marker-end="url(#arrow-red)" />
  <rect x="445" y="828" width="135" height="18" rx="3" fill="#fef2f2" stroke="#fecaca" />
  <text x="512" y="841" fill="#b91c1c" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">[No: Score &lt; Threshold]</text>
  
  <g transform="translate(150, 825)">
    <rect width="260" height="50" rx="25" fill="#fef2f2" stroke="#ef4444" stroke-width="2" filter="url(#shadow)" />
    <text x="130" y="23" fill="#991b1b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Flag as "Unreliable"</text>
    <text x="130" y="40" fill="#b91c1c" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Alert: Route to Radiologist Review</text>
  </g>
  
  <!-- Merge Node -->
  <g transform="translate(625, 915)">
    <polygon points="25,0 50,15 25,30 0,15" fill="#64748b" stroke="#334155" stroke-width="1.5" />
  </g>
  
  <path d="M 1020 875 L 1020 930 L 675 930" fill="none" stroke="#059669" stroke-width="2" />
  <path d="M 280 875 L 280 930 L 625 930" fill="none" stroke="#dc2626" stroke-width="2" />
  
  <path d="M 650 945 L 650 970" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Final Activity: Display & Log -->
  <g transform="translate(450, 970)">
    <rect width="400" height="42" rx="21" fill="#f8fafc" stroke="#334155" stroke-width="1.8" />
    <text x="200" y="26" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Log Analysis, Persist Records &amp; Display Dashboard Views</text>
  </g>
  
  <path d="M 850 991 L 910 991" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- Final State Node (Bullseye) -->
  <circle cx="935" cy="991" r="16" fill="#ffffff" stroke="#0f172a" stroke-width="2" />
  <circle cx="935" cy="991" r="10" fill="#0f172a" />
  <text x="965" y="995" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">Final State</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_7_activity_diagram")

# -------------------------------------------------------------
# 10. Figure 4.8: UML Class Diagram
# -------------------------------------------------------------
def build_fig_4_8():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 1150" width="1450" height="1150">
  <rect width="1450" height="1150" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1350" height="55" rx="8" fill="#1e293b" />
  <text x="725" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.8: UML Class Diagram (Detailed Object-Oriented Design)</text>
  
  <!-- ROW 1: DatasetLoader -> Preprocessor -> TumourClassifier -->
  <!-- Class 1: DatasetLoader -->
  <g transform="translate(60, 110)">
    <rect width="360" height="230" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="360" height="35" rx="8" fill="#2563eb" />
    <rect y="25" width="360" height="10" fill="#2563eb" />
    <text x="180" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">DatasetLoader</text>
    
    <!-- Attributes -->
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- dataset_dir: str</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- patient_splits: dict[str, list]</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- batch_size: int = 32</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- image_mask_pairs: list[tuple]</text>
    
    <line x1="0" y1="130" x2="360" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <!-- Methods -->
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ load_slices_and_masks(): list</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ get_patient_stratified_split(): tuple</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ verify_mask_alignment(): bool</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ get_data_loader(split: str): DataLoader</text>
  </g>
  
  <!-- Class 2: Preprocessor -->
  <g transform="translate(540, 110)">
    <rect width="360" height="230" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="360" height="35" rx="8" fill="#2563eb" />
    <rect y="25" width="360" height="10" fill="#2563eb" />
    <text x="180" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Preprocessor</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- target_size: tuple[int, int] = (224, 224)</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- norm_mean: list[float] = [0.485, ...]</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- norm_std: list[float] = [0.229, ...]</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- intensity_range: tuple[float, float]</text>
    
    <line x1="0" y1="130" x2="360" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ resize(img: np.ndarray): np.ndarray</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ normalize_intensity(img): Tensor</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ synchronized_augment(img, mask): tuple</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ to_tensor(img: np.ndarray): Tensor</text>
  </g>
  
  <!-- Class 3: TumourClassifier -->
  <g transform="translate(1020, 110)">
    <rect width="370" height="230" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="370" height="35" rx="8" fill="#2563eb" />
    <rect y="25" width="370" height="10" fill="#2563eb" />
    <text x="185" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">TumourClassifier</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- backbone: str = "resnet50"</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- num_classes: int = 3</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- class_names: list[str] = [gli, men, pit]</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- weights_path: str</text>
    
    <line x1="0" y1="130" x2="370" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ forward(x: Tensor): Tensor</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ predict(x: Tensor): tuple[str, float]</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_loss(logits, targets): Tensor</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ get_target_layer_activations(): Tensor</text>
  </g>
  
  <!-- Connector R1: DatasetLoader -> Preprocessor -->
  <path d="M 420 220 L 540 220" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <text x="480" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">feeds</text>
  
  <!-- Connector R2: Preprocessor -> TumourClassifier -->
  <path d="M 900 220 L 1020 220" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <text x="960" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">provides x'</text>
  
  <!-- ROW 2: ExplanationGenerator & Perturbation -->
  <!-- Class 4: ExplanationGenerator -->
  <g transform="translate(1020, 420)">
    <rect width="370" height="230" rx="8" fill="#ffffff" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)" />
    <rect width="370" height="35" rx="8" fill="#0284c7" />
    <rect y="25" width="370" height="10" fill="#0284c7" />
    <text x="185" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">ExplanationGenerator</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- target_layer: str = "layer4[-1]"</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- method: str = "Grad-CAM"</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- colormap: str = "jet"</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- gradients: Tensor</text>
    
    <line x1="0" y1="130" x2="370" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ generate_cam(img: Tensor, cls: int): Array</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ extract_gradients(): void</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ overlay_on_image(img, cam): np.ndarray</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ normalize_heatmap(cam: Array): Array</text>
  </g>
  
  <!-- Class 5: Perturbation -->
  <g transform="translate(60, 420)">
    <rect width="360" height="230" rx="8" fill="#ffffff" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <rect width="360" height="35" rx="8" fill="#a21caf" />
    <rect y="25" width="360" height="10" fill="#a21caf" />
    <text x="180" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Perturbation</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- noise_stds: list[float] = [0.01, 0.05]</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- blur_radii: list[float] = [1.0, 2.0]</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- contrast_ranges: tuple[float, float]</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- max_shift_px: int = 3</text>
    
    <line x1="0" y1="130" x2="360" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ add_gaussian_noise(img, std): Tensor</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ apply_gaussian_blur(img, radius): Tensor</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ alter_brightness_contrast(img): Tensor</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ generate_perturbation_suite(img): list</text>
  </g>
  
  <!-- Connector: ExplanationGenerator uses TumourClassifier -->
  <path d="M 1205 340 L 1205 420" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="1235" y="380" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">uses</text>
  
  <!-- ROW 3: LocalizationEvaluator & StabilityEvaluator -->
  <!-- Class 6: LocalizationEvaluator -->
  <g transform="translate(1020, 730)">
    <rect width="370" height="230" rx="8" fill="#ffffff" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect width="370" height="35" rx="8" fill="#059669" />
    <rect y="25" width="370" height="10" fill="#059669" />
    <text x="185" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">LocalizationEvaluator</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- binarize_threshold: float = 0.5</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- binarize_strategy: str = "otsu"</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- current_mask: np.ndarray</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- score_cache: dict[str, float]</text>
    
    <line x1="0" y1="130" x2="370" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ binarize_heatmap(cam: Array): Array</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_iou(cam_bin, mask): float</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_dice(cam_bin, mask): float</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ pointing_game_and_energy(cam, mask): dict</text>
  </g>
  
  <!-- Class 7: StabilityEvaluator -->
  <g transform="translate(60, 730)">
    <rect width="360" height="230" rx="8" fill="#ffffff" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <rect width="360" height="35" rx="8" fill="#a21caf" />
    <rect y="25" width="360" height="10" fill="#a21caf" />
    <text x="180" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">StabilityEvaluator</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- num_perturbations: int = 5</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- ssim_window_size: int = 7</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- orig_heatmap: np.ndarray</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- perturbed_heatmaps: list[np.ndarray]</text>
    
    <line x1="0" y1="130" x2="360" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ evaluate_stability(h, [hk]): tuple</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_pairwise_ssim(h1, h2): float</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_spearman_rank(h1, h2): float</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ aggregate_stability_metrics(): dict</text>
  </g>
  
  <!-- Connector: StabilityEvaluator uses Perturbation -->
  <path d="M 240 650 L 240 730" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="265" y="690" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">uses</text>
  
  <!-- Connector: ExplanationGenerator -> LocalizationEvaluator -->
  <path d="M 1205 650 L 1205 730" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <text x="1235" y="690" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">supplies h</text>
  
  <!-- Connector: ExplanationGenerator -> StabilityEvaluator -->
  <path d="M 1020 540 C 700 540, 500 680, 420 780" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <text x="690" y="590" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">supplies original &amp; perturbed cams</text>
  
  <!-- ROW 4 / CENTER: ReliabilityAnalyzer -->
  <!-- Class 8: ReliabilityAnalyzer -->
  <g transform="translate(530, 560)">
    <rect width="380" height="260" rx="8" fill="#ffffff" stroke="#db2777" stroke-width="2.5" filter="url(#shadow)" />
    <rect width="380" height="35" rx="8" fill="#be185d" />
    <rect y="25" width="380" height="10" fill="#be185d" />
    <text x="190" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">ReliabilityAnalyzer</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- reliability_threshold: float = 0.70</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- weights: dict[str, float]</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- baseline_type: str = "confidence_only"</text>
    <text x="15" y="118" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- audit_history: list[dict]</text>
    
    <line x1="0" y1="130" x2="380" y2="130" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="152" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ fuse_evidence(conf, loc, stab): float</text>
    <text x="15" y="174" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ assign_reliability_flag(score): str</text>
    <text x="15" y="196" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compute_auroc_auprc(y_true, scores): dict</text>
    <text x="15" y="218" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ risk_coverage_analysis(): dict</text>
    <text x="15" y="240" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ compare_with_baseline(): dict</text>
  </g>
  
  <!-- Evaluators -> ReliabilityAnalyzer -->
  <path d="M 420 840 L 530 760" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />
  <text x="440" y="790" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600">stability scores</text>
  
  <path d="M 1020 840 L 910 760" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />
  <text x="965" y="820" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle">localization scores</text>
  
  <!-- ROW 5: ResultsDashboard -->
  <!-- Class 9: ResultsDashboard -->
  <g transform="translate(510, 890)">
    <rect width="420" height="210" rx="8" fill="#ffffff" stroke="#f59e0b" stroke-width="2" filter="url(#shadow)" />
    <rect width="420" height="35" rx="8" fill="#d97706" />
    <rect y="25" width="420" height="10" fill="#d97706" />
    <text x="210" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">ResultsDashboard</text>
    
    <text x="15" y="58" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- app_mode: str = "clinical_audit"</text>
    <text x="15" y="78" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- uploaded_scan: np.ndarray</text>
    <text x="15" y="98" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">- active_result: dict</text>
    
    <line x1="0" y1="110" x2="420" y2="110" stroke="#cbd5e1" stroke-width="1.2" />
    
    <text x="15" y="132" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ render_upload_interface(): void</text>
    <text x="15" y="154" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ display_prediction_panel(pred, conf): void</text>
    <text x="15" y="176" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ display_heatmaps_and_mask_comparison(): void</text>
    <text x="15" y="198" fill="#0f172a" font-family="'Courier New', monospace" font-size="11.5">+ render_reliability_badge_and_export(): void</text>
  </g>
  
  <!-- Connector: ReliabilityAnalyzer -> ResultsDashboard -->
  <path d="M 720 820 L 720 890" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="790" y="860" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">sends findings</text>
  
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_8_class_diagram")

# -------------------------------------------------------------
# 11. Figure 4.9: UML Sequence Diagram
# -------------------------------------------------------------
def build_fig_4_9():
    lifelines = [
        ("User", "Actor", 100),
        ("ui", "Dashboard UI", 280),
        ("ctrl", "Pipeline Controller", 490),
        ("clf", "Tumour Classifier", 700),
        ("exp", "Explanation Generator", 910),
        ("eval", "Evaluation Modules", 1120),
        ("rel", "Reliability Analyzer", 1310)
    ]
    
    lines_svg = ""
    boxes_svg = ""
    for name, title, x in lifelines:
        lines_svg += f"""
        <line x1="{x}" y1="160" x2="{x}" y2="920" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="6,4" />
        """
        boxes_svg += f"""
        <g transform="translate({x - 75}, 110)">
          <rect width="150" height="48" rx="6" fill="#1e293b" />
          <text x="75" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">{name}</text>
          <text x="75" y="40" fill="#93c5fd" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" text-anchor="middle">{title}</text>
        </g>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 1020" width="1450" height="1020">
  <rect width="1450" height="1020" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1350" height="55" rx="8" fill="#1e293b" />
  <text x="725" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.9: UML Sequence Diagram (System Interaction Lifecycle)</text>
  
  <!-- Lifelines -->
  {lines_svg}
  {boxes_svg}
  
  <!-- Messages -->
  <!-- 1. User -> UI -->
  <path d="M 100 190 L 280 190" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="190" y="180" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">1: uploadMRI(scan, mask)</text>
  
  <!-- 2. UI -> Controller -->
  <path d="M 280 230 L 490 230" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="385" y="220" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">2: runAnalysis(image, mask)</text>
  
  <!-- 3. Controller -> Classifier -->
  <path d="M 490 270 L 700 270" stroke="#2563eb" stroke-width="2" marker-end="url(#arrow-blue)" />
  <text x="595" y="260" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">3: predict(preprocessed_image)</text>
  
  <!-- 4. Classifier --> Controller -->
  <path d="M 700 310 L 490 310" stroke="#2563eb" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow-blue)" />
  <text x="595" y="302" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">4: return (predicted_class, confidence)</text>
  
  <!-- 5. Controller -> Explainer -->
  <path d="M 490 350 L 910 350" stroke="#0ea5e9" stroke-width="2" marker-end="url(#arrow-blue)" />
  <text x="700" y="340" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">5: generateGradCAM(image, predicted_class)</text>
  
  <!-- 6. Explainer --> Controller -->
  <path d="M 910 390 L 490 390" stroke="#0ea5e9" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow-blue)" />
  <text x="700" y="382" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">6: return heatmap [h]</text>
  
  <!-- 7. Controller -> Evaluators (Localization) -->
  <path d="M 490 430 L 1120 430" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />
  <text x="805" y="420" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">7: computeLocalization(heatmap, mask)</text>
  
  <!-- 8. Evaluators --> Controller -->
  <path d="M 1120 470 L 490 470" stroke="#059669" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow-green)" />
  <text x="805" y="462" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">8: return localization_scores (IoU, Dice, PointingGame)</text>
  
  <!-- LOOP FRAGMENT: Perturbations -->
  <g transform="translate(440, 500)">
    <rect width="720" height="175" rx="6" fill="#fdf4ff" fill-opacity="0.3" stroke="#c026d3" stroke-width="1.5" stroke-dasharray="4,2" />
    <rect width="210" height="24" rx="4" fill="#c026d3" />
    <text x="105" y="16" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">loop [for each perturbation k = 1..K]</text>
  </g>
  
  <!-- 9. Controller -> Explainer (Perturbed) -->
  <path d="M 490 545 L 910 545" stroke="#a21caf" stroke-width="2" marker-end="url(#arrow)" />
  <text x="700" y="535" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">9: generateGradCAM(perturbed_image_k, class)</text>
  
  <!-- 10. Explainer --> Controller -->
  <path d="M 910 585 L 490 585" stroke="#a21caf" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="700" y="577" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">10: return perturbed_heatmap_k</text>
  
  <!-- 11. Controller -> Evaluators (Stability) -->
  <path d="M 490 625 L 1120 625" stroke="#a21caf" stroke-width="2" marker-end="url(#arrow)" />
  <text x="805" y="615" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">11: evaluateStability(orig_heatmap, [perturbed_heatmaps])</text>
  
  <!-- 12. Evaluators --> Controller -->
  <path d="M 1120 660 L 490 660" stroke="#a21caf" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="805" y="652" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">12: return stability_scores (mean_SSIM, Spearman_rank)</text>
  
  <!-- 13. Controller -> Reliability Analyzer -->
  <path d="M 490 715 L 1310 715" stroke="#db2777" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="900" y="705" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">13: assessReliability(confidence, localization_scores, stability_scores)</text>
  
  <!-- 14. Reliability Analyzer --> Controller -->
  <path d="M 1310 760 L 490 760" stroke="#db2777" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="900" y="750" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">14: return (reliability_score, reliability_flag)</text>
  
  <!-- 15. Controller --> UI -->
  <path d="M 490 815 L 280 815" stroke="#334155" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="385" y="805" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">15: deliverCompleteEvaluation()</text>
  
  <!-- 16. UI --> User -->
  <path d="M 280 875 L 100 875" stroke="#059669" stroke-width="2.5" marker-end="url(#arrow-green)" />
  <text x="190" y="852" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">16: renderDashboard()</text>
  <text x="190" y="867" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">(prediction, overlay, scores, flag)</text>
  
  <!-- Footer Note -->
  <text x="725" y="975" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Full synchronous invocation sequence showing loop over perturbations and subsequent evidence fusion.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_9_sequence_diagram")

# -------------------------------------------------------------
# 12. Figure 4.10: UML Component Diagram
# -------------------------------------------------------------
def build_fig_4_10():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 950" width="1350" height="950">
  <rect width="1350" height="950" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1250" height="55" rx="8" fill="#1e293b" />
  <text x="675" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.10: UML Component Diagram</text>
  
  <!-- Top Component: Dashboard -->
  <g transform="translate(480, 100)">
    <rect width="380" height="110" rx="8" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <!-- Component Tabs on left -->
    <rect x="-12" y="18" width="24" height="15" fill="#2563eb" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#2563eb" rx="2" />
    <text x="190" y="32" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="190" y="58" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Dashboard (Streamlit Web UI)</text>
    <text x="190" y="82" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" text-anchor="middle">Upload, Visual Heatmaps, Metrics &amp; Audit Badges</text>
  </g>
  
  <!-- Middle Central Component: Pipeline Controller -->
  <g transform="translate(480, 290)">
    <rect width="380" height="115" rx="8" fill="#f0f9ff" stroke="#0284c7" stroke-width="2.5" filter="url(#shadow)" />
    <rect x="-12" y="20" width="24" height="15" fill="#0284c7" rx="2" />
    <rect x="-12" y="52" width="24" height="15" fill="#0284c7" rx="2" />
    <text x="190" y="34" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="190" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Pipeline Controller</text>
    <text x="190" y="86" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" text-anchor="middle">Master Orchestration, Configuration &amp; Event Dispatch</text>
  </g>
  
  <!-- Dependency Dashboard -> Controller -->
  <path d="M 670 210 L 670 290" stroke="#334155" stroke-width="2" stroke-dasharray="5,4" marker-end="url(#arrow)" />
  <circle cx="670" cy="245" r="9" fill="#ffffff" stroke="#334155" stroke-width="1.8" />
  <text x="735" y="249" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600">IPipelineControl</text>
  
  <!-- Surrounding Core Modules in Layer -->
  <!-- Module 1: Data Module (Left) -->
  <g transform="translate(60, 480)">
    <rect width="260" height="110" rx="8" fill="#f8fafc" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect x="-12" y="18" width="24" height="15" fill="#10b981" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#10b981" rx="2" />
    <text x="130" y="30" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="130" y="54" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Data Module</text>
    <text x="130" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Dataset Loading &amp; Pre-processing</text>
  </g>
  
  <!-- Module 2: Model Module (Center Left) -->
  <g transform="translate(365, 480)">
    <rect width="270" height="110" rx="8" fill="#f8fafc" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)" />
    <rect x="-12" y="18" width="24" height="15" fill="#3b82f6" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#3b82f6" rx="2" />
    <text x="135" y="30" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="135" y="54" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Model Module</text>
    <text x="135" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">CNN Classifier &amp; Softmax Head</text>
  </g>
  
  <!-- Module 3: Explanation Module (Center Right) -->
  <g transform="translate(685, 480)">
    <rect width="270" height="110" rx="8" fill="#f8fafc" stroke="#8b5cf6" stroke-width="2" filter="url(#shadow)" />
    <rect x="-12" y="18" width="24" height="15" fill="#8b5cf6" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#8b5cf6" rx="2" />
    <text x="135" y="30" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="135" y="54" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Explanation Module</text>
    <text x="135" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Grad-CAM &amp; Saliency Gradients</text>
  </g>
  
  <!-- Module 4: Evaluation Module (Right) -->
  <g transform="translate(1005, 480)">
    <rect width="275" height="110" rx="8" fill="#f8fafc" stroke="#ec4899" stroke-width="2" filter="url(#shadow)" />
    <rect x="-12" y="18" width="24" height="15" fill="#ec4899" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#ec4899" rx="2" />
    <text x="137" y="30" fill="#be185d" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="137" y="54" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Evaluation Module</text>
    <text x="137" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Localization (IoU) &amp; Stability (SSIM)</text>
  </g>
  
  <!-- Connectors from Controller to Modules -->
  <path d="M 520 405 C 480 435, 230 435, 200 480" fill="none" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 580 405 L 515 480" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 760 405 L 820 480" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 820 405 C 860 435, 1100 435, 1140 480" fill="none" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  
  <!-- Module 5: Reliability Module -->
  <g transform="translate(480, 640)">
    <rect width="380" height="100" rx="8" fill="#fdf2f8" stroke="#db2777" stroke-width="2.5" filter="url(#shadow)" />
    <rect x="-12" y="18" width="24" height="15" fill="#db2777" rx="2" />
    <rect x="-12" y="48" width="24" height="15" fill="#db2777" rx="2" />
    <text x="190" y="30" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="190" y="54" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Reliability Module</text>
    <text x="190" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" text-anchor="middle">Evidence Fusion Model, Threshold Flagging &amp; AUROC</text>
  </g>
  <path d="M 670 405 L 670 640" stroke="#db2777" stroke-width="2" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <text x="735" y="625" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold">IReliability</text>
  
  <!-- Bottom Component: Storage Component -->
  <g transform="translate(300, 785)">
    <rect width="740" height="110" rx="8" fill="#f8fafc" stroke="#64748b" stroke-width="2" filter="url(#shadow)" />
    <rect x="-12" y="20" width="24" height="15" fill="#64748b" rx="2" />
    <rect x="-12" y="52" width="24" height="15" fill="#64748b" rx="2" />
    <text x="370" y="32" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;component&gt;&gt;</text>
    <text x="370" y="58" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Storage Subsystem (Files, Checkpoints &amp; Logs)</text>
    <text x="370" y="84" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">MRI Slices &amp; Masks | PyTorch .pt Model Weights | SQLite / CSV Audit Records</text>
  </g>
  
  <!-- Links to Storage -->
  <path d="M 190 590 C 190 730, 360 740, 420 785" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 500 590 L 500 785" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 820 590 L 820 785" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 1140 590 C 1140 730, 950 740, 890 785" fill="none" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <path d="M 670 740 L 670 785" stroke="#64748b" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  
  <!-- Footer Note -->
  <text x="675" y="925" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Modular component decoupling: Dashboard interacts solely with Controller, which dispatches requests to domain modules and data storage.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_10_component_diagram")

# -------------------------------------------------------------
# 13. Figure 4.11: UML Deployment Diagram
# -------------------------------------------------------------
def build_fig_4_11():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 900" width="1350" height="900">
  <rect width="1350" height="900" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1250" height="55" rx="8" fill="#1e293b" />
  <text x="675" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.11: UML Deployment Diagram (Physical Node Deployment)</text>
  
  <!-- NODE 1: Reviewer Client Device (Top Left) -->
  <g transform="translate(60, 130)">
    <!-- 3D Node box -->
    <path d="M 0 20 L 20 0 L 360 0 L 340 20 Z" fill="#cbd5e1" stroke="#475569" stroke-width="1.5" />
    <path d="M 360 0 L 360 210 L 340 230 L 340 20 Z" fill="#94a3b8" stroke="#475569" stroke-width="1.5" />
    <rect x="0" y="20" width="340" height="210" fill="#f8fafc" stroke="#475569" stroke-width="1.5" filter="url(#shadow)" />
    
    <text x="170" y="50" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">&lt;&lt;device&gt;&gt;</text>
    <text x="170" y="72" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Client Workstation / Laptop</text>
    
    <!-- Nested Execution Environment -->
    <rect x="25" y="95" width="290" height="110" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" />
    <text x="170" y="120" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;execution environment&gt;&gt;</text>
    <text x="170" y="142" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Web Browser (Chrome / Edge)</text>
    
    <!-- Artifact -->
    <rect x="45" y="155" width="250" height="36" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-width="1" />
    <text x="170" y="178" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle">&lt;&lt;artifact&gt;&gt; Streamlit Client SPA</text>
  </g>
  
  <!-- NODE 2: Application Server Node (Top Right) -->
  <g transform="translate(760, 110)">
    <path d="M 0 20 L 20 0 L 520 0 L 500 20 Z" fill="#cbd5e1" stroke="#475569" stroke-width="1.5" />
    <path d="M 520 0 L 520 280 L 500 300 L 500 20 Z" fill="#94a3b8" stroke="#475569" stroke-width="1.5" />
    <rect x="0" y="20" width="500" height="280" fill="#f8fafc" stroke="#475569" stroke-width="1.5" filter="url(#shadow)" />
    
    <text x="250" y="50" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">&lt;&lt;execution environment&gt;&gt;</text>
    <text x="250" y="72" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Application Server (Python 3.10+ Host)</text>
    
    <!-- Artifacts inside App Server -->
    <rect x="25" y="95" width="450" height="50" rx="6" fill="#f0f9ff" stroke="#0ea5e9" stroke-width="1.2" />
    <text x="40" y="118" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Streamlit Web Server Engine</text>
    <text x="40" y="135" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Hosts interactive portal, widgets, and dynamic UI state</text>
    
    <rect x="25" y="160" width="450" height="55" rx="6" fill="#f0f9ff" stroke="#0ea5e9" stroke-width="1.2" />
    <text x="40" y="183" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Pipeline Controller Service</text>
    <text x="40" y="202" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Dispatches tasks to GPU and coordinates evidence fusion</text>
    
    <rect x="25" y="230" width="450" height="48" rx="6" fill="#f0f9ff" stroke="#0ea5e9" stroke-width="1.2" />
    <text x="40" y="253" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Evaluation &amp; Scoring Framework</text>
    <text x="40" y="270" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">IoU, Dice, Pointing Game, SSIM, Spearman Rank &amp; AUROC</text>
  </g>
  
  <!-- Communication Link 1: Client to App Server -->
  <line x1="400" y1="210" x2="760" y2="210" stroke="#2563eb" stroke-width="2.5" />
  <rect x="490" y="180" width="180" height="26" rx="4" fill="#ffffff" stroke="#93c5fd" stroke-width="1" />
  <text x="580" y="198" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">HTTP / WebSockets : 8501</text>
  
  <!-- NODE 3: GPU Compute Node (Bottom Right) -->
  <g transform="translate(760, 470)">
    <path d="M 0 20 L 20 0 L 520 0 L 500 20 Z" fill="#cbd5e1" stroke="#475569" stroke-width="1.5" />
    <path d="M 520 0 L 520 300 L 500 320 L 500 20 Z" fill="#94a3b8" stroke="#475569" stroke-width="1.5" />
    <rect x="0" y="20" width="500" height="300" fill="#f8fafc" stroke="#475569" stroke-width="1.5" filter="url(#shadow)" />
    
    <text x="250" y="50" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">&lt;&lt;device&gt;&gt;</text>
    <text x="250" y="72" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">GPU Compute Server (NVIDIA CUDA Node)</text>
    
    <!-- Environment -->
    <rect x="25" y="95" width="450" height="50" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="1.2" />
    <text x="40" y="118" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;runtime&gt;&gt; PyTorch 2.x &amp; CUDA Drivers</text>
    <text x="40" y="135" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Tensor parallel processing, automatic mixed precision</text>
    
    <!-- Artifacts -->
    <rect x="25" y="160" width="450" height="50" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="1.2" />
    <text x="40" y="183" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; ResNet50 / DenseNet121 Deep Classifier</text>
    <text x="40" y="200" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Forward pass inference, feature extractors, softmax confidence</text>
    
    <rect x="25" y="225" width="450" height="50" rx="6" fill="#ecfdf5" stroke="#10b981" stroke-width="1.2" />
    <text x="40" y="248" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Grad-CAM &amp; Perturbation Engine</text>
    <text x="40" y="265" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Gradient backpropagation &amp; multi-pass perturbed heatmaps</text>
  </g>
  
  <!-- Communication Link 2: App Server to GPU Compute -->
  <line x1="1010" y1="410" x2="1010" y2="470" stroke="#059669" stroke-width="2.5" />
  <rect x="915" y="425" width="190" height="26" rx="4" fill="#ffffff" stroke="#a7f3d0" stroke-width="1" />
  <text x="1010" y="443" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">PyTorch CUDA API / PCIe</text>
  
  <!-- NODE 4: Storage Subsystem (Bottom Left) -->
  <g transform="translate(60, 470)">
    <path d="M 0 20 L 20 0 L 520 0 L 500 20 Z" fill="#cbd5e1" stroke="#475569" stroke-width="1.5" />
    <path d="M 520 0 L 520 300 L 500 320 L 500 20 Z" fill="#94a3b8" stroke="#475569" stroke-width="1.5" />
    <rect x="0" y="20" width="500" height="300" fill="#f8fafc" stroke="#475569" stroke-width="1.5" filter="url(#shadow)" />
    
    <text x="250" y="50" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">&lt;&lt;storage device&gt;&gt;</text>
    <text x="250" y="72" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Local SSD / High-Speed Storage Subsystem</text>
    
    <rect x="25" y="95" width="450" height="50" rx="6" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.2" />
    <text x="40" y="118" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Brain MRI Slices &amp; Expert Masks Store</text>
    <text x="40" y="135" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Figshare CE-T1 (3,064 slices) &amp; BraTS benchmarks</text>
    
    <rect x="25" y="160" width="450" height="50" rx="6" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.2" />
    <text x="40" y="183" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Trained Model Checkpoints (.pth / .pt)</text>
    <text x="40" y="200" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Pre-trained backbones &amp; best validation checkpoint weights</text>
    
    <rect x="25" y="225" width="450" height="50" rx="6" fill="#f5f3ff" stroke="#8b5cf6" stroke-width="1.2" />
    <text x="40" y="248" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">&lt;&lt;artifact&gt;&gt; Experiment Results &amp; Audit DB (SQLite / CSV)</text>
    <text x="40" y="265" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">Logged prediction histories, stability scores &amp; reliability flags</text>
  </g>
  
  <!-- Communication Link 3: App Server to Storage -->
  <path d="M 760 350 L 560 520" stroke="#8b5cf6" stroke-width="2.2" />
  <rect x="580" y="420" width="160" height="26" rx="4" fill="#ffffff" stroke="#ddd6fe" stroke-width="1" />
  <text x="660" y="438" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Local File I/O / OS Paths</text>
  
  <!-- Communication Link 4: GPU to Storage (weights and dataset) -->
  <line x1="560" y1="620" x2="760" y2="620" stroke="#8b5cf6" stroke-width="2.2" />
  <rect x="590" y="605" width="140" height="26" rx="4" fill="#ffffff" stroke="#ddd6fe" stroke-width="1" />
  <text x="660" y="623" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Direct Disk I/O (DMA)</text>
  
  <!-- Footer Note -->
  <text x="675" y="855" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Deployment architecture can run locally on an AI workstation or inside Google Colab / Kaggle cloud with GPU acceleration.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_11_deployment_diagram")

# -------------------------------------------------------------
# 14. Figure 4.12: UML State Machine Diagram
# -------------------------------------------------------------
def build_fig_4_12():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 950" width="1350" height="950">
  <rect width="1350" height="950" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1250" height="55" rx="8" fill="#1e293b" />
  <text x="675" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.12: UML State Machine Diagram (Inference &amp; Verification State Lifecycle)</text>
  
  <!-- Start Node -->
  <circle cx="100" cy="180" r="16" fill="#0f172a" />
  <text x="100" y="145" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic" text-anchor="middle">Start</text>
  
  <path d="M 116 180 L 180 180" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  
  <!-- STATE 1: Idle -->
  <g transform="translate(180, 140)">
    <rect width="200" height="80" rx="12" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="200" height="28" rx="12" fill="#2563eb" />
    <rect y="18" width="200" height="10" fill="#2563eb" />
    <text x="100" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Idle</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / initialize_session()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / wait_for_user_upload()</text>
  </g>
  
  <!-- Transition 1: Idle -> Image Loaded -->
  <path d="M 380 180 L 480 180" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="430" y="165" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">upload_scan [valid]</text>
  
  <!-- STATE 2: Image Loaded -->
  <g transform="translate(480, 140)">
    <rect width="230" height="80" rx="12" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="230" height="28" rx="12" fill="#2563eb" />
    <rect y="18" width="230" height="10" fill="#2563eb" />
    <text x="115" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Image Loaded</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / parse_dicom_or_png()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / preprocess_and_normalize()</text>
  </g>
  
  <!-- Transition 2: Image Loaded -> Classified -->
  <path d="M 710 180 L 810 180" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="760" y="165" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">run_classifier()</text>
  
  <!-- STATE 3: Classified -->
  <g transform="translate(810, 140)">
    <rect width="230" height="80" rx="12" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="230" height="28" rx="12" fill="#2563eb" />
    <rect y="18" width="230" height="10" fill="#2563eb" />
    <text x="115" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Classified</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / forward_pass()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">exit / record_class_and_conf()</text>
  </g>
  
  <!-- Transition 3: Classified -> Explained -->
  <path d="M 1040 180 L 1110 180 C 1140 180, 1140 280, 1140 330" fill="none" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="1185" y="260" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">explain()</text>
  
  <!-- STATE 4: Explained -->
  <g transform="translate(1010, 330)">
    <rect width="250" height="85" rx="12" fill="#f8fafc" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)" />
    <rect width="250" height="28" rx="12" fill="#0284c7" />
    <rect y="18" width="250" height="10" fill="#0284c7" />
    <text x="125" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Explained</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / gradcam_backward()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / generate_heatmap_overlay()</text>
  </g>
  
  <!-- Transition 4: Explained -> Evaluated -->
  <path d="M 1010 375 L 850 375" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="930" y="365" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">eval_localization(mask)</text>
  
  <!-- STATE 5: Evaluated (Localization) -->
  <g transform="translate(580, 330)">
    <rect width="270" height="85" rx="12" fill="#f8fafc" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect width="270" height="28" rx="12" fill="#059669" />
    <rect y="18" width="270" height="10" fill="#059669" />
    <text x="135" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Evaluated (Localization)</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / binarize_heatmap()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / compute_iou_dice_energy()</text>
  </g>
  
  <!-- Transition 5: Evaluated -> Perturbation Analysis -->
  <path d="M 580 375 L 430 375" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="505" y="365" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">perturb_inputs()</text>
  
  <!-- STATE 6: Perturbation Analysis -->
  <g transform="translate(150, 330)">
    <rect width="280" height="85" rx="12" fill="#f8fafc" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <rect width="280" height="28" rx="12" fill="#a21caf" />
    <rect y="18" width="280" height="10" fill="#a21caf" />
    <text x="140" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Perturbation Analysis</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / for_k_in_perturbations: re_explain()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">exit / compute_ssim_and_rank_corr()</text>
  </g>
  
  <!-- Transition 6: Perturbation Analysis -> Scored -->
  <path d="M 290 415 L 290 520" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="345" y="470" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">fuse_evidence()</text>
  
  <!-- STATE 7: Scored -->
  <g transform="translate(150, 520)">
    <rect width="280" height="85" rx="12" fill="#fdf2f8" stroke="#db2777" stroke-width="2.5" filter="url(#shadow)" />
    <rect width="280" height="28" rx="12" fill="#be185d" />
    <rect y="18" width="280" height="10" fill="#be185d" />
    <text x="140" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Scored (Reliability)</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / calculate_composite_score()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / compare_against_threshold()</text>
  </g>
  
  <!-- Transition 7: Scored -> Result Ready -->
  <path d="M 430 562 L 600 562" stroke="#334155" stroke-width="2" marker-end="url(#arrow)" />
  <text x="515" y="550" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">set_flag()</text>
  
  <!-- STATE 8: Result Ready -->
  <g transform="translate(600, 520)">
    <rect width="270" height="85" rx="12" fill="#f0fdf4" stroke="#10b981" stroke-width="2.5" filter="url(#shadow)" />
    <rect width="270" height="28" rx="12" fill="#059669" />
    <rect y="18" width="270" height="10" fill="#059669" />
    <text x="135" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Result Ready</text>
    <text x="15" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / render_dashboard_panels()</text>
    <text x="15" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / allow_report_pdf_download()</text>
  </g>
  
  <!-- Transition 8: Result Ready -> Idle (New Analysis) -->
  <path d="M 735 520 C 735 480, 280 480, 280 220" fill="none" stroke="#2563eb" stroke-width="1.8" stroke-dasharray="5,3" marker-end="url(#arrow-blue)" />
  <text x="460" y="465" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">new_analysis() / reset</text>
  
  <!-- STATE 9: Error State (Bottom) -->
  <g transform="translate(480, 740)">
    <rect width="360" height="85" rx="12" fill="#fef2f2" stroke="#ef4444" stroke-width="2.5" filter="url(#shadow)" />
    <rect width="360" height="28" rx="12" fill="#dc2626" />
    <rect y="18" width="360" height="10" fill="#dc2626" />
    <text x="180" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Error State</text>
    <text x="20" y="48" fill="#991b1b" font-family="'Segoe UI', Arial, sans-serif" font-size="11">entry / log_error_details() &amp; notify_user()</text>
    <text x="20" y="68" fill="#991b1b" font-family="'Segoe UI', Arial, sans-serif" font-size="11">do / display_clinical_advisory_alert()</text>
  </g>
  
  <!-- Error Transitions to Error State -->
  <path d="M 600 220 C 600 300, 520 650, 520 740" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow-red)" />
  <path d="M 925 220 C 925 350, 680 650, 680 740" fill="none" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow-red)" />
  <text x="740" y="695" fill="#dc2626" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold">[invalid image / dimension mismatch / exception]</text>
  
  <!-- Reset from Error to Idle -->
  <path d="M 480 780 C 100 780, 80 350, 200 220" fill="none" stroke="#dc2626" stroke-width="1.8" marker-end="url(#arrow-red)" />
  <text x="120" y="550" fill="#dc2626" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold">dismiss_error() / reset</text>
  
  <!-- Footer Note -->
  <text x="675" y="885" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">State transitions guarantee deterministic execution of inference, Grad-CAM generation, perturbation checks, and fail-safe error recovery.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_12_state_machine_diagram")

if __name__ == "__main__":
    print("Building Part 3 diagrams...")
    build_fig_4_7()
    build_fig_4_8()
    build_fig_4_9()
    build_fig_4_10()
    build_fig_4_11()
    build_fig_4_12()
    print("Part 3 complete!")
