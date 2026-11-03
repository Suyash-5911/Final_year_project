import os
from render_utils import render_svg_to_png_and_jpg
from build_part1 import SVG_DEFS

# -------------------------------------------------------------
# 5. Figure 4.3: DFD Level 1
# -------------------------------------------------------------
def build_fig_4_3():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1450 1000" width="1450" height="1000">
  <rect width="1450" height="1000" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Title Header -->
  <rect x="50" y="25" width="1350" height="55" rx="8" fill="#1e293b" />
  <text x="725" y="60" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.3: Data Flow Diagram (DFD Level 1 - Subsystem Overview)</text>
  
  <!-- External Entities -->
  <!-- Researcher (Top Left) -->
  <g transform="translate(60, 120)">
    <rect width="210" height="100" rx="8" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="210" height="28" rx="8" fill="#2563eb" />
    <rect y="20" width="210" height="8" fill="#2563eb" />
    <text x="105" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;External Entity&gt;&gt;</text>
    <text x="105" y="55" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Researcher / Student</text>
    <text x="105" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Config, Data &amp; Masks</text>
  </g>
  
  <!-- Radiologist / Reviewer (Bottom Right) -->
  <g transform="translate(1180, 780)">
    <rect width="210" height="100" rx="8" fill="#f8fafc" stroke="#7c3aed" stroke-width="2" filter="url(#shadow)" />
    <rect width="210" height="28" rx="8" fill="#7c3aed" />
    <rect y="20" width="210" height="8" fill="#7c3aed" />
    <text x="105" y="20" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&lt;&lt;External Entity&gt;&gt;</text>
    <text x="105" y="55" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Radiologist / Reviewer</text>
    <text x="105" y="78" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Clinical Inspection &amp; Audit</text>
  </g>
  
  <!-- Data Stores -->
  <!-- D1: MRI Images and Masks -->
  <g transform="translate(60, 360)">
    <path d="M 0 0 L 210 0 M 0 60 L 210 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="210" height="60" fill="#f0f9ff" />
    <line x1="45" y1="0" x2="45" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="22" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">D1</text>
    <text x="125" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">MRI Images &amp; Masks</text>
    <text x="125" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Raw Scans &amp; Expert Masks</text>
  </g>
  
  <!-- D2: Model Weights -->
  <g transform="translate(420, 120)">
    <path d="M 0 0 L 220 0 M 0 60 L 220 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="220" height="60" fill="#f0f9ff" />
    <line x1="45" y1="0" x2="45" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="22" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">D2</text>
    <text x="132" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Model Weights</text>
    <text x="132" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">PyTorch Trained Checkpoints</text>
  </g>
  
  <!-- D3: Heatmaps and Metrics -->
  <g transform="translate(1180, 360)">
    <path d="M 0 0 L 220 0 M 0 60 L 220 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="220" height="60" fill="#f0f9ff" />
    <line x1="45" y1="0" x2="45" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="22" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">D3</text>
    <text x="132" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Heatmaps &amp; Metrics</text>
    <text x="132" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Grad-CAM, IoU, SSIM Scores</text>
  </g>
  
  <!-- D4: Experiment Logs -->
  <g transform="translate(780, 800)">
    <path d="M 0 0 L 220 0 M 0 60 L 220 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="220" height="60" fill="#f0f9ff" />
    <line x1="45" y1="0" x2="45" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="22" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">D4</text>
    <text x="132" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Experiment Logs</text>
    <text x="132" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Run History &amp; Reliability Logs</text>
  </g>
  
  <!-- 6 Processes -->
  <!-- Process 1.0: Data Preparation -->
  <g transform="translate(60, 520)">
    <circle cx="100" cy="80" r="75" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">1.0</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Data</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Preparation</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">&amp; Augmentation</text>
  </g>
  
  <!-- Process 2.0: Classification -->
  <g transform="translate(420, 310)">
    <circle cx="100" cy="80" r="75" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">2.0</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Tumour</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Classification</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">CNN Inference</text>
  </g>
  
  <!-- Process 3.0: Explanation Generation -->
  <g transform="translate(800, 240)">
    <circle cx="100" cy="80" r="75" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">3.0</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Explanation</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Generation</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Grad-CAM Mapping</text>
  </g>
  
  <!-- Process 4.0: Evidence Evaluation -->
  <g transform="translate(800, 520)">
    <circle cx="100" cy="80" r="80" fill="#ecfdf5" stroke="#059669" stroke-width="3" filter="url(#shadow)" />
    <line x1="20" y1="50" x2="180" y2="50" stroke="#a7f3d0" stroke-width="1.5" />
    <text x="100" y="40" fill="#059669" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">4.0</text>
    <text x="100" y="73" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Evidence Evaluation</text>
    <text x="100" y="93" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Localization (IoU/Dice)</text>
    <text x="100" y="110" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">&amp; Stability (SSIM/Rank)</text>
  </g>
  
  <!-- Process 5.0: Reliability Assessment -->
  <g transform="translate(420, 680)">
    <circle cx="100" cy="80" r="75" fill="#fdf2f8" stroke="#db2777" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#fbcfe8" stroke-width="1.5" />
    <text x="100" y="40" fill="#db2777" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">5.0</text>
    <text x="100" y="75" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Reliability</text>
    <text x="100" y="95" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Assessment</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Score &amp; Flag Fusion</text>
  </g>
  
  <!-- Process 6.0: Result Presentation -->
  <g transform="translate(1180, 560)">
    <circle cx="100" cy="80" r="75" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">6.0</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Result</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Presentation</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Dashboard Views</text>
  </g>
  
  <!-- DATA FLOW ARROWS & LABELS -->
  <!-- 1. Researcher -> D1 -->
  <path d="M 160 220 L 160 360" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="70" y="270" width="120" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="130" y="284" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Dataset &amp; Masks</text>
  
  <!-- 2. D1 -> Process 1.0 -->
  <path d="M 160 420 L 160 520" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="80" y="460" width="110" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="135" y="474" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Raw MRI Images</text>
  
  <!-- 3. Process 1.0 -> Process 2.0 -->
  <path d="M 235 570 C 330 540, 380 460, 430 430" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="270" y="490" width="140" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="340" y="504" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Pre-processed Images</text>
  
  <!-- 4. D2 <-> Process 2.0 -->
  <path d="M 520 180 L 520 310" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <path d="M 505 310 L 505 180" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="430" y="235" width="145" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="502" y="249" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Read/Write Weights</text>
  
  <!-- 5. Process 2.0 -> Process 3.0 -->
  <path d="M 595 365 C 670 330, 720 310, 800 310" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="635" y="315" width="150" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="710" y="329" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Predicted Class &amp; Feature</text>
  
  <!-- 6. Process 2.0 -> Process 5.0 -->
  <path d="M 520 460 L 520 680" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="445" y="560" width="120" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="505" y="574" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Class &amp; Confidence</text>
  
  <!-- 7. Process 3.0 -> Process 4.0 -->
  <path d="M 900 390 L 900 520" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="830" y="445" width="140" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="900" y="459" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Grad-CAM Heatmaps</text>
  
  <!-- 8. Process 3.0 -> D3 -->
  <path d="M 975 320 C 1050 320, 1100 360, 1180 375" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="1035" y="335" width="115" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="1092" y="349" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Store Heatmaps</text>
  
  <!-- 9. D1 -> Process 4.0 (Masks) -->
  <path d="M 270 380 C 450 370, 700 480, 805 560" fill="none" stroke="#0284c7" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow-blue)" />
  <rect x="480" y="450" width="140" height="20" rx="3" fill="#f0f9ff" stroke="#bae6fd" />
  <text x="550" y="464" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Expert Tumour Masks</text>
  
  <!-- 10. Process 4.0 -> D3 -->
  <path d="M 980 570 C 1060 550, 1130 460, 1185 410" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="1040" y="480" width="130" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="1105" y="494" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Store Evidence Scores</text>
  
  <!-- 11. Process 4.0 -> Process 5.0 -->
  <path d="M 800 620 C 720 660, 670 700, 595 730" fill="none" stroke="#059669" stroke-width="2" marker-end="url(#arrow-green)" />
  <rect x="645" y="665" width="150" height="20" rx="3" fill="#ecfdf5" stroke="#a7f3d0" />
  <text x="720" y="679" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">Loc &amp; Stability Scores</text>
  
  <!-- 12. Process 5.0 -> D4 -->
  <path d="M 595 780 L 780 810" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="635" y="785" width="120" height="20" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="695" y="799" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Reliability Logs</text>
  
  <!-- 13. Process 5.0 -> Process 6.0 -->
  <path d="M 595 750 C 780 770, 990 730, 1180 655" fill="none" stroke="#db2777" stroke-width="2" marker-end="url(#arrow)" />
  <rect x="850" y="725" width="160" height="20" rx="3" fill="#fdf2f8" stroke="#fbcfe8" />
  <text x="930" y="739" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">Reliability Flag &amp; Metrics</text>
  
  <!-- 14. Process 6.0 -> Radiologist -->
  <path d="M 1285 710 L 1285 780" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow)" />
  <rect x="1205" y="735" width="155" height="20" rx="3" fill="#f5f3ff" stroke="#ddd6fe" />
  <text x="1282" y="749" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">Rendered Heatmap &amp; Flag</text>
  
  <!-- Radiologist Scan Upload -> Process 1.0 -->
  <path d="M 1180 840 C 700 980, 150 900, 140 670" fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <rect x="520" y="905" width="170" height="22" rx="3" fill="#f5f3ff" stroke="#ddd6fe" />
  <text x="605" y="920" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">Uploaded Scan &amp; Analysis Query</text>
  
  <!-- Footer Note -->
  <text x="725" y="970" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Six functional processes interconnect with 4 data repositories to achieve evidence-aware classification and auditability.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_3_dfd_level_1")

# -------------------------------------------------------------
# 6. Figure 4.4: DFD Level 2 (Expansion of Process 4.0)
# -------------------------------------------------------------
def build_fig_4_4():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 820" width="1350" height="820">
  <rect width="1350" height="820" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="25" width="1250" height="55" rx="8" fill="#1e293b" />
  <text x="675" y="60" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.4: Data Flow Diagram (DFD Level 2 - Expansion of Process 4.0: Evidence Evaluation)</text>
  
  <!-- Subsystem Boundary Box -->
  <rect x="50" y="105" width="1250" height="660" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5" stroke-dasharray="6,4" />
  <text x="75" y="130" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">BOUNDARY: Process 4.0 Evidence Evaluation Sub-system</text>
  
  <!-- External Inputs to 4.0 -->
  <!-- Heatmap from 3.0 -->
  <g transform="translate(80, 160)">
    <rect width="180" height="60" rx="6" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow)" />
    <text x="90" y="28" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">From Process 3.0</text>
    <text x="90" y="48" fill="#3b82f6" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Grad-CAM Heatmap [h]</text>
  </g>
  
  <!-- Expert Mask from D1 -->
  <g transform="translate(80, 310)">
    <path d="M 0 0 L 180 0 M 0 60 L 180 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="180" height="60" fill="#f0f9ff" />
    <line x1="35" y1="0" x2="35" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="17" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">D1</text>
    <text x="107" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Expert Tumour Mask</text>
    <text x="107" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Ground Truth Mask [m]</text>
  </g>
  
  <!-- Input Image & Model from D2 -->
  <g transform="translate(80, 480)">
    <path d="M 0 0 L 180 0 M 0 60 L 180 60" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="180" height="60" fill="#f0f9ff" />
    <line x1="35" y1="0" x2="35" y2="60" stroke="#0284c7" stroke-width="2" />
    <text x="17" y="36" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">D2</text>
    <text x="107" y="28" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Model &amp; Preproc Image</text>
    <text x="107" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Weights &amp; Scan [x']</text>
  </g>
  
  <!-- 4 Sub-Processes -->
  <!-- 4.1 Binarize Heatmap -->
  <g transform="translate(360, 150)">
    <circle cx="100" cy="80" r="75" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="50" x2="175" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">4.1</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Binarize</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Heatmap</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Threshold / Otsu [hb]</text>
  </g>
  
  <!-- 4.2 Compute Localization Scores -->
  <g transform="translate(690, 220)">
    <circle cx="110" cy="90" r="85" fill="#ecfdf5" stroke="#059669" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="55" x2="195" y2="55" stroke="#a7f3d0" stroke-width="1.5" />
    <text x="110" y="42" fill="#059669" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">4.2</text>
    <text x="110" y="78" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Compute Localization</text>
    <text x="110" y="98" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Scores</text>
    <text x="110" y="118" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">• IoU = |hb ∩ m| / |hb ∪ m|</text>
    <text x="110" y="133" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">• Dice = 2|hb ∩ m| / (|hb| + |m|)</text>
    <text x="110" y="148" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">• Pointing Game &amp; Energy</text>
  </g>
  
  <!-- 4.3 Generate Perturbed Images and Re-explain -->
  <g transform="translate(360, 480)">
    <circle cx="100" cy="85" r="80" fill="#f8fafc" stroke="#2563eb" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="20" y1="50" x2="180" y2="50" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="100" y="40" fill="#2563eb" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">4.3</text>
    <text x="100" y="75" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Generate Perturbed</text>
    <text x="100" y="95" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Images &amp; Re-explain</text>
    <text x="100" y="115" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Gaussian, Blur, Shifts</text>
    <text x="100" y="130" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">K Heatmaps [hk]</text>
  </g>
  
  <!-- 4.4 Compute Stability Scores -->
  <g transform="translate(690, 490)">
    <circle cx="110" cy="90" r="85" fill="#fdf4ff" stroke="#c026d3" stroke-width="2.5" filter="url(#shadow)" />
    <line x1="25" y1="55" x2="195" y2="55" stroke="#f5d0fe" stroke-width="1.5" />
    <text x="110" y="42" fill="#c026d3" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">4.4</text>
    <text x="110" y="78" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Compute Stability</text>
    <text x="110" y="98" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Scores</text>
    <text x="110" y="118" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">• SSIM: 1/K Σ sim(h, hk)</text>
    <text x="110" y="133" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">• Spearman Rank Correlation (ρ)</text>
  </g>
  
  <!-- Outputs to Process 5.0 and D3 -->
  <!-- Process 5.0 Destination -->
  <g transform="translate(1080, 280)">
    <rect width="190" height="90" rx="8" fill="#fdf2f8" stroke="#db2777" stroke-width="2" filter="url(#shadow)" />
    <text x="95" y="28" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">To Process 5.0</text>
    <text x="95" y="52" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Reliability Assessment</text>
    <text x="95" y="74" fill="#db2777" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600" text-anchor="middle">Evidence Fusion Vector</text>
  </g>
  
  <!-- D3 Destination -->
  <g transform="translate(1080, 480)">
    <path d="M 0 0 L 190 0 M 0 65 L 190 65" stroke="#0284c7" stroke-width="2" />
    <rect x="0" y="0" width="190" height="65" fill="#f0f9ff" />
    <line x1="35" y1="0" x2="35" y2="65" stroke="#0284c7" stroke-width="2" />
    <text x="17" y="38" fill="#0284c7" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">D3</text>
    <text x="112" y="30" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Heatmaps &amp; Metrics</text>
    <text x="112" y="50" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" text-anchor="middle">Store IoU, SSIM, Ranks</text>
  </g>
  
  <!-- Connectors inside DFD Level 2 -->
  <!-- 1. Heatmap -> 4.1 -->
  <path d="M 260 190 L 360 215" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="270" y="180" width="85" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="312" y="193" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Heatmap [h]</text>
  
  <!-- 2. Heatmap -> 4.4 (for comparison) -->
  <path d="M 260 210 C 330 250, 600 350, 695 530" fill="none" stroke="#334155" stroke-width="1.8" stroke-dasharray="4,3" marker-end="url(#arrow)" />
  <rect x="440" y="380" width="105" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="492" y="393" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Original Heatmap [h]</text>
  
  <!-- 3. 4.1 -> 4.2 -->
  <path d="M 535 240 L 690 270" stroke="#2563eb" stroke-width="1.8" marker-end="url(#arrow-blue)" />
  <rect x="555" y="235" width="125" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="617" y="248" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Binarized Mask [hb]</text>
  
  <!-- 4. Mask from D1 -> 4.2 -->
  <path d="M 260 340 L 690 315" stroke="#0284c7" stroke-width="1.8" marker-end="url(#arrow-blue)" />
  <rect x="390" y="318" width="125" height="18" rx="3" fill="#f0f9ff" stroke="#bae6fd" />
  <text x="452" y="331" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Ground Truth Mask [m]</text>
  
  <!-- 5. Image & Model from D2 -> 4.3 -->
  <path d="M 260 520 L 360 545" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <rect x="268" y="515" width="90" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="313" y="528" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Scan &amp; Weights</text>
  
  <!-- 6. 4.3 -> 4.4 -->
  <path d="M 540 570 L 690 575" stroke="#2563eb" stroke-width="1.8" marker-end="url(#arrow-blue)" />
  <rect x="555" y="555" width="120" height="18" rx="3" fill="#ffffff" stroke="#cbd5e1" />
  <text x="615" y="568" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="9.5" text-anchor="middle">Perturbed Maps [hk]</text>
  
  <!-- 7. 4.2 -> Process 5.0 -->
  <path d="M 910 270 L 1080 305" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-green)" />
  <rect x="925" y="265" width="145" height="22" rx="3" fill="#ecfdf5" stroke="#a7f3d0" />
  <text x="997" y="280" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">Localization Scores L</text>
  
  <!-- 8. 4.4 -> Process 5.0 -->
  <path d="M 905 520 C 970 480, 1020 400, 1080 350" fill="none" stroke="#c026d3" stroke-width="2.2" marker-end="url(#arrow)" />
  <rect x="945" y="425" width="135" height="22" rx="3" fill="#fdf4ff" stroke="#f5d0fe" />
  <text x="1012" y="440" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" text-anchor="middle">Stability Scores T</text>
  
  <!-- 9. 4.2 & 4.4 -> D3 -->
  <path d="M 905 320 C 980 350, 1030 450, 1080 495" fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  <path d="M 910 600 L 1080 540" stroke="#334155" stroke-width="1.8" marker-end="url(#arrow)" />
  
  <!-- Footer Note -->
  <text x="675" y="790" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Detailed sub-process breakdown showing binarization, IoU/Dice localization, perturbation loop, and SSIM stability calculation.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_4_dfd_level_2")

# -------------------------------------------------------------
# 7. Figure 4.5: ER Diagram
# -------------------------------------------------------------
def build_fig_4_5():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1400 950" width="1400" height="950">
  <rect width="1400" height="950" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1300" height="55" rx="8" fill="#1e293b" />
  <text x="700" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.5: Entity-Relationship (ER) Diagram (Logical Data Model)</text>
  
  <!-- Entity 1: PATIENT -->
  <g transform="translate(60, 130)">
    <rect width="260" height="180" rx="8" fill="#ffffff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="260" height="35" rx="8" fill="#2563eb" />
    <rect y="25" width="260" height="10" fill="#2563eb" />
    <text x="130" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">PATIENT</text>
    
    <text x="20" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• patient_id (PK)</text>
    <line x1="20" y1="64" x2="115" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="20" y="90" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• deidentified_code: varchar</text>
    <text x="20" y="118" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• age_group: varchar</text>
    <text x="20" y="146" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• gender: char(1)</text>
  </g>
  
  <!-- Relationship 1: Has (1 : N) -->
  <g transform="translate(410, 180)">
    <polygon points="50,0 100,35 50,70 0,35" fill="#eff6ff" stroke="#3b82f6" stroke-width="2" />
    <text x="50" y="40" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Has</text>
  </g>
  <line x1="320" y1="215" x2="410" y2="215" stroke="#334155" stroke-width="2" />
  <text x="340" y="205" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  <line x1="510" y1="215" x2="600" y2="215" stroke="#334155" stroke-width="2" />
  <text x="580" y="205" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">N</text>
  
  <!-- Entity 2: MRI_IMAGE -->
  <g transform="translate(600, 110)">
    <rect width="280" height="220" rx="8" fill="#ffffff" stroke="#0284c7" stroke-width="2" filter="url(#shadow)" />
    <rect width="280" height="35" rx="8" fill="#0284c7" />
    <rect y="25" width="280" height="10" fill="#0284c7" />
    <text x="140" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">MRI_IMAGE</text>
    
    <text x="20" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• image_id (PK)</text>
    <line x1="20" y1="64" x2="105" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="20" y="88" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">• patient_id (FK)</text>
    <text x="20" y="116" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• file_path: text</text>
    <text x="20" y="144" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• slice_index: int</text>
    <text x="20" y="172" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• true_label: varchar (glioma/men/pit)</text>
    <text x="20" y="200" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• split_tag: varchar (train/val/test)</text>
  </g>
  
  <!-- Relationship 2: Contains (1 : 1) to Tumour Mask -->
  <g transform="translate(970, 180)">
    <polygon points="50,0 100,35 50,70 0,35" fill="#f0fdf4" stroke="#10b981" stroke-width="2" />
    <text x="50" y="40" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Contains</text>
  </g>
  <line x1="880" y1="215" x2="970" y2="215" stroke="#334155" stroke-width="2" />
  <text x="900" y="205" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  <line x1="1070" y1="215" x2="1140" y2="215" stroke="#334155" stroke-width="2" />
  <text x="1120" y="205" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  
  <!-- Entity 3: TUMOUR_MASK -->
  <g transform="translate(1140, 125)">
    <rect width="220" height="185" rx="8" fill="#ffffff" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="35" rx="8" fill="#059669" />
    <rect y="25" width="220" height="10" fill="#059669" />
    <text x="110" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">TUMOUR_MASK</text>
    
    <text x="15" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• mask_id (PK)</text>
    <line x1="15" y1="64" x2="100" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="15" y="88" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">• image_id (FK)</text>
    <text x="15" y="116" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• file_path: text</text>
    <text x="15" y="144" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• annotator_id: varchar</text>
    <text x="15" y="170" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• mask_area_px: int</text>
  </g>
  
  <!-- Relationship 3: Evaluated_In (1 : N) from MRI_IMAGE to PREDICTION -->
  <g transform="translate(690, 390)">
    <polygon points="50,0 100,35 50,70 0,35" fill="#eff6ff" stroke="#3b82f6" stroke-width="2" />
    <text x="50" y="40" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Evaluated_In</text>
  </g>
  <line x1="740" y1="330" x2="740" y2="390" stroke="#334155" stroke-width="2" />
  <text x="755" y="360" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  <line x1="740" y1="460" x2="740" y2="520" stroke="#334155" stroke-width="2" />
  <text x="755" y="495" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">N</text>
  
  <!-- Entity 4: PREDICTION -->
  <g transform="translate(600, 520)">
    <rect width="280" height="210" rx="8" fill="#ffffff" stroke="#8b5cf6" stroke-width="2" filter="url(#shadow)" />
    <rect width="280" height="35" rx="8" fill="#7c3aed" />
    <rect y="25" width="280" height="10" fill="#7c3aed" />
    <text x="140" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">PREDICTION</text>
    
    <text x="20" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• prediction_id (PK)</text>
    <line x1="20" y1="64" x2="135" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="20" y="88" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">• image_id (FK)</text>
    <text x="20" y="116" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• predicted_label: varchar</text>
    <text x="20" y="144" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• confidence: float (0.0 - 1.0)</text>
    <text x="20" y="172" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• model_version: varchar</text>
    <text x="20" y="198" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• inference_time_ms: float</text>
  </g>
  
  <!-- Relationship 4: Generates (1 : N) to EXPLANATION -->
  <g transform="translate(410, 580)">
    <polygon points="50,0 100,35 50,70 0,35" fill="#fdf4ff" stroke="#c026d3" stroke-width="2" />
    <text x="50" y="40" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Generates</text>
  </g>
  <line x1="600" y1="615" x2="510" y2="615" stroke="#334155" stroke-width="2" />
  <text x="575" y="605" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  <line x1="410" y1="615" x2="320" y2="615" stroke="#334155" stroke-width="2" />
  <text x="340" y="605" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">N</text>
  
  <!-- Entity 5: EXPLANATION -->
  <g transform="translate(60, 520)">
    <rect width="260" height="210" rx="8" fill="#ffffff" stroke="#c026d3" stroke-width="2" filter="url(#shadow)" />
    <rect width="260" height="35" rx="8" fill="#a21caf" />
    <rect y="25" width="260" height="10" fill="#a21caf" />
    <text x="130" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">EXPLANATION</text>
    
    <text x="20" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• explanation_id (PK)</text>
    <line x1="20" y1="64" x2="145" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="20" y="88" fill="#701a75" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">• prediction_id (FK)</text>
    <text x="20" y="116" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• method: varchar (Grad-CAM)</text>
    <text x="20" y="144" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• map_path: text</text>
    <text x="20" y="172" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• is_perturbed: boolean</text>
    <text x="20" y="198" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• perturbation_type: varchar</text>
  </g>
  
  <!-- Relationship 5: Evaluated_By (1 : 1) to EVIDENCE_SCORE -->
  <g transform="translate(960, 580)">
    <polygon points="50,0 100,35 50,70 0,35" fill="#fdf2f8" stroke="#db2777" stroke-width="2" />
    <text x="50" y="40" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Produces</text>
  </g>
  <line x1="880" y1="615" x2="960" y2="615" stroke="#334155" stroke-width="2" />
  <text x="900" y="605" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  <line x1="1060" y1="615" x2="1140" y2="615" stroke="#334155" stroke-width="2" />
  <text x="1120" y="605" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">1</text>
  
  <!-- Entity 6: EVIDENCE_SCORE -->
  <g transform="translate(1140, 480)">
    <rect width="220" height="280" rx="8" fill="#ffffff" stroke="#db2777" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="35" rx="8" fill="#be185d" />
    <rect y="25" width="220" height="10" fill="#be185d" />
    <text x="110" y="24" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">EVIDENCE_SCORE</text>
    
    <text x="15" y="60" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• score_id (PK)</text>
    <line x1="15" y1="64" x2="105" y2="64" stroke="#0f172a" stroke-width="1" />
    <text x="15" y="85" fill="#be185d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-style="italic">• prediction_id (FK)</text>
    <text x="15" y="108" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• iou: float</text>
    <text x="15" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• dice: float</text>
    <text x="15" y="152" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• pointing_game: boolean</text>
    <text x="15" y="174" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• energy_in_mask: float</text>
    <text x="15" y="196" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• stability_ssim: float</text>
    <text x="15" y="218" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• stability_rank: float</text>
    <text x="15" y="242" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• reliability_score: float</text>
    <text x="15" y="266" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold">• flag: varchar (Reliable/Unreliable)</text>
  </g>
  
  <!-- Legend / Notes -->
  <g transform="translate(60, 780)">
    <rect width="1300" height="120" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1" />
    <text x="25" y="28" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold">Entity-Relationship Rules &amp; Cardinalities:</text>
    <text x="25" y="55" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• One Patient has multiple MRI Scans (1 : N) with patient-wise stratification to eliminate cross-fold patient leakage.</text>
    <text x="25" y="78" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• Each MRI Image has exactly one expert-annotated tumour mask (1 : 1) and can be evaluated across multiple model checkpoints (1 : N).</text>
    <text x="25" y="102" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5">• Each Prediction produces 1 unperturbed Grad-CAM explanation plus K perturbed explanations (1 : N), synthesizing into 1 Evidence Score tuple (1 : 1).</text>
  </g>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_5_er_diagram")

# -------------------------------------------------------------
# 8. Figure 4.6: UML Use Case Diagram
# -------------------------------------------------------------
def build_fig_4_6():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 900" width="1350" height="900">
  <rect width="1350" height="900" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="20" width="1250" height="55" rx="8" fill="#1e293b" />
  <text x="675" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.6: UML Use Case Diagram</text>
  
  <!-- System Boundary -->
  <rect x="280" y="95" width="790" height="740" rx="12" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
  <rect x="280" y="95" width="790" height="40" rx="12" fill="#2563eb" />
  <rect x="280" y="125" width="790" height="10" fill="#2563eb" />
  <text x="675" y="122" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">System Boundary: Evidence-Aware Brain Tumour Classification System</text>
  
  <!-- Actor 1: Researcher (Left) -->
  <g transform="translate(60, 320)">
    <!-- Stick Figure -->
    <circle cx="70" cy="40" r="22" fill="#eff6ff" stroke="#1e40af" stroke-width="2.5" />
    <line x1="70" y1="62" x2="70" y2="130" stroke="#1e40af" stroke-width="2.5" />
    <line x1="25" y1="85" x2="115" y2="85" stroke="#1e40af" stroke-width="2.5" />
    <line x1="70" y1="130" x2="35" y2="190" stroke="#1e40af" stroke-width="2.5" />
    <line x1="70" y1="130" x2="105" y2="190" stroke="#1e40af" stroke-width="2.5" />
    <text x="70" y="225" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Researcher /</text>
    <text x="70" y="245" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Student</text>
  </g>
  
  <!-- Actor 2: Radiologist / Reviewer (Right) -->
  <g transform="translate(1150, 320)">
    <!-- Stick Figure -->
    <circle cx="70" cy="40" r="22" fill="#fdf4ff" stroke="#701a75" stroke-width="2.5" />
    <line x1="70" y1="62" x2="70" y2="130" stroke="#701a75" stroke-width="2.5" />
    <line x1="25" y1="85" x2="115" y2="85" stroke="#701a75" stroke-width="2.5" />
    <line x1="70" y1="130" x2="35" y2="190" stroke="#701a75" stroke-width="2.5" />
    <line x1="70" y1="130" x2="105" y2="190" stroke="#701a75" stroke-width="2.5" />
    <text x="70" y="225" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Radiologist /</text>
    <text x="70" y="245" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">Reviewer</text>
  </g>
  
  <!-- Use Cases Inside Boundary -->
  <!-- UC1: Load dataset and masks (Researcher) -->
  <g transform="translate(340, 160)">
    <ellipse cx="110" cy="30" rx="100" ry="28" fill="#ffffff" stroke="#0284c7" stroke-width="1.8" filter="url(#shadow)" />
    <text x="110" y="35" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Load Dataset &amp; Masks</text>
  </g>
  
  <!-- UC2: Train classifier (Researcher) -->
  <g transform="translate(340, 240)">
    <ellipse cx="110" cy="30" rx="100" ry="28" fill="#ffffff" stroke="#0284c7" stroke-width="1.8" filter="url(#shadow)" />
    <text x="110" y="35" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Train Classifier</text>
  </g>
  
  <!-- UC3: Evaluate localization and stability (Researcher) -->
  <g transform="translate(330, 325)">
    <ellipse cx="120" cy="35" rx="115" ry="32" fill="#eff6ff" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <text x="120" y="33" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Evaluate Localization</text>
    <text x="120" y="50" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">&amp; Stability</text>
  </g>
  
  <!-- UC4: Compare with confidence baseline (Researcher) -->
  <g transform="translate(340, 420)">
    <ellipse cx="110" cy="32" rx="105" ry="30" fill="#ffffff" stroke="#0284c7" stroke-width="1.8" filter="url(#shadow)" />
    <text x="110" y="30" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Compare with Confidence</text>
    <text x="110" y="47" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Baseline (AUROC/AUPRC)</text>
  </g>
  
  <!-- INCLUDED USE CASES IN CENTER -->
  <!-- UC_INC1: Generate visual explanation -->
  <g transform="translate(620, 275)">
    <ellipse cx="120" cy="32" rx="115" ry="30" fill="#f0fdf4" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,2" filter="url(#shadow)" />
    <text x="120" y="28" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;include&gt;&gt;</text>
    <text x="120" y="46" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Generate Visual Explanation</text>
  </g>
  
  <!-- UC_INC2: Perturb input and re-explain -->
  <g transform="translate(620, 365)">
    <ellipse cx="120" cy="32" rx="115" ry="30" fill="#f0fdf4" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,2" filter="url(#shadow)" />
    <text x="120" y="28" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;include&gt;&gt;</text>
    <text x="120" y="46" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Perturb Input &amp; Re-explain</text>
  </g>
  
  <!-- UC_INC3: Pre-process MRI -->
  <g transform="translate(620, 520)">
    <ellipse cx="120" cy="30" rx="110" ry="28" fill="#f0fdf4" stroke="#10b981" stroke-width="1.8" stroke-dasharray="4,2" filter="url(#shadow)" />
    <text x="120" y="26" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;include&gt;&gt;</text>
    <text x="120" y="44" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Pre-process MRI</text>
  </g>
  
  <!-- UC_INC4: Estimate reliability score -->
  <g transform="translate(620, 680)">
    <ellipse cx="120" cy="32" rx="115" ry="30" fill="#fdf2f8" stroke="#db2777" stroke-width="1.8" stroke-dasharray="4,2" filter="url(#shadow)" />
    <text x="120" y="28" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="11.5" font-weight="bold" text-anchor="middle">&lt;&lt;include&gt;&gt;</text>
    <text x="120" y="46" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Estimate Reliability Score</text>
  </g>
  
  <!-- RADIOLOGIST / REVIEWER USE CASES (Right) -->
  <!-- UC5: Upload MRI -->
  <g transform="translate(820, 440)">
    <ellipse cx="105" cy="30" rx="100" ry="28" fill="#ffffff" stroke="#7c3aed" stroke-width="1.8" filter="url(#shadow)" />
    <text x="105" y="35" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Upload MRI Scan</text>
  </g>
  
  <!-- UC6: Classify tumour -->
  <g transform="translate(820, 520)">
    <ellipse cx="105" cy="30" rx="100" ry="28" fill="#ffffff" stroke="#7c3aed" stroke-width="1.8" filter="url(#shadow)" />
    <text x="105" y="35" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Classify Tumour</text>
  </g>
  
  <!-- UC7: View explanation overlay -->
  <g transform="translate(820, 600)">
    <ellipse cx="105" cy="30" rx="100" ry="28" fill="#ffffff" stroke="#7c3aed" stroke-width="1.8" filter="url(#shadow)" />
    <text x="105" y="35" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">View Explanation Overlay</text>
  </g>
  
  <!-- UC8: View reliability flag -->
  <g transform="translate(820, 680)">
    <ellipse cx="105" cy="30" rx="100" ry="28" fill="#fdf2f8" stroke="#db2777" stroke-width="2" filter="url(#shadow)" />
    <text x="105" y="35" fill="#9d174d" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">View Reliability Flag</text>
  </g>
  
  <!-- UC9: Download Report -->
  <g transform="translate(820, 760)">
    <ellipse cx="105" cy="28" rx="100" ry="26" fill="#ffffff" stroke="#7c3aed" stroke-width="1.8" filter="url(#shadow)" />
    <text x="105" y="33" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600" text-anchor="middle">Export Analysis Report</text>
  </g>
  
  <!-- ACTOR ASSOCIATIONS -->
  <!-- Researcher associations -->
  <line x1="175" y1="360" x2="340" y2="190" stroke="#334155" stroke-width="1.8" />
  <line x1="175" y1="375" x2="340" y2="270" stroke="#334155" stroke-width="1.8" />
  <line x1="175" y1="390" x2="330" y2="360" stroke="#334155" stroke-width="2" />
  <line x1="175" y1="410" x2="340" y2="450" stroke="#334155" stroke-width="1.8" />
  
  <!-- Radiologist associations -->
  <line x1="1150" y1="380" x2="1025" y2="470" stroke="#334155" stroke-width="1.8" />
  <line x1="1150" y1="395" x2="1025" y2="550" stroke="#334155" stroke-width="1.8" />
  <line x1="1150" y1="415" x2="1025" y2="630" stroke="#334155" stroke-width="1.8" />
  <line x1="1150" y1="435" x2="1025" y2="710" stroke="#334155" stroke-width="2" />
  <line x1="1150" y1="455" x2="1025" y2="790" stroke="#334155" stroke-width="1.8" />
  
  <!-- INCLUDE DEPENDENCY ARROWS (dashed with arrow) -->
  <!-- UC3 -> Generate explanation -->
  <path d="M 540 335 L 625 320" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#include-arrow)" />
  <text x="580" y="315" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-style="italic">«include»</text>
  
  <!-- UC3 -> Perturb input -->
  <path d="M 545 370 L 625 385" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#include-arrow)" />
  <text x="575" y="392" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-style="italic">«include»</text>
  
  <!-- UC6 (Classify) -> Pre-process MRI -->
  <path d="M 820 550 L 740 550" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#include-arrow)" />
  <text x="775" y="542" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-style="italic">«include»</text>
  
  <!-- UC7 (View Explanation) -> Generate explanation -->
  <path d="M 845 605 C 780 570, 750 420, 740 340" fill="none" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#include-arrow)" />
  <text x="785" y="470" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-style="italic">«include»</text>
  
  <!-- UC8 (View Flag) -> Estimate reliability score -->
  <path d="M 820 710 L 740 710" stroke="#475569" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#include-arrow)" />
  <text x="775" y="702" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-style="italic">«include»</text>
  
  <!-- Footer Note -->
  <text x="675" y="865" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Standard UML Use Case model demonstrating primary clinical and research actors, operational boundaries, and modular inclusions.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_6_use_case_diagram")

if __name__ == "__main__":
    print("Building Part 2 diagrams...")
    build_fig_4_3()
    build_fig_4_4()
    build_fig_4_5()
    build_fig_4_6()
    print("Part 2 complete!")
