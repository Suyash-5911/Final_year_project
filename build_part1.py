import os
from render_utils import render_svg_to_png_and_jpg

# Common SVG styles and defs
SVG_DEFS = """
<defs>
  <filter id="shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
    <feDropShadow dx="2" dy="3" stdDeviation="3" flood-opacity="0.12" flood-color="#0f172a" />
  </filter>
  <filter id="glow" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="0" stdDeviation="4" flood-opacity="0.25" flood-color="#2563eb" />
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1 L 10 5 L 0 9 z" fill="#334155" />
  </marker>
  <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1 L 10 5 L 0 9 z" fill="#2563eb" />
  </marker>
  <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1 L 10 5 L 0 9 z" fill="#059669" />
  </marker>
  <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1 L 10 5 L 0 9 z" fill="#dc2626" />
  </marker>
  <marker id="include-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 2 L 8 5 L 0 8" fill="none" stroke="#475569" stroke-width="1.5" />
  </marker>
</defs>
"""

# -------------------------------------------------------------
# 1. Figure 3.1: Iterative and Incremental Model
# -------------------------------------------------------------
def build_fig_3_1():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1300 850" width="1300" height="850">
  <rect width="1300" height="850" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Title Header -->
  <rect x="50" y="25" width="1200" height="55" rx="8" fill="#1e293b" />
  <text x="650" y="60" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 3.1: Iterative and Incremental Development Model</text>
  
  <!-- Increments container -->
  <!-- Increment 1 -->
  <g transform="translate(60, 110)">
    <rect width="220" height="580" rx="10" fill="#f8fafc" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="42" rx="10" fill="#2563eb" />
    <rect y="30" width="220" height="12" fill="#2563eb" />
    <text x="110" y="27" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Increment 1</text>
    <text x="110" y="62" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Dataset &amp; Classifier</text>
    
    <!-- Iteration steps inside Increment 1 -->
    <rect x="15" y="85" width="190" height="60" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
    <text x="110" y="110" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">1. Requirement &amp; Design</text>
    <text x="110" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Dataset splits &amp; CNN choice</text>
    
    <path d="M 110 145 L 110 165" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="165" width="190" height="60" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
    <text x="110" y="190" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">2. Implementation</text>
    <text x="110" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Preprocessing &amp; CNN training</text>
    
    <path d="M 110 225 L 110 245" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="245" width="190" height="60" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
    <text x="110" y="270" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">3. Testing &amp; Verification</text>
    <text x="110" y="290" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Accuracy, F1, Loss check</text>
    
    <path d="M 110 305 L 110 325" stroke="#3b82f6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="325" width="190" height="60" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
    <text x="110" y="350" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">4. Guide Review</text>
    <text x="110" y="370" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Baseline validation</text>
    
    <!-- Iterative Loop arrow inside increment -->
    <path d="M 15 355 C -5 355, -5 115, 15 115" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,3" marker-end="url(#arrow)" />
    <text x="-8" y="240" fill="#d97706" font-family="'Segoe UI', Arial, sans-serif" font-size="10" font-weight="bold" transform="rotate(-90 -8 240)" text-anchor="middle">Iterative Refinement</text>
    
    <rect x="15" y="415" width="190" height="145" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
    <text x="110" y="435" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Deliverable 1</text>
    <text x="25" y="460" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Patient-wise dataset split</text>
    <text x="25" y="485" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Trained CNN backbone</text>
    <text x="25" y="510" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Softmax prediction &amp; conf</text>
    <text x="25" y="535" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Baseline classification metrics</text>
  </g>
  
  <!-- Increment 2 -->
  <g transform="translate(305, 110)">
    <rect width="220" height="580" rx="10" fill="#f8fafc" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="42" rx="10" fill="#0284c7" />
    <rect y="30" width="220" height="12" fill="#0284c7" />
    <text x="110" y="27" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Increment 2</text>
    <text x="110" y="62" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Explanation &amp; Localization</text>
    
    <rect x="15" y="85" width="190" height="60" rx="6" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.2" />
    <text x="110" y="110" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">1. Requirement &amp; Design</text>
    <text x="110" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Grad-CAM &amp; Mask metrics</text>
    
    <path d="M 110 145 L 110 165" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="165" width="190" height="60" rx="6" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.2" />
    <text x="110" y="190" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">2. Implementation</text>
    <text x="110" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Heatmap gen &amp; Binarization</text>
    
    <path d="M 110 225 L 110 245" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="245" width="190" height="60" rx="6" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.2" />
    <text x="110" y="270" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">3. Testing &amp; Verification</text>
    <text x="110" y="290" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">IoU, Dice, Pointing Game</text>
    
    <path d="M 110 305 L 110 325" stroke="#0ea5e9" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="325" width="190" height="60" rx="6" fill="#f0f9ff" stroke="#bae6fd" stroke-width="1.2" />
    <text x="110" y="350" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">4. Guide Review</text>
    <text x="110" y="370" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Spatial alignment checks</text>
    
    <rect x="15" y="415" width="190" height="145" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
    <text x="110" y="435" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Deliverable 2</text>
    <text x="25" y="460" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Grad-CAM heatmaps</text>
    <text x="25" y="485" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• IoU &amp; Dice computations</text>
    <text x="25" y="510" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Pointing game evaluation</text>
    <text x="25" y="535" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Energy inside tumour mask</text>
  </g>
  
  <!-- Increment 3 -->
  <g transform="translate(550, 110)">
    <rect width="220" height="580" rx="10" fill="#f8fafc" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="42" rx="10" fill="#059669" />
    <rect y="30" width="220" height="12" fill="#059669" />
    <text x="110" y="27" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Increment 3</text>
    <text x="110" y="62" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Perturbation &amp; Stability</text>
    
    <rect x="15" y="85" width="190" height="60" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.2" />
    <text x="110" y="110" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">1. Requirement &amp; Design</text>
    <text x="110" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Noise, Blur, Shift specs</text>
    
    <path d="M 110 145 L 110 165" stroke="#10b981" stroke-width="1.5" marker-end="url(#arrow-green)" />
    
    <rect x="15" y="165" width="190" height="60" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.2" />
    <text x="110" y="190" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">2. Implementation</text>
    <text x="110" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Perturbation &amp; Re-explain</text>
    
    <path d="M 110 225 L 110 245" stroke="#10b981" stroke-width="1.5" marker-end="url(#arrow-green)" />
    
    <rect x="15" y="245" width="190" height="60" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.2" />
    <text x="110" y="270" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">3. Testing &amp; Verification</text>
    <text x="110" y="290" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">SSIM &amp; Rank Correlation</text>
    
    <path d="M 110 305 L 110 325" stroke="#10b981" stroke-width="1.5" marker-end="url(#arrow-green)" />
    
    <rect x="15" y="325" width="190" height="60" rx="6" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1.2" />
    <text x="110" y="350" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">4. Guide Review</text>
    <text x="110" y="370" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Robustness sanity check</text>
    
    <rect x="15" y="415" width="190" height="145" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
    <text x="110" y="435" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Deliverable 3</text>
    <text x="25" y="460" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Perturbation engine</text>
    <text x="25" y="485" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Multi-pass heatmaps</text>
    <text x="25" y="510" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• SSIM stability metric</text>
    <text x="25" y="535" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Rank correlation indices</text>
  </g>
  
  <!-- Increment 4 -->
  <g transform="translate(795, 110)">
    <rect width="220" height="580" rx="10" fill="#f8fafc" stroke="#8b5cf6" stroke-width="2" filter="url(#shadow)" />
    <rect width="220" height="42" rx="10" fill="#7c3aed" />
    <rect y="30" width="220" height="12" fill="#7c3aed" />
    <text x="110" y="27" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Increment 4</text>
    <text x="110" y="62" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Reliability Assessment</text>
    
    <rect x="15" y="85" width="190" height="60" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.2" />
    <text x="110" y="110" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">1. Requirement &amp; Design</text>
    <text x="110" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Fusion logic &amp; Baselines</text>
    
    <path d="M 110 145 L 110 165" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="165" width="190" height="60" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.2" />
    <text x="110" y="190" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">2. Implementation</text>
    <text x="110" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Reliability scoring &amp; Flag</text>
    
    <path d="M 110 225 L 110 245" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="245" width="190" height="60" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.2" />
    <text x="110" y="270" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">3. Testing &amp; Verification</text>
    <text x="110" y="290" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">AUROC / AUPRC evaluation</text>
    
    <path d="M 110 305 L 110 325" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="325" width="190" height="60" rx="6" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1.2" />
    <text x="110" y="350" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">4. Guide Review</text>
    <text x="110" y="370" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Hypothesis validation</text>
    
    <rect x="15" y="415" width="190" height="145" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
    <text x="110" y="435" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Deliverable 4</text>
    <text x="25" y="460" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Composite reliability score</text>
    <text x="25" y="485" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Decision threshold flagging</text>
    <text x="25" y="510" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• AUROC / AUPRC gains</text>
    <text x="25" y="535" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Risk-coverage analysis</text>
  </g>
  
  <!-- Increment 5 -->
  <g transform="translate(1040, 110)">
    <rect width="200" height="580" rx="10" fill="#f8fafc" stroke="#f59e0b" stroke-width="2" filter="url(#shadow)" />
    <rect width="200" height="42" rx="10" fill="#d97706" />
    <rect y="30" width="200" height="12" fill="#d97706" />
    <text x="100" y="27" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Increment 5</text>
    <text x="100" y="62" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Dashboard &amp; Release</text>
    
    <rect x="15" y="85" width="170" height="60" rx="6" fill="#fffbeb" stroke="#fde68a" stroke-width="1.2" />
    <text x="100" y="110" fill="#b45309" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">1. Requirement &amp; Design</text>
    <text x="100" y="130" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Streamlit UI &amp; UX wireframe</text>
    
    <path d="M 100 145 L 100 165" stroke="#f59e0b" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="165" width="170" height="60" rx="6" fill="#fffbeb" stroke="#fde68a" stroke-width="1.2" />
    <text x="100" y="190" fill="#b45309" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">2. Implementation</text>
    <text x="100" y="210" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Full pipeline integration</text>
    
    <path d="M 100 225 L 100 245" stroke="#f59e0b" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="245" width="170" height="60" rx="6" fill="#fffbeb" stroke="#fde68a" stroke-width="1.2" />
    <text x="100" y="270" fill="#b45309" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">3. Testing &amp; Verification</text>
    <text x="100" y="290" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">User acceptance testing</text>
    
    <path d="M 100 305 L 100 325" stroke="#f59e0b" stroke-width="1.5" marker-end="url(#arrow-blue)" />
    
    <rect x="15" y="325" width="170" height="60" rx="6" fill="#fffbeb" stroke="#fde68a" stroke-width="1.2" />
    <text x="100" y="350" fill="#b45309" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">4. Guide Review</text>
    <text x="100" y="370" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Final demonstration</text>
    
    <rect x="15" y="415" width="170" height="145" rx="6" fill="#ffffff" stroke="#cbd5e1" stroke-width="1" />
    <text x="100" y="435" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Final Deliverable</text>
    <text x="25" y="460" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Streamlit web app</text>
    <text x="25" y="485" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Visual evidence viewer</text>
    <text x="25" y="510" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Reliability flag indicator</text>
    <text x="25" y="535" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="11">• Exportable PDF report</text>
  </g>
  
  <!-- Inter-increment flow connectors -->
  <path d="M 280 250 L 305 250" stroke="#0284c7" stroke-width="3" marker-end="url(#arrow-blue)" />
  <path d="M 525 250 L 550 250" stroke="#059669" stroke-width="3" marker-end="url(#arrow-green)" />
  <path d="M 770 250 L 795 250" stroke="#7c3aed" stroke-width="3" marker-end="url(#arrow-blue)" />
  <path d="M 1015 250 L 1040 250" stroke="#d97706" stroke-width="3" marker-end="url(#arrow-blue)" />
  
  <!-- Feedback Loops at bottom -->
  <path d="M 1140 705 C 1140 760, 200 760, 200 705" fill="none" stroke="#ef4444" stroke-width="2.5" stroke-dasharray="6,4" marker-end="url(#arrow-red)" />
  <rect x="530" y="742" width="240" height="30" rx="5" fill="#fee2e2" stroke="#fca5a5" stroke-width="1" />
  <text x="650" y="762" fill="#991b1b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">Feedback &amp; Tuning across Increments</text>
  
  <!-- Footer Note -->
  <text x="650" y="815" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Iterative execution within increments ensures agile prototyping; findings from evaluation feed back to optimize earlier stages.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_3_1_iterative_model")

# -------------------------------------------------------------
# 2. Figure 3.2: Timeline Chart (Gantt Chart)
# -------------------------------------------------------------
def build_fig_3_2():
    # Table 3.3 Activities:
    # 1. Literature survey and problem finalization (Aug 2026 – Sep 2026)
    # 2. Requirement analysis and system design (Oct 2026)
    # 3. Dataset preparation and pre-processing (Nov 2026)
    # 4. Classifier training and evaluation (Nov 2026 – Dec 2026)
    # 5. Explanation generation and localization evaluation (Dec 2026 – Jan 2027)
    # 6. Perturbation and stability evaluation (Jan 2027 – Feb 2027)
    # 7. Reliability assessment and baseline comparison (Feb 2027 – Mar 2027)
    # 8. Dashboard development and integration (Mar 2027)
    # 9. Testing, documentation and final report (Apr 2027)

    months = ["Aug 2026", "Sep 2026", "Oct 2026", "Nov 2026", "Dec 2026", "Jan 2027", "Feb 2027", "Mar 2027", "Apr 2027"]
    
    tasks = [
        ("1. Literature survey & problem finalization", 0, 1.95, "#3b82f6", "Aug - Sep 2026"),
        ("2. Requirement analysis & system design", 2, 2.95, "#0ea5e9", "Oct 2026"),
        ("3. Dataset preparation & pre-processing", 3, 3.95, "#14b8a6", "Nov 2026"),
        ("4. Classifier training & evaluation", 3.2, 4.95, "#10b981", "Nov - Dec 2026"),
        ("5. Explanation generation & localization eval", 4.2, 5.95, "#8b5cf6", "Dec 2026 - Jan 2027"),
        ("6. Perturbation & stability evaluation", 5.2, 6.95, "#a855f7", "Jan - Feb 2027"),
        ("7. Reliability assessment & baseline comparison", 6.2, 7.95, "#f59e0b", "Feb - Mar 2027"),
        ("8. Dashboard development & integration", 7.2, 7.95, "#f97316", "Mar 2027"),
        ("9. Testing, documentation & final report", 8, 8.95, "#ef4444", "Apr 2027")
    ]

    col_w = 95
    start_x = 420
    
    month_headers = ""
    grid_lines = ""
    for i, m in enumerate(months):
        x = start_x + i * col_w
        month_headers += f"""
        <rect x="{x}" y="100" width="{col_w}" height="40" fill="{'#f1f5f9' if i % 2 == 0 else '#e2e8f0'}" stroke="#cbd5e1" stroke-width="0.8" />
        <text x="{x + col_w/2}" y="125" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="bold" text-anchor="middle">{m}</text>
        """
        grid_lines += f"""
        <line x1="{x}" y1="140" x2="{x}" y2="640" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3" />
        """
    # final right line
    grid_lines += f"""<line x1="{start_x + len(months)*col_w}" y1="140" x2="{start_x + len(months)*col_w}" y2="640" stroke="#cbd5e1" stroke-width="1.5" />"""

    task_rows = ""
    for idx, (name, s, e, color, dur_label) in enumerate(tasks):
        y = 150 + idx * 52
        bx = start_x + s * col_w
        bw = (e - s + 0.05) * col_w
        
        # alternating background
        bg_col = "#ffffff" if idx % 2 == 0 else "#f8fafc"
        task_rows += f"""
        <rect x="40" y="{y - 10}" width="{start_x + len(months)*col_w - 40}" height="48" fill="{bg_col}" />
        <text x="55" y="{y + 20}" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="600">{name}</text>
        <rect x="{bx}" y="{y}" width="{bw}" height="28" rx="6" fill="{color}" filter="url(#shadow)" />
        <text x="{bx + bw/2}" y="{y + 18}" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="10.5" font-weight="bold" text-anchor="middle">{dur_label}</text>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1350 720" width="1350" height="720">
  <rect width="1350" height="720" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="40" y="25" width="1275" height="55" rx="8" fill="#1e293b" />
  <text x="677" y="60" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 3.2: Project Timeline (Gantt Chart - Academic Year 2026-2027)</text>
  
  <!-- Task Column Header -->
  <rect x="40" y="100" width="380" height="40" fill="#1e293b" stroke="#cbd5e1" stroke-width="0.8" />
  <text x="60" y="125" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold">Project Activity / Work Breakdown</text>
  
  <!-- Months Headers -->
  {month_headers}
  
  <!-- Background Grid -->
  {grid_lines}
  
  <!-- Task Rows and Gantt Bars -->
  {task_rows}
  
  <!-- Outer border -->
  <rect x="40" y="100" width="1235" height="540" fill="none" stroke="#94a3b8" stroke-width="1.5" />
  
  <!-- Milestone indicator / Footer -->
  <g transform="translate(40, 660)">
    <circle cx="20" cy="15" r="7" fill="#10b981" />
    <text x="35" y="19" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600">Model Training Milestones (Dec 2026)</text>
    
    <circle cx="340" cy="15" r="7" fill="#8b5cf6" />
    <text x="355" y="19" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600">XAI &amp; Stability Benchmarking (Jan-Feb 2027)</text>
    
    <circle cx="700" cy="15" r="7" fill="#f59e0b" />
    <text x="715" y="19" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600">Clinical Reliability Fusion &amp; Dashboard (Mar 2027)</text>
    
    <circle cx="1070" cy="15" r="7" fill="#ef4444" />
    <text x="1085" y="19" fill="#334155" font-family="'Segoe UI', Arial, sans-serif" font-size="12" font-weight="600">Final Defense (Apr 2027)</text>
  </g>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_3_2_timeline_chart")

# -------------------------------------------------------------
# 3. Figure 4.1: System Architecture
# -------------------------------------------------------------
def build_fig_4_1():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1300 950" width="1300" height="950">
  <rect width="1300" height="950" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Title Header -->
  <rect x="50" y="20" width="1200" height="55" rx="8" fill="#1e293b" />
  <text x="650" y="55" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.1: Evidence-Aware Brain Tumour Classification System Architecture</text>
  
  <!-- Layer 1: Presentation Layer -->
  <g transform="translate(60, 95)">
    <rect width="1180" height="150" rx="10" fill="#f8fafc" stroke="#3b82f6" stroke-width="2" filter="url(#shadow)" />
    <rect width="1180" height="34" rx="10" fill="#2563eb" />
    <rect y="24" width="1180" height="10" fill="#2563eb" />
    <text x="30" y="23" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold">1. PRESENTATION LAYER (Streamlit Interactive Web Application)</text>
    
    <g transform="translate(25, 45)">
      <!-- UI Modules -->
      <rect x="0" y="8" width="205" height="80" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
      <text x="102" y="38" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">MRI &amp; Mask Uploader</text>
      <text x="102" y="60" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">File drag-drop, DICOM/PNG</text>
      
      <rect x="230" y="8" width="215" height="80" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
      <text x="337" y="38" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Prediction &amp; Conf View</text>
      <text x="337" y="60" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Class probabilities &amp; bar chart</text>
      
      <rect x="470" y="8" width="220" height="80" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
      <text x="580" y="38" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Heatmap Overlay Display</text>
      <text x="580" y="60" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Grad-CAM overlaid on scan</text>
      
      <rect x="715" y="8" width="205" height="80" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
      <text x="817" y="38" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Mask Comparison View</text>
      <text x="817" y="60" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Ground truth vs Saliency</text>
      
      <rect x="945" y="8" width="185" height="80" rx="6" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.2" />
      <text x="1037" y="38" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Reliability &amp; Report</text>
      <text x="1037" y="60" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Status Flag &amp; PDF Export</text>
    </g>
  </g>
  
  <!-- Inter-layer Arrow 1-2 -->
  <path d="M 650 245 L 650 275" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <path d="M 660 275 L 660 245" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="735" y="263" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600">User Requests / Visualization Data</text>
  
  <!-- Layer 2: Application Layer -->
  <g transform="translate(60, 275)">
    <rect width="1180" height="110" rx="10" fill="#f8fafc" stroke="#0ea5e9" stroke-width="2" filter="url(#shadow)" />
    <rect width="1180" height="34" rx="10" fill="#0284c7" />
    <rect y="24" width="1180" height="10" fill="#0284c7" />
    <text x="30" y="23" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold">2. APPLICATION LAYER (Pipeline Controller &amp; Execution Orchestrator)</text>
    
    <g transform="translate(45, 45)">
      <rect x="0" y="5" width="250" height="50" rx="6" fill="#f0f9ff" stroke="#7dd3fc" stroke-width="1.2" />
      <text x="125" y="35" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Workflow Coordinator</text>
      
      <rect x="280" y="5" width="250" height="50" rx="6" fill="#f0f9ff" stroke="#7dd3fc" stroke-width="1.2" />
      <text x="405" y="35" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Configuration Manager</text>
      
      <rect x="560" y="5" width="250" height="50" rx="6" fill="#f0f9ff" stroke="#7dd3fc" stroke-width="1.2" />
      <text x="685" y="35" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Execution Controller</text>
      
      <rect x="840" y="5" width="250" height="50" rx="6" fill="#f0f9ff" stroke="#7dd3fc" stroke-width="1.2" />
      <text x="965" y="35" fill="#0369a1" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Logging &amp; Audit Service</text>
    </g>
  </g>
  
  <!-- Inter-layer Arrow 2-3 -->
  <path d="M 650 385 L 650 415" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <path d="M 660 415 L 660 385" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="735" y="403" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600">Module Dispatch / Evidence Scores</text>
  
  <!-- Layer 3: Analysis Layer -->
  <g transform="translate(60, 415)">
    <rect width="1180" height="260" rx="10" fill="#f8fafc" stroke="#10b981" stroke-width="2" filter="url(#shadow)" />
    <rect width="1180" height="34" rx="10" fill="#059669" />
    <rect y="24" width="1180" height="10" fill="#059669" />
    <text x="30" y="23" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold">3. ANALYSIS LAYER (Deep Learning &amp; Evidence Evaluation Engines)</text>
    
    <!-- Top Row of Analysis Modules -->
    <g transform="translate(25, 48)">
      <rect x="0" y="0" width="265" height="90" rx="6" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.2" />
      <text x="132" y="25" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Pre-processing Module</text>
      <text x="132" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Image Resizing &amp; Normalization</text>
      <text x="132" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Patient-wise Stratified Split</text>
      
      <rect x="290" y="0" width="265" height="90" rx="6" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.2" />
      <text x="422" y="25" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">CNN Classifier</text>
      <text x="422" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Backbone: ResNet50 / DenseNet121</text>
      <text x="422" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Output: Predicted Class &amp; Confidence</text>
      
      <rect x="580" y="0" width="265" height="90" rx="6" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.2" />
      <text x="712" y="25" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Explanation Generator</text>
      <text x="712" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Grad-CAM Activation Mapping</text>
      <text x="712" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Saliency Maps &amp; Heatmap Normalization</text>
      
      <rect x="870" y="0" width="260" height="90" rx="6" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.2" />
      <text x="1000" y="25" fill="#047857" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Perturbation Engine</text>
      <text x="1000" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Controlled Gaussian Noise &amp; Blur</text>
      <text x="1000" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Contrast Changes &amp; Small Shifts</text>
    </g>
    
    <!-- Bottom Row of Analysis Modules -->
    <g transform="translate(25, 150)">
      <rect x="60" y="0" width="310" height="95" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2" />
      <text x="215" y="25" fill="#15803d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Localization Evaluator</text>
      <text x="215" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Heatmap Binarization (Otsu/Threshold)</text>
      <text x="215" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• IoU, Dice Coefficient, Pointing Game</text>
      <text x="215" y="85" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Heatmap Energy inside Expert Mask</text>
      
      <rect x="430" y="0" width="310" height="95" rx="6" fill="#f0fdf4" stroke="#86efac" stroke-width="1.2" />
      <text x="585" y="25" fill="#15803d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Stability Evaluator</text>
      <text x="585" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Re-explain Perturbed MRI Slices</text>
      <text x="585" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Structural Similarity Index (SSIM)</text>
      <text x="585" y="85" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Spearman Rank Correlation (ρ)</text>
      
      <rect x="800" y="0" width="310" height="95" rx="6" fill="#fdf2f8" stroke="#f472b6" stroke-width="1.5" />
      <text x="955" y="25" fill="#be185d" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Reliability Analyzer (Core Innovation)</text>
      <text x="955" y="48" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Evidence Fusion (Conf + Loc + Stability)</text>
      <text x="955" y="68" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• Reliability Threshold Flagging (Reliable/Unreliable)</text>
      <text x="955" y="85" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">• AUROC, AUPRC &amp; Risk-Coverage vs Baseline</text>
    </g>
  </g>
  
  <!-- Inter-layer Arrow 3-4 -->
  <path d="M 650 675 L 650 705" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <path d="M 660 705 L 660 675" stroke="#334155" stroke-width="2.5" marker-end="url(#arrow)" />
  <text x="735" y="693" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="600">Model Weights / Scan Data / Metrics Storage</text>
  
  <!-- Layer 4: Data Layer -->
  <g transform="translate(60, 705)">
    <rect width="1180" height="185" rx="10" fill="#f8fafc" stroke="#8b5cf6" stroke-width="2" filter="url(#shadow)" />
    <rect width="1180" height="34" rx="10" fill="#7c3aed" />
    <rect y="24" width="1180" height="10" fill="#7c3aed" />
    <text x="30" y="23" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold">4. DATA LAYER (File-Based Storage &amp; Experiment Tracking)</text>
    
    <g transform="translate(35, 45)">
      <!-- Cylinder 1 -->
      <g transform="translate(0, 0)">
        <path d="M 0 25 C 0 10, 240 10, 240 25 L 240 100 C 240 115, 0 115, 0 100 Z" fill="#ede9fe" stroke="#a78bfa" stroke-width="1.5" />
        <ellipse cx="120" cy="25" rx="120" ry="15" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5" />
        <text x="120" y="55" fill="#5b21b6" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">MRI Scans &amp; Masks</text>
        <text x="120" y="75" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Cheng (Figshare) 3,064 slices</text>
        <text x="120" y="92" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">BraTS multi-modal MRI</text>
      </g>
      
      <!-- Cylinder 2 -->
      <g transform="translate(290, 0)">
        <path d="M 0 25 C 0 10, 240 10, 240 25 L 240 100 C 240 115, 0 115, 0 100 Z" fill="#ede9fe" stroke="#a78bfa" stroke-width="1.5" />
        <ellipse cx="120" cy="25" rx="120" ry="15" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5" />
        <text x="120" y="55" fill="#5b21b6" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Model Checkpoints</text>
        <text x="120" y="75" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">PyTorch .pt weights</text>
        <text x="120" y="92" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Best checkpoint state_dict</text>
      </g>
      
      <!-- Cylinder 3 -->
      <g transform="translate(580, 0)">
        <path d="M 0 25 C 0 10, 240 10, 240 25 L 240 100 C 240 115, 0 115, 0 100 Z" fill="#ede9fe" stroke="#a78bfa" stroke-width="1.5" />
        <ellipse cx="120" cy="25" rx="120" ry="15" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5" />
        <text x="120" y="55" fill="#5b21b6" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Heatmaps &amp; Maps</text>
        <text x="120" y="75" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Binarized / float32 heatmaps</text>
        <text x="120" y="92" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Perturbation cache arrays</text>
      </g>
      
      <!-- Cylinder 4 -->
      <g transform="translate(870, 0)">
        <path d="M 0 25 C 0 10, 240 10, 240 25 L 240 100 C 240 115, 0 115, 0 100 Z" fill="#ede9fe" stroke="#a78bfa" stroke-width="1.5" />
        <ellipse cx="120" cy="25" rx="120" ry="15" fill="#ddd6fe" stroke="#a78bfa" stroke-width="1.5" />
        <text x="120" y="55" fill="#5b21b6" font-family="'Segoe UI', Arial, sans-serif" font-size="13" font-weight="bold" text-anchor="middle">Logs &amp; Results DB</text>
        <text x="120" y="75" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">CSV logs &amp; SQLite database</text>
        <text x="120" y="92" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="11" text-anchor="middle">Evaluation summary metrics</text>
      </g>
    </g>
  </g>
  
  <!-- Footer Note -->
  <text x="650" y="925" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Layered architecture ensures complete separation of presentation, orchestration, deep learning evidence computation, and data persistence.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_1_system_architecture")

# -------------------------------------------------------------
# 4. Figure 4.2: DFD Level 0 (Context Diagram)
# -------------------------------------------------------------
def build_fig_4_2():
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1300 750" width="1300" height="750">
  <rect width="1300" height="750" fill="#ffffff" />
  {SVG_DEFS}
  
  <!-- Header -->
  <rect x="50" y="25" width="1200" height="55" rx="8" fill="#1e293b" />
  <text x="650" y="60" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="20" font-weight="bold" text-anchor="middle">Figure 4.2: Data Flow Diagram (DFD Level 0 - Context Level)</text>
  
  <!-- External Entity 1: Researcher / ML Student -->
  <g transform="translate(60, 220)">
    <rect width="240" height="260" rx="10" fill="#f8fafc" stroke="#2563eb" stroke-width="2" filter="url(#shadow)" />
    <rect width="240" height="40" rx="10" fill="#2563eb" />
    <rect y="30" width="240" height="10" fill="#2563eb" />
    <text x="120" y="26" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">&lt;&lt;External Entity&gt;&gt;</text>
    <text x="120" y="70" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Researcher / Student</text>
    <line x1="20" y1="90" x2="220" y2="90" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="30" y="125" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Configures experiments</text>
    <text x="30" y="155" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Supplies MRI training dataset</text>
    <text x="30" y="185" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Provides expert masks</text>
    <text x="30" y="215" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Inspects model benchmarks</text>
  </g>
  
  <!-- Central Process 0.0 -->
  <g transform="translate(480, 180)">
    <circle cx="170" cy="170" r="165" fill="#f0fdf4" stroke="#059669" stroke-width="3" filter="url(#shadow)" />
    <circle cx="170" cy="170" r="152" fill="none" stroke="#10b981" stroke-width="1.5" stroke-dasharray="5,4" />
    
    <rect x="70" y="65" width="200" height="32" rx="6" fill="#059669" />
    <text x="170" y="87" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="14" font-weight="bold" text-anchor="middle">Process 0.0</text>
    
    <text x="170" y="145" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="17" font-weight="bold" text-anchor="middle">Evidence-Aware</text>
    <text x="170" y="175" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="17" font-weight="bold" text-anchor="middle">Brain Tumour Classification</text>
    <text x="170" y="205" fill="#0f172a" font-family="'Segoe UI', Arial, sans-serif" font-size="17" font-weight="bold" text-anchor="middle">System</text>
    
    <line x1="85" y1="225" x2="255" y2="225" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="170" y="250" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Classification + Grad-CAM</text>
    <text x="170" y="272" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">+ Evidence &amp; Stability Fusion</text>
  </g>
  
  <!-- External Entity 2: Radiologist / Reviewer -->
  <g transform="translate(1000, 220)">
    <rect width="240" height="260" rx="10" fill="#f8fafc" stroke="#7c3aed" stroke-width="2" filter="url(#shadow)" />
    <rect width="240" height="40" rx="10" fill="#7c3aed" />
    <rect y="30" width="240" height="10" fill="#7c3aed" />
    <text x="120" y="26" fill="#ffffff" font-family="'Segoe UI', Arial, sans-serif" font-size="15" font-weight="bold" text-anchor="middle">&lt;&lt;External Entity&gt;&gt;</text>
    <text x="120" y="70" fill="#1e293b" font-family="'Segoe UI', Arial, sans-serif" font-size="16" font-weight="bold" text-anchor="middle">Radiologist / Reviewer</text>
    <line x1="20" y1="90" x2="220" y2="90" stroke="#cbd5e1" stroke-width="1.5" />
    <text x="30" y="125" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Uploads patient MRI slice</text>
    <text x="30" y="155" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Inspects visual heatmaps</text>
    <text x="30" y="185" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Reviews reliability flag</text>
    <text x="30" y="215" fill="#475569" font-family="'Segoe UI', Arial, sans-serif" font-size="12">• Flags cases for manual review</text>
  </g>
  
  <!-- Flows between Researcher and System -->
  <!-- Top flow: Researcher -> System -->
  <path d="M 300 280 L 485 280" stroke="#2563eb" stroke-width="2.2" marker-end="url(#arrow-blue)" />
  <rect x="315" y="250" width="160" height="24" rx="4" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" />
  <text x="395" y="266" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">MRI Dataset &amp; Masks</text>
  
  <!-- Second flow: Researcher -> System -->
  <path d="M 300 330 L 485 330" stroke="#2563eb" stroke-width="2.2" marker-end="url(#arrow-blue)" />
  <rect x="315" y="305" width="160" height="22" rx="4" fill="#eff6ff" stroke="#bfdbfe" stroke-width="1" />
  <text x="395" y="321" fill="#1e40af" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Model Config &amp; Splits</text>
  
  <!-- Third flow: System -> Researcher -->
  <path d="M 485 410 L 300 410" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-green)" />
  <rect x="315" y="385" width="160" height="22" rx="4" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
  <text x="395" y="401" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Trained Model Weights</text>
  
  <!-- Fourth flow: System -> Researcher -->
  <path d="M 485 460 L 300 460" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-green)" />
  <rect x="315" y="435" width="160" height="22" rx="4" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
  <text x="395" y="451" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">AUROC / Error Reports</text>
  
  <!-- Flows between Radiologist and System -->
  <!-- Top flow: Radiologist -> System -->
  <path d="M 1000 290 L 815 290" stroke="#7c3aed" stroke-width="2.2" marker-end="url(#arrow-blue)" />
  <rect x="825" y="265" width="165" height="22" rx="4" fill="#f5f3ff" stroke="#ddd6fe" stroke-width="1" />
  <text x="907" y="281" fill="#6d28d9" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">MRI Slice &amp; Optional Mask</text>
  
  <!-- Second flow: System -> Radiologist -->
  <path d="M 815 350 L 1000 350" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-green)" />
  <rect x="825" y="325" width="165" height="22" rx="4" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
  <text x="907" y="341" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Tumour Class &amp; Confidence</text>
  
  <!-- Third flow: System -> Radiologist -->
  <path d="M 815 410 L 1000 410" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-green)" />
  <rect x="825" y="385" width="165" height="22" rx="4" fill="#ecfdf5" stroke="#a7f3d0" stroke-width="1" />
  <text x="907" y="401" fill="#065f46" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Grad-CAM Overlay Map</text>
  
  <!-- Fourth flow: System -> Radiologist -->
  <path d="M 815 470 L 1000 470" stroke="#dc2626" stroke-width="2.2" marker-end="url(#arrow-red)" />
  <rect x="820" y="445" width="175" height="22" rx="4" fill="#fef2f2" stroke="#fecaca" stroke-width="1" />
  <text x="907" y="461" fill="#991b1b" font-family="'Segoe UI', Arial, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">Reliability Flag &amp; Diagnostics</text>
  
  <!-- Footer Note -->
  <text x="650" y="680" fill="#64748b" font-family="'Segoe UI', Arial, sans-serif" font-size="12" text-anchor="middle">Context-level boundary showing external user interactions, model provisioning, and clinical advisory reporting.</text>
</svg>"""
    return render_svg_to_png_and_jpg(svg, "fig_4_2_dfd_level_0")

if __name__ == "__main__":
    print("Building Part 1 diagrams...")
    build_fig_3_1()
    build_fig_3_2()
    build_fig_4_1()
    build_fig_4_2()
    print("Part 1 complete!")
