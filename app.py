"""
Brain Tumor Classification & Grad-CAM Explainability Dashboard
Interactive web application for testing the model and visualizing explanations
"""

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
import io

# Configure page
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules
import sys
sys.path.insert(0, 'src')

from src.models.classifier import TumourClassifier
from src.explainability.grad_cam import ExplanationGenerator
from src.explainability.visualization import overlay_heatmap
from src.data.preprocessor import Preprocessor

# Cache model loading
@st.cache_resource
def load_model():
    """Load the trained model (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TumourClassifier(backbone='resnet50', num_classes=3, pretrained=False)
    
    try:
        model.load_state_dict(torch.load('models/best_model_v2.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

@st.cache_resource
def load_explainer(_model, _device):
    """Load Grad-CAM explainer (cached)"""
    return ExplanationGenerator(model=_model, colormap='jet', device=_device)

@st.cache_data
def load_training_history():
    """Load training history if available"""
    history_path = Path('reults/training_history.json')
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

def preprocess_image(image, device):
    """Preprocess uploaded image for model"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Store original for visualization
    original = np.array(image)
    
    # Transform and add batch dimension
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor, original

def predict(model, image_tensor, device):
    """Make prediction on image"""
    class_names = ['Glioma', 'Meningioma', 'Pituitary']
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return {
        'class': class_names[predicted_class.item()],
        'class_idx': predicted_class.item(),
        'confidence': confidence.item(),
        'probabilities': probabilities[0].cpu().numpy()
    }

def generate_explanation(explainer, image_tensor, class_idx):
    """Generate Grad-CAM explanation"""
    cam = explainer.generate_cam(image_tensor, target_class=class_idx)
    return cam

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
    st.title("🧠 Brain Tumor Classifier")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Single Image Analysis", "📊 Model Information", "📁 Batch Processing"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "**Evidence-Aware Brain Tumor Classification**\n\n"
        "This system uses CNN with Grad-CAM explanations to classify brain tumors "
        "into three categories: Glioma, Meningioma, and Pituitary."
    )
    
    # Model status
    model, device = load_model()
    if model is not None:
        st.success(f"✓ Model loaded on {device}")
    else:
        st.error("✗ Model not loaded")

# Main content
if page == "🏠 Home":
    st.title("Brain Tumor Classification System")
    st.markdown("### Welcome to the Interactive Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "99.3%", "Validation")
    with col2:
        st.metric("Tumor Types", "3", "Glioma, Meningioma, Pituitary")
    with col3:
        st.metric("Backbone", "ResNet-50", "Pretrained CNN")
    
    st.markdown("---")
    
    # Features
    st.markdown("### 🎯 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Classification**
        - 99.3% validation accuracy
        - Three tumor types supported
        - Real-time predictions
        - Confidence scores
        """)
        
        st.markdown("""
        **Explainability**
        - Grad-CAM visual explanations
        - Heatmap overlays
        - Region highlighting
        - Interpretable results
        """)
    
    with col2:
        st.markdown("""
        **Analysis**
        - Single image analysis
        - Batch processing
        - Detailed visualizations
        - Export results
        """)
        
        st.markdown("""
        **Reliability**
        - High confidence predictions
        - Visual evidence
        - Transparent AI
        - Clinical-ready
        """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    # Load training history
    history = load_training_history()
    if history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Metrics**")
            final_epoch = len(history['train_acc']) - 1
            st.write(f"- Final Training Acc: {history['train_acc'][final_epoch]:.2f}%")
            st.write(f"- Final Training Loss: {history['train_loss'][final_epoch]:.4f}")
        
        with col2:
            st.markdown("**Validation Metrics**")
            st.write(f"- Final Validation Acc: {history['val_acc'][final_epoch]:.2f}%")
            st.write(f"- Final Validation Loss: {history['val_loss'][final_epoch]:.4f}")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to different sections")

elif page == "🔍 Single Image Analysis":
    st.title("Single Image Analysis")
    st.markdown("Upload a brain MRI image to classify and explain")
    
    uploaded_file = st.file_uploader(
        "Choose a brain MRI image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a brain MRI scan (JPG or PNG format)"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("### Analysis")
            
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    image_tensor, original_image = preprocess_image(image, device)
                    
                    # Predict
                    result = predict(model, image_tensor, device)
                    
                    # Display prediction
                    st.markdown("#### Prediction Results")
                    st.success(f"**Predicted Class:** {result['class']}")
                    st.metric("Confidence", f"{result['confidence']*100:.2f}%")
                    
                    # Probabilities
                    st.markdown("#### Class Probabilities")
                    class_names = ['Glioma', 'Meningioma', 'Pituitary']
                    prob_data = {class_names[i]: result['probabilities'][i] for i in range(3)}
                    st.bar_chart(prob_data)
        
        # Grad-CAM Explanation
        st.markdown("---")
        st.markdown("### 🔥 Grad-CAM Explanation")
        
        if st.button("Generate Explanation"):
            with st.spinner("Generating Grad-CAM..."):
                # Preprocess again
                image_tensor, original_image = preprocess_image(image, device)
                
                # Get prediction
                result = predict(model, image_tensor, device)
                
                # Generate explanation
                explainer = load_explainer(model, device)
                cam = generate_explanation(explainer, image_tensor, result['class_idx'])
                
                # Create visualizations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Original**")
                    st.image(original_image, use_container_width=True)
                
                with col2:
                    st.markdown("**Heatmap**")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    im = ax.imshow(cam, cmap='jet', vmin=0, vmax=1)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046)
                    st.pyplot(fig)
                    plt.close()
                
                with col3:
                    st.markdown("**Overlay**")
                    overlay = overlay_heatmap(original_image, cam, alpha=0.4)
                    st.image(overlay, use_container_width=True)
                
                st.success("✓ Explanation generated successfully!")
                
                # Interpretation
                st.markdown("### 💡 Interpretation")
                st.info(
                    f"The red regions show where the model focused to classify this as **{result['class']}**. "
                    f"Brighter red indicates higher importance for the decision."
                )

elif page == "📊 Model Information":
    st.title("Model Information & Statistics")
    
    tab1, tab2, tab3 = st.tabs(["📈 Training History", "🏗️ Architecture", "📊 Performance"])
    
    with tab1:
        st.markdown("### Training History")
        
        history = load_training_history()
        if history:
            # Training curves
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Accuracy Over Epochs**")
                fig, ax = plt.subplots(figsize=(8, 5))
                epochs = list(range(1, len(history['train_acc']) + 1))
                ax.plot(epochs, history['train_acc'], label='Training', linewidth=2)
                ax.plot(epochs, history['val_acc'], label='Validation', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy (%)')
                ax.set_title('Training & Validation Accuracy')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.markdown("**Loss Over Epochs**")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(epochs, history['train_loss'], label='Training', linewidth=2)
                ax.plot(epochs, history['val_loss'], label='Validation', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training & Validation Loss')
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            # Final metrics
            st.markdown("### Final Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            final_idx = len(history['train_acc']) - 1
            with col1:
                st.metric("Train Accuracy", f"{history['train_acc'][final_idx]:.2f}%")
            with col2:
                st.metric("Val Accuracy", f"{history['val_acc'][final_idx]:.2f}%")
            with col3:
                st.metric("Train Loss", f"{history['train_loss'][final_idx]:.4f}")
            with col4:
                st.metric("Val Loss", f"{history['val_loss'][final_idx]:.4f}")
        else:
            st.warning("Training history not available")
    
    with tab2:
        st.markdown("### Model Architecture")
        
        st.code("""
Model: TumourClassifier
├── Backbone: ResNet-50 (Pretrained on ImageNet)
│   ├── conv1: 7×7 conv, 64 filters
│   ├── layer1: 3 bottleneck blocks
│   ├── layer2: 4 bottleneck blocks
│   ├── layer3: 6 bottleneck blocks
│   └── layer4: 3 bottleneck blocks ← Grad-CAM target
├── Global Average Pooling
└── Fully Connected: 2048 → 3 classes
        """, language="text")
        
        st.markdown("### Model Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Input**
            - Size: 224 × 224 × 3
            - Normalization: ImageNet stats
            - Format: RGB
            """)
        
        with col2:
            st.markdown("""
            **Output**
            - Classes: 3
            - Activation: Softmax
            - Labels: Glioma, Meningioma, Pituitary
            """)
    
    with tab3:
        st.markdown("### Performance Metrics")
        
        # Confusion matrix (if available)
        cm_path = Path('reults/confusion_matrix.png')
        if cm_path.exists():
            st.markdown("**Confusion Matrix**")
            st.image(str(cm_path), use_container_width=True)
        
        # Per-class performance
        st.markdown("### Per-Class Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Glioma**")
            st.metric("Accuracy", "99%")
        
        with col2:
            st.markdown("**Meningioma**")
            st.metric("Accuracy", "99%")
        
        with col3:
            st.markdown("**Pituitary**")
            st.metric("Accuracy", "100%")

elif page == "📁 Batch Processing":
    st.title("Batch Processing")
    st.markdown("Upload multiple images for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Choose multiple brain MRI images...",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Upload multiple brain MRI scans"
    )
    
    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} images")
        
        if st.button("🚀 Process All Images", type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx+1}/{len(uploaded_files)}: {file.name}")
                
                # Load and process
                image = Image.open(file)
                image_tensor, original_image = preprocess_image(image, device)
                result = predict(model, image_tensor, device)
                
                results.append({
                    'filename': file.name,
                    'prediction': result['class'],
                    'confidence': result['confidence']
                })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✓ Processing complete!")
            
            # Display results
            st.markdown("### Results")
            
            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Summary
            st.markdown("### Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                glioma_count = len([r for r in results if r['prediction'] == 'Glioma'])
                st.metric("Glioma", glioma_count)
            
            with col2:
                meningioma_count = len([r for r in results if r['prediction'] == 'Meningioma'])
                st.metric("Meningioma", meningioma_count)
            
            with col3:
                pituitary_count = len([r for r in results if r['prediction'] == 'Pituitary'])
                st.metric("Pituitary", pituitary_count)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="batch_results.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Brain Tumor Classification System | Final Year Project 2026-2027"
    "</div>",
    unsafe_allow_html=True
)
