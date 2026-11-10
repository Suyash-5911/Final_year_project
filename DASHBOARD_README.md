# 🧠 Brain Tumor Classifier - Web Dashboard

Interactive web application for brain tumor classification with Grad-CAM explanations.

## Features

- 🔍 **Single Image Analysis** - Upload and classify individual MRI scans
- 🔥 **Grad-CAM Explanations** - Visual heatmaps showing model focus
- 📊 **Model Information** - Training history and performance metrics
- 📁 **Batch Processing** - Analyze multiple images at once
- 📥 **Export Results** - Download predictions as CSV

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Usage Guide

### Single Image Analysis

1. Navigate to **"🔍 Single Image Analysis"** in the sidebar
2. Click **"Choose a brain MRI image..."** to upload
3. Click **"🔍 Analyze Image"** to get prediction
4. Click **"Generate Explanation"** to see Grad-CAM heatmap

### Model Information

1. Navigate to **"📊 Model Information"**
2. View training curves, architecture, and performance metrics
3. Check confusion matrix and per-class accuracy

### Batch Processing

1. Navigate to **"📁 Batch Processing"**
2. Upload multiple images (JPG/PNG)
3. Click **"🚀 Process All Images"**
4. Download results as CSV

## Screenshots

### Home Page
- Model accuracy metrics
- Key features overview
- Training performance summary

### Single Image Analysis
- Image upload interface
- Real-time prediction
- Confidence scores
- Class probability distribution

### Grad-CAM Explanation
- Original image
- Heatmap visualization
- Overlay with explanation
- Interpretation guidance

### Model Information
- Training/validation curves
- Model architecture diagram
- Confusion matrix
- Per-class metrics

## System Requirements

### Minimum
- Python 3.8+
- 4GB RAM
- CPU only (slower)

### Recommended
- Python 3.10+
- 8GB RAM
- NVIDIA GPU with CUDA (faster)

## Supported Image Formats

- JPG/JPEG
- PNG
- Size: Any (will be resized to 224×224)
- Color: Grayscale or RGB

## Model Details

- **Architecture**: ResNet-50
- **Classes**: Glioma, Meningioma, Pituitary
- **Input Size**: 224×224×3
- **Accuracy**: 99.3% (validation)

## Troubleshooting

### Port Already in Use

```bash
streamlit run app.py --server.port 8502
```

### Model Not Found

Ensure `models/best_model_v2.pth` exists:

```bash
ls -lh models/best_model_v2.pth
```

### CUDA Out of Memory

Add to `~/.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 10
```

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
maxUploadSize = 20
enableCORS = false
```

## Deployment

### Local Network

```bash
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from repository
4. Share public URL

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and run:

```bash
docker build -t brain-tumor-classifier .
docker run -p 8501:8501 brain-tumor-classifier
```

## Development

### Adding New Features

1. Edit `app.py`
2. Streamlit auto-reloads on save
3. Test in browser

### Custom Styling

Add CSS in `app.py`:

```python
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)
```

## Performance Tips

1. **Caching**: Model and explainer are cached automatically
2. **GPU**: Use CUDA for faster inference
3. **Batch Size**: Process multiple images together
4. **Image Size**: Resize large images before upload

## API Reference

### Model Prediction

```python
from src.models.classifier import TumourClassifier

model = TumourClassifier(backbone='resnet50', num_classes=3)
model.load_state_dict(torch.load('models/best_model_v2.pth'))
model.eval()

# Predict
output = model(image_tensor)
probabilities = F.softmax(output, dim=1)
```

### Grad-CAM Generation

```python
from src.explainability.grad_cam import ExplanationGenerator

explainer = ExplanationGenerator(model=model)
cam = explainer.generate_cam(image_tensor, target_class=0)
```

## Project Structure

```
├── app.py                    # Main Streamlit application
├── src/
│   ├── models/
│   │   └── classifier.py     # Model definition
│   ├── explainability/
│   │   ├── grad_cam.py      # Grad-CAM implementation
│   │   └── visualization.py  # Visualization utilities
│   └── data/
│       └── preprocessor.py   # Data preprocessing
├── models/
│   └── best_model_v2.pth    # Trained weights
├── results/
│   ├── training_history.json
│   └── confusion_matrix.png
└── requirements.txt
```

## FAQ

**Q: Can I use my own model?**  
A: Yes, replace `models/best_model_v2.pth` with your weights.

**Q: How do I add more tumor types?**  
A: Retrain the model with additional classes and update `class_names`.

**Q: Can I deploy without GPU?**  
A: Yes, the model works on CPU (slower inference).

**Q: How accurate is the model?**  
A: 99.3% validation accuracy on the BRISC dataset.

**Q: Is this production-ready?**  
A: This is a research prototype. Clinical validation required for production.

## License

Academic project for Bharati Vidyapeeth / SPPU

## Contributors

Final Year Project Team 2026-2027

## Support

For issues or questions:
1. Check troubleshooting section
2. Review documentation
3. Open GitHub issue

---

**Built with** ❤️ **using Streamlit, PyTorch, and Grad-CAM**
