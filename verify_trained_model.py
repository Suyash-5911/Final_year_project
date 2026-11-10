"""
Verify the trained model by testing on sample images
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU mode
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Import model architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.models.classifier import TumourClassifier

def load_model(model_path, device='cpu'):
    """Load the trained model"""
    model = TumourClassifier(backbone='resnet50', num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✓ Model loaded successfully")
    return model

def preprocess_image(image_path):
    """Preprocess image for inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image

def predict(model, image_tensor, device='cpu'):
    """Make prediction on image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities[0]

def main():
    print("=" * 60)
    print("TRAINED MODEL VERIFICATION")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    project_root = Path(__file__).parent
    model_path = project_root / 'models' / 'best_model'
    data_dir = project_root / 'data' / 'brisc' / 'processed'
    class_names = ['glioma', 'meningioma', 'pituitary']
    
    print(f"\nLoading model...")
    model = load_model(model_path, device)
    
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE IMAGES")
    print("=" * 60)
    
    test_results = []
    
    for class_name in class_names:
        class_dir = data_dir / 'test' / class_name
        if not class_dir.exists():
            continue
        
        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        if not image_files:
            continue
        
        image_path = image_files[0]
        print(f"\n📁 Testing: {class_name} - {image_path.name}")
        
        image_tensor, original_image = preprocess_image(image_path)
        predicted_class, confidence, probabilities = predict(model, image_tensor, device)
        predicted_name = class_names[predicted_class]
        
        is_correct = predicted_name == class_name
        status = "✓ CORRECT" if is_correct else "✗ WRONG"
        
        print(f"   Predicted: {predicted_name} ({confidence:.2%})")
        print(f"   Probabilities: ", end="")
        for i, cls in enumerate(class_names):
            print(f"{cls}={probabilities[i]*100:.1f}% ", end="")
        print(f"\n   Status: {status}")
        
        test_results.append({
            'image': original_image,
            'true_class': class_name,
            'predicted_class': predicted_name,
            'confidence': confidence,
            'probabilities': probabilities,
            'correct': is_correct
        })
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    correct = sum(1 for r in test_results if r['correct'])
    total = len(test_results)
    print(f"Samples tested: {total}")
    print(f"Correct predictions: {correct}/{total}")
    if total > 0:
        print(f"Accuracy on samples: {correct/total*100:.1f}%")
    
    print("\n✓ Model verification complete!")
    print("=" * 60)
    
    return test_results

if __name__ == '__main__':
    results = main()
