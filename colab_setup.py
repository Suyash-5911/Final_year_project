"""
Colab Setup Helper - Run this first in your Colab notebook
"""
import sys
import os
from pathlib import Path

def setup_colab_environment():
    """Setup paths and verify structure for Colab"""
    
    # Change to project directory
    if os.path.exists('/content/Final_year_project'):
        os.chdir('/content/Final_year_project')
        print(f"✓ Changed to: {os.getcwd()}")
    else:
        print("✗ Project directory not found. Please clone first:")
        print("  !git clone https://github.com/Suyash-5911/Final_year_project.git")
        return False
    
    # Add to Python path
    project_root = os.getcwd()
    src_path = os.path.join(project_root, 'src')
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    print(f"✓ Added to path: {project_root}")
    print(f"✓ Added to path: {src_path}")
    
    # Verify required files
    print("\nVerifying project structure...")
    required_files = {
        'src/models/classifier.py': 'Model definition',
        'src/explainability/grad_cam.py': 'Grad-CAM implementation',
        'src/explainability/visualization.py': 'Visualization utilities',
        'src/evaluation/localization.py': 'Localization metrics',
        'models/best_model/data.pkl': 'Trained model weights'
    }
    
    all_present = True
    for file, description in required_files.items():
        exists = os.path.exists(file)
        status = "✓" if exists else "✗"
        print(f"  {status} {file} - {description}")
        if not exists:
            all_present = False
    
    if all_present:
        print("\n✅ All files present! Ready to proceed.")
        return True
    else:
        print("\n⚠ Some files are missing. Please check git clone.")
        return False

if __name__ == "__main__":
    setup_colab_environment()
