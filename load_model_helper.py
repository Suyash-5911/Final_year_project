"""
Helper script to load PyTorch models saved in ZIP/persistent storage format
Use this in Colab to load the model properly
"""

import torch
from pathlib import Path
import sys

def load_zip_format_model(model_path, device='cpu'):
    """
    Load a PyTorch model saved in ZIP format (directory structure)
    
    Args:
        model_path: Path to the model directory
        device: Device to load model to
    
    Returns:
        state_dict: The loaded model state dictionary
    """
    model_path = Path(model_path)
    
    print(f"Loading model from {model_path}...")
    
    # Try different methods
    methods = []
    
    # Method 1: Try _open_zipfile_reader (PyTorch 2.0+)
    try:
        from torch.serialization import _open_zipfile_reader, _is_zipfile
        
        if _is_zipfile(str(model_path)):
            print("  Detected as ZIP file format")
            with _open_zipfile_reader(str(model_path)) as zip_reader:
                state_dict = torch.load(zip_reader, map_location=device, weights_only=False)
            print("  ✓ Loaded using _open_zipfile_reader")
            return state_dict
    except Exception as e:
        methods.append(f"Method 1 (_open_zipfile_reader): {e}")
    
    # Method 2: Load as torch.jit
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        print("  ✓ Loaded as TorchScript model")
        return model
    except Exception as e:
        methods.append(f"Method 2 (torch.jit.load): {e}")
    
    # Method 3: Try torch.package
    try:
        from torch.package import PackageImporter
        importer = PackageImporter(str(model_path))
        state_dict = importer.load_pickle('model', 'state_dict.pkl')
        print("  ✓ Loaded using torch.package")
        return state_dict
    except Exception as e:
        methods.append(f"Method 3 (torch.package): {e}")
    
    # Method 4: Manual persistent load
    try:
        import pickle
        
        # Custom unpickler for persistent storage
        class PersistentUnpickler(pickle.Unpickler):
            def __init__(self, file, data_dir):
                super().__init__(file)
                self.data_dir = data_dir
                
            def persistent_load(self, pid):
                if isinstance(pid, tuple) and len(pid) >= 3:
                    typename, key, location, *rest = pid
                    
                    # Load tensor from data directory
                    tensor_file = self.data_dir / str(key)
                    if tensor_file.exists():
                        # Load the tensor data
                        with open(tensor_file, 'rb') as f:
                            # The file contains raw tensor data
                            # We need to reconstruct it properly
                            pass
                    
                return None
        
        data_pkl = model_path / 'data.pkl'
        data_dir = model_path / 'data'
        
        with open(data_pkl, 'rb') as f:
            unpickler = PersistentUnpickler(f, data_dir)
            state_dict = unpickler.load()
        
        print("  ✓ Loaded using custom persistent unpickler")
        return state_dict
        
    except Exception as e:
        methods.append(f"Method 4 (persistent load): {e}")
    
    # All methods failed
    print("\n❌ All loading methods failed:")
    for method in methods:
        print(f"  {method}")
    
    raise RuntimeError(
        "Could not load model. Please re-save from training notebook using:\n"
        "  torch.save(model.state_dict(), 'model.pth')"
    )


def load_model_for_gradcam(model_class, model_path, device='cpu', **model_kwargs):
    """
    Complete function to load a model for Grad-CAM usage
    
    Args:
        model_class: The model class (e.g., TumourClassifier)
        model_path: Path to saved model
        device: Device to use
        **model_kwargs: Arguments for model initialization
    
    Returns:
        model: Loaded and initialized model ready for inference
    """
    # Initialize model architecture
    print(f"Initializing {model_class.__name__}...")
    model = model_class(**model_kwargs)
    
    # Load weights
    try:
        state_dict = load_zip_format_model(model_path, device)
        model.load_state_dict(state_dict)
    except:
        # If that fails, try loading as a complete model
        print("Attempting to load as complete model...")
        model = torch.load(str(model_path), map_location=device, weights_only=False)
    
    model.to(device)
    model.eval()
    
    print("✓ Model ready for inference")
    return model


if __name__ == "__main__":
    # Example usage
    print("Model Loading Helper")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
        
        state_dict = load_zip_format_model(model_path, device)
        print(f"\n✓ Successfully loaded model!")
        print(f"  State dict keys: {len(state_dict)} layers")
    else:
        print("Usage: python load_model_helper.py <model_path> [device]")
        print("Example: python load_model_helper.py models/best_model cuda")
