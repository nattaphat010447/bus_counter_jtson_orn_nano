import os
import sys
from ultralytics import YOLO

# Resolve project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def export_yolo_models():
    # Target models for the pipeline
    target_models = [
        "yolov8n.pt",
        "yolov8n-face.pt"
    ]

    for model_file in target_models:
        model_path = os.path.join(MODELS_DIR, model_file)
        
        # Check model existence before processing
        if not os.path.exists(model_path):
            print(f"[Warning] Model not found: {model_path}. Skipping.")
            continue

        print(f"[Info] Exporting {model_file}...")
        try:
            model = YOLO(model_path)
            
            # Export using CPU to bypass Jetson PyTorch cuDNN tracing bugs
            # dynamic=False is required for stable TensorRT optimization
            model.export(format="onnx", device="cpu", dynamic=False)
            
            print(f"[Success] Exported ONNX for {model_file}")
        except Exception as e:
            print(f"[Error] Failed to export {model_file}: {e}")

if __name__ == "__main__":
    export_yolo_models()