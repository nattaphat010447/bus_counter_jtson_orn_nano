import os
import urllib.request

def download_file(url, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    if os.path.exists(dest_path):
        print(f"[Info] Found existing model: {dest_path}")
        return

    print(f"[Info] Downloading {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[Success] Downloaded: {dest_path}")
    except Exception as e:
        print(f"[Error] Failed to download {dest_path}: {e}")

def main():
    models_to_download = {
        "models/yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "models/yolov8n-face.pt": "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt" 
    }

    print("--- Checking Missing Models ---")
    for file_path, url in models_to_download.items():
        download_file(url, file_path)

if __name__ == "__main__":
    main()