import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import time
import glob
from src.tracker_engine import TrackerEngine
from src.face_engine import FaceEngine

# Helper: List files by extension
def get_media_files(ext_list):
    files = []
    for ext in ext_list:
        files.extend(glob.glob(f"data/*{ext}"))
    return files

# CLI: Prompt user file selection
def select_file(files):
    if not files:
        print("[Error] No files found in data/")
        return None
    
    print("\n--- Available Files ---")
    for i, f in enumerate(files):
        print(f"[{i+1}] {os.path.basename(f)}")
    print("[0] Cancel")
    
    while True:
        try:
            choice = int(input("\nSelect number: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(files):
                return files[choice-1]
            print("Invalid input.")
        except ValueError:
            print("Integer required.")

# Sub-routine: Run tracker engine
def run_tracker(source):
    tracker = TrackerEngine(model_path='models/yolov8n.pt', config_path='configs/tracker_zone_config.json')
    cap = cv2.VideoCapture(source)
    
    print(f"[Info] Running Tracker. Source: {source} (Press 'q' to quit)")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        annotated_frame, counts = tracker.process_frame(frame)
        fps = 1.0 / (time.time() - start_time)
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)
        cv2.imshow("Tracker Test", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Sub-routine: Run face engine
def run_mivolo(source, is_image=False):
    face_analyzer = FaceEngine(detector_path='models/yolov8n-face.pt', mivolo_path='models/mivolo_v2.onnx')
    
    print(f"[Info] Running miVOLO. Source: {source} (Press 'q' to quit)")
    
    # Image logic: 5 passes required to bypass smoothing buffer
    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print("[Error] Invalid image.")
            return
            
        for _ in range(5):
            annotated_frame, stable_faces = face_analyzer.process_frame(frame)
            
        cv2.imshow("miVOLO Image", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Video/Camera logic
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        annotated_frame, stable_faces = face_analyzer.process_frame(frame)
        fps = 1.0 / (time.time() - start_time)
        
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 150), 2, 1, (0, 255, 0), 2)
        cv2.imshow("miVOLO Stream", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Main entry point
def main():
    while True:
        print("\n========================================")
        print("DIAGNOSTIC TEST MENU")
        print("========================================")
        print("[1] Test Zone Tracker (Video File)")
        print("[2] Test Zone Tracker (Live Camera)")
        print("[3] Test miVOLO (Video/Image File)")
        print("[4] Test miVOLO (Live Camera)")
        print("[0] Exit")
        print("========================================")
        
        choice = input("Select operation: ")
        
        # Route logic based on selection
        if choice == '1':
            files = get_media_files(['.mp4', '.avi'])
            selected = select_file(files)
            if selected: run_tracker(selected)
            
        elif choice == '2':
            run_tracker(0)
            
        elif choice == '3':
            files = get_media_files(['.mp4', '.avi', '.png', '.jpg'])
            selected = select_file(files)
            if selected:
                is_img = selected.lower().endswith(('.png', '.jpg'))
                run_mivolo(selected, is_image=is_img)
                
        elif choice == '4':
            run_mivolo(0)
            
        elif choice == '0':
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()