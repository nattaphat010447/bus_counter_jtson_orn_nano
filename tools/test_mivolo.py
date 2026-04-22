import cv2
import numpy as np
import onnxruntime as ort

def test_mivolo(image_path='data/test.png', model_path='models/mivolo_v2.onnx'):
    try:
        session = ort.InferenceSession(model_path)
        input_names = [i.name for i in session.get_inputs()]
        print(f"[Info] Model loaded. Expected inputs: {input_names}")
    except Exception as e:
        print(f"[Error] Load model failed: {e}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[Error] Cannot find image: {image_path}")
        return

    blob = cv2.resize(img, (384, 384))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    input_feed = {
        'pixel_values_face': blob,
        'pixel_values_body': blob
    }

    try:
        outputs = session.run(None, input_feed)
        gender_out = outputs[0][0]
        age = float(outputs[1][0][0])
        gender = "Male" if gender_out[0] > gender_out[1] else "Female"

        print("-" * 30)
        print(f"Result for: {image_path}")
        print(f"Gender : {gender}")
        print(f"Age    : {age:.1f}")
        print("-" * 30)

    except Exception as e:
        print(f"[Error] Inference failed: {e}")

if __name__ == "__main__":
    test_mivolo()