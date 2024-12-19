import threading
import pyttsx3
from ultralytics import YOLO
import cv2
import math
import base64
from inference_sdk import InferenceHTTPClient
from util import CLIENT, MODEL_ID


def speak(text):
    """Handles text-to-speech asynchronously."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    engine.say(text)
    engine.runAndWait()


def video_detection(path_x):
    cap = cv2.VideoCapture(path_x)

    detected_objects = set()  # Track objects announced
    #model = YOLO("../Yolo-Weights/yolo5s.pt")
    while True:
        success, img = cap.read()
        if not success:
            break

        # Convert frame to bytes and then to base64
        _, img_encoded = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Perform inference using Roboflow API
        try:
            # Perform inference
            result = CLIENT.infer(img_base64, model_id=MODEL_ID)

            # Directly access predictions from the dictionary
            predictions = result['predictions']  # No need to call `.json()`
        except Exception as e:
            print(f"Error during inference: {e}")
            break

        frame_objects = set()  # Track objects detected in the current frame

        for prediction in predictions:
            # Extract bounding box and class details
            x, y, w, h = prediction['x'], prediction['y'], prediction['width'], prediction['height']
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            conf = round(prediction['confidence'], 2)
            class_name = prediction['class']

            # Draw bounding box and label on the frame
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            label = f'{class_name} {conf}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Announce detected objects
            if class_name not in detected_objects:
                threading.Thread(target=speak, args=(f"{class_name.replace('_', ' ')} sign detected",)).start()
                detected_objects.add(class_name)
            print(class_name)
            frame_objects.add(class_name)
        #re = model
        # Remove objects that are no longer in the frame from detected_objects
        detected_objects.intersection_update(frame_objects)
        
        yield img

