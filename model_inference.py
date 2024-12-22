from ultralytics import YOLO
import cv2
import numpy as np

def load_model():
    yolo_model = YOLO("model/best.pt")
    return yolo_model


def predict_image(yolo_model, img):
    results = yolo_model.predict(img, conf=0.4)  # Run inference on the image

    detections = results[0].boxes  # Get predictions
    bboxes = []  # List to store bounding boxes
    bbox_metrics = []  # List to store metrics

    # Проход по всем предсказаниям
    for det in detections:
        # Unpack the detection
        x1, y1, x2, y2 = det.xyxy[0]  # Bounding box coordinates
        conf = det.conf[0]  # Confidence scores
        cls = det.cls[0]  # Class index

        bboxes.append([x1.item(), y1.item(), x2.item(), y2.item()])  # Store bounding box coordinates
        bbox_metrics.append({'confidence': conf.item(), 'class': int(cls.item())})  # Store metrics

        # Draw bounding box on the image
        img = cv2.rectangle(np.array(img), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Draw rectangle

    return img, bboxes, bbox_metrics



