import cv2
import numpy as np
import threading
from playsound import playsound
from datetime import datetime
from collections import defaultdict
from ultralytics import YOLO

# Load MobileNet-SSD
net = cv2.dnn.readNetFromCaffe('mobilenet_ssd/deploy.prototxt',
                                'mobilenet_ssd/mobilenet_iter_73000.caffemodel')
mobilenet_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                     "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                     "train", "tvmonitor"]

# Load YOLOv11m and YOLOv8 (wildlife-trained)
yolo11_model = YOLO('yolov11/yolo11m.pt')
yolo8_model = YOLO('runs/detect/wildlife_detector8/weights/best.pt')

CONF_THRESH = 0.5

# Animal classes from all models
ANIMAL_CLASSES = {
    "bird", "cat", "cow", "dog", "horse", "sheep",
    "elephant", "bear", "zebra", "giraffe",
    "buffalo", "rhino"
}

# Alert animals
ALERT_ANIMALS = {"bear", "giraffe", "rhino", "buffalo", "horse", "elephant", "zebra"}

# Sound alert function
def play_alert():
    threading.Thread(target=playsound, args=('sound/alert.wav',), daemon=True).start()

# Format bounding box
def format_bbox(bbox):
    return "[" + ", ".join(str(int(x)) for x in bbox) + "]"

# Log sightings
def log_sighting(timestamp, label, model_name, conf, bbox_str):
    print(f"[{timestamp}] {label} detected by {model_name} (conf: {conf:.2f}) at {bbox_str}")

# Draw detections
def draw_detections(frame, detections):
    for model_name, label, conf, box in detections:
        color = (0, 255, 0)
        if label.lower() in ALERT_ANIMALS:
            color = (0, 0, 255)  # Red for alert species
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Tracking stats
class_counts = defaultdict(int)
model_counts = defaultdict(int)
confidence_scores = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    detections = []
    animal_detected = False

    # MobileNet-SSD detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    output = net.forward()

    for i in range(output.shape[2]):
        conf = output[0, 0, i, 2]
        if conf > CONF_THRESH:
            idx = int(output[0, 0, i, 1])
            label = mobilenet_classes[idx]
            box = output[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            bbox = [x1, y1, x2, y2]
            detections.append(('MobileNet', label, conf, bbox))

            class_counts[label] += 1
            model_counts['MobileNet'] += 1
            confidence_scores.append(conf)

            if label.lower() in ANIMAL_CLASSES:
                animal_detected = True
                log_sighting(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, 'MobileNet', conf, format_bbox(bbox))
                if label.lower() in ALERT_ANIMALS:
                    print(f"\033[91mALERT: {label.upper()} detected!\033[0m")
                    play_alert()

    # YOLOv11m detection
    results11 = yolo11_model.predict(source=frame, verbose=False)
    for r in results11:
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf > CONF_THRESH:
                label = yolo11_model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                detections.append(('YOLOv11m', label, conf, bbox))

                class_counts[label] += 1
                model_counts['YOLOv11m'] += 1
                confidence_scores.append(conf)

                if label.lower() in ANIMAL_CLASSES:
                    animal_detected = True
                    log_sighting(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, 'YOLOv11m', conf, format_bbox(bbox))
                    if label.lower() in ALERT_ANIMALS:
                        print(f"\033[91mALERT: {label.upper()} detected!\033[0m")
                        play_alert()

    # YOLOv8 detection
    results8 = yolo8_model(frame, imgsz=640)
    for r in results8:
        for box in r.boxes:
            conf = box.conf.item()
            cls = int(box.cls.item())
            if conf > CONF_THRESH:
                label = yolo8_model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2, y2]
                detections.append(('YOLOv8', label, conf, bbox))

                class_counts[label] += 1
                model_counts['YOLOv8'] += 1
                confidence_scores.append(conf)

                if label.lower() in ANIMAL_CLASSES:
                    animal_detected = True
                    log_sighting(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, 'YOLOv8', conf, format_bbox(bbox))
                    if label.lower() in ALERT_ANIMALS:
                        print(f"\033[91mALERT: {label.upper()} detected!\033[0m")
                        play_alert()

    # Draw and show
    draw_detections(frame, detections)
    cv2.imshow("Triple Model Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
