# 🐾 Wildlife Detection System

A real-time **multi-model animal detection and alert system** using **MobileNet-SSD**, **YOLOv11m**, and a **YOLOv8 wildlife-trained model**.  
This project detects animals from webcam/video input and triggers alerts (sound + console messages) for critical species such as **bear, giraffe, rhino, buffalo, horse, elephant, and zebra**.

---

## ✨ Features
- 🔍 **Triple-model detection**: Combines MobileNet-SSD, YOLOv11m, and YOLOv8 for robust accuracy.  
- 🐘 **Animal alerts**: Plays a unified `alert.wav` sound and prints red console messages when target animals are detected.  
- 🎯 **Multi-scale detection**: Detects both small and large animals using SSD and YOLO feature maps.  
- 📊 **Tracking metrics**: Counts detections per class/model and logs confidence scores.  
- 🎥 **Real-time webcam feed**: Annotated bounding boxes with species names and confidence levels.  

---

## 📂 Folder Structure
```
animal_detection_project/
├── main.py                          # Combined detection script
├── mobilenet_ssd/
│   ├── deploy.prototxt              # MobileNet-SSD architecture
│   └── mobilenet_iter_73000.caffemodel  # Pretrained weights
├── yolov11/
│   └── yolo11m.pt                   # YOLOv11m weights
├── runs/
│   └── detect/
│       └── wildlife_detector8/
│           └── weights/
│               └── best.pt          # Trained YOLOv8 weights
├── sound/
│   └── alert.wav                    # Alert sound
├── african-wildlife.yaml            # Dataset config for YOLOv8 training
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/animal_detection_project.git
   cd animal_detection_project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the model files in place:
   - `mobilenet_ssd/deploy.prototxt`
   - `mobilenet_ssd/mobilenet_iter_73000.caffemodel`
   - `yolov11/yolo11m.pt`
   - `runs/detect/wildlife_detector8/weights/best.pt`

---

## 🚀 Usage
Run the detection script:
```bash
python main.py
```

- Press **ESC** to exit the webcam feed.  
- When any of the 7 alert animals are detected, you’ll hear `alert.wav` and see a red console message like:
  ```
  ALERT: ELEPHANT detected!
  ```

---

## 📊 Supported Animals
- MobileNet-SSD: `bird, cat, cow, dog, horse, sheep`
- YOLOv11m: `elephant, bear, zebra, giraffe`
- YOLOv8 (wildlife-trained): `buffalo, rhino, zebra, elephant`

**Alert Animals:** `bear, giraffe, rhino, buffalo, horse, elephant, zebra`

---

## 🛠 Requirements
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- Playsound
- Matplotlib, Seaborn, scikit-learn (optional for metrics)

Install all with:
```bash
pip install ultralytics opencv-python numpy playsound matplotlib seaborn scikit-learn
```

---

## 📌 Future Improvements
- Save detection logs to CSV/JSON  
- Confusion matrix visualization for model benchmarking  
- Support for video file input instead of webcam  
- GUI overlay for alerts  

---

## 📜 License
This project is licensed under the Mozilla Public License 2.0 (MPL-2.0)– see the [LICENSE](LICENSE) file for details.

---
