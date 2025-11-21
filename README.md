🚗 Car Accident Detection System using YOLOv8 & OpenCV
This project is a real-time car accident detection system built using YOLOv8, OpenCV, and a custom-trained dataset.
It detects vehicle crashes, collisions, and accident scenarios in both video streams and live camera feeds, and highlights accident events using bounding boxes and alerts.

📌 Project Overview
The system analyzes each video frame, detects objects such as cars, motorcycles, persons, and identifies potential accident events based on unusual collision patterns.
Key steps include:
✔ Custom dataset preparation (labeling accident frames)
✔ Training YOLOv8 on accident scenarios
✔ Real-time detection using OpenCV
✔ Alert system triggered when an accident is detected
✔ Visualization using bounding boxes

📂 Repository Structure
You can use the following folder structure in your repository:
car_accident_detection/
│
├── train/                         # Training images
├── test/                          # Testing images
├── valid/                         # Validation images
├── processed/                     # Processed frames / logs
├── uploads/                       # Input media (optional)
├── runs/                          # YOLOv8 auto-generated results
│
├── app.py                         # Main detection script / Streamlit app
├── data.yaml                      # Dataset configuration for YOLOv8
├── best.pt                        # Trained YOLOv8 model (ignored in .gitignore)
├── yolov8s.pt                     # Base model (ignored in .gitignore)
├── yolov8n.pt                     # Base model (ignored in .gitignore)
│
├── detecting_car_accident__accident_detection_customized.ipynb
├── deployment_accident_car_detection.ipynb
├── ML_model_testing.ipynb         # ML evaluation notebook
│
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore rules


🧠 Technologies Used
Python
YOLOv8 (Ultralytics)
OpenCV
NumPy
Pandas
PyTorch
Kaggle Accident Dataset

🛠️ Dataset Preparation
Accident videos were sourced from Kaggle.
Frames were extracted and manually labeled using LabelImg or Roboflow.
Labels included:
car
motorcycle
person
accident (custom class for collision)
A dataset split: train / test / valid.
The dataset config is defined in data.yaml.

🏋️ Model Training (YOLOv8)
Training was performed using the following command:
yolo train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640 plots=True

Outputs include:
✔ best.pt → best trained model
✔ Training charts
✔ Confusion matrix
✔ Precision-Recall curves

🎥 Run the Accident Detection
1️⃣ Detect from Video
python app.py --source input_video.mp4

2️⃣ Live Webcam Detection
python app.py --source 0

3️⃣ Streamlit App (if included)
streamlit run app.py


📊 Results
Detects accident scenes with high accuracy
Works in real-time (subject to hardware capability)
Identifies multiple objects simultaneously
Bounding boxes highlight accident frames
Stable detection even in low-light videos

🌟 Model Highlights
Uses enhanced PANet feature fusion (YOLOv8 architecture)
Optimized for multi-scale detection
Lightweight and fast
Accurate on custom dataset


Can detect:
✔ Car-to-car crash
✔ Car hitting a person
✔ Motorcycle crash
✔ High-impact accidents

🚀 Future Improvements
Severity classification (minor / moderate / major)
Accident anticipation (predict crash before it happens)
Real-time CCTV integration
Deployment on edge devices (Jetson Nano / Raspberry Pi)
SMS/email alert system



👩‍💻 Author
Sakshi Godse
M.Tech Artificial Intelligence
GitHub: @sakshigodse05

