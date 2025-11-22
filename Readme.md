# Face & Eye Liveness Detection – Depth Map, Gaze Tracking, and Audio Verification

This project implements a complete multimodal authentication system that combines **face liveness detection**, **depth-map analysis**, **eye gaze tracking**, and **audio verification**. It uses computer vision, convolutional neural networks, and audio processing to determine whether a user is genuine or spoofed.

---

## 1. Project Overview

The authentication workflow includes:

- Face liveness detection using depth-map prediction  
- Eye gaze tracking using 68 facial landmarks  
- Audio verification using live speech recognition  
- Real-time prediction using OpenCV & TensorFlow  
- Optional Django-based web interface  

The model is trained using a custom CNN architecture and deployed for real-time authentication.

---

## 2. Installation

### TensorFlow GPU Setup (if using GPU)

Install the following manually:

- **CUDA Toolkit**  
  https://developer.nvidia.com/cuda-toolkit

- **cuDNN**  
  https://developer.nvidia.com/cudnn

- **Visual Studio C++ Build Tools** (Required for dlib)  
  https://visualstudio.microsoft.com/downloads  
  → Select: **Desktop Development with C++**

---

## 3. Model Setup

Navigate to:
Face live/gaze_tracking/

Create:
trained_models/
Download: shape_predictor_68_face_landmarks.dat

From:  
https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat

Move it into:
gaze_tracking/trained_models/

---

## 4. Python Dependencies

### Upgrade pip
```
pip install --upgrade pip
```

### Core Libraries
```
pip install opencv-python
pip install opencv-contrib-python
pip install tensorflow
pip install numpy
```

### Audio Dependencies
```
pip install speechrecognition pyaudio pvrecorder
pip install pipwin
pipwin install pyaudio
```

### Additional Required Libraries
```
pip install -r requirements.txt
pip install gaze-tracking dlib imutils
```
---

## 5. Virtual Environment (Optional)
```
python -m venv venv
venv\Scripts\activate
```
---

## 6. Running the Project
### Train the Model
```
python FaceEye_Train.py
```

### Run the Web Server (Django)
```
cd verification_project
python manage.py runserver
```

---

## 7. System Architecture

### Depth Map CNN Model
- 3 convolutional layers (ReLU)
- 1 fully connected output layer
- Trained for **10 epochs**
- Batch size: **32**
- Train-test split: **75:25**

### Real-Time Authentication Pipeline
1. User initiates authentication  
2. System displays lighting/distance instructions  
3. Gaze tracking begins  
4. User reads text for audio verification  
5. Depth-map based liveness detection  
6. Combined result → **Success / Failure**

---

## 8. Troubleshooting

- Ensure camera permissions are enabled  
- Use a well-lit environment  
- Reduce background noise for audio tests  
- Reinstall PyAudio with pipwin if errors occur  

---

## 9. Repository Structure
```
Eye_Face_Detection/
│
├── FaceEye_Train.py
├── verification_project/
├── gaze_tracking/
│ └── trained_models/
│ └── shape_predictor_68_face_landmarks.dat
├── requirements.txt
└── README.md
```

---

## 10. License

```This project is open-source under the **MIT License**.```
