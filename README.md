
# Hand Sign Recognition using Deep Learning

This project implements a real-time hand sign recognition system using a trained Convolutional Neural Network (CNN) model. It captures live video from your webcam, detects hand gestures in a specified Region of Interest (ROI), and predicts the corresponding sign language character (A-Z, space, delete, or nothing).

## 🚀 Features

- Real-time webcam feed and prediction
- Hand sign classification for 26 alphabets and 3 special commands (`space`, `del`, `nothing`)
- Uses a custom-trained deep learning model
- Visual feedback of prediction and confidence score

## 🛠️ Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy

## 📁 Project Structure

```
hand-sign-recognition/
│
├── hand_sign_model_clean.h5      # Trained CNN model
├── hand_sign_recognition.py      # Main Python script for real-time detection
└── README.md                     # Project documentation
```

## 📷 How It Works

1. The webcam captures the live feed.
2. A fixed ROI (Region Of Interest) is defined for the user to show hand gestures.
3. The ROI is preprocessed and passed through the trained model.
4. The predicted character and confidence score are displayed on the screen in real time.

## 🧠 Model Details

The model was trained on clean hand gesture images (64x64) normalized to [0, 1]. It supports classification for:
- Alphabets: `A-Z`
- Special commands: `space`, `del`, `nothing`

## 🖥️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/hand-sign-recognition.git
cd hand-sign-recognition
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

_Example dependencies (add these to `requirements.txt`):_
```
tensorflow
opencv-python
numpy
```

### 3. Add Your Trained Model

Ensure `hand_sign_model_clean.h5` (your trained model) is in the same directory.

### 4. Run the Script

```bash
python hand_sign_recognition.py
```

Press `q` to quit the webcam stream.

## 📌 Notes

- Ensure good lighting and clear hand visibility for better predictions.
- The model expects hands to be shown within the ROI box (green rectangle).
- You can retrain the model or fine-tune it on more diverse or personalized data for better accuracy.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

Feel free to contribute by raising issues or creating pull requests!
﻿# Sign-Language-Detector
