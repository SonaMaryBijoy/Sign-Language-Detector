import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = load_model("hand_sign_model_clean.h5")

# Class labels (A to Z)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror view

    # Define ROI (Region Of Interest) — where the user shows the hand
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess ROI for prediction
    roi_resized = cv2.resize(roi, (64, 64))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_array = img_to_array(roi_rgb) / 255.0
    roi_array = np.expand_dims(roi_array, axis=0)

    # Make prediction
    prediction = model.predict(roi_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    confidence = prediction[predicted_index]

    # Display prediction
    text = f"{predicted_label} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # Show the frames
    cv2.imshow("Hand Sign Recognition", frame)
    cv2.imshow("ROI", roi_resized)  # Optional: show the hand crop

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
