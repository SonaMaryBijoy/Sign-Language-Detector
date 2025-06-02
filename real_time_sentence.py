import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained hand sign model
model = load_model("hand_sign_model_clean.h5")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Class labels matching your model output
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Variables to track sentence and prediction stability
sentence = ""
previous_label = ""
delay_counter = 0
delay_threshold = 7 # Number of consecutive frames for stable detection

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image for easier interaction

    # Define region of interest (ROI) coordinates
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Preprocess ROI for model input
    roi_resized = cv2.resize(roi, (64, 64))
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    roi_array = img_to_array(roi_rgb) / 255.0
    roi_array = np.expand_dims(roi_array, axis=0)

    # Predict the sign in ROI
    prediction = model.predict(roi_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = classes[predicted_index]
    confidence = prediction[predicted_index]

    # Debug: print prediction and confidence
    print(f"Predicted: {predicted_label}, Confidence: {confidence:.2f}", flush=True)

    # Stable prediction check
    if predicted_label == previous_label:
        delay_counter += 1
    else:
        delay_counter = 0
        previous_label = predicted_label

    # If prediction stable for enough frames, update sentence
    if delay_counter >= delay_threshold:
        if predicted_label == "space":
            sentence += " "
            engine.say("space")
            engine.runAndWait()
        elif predicted_label == "del":
            sentence = sentence[:-1]
            engine.say("delete")
            engine.runAndWait()
        elif predicted_label != "nothing":
            sentence += predicted_label
            engine.say(predicted_label)
            engine.runAndWait()

        delay_counter = 0  # Reset counter after updating sentence

    # Print current sentence to terminal every frame
    print(f"Current Sentence: {sentence}", flush=True)

    # Show prediction and sentence on video feed
    cv2.putText(frame, f"{predicted_label} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, "Sentence: " + sentence, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display frames
    cv2.imshow("Hand Sign Recognition", frame)
    cv2.imshow("ROI", roi_resized)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
