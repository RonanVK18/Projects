import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from drawing_utils_patch import redraw_landmarks

import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.drawing_utils as mp_drawing

# Load the trained model
model = load_model('/home/raphaelcg/SLD_model.h5')

# Define a list of possible signs
signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','Space','Nothing']

# Initialize the MediaPipe Hands object
mp_hands = mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

selected_keypoints = [8, 12, 16, 20,4, 5, 9, 13, 17]

# Open a video capture object
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read each frame from the camera
    ret, frame = cap.read()

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results= hands.process(frame_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        # Get the first hand landmark
        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract the keypoints from the hand landmark
        keypoints = np.array([[p.x, p.y] for p in hand_landmarks.landmark])

        # Preprocess the keypoints as a 64x64 image
        input_image=np.full((64,64,3),128)
        for idx in selected_keypoints:
            x, y=int(keypoints[idx][0]*64), int(keypoints[idx][1] * 64)
            if 0 <= x < 64 and 0 <=y <= 64:
                input_image[y, x]= 255

        input_data=np.expand_dims(input_image, axis=0)

        try:
            # Predict the sign using the model
            prediction = model.predict(input_data)[0]

            # Get the index of the predicted sign
            predicted_sign_index = np.argmax(prediction)

            # Get the predicted sign name
            predicted_sign_name = signs[predicted_sign_index]

        except IndexError:
            predicted_sign_index=-1
            predicted_sign_name='Unrecognized Sign'

        # Draw the hand landmarks on the frame
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the predicted sign
        cv2.putText(frame, f'Prediction: {predicted_sign_name}',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Sign Language Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()



