import time

import cv2
import os
from matplotlib import text
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import lernet.network
import numpy as np

# Load model
model = lernet.network.FromFile("biggermodel(98).npz")

gesture_labels = ['call', 'dislike', 'fist', 'four', 'like', 'one', 'no_gesture', 'ok', 'one', 'palm', 'two', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted']
gesture_dict = {i: label for i, label in enumerate(gesture_labels)}

# Settings and landmarker model
resolution = (2560, 1440)

root_dir = os.path.abspath(os.path.dirname(__file__))
base_options = python.BaseOptions(
	model_asset_path=os.path.join(root_dir, "hand_landmarker.task")
)

options = vision.HandLandmarkerOptions(
	base_options=base_options,
	num_hands=10,
	running_mode=vision.RunningMode.VIDEO
)

# Model and constants
detector = vision.HandLandmarker.create_from_options(options)
font_type = cv2.FONT_HERSHEY_SIMPLEX
col = (0, 0, 255)

hand_connections = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17)
]

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

print(f"Requested resolution: {resolution[0]} x {resolution[1]}")
print("Actual resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

frame_timestamp = 0
prev_time = time.time()

# Main loop
while cap.isOpened():
	success, frame = cap.read()
	if not success:
		break

	#frame = cv2.flip(frame, 1)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
	result = detector.detect_for_video(mp_image, frame_timestamp)

	frame_timestamp += 1

	# calculate FPS
	current_time = time.time()
	fps = 1 / (current_time - prev_time) if current_time > prev_time else 0.0
	prev_time = current_time

	# Draw FPS
	cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), font_type, 1, (0, 255, 0), 2)

	# Draw hand landmarks
	for i in range(len(result.hand_landmarks)):
		h, w, _ = frame.shape

		hand_landmarks = result.hand_landmarks[i]
		handedness = result.handedness[i][0]

		# Predict gesture
		flat_landmarks = []
		for landmark in hand_landmarks:
			flat_landmarks.append(landmark.x)
			flat_landmarks.append(landmark.y)

		prediction_gesture = gesture_dict[model.Predict(np.array([flat_landmarks]))]

		# Draw landmarks
		for j, lm in enumerate(hand_landmarks):
			x, y = int(lm.x * w), int(lm.y * h)
			cv2.putText(frame, str(j), (x + 5, y), font_type, 0.5, (255, 255, 255), 1)
			cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

		# Draw connections
		for connection in hand_connections:
			a, b = connection
			x1, y1 = int(hand_landmarks[a].x * w), int(hand_landmarks[a].y * h)
			x2, y2 = int(hand_landmarks[b].x * w), int(hand_landmarks[b].y * h)
			cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

		# Draw bounding box and label
		xs = [lm.x for lm in hand_landmarks]
		ys = [lm.y for lm in hand_landmarks]
		min_x, max_x = int(min(xs) * w), int(max(xs) * w)
		min_y, max_y = int(min(ys) * h), int(max(ys) * h)

		# Get hand label and score
		label = handedness.category_name
		score = handedness.score
		
		# Draw label and bounding box
		cv2.putText(frame, f"{label} ({score:.2f}) Gesture: {prediction_gesture}", (min_x, min_y - 10), font_type, 0.7, col, 2)
		cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), col, 2)

	cv2.imshow("Hand Tracking", frame)

	if cv2.waitKey(1) & 0b1111111 == ord('q'):
		break

# Cleanup
cap.release()
cv2.destroyAllWindows()
detector.close()