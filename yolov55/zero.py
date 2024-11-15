#!/usr/bin/env python3
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import cv2 
import numpy as np
import torch

# cap1 = cv2.VideoCapture(0)  # First device
cap2 = cv2.VideoCapture(0)  # Second device
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# loading custom yolov5 model using torch 
model = torch.hub.load('yolov5', 'custom', path='silo.pt', source='local')

# accessing external camera
cap = cv2.VideoCapture(0)

while True:
    print("before")
    width = cap2.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("Current resolution:", width, "x", height)
    ret, frame = cap.read()
    results = model(frame)
    detections = results.pred[0]

    if len(detections):

        print(detections[:])

        # Find the index of the detection with the highest confidence
        highest_confidence_index = np.argmax(detections[:, 4])
        print(highest_confidence_index)

        x1, y1, x2, y2, conf = detections[highest_confidence_index][:5]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Draw a bounding box around the detected object on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame with the detected object
    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()