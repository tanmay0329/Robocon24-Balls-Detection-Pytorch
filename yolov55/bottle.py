# #!/usr/bin/env python3
# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath
# import torch
# import cv2
# import json

# offset_x = 325
# offset_y = 420
# null = {"data_x": "1000", "data_y": "1000", "detected": "0"}

# # Loading custom YOLOv5 model using torch 
# model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')

# # Accessing the external camera
# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from the camera")
#         break
    
#     results = model(frame)
#     det = results.pred[0]
#     object_detected = False  # Reset flag for each frame
#     error_x = "1000"  # Default value for error_x
#     error_y = "1000"  # Default value for error_y
#     if len(det) == 0:  # If no detections
#         send = json.dumps(null)
#         print(send)
#     else:
#         for *xyxy, conf, cls in reversed(det):
#             if conf > 0.80:
#                 c = int(cls)  # integer class
#                 confidence = float(conf)
#                 error_x = (offset_x - (xyxy[0] + xyxy[2]) / 2) / 8.5
#                 error_y = (offset_y - (xyxy[1] + xyxy[3]) / 2) / 15
#                 object_detected = True  # Set flag to indicate object detection in this frame
        
#         send = {
#             "data_x": str(int(error_x)),
#             "data_y": str(int(error_y)),
#             # "data": 2000,          # Placeholder value for data
#             # "value": 0,            # Placeholder value for value
#             "detected": "1" if object_detected else "0"  # Indicates object detection
#         }
#         json_data = json.dumps(send)
#         print(json_data)

#     cv2.imshow("Camera Feed", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import RPi.GPIO as GPIO
import time
import pathlib
import torch
import cv2
import json

offset_x = 325
offset_y = 420
null = {"data_x": "1000", "data_y": "1000", "detected": "0"}

# Loading custom YOLOv5 model using torch 
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')

# GPIO pin for controlling the motor
motor_pin = 18

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(motor_pin, GPIO.OUT)

# Function to control the motor
def control_motor(detected):
    if detected:
        print("Object Detected - Start Motor")
        GPIO.output(motor_pin, GPIO.HIGH)  # Start the motor
    else:
        print("No Object Detected - Stop Motor")
        GPIO.output(motor_pin, GPIO.LOW)  # Stop the motor

# Accessing the external camera
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from the camera")
        break
    
    results = model(frame)
    det = results.pred[0]
    object_detected = False  # Reset flag for each frame
    error_x = "1000"  # Default value for error_x
    error_y = "1000"  # Default value for error_y
    if len(det) == 0:  # If no detections
        send = json.dumps(null)
        print(send)
    else:
        for *xyxy, conf, cls in reversed(det):
            if conf > 0.80:
                c = int(cls)  # integer class
                confidence = float(conf)
                error_x = (offset_x - (xyxy[0] + xyxy[2]) / 2) / 8.5
                error_y = (offset_y - (xyxy[1] + xyxy[3]) / 2) / 15
                object_detected = True  # Set flag to indicate object detection in this frame
        
        send = {
            "data_x": str(int(error_x)),
            "data_y": str(int(error_y)),
            # "data": 2000,          # Placeholder value for data
            # "value": 0,            # Placeholder value for value
            "detected": "1" if object_detected else "0"  # Indicates object detection
        }
        json_data = json.dumps(send)
        print(json_data)

    # Control the motor based on object detection
    control_motor(object_detected)

    cv2.imshow("Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup GPIO
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
