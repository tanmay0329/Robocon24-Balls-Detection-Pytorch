#!/usr/bin/env python3
import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import torch
import cv2
import json

offset_x = 325
offset_y = 420
paddy_rice = 0
empty_grain = 1
null = {"data_x": 1000, "data_y": 1000, "data": 1000, "value": 1}

# Loading custom YOLOv5 model using torch 
model = torch.hub.load('yolov5', 'custom', path='silo_best.pt', source='local')
# def open_camera():
#     global cap
#     cap = cv2.VideoCapture('/dev/lenovo')
#     cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# Accessing the external camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from the camera")
        break
    
    results = model(frame)
    det = results.pred[0]
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
                send = {
                    "data_x": str(int(error_x)),
                    "data_y": str(int(error_y)),
                    "data": 2000,
                    "value": 0,
                    "cls": str(c),
                    "nearest": -1 if c == paddy_rice else -2
                }
                json_data = json.dumps(send)
                print(json_data)

        cv2.imshow("Camera Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# #!/usr/bin/env python3
# import rospy
# from sensor_msgs.msg import Image
# from std_msgs.msg import String
# import cv2
# import json
# import pathlib
# import torch

# offset_x = 325
# offset_y = 420
# paddy_rice = 0
# empty_grain = 1
# null = {"data_x": 1000, "data_y": 1000, "data": 1000, "value": 1}

# model = torch.hub.load('yolov5', 'custom', path='blue.pt', source='local')
# cap = cv2.VideoCapture(1)
# rospy.init_node('object_detector_publisher')
# pub = rospy.Publisher('detected_objects', String, queue_size=10)

# while not rospy.is_shutdown():
#     ret, frame = cap.read()
#     if not ret:
#         rospy.logerr("Failed to capture frame from the camera")
#         break
    
#     results = model(frame)
#     det = results.pred[0]
#     if len(det) == 0:
#         send = json.dumps(null)
#         pub.publish(send)
#     else:
#         for *xyxy, conf, cls in reversed(det):
#             if conf > 0.80:
#                 c = int(cls)  
                
#                 error_x = (offset_x - (xyxy[0] + xyxy[2]) / 2) / 8.5
#                 error_y = (offset_y - (xyxy[1] + xyxy[3]) / 2) / 15
#                 send = {
#                     "data_x": str(int(error_x)),
#                     "data_y": str(int(error_y)),
#                     "data": 2000,
#                     "value": 0,
#                     "cls": str(c),
#                     "nearest": -1 if c == paddy_rice else -2
#                 }
#                 json_data = json.dumps(send)
#                 pub.publish(json_data)

#         cv2.imshow("Camera Feed", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


