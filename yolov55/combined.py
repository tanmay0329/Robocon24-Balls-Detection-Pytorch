import pathlib
pathlib.PosixPath = pathlib.WindowsPath
import torch
import cv2
import json
import numpy as np
import threading

flag=True
def flag_changer(data):
    
    flag=data.data
    print(flag)

def create_yellow_mask(image, x1, y1, x2, y3):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y3)

    cropped_image = image[y1:y2, x1:x2]
    
    if cropped_image is not None and not cropped_image.size == 0:  
        hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([20, 100, 100], dtype=np.uint8) 
        upper_yellow = np.array([30, 255, 255], dtype=np.uint8)  
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        
        return yellow_mask
    else:
        print("Failed to crop the image or cropped image is empty.")

error_x=0
center_x=0
error_y=0
center_y=0
empty = 20
none = 10
one_red_ball_silo = 40
red_domination_silo = 60
one_white_ball_silo = 30
white_domination_silo = 50
variables = [empty,one_red_ball_silo, red_domination_silo, one_white_ball_silo, white_domination_silo, ]
silo = [-1]
variable_set = {"None":-1,"empty":empty, "red ball silo":one_red_ball_silo, "white ball silo":one_white_ball_silo, "red domination silo":red_domination_silo, "white domination silo":white_domination_silo} 
maximum = empty

# print(f"{key}")
# print(f"{max(silo)}")
model = torch.hub.load('yolov5', 'custom', path='blue.pt', source='local')
      
offset_x = 325
offset_y = 420
paddy_rice = 0
empty_grain = 1
null = {"data_x": 1000, "data_y": 1000, "data": 1000, "value": 1}

# Accessing the external camera
cap1 = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)
print("SILO  - ",silo)
prime = max(silo)
def siloDetection(cap):
    while True:
        print("IN SILO FUNC")
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from the camera")
            break
        
        results = model(frame)
        det = results.pred[0]
        for *xyxy, conf, cls in reversed(det):
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class                    
                                            
                            if int(c) == 0:
                                print(f"Empty silo")
                                silo.append(empty)
                            if int(c) == 1:
                                print(f"None")
                                silo.append(none)
                            if int(c) == 2:
                                print(f"red ball silo")
                                silo.append(one_red_ball_silo)
                            if int(c) == 3:
                                print(f"red domination silo")
                                silo.append(red_domination_silo)
                            if int(c) == 4:
                                print(f"white ball silo")
                                silo.append(one_white_ball_silo)
                            if int(c) == 5:
                                print(f"white domination silo")
                                silo.append(white_domination_silo)


                        key = list(filter(lambda x: variable_set[x] == prime, variable_set))[0]
                        # print(f"{key}")
                        # print(f"{max(silo)}")
                        center_x = (xyxy[0] + xyxy[2]) / 2
                        center_y = (xyxy[1] + xyxy[3]) / 2
                        
                        yellow_mask = create_yellow_mask(frame, xyxy[0],xyxy[1],xyxy[2],xyxy[3]+50)
                        # print(center_x,center_y)
                        error_x=(320-center_x)/8.5
                        error_y=(240-center_y)/8.5
                        silo_detected=3000
                        silo_detected=str(int(silo_detected))
                        no_silo=1000
                        if(-37<error_x<-25):
                            silo_number=1
                        if(-25<=error_x<-10):
                            silo_number=2
                        if(-10<=error_x<10):
                            silo_number=3
                        if(10<=error_x<25):
                            silo_number=4
                        if(error_x>=25 ):
                            silo_number=5
                        no_silo=str(int(no_silo))
                        data_x=str(int(error_x))
                        data_y=str(int(error_y))
                        #json communication code
                        send = {
                        "data_x":data_x,"data_y":data_y,"data_silo":silo_detected,"silo_number":silo_number,"value":2
            
                        }
                        json_data = json.dumps(send)
                        # null={
                        #     "data_x":1000,"data_y":1000,"data":1000
                        # }
                        # json_null=json.dumps(null)
                        yellow_present = cv2.countNonZero(yellow_mask) 
                        # print(yellow_present)
                        if yellow_present>0  and flag==True:
                            
                            print(json_data.encode('utf-8'))
                            # publisher2.publish(json_data)
                            flag=False
                        else:
                            
                            # print(json_null.encode('utf-8'))
                            # publisher2(json_null)
                            pass


        cv2.imshow("Camera Feed for Silo", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        



def ballDetection(cap1):
    while True:
        print("IN BALL FUNC")
        ret, frame = cap1.read()
        if not ret:
            print("Failed to capture frame from the camera")
            break
        
        results = model(frame)
        det = results.pred[0]
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)  # integer class
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
            print(f"Class: {c}, JSON: {json_data}")
            
            # Print red ball distance if paddy_rice class
            if c == paddy_rice:
                print("Red ball distance:", ((error_x)**2 + (error_y)**2) ** 0.5)
            # Print blue ball distance if empty_grain class
            elif c == empty_grain:
                print("Blue ball distance:", ((error_x)**2 + (error_y)**2) ** 0.5)

        cv2.imshow("Camera Feed for Ball", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


t1 = threading.Thread(target=siloDetection, args=(cap,))
t2 = threading.Thread(target=ballDetection, args=(cap1,))
t1.start()
t2.start()
try:
    t1.join()
    t2.join()
finally:
    cap.release()
    cap1.release()
    cv2.destroyAllWindows()