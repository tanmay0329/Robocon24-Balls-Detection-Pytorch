#!/usr/bin/env python3
import argparse
import os
import platform
import sys
#import serial
#import time
import json
from pathlib import Path
import numpy as np
import torch
import cv2
import json
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool
# import pyautogui
nodeName='silo_detector'
topicName='/sending_to_lidar'

flag=False
publisher2=rospy.Publisher(topicName,String,queue_size=5)

def flag_changer(data):
    global flag
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

#ser = serial.Serial('/dev/ttyTHS1', 9600) 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
null={"data_x":1000,"data_y":1000,"data_silo":1000,"silo_number":0,"value":3}
json_null=json.dumps(null)


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)  
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            global flag
#code for bounding box
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, 1] = det[:, 1]   # Increase y1 by 50
                det[:, 3] = det[:, 3] + 20
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                 # Increase the size of the bounding box from the lower side
                
                       # Check for the presence of yellow color within the modified bounding box
                 # You need to define the create_yellow_mask function
#code for priority
                empty = 20
                none = 10
                one_red_ball_silo = 40
                red_domination_silo = 60
                one_white_ball_silo = 30
                white_domination_silo = 50
               


                variables = [empty,one_red_ball_silo, red_domination_silo, one_white_ball_silo, white_domination_silo, ]
                silo = [-1]
                variable_set = {"None":-1,"empty":empty, "red ball silo":one_red_ball_silo, "white ball silo":one_white_ball_silo, "red domination silo":red_domination_silo, "white domination silo":white_domination_silo}

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    
                    maximum = empty


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
                    
                print("SILO  - ",silo)
                prime = max(silo)
                key = list(filter(lambda x: variable_set[x] == prime, variable_set))[0]
                # print(f"{key}")
                # print(f"{max(silo)}")


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    

                    # if int(cls) == variable_set[key]:
                    #     center_x = (xyxy[0] + xyxy[2]) / 2
                    #     center_y = (xyxy[1] + xyxy[3]) / 2
                    #     print(f"Center coordinates of class {key}: ({center_x}, {center_y})")
                    #     error=320-center_x
                    #     print(f"error={error}")
#code for centre
                    if int(cls)==variable_set[key]:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    center_y = (xyxy[1] + xyxy[3]) / 2
                    yellow_mask = create_yellow_mask(imc, xyxy[0],xyxy[1],xyxy[2],xyxy[3]+50)
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
                        publisher2.publish(json_data)
                        flag=False
                    else:
                        
                        # print(json_null.encode('utf-8'))
                        # publisher2(json_null)
                        pass

                    # print(f"{int(error)}")
                    
                    # if(center_x>320):
                    #     print(f"right")
                    # if(center_x<320):
                    #     print(f"left")
                    

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        # if label=='red_ball':

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            elif flag==True:
                print(json_null.encode('utf-8'))
                publisher2.publish(json_null)
                flag=False
            else:
                print("in else")
                pass
    

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'blueball.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / '/dev/video0', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == '__main__':
    send=0.0
    rospy.init_node(nodeName,anonymous=True)
    rospy.Subscriber("/checker",Bool,flag_changer)
    #pub = rospy.Publisher('chatter', String, queue_size=10)
    #rospy.init_node('talker3', anonymous=True)
    try:
        opt = parse_opt()
        main(opt)
        #talker()
    except rospy.ROSInterruptException:
        pass































# #!/usr/bin/env python3
# import argparse
# import os
# import platform
# import sys
# #import serial
# #import time
# import json
# from pathlib import Path
# import numpy as np
# import torch
# import cv2
# import json
# import rospy
# from std_msgs.msg import String
# # import pyautogui
# nodeName='messagepublisherofsilo'
# topicName='information'
# rospy.init_node(nodeName,anonymous=True)

# publisher2=rospy.Publisher(topicName,String,queue_size=5)

# def create_yellow_mask(image, x1, y1, x2, y3):
#     x1 = int(x1)
#     y1 = int(y1)
#     x2 = int(x2)
#     y2 = int(y3)
    
#     cropped_image = image[y1:y2, x1:x2]
    
#     if cropped_image is not None and not cropped_image.size == 0:  
#         hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        
#         lower_yellow = np.array([20, 100, 100], dtype=np.uint8) 
#         upper_yellow = np.array([30, 255, 255], dtype=np.uint8)  
#         yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
        
#         return yellow_mask
#     else:
#         print("Failed to crop the image or cropped image is empty.")

# error_x=0
# center_x=0
# error_y=0
# center_y=0
# #ser = serial.Serial('/dev/ttyTHS1', 9600) 
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from ultralytics.utils.plotting import Annotator, colors, save_one_box

# from models.common import DetectMultiBackend
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
# from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
#                            increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# from utils.torch_utils import select_device, smart_inference_mode
# null={"data_x":1000,"data_y":1000,"data_silo":1000}
# json_null=json.dumps(null)


# @smart_inference_mode()
# def run(
#         weights=ROOT / 'yolov5s.pt',  # model path or triton URL
#         source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
#         data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
#         imgsz=(640, 640),  # inference size (height, width)
#         conf_thres=0.25,  # confidence threshold
#         iou_thres=0.45,  # NMS IOU threshold
#         max_det=1000,  # maximum detections per image
#         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
#         view_img=False,  # show results
#         save_txt=False,  # save results to *.txt
#         save_conf=False,  # save confidences in --save-txt labels
#         save_crop=False,  # save cropped prediction boxes
#         nosave=False,  # do not save images/videos
#         classes=None,  # filter by class: --class 0, or --class 0 2 3
#         agnostic_nms=False,  # class-agnostic NMS
#         augment=False,  # augmented inference
#         visualize=False,  # visualize features
#         update=False,  # update all models
#         project=ROOT / 'runs/detect',  # save results to project/name
#         name='exp',  # save results to project/name
#         exist_ok=False,  # existing project/name ok, do not increment
#         line_thickness=3,  # bounding box thickness (pixels)
#         hide_labels=False,  # hide labels
#         hide_conf=False,  # hide confidences
#         half=False,  # use FP16 half-precision inference
#         dnn=False,  # use OpenCV DNN for ONNX inference
#         vid_stride=1,  # video frame-rate stride
# ):
#     source = str(source)
#     save_img = not nosave and not source.endswith('.txt')  # save inference images
#     is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
#     is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
#     webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
#     screenshot = source.lower().startswith('screen')
#     if is_url and is_file:
#         source = check_file(source)  # download

#     # Directories
#     save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
#     (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

#     # Load model
#     device = select_device(device)
#     model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
#     stride, names, pt = model.stride, model.names, model.pt
#     imgsz = check_img_size(imgsz, s=stride)  # check image size

#     # Dataloader
#     bs = 1  # batch_size
#     if webcam:
#         view_img = check_imshow(warn=True)
#         dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#         bs = len(dataset)
#     elif screenshot:
#         dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
#     else:
#         dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
#     vid_path, vid_writer = [None] * bs, [None] * bs

#     # Run inference
#     model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
#     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
#     for path, im, im0s, vid_cap, s in dataset:
#         with dt[0]:
#             im = torch.from_numpy(im).to(model.device)  
#             im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
#             im /= 255  # 0 - 255 to 0.0 - 1.0
#             if len(im.shape) == 3:
#                 im = im[None]  # expand for batch dim

#         # Inference
#         with dt[1]:
#             visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
#             pred = model(im, augment=augment, visualize=visualize)

#         # NMS
#         with dt[2]:
#             pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

#         # Second-stage classifier (optional)
#         # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

#         # Process predictions
#         for i, det in enumerate(pred):  # per image
#             seen += 1
#             if webcam:  # batch_size >= 1
#                 p, im0, frame = path[i], im0s[i].copy(), dataset.count
#                 s += f'{i}: '
#             else:
#                 p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

#             p = Path(p)  # to Path
#             save_path = str(save_dir / p.name)  # im.jpg
#             txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
#             s += '%gx%g ' % im.shape[2:]  # print string
#             gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             imc = im0.copy() if save_crop else im0  # for save_crop
#             annotator = Annotator(im0, line_width=line_thickness, example=str(names))
# #code for bounding box
#             if len(det):
#                 # Rescale boxes from img_size to im0 size
#                 det[:, 1] = det[:, 1]   # Increase y1 by 50
#                 det[:, 3] = det[:, 3] + 20
#                 det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
#                  # Increase the size of the bounding box from the lower side
                
#                        # Check for the presence of yellow color within the modified bounding box
#                  # You need to define the create_yellow_mask function
# #code for priority
#                 empty = 20
#                 none = 10
#                 one_red_ball_silo = 40
#                 red_domination_silo = 60
#                 one_white_ball_silo = 30
#                 white_domination_silo = 50
               


#                 variables = [empty,one_red_ball_silo, red_domination_silo, one_white_ball_silo, white_domination_silo, ]
#                 silo = [-1]
#                 variable_set = {"None":-1,"empty":empty, "red ball silo":one_red_ball_silo, "white ball silo":one_white_ball_silo, "red domination silo":red_domination_silo, "white domination silo":white_domination_silo}

#                 # Print results
#                 for c in det[:, 5].unique():
#                     n = (det[:, 5] == c).sum()  # detections per class
#                     s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    
#                     maximum = empty


#                     if int(c) == 0:
#                         print(f"Empty silo")
#                         silo.append(empty)
#                     if int(c) == 1:
#                         print(f"None")
#                         silo.append(none)
#                     if int(c) == 2:
#                         print(f"red ball silo")
#                         silo.append(one_red_ball_silo)
#                     if int(c) == 3:
#                         print(f"red domination silo")
#                         silo.append(red_domination_silo)
#                     if int(c) == 4:
#                         print(f"white ball silo")
#                         silo.append(one_white_ball_silo)
#                     if int(c) == 5:
#                         print(f"white domination silo")
#                         silo.append(white_domination_silo)
                    
#                 print("SILO  - ",silo)
#                 prime = max(silo)
#                 key = list(filter(lambda x: variable_set[x] == prime, variable_set))[0]
#                 # print(f"{key}")
#                 # print(f"{max(silo)}")


#                 # Write results
#                 for *xyxy, conf, cls in reversed(det):
                    

#                     # if int(cls) == variable_set[key]:
#                     #     center_x = (xyxy[0] + xyxy[2]) / 2
#                     #     center_y = (xyxy[1] + xyxy[3]) / 2
#                     #     print(f"Center coordinates of class {key}: ({center_x}, {center_y})")
#                     #     error=320-center_x
#                     #     print(f"error={error}")
# #code for centre
#                     if int(cls)==variable_set[key]:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f'{txt_path}.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#                     center_x = (xyxy[0] + xyxy[2]) / 2
#                     center_y = (xyxy[1] + xyxy[3]) / 2
#                     yellow_mask = create_yellow_mask(imc, xyxy[0],xyxy[1],xyxy[2],xyxy[3]+50)
#                     # print(center_x,center_y)
#                     error_x=(320-center_x)/8.5
#                     error_y=(240-center_y)/8.5
#                     silo_detected=3000
#                     silo_detected=str(int(silo_detected))
#                     no_silo=1000
#                     no_silo=str(int(no_silo))
#                     data_x=str(int(error_x))
#                     data_y=str(int(error_y))
# #json communication code
#                     send = {
#                     "data_x":data_x,"data_y":data_y,"data_silo":silo_detected
        
#                     }
#                     json_data = json.dumps(send)
#                     # null={
#                     #     "data_x":1000,"data_y":1000,"data":1000
#                     # }
#                     # json_null=json.dumps(null)
#                     yellow_present = cv2.countNonZero(yellow_mask) 
#                     # print(yellow_present)
#                     if yellow_present>0:
                        
#                         print(json_data.encode('utf-8'))
#                         publisher2.publish(json_data)
#                     else:
                        
#                         # print(json_null.encode('utf-8'))
#                         # publisher2(json_null)
#                         pass

#                     # print(f"{int(error)}")
                    
#                     # if(center_x>320):
#                     #     print(f"right")
#                     # if(center_x<320):
#                     #     print(f"left")
                    

#                     if save_txt:  # Write to file
#                         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
#                         with open(f'{txt_path}.txt', 'a') as f:
#                             f.write(('%g ' * len(line)).rstrip() % line + '\n')
#                     if save_img or save_crop or view_img:  # Add bbox to image
#                         c = int(cls)  # integer class
#                         label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
#                         annotator.box_label(xyxy, label, color=colors(c, True))
#                         # if label=='red_ball':

#                     if save_crop:
#                         save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
#             else:
#                 print(json_null.encode('utf-8'))
#                 publisher2.publish(json_null)
    

#             # Stream results
#             im0 = annotator.result()
#             if view_img:
#                 if platform.system() == 'Linux' and p not in windows:
#                     windows.append(p)
#                     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
#                 cv2.imshow(str(p), im0)
#                 cv2.waitKey(1)  # 1 millisecond

#             # Save results (image with detections)
#             if save_img:
#                 if dataset.mode == 'image':
#                     cv2.imwrite(save_path, im0)
#                 else:  # 'video' or 'stream'
#                     if vid_path[i] != save_path:  # new video
#                         vid_path[i] = save_path
#                         if isinstance(vid_writer[i], cv2.VideoWriter):
#                             vid_writer[i].release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, im0.shape[1], im0.shape[0]
#                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#                         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer[i].write(im0)

#         # Print time (inference-only)
#         # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

#     # Print results
#     t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
#     LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
#     if save_txt or save_img:
#         s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
#         LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
#     if update:
#         strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '300_epochs.pt', help='model path or triton URL')
#     parser.add_argument('--source', type=str, default=ROOT / '/dev/video0', help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.65, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))

# if __name__ == '__main__':
#     send=0.0
#     #pub = rospy.Publisher('chatter', String, queue_size=10)
#     #rospy.init_node('talker3', anonymous=True)
#     try:
#         opt = parse_opt()
#         main(opt)
#         #talker()
#     except rospy.ROSInterruptException:
#         pass
