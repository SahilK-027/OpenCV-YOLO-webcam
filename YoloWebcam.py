# =======================================================================================================
# Dependencies
# =======================================================================================================
from ultralytics import YOLO
import cv2
import cvzone
import math

# =======================================================================================================
# Creating webcam Object
# =======================================================================================================
cap = cv2.VideoCapture(0)
# Setting Width and height of webcam
cap.set(3, 1280)
cap.set(4, 720)

# video as feed
# cap = cv2.VideoCapture('Videos/3.mp4')

# =======================================================================================================
# Creating YOLO model
# =======================================================================================================
model = YOLO('yolov8n.pt')
classnames = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
             'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
             'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
             'ring', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
             'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
             'tomato', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
             'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
             'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
             'teddy bear', 'hair drier', 'toothbrush' ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for i in results:
        boundingBoxes = i.boxes
        for box in boundingBoxes:
            # Drawing bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Drawing confidence value
            conf = math.ceil(box.conf[0] * 100) / 100

            # Drawing class name value
            cls = int(box.cls[0])
            currentClass = classnames[cls]
            # if currentClass == 'knife' and conf > 0.3:
            cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(40, y1)))

            cvzone.cornerRect(img, (x1, y1, w, h))

    cv2.imshow('Image', img)
    cv2.waitKey(1)