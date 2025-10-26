# phone_detector.py
from ultralytics import YOLO
import cv2
import numpy as np
import time

# You can use 'yolov8n.pt' (nano) for speed; it detects many classes including 'cell phone' in COCO.
class PhoneDetector:
    def __init__(self, model_name="yolov8n", conf_thresh=0.2):
        # model_name can be 'yolov8n.pt' or 'yolov8n'
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            print("Error loading YOLO model:", e)
            self.model = None
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        """
        Returns list of detections: [{'box':(x1,y1,x2,y2), 'conf':0.9, 'class':class_name}]
        and boolean phone_present.
        """
        if self.model is None:
            return [], False

        # ultralytics returns results; keep only detections with class name 'cell phone' or similar
        results = self.model(frame, imgsz=640, conf=self.conf_thresh, verbose=False)
        dets = []
        phone_present = False
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1,y1,x2,y2 = map(int, xyxy)
                # map class id to name using model.names
                name = self.model.names.get(cls, str(cls))
                dets.append({"box":(x1,y1,x2,y2), "conf":conf, "class":name})
                # detect phone keywords
                if name.lower() in ("cell phone", "cellphone", "phone", "mobile phone", "mobile"):
                    phone_present = True
        return dets, phone_present
