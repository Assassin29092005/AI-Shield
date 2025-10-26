import cv2
import numpy as np
import time
# We use ultralytics for YOLO, ensure it's installed: pip install ultralytics
from ultralytics import YOLO
import os

# A list of keywords to identify as "phone" or "electronic device"
# This makes the detection robust to different model class labels
ELECTRONICS_KEYWORDS = [
    'phone', 'cell', 'mobile', 'laptop', 'tv', 'monitor', 'computer',
    'keyboard', 'mouse', 'remote', 'tablet', 'ipad', 'macbook', 'smartphone'
]

class PhoneDetector:
    """
    A simple, self-contained class for detecting phones and electronics.
    This module is imported by app3.py and knows nothing about the main session.
    """
    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.25):
        """
        Initializes and loads the YOLO model.
        """
        self.model = None
        if not os.path.exists(model_name):
            print(f"[PhoneDetector] WARNING: Model file not found at {model_name}.")
            print("[PhoneDetector] Phone detection will be disabled.")
            return

        try:
            print(f"[PhoneDetector] Loading model {model_name}...")
            self.model = YOLO(model_name)
            self.conf_thresh = conf_thresh
            print("[PhoneDetector] Model loaded successfully.")
        except Exception as e:
            print(f"[PhoneDetector] ERROR: Could not load model {model_name}. {e}")
            print("[PhoneDetector] Please ensure 'ultralytics' is installed and the model file is accessible.")
            self.model = None

    def detect(self, frame: np.ndarray) -> tuple[bool, list]:
        """
        Detects electronic devices in a single frame.

        Args:
            frame: A single cv2 image (numpy array).

        Returns:
            A tuple: (device_detected, bounding_boxes)
            - device_detected (bool): True if any matching device was found.
            - bounding_boxes (list): A list of boxes for detected items.
        """
        if self.model is None:
            return False, []

        device_detected = False
        bounding_boxes = []

        try:
            # Run inference
            results = self.model(frame, conf=self.conf_thresh, verbose=False)
            
            # Process results
            for result in results:
                names = result.names  # Get class names
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = names[class_id].lower().replace('_', '').strip()
                    
                    # --- DEBUG ---
                    # print(f"Detected: {class_name}") 
                    
                    # Check if the detected class is in our keywords
                    if any(k in class_name for k in ELECTRONICS_KEYWORDS):
                        device_detected = True
                        bounding_boxes.append(box.xyxy[0].cpu().numpy().astype(int))
                        
        except Exception as e:
            print(f"[PhoneDetector] ERROR during detection: {e}")

        return device_detected, bounding_boxes

