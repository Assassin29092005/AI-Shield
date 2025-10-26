# face_gaze_headpose.py
import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

# 3D model points of selected facial landmarks for solvePnP (approx)
# Using nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
# Indices correspond to Mediapipe face mesh landmarks.
# We pick robust points: tip of nose (1), chin (152), left eye outer (33), right eye outer (263), left mouth (61), right mouth (291)
LANDMARKS_PNP = [1, 152, 33, 263, 61, 291]

# 3D model coordinates in mm for the above facial points (approximate generic face model)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],         # nose tip
    [0.0, -63.6, -12.5],     # chin
    [-43.3, 32.7, -26.0],    # left eye corner
    [43.3, 32.7, -26.0],     # right eye corner
    [-28.9, -28.9, -20.0],   # left mouth
    [28.9, -28.9, -20.0],    # right mouth
], dtype=np.float64)

class FaceGazeHeadpose:
    def __init__(self, static_image_mode=False, max_faces=1, refine_landmarks=True):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.last_landmarks = None
        self.ema_head = None
        self.ema_gaze = None

    def process(self, frame):
        """
        Returns: dict with
          - face_present (bool)
          - landmarks (list of (x,y) normalized)
          - head_yaw, head_pitch, head_roll (degrees)
          - gaze (dx, dy) normalized where (0,0) means looking center
          - left_iris_center, right_iris_center (normalized coords)
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.face_mesh.process(img_rgb)
        if not results.multi_face_landmarks:
            return {"face_present": False}

        lm = results.multi_face_landmarks[0].landmark
        # store normalized x,y
        lm_xy = [(p.x, p.y, p.z) for p in lm]
        self.last_landmarks = lm_xy

        # Head pose (POS using solvePnP)
        image_points = []
        for idx in LANDMARKS_PNP:
            x, y = int(lm[idx].x * w), int(lm[idx].y * h)
            image_points.append((x, y))
        image_points = np.array(image_points, dtype=np.float64)

        cam_matrix = np.array([[w, 0, w/2],
                               [0, w, h/2],
                               [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4,1))  # assume no lens distortion

        success, rotation_vec, translation_vec = cv2.solvePnP(
            MODEL_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rotation_vec)
        sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            x_angle = math.atan2(rmat[2,1], rmat[2,2])
            y_angle = math.atan2(-rmat[2,0], sy)
            z_angle = math.atan2(rmat[1,0], rmat[0,0])
        else:
            x_angle = math.atan2(-rmat[1,2], rmat[1,1])
            y_angle = math.atan2(-rmat[2,0], sy)
            z_angle = 0
        # convert to degrees
        pitch = np.degrees(x_angle)   # up / down
        yaw = np.degrees(y_angle)     # left / right
        roll = np.degrees(z_angle)    # tilt

        # Gaze estimation via iris relative position inside eye region
        # Mediapipe iris landmark indices: left iris center ~ 468, right iris center ~ 473
        left_iris_index = 468
        right_iris_index = 473
        left_iris = lm[left_iris_index]
        right_iris = lm[right_iris_index]

        # compute gaze as offset from eye center
        # eye center approximate: average of eye landmarks - for simplicity use specific landmarks
        left_eye_center = np.array([(lm[133].x + lm[33].x)/2, (lm[133].y + lm[33].y)/2])
        right_eye_center = np.array([(lm[362].x + lm[263].x)/2, (lm[362].y + lm[263].y)/2])
        left_iris_center = np.array([left_iris.x, left_iris.y])
        right_iris_center = np.array([right_iris.x, right_iris.y])

        # dx,dy in normalized coords relative to eye center (positive dx -> looking right in image coords)
        left_gaze = left_iris_center - left_eye_center
        right_gaze = right_iris_center - right_eye_center
        # average gaze
        gaze = (left_gaze + right_gaze) / 2.0  # small negative y is up in image coords

        # Normalize/scale gaze for stability
        gaze_norm = np.clip(gaze*10.0, -1.0, 1.0)  # scale factor chosen experimentally

        out = {
            "face_present": True,
            "landmarks": lm_xy,
            "head_yaw": float(yaw),
            "head_pitch": float(pitch),
            "head_roll": float(roll),
            "gaze_dx": float(gaze_norm[0]),
            "gaze_dy": float(gaze_norm[1]),
            "left_iris": (left_iris.x, left_iris.y),
            "right_iris": (right_iris.x, right_iris.y),
        }
        return out
