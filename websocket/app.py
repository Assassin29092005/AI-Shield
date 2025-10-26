import cv2
import numpy as np
import time
from insightface.app import FaceAnalysis
from phone_detector import PhoneDetector
from attention_engine import AttentionEngine

# -------------------------
# CONFIGURATION
# -------------------------
AUTHORIZED_THRESHOLD = 0.4
# We no longer use a time-based interval, we use a frame-count interval
# FACE_CHECK_INTERVAL = 0.5 

# -------------------------
# LOAD MODELS
# -------------------------
face_app = None
phone_detector = None 

try:
    print("Loading InsightFace model (buffalo_s)...")
    face_app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition', 'pose'])
    face_app.prepare(ctx_id=-1)
    print("InsightFace model loaded.")

    print("Loading PhoneDetector model (yolov8n.pt)...")
    phone_detector = PhoneDetector(model_name="yolov8n.pt", conf_thresh=0.25)
    print("PhoneDetector model loaded.")

except Exception as e:
    print(f"FATAL ERROR: Could not load ML models: {e}")

# -------------------------
# ProctoringSession Class
# -------------------------
class ProctoringSession:
    """
    Manages the state and logic for a single proctoring session.
    Uses AttentionEngine for focus scoring.
    """
    
    def __init__(self):
        """
        Initializes all state variables for one user session.
        """
        print(f"[ProctoringSession] New session created.")
        self.engine = AttentionEngine()
        self.ref_embedding = None 
        
        # State tracking
        self.last_faces = []
        self.last_authorized = False
        self.frame_count = 0
        
        # --- NEW COOLDOWN LOGIC ---
        self.ALERT_COOLDOWN = 5.0  # 5 seconds
        self.last_alert_times = {} # Stores last sent time for each event type
        # --- END NEW COOLDOWN LOGIC ---
        
        print("[ProctoringSession] Session initialized.")

    def _can_send_alert(self, event_type: str, now: float) -> bool:
        """
        Checks if the cooldown for a specific alert type has passed.
        If it has, updates the timestamp and returns True.
        """
        last_sent = self.last_alert_times.get(event_type, 0)
        
        if (now - last_sent) >= self.ALERT_COOLDOWN:
            # Cooldown has passed, update the time and allow sending
            self.last_alert_times[event_type] = now
            return True
        
        # Still in cooldown for this specific alert type
        return False

    def register_user_from_frame(self, frame_bytes: bytes) -> tuple[bool, str]:
        """
        Registers a user by creating a reference embedding from their first frame.
        """
        if self.ref_embedding is not None:
            return False, "User is already registered."

        try:
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                return False, "Invalid image data."
                
            faces = face_app.get(frame)
            
            if not faces:
                return False, "No face found in registration frame."
            if len(faces) > 1:
                return False, "Multiple faces found. Only one person allowed."
                
            # Store the embedding
            self.ref_embedding = faces[0].embedding
            self.last_faces = faces # Store for the first frame
            self.last_authorized = True
            print("User registration successful. Reference embedding stored.")
            return True, "User authorized."
                
        except Exception as e:
            print(f"ERROR during registration: {e}")
            return False, "An internal error occurred."

    def process_frame(self, frame_bytes: bytes) -> list[dict]:
        """
        Processes a single frame using the AttentionEngine.
        NOW runs phone detection every frame and face detection every 4th frame.
        """
        try:
            frame = self._decode_frame(frame_bytes)
            if frame is None:
                return [{"event": "ERROR", "message": "Invalid frame data."}]

            now = time.time()
            events = []
            
            # Increment frame count
            self.frame_count += 1
            
            # --- NEW OPTIMIZED LOGIC ---
            
            # By default, use the cached (last known) face values
            faces = self.last_faces
            authorized = self.last_authorized
            
            # --- TASK 1: Run Phone Detection (Every Frame) ---
            phone_present, _ = phone_detector.detect(frame)
            
            # --- TASK 2: Run Face Analysis (Every 4th Frame) ---
            if self.frame_count % 4 == 1:
                faces = face_app.get(frame)
                self.last_faces = faces # Cache the result
                
                if not faces:
                    authorized = False
                elif len(faces) > 1:
                    authorized = False
                    # --- Apply Cooldown Check ---
                    if self._can_send_alert("MULTIPLE_FACES", now):
                        events.append({"event": "MULTIPLE_FACES", "count": len(faces), "timestamp": now})
                else: # Exactly one face
                    sim = np.dot(self.ref_embedding, faces[0].embedding) / (
                        np.linalg.norm(self.ref_embedding) * np.linalg.norm(faces[0].embedding)
                    )
                    if sim <= AUTHORIZED_THRESHOLD:
                        authorized = False
                        # --- Apply Cooldown Check ---
                        if self._can_send_alert("UNAUTHORIZED_PERSON", now):
                            events.append({"event": "UNAUTHORIZED_PERSON", "similarity": float(sim), "timestamp": now})
                    else:
                        authorized = True
                self.last_authorized = authorized # Cache the result
            
            # --- END NEW OPTIMIZED LOGIC ---

            person_count = len(faces)
            face_present = (person_count == 1 and authorized)

            # --- Build 'obs' for AttentionEngine ---
            obs = {
                'face_present': face_present,
                'head_yaw': 0.0,
                'head_pitch': 0.0,
                'gaze_dx': 0.0,
                'gaze_dy': 0.0,
                'phone_present': phone_present # This is updated every frame
            }

            if face_present:
                pose = faces[0].pose
                if pose is not None:
                    obs['head_pitch'] = pose[0]
                    obs['head_yaw'] = pose[1]

            # --- Run the AttentionEngine ---
            engine_results = self.engine.update(obs)
            
            # --- Create Events from Engine Results ---
            
            # ATTENTION_UPDATE is NOT cooled down. It's sent every time.
            events.append({
                "event": "ATTENTION_UPDATE",
                "score": int(engine_results['score']),
                "label": engine_results['label'],
                "timestamp": now
            })
            
            # Check alerts from the engine, applying cooldowns
            for alert_message in engine_results['alerts']:
                if "Face lost" in alert_message:
                    if not any(e['event'] == 'FACE_NOT_FOUND' for e in events) and self._can_send_alert("FACE_NOT_FOUND", now):
                        events.append({"event": "FACE_NOT_FOUND", "timestamp": now})
                
                if "Phone detected" in alert_message:
                    if not any(e['event'] == 'PHONE_DETECTED' for e in events) and self._can_send_alert("PHONE_DETECTED", now):
                        events.append({"event": "PHONE_DETECTED", "timestamp": now})
            
            # --- MODIFIED LOGIC ---
            # Check score from the engine, applying cooldown
            current_score = int(engine_results['score'])
            if current_score < 40:
                if self._can_send_alert("LOOKING_AWAY", now):
                    events.append({"event": "LOOKING_AWAY", "score": current_score, "timestamp": now})
            # --- END MODIFIED LOGIC ---
            
            # Final check for face_present, applying cooldown
            if not face_present and not any(e['event'] == 'FACE_NOT_FOUND' for e in events):
                 if self._can_send_alert("FACE_NOT_FOUND", now):
                    events.append({"event": "FACE_NOT_FOUND", "timestamp": now})

            return events
            
        except Exception as e:
            print(f"ERROR processing frame: {e}")
            import traceback
            traceback.print_exc()
            return [{"event": "ERROR", "message": "An internal error occurred processing the frame."}]
            
    def _decode_frame(self, frame_bytes: bytes) -> np.ndarray | None:
        """Helper to convert raw bytes to a CV2 frame."""
        try:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"Could not decode frame: {e}")
            return None

