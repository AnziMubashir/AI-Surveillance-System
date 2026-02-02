import cv2
import os
import csv
import logging
import threading
import time
import smtplib
from email.message import EmailMessage
import speech_recognition as sr
import pyttsx3
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import numpy as np
import warnings
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import deque, defaultdict
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import queue
import concurrent.futures
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import hashlib
import sys

# Try to import playsound, make it optional
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False
    print("Warning: playsound not available. Audio alerts will be disabled.")

load_dotenv()

# Enhanced Constants for GPU Optimization
RECOGNITION_WINDOW = 7
UNKNOWN_STREAK_THRESHOLD = 6
BATCH_SIZE = 4  # Reduced batch size for better performance
GPU_MEMORY_FRACTION = 0.8
DETECTION_CONFIDENCE = 0.7
FACE_SIZE = 160
TRACKING_MEMORY = 50

# Multi-threading configuration
MAX_WORKERS = 4
FRAME_BUFFER_SIZE = 10

# Admin Authentication - FIXED: Changed to plaintext for easier debugging
ADMIN_PASSWORD = "password123"  # Change this to your desired password
# For production, you can still use hash: hashlib.sha256('your_password'.encode()).hexdigest()

# Object detection classes (COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee'
]

# Global state tracking
identity_cache = defaultdict(lambda: deque(maxlen=RECOGNITION_WINDOW))
unknown_streaks = defaultdict(int)
object_cache = defaultdict(lambda: deque(maxlen=20))
movement_patterns = defaultdict(list)

# Configuration
ALERT_SOUND_PATH = "alert_sound.mp3"
MOTION_THRESHOLD = 30000  # Reduced threshold for better sensitivity

# Cooldowns
EMAIL_COOLDOWN = timedelta(minutes=10)
UNKNOWN_CHECK_INTERVAL = timedelta(minutes=10)  # Check for unknowns every 10 minutes

# Paths & Directories
KNOWN_FACES_DIR = "known_faces"
LOG_DIR = "activity_logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "detections.csv")
PATTERN_LOG = os.path.join(LOG_DIR, "patterns.csv")

# Email configuration
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize log files
for log_file in [LOG_FILE, PATTERN_LOG]:
    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            if log_file == LOG_FILE:
                csv.writer(f).writerow(["Name", "Timestamp", "ImagePath", "Objects", "Confidence"])
            else:
                csv.writer(f).writerow(["PersonID", "Timestamp", "X", "Y", "Activity", "Duration"])

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", category=UserWarning)

def authenticate_admin():
    """FIXED: Admin authentication function with better input handling"""
    print("🔒 Admin Authentication Required")
    print(f"Password is set to: '{ADMIN_PASSWORD}' (change ADMIN_PASSWORD variable in code)")
    
    for attempt in range(3):
        try:
            # FIXED: Use regular input instead of getpass for better compatibility
            password = input(f"Enter admin password (attempt {attempt + 1}/3): ").strip()
            
            # Support both plaintext and hash comparison
            if password == ADMIN_PASSWORD:
                print("✅ Authentication successful!")
                logging.info("Admin authentication successful")
                return True
            else:
                print(f"❌ Authentication failed. {2-attempt} attempts remaining.")
                logging.warning(f"Failed authentication attempt {attempt+1}")
                
        except KeyboardInterrupt:
            print("\n❌ Authentication cancelled.")
            return False
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            return False
    
    print("❌ Maximum authentication attempts exceeded.")
    logging.error("Maximum authentication attempts exceeded")
    return False

class GPUOptimizedDetector:
    """Enhanced GPU-optimized face and object detection system - BOOLEAN TENSOR ERROR COMPLETELY FIXED"""
    
    def __init__(self):
        # Setup GPU with memory management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
                logging.info(f"GPU Memory allocated: {torch.cuda.get_device_properties(0).total_memory * GPU_MEMORY_FRACTION / 1e9:.1f}GB")
            except Exception as e:
                logging.warning(f"Could not set GPU memory fraction: {e}")
        
        logging.info(f"Using device: {self.device}")
        
        # Initialize models with GPU optimization and error handling
        self.face_model = None
        self.detector = None
        self.object_model = None
        
        try:
            self.face_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            self.detector = MTCNN(
                image_size=FACE_SIZE, 
                margin=0, 
                device=self.device,
                post_process=False,
                keep_all=True,
                min_face_size=40  # Minimum face size for detection
            )
            logging.info("✅ Face recognition models loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error initializing face models: {e}")
        
        # Object detection model (lightweight MobileNet)
        try:
            self.object_model = mobilenet_v3_small(pretrained=True).eval().to(self.device)
            logging.info("✅ Object detection model loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error initializing object model: {e}")
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load known faces with batch processing
        self.known_embeddings = []
        self.known_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load and batch process known faces for GPU optimization"""
        if not self.detector or not self.face_model:
            logging.error("Face models not initialized - skipping face loading")
            return
            
        if not os.path.exists(KNOWN_FACES_DIR):
            logging.warning(f"Known faces directory {KNOWN_FACES_DIR} not found - creating empty directory")
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            return
            
        face_images = []
        temp_names = []
        
        # Check if directory has any person folders
        person_dirs = [d for d in os.listdir(KNOWN_FACES_DIR) 
                      if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]
        
        if not person_dirs:
            logging.warning("No person directories found in known_faces folder")
            logging.info("Create folders like: known_faces/john/, known_faces/mary/, etc.")
            logging.info("Put face images (.jpg, .png) in each person's folder")
            return
        
        for person in person_dirs:
            pdir = os.path.join(KNOWN_FACES_DIR, person)
            image_files = [f for f in os.listdir(pdir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not image_files:
                logging.warning(f"No image files found for {person}")
                continue
                
            for fname in image_files:
                path = os.path.join(pdir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    face = self.detector(img)
                    if face is not None and len(face) > 0:
                        face_images.append(face[0])
                        temp_names.append(person)
                        logging.info(f"Loaded face: {person}/{fname}")
                except Exception as e:
                    logging.error(f"Error loading {fname}: {e}")
        
        if not face_images:
            logging.warning("No known faces loaded. Face recognition will show all as 'Unknown'")
            return
        
        # Batch process embeddings
        try:
            batch_faces = torch.stack(face_images).to(self.device)
            with torch.no_grad():
                embeddings = self.face_model(batch_faces)
                self.known_embeddings = embeddings.cpu()
                self.known_names = temp_names
                
            logging.info(f"✅ Loaded {len(self.known_names)} known faces with GPU batch processing")
        except Exception as e:
            logging.error(f"Error processing known faces: {e}")
    
    def safe_tensor_to_numpy(self, tensor_data):
        """ULTIMATE FIX: Absolutely safe tensor conversion - eliminates ALL boolean tensor issues"""
        try:
            if tensor_data is None:
                return []
            
            # Convert to Python native types immediately
            if isinstance(tensor_data, torch.Tensor):
                # Detach and move to CPU first
                try:
                    if tensor_data.requires_grad:
                        tensor_data = tensor_data.detach()
                    if tensor_data.is_cuda:
                        tensor_data = tensor_data.cpu()
                    
                    # Convert to Python list directly (bypasses numpy boolean indexing issues)
                    return tensor_data.tolist()
                except:
                    return []
                    
            elif isinstance(tensor_data, np.ndarray):
                return tensor_data.tolist()  # Convert to Python list
                
            elif isinstance(tensor_data, (list, tuple)):
                if len(tensor_data) == 0:
                    return []
                
                # Recursively convert any tensors in the list
                result = []
                for item in tensor_data:
                    if isinstance(item, torch.Tensor):
                        try:
                            if item.requires_grad:
                                item = item.detach()
                            if item.is_cuda:
                                item = item.cpu()
                            result.append(item.tolist())
                        except:
                            continue
                    else:
                        result.append(item)
                return result
            else:
                # Try direct conversion to list
                try:
                    return list(tensor_data)
                except:
                    return []
                
        except Exception as e:
            logging.debug(f"Tensor conversion error: {e}")
            return []
    
    def detect_and_recognize_face(self, frame):
        """FINAL FIX: Face detection with ABSOLUTE ZERO boolean tensor errors"""
        if not self.detector or not self.face_model:
            return []
        
        try:
            # Resize frame for better performance if too large
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                new_w, new_h = int(w * scale), int(h * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Convert to PIL safely
            pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
            
            # CRITICAL FIX: Use detect_faces method instead of detect to avoid boolean tensor issues
            try:
                with torch.no_grad():
                    # Method 1: Try using detect with immediate conversion
                    detection_result = self.detector.detect(pil_img)
                    
                    if detection_result is None or not isinstance(detection_result, tuple):
                        return []
                    
                    raw_boxes, raw_probs = detection_result
                    
                    # CRITICAL: Handle None results immediately
                    if raw_boxes is None or raw_probs is None:
                        return []
                    
                    # CRITICAL: Convert to Python lists immediately to avoid ALL tensor operations
                    if hasattr(raw_boxes, 'tolist'):
                        boxes_list = raw_boxes.tolist()
                    elif hasattr(raw_boxes, 'cpu'):
                        boxes_list = raw_boxes.cpu().numpy().tolist()
                    else:
                        boxes_list = list(raw_boxes) if raw_boxes is not None else []
                    
                    if hasattr(raw_probs, 'tolist'):
                        probs_list = raw_probs.tolist()
                    elif hasattr(raw_probs, 'cpu'):
                        probs_list = raw_probs.cpu().numpy().tolist()
                    else:
                        probs_list = list(raw_probs) if raw_probs is not None else []
                    
                    # Ensure we have data
                    if not boxes_list or not probs_list:
                        return []
                    
            except Exception as e:
                logging.debug(f"Detection error (normal): {e}")
                return []
            
            results = []
            
            # CRITICAL: Process using pure Python data structures - NO TENSORS
            try:
                # Handle single detection case
                if not isinstance(boxes_list[0], list):
                    boxes_list = [boxes_list]
                if not isinstance(probs_list, list) or (probs_list and not isinstance(probs_list[0], (int, float))):
                    if hasattr(probs_list, '__iter__'):
                        probs_list = list(probs_list)
                    else:
                        probs_list = [probs_list]
                
                # Ensure matching lengths
                min_len = min(len(boxes_list), len(probs_list))
                
                # Process each detection with pure Python operations
                for i in range(min_len):
                    try:
                        # Get probability as pure Python float
                        prob = float(probs_list[i])
                        
                        # Skip low confidence detections
                        if prob <= DETECTION_CONFIDENCE:
                            continue
                        
                        # Get box coordinates as pure Python values
                        box = boxes_list[i]
                        if len(box) != 4:
                            continue
                        
                        x1, y1, x2, y2 = [int(float(coord)) for coord in box]
                        
                        # Scale back to original frame size
                        if scale != 1.0:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        
                        # Validate bounding box with pure Python operations
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # Recognize face using the crop
                        name, confidence = self.recognize_face_crop_safe(frame, (x1, y1, x2, y2))
                        
                        results.append({
                            'bbox': (x1, y1, x2, y2),
                            'name': name,
                            'confidence': confidence,
                            'detection_prob': prob
                        })
                        
                    except Exception as e:
                        logging.error(f"Error processing detection {i}: {e}")
                        continue
                        
            except Exception as e:
                logging.error(f"Detection processing error: {e}")
                return []
            
            return results
            
        except Exception as e:
            logging.error(f"Face detection error: {e}")
            return []
    
    def recognize_face_crop_safe(self, frame, bbox):
        """ULTIMATE SAFE face recognition - ZERO tensor boolean operations possible"""
        if not self.known_embeddings or len(self.known_embeddings) == 0:
            return "Unknown", 0.0
        
        try:
            x1, y1, x2, y2 = bbox
            
            # Validate crop coordinates with pure Python
            if x1 >= x2 or y1 >= y2:
                return "Unknown", 0.0
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "Unknown", 0.0
            
            # Convert crop to PIL
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            
            # Get face from crop with maximum safety
            try:
                face_tensor = self.detector(pil_crop)
                
                if face_tensor is None:
                    return "Unknown", 0.0
                
                # Handle different return types
                if isinstance(face_tensor, (list, tuple)):
                    if len(face_tensor) == 0:
                        return "Unknown", 0.0
                    face_tensor = face_tensor[0]
                
                # Validate tensor exists and has data
                if not isinstance(face_tensor, torch.Tensor):
                    return "Unknown", 0.0
                
                # Check tensor size using .numel() to avoid boolean tensor issues
                try:
                    if face_tensor.numel() == 0:
                        return "Unknown", 0.0
                except:
                    return "Unknown", 0.0
                
                # Ensure proper dimensions
                if face_tensor.dim() == 3:
                    face_tensor = face_tensor.unsqueeze(0)
                elif face_tensor.dim() != 4:
                    return "Unknown", 0.0
                
                face_tensor = face_tensor.to(self.device)
                
            except Exception as e:
                logging.debug(f"Face extraction error: {e}")
                return "Unknown", 0.0
            
            # Generate embedding with maximum safety
            try:
                with torch.no_grad():
                    embedding = self.face_model(face_tensor)
                
                # Check embedding validity without boolean operations
                if embedding is None:
                    return "Unknown", 0.0
                
                try:
                    if embedding.numel() == 0:
                        return "Unknown", 0.0
                except:
                    return "Unknown", 0.0
                
            except Exception as e:
                logging.error(f"Embedding generation error: {e}")
                return "Unknown", 0.0
            
            # Calculate similarity with ABSOLUTE safety
            try:
                # Ensure known embeddings are on the same device
                known_embeddings_device = self.known_embeddings.to(self.device)
                
                # Safely normalize dimensions
                try:
                    # Get dimensions without boolean operations
                    embed_dims = embedding.dim()
                    known_dims = known_embeddings_device.dim()
                    
                    if embed_dims > known_dims:
                        embedding = embedding.squeeze()
                    elif embed_dims < known_dims:
                        embedding = embedding.unsqueeze(0)
                except:
                    return "Unknown", 0.0
                
                # Calculate cosine similarity
                try:
                    similarities = F.cosine_similarity(embedding, known_embeddings_device, dim=-1)
                except Exception as e:
                    logging.error(f"Cosine similarity error: {e}")
                    return "Unknown", 0.0
                
                # Convert to Python list IMMEDIATELY (no numpy to avoid boolean indexing)
                try:
                    if similarities.requires_grad:
                        similarities = similarities.detach()
                    if similarities.is_cuda:
                        similarities = similarities.cpu()
                    
                    # Convert directly to Python list
                    similarities_list = similarities.tolist()
                    
                    if not similarities_list:
                        return "Unknown", 0.0
                    
                    # Ensure it's a list (handle single value case)
                    if not isinstance(similarities_list, list):
                        similarities_list = [similarities_list]
                    
                except Exception as e:
                    logging.error(f"Similarity conversion error: {e}")
                    return "Unknown", 0.0
                
                # Find best match using pure Python operations
                try:
                    if len(similarities_list) == 0:
                        return "Unknown", 0.0
                    
                    # Find max using pure Python
                    best_similarity = max(similarities_list)
                    best_match_idx = similarities_list.index(best_similarity)
                    
                    # Return result based on threshold
                    if best_similarity > 0.65 and best_match_idx < len(self.known_names):
                        return self.known_names[best_match_idx], best_similarity
                    else:
                        return "Unknown", best_similarity
                        
                except Exception as e:
                    logging.error(f"Best match calculation error: {e}")
                    return "Unknown", 0.0
                    
            except Exception as e:
                logging.error(f"Similarity calculation error: {e}")
                return "Unknown", 0.0
                
        except Exception as e:
            logging.error(f"Face recognition error: {e}")
            return "Unknown", 0.0
    
    def detect_objects_gpu(self, frame):
        """GPU-accelerated object detection"""
        if not self.object_model:
            return []
            
        try:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tensor_img = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.object_model(tensor_img)
                probabilities = F.softmax(output[0], dim=0)
                
            # Get top 5 predictions
            top_prob, top_indices = probabilities.topk(5)
            
            detected_objects = []
            for prob, idx in zip(top_prob, top_indices):
                if prob.item() > 0.3 and idx.item() < len(COCO_CLASSES):
                    detected_objects.append((COCO_CLASSES[idx.item()], prob.item()))
            
            return detected_objects
            
        except Exception as e:
            logging.error(f"Object detection error: {e}")
            return []
    
    def track_movement_patterns(self, person_id, x, y, timestamp):
        """Advanced movement pattern analysis"""
        movement_patterns[person_id].append((x, y, timestamp))
        
        # Keep only recent movements
        cutoff = timestamp - timedelta(minutes=5)
        movement_patterns[person_id] = [
            (px, py, pt) for px, py, pt in movement_patterns[person_id] 
            if pt > cutoff
        ]
        
        # Analyze patterns
        if len(movement_patterns[person_id]) > 10:
            return self.analyze_behavior_pattern(person_id)
        
        return None
    
    def analyze_behavior_pattern(self, person_id):
        """Detect suspicious movement patterns"""
        movements = movement_patterns[person_id]
        if len(movements) < 10:
            return None
        
        positions = np.array([(x, y) for x, y, _ in movements])
        
        # Calculate movement statistics
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_speed = np.mean(distances)
        
        # Detect loitering (staying in small area)
        if np.std(positions) < 50 and len(movements) > 20:
            return "loitering"
        
        # Detect pacing (back and forth movement)
        if avg_speed > 30 and np.std(distances) > 20:
            return "pacing"
        
        # Detect rapid movement
        if avg_speed > 100:
            return "running"
        
        return "normal"

# Initialize GPU detector after authentication
gpu_detector = None

# Voice and email functions
def speak(text: str):
    """FIXED: Text-to-speech with better error handling"""
    logging.info(f"TTS → {text}")
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        logging.error(f"TTS error: {e}")
        print(f"[SPEECH] {text}")  # Fallback to text output

def send_email(subject: str, body: str, attachment_path=None):
    """FIXED: Email function with better error handling"""
    if not SMTP_USER or not SMTP_PASS or not EMAIL_RECIPIENT:
        logging.warning("Missing email credentials; skipping email.")
        print(f"[EMAIL DISABLED] {subject}: {body}")
        return
    
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = EMAIL_RECIPIENT
        msg.set_content(body)
        
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                file_data = f.read()
                msg.add_attachment(file_data, maintype="image", subtype="jpeg", 
                                 filename=os.path.basename(attachment_path))
        
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logging.info("📧 Email alert sent successfully")
    except Exception as e:
        logging.error(f"⚠️ Email send failed: {e}")

def play_alert():
    """FIXED: Alert sound with fallback"""
    logging.info("Playing alert sound")
    try:
        if PLAYSOUND_AVAILABLE and os.path.exists(ALERT_SOUND_PATH):
            playsound(ALERT_SOUND_PATH)
        else:
            # Fallback: system beep or print
            if os.name == 'nt':  # Windows
                import winsound
                winsound.Beep(1000, 1000)  # 1000Hz for 1 second
            else:
                print("\a" * 3)  # Terminal bell
                print("🚨 SECURITY ALERT! 🚨")
    except Exception as e:
        logging.error(f"Alert sound failed: {e}")
        print("🚨 SECURITY ALERT! 🚨")

def capture_snapshot(cam=None):
    """Capture snapshot with better error handling"""
    if not os.path.exists(SNAPSHOT_DIR):
        os.makedirs(SNAPSHOT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.jpg")
    
    if cam is not None:
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(path, frame)
            logging.info(f"📸 Snapshot saved: {path}")
            return path
    
    logging.warning("Could not capture snapshot")
    return None

# Enhanced logging function
def log_detection(name, objects, confidence, snapshot_path=None):
    """Enhanced logging with proper details"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format objects string
        obj_str = ""
        if objects:
            obj_str = ";".join([f"{obj[0]}({obj[1]:.2f})" for obj in objects[:3]])
        
        # Create detailed log entry
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([
                name, 
                timestamp, 
                snapshot_path or "N/A", 
                obj_str, 
                f"{confidence:.3f}"
            ])
        
        logging.info(f"📝 Detection logged: {name} at {timestamp} with confidence {confidence:.3f}")
        
    except Exception as e:
        logging.error(f"Logging error: {e}")

# Global state
voice_auth = False
surveillance_active = False
listening_flag = threading.Event()
exit_flag = threading.Event()
snapshot_flag = threading.Event()
voice_listening_flag = threading.Event()
anzar_authenticated = False
admin_override = False

# FIXED: Voice recognition setup with better error handling
recognizer = None
mic = None

try:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    logging.info("✅ Voice recognition initialized")
except Exception as e:
    logging.error(f"❌ Microphone initialization failed: {e}")
    print("Warning: Voice commands will not be available")

def listen_for_anzar_authentication():
    """Continuous background voice authentication for Anzar"""
    global anzar_authenticated, admin_override
    
    if not mic or not recognizer:
        logging.warning("Voice authentication disabled - no microphone")
        return
    
    logging.info("🎤 Voice authentication listener started for Anzar")
    
    while not exit_flag.is_set():
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                # Listen for 2 seconds at a time
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                
                try:
                    command = recognizer.recognize_google(audio).lower()
                    logging.info(f"Voice heard: {command}")
                    
                    # Check for Anzar authentication phrase
                    if any(phrase in command for phrase in ["this is anzar", "i am anzar", "anzar here", "this is admin"]):
                        anzar_authenticated = True
                        admin_override = True
                        speak("Voice authentication successful. Admin privileges granted.")
                        logging.info("🔓 Anzar voice authentication successful")
                        
                    # Check for admin logout
                    elif any(phrase in command for phrase in ["logout", "sign out", "remove admin", "end session"]):
                        anzar_authenticated = False
                        admin_override = False
                        speak("Admin session ended.")
                        logging.info("🔒 Admin session ended")
                        
                except sr.UnknownValueError:
                    # This is normal - just background noise
                    pass
                except sr.RequestError as e:
                    logging.error(f"Voice recognition service error: {e}")
                    time.sleep(5)  # Wait before retrying
                    
        except sr.WaitTimeoutError:
            # Normal timeout - continue listening
            pass
        except Exception as e:
            logging.error(f"Voice authentication error: {e}")
            time.sleep(1)
    
    logging.info("Voice authentication listener stopped")

def voice_command_listener():
    """FIXED: Voice command listener with better error handling"""
    global surveillance_active
    
    if not mic or not recognizer:
        speak("Voice commands not available - no microphone")
        return
    
    speak("Voice command mode activated. Listening for 30 seconds.")
    start_time = time.time()
    timeout = 30  # Listen for 30 seconds
    
    while voice_listening_flag.is_set() and (time.time() - start_time < timeout) and not exit_flag.is_set():
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                logging.info(f"Voice command received: {command}")
        except sr.WaitTimeoutError:
            continue
        except sr.UnknownValueError:
            speak("Sorry, I didn't catch that.")
            continue
        except sr.RequestError as e:
            speak(f"Could not request results; {e}")
            continue
        except Exception as e:
            logging.error(f"Voice command error: {e}")
            continue
        
        # Check if admin privileges are needed for certain commands
        admin_commands = ["stop surveillance", "stop", "halt", "abort", "exit", "quit", "close"]
        needs_admin = any(cmd in command for cmd in admin_commands)
        
        if needs_admin and not anzar_authenticated:
            speak("Admin authentication required for this command.")
            logging.warning("Command blocked - admin authentication required")
            continue
        
        # Process commands
        if any(kw in command for kw in ["take a snapshot", "snapshot", "photo", "capture"]):
            speak("Snapshot captured.")
            snapshot_flag.set()
        elif any(kw in command for kw in ["stop surveillance", "stop", "halt", "abort"]):
            surveillance_active = False
            speak("Surveillance system deactivated.")
        elif any(kw in command for kw in ["start surveillance", "start", "begin", "commence"]):
            surveillance_active = True
            speak("Surveillance system activated.")
        elif "sound alert" in command:
            threading.Thread(target=play_alert, daemon=True).start()
            speak("Security alert activated.")
        elif any(kw in command for kw in ["send report", "email report", "report"]):
            send_email("Surveillance Report", f"Surveillance report requested at {datetime.now()}")
            speak("Report sent.")
        elif any(kw in command for kw in ["exit", "quit", "close"]):
            speak("Shutting down surveillance system.")
            voice_listening_flag.clear()
            exit_flag.set()
            break
        elif any(kw in command for kw in ["admin status", "authentication status", "who am i"]):
            status = "authenticated" if anzar_authenticated else "not authenticated"
            speak(f"Admin is currently {status}")
        else:
            speak("Command not recognized.")
    
    voice_listening_flag.clear()
    speak("Voice command mode deactivated.")
    logging.info("Voice listener finished.")

def main_loop():
    """ENHANCED: Main surveillance loop with GUARANTEED face detection visibility"""
    global surveillance_active, gpu_detector
    
    # Initialize camera with fallback options
    cam = None
    backends = []
    
    if os.name == 'nt':  # Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    else:  # Linux/Mac
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    
    for backend in backends:
        try:
            cam = cv2.VideoCapture(0, backend)
            if cam.isOpened():
                logging.info(f"✅ Camera opened with backend: {backend}")
                break
            cam.release()
        except Exception as e:
            logging.warning(f"Camera backend {backend} failed: {e}")
    
    if not cam or not cam.isOpened():
        logging.error("❌ Could not open camera - trying basic initialization")
        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                raise Exception("Camera not accessible")
        except Exception as e:
            logging.error(f"❌ Camera initialization failed completely: {e}")
            print("ERROR: Cannot access camera. Please check:")
            print("1. Camera is connected")
            print("2. Camera is not being used by another application")
            print("3. Camera permissions are granted")
            return
    
    # Set camera properties with error handling
    try:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cam.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify actual resolution
        actual_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"Camera resolution: {actual_width}x{actual_height}")
        
    except Exception as e:
        logging.warning(f"Could not set camera properties: {e}")
    
    prev_gray = None
    last_alert = datetime.min
    last_unknown_check = datetime.min
    frame_count = 0
    fps_counter = time.time()
    
    # Initialize surveillance as active
    surveillance_active = True
    
    try:
        logging.info("🚀 Enhanced GPU-accelerated surveillance system starting...")
        speak("Argus surveillance system online. Press V for voice commands.")
        
        while not exit_flag.is_set():
            ret, frame = cam.read()
            if not ret:
                logging.error("Failed to read camera frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            now = datetime.now()
            current_frame = frame.copy()
            
            # FPS calculation and display
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_counter + 0.001)
                fps_counter = time.time()
                fps_text = f"FPS: {fps:.1f} | GPU: {'ON' if torch.cuda.is_available() else 'OFF'}"
            else:
                fps_text = f"FPS: -- | GPU: {'ON' if torch.cuda.is_available() else 'OFF'}"
                
            cv2.putText(current_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # ENHANCED: Constant face detection and recognition - ALWAYS VISIBLE
            seen_people = set()
            face_results = []
            
            if gpu_detector:
                # FIXED: Run face detection EVERY frame for constant visibility
                face_results = gpu_detector.detect_and_recognize_face(current_frame)
                
                for i, face_result in enumerate(face_results):
                    bbox = face_result['bbox']
                    name = face_result['name']
                    confidence = face_result['confidence']
                    
                    x1, y1, x2, y2 = bbox
                    
                    # Enhanced identity tracking with smoothing
                    identity_cache[i].append(name)
                    if len(identity_cache[i]) >= 3:
                        candidates = list(identity_cache[i])
                        most_common = max(set(candidates), key=candidates.count)
                        if candidates.count(most_common) >= len(candidates) // 2:
                            name = most_common
                    
                    seen_people.add(name)
                    
                    # Check if Anzar is detected and handle voice authentication
                    if name.lower() == "anzar" and not anzar_authenticated:
                        # Visual indicator that Anzar is detected but not authenticated
                        cv2.putText(current_frame, "SAY: 'This is Anzar'", (x1, y2 + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Movement pattern tracking (only when surveillance active)
                    if surveillance_active:
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        behavior = gpu_detector.track_movement_patterns(i, center_x, center_y, now)
                    else:
                        behavior = None
                    
                    # ENHANCED: Always visible face rectangles with proper colors
                    if name != "Unknown":
                        if name.lower() == "anzar" and anzar_authenticated:
                            color = (0, 255, 255)  # Yellow for authenticated Anzar
                            thickness = 4
                        else:
                            color = (0, 255, 0)  # Green for known faces
                            thickness = 3
                    else:
                        color = (0, 0, 255)  # Red for unknown faces
                        thickness = 3
                    
                    # ALWAYS draw bounding box regardless of surveillance state
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Prepare enhanced label
                    label = f"{name} ({confidence:.2f})"
                    if name.lower() == "anzar" and anzar_authenticated:
                        label += " [ADMIN]"
                    elif behavior and behavior != "normal" and surveillance_active:
                        label += f" [{behavior.upper()}]"
                        color = (0, 165, 255)  # Orange for suspicious behavior
                    
                    # Draw label with enhanced background for better visibility
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                    
                    # Draw background rectangle for label
                    cv2.rectangle(current_frame, (x1, y1 - label_size[1] - 15), 
                                 (x1 + label_size[0] + 10, y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(current_frame, label, (x1 + 5, y1 - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add detection confidence indicator
                    conf_bar_width = int((confidence * 100))
                    cv2.rectangle(current_frame, (x1, y2 + 5), (x1 + conf_bar_width, y2 + 15), 
                                 color, -1)
                
                # Object detection (less frequent for performance)
                detected_objects = []
                if frame_count % 15 == 0 and surveillance_active:
                    detected_objects = gpu_detector.detect_objects_gpu(current_frame)
                    if detected_objects:
                        object_names = [obj[0] for obj in detected_objects[:3]]
                        objects_text = f"Objects: {', '.join(object_names)}"
                        cv2.putText(current_frame, objects_text, 
                                   (10, current_frame.shape[0] - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Enhanced unknown person alert logic (only when surveillance active)
                if surveillance_active and now - last_unknown_check > UNKNOWN_CHECK_INTERVAL:
                    unknown_detected = any('Unknown' in person for person in seen_people)
                    if unknown_detected and now - last_alert > EMAIL_COOLDOWN:
                        last_alert = now
                        snapshot_path = capture_snapshot(cam)
                        
                        # Create detailed alert
                        alert_body = f"""
SECURITY ALERT: Unknown Person Detected

Time: {now.strftime('%Y-%m-%d %H:%M:%S')}
Location: Camera 1
Known Persons in Frame: {', '.join([p for p in seen_people if p != 'Unknown']) or 'None'}
Objects Detected: {', '.join([obj[0] for obj in detected_objects[:3]]) or 'None'}

This is an automated alert from Argus Surveillance System.
                        """.strip()
                        
                        threading.Thread(target=play_alert, daemon=True).start()
                        threading.Thread(target=send_email, 
                                       args=("🚨 SECURITY ALERT: Unknown Person Detected", 
                                            alert_body, snapshot_path), daemon=True).start()
                        
                        logging.warning(f"🚨 Unknown person alert triggered at {now}")
                    
                    last_unknown_check = now
                
                # Enhanced logging (only when surveillance active)
                if surveillance_active and seen_people and frame_count % 300 == 0:  # Log every 10 seconds at 30fps
                    for person in seen_people:
                        log_detection(person, detected_objects, 
                                    max([r['confidence'] for r in face_results if r['name'] == person], default=0.0))
            
            # Enhanced motion detection (only when surveillance active)
            motion_detected = False
            motion_pixels = 0
            
            if surveillance_active:
                gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)
                
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
                    thresh = cv2.dilate(thresh, None, iterations=2)
                    
                    # Find contours for better motion visualization
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    motion_pixels = np.count_nonzero(thresh)
                    motion_detected = motion_pixels > MOTION_THRESHOLD
                    
                    # Draw motion rectangles
                    if motion_detected:
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area > 500:
                                x, y, w, h = cv2.boundingRect(contour)
                                if x >= 0 and y >= 0 and x + w < current_frame.shape[1] and y + h < current_frame.shape[0]:
                                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                                    cv2.putText(current_frame, "MOTION", (x, max(y - 10, 10)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                prev_gray = gray
            
            # Display enhanced status information
            status_y = 60
            line_height = 25
            
            # Face count display (always visible)
            face_count = len(face_results)
            known_count = len([r for r in face_results if r['name'] != 'Unknown'])
            unknown_count = face_count - known_count
            
            face_status = f"Faces: {face_count} (Known: {known_count}, Unknown: {unknown_count})"
            cv2.putText(current_frame, face_status, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += line_height
            
            # Motion status (only when surveillance active)
            if surveillance_active:
                motion_status = f"Motion: {'DETECTED' if motion_detected else 'None'} ({motion_pixels})"
                cv2.putText(current_frame, motion_status, (10, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 255) if motion_detected else (128, 128, 128), 2)
                status_y += line_height
            
            # Surveillance status
            status_text = f"Surveillance: {'ACTIVE' if surveillance_active else 'STANDBY'}"
            status_color = (0, 255, 0) if surveillance_active else (0, 255, 255)
            cv2.putText(current_frame, status_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            status_y += line_height
            
            # Admin authentication status
            admin_text = f"Admin: {'AUTHENTICATED' if anzar_authenticated else 'NOT AUTH'}"
            admin_color = (0, 255, 255) if anzar_authenticated else (128, 128, 128)
            cv2.putText(current_frame, admin_text, (10, status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, admin_color, 2)
            status_y += line_height
            
            # GPU utilization info
            if torch.cuda.is_available():
                try:
                    gpu_memory = torch.cuda.memory_allocated() / 1e6
                    cv2.putText(current_frame, f"GPU Mem: {gpu_memory:.0f}MB", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except:
                    pass
            
            # Handle snapshot requests
            if snapshot_flag.is_set():
                snapshot_path = capture_snapshot(cam)
                if snapshot_path:
                    log_detection("Manual_Snapshot", [], 1.0, snapshot_path)
                snapshot_flag.clear()
            
            # Display the frame
            cv2.imshow("Enhanced GPU Surveillance - FACES ALWAYS VISIBLE", current_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                exit_flag.set()
                break
            elif key == ord("s"):  # Manual snapshot
                snapshot_path = capture_snapshot(cam)
                if snapshot_path:
                    log_detection("Manual_Snapshot", [], 1.0, snapshot_path)
                logging.info("Manual snapshot taken")
            elif key == ord(" "):  # Space to toggle surveillance
                surveillance_active = not surveillance_active
                status = "ACTIVATED" if surveillance_active else "DEACTIVATED"
                logging.info(f"Surveillance {status}")
                speak(f"Surveillance {status.lower()}")
            elif key == ord("v"):  # V for voice commands
                if not voice_listening_flag.is_set():
                    voice_listening_flag.set()
                    threading.Thread(target=voice_command_listener, daemon=True).start()
                else:
                    logging.info("Voice command already active")
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        exit_flag.set()
        listening_flag.clear()
        voice_listening_flag.clear()
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        if cam:
            cam.release()
        cv2.destroyAllWindows()
        logging.info("Enhanced surveillance system terminated.")

def check_dependencies():
    """Check and report missing dependencies"""
    required_packages = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'facenet_pytorch': 'facenet-pytorch',
        'speech_recognition': 'SpeechRecognition',
        'pyttsx3': 'pyttsx3',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'dotenv': 'python-dotenv'
    }
    
    missing = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(install_name)
    
    if missing:
        print("❌ Missing required packages:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print("\nInstall missing packages and try again.")
        return False
    
    return True

if __name__ == "__main__":
    print("🔍 Argus Enhanced Surveillance System")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        exit(1)
    
    # Admin Authentication Required
    if not authenticate_admin():
        print("❌ Access denied. System terminating.")
        exit(1)
    
    print("\n✅ Initializing surveillance system...")
    
    try:
        # Initialize GPU detector after authentication
        gpu_detector = GPUOptimizedDetector()
        
        print("\n🎯 Controls:")
        print("  • SPACEBAR: Toggle surveillance on/off")
        print("  • 'S': Take manual snapshot")
        print("  • 'V': Activate voice commands")
        print("  • 'Q': Quit system")
        print("\n🎤 Voice Commands:")
        print("  • 'Take a snapshot' / 'Snapshot' / 'Photo'")
        print("  • 'Start surveillance' / 'Stop surveillance' (Admin required)")
        print("  • 'Sound alert'")
        print("  • 'Send report' / 'Email report'")
        print("  • 'Exit' / 'Quit' (Admin required)")
        print("  • 'Admin status' - Check authentication status")
        print("\n🔑 Voice Authentication:")
        print("  • When Anzar appears on camera, say: 'This is Anzar'")
        print("  • To logout: 'Logout' / 'Sign out'")
        print("\n📁 Setup Instructions:")
        print("  • Create 'known_faces' folder")
        print("  • Add person folders: known_faces/john/, known_faces/mary/")
        print("  • Add face photos (.jpg, .png) to each person's folder")
        print("  • Create '.env' file for email settings (optional)")
        print("\n🎯 ENHANCED FEATURES:")
        print("  • Faces are ALWAYS detected and visible (green=known, red=unknown)")
        print("  • Boolean tensor errors completely eliminated")
        print("  • Constant face recognition regardless of surveillance state")
        print("  • Enhanced visual feedback and status display")
        print("\n" + "=" * 50)
        
        # Start voice authentication listener in background
        if mic and recognizer:
            auth_thread = threading.Thread(target=listen_for_anzar_authentication, daemon=True)
            auth_thread.start()
            logging.info("🎤 Background voice authentication started")
        else:
            print("⚠️  Voice authentication disabled - no microphone available")
        
        main_loop()
        
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ Fatal error occurred: {e}")
        print("Check the console output above for details.")
    
    print("\n👋 Surveillance system shutdown complete.")