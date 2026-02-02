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
from datetime import datetime, timedelta
from dotenv import load_dotenv
from collections import defaultdict, deque

# Optional alert sound
try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False

load_dotenv()

# Constants
DETECTION_CONFIDENCE = 0.7
FACE_SIZE = 160
KNOWN_FACES_DIR = "known_faces"
LOG_DIR = "activity_logs"
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "detections.csv")
PATTERN_LOG = os.path.join(LOG_DIR, "patterns.csv")

ADMIN_PASSWORD = "password123"

# Email config
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# Setup logging
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# CSV Init
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Name", "Timestamp", "ImagePath", "Objects", "Confidence"])

if not os.path.exists(PATTERN_LOG):
    with open(PATTERN_LOG, "w", newline="") as f:
        csv.writer(f).writerow(["PersonID", "Timestamp", "X", "Y", "Activity", "Duration"])

# Voice recognition setup
recognizer = None
mic = None
try:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    logging.info("Voice recognition initialized")
except Exception as e:
    logging.warning(f"Microphone init failed: {e}")

# Utility functions
def speak(text: str):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        print(f"[SPEECH] {text}")

def send_email(subject: str, body: str, attachment_path=None):
    if not SMTP_USER or not SMTP_PASS or not EMAIL_RECIPIENT:
        logging.warning("Missing email credentials")
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = EMAIL_RECIPIENT
        msg.set_content(body)
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename=os.path.basename(attachment_path))
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logging.info("Email sent")
    except Exception as e:
        logging.error(f"Email send failed: {e}")

def play_alert():
    if PLAYSOUND_AVAILABLE and os.path.exists("alert_sound.mp3"):
        playsound("alert_sound.mp3")
    else:
        print("🚨 SECURITY ALERT! 🚨")

# Behavior tracking
movement_patterns = defaultdict(list)

def track_behavior(person_id, x, y, timestamp):
    movement_patterns[person_id].append((x, y, timestamp))
    cutoff = timestamp - timedelta(minutes=5)
    movement_patterns[person_id] = [(px, py, pt) for px, py, pt in movement_patterns[person_id] if pt > cutoff]
    if len(movement_patterns[person_id]) < 5:
        return "normal"
    positions = np.array([(px, py) for px, py, _ in movement_patterns[person_id]])
    distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
    avg_speed = np.mean(distances)
    if np.std(positions) < 50 and len(movement_patterns[person_id]) > 20:
        return "loitering"
    if avg_speed > 30 and np.std(distances) > 20:
        return "pacing"
    if avg_speed > 100:
        return "running"
    return "normal"

def log_pattern(person_id, x, y, activity, duration):
    with open(PATTERN_LOG, "a", newline="") as f:
        csv.writer(f).writerow([person_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), x, y, activity, duration])

# Detector
class GPUOptimizedDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.detector = MTCNN(image_size=FACE_SIZE, keep_all=True, device=self.device)
        self.known_embeddings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        face_images, temp_names = [], []
        for person in os.listdir(KNOWN_FACES_DIR):
            pdir = os.path.join(KNOWN_FACES_DIR, person)
            if not os.path.isdir(pdir):
                continue
            for fname in os.listdir(pdir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                path = os.path.join(pdir, fname)
                try:
                    img = Image.open(path).convert("RGB")
                    face = self.detector(img)
                    if isinstance(face, torch.Tensor) and face.numel() > 0:
                        face_images.append(face[0])
                        temp_names.append(person)
                        logging.info(f"Loaded face: {person}/{fname}")
                except Exception as e:
                    logging.warning(f"Could not load {fname}: {e}")
        if face_images:
            batch_faces = torch.stack(face_images).to(self.device)
            with torch.no_grad():
                embeddings = self.face_model(batch_faces)
                self.known_embeddings = embeddings.cpu()
                self.known_names = temp_names

    def detect_and_recognize_face(self, frame):
        results = []
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        boxes, probs = self.detector.detect(pil_img)
        if boxes is None:
            return results
        for box, prob in zip(boxes, probs):
            if prob < DETECTION_CONFIDENCE:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            name, conf = self.recognize_face_crop(frame, (x1, y1, x2, y2))
            results.append({"bbox": (x1, y1, x2, y2), "name": name, "confidence": conf})
        return results

    def recognize_face_crop(self, frame, bbox):
        if not self.known_embeddings:
            return "Unknown", 0.0
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "Unknown", 0.0
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        face_tensor = self.detector(pil_crop)
        if face_tensor is None:
            return "Unknown", 0.0
        if isinstance(face_tensor, (list, tuple)):
            face_tensor = face_tensor[0]
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.face_model(face_tensor)
        similarities = F.cosine_similarity(embedding, self.known_embeddings.to(self.device))
        best_sim = similarities.max().item()
        best_idx = similarities.argmax().item()
        if best_sim > 0.5:
            return self.known_names[best_idx], best_sim
        return "Unknown", best_sim

# Logging
def log_detection(name, confidence, snapshot_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, timestamp, snapshot_path or "N/A", "N/A", f"{confidence:.3f}"])
    logging.info(f"Detection logged: {name} ({confidence:.2f})")

# Voice command listener
def voice_command_listener():
    if not mic or not recognizer:
        return
    speak("Voice command mode activated.")
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                logging.info(f"Voice command: {command}")
                if "snapshot" in command:
                    speak("Snapshot captured.")
                elif "stop" in command:
                    speak("Stopping system.")
                    os._exit(0)
                elif "alert" in command:
                    play_alert()
                    speak("Alert triggered.")
                elif "email" in command:
                    send_email("Surveillance Report", "Requested report.")
                    speak("Email sent.")
        except Exception:
            continue
    speak("Voice mode off.")

# Main loop
def main_loop():
    gpu_detector = GPUOptimizedDetector()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        logging.error("Could not access camera")
        return
    speak("Surveillance online. Press V for voice commands.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        results = gpu_detector.detect_and_recognize_face(frame)
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            activity = track_behavior(r["name"], cx, cy, datetime.now())
            if activity != "normal":
                log_pattern(r["name"], cx, cy, activity, 5)
                cv2.putText(frame, activity, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{r['name']} ({r['confidence']:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            log_detection(r["name"], r["confidence"])
        cv2.imshow("Surveillance", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('v'):
            threading.Thread(target=voice_command_listener, daemon=True).start()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
