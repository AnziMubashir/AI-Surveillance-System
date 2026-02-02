import cv2
import os
import csv
import logging
import threading
import time
import smtplib
from email.message import EmailMessage
import torch
import pyttsx3
import speech_recognition as sr
from playsound import playsound
from datetime import datetime, timedelta
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

# === Configuration ===
ALERT_SOUND_PATH   = "alert_sound.mp3"
SPEAK_COOLDOWN     = timedelta(seconds=10)
ALERT_COOLDOWN     = timedelta(minutes=5)
MOTION_THRESHOLD   = 50000

KNOWN_FACES_DIR    = "known_faces"
LOG_DIR            = "activity_logs"
SNAPSHOT_DIR       = os.path.join(LOG_DIR, "snapshots")
LOG_FILE           = os.path.join(LOG_DIR, "detections.csv")

EMAIL_RECIPIENT    = "you@example.com"
SMTP_SERVER        = "smtp.gmail.com"
SMTP_PORT          = 465
SMTP_USER          = os.getenv("SMTP_USER")
SMTP_PASS          = os.getenv("SMTP_PASS")
EMAIL_COOLDOWN     = timedelta(minutes=10)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(["Name", "Timestamp", "ImagePath"])
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === TTS & Alert Helpers ===
tts_engine = pyttsx3.init()

def speak(text: str):
    logging.info(f"TTS → {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

def play_alert():
    logging.info("Playing alert sound")
    try:
        playsound(ALERT_SOUND_PATH)
    except Exception as e:
        logging.error(f"Alert sound failed: {e}")

# === Email Alert Helper ===
_last_email = datetime.min

def send_email(message: str):
    global _last_email
    now = datetime.now()
    if now - _last_email < EMAIL_COOLDOWN:
        return
    _last_email = now

    if not SMTP_USER or not SMTP_PASS:
        logging.warning("Missing SMTP credentials; skipping email.")
        return

    try:
        msg = EmailMessage()
        msg["Subject"] = "Surveillance Alert"
        msg["From"]    = SMTP_USER
        msg["To"]      = EMAIL_RECIPIENT
        msg.set_content(message)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logging.info("Email alert sent.")
    except Exception as e:
        logging.error(f"Email send failed: {e}")

# === Surveillance State ===
surveillance_active = True
snapshot_flag = threading.Event()

# === Voice-Command Thread ===
recognizer     = sr.Recognizer()
mic            = sr.Microphone()
listening_flag = threading.Event()
voice_auth     = False

def voice_command_listener():
    global voice_auth, surveillance_active

    speak("Voice module active. Please authenticate by saying: This is Anzar.")

    while listening_flag.is_set():
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                logging.info(f"Voice heard: {text}")
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                speak("Sorry, I didn't catch that.")
                continue
            except sr.RequestError:
                speak("Speech service unavailable.")
                continue

        if not voice_auth:
            if "this is anzar" in text:
                voice_auth = True
                speak("Voice authenticated. You may now give commands.")
            else:
                speak("Please say: This is Anzar.")
            continue

        if "start surveillance" in text:
            surveillance_active = True
            speak("Surveillance started.")
        elif "stop surveillance" in text:
            surveillance_active = False
            speak("Surveillance stopped.")
        elif "take snapshot" in text:
            snapshot_flag.set()
            speak("Snapshot taken.")
        elif "sound alert" in text:
            threading.Thread(target=play_alert, daemon=True).start()
            speak("Alert sounded.")
        elif "send report" in text:
            send_email("Report requested by voice.")
            speak("Report sent.")
        elif "logout" in text or "deactivate voice" in text:
            voice_auth = False
            speak("Voice deactivated. Please authenticate again to continue.")
        else:
            speak("Command not recognized.")

    voice_auth = False
    logging.info("Voice listener exiting.")

# === Load Models ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn  = MTCNN(image_size=160, margin=0, device=device)

# === Load Known Faces ===
known_embeddings, known_names = [], []

for person in os.listdir(KNOWN_FACES_DIR):
    pdir = os.path.join(KNOWN_FACES_DIR, person)
    if not os.path.isdir(pdir):
        continue

    loaded = False
    for fname in os.listdir(pdir):
        path = os.path.join(pdir, fname)
        try:
            img = Image.open(path).convert('RGB')
            face = mtcnn(img)
            if face is None:
                logging.warning(f"[WARN] No face detected in {fname} (person: {person})")
                continue

            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                emb = model(face).cpu()

            if emb.numel() == 0:
                logging.warning(f"[WARN] Empty embedding for {fname} (person: {person})")
                continue

            known_embeddings.append(emb)
            known_names.append(person)
            loaded = True

        except Exception as e:
            logging.error(f"[ERROR] Failed to process {fname} (person: {person}): {e}")

    if not loaded:
        logging.warning(f"[WARN] No valid faces loaded for {person}")

if not known_embeddings:
    logging.error("[FATAL] No known face embeddings loaded. Check your known_faces directory and images.")
    speak("Fatal error. No known face embeddings were loaded. Shutting down.")
    exit(1)

try:
    known_embeddings = torch.cat(known_embeddings)
except Exception as e:
    logging.error(f"[FATAL] Failed to concatenate embeddings: {e}")
    speak("Fatal error. Could not process known faces. Shutting down.")
    exit(1)

if known_embeddings.numel() == 0:
    logging.error("[FATAL] Known embeddings tensor is empty after concatenation.")
    speak("Fatal error. No valid face embeddings found. Shutting down.")
    exit(1)

logging.info(f"Loaded {len(known_names)} known faces: {set(known_names)}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)
if not cap.isOpened():
    logging.error("Cannot open camera; exiting.")
    exit(1)

logging.info("Surveillance started. Press 'q' to quit.")
prev_time  = time.time()
prev_gray  = None
last_speak = {}
last_alert = datetime.min
last_anzar_greet = datetime.min

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logging.warning("Empty frame received from camera.")
            time.sleep(0.1)
            continue

        now_t = time.time()
        fps   = 1/(now_t - prev_time); prev_time = now_t
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        rgb, seen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), set()
        boxes, _ = mtcnn.detect(Image.fromarray(rgb))
        if boxes is not None:
            for b in boxes:
                x1,y1,x2,y2 = map(int,b)
                crop = frame[y1:y2, x1:x2]
                fimg = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensor = mtcnn(fimg)
                if tensor is None: continue
                tensor = tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    if device.type=='cuda': torch.cuda.synchronize()
                    emb = model(tensor)
                sim = torch.nn.functional.cosine_similarity(emb.cpu(), known_embeddings)
                score, idx = sim.max().item(), sim.argmax().item()
                name = known_names[idx] if score>0.65 else "Unknown"
                seen.add(name)

                col = (0,255,0) if name!="Unknown" else (0,0,255)
                lbl = f"{name} ({score:.2f})" if name!="Unknown" else name
                cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
                cv2.putText(frame,lbl,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

                now_dt = datetime.now()
                if name=="Unknown":
                    if now_dt - last_alert >= ALERT_COOLDOWN:
                        threading.Thread(target=play_alert, daemon=True).start()
                        send_email(f"🚨 Unknown at {now_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        last_alert = now_dt

        anzar = "Anzar" in seen
        now_dt = datetime.now()
        if anzar and (now_dt - last_anzar_greet >= timedelta(minutes=10)):
            last_anzar_greet = now_dt
            threading.Thread(target=speak, args=("Hello Anzar",), daemon=True).start()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)
        mot = False
        if prev_gray is not None:
            d = cv2.absdiff(prev_gray, gray)
            t = cv2.threshold(d,25,255,cv2.THRESH_BINARY)[1]
            t = cv2.dilate(t,None,iterations=2)
            mot = cv2.countNonZero(t)>MOTION_THRESHOLD
        prev_gray = gray

        if surveillance_active and mot and seen:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE,'a',newline='') as f:
                csv.writer(f).writerow([";".join(seen), ts, "N/A"])
            logging.info(f"Logged {seen} at {ts}")

        if anzar and not listening_flag.is_set():
            listening_flag.set()
            threading.Thread(target=voice_command_listener, daemon=True).start()
            logging.info("Voice listener started.")
        elif not anzar and listening_flag.is_set():
            listening_flag.clear()
            speak("Voice module deactivated.")
            logging.info("Voice listener stopped.")

        if snapshot_flag.is_set():
            snap_path = os.path.join(SNAPSHOT_DIR, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            cv2.imwrite(snap_path, frame)
            logging.info(f"Snapshot saved: {snap_path}")
            snapshot_flag.clear()

        cv2.imshow("Surveillance Feed", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break

except KeyboardInterrupt:
    logging.info("Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    logging.info("System terminated.")
