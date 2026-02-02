import face_recognition
import cv2
import os
import numpy as np
from PIL import Image
import csv
from datetime import datetime, timedelta
import logging

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# === Motion Detection Setup ===
prev_gray = None
motion_threshold = 50000  # tweak this for sensitivity

# === Face Recognition Setup ===
known_face_encodings = []
known_face_names = []

known_faces_dir = "known_faces"
log_dir = "activity_logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, "detections.csv")
snapshot_dir = os.path.join(log_dir, "snapshots")
os.makedirs(snapshot_dir, exist_ok=True)

# Dicts to track last log/snapshot times
last_logged_time = {}
last_snapshot_time = {}

# === Create CSV Log File ===
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Timestamp", "ImagePath"])

# === Load Known Faces ===
for person_name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, person_name)
    if not os.path.isdir(person_dir):
        continue

    person_encodings = []

    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                logging.info(f"Converting {filename} from {pil_image.mode} to RGB")
                pil_image = pil_image.convert("RGB")
                pil_image.save(image_path)

            image = np.asarray(pil_image, dtype=np.uint8)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                person_encodings.append(encodings[0])
                logging.info(f"Processed: {filename}")
            else:
                logging.warning(f"No faces found in: {filename}")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    if person_encodings:
        averaged_encoding = np.mean(person_encodings, axis=0)
        known_face_encodings.append(averaged_encoding)
        known_face_names.append(person_name)
        logging.info(f"Averaged encoding added for {person_name}")
    else:
        logging.warning(f"No valid encodings found for {person_name}")

# === Start Webcam ===
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not video_capture.isOpened():
    logging.error("Could not open webcam.")
    exit()

logging.info("Surveillance system active. Press 'q' to quit.")

# === Main Loop ===
while True:
    ret, frame = video_capture.read()
    if not ret:
        logging.warning("Failed to grab frame.")
        continue

    # --- Motion Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    motion_detected = False
    if prev_gray is not None:
        frame_diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        motion_level = cv2.countNonZero(thresh)
        if motion_level > motion_threshold:
            motion_detected = True
            logging.info(f"MOTION detected with intensity: {motion_level}")

    prev_gray = gray

    if not motion_detected:
        cv2.imshow("Surveillance Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- Face Recognition Pipeline ---
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    small_frame = cv2.resize(enhanced_frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    recognized_faces = []

    for (location, encoding) in zip(face_locations, face_encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.58)
        face_distances = face_recognition.face_distance(known_face_encodings, encoding)

        if matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        recognized_faces.append((name, location))

       for name, location in recognized_faces:
     # --- Logging ---
     now = datetime.now()
     last_seen = last_logged_time.get(name, now - timedelta(minutes=11))
     if (now - last_seen) >= timedelta(minutes=10):
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        snapshot_path = "N/A"

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, timestamp, snapshot_path])
            logging.info(f"LOGGED {name} at {timestamp}")

        last_logged_time[name] = now

     # --- Snapshot ---
     if name != "Unknown":
        last_snap = last_snapshot_time.get(name)
        if last_snap is None or (now - last_snap) >= timedelta(minutes=10):
            snapshot_filename = f"{name}_{now.strftime('%Y%m%d_%H%M%S')}.jpg"
            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
            cv2.imwrite(snapshot_path, frame)

            with open(log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S"), snapshot_path])
                logging.info(f"SNAPSHOT {name} saved at {snapshot_path}")

            last_snapshot_time[name] = now
        else:
            logging.info(f"Snapshot for {name} skipped — already captured within last 10 mins.")

    # --- Display ---
    for name, (top, right, bottom, left) in recognized_faces:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 1)

    cv2.imshow("Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
video_capture.release()
cv2.destroyAllWindows()
logging.info("System terminated.")