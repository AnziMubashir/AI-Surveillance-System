# ArgusSystems: Multi-Modal AI Smart Surveillance System

## 📌 Overview
An intelligent, GPU-accelerated surveillance solution that combines **face recognition**, **voice authentication**, and **real-time alerts** into a single adaptive security platform. Designed for efficiency, accuracy, and minimal false positives.

---

## ✨ Features

- **Motion-Triggered Face Recognition**  
  Activates only when motion is detected, optimizing performance and reducing unnecessary computation.

- **GPU-Accelerated Deep Learning**  
  Powered by **PyTorch + CUDA** for faster detection and recognition.

- **Biometric Multi-Factor Authentication**  
  Combines **voice recognition** and **facial recognition** for secure admin access and voice command execution.

- **Intelligent Event Filtering**  
  Avoids duplicate alerts by suppressing frequent detections of the same recognized individual within a short time window.

- **Real-Time Alerts & Logging**  
  - Sends **email notifications** with snapshots and event details.  
  - Plays **audible alerts** for unknown detections.  
  - Maintains detailed logs with timestamps and captured images.

- **Optimized Recognition Pipeline**  
  Uses averaged embeddings from multiple input images to improve recognition accuracy.

---

## System Architecture Overview

The SurveillanceProject system is designed as a modular, event-driven application that combines real-time computer vision, voice recognition, and alerting. Here’s how the architecture is structured:

---

### 1. **Input Layer**
- **Camera Input:**  
  Captures live video frames from the webcam using OpenCV.
- **Microphone Input:**  
  Captures audio for voice command recognition using the SpeechRecognition library.

### 2. **Processing Layer**
- **Face Detection & Recognition:**  
  - Uses MTCNN (Multi-task Cascaded Convolutional Networks) to detect faces in each frame.
  - Extracts face embeddings with a pre-trained InceptionResnetV1 model (facenet-pytorch).
  - Compares detected faces to known embeddings to identify individuals.
- **Motion Detection:**  
  - Uses frame differencing and thresholding to detect significant motion in the scene.

### 3. **Control & State Management**
- **Voice Command Thread:**  
  - Runs in parallel to the main video loop.
  - Only accepts commands if the admin ("Anzar") is detected in the current frame.
  - Requires voice authentication before executing commands.
- **Surveillance State:**  
  - Maintains flags for surveillance activity, snapshot requests, and admin presence.
  - Uses thread-safe events and shared variables for inter-thread communication.

### 4. **Output & Actions**
- **Visual Feedback:**  
  - Annotates video frames with bounding boxes, names, and confidence scores.
  - Displays the live feed with overlays.
- **Audio Feedback:**  
  - Uses TTS (pyttsx3) to provide spoken responses and alerts.
- **Logging:**  
  - Logs recognized faces and motion events to a CSV file.
  - Saves snapshots on command or on certain events.
- **Alerting:**  
  - Plays an alert sound and sends email notifications when unknown faces are detected or on command.

### 5. **Persistence & Configuration**
- **Known Faces Database:**  
  - Stores reference images in a directory structure (`known_faces/<person_name>/`).
- **Environment Variables:**  
  - Stores sensitive configuration (SMTP credentials, email recipient) in a `.env` file.

### 6. **Threading & Synchronization**
- The main video loop and the voice command listener run in separate threads.
- Shared state (such as admin presence) is managed using thread-safe events to ensure reliable coordination.

### **Summary Diagram**
+-------------------+      +-------------------+
|   Camera Input    |----->|  Face Detection   |
+-------------------+      +-------------------+
                                 |
                                 v
+-------------------+      +-------------------+      +-------------------+
| Microphone Input  |----->| Voice Recognition |----->| Command Execution |
+-------------------+      +-------------------+      +-------------------+
                                 |                          |
                                 v                          v
                        +-------------------+      +-------------------+
                        |  State Management |<---->|  Motion Detection |
                        +-------------------+      +-------------------+
                                 |                          |
                                 v                          v
                        +-------------------+      +-------------------+
                        |   Logging/Alerts  |      |   Visual Output   |
                        +-------------------+      +-------------------+


## Upcoming Features
The SurveillanceProject is under active development. Planned enhancements include:

### 🔌 Hardware & Sensor Integration
- **IoT & Sensor Support:**  
  Integrate with external sensors (e.g., PIR motion sensors, door/window contacts, temperature/humidity sensors) for richer event triggers and smarter automation.
- **Hardware Controls:**  
  Support for relays, alarms, smart locks, or lights—allowing the system to physically respond to events (e.g., lock doors on unknown detection).

---

### 🧠 Advanced Algorithms
- **Behavioral Pattern Recognition:**  
  Analyze movement and activity patterns to detect unusual or suspicious behavior over time.
- **Object Detection:**  
  Incorporate object detection (e.g., YOLO, SSD) to recognize and log specific items (bags, packages, weapons, etc.) in the camera feed.
- **Multi-person Tracking:**  
  Track multiple individuals across frames for better event correlation and analytics.

---

### 🛡️ Security & Privacy Focus
- **No Web UI/Flask:**  
  By design, the system does **not** use Flask or any web-based UI, ensuring all processing and control remain local for maximum security and privacy.
- **Local-Only Operation:**  
  All data, logs, and video remain on the device; no cloud or remote access is required or planned.

---

### 💡 Other Planned Improvements
- **Configurable Admins:**  
  Support for multiple trusted users, each with their own authentication phrase.
- **Customizable Voice Commands:**  
  Allow users to define their own command phrases and actions.
- **Improved Logging & Analytics:**  
  More detailed event logs, statistics, and possibly local dashboard summaries (still local, not web-based).

---

**Have a feature request?**  
Feel free to open an issue or contribute ideas!


