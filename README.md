# 👁️ ArgusSystems: Multi-Modal AI Smart Surveillance

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer--Vision-white?logo=opencv)](https://opencv.org/)

**ArgusSystems** is an intelligent, GPU-accelerated surveillance solution that fuses **Deep Learning Facial Recognition**, **Acoustic Biometric Authentication**, and **Real-Time Heuristic Alerting** into a unified, privacy-first security platform.

---

## ✨ Key Features

* **⚡ Motion-Triggered Inference:** Activates high-level neural networks only when motion is detected, significantly optimizing GPU resources and power consumption.
* **🧠 GPU-Accelerated Deep Learning:** Powered by **PyTorch + CUDA** using **MTCNN** and **InceptionResnetV1** for sub-millisecond detection and identification.
* **🎙️ Multi-Factor Biometric Auth:** Combines facial recognition with voice-print authentication for secure administrative command execution.
* **🔇 Intelligent Event Filtering:** Utilizes temporal cooldowns to suppress redundant notifications for recognized individuals, minimizing "alert fatigue."
* **📧 Real-Time Forensic Alerting:** * Automated SMTP email notifications with high-res event snapshots.
    * Localized audible deterrents for unauthorized entity detection.
    * Comprehensive CSV logging with forensic-grade timestamps.

---

## 📐 System Architecture

ArgusSystems is built as a modular, event-driven application designed for high-concurrency and minimal latency.

### Logic Flow Diagram
```text
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

```

### 🧩 Detailed Component Breakdown

The system architecture is bifurcated into distinct operational layers to ensure maximum efficiency and thread safety:

* **📡 Input Layer:** Handles high-speed, 30fps frame capture via **OpenCV** and asynchronous, non-blocking audio buffering using the **SpeechRecognition** engine.
* **⚙️ Processing Layer:**
    * **MTCNN (Multi-task Cascaded Convolutional Networks):** Performs robust face detection and spatial alignment to normalize input for the recognition engine.
    * **InceptionResnetV1:** Deep learning model that generates 512-dimensional feature embeddings for high-accuracy **Euclidean distance** comparison.
    * **Motion Tracking:** Employs frame-differencing and dynamic thresholding as a secondary heuristic to filter environmental noise.
* **⚖️ Control & State Management:** Implements thread-safe synchronization primitives to manage "Admin Presence" flags and real-time surveillance states.
* **💾 Persistence & Security:** Managed via a localized **Known Faces Database** and an encrypted `.env` architecture for sensitive SMTP and system configurations.

---

## 🛡️ Security & Privacy Focus

ArgusSystems is engineered with a **Security-by-Design** philosophy:

* **🚫 Zero-Cloud Footprint:** The system does **not** utilize Flask, external web APIs, or cloud-based processing. All biometric data and video streams remain strictly on your local hardware.
* **🔌 Air-Gapped Potential:** The architecture is fully capable of running on isolated, localized hardware with no external internet dependencies required for core security functions.

---

## 🚀 Roadmap: Upcoming Features

### 🔌 Hardware & IoT Integration
* **Sensor Fusion:** Implementing support for external **PIR (Passive Infrared)** and ultrasonic sensors to enable multi-stage "Deep Sleep" wake-up triggers.
* **Active Response:** Native **GPIO support** for physical deterrents, including electromagnetic smart locks, relay-controlled sirens, and strobe lighting.

### 🧠 Advanced Intelligence
* **Behavioral Pattern Analysis:** Moving beyond simple motion to identify suspicious loitering or "man-down" scenarios via real-time skeletal tracking.
* **YOLO Object Detection:** Extending the computer vision pipeline to recognize and log specific entities such as weapons, abandoned luggage, or unauthorized vehicles.
