from flask import Flask, render_template_string, send_from_directory, jsonify
from flask_cors import CORS
import os
import csv
from datetime import datetime, timedelta
from collections import defaultdict, Counter

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "activity_logs")
SNAPSHOT_DIR = os.path.join(LOG_DIR, "snapshots")
LOG_FILE = os.path.join(LOG_DIR, "detections.csv")
PATTERN_LOG = os.path.join(LOG_DIR, "patterns.csv")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# Create directories if they don't exist
for directory in [LOG_DIR, SNAPSHOT_DIR, KNOWN_FACES_DIR]:
    os.makedirs(directory, exist_ok=True)

# Create CSV files with headers if they don't exist
def ensure_log_files():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Timestamp', 'ImagePath', 'Objects', 'Confidence'])
    
    if not os.path.exists(PATTERN_LOG):
        with open(PATTERN_LOG, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['PersonID', 'Timestamp', 'X', 'Y', 'Activity', 'Duration'])

def read_detection_logs():
    detections = []
    ensure_log_files()
    
    if not os.path.exists(LOG_FILE):
        return detections
    
    try:
        with open(LOG_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'Name' in row and row['Name'].strip():
                    # Clean up the data
                    clean_row = {
                        'Name': row.get('Name', '').strip(),
                        'Timestamp': row.get('Timestamp', '').strip(),
                        'ImagePath': row.get('ImagePath', '').strip(),
                        'Objects': row.get('Objects', '').strip(),
                        'Confidence': row.get('Confidence', '0').strip()
                    }
                    if clean_row['Name']:  # Only add if name is not empty
                        detections.append(clean_row)
    except Exception as e:
        print(f"Error reading detection logs: {e}")
    
    return detections

def read_pattern_logs():
    patterns = []
    ensure_log_files()
    
    if not os.path.exists(PATTERN_LOG):
        return patterns
    
    try:
        with open(PATTERN_LOG, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and 'PersonID' in row and row['PersonID'].strip():
                    patterns.append(row)
    except Exception as e:
        print(f"Error reading pattern logs: {e}")
    
    return patterns

def get_enhanced_stats():
    detections = read_detection_logs()
    patterns = read_pattern_logs()
    now = datetime.now()

    # Count known people from directory
    known_people_count = 0
    if os.path.exists(KNOWN_FACES_DIR):
        try:
            known_people_count = len([d for d in os.listdir(KNOWN_FACES_DIR)
                                     if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
        except:
            known_people_count = 0

    # Count detection statistics
    unknown_alerts = 0
    total_detections = len(detections)
    confidence_scores = []
    object_detections = defaultdict(int)
    
    for d in detections:
        try:
            # Count unknown alerts
            if 'Unknown' in d.get('Name', ''):
                unknown_alerts += 1
            
            # Parse objects
            objects = d.get('Objects', '')
            if objects and objects != 'N/A':
                for obj in objects.split(';'):
                    if '(' in obj and ')' in obj:
                        obj_name = obj.split('(')[0].strip()
                        if obj_name:
                            object_detections[obj_name] += 1
            
            # Parse confidence
            confidence = d.get('Confidence', '0')
            try:
                if confidence and confidence != 'N/A':
                    conf_val = float(confidence)
                    if 0 <= conf_val <= 1:
                        confidence_scores.append(conf_val)
            except:
                pass
        except Exception as e:
            print(f"Error processing detection: {e}")

    # Count behavior patterns
    behavior_counts = defaultdict(int)
    for p in patterns:
        activity = p.get('Activity', '')
        if activity and activity != 'normal':
            behavior_counts[activity] += 1

    # Calculate recent detections (last 7 days)
    recent_detections = []
    for d in detections:
        try:
            timestamp_str = d.get('Timestamp', '')
            if timestamp_str:
                dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                if (now - dt).days < 7:
                    recent_detections.append(d)
        except:
            continue

    # Calculate uptime
    uptime_hours = 24
    if detections:
        try:
            first_detection = datetime.strptime(detections[0]['Timestamp'], '%Y-%m-%d %H:%M:%S')
            uptime_delta = now - first_detection
            uptime_hours = max(1, int(uptime_delta.total_seconds() / 3600))
        except:
            pass

    # Calculate average confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

    # Find most detected object
    most_detected_object = 'None'
    if object_detections:
        most_detected_object = max(object_detections.items(), key=lambda x: x[1])[0]

    # Calculate detection accuracy
    detection_accuracy = 0
    if total_detections > 0:
        detection_accuracy = ((total_detections - unknown_alerts) / total_detections) * 100

    return {
        'total_detections': total_detections,
        'known_people': known_people_count,
        'unknown_alerts': unknown_alerts,
        'uptime': f'{uptime_hours}h',
        'object_detections': dict(object_detections),
        'avg_confidence': round(avg_confidence, 2),
        'recent_detections_week': len(recent_detections),
        'behavior_patterns': dict(behavior_counts),
        'detection_accuracy': round(detection_accuracy, 1),
        'most_detected_object': most_detected_object
    }

def get_hourly_stats(detections):
    hours_24 = defaultdict(int)
    hours_week = defaultdict(int)
    now = datetime.now()

    for detection in detections:
        try:
            dt = datetime.strptime(detection.get('Timestamp', ''), '%Y-%m-%d %H:%M:%S')
            if now - dt <= timedelta(hours=24):
                hours_24[dt.hour] += 1
            if now - dt <= timedelta(days=7):
                hours_week[dt.hour] += 1
        except:
            continue

    return {
        '24h': [hours_24.get(i, 0) for i in range(24)],
        'week': [hours_week.get(i, 0) for i in range(24)]
    }

def get_behavioral_insights():
    patterns = read_pattern_logs()
    detections = read_detection_logs()
    now = datetime.now()

    daily_patterns = defaultdict(lambda: defaultdict(int))
    hourly_behaviors = defaultdict(lambda: defaultdict(int))

    for p in patterns:
        try:
            timestamp = datetime.strptime(p.get('Timestamp', ''), '%Y-%m-%d %H:%M:%S')
            activity = p.get('Activity', 'normal')

            if now - timestamp <= timedelta(days=30):
                day_of_week = timestamp.strftime('%A')
                hour = timestamp.hour

                daily_patterns[day_of_week][activity] += 1
                hourly_behaviors[hour][activity] += 1
        except:
            continue

    person_frequency = defaultdict(int)
    for d in detections:
        names = d.get('Name', '').split(';')
        for name in names:
            name = name.strip()
            if name and name != 'Unknown':
                person_frequency[name] += 1

    suspicious_activities = len([p for p in patterns if p.get('Activity') in ['loitering', 'pacing']])

    return {
        'daily_patterns': {k: dict(v) for k, v in daily_patterns.items()},
        'hourly_behaviors': {k: dict(v) for k, v in hourly_behaviors.items()},
        'person_frequency': dict(person_frequency),
        'suspicious_activities': suspicious_activities
    }

def get_recent_snapshots():
    snapshots = []

    if os.path.exists(SNAPSHOT_DIR):
        files = []
        try:
            for f in os.listdir(SNAPSHOT_DIR):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(SNAPSHOT_DIR, f)
                    if os.path.exists(file_path):
                        files.append((f, os.path.getmtime(file_path), os.path.getsize(file_path)))
        except:
            pass

        files.sort(key=lambda x: x[1], reverse=True)
        for filename, mtime, size in files[:12]:
            try:
                snapshots.append({
                    'url': f'/snapshots/{filename}',
                    'timestamp': datetime.fromtimestamp(mtime).strftime('%H:%M:%S'),
                    'date': datetime.fromtimestamp(mtime).strftime('%m/%d'),
                    'size': f"{size // 1024}KB"
                })
            except:
                continue

    return snapshots

def create_sample_data():
    """Create sample data if logs are empty for demonstration"""
    detections = read_detection_logs()
    patterns = read_pattern_logs()
    
    if len(detections) < 5:
        print("📝 Creating sample detection data...")
        ensure_log_files()
        
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            now = datetime.now()
            
            sample_data = [
                ('Anzar', now - timedelta(hours=1), 'person(0.95);backpack(0.87)', '0.950'),
                ('Unknown', now - timedelta(hours=2), 'person(0.67)', '0.670'),
                ('Anzar', now - timedelta(hours=3), 'person(0.92);car(0.78)', '0.920'),
                ('John', now - timedelta(hours=4), 'person(0.89)', '0.890'),
                ('Unknown', now - timedelta(hours=5), 'person(0.65);motorcycle(0.55)', '0.650'),
                ('Anzar', now - timedelta(minutes=30), 'person(0.96)', '0.960'),
                ('Sarah', now - timedelta(hours=6), 'person(0.91);handbag(0.72)', '0.910'),
            ]
            
            for name, timestamp, objects, confidence in sample_data:
                writer.writerow([
                    name,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'N/A',
                    objects,
                    confidence
                ])
    
    if len(patterns) < 3:
        print("📝 Creating sample pattern data...")
        with open(PATTERN_LOG, 'a', newline='') as f:
            writer = csv.writer(f)
            now = datetime.now()
            
            pattern_data = [
                ('Anzar', now - timedelta(hours=2), 100, 200, 'normal', 5),
                ('Unknown', now - timedelta(hours=1), 150, 250, 'loitering', 15),
                ('Anzar', now - timedelta(minutes=30), 120, 220, 'pacing', 8),
            ]
            
            for person_id, timestamp, x, y, activity, duration in pattern_data:
                writer.writerow([
                    person_id,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    x, y, activity, duration
                ])

# Routes
@app.route('/')
def dashboard():
    """Serve the enhanced dashboard HTML"""
    dashboard_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ArgusSystems | Surveillance Intelligence</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;1,400&family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
    <style>
        :root {
            --primary: #8B7355;
            --secondary: #A0956B;
            --accent: #D4AF37;
            --accent-secondary: #CD853F;
            --paper: #F5F5DC;
            --parchment: #FDF5E6;
            --ink: #2F2F2F;
            --sepia: #704214;
            --vintage-gold: #B8860B;
            --warm-gray: #8B8680;
            --cream: #FFFDD0;
            --old-lace: #FDF5E6;
            --success: #228B22;
            --warning: #DAA520;
            --danger: #8B0000;
            --glass: rgba(139, 115, 85, 0.1);
            --glass-border: rgba(139, 115, 85, 0.2);
        }
        * {margin: 0; padding: 0; box-sizing: border-box;}
        body {
            font-family: 'Crimson Text', serif;
            background: linear-gradient(135deg, var(--parchment) 0%, var(--paper) 50%, var(--old-lace) 100%);
            background-attachment: fixed;
            color: var(--ink);
            overflow-x: hidden;
            min-height: 100vh;
            position: relative;
        }
        body::before {
            content: '';
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: 
                radial-gradient(circle at 15% 85%, rgba(212, 175, 55, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 85% 15%, rgba(205, 133, 63, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(139, 115, 85, 0.03) 0%, transparent 50%);
            z-index: -2;
        }
        body::after {
            content: '';
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: 
                repeating-linear-gradient(
                    0deg,
                    transparent,
                    transparent 2px,
                    rgba(139, 115, 85, 0.01) 2px,
                    rgba(139, 115, 85, 0.01) 4px
                );
            z-index: -1;
            pointer-events: none;
        }
        .navbar {
            position: fixed; top: 0; width: 100%;
            background: rgba(245, 245, 220, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid var(--primary);
            box-shadow: 0 2px 10px rgba(139, 115, 85, 0.2);
            z-index: 1000;
            padding: 1rem 0;
        }
        .nav-container {
            max-width: 1400px;
            margin: 0 auto;
            display: flex; justify-content: space-between; align-items: center; padding: 0 2rem;
        }
        .logo {
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            font-weight: 900;
            color: var(--primary);
            text-shadow: 2px 2px 4px rgba(139, 115, 85, 0.3);
            letter-spacing: 2px;
        }
        .system-badge {
            background: var(--accent);
            color: var(--ink);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: 600;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
        }
        .hero-section {
            height: 100vh;
            display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;
            position: relative;
            background: radial-gradient(circle at center, var(--cream) 0%, var(--parchment) 100%);
        }
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 5rem;
            font-weight: 900;
            color: var(--primary);
            margin-bottom: 1rem;
            text-shadow: 3px 3px 6px rgba(139, 115, 85, 0.3);
            letter-spacing: 3px;
        }
        .hero-subtitle {
            font-size: 1.6rem;
            color: var(--sepia);
            margin-bottom: 2rem;
            font-weight: 400;
            font-style: italic;
            letter-spacing: 1px;
        }
        .system-status {
            display: flex;
            align-items: center;
            gap: 1rem;
            background: var(--glass);
            backdrop-filter: blur(10px);
            border: 2px solid var(--glass-border);
            border-radius: 30px;
            padding: 1rem 2rem;
            box-shadow: 0 4px 15px rgba(139, 115, 85, 0.2);
        }
        .status-dot {
            width: 14px; height: 14px; border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 10px var(--success);
            animation: vintage-pulse 2s infinite;
        }
        @keyframes vintage-pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.7; transform: scale(1.1); }
        }
        .main-content { padding: 4rem 2rem; max-width: 1400px; margin: 0 auto; }
        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 2.8rem;
            font-weight: 700;
            margin-bottom: 2rem;
            text-align: center;
            color: var(--primary);
            text-shadow: 2px 2px 4px rgba(139, 115, 85, 0.2);
            position: relative;
        }
        .section-title::after {
            content: '';
            position: absolute;
            bottom: -10px;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 3px;
            background: linear-gradient(to right, var(--accent), var(--accent-secondary));
            border-radius: 2px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin-bottom: 3rem;
        }
        .stat-card {
            background: var(--cream);
            border: 2px solid var(--warm-gray);
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            box-shadow: 0 4px 12px rgba(139, 115, 85, 0.15);
        }
        .stat-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(139, 115, 85, 0.25);
            border-color: var(--accent);
        }
        .stat-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(to right, var(--accent), var(--accent-secondary));
            border-radius: 15px 15px 0 0;
        }
        .stat-icon { font-size: 3rem; margin-bottom: 1rem; color: var(--primary);}
        .stat-value {
            font-family: 'Libre Baskerville', serif;
            font-size: 3rem; font-weight: 700; color: var(--ink); margin-bottom: 0.5rem;
        }
        .stat-label {
            font-size: 1.1rem; color: var(--sepia); font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
        }
        .charts-section {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-bottom: 4rem;
        }
        .chart-container {
            background: var(--cream);
            border: 2px solid var(--warm-gray);
            border-radius: 15px;
            padding: 2rem;
            position: relative;
            box-shadow: 0 4px 12px rgba(139, 115, 85, 0.15);
        }
        .chart-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--ink);
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        .chart-title i { color: var(--primary);}
        .enhanced-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 2rem;
            margin-bottom: 4rem;
        }
        .activity-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 4rem;
        }
        .activity-feed {
            background: var(--cream);
            border: 2px solid var(--warm-gray);
            border-radius: 15px;
            padding: 2rem;
            max-height: 500px;
            overflow-y: auto;
            box-shadow: 0 4px 12px rgba(139, 115, 85, 0.15);
        }
        .activity-item {
            display: flex;
            align-items: center;
            gap: 1rem;
            padding: 1rem;
            border-bottom: 1px solid var(--glass-border);
            transition: all 0.3s ease;
            border-radius: 8px;
            margin-bottom: 0.5rem;
        }
        .activity-item:hover {
            background: var(--glass);
            transform: translateX(5px);
        }
        .activity-icon {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
        }
        .activity-icon.success {
            background: rgba(34, 139, 34, 0.2);
            color: var(--success);
            border: 2px solid var(--success);
        }
        .activity-icon.warning {
            background: rgba(218, 165, 32, 0.2);
            color: var(--warning);
            border: 2px solid var(--warning);
        }
        .activity-icon.danger {
            background: rgba(139, 0, 0, 0.2);
            color: var(--danger);
            border: 2px solid var(--danger);
        }
        .people-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        .person-card {
            background: var(--parchment);
            border: 2px solid var(--warm-gray);
            border-radius: 15px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(139, 115, 85, 0.1);
        }
        .person-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 20px rgba(139, 115, 85, 0.2);
            border-color: var(--accent);
        }
        .person-avatar {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            margin: 0 auto 1rem;
            background: linear-gradient(45deg, var(--primary), var(--secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: 700;
            color: white;
            border: 3px solid var(--accent);
        }
        .snapshots-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 1.5rem;
            margin-top: 1.5rem;
        }
        .snapshot-item {
            aspect-ratio: 16/9;
            border-radius: 12px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
            background: var(--parchment);
            border: 2px solid var(--warm-gray);
            box-shadow: 0 3px 10px rgba(139, 115, 85, 0.1);
        }
        .snapshot-item:hover {
            transform: scale(1.05) rotate(1deg);
            box-shadow: 0 6px 20px rgba(139, 115, 85, 0.3);
            border-color: var(--accent);
        }
        .snapshot-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .snapshot-overlay {
            position: absolute;
            bottom: 0; left: 0; right: 0;
            background: linear-gradient(transparent, rgba(47, 47, 47, 0.9));
            color: white;
            padding: 1rem 0.5rem;
            font-size: 0.85rem;
            text-align: center;
        }
        .behavior-insights {
            background: var(--cream);
            border: 2px solid var(--warm-gray);
            border-radius: 15px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(139, 115, 85, 0.15);
        }
        .insight-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }
        .insight-card {
            background: var(--parchment);
            border: 1px solid var(--glass-border);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
        }
        .insight-number {
            font-family: 'Libre Baskerville', serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        .insight-label {
            color: var(--sepia);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.9rem;
        }
        .loading {
            display: inline-block;
            width: 24px; height: 24px;
            border: 3px solid var(--glass-border);
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            animation: vintage-spin 1s linear infinite;
        }
        @keyframes vintage-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .vintage-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            background: var(--parchment);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(139, 115, 85, 0.1);
        }
        .vintage-table th,
        .vintage-table td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--glass-border);
        }
        .vintage-table th {
            background: var(--primary);
            color: var(--cream);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .vintage-table tbody tr:hover { background: var(--glass);}
        .error-message {
            text-align: center;
            color: var(--danger);
            font-style: italic;
            padding: 2rem;
        }
        .no-data {
            text-align: center;
            color: var(--sepia);
            font-style: italic;
            padding: 2rem;
        }
        @media (max-width: 1024px) {
            .charts-section, .activity-section { grid-template-columns: 1fr; }
            .enhanced-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 768px) {
            .hero-title { font-size: 3rem; }
            .stats-grid { grid-template-columns: 1fr;}
            .nav-container { flex-direction: column; gap: 1rem;}
        }
        ::-webkit-scrollbar { width: 12px;}
        ::-webkit-scrollbar-track { background: var(--parchment); border-radius: 6px;}
        ::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 6px;}
        ::-webkit-scrollbar-thumb:hover { background: var(--secondary);}
    </style>
</head>
<body>
    <nav class="navbar" id="navbar">
        <div class="nav-container">
            <div class="logo">ArgusSystems</div>
            <div class="system-badge">
                <i class="fas fa-shield-alt"></i>
                Smart Intelligence
            </div>
        </div>
    </nav>
    <section class="hero-section" id="dashboard">
        <h1 class="hero-title">ARGUS SYSTEMS</h1>
        <p class="hero-subtitle">-Intelligent Surveillance, Accessible for All-</p>
        <div class="system-status">
            <div class="status-dot"></div>
            <span id="system-status-text">System Operational</span>
        </div>
    </section>
    <main class="main-content">
        <section class="stats-section" id="analytics">
            <h2 class="section-title">System Analytics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-eye"></i></div>
                    <div class="stat-value" id="total-detections">0</div>
                    <div class="stat-label">Total Detections</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-users"></i></div>
                    <div class="stat-value" id="known-people">0</div>
                    <div class="stat-label">Known Persons</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-exclamation-triangle"></i></div>
                    <div class="stat-value" id="unknown-alerts">0</div>
                    <div class="stat-label">Unknown Alerts</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-percentage"></i></div>
                    <div class="stat-value" id="detection-accuracy">0%</div>
                    <div class="stat-label">Detection Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-clock"></i></div>
                    <div class="stat-value" id="uptime">0h</div>
                    <div class="stat-label">System Uptime</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-star"></i></div>
                    <div class="stat-value" id="avg-confidence">0.0</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-cube"></i></div>
                    <div class="stat-value" id="most-detected-object">None</div>
                    <div class="stat-label">Top Object Detected</div>
                </div>
                <div class="stat-card">
                    <div class="stat-icon"><i class="fas fa-calendar-week"></i></div>
                    <div class="stat-value" id="weekly-detections">0</div>
                    <div class="stat-label">This Week</div>
                </div>
            </div>
        </section>
        <section class="charts-section">
            <div class="chart-container">
                <h3 class="chart-title"><i class="fas fa-chart-bar"></i>Activity Timeline (Last 24 Hours)</h3>
                <canvas id="activityChart" height="300"></canvas>
            </div>
            <div class="chart-container">
                <h3 class="chart-title"><i class="fas fa-user-clock"></i>Person Frequency Analysis</h3>
                <table class="vintage-table" id="person-frequency-table">
                    <thead>
                        <tr>
                            <th>Person</th>
                            <th>Detections</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="person-frequency-body">
                        <tr><td colspan="3"><div class="loading"></div></td></tr>
                    </tbody>
                </table>
            </div>
        </section>
        <section class="activity-section">
            <div class="activity-feed" id="activity">
                <h3 class="chart-title"><i class="fas fa-bell"></i>Recent Activity Log</h3>
                <div id="activity-list"><div class="loading"></div></div>
            </div>
            <div class="chart-container" id="people">
                <h3 class="chart-title"><i class="fas fa-user-friends"></i>Registered Personnel</h3>
                <div class="people-grid" id="people-grid">
                    <div class="loading"></div>
                </div>
            </div>
        </section>
        <section class="chart-container">
            <h3 class="chart-title"><i class="fas fa-camera-retro"></i>Recent Surveillance Captures</h3>
            <div class="snapshots-grid" id="snapshots-grid">
                <div class="loading"></div>
            </div>
        </section>
        <section class="chart-container">
            <h3 class="chart-title"><i class="fas fa-chart-area"></i>Object Detection Intelligence</h3>
            <div class="insight-grid" id="object-insights">
                <div class="loading"></div>
            </div>
        </section>
    </main>
    <script>
// Global chart variables
let activityChart;

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Dashboard initializing...');
    initializeCharts();
    loadDashboardData();
    setInterval(loadDashboardData, 10000); // Update every 10 seconds
});

function initializeCharts() {
    try {
        const activityCtx = document.getElementById('activityChart').getContext('2d');
        activityChart = new Chart(activityCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                datasets: [{
                    label: 'Detections',
                    data: new Array(24).fill(0),
                    backgroundColor: 'rgba(139, 115, 85, 0.2)',
                    borderColor: '#8B7355',
                    borderWidth: 3,
                    pointBackgroundColor: '#D4AF37',
                    pointBorderColor: '#2F2F2F',
                    pointRadius: 6,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#2F2F2F', font: { family: 'Crimson Text' } }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(139, 115, 85, 0.3)' },
                        ticks: {
                            color: '#704214',
                            font: { family: 'Crimson Text' }
                        }
                    },
                    x: {
                        grid: { color: 'rgba(139, 115, 85, 0.3)' },
                        ticks: {
                            color: '#704214',
                            font: { family: 'Crimson Text' }
                        }
                    }
                }
            }
        });
        console.log('✅ Charts initialized');
    } catch (error) {
        console.error('❌ Error initializing charts:', error);
    }
}

async function loadDashboardData() {
    try {
        console.log('📡 Fetching dashboard data...');
        const response = await fetch('/api/enhanced-dashboard-data');
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('📊 Data received:', data);
        
        updateEnhancedStats(data.stats);
        updateActivity(data.activity);
        updateCharts(data);
        updateSnapshots(data.snapshots);
        updateObjectInsights(data.object_insights);
        loadPeople();
        
        console.log('✅ Dashboard updated successfully');
        
    } catch (error) {
        console.error('❌ Failed to load dashboard data:', error);
        showErrorState();
    }
}

function updateEnhancedStats(stats) {
    try {
        if (!stats) {
            console.warn('⚠️ No stats data provided');
            return;
        }
        
        console.log('📈 Updating stats:', stats);
        
        document.getElementById('total-detections').textContent = stats.total_detections || 0;
        document.getElementById('known-people').textContent = stats.known_people || 0;
        document.getElementById('unknown-alerts').textContent = stats.unknown_alerts || 0;
        document.getElementById('detection-accuracy').textContent = (stats.detection_accuracy || 0) + '%';
        document.getElementById('uptime').textContent = stats.uptime || '0h';
        document.getElementById('avg-confidence').textContent = stats.avg_confidence || '0.0';
        document.getElementById('most-detected-object').textContent = stats.most_detected_object || 'None';
        document.getElementById('weekly-detections').textContent = stats.recent_detections_week || 0;
        
    } catch (error) {
        console.error('❌ Error updating stats:', error);
    }
}

function updateCharts(data) {
    try {
        // Activity Chart
        if (data.hourly_stats && data.hourly_stats['24h']) {
            activityChart.data.datasets[0].data = data.hourly_stats['24h'];
            activityChart.update('none');
        }
        
        // Person Frequency Table
        if (data.behavioral_insights && data.behavioral_insights.person_frequency) {
            updatePersonFrequencyTable(data.behavioral_insights.person_frequency);
        }
        
        console.log('📊 Charts updated');
        
    } catch (error) {
        console.error('❌ Error updating charts:', error);
    }
}

function updatePersonFrequencyTable(personFreq) {
    try {
        const tbody = document.getElementById('person-frequency-body');
        
        if (!personFreq || Object.keys(personFreq).length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" class="no-data">No person data available</td></tr>';
            return;
        }
        
        const sortedPersons = Object.entries(personFreq)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10);
        
        tbody.innerHTML = sortedPersons.map(([name, count]) => `
            <tr>
                <td style="font-weight: 600; color: var(--primary);">${name}</td>
                <td>${count}</td>
                <td style="color: var(--success);">Active</td>
            </tr>
        `).join('');
        
    } catch (error) {
        console.error('❌ Error updating person frequency table:', error);
    }
}

function updateActivity(activities) {
    try {
        const activityList = document.getElementById('activity-list');
        
        if (!activities || activities.length === 0) {
            activityList.innerHTML = '<div class="no-data">No recent surveillance activity detected</div>';
            return;
        }
        
        activityList.innerHTML = activities.map(activity => `
            <div class="activity-item">
                <div class="activity-icon ${activity.type}">
                    <i class="fas ${activity.icon}"></i>
                </div>
                <div class="activity-text" style="color: var(--ink); font-weight: 500;">
                    ${activity.text}
                </div>
            </div>
        `).join('');
        
        console.log('📝 Activity updated');
        
    } catch (error) {
        console.error('❌ Error updating activity:', error);
    }
}

async function loadPeople() {
    try {
        const response = await fetch('/api/known-people');
        const people = await response.json();
        updatePeople(people);
    } catch (error) {
        console.error('❌ Failed to load people:', error);
        updatePeople([]);
    }
}

function updatePeople(people) {
    try {
        const peopleGrid = document.getElementById('people-grid');
        
        if (!people || people.length === 0) {
            peopleGrid.innerHTML = '<div class="no-data">No registered personnel found</div>';
            return;
        }
        
        peopleGrid.innerHTML = people.map(person => `
            <div class="person-card">
                <div class="person-avatar">${person.name[0].toUpperCase()}</div>
                <div style="color: var(--ink); font-weight: 600; margin-bottom: 0.5rem; font-family: 'Playfair Display', serif;">${person.name}</div>
                <div style="color: var(--sepia); font-size: 0.95rem;">${person.images} photographs</div>
                <div style="color: var(--primary); font-size: 0.85rem; font-weight: 600;">Last: ${person.lastSeen}</div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('❌ Error updating people:', error);
    }
}

function updateSnapshots(snapshots) {
    try {
        const snapshotsGrid = document.getElementById('snapshots-grid');
        
        if (!snapshots || snapshots.length === 0) {
            snapshotsGrid.innerHTML = '<div class="no-data">No surveillance captures available</div>';
            return;
        }
        
        snapshotsGrid.innerHTML = snapshots.map(snapshot => `
            <div class="snapshot-item">
                <img src="${snapshot.url}" alt="Surveillance Capture" 
                     onerror="this.style.display='none'; this.parentNode.innerHTML='<div class=\\'no-data\\'>Image not found</div>'">
                <div class="snapshot-overlay">
                    <div style="font-weight: 600;">${snapshot.timestamp}</div>
                    <div style="font-size: 0.75rem; opacity: 0.8;">${snapshot.date} • ${snapshot.size}</div>
                </div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('❌ Error updating snapshots:', error);
    }
}

function updateObjectInsights(objectInsights) {
    try {
        const container = document.getElementById('object-insights');
        
        if (!objectInsights || Object.keys(objectInsights).length === 0) {
            container.innerHTML = '<div class="no-data">No object detection data available</div>';
            return;
        }
        
        const topObjects = Object.entries(objectInsights)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 6);
        
        container.innerHTML = topObjects.map(([object, count]) => `
            <div class="insight-card">
                <div class="insight-number">${count}</div>
                <div class="insight-label">${object}</div>
            </div>
        `).join('');
        
    } catch (error) {
        console.error('❌ Error updating object insights:', error);
    }
}

function showErrorState() {
    // Show error state for all sections
    const errorHtml = '<div class="error-message">Unable to load data. Please check if surveillance system is running.</div>';
    
    document.getElementById('activity-list').innerHTML = errorHtml;
    document.getElementById('people-grid').innerHTML = errorHtml;
    document.getElementById('snapshots-grid').innerHTML = errorHtml;
    document.getElementById('object-insights').innerHTML = errorHtml;
}
    </script>
</body>
</html>"""
    return dashboard_html

@app.route('/api/enhanced-dashboard-data')
def enhanced_dashboard_data():
    try:
        print("🔄 Processing dashboard data request...")
        
        detections = read_detection_logs()
        patterns = read_pattern_logs()
        recent_detections = detections[-20:] if detections else []

        print(f"📊 Found {len(detections)} detections, {len(patterns)} patterns")

        # Build activity log
        activity = []
        for d in reversed(recent_detections):
            try:
                names = d.get('Name', 'Unknown').split(';')
                timestamp = d.get('Timestamp', 'Unknown time')
                objects = d.get('Objects', '')
                confidence = d.get('Confidence', '')

                for name in names:
                    name = name.strip()
                    if name:
                        activity_text = f"{name} detected at {timestamp}"
                        if objects and objects != 'N/A':
                            obj_list = [obj.split('(')[0] for obj in objects.split(';')[:2] if '(' in obj]
                            if obj_list:
                                activity_text += f" with {', '.join(obj_list)}"
                        if confidence and confidence != 'N/A':
                            activity_text += f" (Confidence: {confidence})"

                        activity.append({
                            'type': 'danger' if 'Unknown' in name else 'success',
                            'icon': 'fa-exclamation-triangle' if 'Unknown' in name else 'fa-user-check',
                            'text': activity_text
                        })
            except Exception as e:
                print(f"⚠️ Error processing detection activity: {e}")

        # Add pattern activities
        for p in patterns[-5:]:
            try:
                activity_type = p.get('Activity', 'normal')
                if activity_type in ['loitering', 'pacing', 'running']:
                    timestamp = p.get('Timestamp', 'Unknown time')
                    person_id = p.get('PersonID', 'Unknown')
                    activity.append({
                        'type': 'warning',
                        'icon': 'fa-walking',
                        'text': f"Behavioral pattern: {activity_type} detected (Person {person_id}) at {timestamp}"
                    })
            except Exception as e:
                print(f"⚠️ Error processing pattern activity: {e}")

        stats = get_enhanced_stats()
        behavioral_insights = get_behavioral_insights()
        hourly_stats = get_hourly_stats(detections)
        snapshots = get_recent_snapshots()

        response_data = {
            'stats': stats,
            'activity': activity[:15],
            'hourly_stats': hourly_stats,
            'snapshots': snapshots,
            'behavioral_insights': behavioral_insights,
            'object_insights': stats.get('object_detections', {})
        }

        print(f"✅ Dashboard data prepared successfully with {len(activity)} activities")
        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in enhanced_dashboard_data: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty data structure to prevent frontend errors
        return jsonify({
            'stats': {
                'total_detections': 0,
                'known_people': 0,
                'unknown_alerts': 0,
                'uptime': '0h',
                'detection_accuracy': 0,
                'avg_confidence': 0.0,
                'most_detected_object': 'None',
                'recent_detections_week': 0,
                'behavior_patterns': {},
                'object_detections': {}
            },
            'activity': [],
            'hourly_stats': {'24h': [0] * 24, 'week': [0] * 24},
            'snapshots': [],
            'behavioral_insights': {'person_frequency': {}},
            'object_insights': {}
        })

@app.route('/api/dashboard-data')
def dashboard_data():
    return enhanced_dashboard_data()

@app.route('/api/known-people')
def known_people():
    try:
        detections = read_detection_logs()
        people = []

        # Build last seen map
        last_seen_map = {}
        for d in reversed(detections):
            try:
                names = d.get('Name', '').split(';')
                timestamp = d.get('Timestamp', '')
                for name in names:
                    name = name.strip()
                    if name and name != 'Unknown' and name not in last_seen_map:
                        last_seen_map[name] = timestamp
            except:
                continue

        # Get people from known faces directory
        if os.path.exists(KNOWN_FACES_DIR):
            try:
                for person in os.listdir(KNOWN_FACES_DIR):
                    person_dir = os.path.join(KNOWN_FACES_DIR, person)
                    if os.path.isdir(person_dir):
                        try:
                            images = len([f for f in os.listdir(person_dir)
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                            last_seen = 'Never'
                            if person in last_seen_map:
                                try:
                                    dt = datetime.strptime(last_seen_map[person], '%Y-%m-%d %H:%M:%S')
                                    diff = datetime.now() - dt
                                    if diff.days > 0:
                                        last_seen = f'{diff.days} days ago'
                                    elif diff.seconds > 3600:
                                        last_seen = f'{diff.seconds // 3600} hours ago'
                                    elif diff.seconds > 60:
                                        last_seen = f'{diff.seconds // 60} minutes ago'
                                    else:
                                        last_seen = 'Just now'
                                except:
                                    pass

                            people.append({
                                'name': person,
                                'images': images,
                                'lastSeen': last_seen
                            })
                        except Exception as e:
                            print(f"Error processing person {person}: {e}")
            except Exception as e:
                print(f"Error reading known faces directory: {e}")

        print(f"👥 Found {len(people)} known people")
        return jsonify(people)

    except Exception as e:
        print(f"❌ Error getting known people: {e}")
        return jsonify([])

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    try:
        return send_from_directory(SNAPSHOT_DIR, filename)
    except Exception as e:
        print(f"❌ Error serving snapshot {filename}: {e}")
        return "File not found", 404

@app.route('/api/stats')
def stats():
    return jsonify(get_enhanced_stats())

@app.route('/api/behavioral-patterns')
def behavioral_patterns():
    return jsonify(get_behavioral_insights())

if __name__ == '__main__':
    print(f"""
📜 ArgusSystems Enhanced Dashboard Starting...
📁 Log Directory: {LOG_DIR}
📸 Snapshots: {SNAPSHOT_DIR}
👥 Known Faces: {KNOWN_FACES_DIR}
🌐 Dashboard: http://localhost:5000
🔧 Enhanced with robust error handling and data validation
    """)
    
    # Create sample data for demonstration
    create_sample_data()
    
    app.run(debug=True, host='0.0.0.0', port=5000)