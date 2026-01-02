#!/usr/bin/env python3
"""
Unified AI Tracking + Guess-and-Stop Safety + Emergency Stop + Surgeon Override Capture
with On-Screen Alerts

Continuous danger alarm (safe, non-freezing implementation)
"""

import cv2
import tempfile
import os
import numpy as np
import socket
import time
import sys
import subprocess
from datetime import datetime
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import speech_recognition as sr
import pyttsx3
import threading
import queue
import subprocess

# ================================
# TTS INIT
# ================================
try:
    # Keep pyttsx3 for startup check, but use 'say' for runtime async
    tts_engine = pyttsx3.init()
    tts_engine.say("TTS initialized successfully")
    tts_engine.runAndWait()
    print("TTS test successful")
except Exception as e:
    print(f"TTS init/test error: {e}")
    tts_engine = None

# Thread-safe speech queue preventing thread explosion
speech_queue = queue.Queue(maxsize=10)

def speech_worker_thread():
    """Background worker to process speech commands sequentially."""
    while True:
        text = speech_queue.get()
        if text is None: break  # Sentinel to stop
        try:
            subprocess.run(["say", text], check=False)
        except Exception as e:
            print(f"Speech error: {e}")
        speech_queue.task_done()

# Start speech worker once
threading.Thread(target=speech_worker_thread, daemon=True).start()

def speak_async(text):
    """Non-blocking speech request using the shared queue. Drops old alerts if full."""
    try:
        speech_queue.put_nowait(text)
    except queue.Full:
        pass # Drop alert if queue is backed up to prevent lag

# ================================
# CONFIG
# ================================
TRACK_CLASS        = "vertebral-body-tumor"
ALLOWED_LABELS     = {"vertebral-body-tumor"}
FORBIDDEN_LABELS   = {"nerve"}
SENSITIVE_CLASSES  = {"nerve", "SpinalCord"}

AUTO_TRACK         = True
CONFIDENCE_THRESH  = 0.40
IOU_THRESH         = 0.65
TOLERANCE_PER      = 0.10
CAUTION_DISTANCE_MM = 15.0   # Warn surgeon and halt robot early
CRITICAL_DISTANCE_MM = 5.0   # Pulsing "DANGER" overlay only at extreme proximity
MIN_DISTANCE_MM    = 20.0
PIXEL_TO_MM        = 1.0
HORIZ_OFFSET_PX    = 50
VERT_OFFSET_PX     = -30

ROBOT_HOST         = "moonshot5.local"
ROBOT_ON           = False
RETRACT_CMD        = "move -1 -1"
HALT_CMD           = "move 0 0"
ROBOT_AI_SPEED     = 12
AI_SPEED_INC       = 4

RF_MODEL_ID        = "Roboflow model_url"
RF_API_KEY         = "RoboFlow_API_key"
RF_API_URL         = "https://detect.roboflow.com"
INF_W, INF_H       = 640, 640  # Standard AI processing size

CAMERA_PORT        = 0
DISPLAY_WINDOW     = "Frame"
RECORD_VIDEO       = False
LOOP_SLEEP_SEC     = 0.05
OVERRIDE_DIR       = "override_frames"

# ================================
# SETUP
# ================================
try:
    config = InferenceConfiguration(CONFIDENCE_THRESH, IOU_THRESH)
    client = InferenceHTTPClient(api_url=RF_API_URL, api_key=RF_API_KEY)
    client.configure(config)
    client.select_model(RF_MODEL_ID)
except Exception as e:
    print(f"Failed to initialize Roboflow client: {e}")
    sys.exit(1)

print("Initializing camera...")
if sys.platform.startswith("win"): # Windows
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_MSMF)
elif sys.platform.startswith("darwin"): # macOS
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAMERA_PORT) # Linux or other

if not cap.isOpened():
    print("Camera error")
    sys.exit(1)

# Request best resolution but target 1080p UI
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Native capture size
_fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
_fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
print(f"Native capture: {_fw}x{_fh}")

# Standardized high-res display dimensions
FH = 4000
FW = int(_fw * (FH / _fh))
UI_SCALE = FH / 720.0
print(f"Upscaled UI dimensions: {FW}x{FH}")

if RECORD_VIDEO:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(f"record_{ts}.mp4", cv2.VideoWriter_fourcc(*"XVID"), 7.0, (FW, FH))
    print("Video recording is enabled.")
else:
    out = None

# Side panel configuration
SIDE_PANEL_WIDTH = int(500 * UI_SCALE)
TOTAL_WIDTH = FW + SIDE_PANEL_WIDTH
TOTAL_HEIGHT = FH

# ================================
# STATE
# ================================
ai_mode = False
running = True
system_active = True
lock = False                    # True when target is centered within tolerance
last_ai_mode_sent = False       # Tracks if AI mode command was sent to robot

last_alert_text = ""            # Text for on-screen alert
last_alert_ts = 0.0             # Timestamp when alert was set (for fade-out)
last_distance_text = ""         # Text for distance display (e.g., to closest forbidden)
last_tracked_tumor_text = ""    # Text for tracked tumor info

is_danger = False               # Flag for danger overlay
last_is_danger = False          # To track danger state changes
danger_pulse = 0.0              # Pulsing animation for danger state

# position_history = []           # Removed (unused)
initial_sep_dist = None         # To keep constant separation distance between two tumors
initial_tumor_size = None       # To keep constant initial tumor size
locked_tumour_dist = None       # Locked distance when tumor is in "red" proximity

# UI Animation state
fps_counter = 0
last_fps_time = time.time()
current_fps = 0
connection_status = "Disconnected"
last_inference_success = 0.0

# Expert Features State
distance_history = []           # Last 100 frames of closest forbidden distance
severity_history = []           # For temporal smoothing of danger states
event_log = []                  # List of (timestamp, event_text)
MAX_EVENT_LOG = 5
MAX_DIST_HISTORY = 100
MAX_SEVERITY_HISTORY = 5        # Frames to smooth
TRACKING_SMOOTH_HISTORY = 3     # Window for fluid robot motion
tracking_history = []           # Rolling average of tumor centers

# Background Inference State
inference_queue = queue.Queue(maxsize=1)
latest_predictions = []
inference_lock = threading.Lock()
last_command_time = 0
COMMAND_THROTTLE_SEC = 0.05

def inference_worker():
    """Background thread to handle potentially slow cloud inference calls."""
    global latest_predictions, connection_status, last_inference_success
    while running:
        try:
            # Wait for a new frame, but don't block forever so we can check 'running'
            frame = inference_queue.get(timeout=1)
            if frame is None: continue
            
            resp = client.infer(frame)
            preds = resp.get("predictions", [])
            
            with inference_lock:
                latest_predictions = preds
                if connection_status != "Connected":
                    speak_async("Inference Connected")
                connection_status = "Connected"
                last_inference_success = time.time()
                
            inference_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            with inference_lock:
                connection_status = "Disconnected"
            time.sleep(1) # Backoff on error

def add_event(text):
    """Add an event to the rolling event log with a color-coded timestamp."""
    global event_log
    ts = datetime.now().strftime("%H:%M:%S")
    event_log.insert(0, (ts, text))
    if len(event_log) > MAX_EVENT_LOG:
        event_log.pop()
    print(f"EVENT: {ts} - {text}")

# ================================
# SAFE CONTINUOUS DANGER SOUND
# ================================
danger_sound_process = None

def play_danger_sound(loop=False): # Plays a danger sound, optionally looping
    global danger_sound_process
    if sys.platform.startswith("darwin") and loop: # macOS specific sound command
        if danger_sound_process is not None:
            if danger_sound_process.poll() is not None:
                danger_sound_process = None
        
        if danger_sound_process is None: # If not playing or finished
            try: # Use subprocess for robust background sound playback
                danger_sound_process = subprocess.Popen(["afplay", "-r", "0.5", "/System/Library/Sounds/Basso.aiff"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e: print(f"Sound play error: {e}")

def stop_danger_sound():
    global danger_sound_process
    if danger_sound_process is not None:
        try:
            danger_sound_process.terminate()
            danger_sound_process.wait(timeout=1)
        except:
            pass # Ignore if process is already dead or termination fails
        danger_sound_process = None

# ================================
# MODERN UI HELPER FUNCTIONS
# ================================
def draw_rounded_rect(img, pt1, pt2, color, thickness=1, radius=10, fill=False):
    """Draw a rounded rectangle with dynamic scaling."""
    radius = int(radius * UI_SCALE)
    if thickness > 0:
        thickness = max(1, int(thickness * UI_SCALE))
    
    x1, y1 = pt1
    x2, y2 = pt2
    
    if fill:
        thickness = -1
    
    # Draw the main rectangle
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw the corners
    if fill:
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)
    else:
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

# ================================
# DRAWING HELPERS (PREMIUM UI)
# ================================
CYBER_CYAN = (255, 240, 0)   # BGR
DEEP_SLATE = (40, 40, 50)    # BGR
GLASS_ALPHA = 0.6

def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=10):
    """Draws a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    
    # Clamping radius
    radius = min(radius, w // 2, h // 2)
    
    # Draw straight parts
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Draw corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_glass_panel(img, pt1, pt2, color=(40, 40, 40), alpha=0.7, border_color=(100, 200, 255), border_thickness=2):
    """Draw a glassmorphism-style panel using optimized ROI processing."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Safety: Clamp to image bounds
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1: return

    # Optimization: Work only on the ROI (Region of Interest)
    # This avoids copying the entire 4K/8K buffer for every small label
    roi = img[y1:y2, x1:x2]
    overlay = roi.copy()
    
    # Draw filled rounded rect on local ROI coordinates (0, 0) -> (width, height)
    local_w = x2 - x1
    local_h = y2 - y1
    draw_rounded_rect(overlay, (0, 0), (local_w, local_h), color, thickness=-1, radius=15)
    
    # Blend and place back
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi
    
    # Draw border on global image for sharpness
    draw_rounded_rect(img, (x1, y1), (x2, y2), border_color, thickness=border_thickness, radius=15)

def draw_outlined_text(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.6, color=(255, 255, 255), thickness=2, outline_color=(0, 0, 0), outline_thickness=4):
    """Draw text with scaled thickness and size."""
    scaled_scale = scale * UI_SCALE
    scaled_thickness = max(1, int(thickness * UI_SCALE))
    scaled_outline = max(1, int(outline_thickness * UI_SCALE))
    
    x, y = pos
    # Draw outline
    cv2.putText(img, text, (x, y), font, scaled_scale, outline_color, scaled_outline, cv2.LINE_AA)
    # Draw text
    cv2.putText(img, text, (x, y), font, scaled_scale, color, scaled_thickness, cv2.LINE_AA)

def draw_status_indicator(img, pos, status, label=""):
    """Draw a status indicator dot with dynamic scaling."""
    x, y = pos
    colors = {
        "active": (0, 255, 0),
        "inactive": (100, 100, 100),
        "danger": (0, 0, 255),
        "warning": (0, 165, 255)
    }
    color = colors.get(status, (255, 255, 255))
    
    base_r = int(8 * UI_SCALE)
    outer_r = int(12 * UI_SCALE)
    
    # Draw pulsing circle for active states
    if status in ["active", "danger"]:
        pulse = int(abs(np.sin(time.time() * 3)) * (5 * UI_SCALE))
        cv2.circle(img, (x, y), base_r + pulse, color, -1)
        cv2.circle(img, (x, y), outer_r + pulse, color, max(1, int(2 * UI_SCALE)))
    else:
        cv2.circle(img, (x, y), base_r, color, -1)
    
    if label:
        draw_outlined_text(img, label, (x + int(20 * UI_SCALE), y + int(5 * UI_SCALE)), scale=0.5, thickness=1, outline_thickness=3)

def get_distance_color(distance_mm, threshold=MIN_DISTANCE_MM):
    """Get color based on distance (green -> yellow -> red)."""
    if distance_mm >= threshold * 2:
        return (0, 255, 0)  # Green - safe
    elif distance_mm >= threshold:
        ratio = (distance_mm - threshold) / threshold
        return (0, int(255 * ratio), int(255 * (1 - ratio)))  # Yellow - caution
    else:
        return (0, 0, 255)  # Red - danger

def draw_proximity_radar(img, pos, objects, radius=80):
    """Draw a topographic radar with dynamic scaling."""
    radius = int(radius * UI_SCALE)
    cx, cy = pos
    
    # Draw radar circles
    for r in [radius, radius*0.6, radius*0.3]:
        cv2.circle(img, (cx, cy), int(r), (80, 80, 100), max(1, int(1 * UI_SCALE)))
    
    # Axis lines
    cv2.line(img, (cx-radius, cy), (cx+radius, cy), (80, 80, 100), max(1, int(1 * UI_SCALE)))
    cv2.line(img, (cx, cy-radius), (cx, cy+radius), (80, 80, 100), max(1, int(1 * UI_SCALE)))
    
    # Tool at center
    cv2.circle(img, (cx, cy), max(1, int(3 * UI_SCALE)), (255, 0, 255), -1)
    
    # Map objects
    for obj in objects:
        if obj.get("class") in FORBIDDEN_LABELS:
            obj_center = obj.get("center")
            if obj_center:
                # Calculate relative position scaled to radar
                dx = (obj_center[0] - FW/2) * (radius / (FW/2))
                dy = (obj_center[1] - FH/2) * (radius / (FH/2))
                
                # Constrain to radar circle
                dist = (dx**2 + dy**2)**0.5
                if dist > radius:
                    scale = radius / dist
                    dx *= scale
                    dy *= scale
                
                # Draw on radar
                color = get_distance_color(dist * PIXEL_TO_MM / (radius / 100)) # Simple scale
                cv2.circle(img, (int(cx + dx), int(cy + dy)), max(1, int(4 * UI_SCALE)), color, -1)

def draw_distance_graph(img, pos, history, width=320, height=80):
    """Draw a trend graph for distances."""
    x_start, y_start = pos
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (40, 40, 50), -1)
    cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (100, 100, 120), 1)
    
    if len(history) < 2:
        return
        
    points = []
    max_val = 150 # mm scale max
    for i, val in enumerate(history):
        px = x_start + int(i * (width / MAX_DIST_HISTORY))
        py = y_start + height - int(min(val, max_val) * (height / max_val))
        points.append((px, py))
        
    for i in range(len(points) - 1):
        color = get_distance_color(history[i])
        cv2.line(img, points[i], points[i+1], color, 2)


# ================================
# UTILITIES
# ================================
def ensure_ai_on():
    global last_ai_mode_sent
    if ROBOT_ON and not last_ai_mode_sent:
        sock.sendto(b"ai_mode on", ROBOT_ADDR)
        last_ai_mode_sent = True

def ensure_ai_off():
    global last_ai_mode_sent
    if ROBOT_ON and last_ai_mode_sent:
        sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
        sock.sendto(b"ai_mode off", ROBOT_ADDR)
        last_ai_mode_sent = False

def draw_detection(frame, det, color=(0, 255, 0), tool_center=None):
    """Draws a single detection on the frame with enhanced visuals."""
    pts = det.get("points", [])
    if not pts:
        return
    
    # Calculate center and distance to tool
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    
    # Auto-adjust color based on distance if tool_center provided
    if tool_center and det.get("class") in FORBIDDEN_LABELS:
        dist_px = ((tool_center[0] - cx)**2 + (tool_center[1] - cy)**2)**0.5
        dist_mm = dist_px * PIXEL_TO_MM
        color = get_distance_color(dist_mm)
    
    # Draw contour with thicker scaled line
    contour = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [contour], True, color, max(1, int(3 * UI_SCALE)))
    
    # Draw distance line to tool if applicable
    if tool_center and det.get("class") in FORBIDDEN_LABELS:
        cv2.line(frame, (int(tool_center[0]), int(tool_center[1])), (int(cx), int(cy)), color, max(1, int(2 * UI_SCALE)), cv2.LINE_AA)
        # Distance label on line midpoint
        mid_x = int((tool_center[0] + cx) / 2)
        mid_y = int((tool_center[1] + cy) / 2)
        dist_text = f"{dist_mm:.1f}mm"
        draw_outlined_text(frame, dist_text, (mid_x, mid_y), scale=0.5, thickness=1, outline_thickness=3)
    
    # Enhanced label with background
    display_name = det.get('class','Unknown').replace("-", " ").replace("_", " ").title()
    conf_val = det.get('confidence', 0)
    label = f"{display_name} [{conf_val:.0%}]"
    
    min_x = int(min(xs))
    min_y = int(min(ys))
    
    # Draw glass panel for label
    label_height = int(35 * UI_SCALE)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * UI_SCALE, max(1, int(1 * UI_SCALE)))
    
    # Position label above the detection
    label_y = min_y - int(10 * UI_SCALE)
    draw_glass_panel(frame, (min_x, label_y - label_height), (min_x + text_w + int(20 * UI_SCALE), label_y), 
                     color=(20, 20, 20), alpha=0.8, border_color=color, border_thickness=1)
    
    draw_outlined_text(frame, label, (min_x + int(10 * UI_SCALE), label_y - int(10 * UI_SCALE)), scale=0.5, thickness=1, color=(255, 255, 255))

def set_alert(text):
    """Sets an on-screen and TTS alert."""
    global last_alert_text, last_alert_ts
    now = time.time()
    if now - last_alert_ts < 0.8: # Throttle alerts to avoid spamming
        return # Don't set new alert if one was just set
    last_alert_text = text
    last_alert_ts = now
    print(text)
    speak_async(text)

def overlay_alerts(frame, detections=[]):
    """Enhanced overlay system with Ambient Glass UI and dedicated analytics."""
    global current_fps, fps_counter, last_fps_time, connection_status, distance_history, event_log
    
    # Update FPS
    fps_counter += 1
    if time.time() - last_fps_time >= 1.0:
        current_fps = fps_counter
        fps_counter = 0
        last_fps_time = time.time()
    
    # === AMBIENT GLASS BACKGROUND ===
    # Optimization: Compute blur on low-res image to save massive memory/CPU
    # 1. Downscale to 10% size
    small_w, small_h = max(1, TOTAL_WIDTH // 10), max(1, TOTAL_HEIGHT // 10)
    small_bg = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Apply blur on small image (fast)
    small_bg = cv2.blur(small_bg, (5, 5)) 
    
    # 3. Upscale back to full size
    ambient_bg = cv2.resize(small_bg, (TOTAL_WIDTH, TOTAL_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # 4. Darken (Optimized: Scale values down instead of allocating a huge zero array)
    display = ambient_bg
    display = cv2.convertScaleAbs(display, alpha=0.4, beta=0)
    
    # === FOREGROUND VIDEO ===
    # Ensure camera feed is crisp on the left
    display[0:FH, 0:FW] = frame
    
    # === SIDE PANEL (Right side) ===
    panel_x = FW
    
    # Draw Glass Sidebar (Semi-transparent overlay)
    # Since 'display' already has the blurred BG on the right, we just add the tint
    sidebar_overlay = display.copy()
    cv2.rectangle(sidebar_overlay, (panel_x, 0), (TOTAL_WIDTH, TOTAL_HEIGHT), (20, 25, 30), -1) # Dark tint
    cv2.addWeighted(sidebar_overlay, 0.7, display, 0.3, 0, display)
    
    # Sidebar separator line (Cyber Cyan)
    cv2.line(display, (panel_x, 0), (panel_x, TOTAL_HEIGHT), CYBER_CYAN, max(1, int(2 * UI_SCALE)))
    
    y_pos = int(30 * UI_SCALE)
    padding = int(20 * UI_SCALE)
    
    # === HEADER SECTION ===
    draw_outlined_text(display, "SURGICAL AI", (panel_x + padding, y_pos), scale=0.9, thickness=2, color=CYBER_CYAN)
    y_pos += int(35 * UI_SCALE)
    draw_outlined_text(display, "COMMAND CENTER", (panel_x + padding, y_pos), scale=0.7, thickness=2, color=(200, 200, 200))
    y_pos += int(40 * UI_SCALE)
    
    # Current time & FPS
    curr_t = datetime.now().strftime("%H:%M:%S")
    draw_outlined_text(display, f"TIME: {curr_t} | FPS: {current_fps}", (panel_x + padding, y_pos), scale=0.5, thickness=1, color=(150, 150, 150))
    y_pos += int(40 * UI_SCALE)
    
    # === STATUS SECTION ===
    cv2.line(display, (panel_x + 10, y_pos - 10), (panel_x + SIDE_PANEL_WIDTH - 10, y_pos - 10), (60, 60, 80), max(1, int(2 * UI_SCALE)))
    y_pos += int(10 * UI_SCALE)
    
    draw_outlined_text(display, "SYSTEM STATUS", (panel_x + padding, y_pos), scale=0.7, thickness=2, color=CYBER_CYAN)
    y_pos += int(35 * UI_SCALE)
    
    ai_status = "active" if ai_mode else "inactive"
    draw_status_indicator(display, (panel_x + padding, y_pos - int(5 * UI_SCALE)), ai_status, f"AI Mode: {'ON' if ai_mode else 'OFF'}")
    y_pos += int(25 * UI_SCALE)
    
    sys_status = "active" if system_active else "danger"
    draw_status_indicator(display, (panel_x + padding, y_pos - int(5 * UI_SCALE)), sys_status, f"Safety: {'ACTIVE' if system_active else 'HALTED'}")
    y_pos += int(25 * UI_SCALE)
    
    conn_status = "active" if time.time() - last_inference_success < 2.0 else "inactive"
    draw_status_indicator(display, (panel_x + padding, y_pos - int(5 * UI_SCALE)), conn_status, f"Inference: {connection_status}")
    y_pos += int(40 * UI_SCALE)
    
    # === PROXIMITY RADAR ===
    radar_y_center = y_pos + int(80 * UI_SCALE)
    draw_proximity_radar(display, (panel_x + SIDE_PANEL_WIDTH//2, radar_y_center), detections, radius=70)
    y_pos += int(170 * UI_SCALE)
    
    # === DISTANCE ANALYTICS ===
    draw_outlined_text(display, "DISTANCE TREND (mm)", (panel_x + padding, y_pos), scale=0.5, thickness=1, color=(150, 150, 150))
    y_pos += int(15 * UI_SCALE)
    draw_distance_graph(display, (panel_x + int(20 * UI_SCALE), y_pos), distance_history, width=int(360 * UI_SCALE), height=int(60 * UI_SCALE))
    y_pos += int(85 * UI_SCALE)
    
    # === EVENT LOG ===
    draw_outlined_text(display, "SYSTEM LOG", (panel_x + padding, y_pos), scale=0.7, thickness=2, color=CYBER_CYAN)
    y_pos += int(30 * UI_SCALE)
    
    for ts, text in event_log:
        draw_outlined_text(display, f"[{ts}] {text}", (panel_x + int(20 * UI_SCALE), y_pos), scale=0.45, thickness=1, color=(180, 180, 180))
        y_pos += int(20 * UI_SCALE)
        
    # === BOTTOM SAFETY INDICATOR ===
    y_pos = TOTAL_HEIGHT - int(130 * UI_SCALE)
    safety_text = "SURGEON: SAFE" if system_active else "WARNING: SURGEON"
    safety_color = (0, 255, 0) if system_active else (0, 0, 255)
    
    # Glass panel for safety indicator
    draw_glass_panel(display, (panel_x + 10, y_pos), (panel_x + SIDE_PANEL_WIDTH - 10, y_pos + int(50 * UI_SCALE)), 
                     color=(0, 40, 0) if system_active else (50, 0, 0), alpha=0.5, border_color=safety_color)
    
    (tw, th), _ = cv2.getTextSize(safety_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * UI_SCALE, max(1, int(2 * UI_SCALE)))
    draw_outlined_text(display, safety_text, (panel_x + (SIDE_PANEL_WIDTH - tw)//2, y_pos + int(35 * UI_SCALE)), scale=0.7, thickness=2, color=safety_color)
    
    # === DANGER OVERLAY (On Feed) ===
    if is_danger:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (FW, FH), (0, 0, 255), -1)
        pulse = 0.1 + 0.15 * abs(np.sin(time.time() * 5))
        cv2.addWeighted(overlay[0:FH, 0:FW], pulse, display[0:FH, 0:FW], 1 - pulse, 0, display[0:FH, 0:FW])
        draw_outlined_text(display, "CRITICAL PROXIMITY", (FW//2 - 250, FH//2), scale=1.5, thickness=4, outline_thickness=8, color=(255, 255, 255), outline_color=(0, 0, 150))
    
    return display


# ================================
# SPEECH RECOGNITION
# ================================
def speech_callback(recognizer, audio):
    global ai_mode
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Voice heard: {command}")
        if "stop" in command or "halt" in command:
            ai_mode = False
            stop_danger_sound()
            set_alert("VOICE STOP TRIGGERED")
    except sr.UnknownValueError:
        pass
    except Exception as e:
        print(f"Speech error: {e}")

# ================================
# SAFETY CHECK FUNCTIONS
# ================================
def bbox_overlap(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    return x1 < x2 and y1 < y2

def stop_tool():
    set_alert("TOOL MOVEMENT HALTED")

def alert_doctor(label):
    set_alert(f"ALERT: Contact with restricted structure ({label})")

def continue_motion():
    if system_active:
        pass 

def check_contact(tool_bbox, detected_objects):
    """
    Consolidated safety check with tiered alerts and Polygon-Precision.
    Returns: (is_safe, alert_message, severity)
    Severity: 0=Safe, 1=Caution (Halt), 2=Critical Danger
    """
    tool_center = ((tool_bbox[0] + tool_bbox[2]) / 2, (tool_bbox[1] + tool_bbox[3]) / 2)
    highest_severity = 0
    best_msg = ""
    
    for obj in detected_objects:
        obj_label = obj["class"]
        pts = obj.get("points", [])
        if not pts: continue
        
        # EXPERT: Calculate distance to the CLOSEST point in the polygon
        # This is much more accurate than center-to-center distance
        pts_arr = np.array([[p["x"], p["y"]] for p in pts])
        dists = np.sqrt(np.sum((pts_arr - tool_center)**2, axis=1))
        min_pt_dist_mm = np.min(dists) * PIXEL_TO_MM
        
        # Proximity Check (Only for Forbidden/Sensitive structures)
        if obj_label in FORBIDDEN_LABELS or obj_label in SENSITIVE_CLASSES:
            # 1. Critical Proximity (Visual Danger)
            if min_pt_dist_mm < CRITICAL_DISTANCE_MM:
                if "nerve" in obj_label.lower():
                    return False, "Warning: Surgeon Nerve Detected", 2
                return False, f"CRITICAL: Extremely close to {obj_label.upper()} ({min_pt_dist_mm:.1f}mm)", 2
            
            # 2. Caution/Halt (Robot Safety)
            if min_pt_dist_mm < MIN_DISTANCE_MM:
                highest_severity = max(highest_severity, 1)
                best_msg = f"RESTRICTED: Near {obj_label.upper()} ({min_pt_dist_mm:.1f}mm)"

        # 3. Physical Collision (Box overlap as fallback)
        if bbox_overlap(tool_bbox, obj["bbox"]):
            if obj_label in FORBIDDEN_LABELS or obj_label in SENSITIVE_CLASSES:
                return False, f"FATAL ERROR: Contact with {obj_label.upper()}!", 2

    if highest_severity == 1:
        return False, best_msg, 1
        
    return True, "", 0

def init_speech_recognition():
    print("Initializing speech recognition...")
    try:
        rec = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            print("Adjusting for ambient noise...")
            rec.adjust_for_ambient_noise(source, duration=1.0)
        stop_listening = rec.listen_in_background(mic, speech_callback)
        print("Speech recognition active.")
        return stop_listening
    except Exception as e:
        print(f"Speech init failed: {e}")
        return None

# ================================
# MAIN LOOP
# ================================
print("Controls: 'a' toggle AI, 'q' quit, 's' surgeon stop")

def main():
    print("Starting main loop")
    
    # Initialize window with NORMAL flag to allow full-screen toggling
    cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(DISPLAY_WINDOW, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    
    global running, ai_mode, out, system_active, last_alert_text, last_alert_ts, last_distance_text, last_tracked_tumor_text, is_danger, last_is_danger, initial_sep_dist, initial_tumor_size, locked_tumour_dist, lock, position_history, connection_status, last_inference_success, last_command_time
    stop_listening = init_speech_recognition()

    # Start background inference thread
    inf_thread = threading.Thread(target=inference_worker, daemon=True)
    inf_thread.start()

    try:
        while running:
            ret, _raw_frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Faster upscaling for 1080p
            frame = cv2.resize(_raw_frame, (FW, FH), interpolation=cv2.INTER_LINEAR)

            # Enhanced tool crosshair
            c_radius = int(60 * UI_SCALE)
            l_len = int(70 * UI_SCALE)
            tool_center = (FW//2, FH//2)
            tool_bbox = (FW//2 - c_radius, FH//2 - c_radius, FW//2 + c_radius, FH//2 + c_radius)
            
            # Tool crosshair removed as requested for a cleaner view

            detected_objects_for_safety = [] # Initialize empty for loop start
            if ai_mode:
                ensure_ai_on() # Ensure robot is in AI mode
                
                # Ultra-Latency: Push DOWNSCALED frame for much faster cloud transmission
                if inference_queue.empty():
                    try:
                        # Resize to 640x640 for the AI
                        inf_frame = cv2.resize(_raw_frame, (INF_W, INF_H), interpolation=cv2.INTER_LINEAR)
                        inference_queue.put_nowait(inf_frame)
                    except queue.Full:
                        pass
                
                # Use LATEST predictions from background thread (Non-blocking)
                # We copy the list and dicts to avoid modifying the background buffer by reference
                with inference_lock:
                    preds = [d.copy() for d in latest_predictions]

                # Coordinate scaling: from 640 (AI) back to FW (1080p UI)
                scale_x = FW / INF_W
                scale_y = FH / INF_H
                
                tool_center = (FW/2, FH/2)
                min_forbidden_dist_mm = float("inf")
                
                detected_objects_for_safety = []
                tumors_for_tracking = []

                # 1. Collect and Process all detections
                if preds:
                    for det in preds:
                        obj_label = det.get("class", "unknown")
                        # Rescale coordinates to match 1080p display
                        raw_pts = det.get("points", [])
                        pts = []
                        for p in raw_pts:
                            pts.append({"x": p["x"] * scale_x, "y": p["y"] * scale_y})
                        
                        if not pts: continue

                        xs = [p["x"] for p in pts]; ys = [p["y"] for p in pts]
                        bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                        center = (sum(xs)/len(xs), sum(ys)/len(ys))
                        
                        # Update original det object with rescaled points for drawing/tracking
                        det["points"] = pts
                        
                        obj_data = {"class": obj_label, "bbox": bbox, "center": center, "points": pts}
                        detected_objects_for_safety.append(obj_data)

                        # Update metrics
                        if obj_label in FORBIDDEN_LABELS:
                            dist_mm = (((tool_center[0] - center[0])**2 + (tool_center[1] - center[1])**2)**0.5) * PIXEL_TO_MM
                            min_forbidden_dist_mm = min(min_forbidden_dist_mm, dist_mm)
                        
                        if obj_label == TRACK_CLASS:
                            tumors_for_tracking.append(det)
                            draw_detection(frame, det, color=(0, 255, 0), tool_center=tool_center)
                        else:
                            draw_detection(frame, det, color=(0, 165, 255), tool_center=tool_center)
                else:
                    print("No objects detected by the model.")

                # 2. Perform Consolidated Safety Check
                safe_to_move, safety_msg, severity = check_contact(tool_bbox, detected_objects_for_safety)
                
                # Expert: Temporal Smoothing of Severity
                severity_history.append(severity)
                if len(severity_history) > MAX_SEVERITY_HISTORY:
                    severity_history.pop(0)
                
                # Determine "Smoothed" Severity (Consensus)
                # If majority of frames show severity >= 1, we halt.
                # If majority of frames show severity == 2, we show visual danger.
                smoothed_severity = max(set(severity_history), key=severity_history.count) if severity_history else 0

                # 3. Handle Safety state
                if smoothed_severity > 0:
                    if system_active: # Just entered caution/danger
                        # Use the real-time msg for logging the direct breach
                        log_msg = safety_msg if safety_msg else "RESTRICTED"
                        add_event(f"⚠️ {log_msg}")
                        set_alert(log_msg)
                        if ROBOT_ON: sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
                        system_active = False
                    
                    if smoothed_severity == 2: # Critical Danger UI
                        is_danger = True
                        play_danger_sound(loop=True)
                    else: # Caution only (Stable)
                        is_danger = False
                        stop_danger_sound()
                else:
                    if not system_active: # Recovery consensus
                        add_event("✅ Safety Restored")
                        system_active = True
                    is_danger = False
                    stop_danger_sound()

                # Update UI Metric String
                last_distance_text = f"Closest forbidden: {min_forbidden_dist_mm:.1f} mm" if min_forbidden_dist_mm < float('inf') else ""

                # Update Distance History
                val_to_plot = min(min_forbidden_dist_mm, 150.0) if min_forbidden_dist_mm < float('inf') else 150.0
                distance_history.append(val_to_plot)
                if len(distance_history) > MAX_DIST_HISTORY:
                    distance_history.pop(0)

                # Tumor tracking logic
                if safe_to_move and tumors_for_tracking:
                    # Initialize initial_tumor_size if not set
                    if initial_tumor_size is None:
                        det = tumors_for_tracking[0]
                        pts = det["points"]
                        xs = [p["x"] for p in pts]
                        ys = [p["y"] for p in pts]
                        w_px = max(xs) - min(xs)
                        h_px = max(ys) - min(ys)
                        initial_tumor_size = (w_px * PIXEL_TO_MM, h_px * PIXEL_TO_MM)

                    # Find closest tumor for tracking and display
                    min_dist_tumor = float('inf')
                    tracked_tumor_center = None
                    for det in tumors_for_tracking:
                        pts = det.get("points", [])
                        if not pts: continue
                        cx_det = sum(p["x"] for p in pts) / len(pts)
                        cy_det = sum(p["y"] for p in pts) / len(pts)
                        dist = ((tool_center[0] - cx_det)**2 + (tool_center[1] - cy_det)**2)**0.5
                        if dist < min_dist_tumor:
                            min_dist_tumor = dist
                            tracked_tumor_center = (cx_det, cy_det)

                    tumour_dist_mm = min_dist_tumor * PIXEL_TO_MM
                    last_tracked_tumor_text = f"Tracked {TRACK_CLASS}: {tumour_dist_mm:.1f} mm"

                    # Anomaly detection and robot control (simplified for clarity)
                    if tracked_tumor_center:
                        cx, cy = tracked_tumor_center
                        
                        # EXPERT: Smoothing for "Happy Movement" (Fluid motion)
                        tracking_history.append((cx, cy))
                        if len(tracking_history) > TRACKING_SMOOTH_HISTORY:
                            tracking_history.pop(0)
                        
                        avg_cx = sum(p[0] for p in tracking_history) / len(tracking_history)
                        avg_cy = sum(p[1] for p in tracking_history) / len(tracking_history)
                        
                        # Apply offsets to smoothed position
                        cx_adj = avg_cx + HORIZ_OFFSET_PX
                        cy_adj = avg_cy + VERT_OFFSET_PX
                        # Normalized displacement
                        dx = (cx_adj - FW / 2) / (FW / 2)
                        dy = (cy_adj - FH / 2) / (FH / 2)

                        # Tolerance
                        if abs(dx) <= TOLERANCE_PER: dx = 0.0
                        if abs(dy) <= TOLERANCE_PER: dy = 0.0

                        # Decide command
                        if dx == 0.0 and dy == 0.0:
                            if not lock:
                                add_event(f"🎯 TARGET LOCKED: {TRACK_CLASS}")
                                lock = True
                            cmd = HALT_CMD
                        else:
                            if lock:
                                add_event("Searching for Target...")
                            lock = False
                            cmd = f"move {dx} {dy}"
                            print(f"dx: {dx*100:.2f}%, dy: {dy*100:.2f}%")
                        
                        if ROBOT_ON:
                            # Command Throttling for smoothness
                            if time.time() - last_command_time > COMMAND_THROTTLE_SEC:
                                sock.sendto(cmd.encode(), ROBOT_ADDR)
                                last_command_time = time.time()

                        # One-shot behavior: disengage on lock
                        if (not AUTO_TRACK) and (dx == 0.0 and dy == 0.0):
                            ensure_ai_off()
                            ai_mode = False
                            set_alert("AI mode: OFF (one-shot complete)")

                elif safe_to_move and not tumors_for_tracking:
                    # No tumor detections, but safe to move otherwise
                    if ROBOT_ON:
                        sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
                    if not AUTO_TRACK:
                        ensure_ai_off()
                        ai_mode = False
                        add_event("AI: Target Lost (Mode OFF)")
                    else:
                        add_event("AI: Target Lost (Searching...)")
                    last_tracked_tumor_text = ""
                    lock = False
                # If not safe_to_move, the check_contact function already handled halting

            display = overlay_alerts(frame, detections=detected_objects_for_safety)
            
            # FORCE FILL FIX: Disabled for memory safety diagnostic
            # Manual resizing loop suspected of causing OOM on macOS
            # try:
            #     rect = cv2.getWindowImageRect(DISPLAY_WINDOW)
            #     ...
            # except Exception:
            #     pass

            # Display the frame
            cv2.imshow(DISPLAY_WINDOW, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('a'):
                ai_mode = not ai_mode
                if ai_mode:
                    add_event("Mode: AI Engagement ON")
                    speak_async("AI Mode Detected")
                else:
                    add_event("Mode: Manual Control")
                    speak_async("Manual Control")
                    stop_danger_sound()
                    ensure_ai_off()
            elif key == ord('s'):
                add_event("SURGEON STOP TRIGGERED")
                system_active = False
                if ROBOT_ON:
                    sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
                ai_mode = False
            elif key == ord('f'):
                # NUCLEAR OPTION: Destroy and Re-create to clear stuck aspect ratio
                cv2.destroyWindow(DISPLAY_WINDOW)
                cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
                # Removed FREERATIO per user request
                cv2.setWindowProperty(DISPLAY_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                add_event("Window: FULL SCREEN (Reset)")
            elif key == ord('p'):
                cv2.destroyWindow(DISPLAY_WINDOW)
                cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)  
                cv2.resizeWindow(DISPLAY_WINDOW, FW, FH)
                add_event("Window: NORMAL SIZE (Reset)")

    finally:
        print("Exiting main loop")
        if stop_listening:
            stop_listening(wait_for_stop=False)
            print("Speech recognition stopped.")
        stop_danger_sound()
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    print("Script started")
    main()
    print("Script finished")
