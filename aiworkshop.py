#!/usr/bin/env python3
"""
Unified AI Tracking + Guess-and-Stop Safety + Emergency Stop + Surgeon Override Capture
with On-Screen Alerts

- Tracks 'vertebral-body-tumour' via Roboflow detections.
- Safety layer checks confidence and geometry; warns/halts near sensitive structures.
- Emergency stop: if tool bbox overlaps forbidden/unknown structures, warn + halt (surgeon can override anytime).
- Surgeon override: press 's' → logs reason + saves current frame.
- Overlays alerts directly on the video feed for clear surgeon feedback.
"""

import cv2
import tempfile
import os
import numpy as np
import socket
import time
import sys
from datetime import datetime
from inference_sdk import InferenceConfiguration, InferenceHTTPClient
import speech_recognition as sr
import pyttsx3

# Initialize TTS engine
try:
    tts_engine = pyttsx3.init()
    # Test TTS
    tts_engine.say("TTS initialized successfully")
    tts_engine.runAndWait()
    print("TTS test successful")
except Exception as e:
    print(f"TTS init/test error: {e}")
    tts_engine = None

# ================================
# CONFIGURABLE CONSTANTS
# ================================

# Labels (match your Roboflow dataset names exactly)
TRACK_CLASS        = "vertebral-body-tumor"
ALLOWED_LABELS     = {"vertebral-body-tumor"}
FORBIDDEN_LABELS   = {"nerve", "bone"}           # add/remove as per your labels
SENSITIVE_CLASSES  = {"nerve", "SpinalCord"}     # used for proximity warnings

# AI tracking and detection
AUTO_TRACK         = True
CONFIDENCE_THRESH  = 0.55
IOU_THRESH         = 0.65
TOLERANCE_PER      = 0.10                        # normalized tolerance to lock
MIN_DISTANCE_MM    = 20                          # minimum safe distance in mm from forbidden structures
PIXEL_TO_MM        = 1.0                         # conversion factor: pixels to mm (adjust based on calibration)
HORIZ_OFFSET_PX    = 50
VERT_OFFSET_PX     = -30

# Robot control
ROBOT_HOST         = "moonshot5.local"
ROBOT_ON           = False
RETRACT_CMD        = "move -1 -1"                # symbolic retract
HALT_CMD           = "move 0 0"                  # stop motors
ROBOT_AI_SPEED     = 12
AI_SPEED_INC       = 4

# Roboflow inference
RF_MODEL_ID        = "my-first-project-s3pmu/5"
RF_API_KEY         = "UnCpj4TyfcueHzcoVikk"
RF_API_URL         = "http://localhost:9001"

# Camera and UI
CAMERA_PORT        = 0
DISPLAY_WINDOW     = "Frame"
RECORD_VIDEO       = False
LOOP_SLEEP_SEC     = 0.05

# Storage
OVERRIDE_DIR       = "override_frames"

# ================================
# SETUP
# ================================
config = InferenceConfiguration(CONFIDENCE_THRESH, IOU_THRESH)
client = InferenceHTTPClient(api_url=RF_API_URL, api_key=RF_API_KEY)
client.configure(config)
client.select_model(RF_MODEL_ID)

# Platform-specific camera backend
if sys.platform.startswith("win"):
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_MSMF)
elif sys.platform.startswith("darwin"):
    cap = cv2.VideoCapture(CAMERA_PORT, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAMERA_PORT)

if not cap.isOpened():
    print(f"Error: Failed to open camera at port {CAMERA_PORT}. Please check camera connection and try again.")
    sys.exit(1)

FW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
FH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

if RECORD_VIDEO:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = cv2.VideoWriter(
        f"record_{ts}.mp4",
        cv2.VideoWriter_fourcc(*"XVID"),
        7.0,
        (FW, FH)
    )
else:
    out = None

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ROBOT_ADDR = (ROBOT_HOST, 5005)

def prime_robot_speeds():
    if ROBOT_ON:
        sock.sendto(f"ai_speed {ROBOT_AI_SPEED}".encode(), ROBOT_ADDR)
        sock.sendto(f"ai_speed_inc {AI_SPEED_INC/100}".encode(), ROBOT_ADDR)

prime_robot_speeds()
cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
os.makedirs(OVERRIDE_DIR, exist_ok=True)

# ================================
# STATE
# ================================
ai_mode = False
running = True
lock = False
last_ai_mode_sent = False
system_active = True            # for emergency-stop messages
last_alert_text = ""            # text to overlay
last_alert_ts = 0.0             # when alert was set (for fade-out)
last_distance_text = ""         # distance to closest forbidden
is_danger = False               # flag for danger overlay
last_is_danger = False          # to track danger state changes
last_tumour_danger = False      # to track tumour proximity danger
position_history = []           # for anomaly detection
last_proximity_alert_time = 0   # to throttle proximity alerts
initial_sep_dist = None         # to keep constant
initial_tumor_size = None       # to keep constant
locked_tumour_dist = None       # lock distance when red

def play_danger_sound():
    try:
        os.system('afplay /System/Library/Sounds/Basso.aiff &')
    except Exception as e:
        print(f"Sound play error: {e}")

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

def draw_detection(frame, det, color=(0, 255, 0)):
    pts = det.get("points", [])
    if not pts:
        return
    contour = np.array([[int(p["x"]), int(p["y"])] for p in pts], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [contour], True, color, 2)
    lx, ly = int(pts[0]["x"]), int(pts[0]["y"]) - 10
    label = f"{det.get('class','Unknown')} {det.get('confidence',0):.2f}"
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # Add size measurement
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    w_px = max(xs) - min(xs)
    h_px = max(ys) - min(ys)
    w_mm = w_px * PIXEL_TO_MM
    h_mm = h_px * PIXEL_TO_MM
    if det.get("class") == TRACK_CLASS and initial_tumor_size:
        w_mm, h_mm = initial_tumor_size
    size_text = f"{w_mm:.1f}x{h_mm:.1f} mm"
    cv2.putText(frame, size_text, (lx, ly + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def set_alert(text, color=(0, 0, 255)):
    # store text for overlay and timestamp
    global last_alert_text, last_alert_ts
    last_alert_text = text
    last_alert_ts = time.time()
    print(text)  # also print to console for logging
    # Voice alert
    if tts_engine:
        try:
            print(f"Speaking: {text}")
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS speak error: {e}")
    else:
        print("TTS not available")

def overlay_alerts(frame):
    # show alert for ~2 seconds from last set, fade by time
    if last_alert_text:
        elapsed = time.time() - last_alert_ts
        if elapsed <= 2.5:
            # red bar at top
            cv2.rectangle(frame, (0, 0), (FW, 32), (0, 0, 255), thickness=-1)
            cv2.putText(frame, last_alert_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Show distance at bottom if available
    if last_distance_text:
        # Extract mm value for coloring
        try:
            mm_value = float(last_distance_text.split()[-2])
            color = (0, 0, 255) if mm_value < MIN_DISTANCE_MM else (0, 255, 0)
        except:
            color = (255, 255, 255)
        cv2.putText(frame, last_distance_text, (10, FH - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Danger overlay if too close to forbidden
    if is_danger:
        # Semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FW, FH), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        # Big DANGER text
        cv2.putText(frame, "DANGER", (FW//2 - 150, FH//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 8)

# ================================
# SAFETY + EMERGENCY STOP
# ================================
def is_confident(det):
    return float(det.get("confidence", 0.0)) >= CONFIDENCE_THRESH

def has_valid_geometry(det):
    pts = det.get("points", [])
    if len(pts) < 4:
        return False
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return (w > 3) and (h > 3)

def predict_tumor_response(det):
    # placeholder rules; extend with learned model if available
    return is_confident(det) and has_valid_geometry(det)

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
        # short info message; overlay less prominently
        set_alert("Safe interaction — movement allowed")

def check_contact(tool_bbox, detected_objects):
    # Returns True if motion allowed; False if halting
    global system_active
    tool_center = ((tool_bbox[0] + tool_bbox[2]) / 2, (tool_bbox[1] + tool_bbox[3]) / 2)
    for obj in detected_objects:
        if bbox_overlap(tool_bbox, obj["bbox"]):
            if obj["label"] in FORBIDDEN_LABELS:
                system_active = False
                stop_tool()
                alert_doctor(obj["label"])
                return False
            if obj["label"] in ALLOWED_LABELS:
                continue_motion()
                return True
            # Unknown → unsafe
            system_active = False
            stop_tool()
            alert_doctor("unknown structure")
            return False
        # Proximity check for forbidden structures
        if obj["label"] in FORBIDDEN_LABELS:
            pts = obj.get("points", [])
            if pts:
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                obj_center = (sum(xs)/len(xs), sum(ys)/len(ys))
                dist = ((tool_center[0] - obj_center[0])**2 + (tool_center[1] - obj_center[1])**2)**0.5
                dist_mm = dist * PIXEL_TO_MM
                if dist_mm < MIN_DISTANCE_MM:
                    system_active = False
                    stop_tool()
                    alert_doctor(f"too close to {obj['label']}")
                    return False
        # Check for overlap with sensitive structures (touching)
        if obj["label"] in SENSITIVE_CLASSES and bbox_overlap(tool_bbox, obj["bbox"]):
            set_alert(f"Touching sensitive organ: {obj['label']}")
            play_danger_sound()
            system_active = False
            stop_tool()
            return False
    continue_motion()
    return True

# ================================
# LOGGING
# ================================
def log_surgeon_override(reason, frame=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # record text log
    with open("surgeon_feedback.log", "a", encoding="utf-8") as f:
        f.write(f"{ts} | {reason}\n")
    set_alert(f"Surgeon override logged: {reason}")
    # save frame evidence
    if frame is not None:
        fname = os.path.join(OVERRIDE_DIR, f"override_{ts}_{reason.replace(' ', '_')}.jpg")
        cv2.imwrite(fname, frame)
        print(f"Override frame saved: {fname}")

# ================================
# MAIN LOOP
# ================================
print("Controls: 'a' toggle AI, 'q' quit, 's' surgeon stop+reason")

try:
    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # draw tool bbox (visual cue) centered in the frame
        tool_bbox = (FW // 2 - 50, FH // 2 - 50, FW // 2 + 50, FH // 2 + 50)
        cv2.rectangle(frame, (tool_bbox[0], tool_bbox[1]), (tool_bbox[2], tool_bbox[3]), (255, 0, 255), 1)

        if ai_mode:
            # Inference via temp file
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, frame)
                    tmp_path = tmp.name
                resp = client.infer(tmp_path)
                preds = resp.get("predictions", [])
            except Exception as e:
                preds = []
                if ROBOT_ON:
                    sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
                set_alert(f"Inference error: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            # Draw detections
            for det in preds:
                draw_detection(frame, det)

            # Build list of detected objects (bbox) for emergency stop
            detected_objects = []
            for det in preds:
                pts = det.get("points", [])
                if not pts:
                    continue
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                center = (sum(xs)/len(xs), sum(ys)/len(ys))
                detected_objects.append({"label": det.get("class", "unknown"), "bbox": bbox, "center": center})

            # Calculate distance to closest object and check for danger
            tool_center = (FW / 2, FH / 2)
            min_dist = float('inf')
            min_dist_forbidden = float('inf')
            for obj in detected_objects:
                if obj["center"]:
                    dist = ((tool_center[0] - obj["center"][0])**2 + (tool_center[1] - obj["center"][1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                    if obj["label"] in FORBIDDEN_LABELS and dist < min_dist_forbidden:
                        min_dist_forbidden = dist
            min_dist_mm = min_dist * PIXEL_TO_MM if min_dist < float('inf') else float('inf')
            min_dist_forbidden_mm = min_dist_forbidden * PIXEL_TO_MM if min_dist_forbidden < float('inf') else float('inf')
            is_danger = min_dist_forbidden_mm < MIN_DISTANCE_MM if min_dist_forbidden < float('inf') else False
            # Alert on danger state change
            if is_danger and not last_is_danger:
                set_alert("Danger: Too close to forbidden structure")
                play_danger_sound()
            last_is_danger = is_danger
            # Proximity alert for any object
            if min_dist_mm < 30 and time.time() - last_proximity_alert_time > 5:
                set_alert("Warning: Object too close")
                last_proximity_alert_time = time.time()
            # Distance text will be set in tumor tracking
            print(f"Detected labels: {[obj['label'] for obj in detected_objects]}, Min dist: {min_dist:.1f} px ({min_dist_mm:.1f} mm), Danger: {is_danger}")

            # Emergency stop check (warn + halt if needed)
            safe_to_move = check_contact(tool_bbox, detected_objects)

            # Tumor tracking only if safe and tumor present
            tumors = [d for d in preds if d.get("class") == TRACK_CLASS]
            if tumors and initial_tumor_size is None:
                det = tumors[0]
                pts = det["points"]
                xs = [p["x"] for p in pts]
                ys = [p["y"] for p in pts]
                w_px = max(xs) - min(xs)
                h_px = max(ys) - min(ys)
                initial_tumor_size = (w_px * PIXEL_TO_MM, h_px * PIXEL_TO_MM)
            if safe_to_move and tumors:
                # Find closest tumor for display
                tool_center = (FW / 2, FH / 2)
                min_dist_tumor = float('inf')
                for det in tumors:
                    pts = det["points"]
                    xs = [p["x"] for p in pts]
                    ys = [p["y"] for p in pts]
                    cx_det = sum(xs)/len(xs)
                    cy_det = sum(ys)/len(ys)
                    dist = ((tool_center[0] - cx_det)**2 + (tool_center[1] - cy_det)**2)**0.5
                    if dist < min_dist_tumor:
                        min_dist_tumor = dist
                # Set initial distances
                tumour_dist_mm = min_dist_tumor * PIXEL_TO_MM
                display_dist = locked_tumour_dist if locked_tumour_dist is not None else tumour_dist_mm
                last_distance_text = f"Screen distance to closest tumour: {display_dist:.1f} mm"
                # Alert if too close to tumour
                if tumour_dist_mm < MIN_DISTANCE_MM and not last_tumour_danger:
                    set_alert("Warning: Too close to tumour")
                    play_danger_sound()
                last_tumour_danger = tumour_dist_mm < MIN_DISTANCE_MM
                if last_tumour_danger and locked_tumour_dist is None:
                    locked_tumour_dist = tumour_dist_mm
                # Tumor separation if exactly two
                if len(tumors) == 2:
                    det1 = tumors[0]
                    pts1 = det1["points"]
                    xs1 = [p["x"] for p in pts1]
                    ys1 = [p["y"] for p in pts1]
                    cx1 = sum(xs1)/len(xs1)
                    cy1 = sum(ys1)/len(ys1)
                    det2 = tumors[1]
                    pts2 = det2["points"]
                    xs2 = [p["x"] for p in pts2]
                    ys2 = [p["y"] for p in pts2]
                    cx2 = sum(xs2)/len(xs2)
                    cy2 = sum(ys2)/len(ys2)
                    sep_dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5 * PIXEL_TO_MM
                    if initial_sep_dist is None:
                        initial_sep_dist = sep_dist
                    last_distance_text += f", Screen distance between two tumours: {initial_sep_dist:.1f} mm"
                # Compute average center for tracking
                all_pts = []
                for det in tumors:
                    all_pts.extend(det["points"])
                xs = [p["x"] for p in all_pts]
                ys = [p["y"] for p in all_pts]
                cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
                # Anomaly detection: check for sudden movement
                if position_history:
                    prev_cx, prev_cy = position_history[-1]
                    dist_moved_px = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                    dist_moved_mm = dist_moved_px * PIXEL_TO_MM
                    if dist_moved_mm > 20:  # threshold for sudden movement
                        set_alert("Sudden movement detected")
                position_history.append((cx, cy))
                if len(position_history) > 5:
                    position_history.pop(0)
                # apply offsets
                cx_adj = cx + HORIZ_OFFSET_PX
                cy_adj = cy + VERT_OFFSET_PX
                # normalized displacement
                dx = (cx_adj - FW / 2) / (FW / 2)
                dy = (cy_adj - FH / 2) / (FH / 2)
                # tolerance
                if abs(dx) <= TOLERANCE_PER:
                    dx = 0.0
                if abs(dy) <= TOLERANCE_PER:
                    dy = 0.0
                # decide command
                if dx == 0.0 and dy == 0.0:
                    if not lock:
                        set_alert("Target Locked")
                        lock = True
                    cmd = HALT_CMD
                else:
                    lock = False
                    cmd = f"move {dx} {dy}"
                    print(f"dx: {dx*100:.2f}%, dy: {dy*100:.2f}%")
                # send command
                if ROBOT_ON:
                    ensure_ai_on()
                    sock.sendto(cmd.encode(), ROBOT_ADDR)
                # one-shot behavior: disengage on lock
                if (not AUTO_TRACK) and (dx == 0.0 and dy == 0.0):
                    ensure_ai_off()
                    ai_mode = False
                    set_alert("AI mode: OFF (one-shot complete)")

            elif safe_to_move and not tumors:
                # no tumor detections
                if ROBOT_ON:
                    sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)
                if not AUTO_TRACK:
                    ensure_ai_off()
                    ai_mode = False
                    set_alert("AI mode: OFF (one-shot lost target)")
                else:
                    set_alert("No detections (robot stopped, AI searching)")
                last_distance_text = ""
            else:
                # unsafe (emergency stop triggered)
                if ROBOT_ON:
                    sock.sendto(HALT_CMD.encode(), ROBOT_ADDR)

            time.sleep(LOOP_SLEEP_SEC)

        # overlay alerts bar if any
        overlay_alerts(frame)

        # record + show
        if RECORD_VIDEO and out:
            out.write(frame)
        cv2.imshow(DISPLAY_WINDOW, frame)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
            ensure_ai_off()
        elif key == ord('a'):
            ai_mode = not ai_mode
            set_alert("AI mode: ON" if ai_mode else "AI mode: OFF")
            if not ai_mode:
                ensure_ai_off()
            else:
                prime_robot_speeds()
                ensure_ai_on()
        elif key == ord('s'):
            # surgeon override + frame capture
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    print("Speak your reason for stopping (you have 10 seconds):")
                    audio = r.listen(source, timeout=10)
                try:
                    reason = r.recognize_google(audio)
                    print(f"Recognized: {reason}")
                except sr.UnknownValueError:
                    reason = "Speech not understood"
                    print("Could not understand audio")
                except sr.RequestError:
                    reason = "Speech recognition unavailable"
                    print("Speech recognition service error")
            except Exception as e:
                reason = "Microphone error"
                print(f"Microphone error: {e}")
            log_surgeon_override(reason, frame)
            ensure_ai_off()
            ai_mode = False

except KeyboardInterrupt:
    ensure_ai_off()
finally:
    # cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    try:
        sock.close()
    except Exception:
        pass


