from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pickle
import numpy as np
import datetime
import pyttsx3
import threading
import time

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_FILE = "encodings.pickle" 
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
ATTENDANCE_FILE = "attendance_log.csv"
COOLDOWN_SECONDS = 30

# Global State
camera = None
outputFrame = None
lock = threading.Lock()
app_mode = 'idle'
reg_name = ""
reg_emp_id = ""
reg_counter = 0
reg_faces = []
reg_status_msg = "idle"
today_log = [] # Fix: Initialize today_log
reg_faces_color = [] # New: Store color faces for DB
latest_recognition = {"name": "", "emp_id": "", "time": 0}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

import mysql.connector

# --- DATABASE CONFIG ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '', # Default XAMPP/WAMP password
    'database': 'face'
}

def speak_async(text):
    def task():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")
    threading.Thread(target=task).start()

def get_db_connection(db_name=None):
    conf = DB_CONFIG.copy()
    if db_name:
        conf['database'] = db_name
    else:
        conf.pop('database', None) # Connect to server only
        
    try:
        conn = mysql.connector.connect(**conf)
        return conn
    except mysql.connector.Error as err:
        print(f"Database Connect Error: {err}")
        return None

def init_db():
    # 1. Create DB if not exists
    conn = get_db_connection() # No DB specified
    if not conn:
        print("CRITICAL: COULD NOT CONNECT TO MYSQL SERVER.")
        return

    cursor = conn.cursor()
    try:
        cursor.execute("CREATE DATABASE IF NOT EXISTS face")
        print("Database 'face' checked/created.")
    except Exception as e:
        print(f"Error creating DB: {e}")
        
    conn.close()
    
    # 2. Create Tables
    conn = get_db_connection('face')
    if not conn:
        print("Error connecting to 'face' database.")
        return
        
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            emp_id VARCHAR(50) UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            name VARCHAR(100),
            emp_id VARCHAR(50),
            timestamp DATETIME,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Add profile_image column if not exists
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN profile_image LONGBLOB")
    except:
        pass

    conn.commit()
    conn.close()
    print("Tables initialized.")

init_db()

def load_resources():
    global user_db, recognizer
    user_db = {}
    
    if os.path.exists(MODEL_FILE):
        try:
             recognizer.read(MODEL_FILE)
             print("Model loaded.")
        except:
             pass
    
    # Load users from DB
    conn = get_db_connection('face')
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, emp_id FROM users")
        for row in cursor.fetchall():
            user_db[row['id']] = {
                "name": row['name'],
                "emp_id": row['emp_id']
            }
        conn.close()

def log_attendance(user_data):
    global today_log
    name = user_data['name']
    emp_id = user_data.get('emp_id', 'N/A')
    user_pk = -1
    
    # Find user_pk from user_db (reverse lookup or just passed in?)
    # user_data coming from process_video_stream is fetched from user_db, so we don't strictly have PK there unless we add it to user_db values.
    # Let's verify user_db structure from load_resources: user_db[id] = {...}
    # So we need to pass the ID from the detector loop.
    pass # logic moved to inside detection loop to get ID

def log_attendance_db(internal_id, user_data):
    global today_log
    name = user_data['name']
    emp_id = user_data.get('emp_id', 'N/A')
    
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check cooldown in memory
    for entry in today_log:
        if entry['id'] == emp_id and entry['name'] == name:
             last_time = datetime.datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
             if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
                 return

    # Add to memory
    entry = {"name": name, "id": emp_id, "time": time_str}
    today_log.append(entry)
    
    # Write to DB
    conn = get_db_connection('face')
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (user_id, name, emp_id, timestamp) VALUES (%s, %s, %s, %s)", 
                      (internal_id, name, emp_id, now))
        conn.commit()
        conn.close()
    
    threading.Thread(target=speak_async, args=(f"Thank you {name}",)).start()

def save_model():
    global user_db, reg_faces, reg_faces_color, reg_name, reg_emp_id, recognizer
    
    if not reg_faces:
        print("Error: No faces captured.")
        return False

    try:
        conn = get_db_connection('face')
        if not conn: return False
        
        cursor = conn.cursor()
        
        # 1. Insert into MySQL
        try:
            # Prepare image
            img_blob = None
            if reg_faces_color:
                # Use the middle frame for worst-case blur avoidance
                best_frame = reg_faces_color[len(reg_faces_color)//2] 
                success, encoded = cv2.imencode('.jpg', best_frame)
                if success:
                    img_blob = encoded.tobytes()

            cursor.execute("INSERT INTO users (name, emp_id, profile_image) VALUES (%s, %s, %s)", (reg_name, reg_emp_id, img_blob))
            conn.commit()
            new_internal_id = cursor.lastrowid
        except mysql.connector.Error as err:
            print(f"User exists or DB error: {err}")
            # If duplicate, fetch existing ID
            cursor.execute("SELECT id FROM users WHERE emp_id = %s", (reg_emp_id,))
            result = cursor.fetchone()
            if result:
                 new_internal_id = result[0]
            else:
                 return False # Should not happen
        
        conn.close()

        # 2. Update Local Cache
        user_db[new_internal_id] = {
            "name": reg_name,
            "emp_id": reg_emp_id
        }
            
        # 3. Update Recognizer
        print("Training LBPH model...")
        faces_np = [np.array(f, dtype=np.uint8) for f in reg_faces]
        ids_np = np.array([new_internal_id]*len(reg_faces))

        if os.path.exists(MODEL_FILE):
             # Force full retrain if possible or update? 
             # LBPH update works, but sometimes easier to just load all images. 
             # Since we don't store images, we MUST use update.
             recognizer.update(faces_np, ids_np)
        else:
            recognizer.train(faces_np, ids_np)
            
        recognizer.write(MODEL_FILE)
        print(f"Saved model for {reg_name} (ID: {new_internal_id})")
        return True
        
    except Exception as e:
        print(f"CRITICAL ERROR SAVING MODEL: {e}")
        return False

# --- CAMERA HANDLING ---
class VideoCamera(object):
    def __init__(self):
        # Try DirectShow first (faster on Windows)
        print("Initializing Camera...")
        self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not self.video.isOpened():
             print("DirectShow failed, trying default...")
             self.video = cv2.VideoCapture(0)
             
        # Reverted to default resolution to fix black screen issue on incompatible cameras
        # self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.video.isOpened():
            print("CRITICAL: Camera could not be opened!")
            raise Exception("Could not open video device")
             
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        if not self.video.isOpened(): return None
        success, frame = self.video.read()
        if not success: return None
        return frame

        if not success: return None
        return frame

def draw_corner_rect(img, bbox, color, thickness=3, length=30):
    """Draws only the corners of a bounding box (viewfinder style)."""
    x, y, w, h = bbox
    
    # Top Left
    cv2.line(img, (x, y), (x + length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + length), color, thickness)
    
    # Top Right
    cv2.line(img, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), color, thickness)
    
    # Bottom Left
    cv2.line(img, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), color, thickness)
    
    # Bottom Right
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), color, thickness)

def process_video_stream():
    global outputFrame, lock, camera, app_mode
    global reg_counter, reg_faces, reg_faces_color, reg_status_msg, latest_recognition
    
    # Initialize CLAHE for lighting correction (Bright/Dark handling)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    
    # Initialize with a black frame so the browser has something to show immediately
    with lock:
        outputFrame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(outputFrame, "Starting Camera...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Retry camera connection mechanism
    while camera is None:
        try:
            camera = VideoCamera()
            print("Camera connected successfully!")
        except Exception as e:
             time.sleep(1)
             print(f"Waiting for camera... {e}")

    while True:
        try:
             # ...
            frame = camera.get_frame()
            
            if frame is None:
                # Camera disconnected? Try to reconnect
                print("Frame is None, reconnecting...")
                try:
                    # Do not delete the global variable name!
                    # Just release if possible (though __del__ handles it) and reassign
                    camera = None 
                    time.sleep(1)
                    camera = VideoCamera()
                    continue
                except Exception as e:
                    time.sleep(1)
                    continue
            
                    time.sleep(1)
                    continue
            
            # --- REMOVED HDR/CLAHE FROM DISPLAY FRAME TO FIX BLUR ---
            # The processing was causing graininess. We keep frame natural.
            # -------------------------------------------

            # Always show mode for debugging
            cv2.putText(frame, f"Mode: {app_mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if app_mode == 'idle':
                 # Transparent Overlay (Dimmed)
                 overlay = frame.copy()
                 cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,0,0), -1)
                 cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                 
            elif app_mode == 'registration':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = clahe.apply(gray) # Apply lighting correction
                faces = face_cascade.detectMultiScale(gray, 1.1, 4) 
                
                # Show processing status
                if reg_status_msg == "training":
                     cv2.putText(frame, "TRAINING DO NOT CLOSE...", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif reg_status_msg == "complete":
                     cv2.putText(frame, "SAVED SUCCESSFULLY!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    # Normal Capture Mode
                    if len(faces) == 0:
                         cv2.putText(frame, "LOOK AT CAMERA", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                         
                    for (x,y,w,h) in faces:
                        # Draw Corner Rect instead of full rect
                        draw_corner_rect(frame, (x,y,w,h), (0, 255, 255), 3, 25)
                        
                        if reg_counter < 40: 
                            reg_faces.append(gray[y:y+h, x:x+w])
                            reg_faces_color.append(frame[y:y+h, x:x+w])
                            reg_counter += 1
                        
                        if reg_counter >= 40:
                            # Only start training ONE time
                            if reg_status_msg != "training" and reg_status_msg != "complete":
                                reg_status_msg = "training"
                                threading.Thread(target=finish_registration).start()
                    
                    cv2.putText(frame, f"Capturing: {reg_counter}/40", (20, 450), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                           
            elif app_mode == 'attendance':
                # Optimization: Resize frame for faster detection (50% scale)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                small_gray = clahe.apply(small_gray) # Apply lighting correction

                faces_small = face_cascade.detectMultiScale(small_gray, 1.1, 4)
                
                # Scale faces back to original size
                faces = []
                for (x, y, w, h) in faces_small:
                    faces.append((x*2, y*2, w*2, h*2))

                
                if not user_db:
                     cv2.putText(frame, "NO USERS REGISTERED", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if len(faces) == 0:
                     cv2.putText(frame, "Looking for faces...", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
                for (x,y,w,h) in faces:
                    name = "Unknown"
                    emp_id_disp = ""
                    color = (0, 0, 255)
    
                    if user_db:
                        try:
                            # Use full resolution gray frame with CLAHE for recognition
                            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            gray_full = clahe.apply(gray_full)

                            id_, conf = recognizer.predict(gray_full[y:y+h, x:x+w])
                            print(f"Detected: ID={id_} Conf={conf}")
                            
                            # LBPH Confidence: Lower is better. 
                            # 0 = Absolute match. 
                            # < 50 = Good match.
                            # < 100 = Loose match.
                            if conf < 55:  # Relaxed threshold (was 45) to reduce "Unknown" issues
                                user_data = user_db.get(id_)
                                if user_data:
                                    name = user_data['name']
                                    emp_id_disp = user_data.get('emp_id', '')
                                    color = (0, 255, 0)
                                    # Log to DB
                                    log_attendance_db(id_, user_data)
                                    
                                    # Update global state for UI Overlay
                                    latest_recognition = {
                                        "name": name,
                                        "emp_id": emp_id_disp,
                                        "time": time.time()
                                    }
                                else:
                                    print(f"ID {id_} not found in DB cache: {user_db.keys()}")
                        except Exception as e:
                            print(f"Prediction Error: {e}")
                            pass
                    
                    
                    draw_corner_rect(frame, (x,y,w,h), color, 3, 25)
                    
                    # Only show text if Unknown, otherwise the UI Card handles it
                    if name == "Unknown":
                         cv2.putText(frame, f"Unknown ({round(conf)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                         # Show PRESENT text below the face bracket
                         cv2.putText(frame, "PRESENT", (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
            with lock:
                outputFrame = frame.copy()
            time.sleep(0.01)

        except Exception as e:
            print(f"Stream Error: {e}")
            time.sleep(1)

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def finish_registration():
    global reg_status_msg, reg_faces, app_mode
    try:
        if save_model():
            reg_status_msg = "complete"
            speak_async("Registration complete.")
            time.sleep(3) # Show success message
        else:
            speak_async("Error saving model.")
            time.sleep(2)
    except:
        pass
        
    reg_status_msg = "idle"
    reg_faces = [] 
    reg_faces_color = []
    app_mode = 'idle' # Go back to home

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode')
def set_mode():
    global app_mode
    mode = request.args.get('mode')
    if mode in ['idle', 'attendance', 'registration']:
        app_mode = mode
        return jsonify({"success": True, "mode": app_mode})
    return jsonify({"success": False})

@app.route('/register_start')
def register_start():
    global app_mode, reg_name, reg_emp_id, reg_counter, reg_faces, reg_status_msg
    name = request.args.get('name')
    emp_id = request.args.get('emp_id')
    
    if name and emp_id:
        reg_name = name
        reg_emp_id = emp_id
        reg_counter = 0
        reg_faces = []
        reg_faces_color = []
        reg_status_msg = "processing"
        app_mode = 'registration'
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/registration_status')
def registration_status():
    return jsonify({
        "status": reg_status_msg,
        "progress": reg_counter,
        "mode": app_mode
    })

@app.route('/get_attendance')
def get_attendance():
    return jsonify(today_log)

@app.route('/current_recognition')
def current_recognition():
    # Only return if happened in last 3 seconds
    if time.time() - latest_recognition['time'] < 4.0:
        return jsonify(latest_recognition)
    return jsonify({}) # Empty if no recent match

@app.route('/user_image/<emp_id>')
def user_image(emp_id):
    # Fetch blob from DB
    try:
        conn = get_db_connection('face')
        cursor = conn.cursor()
        cursor.execute("SELECT profile_image FROM users WHERE emp_id = %s", (emp_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row and row[0]:
            return Response(row[0], mimetype='image/jpeg')
    except:
        pass
    # Return placeholder or empty
    return "", 404

if __name__ == '__main__':
    load_resources() # Fixed: Load users and model
    t = threading.Thread(target=process_video_stream)
    t.daemon = True
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

