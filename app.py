from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
import datetime
import threading
import time
import base64
import mysql.connector
from urllib.parse import urlparse

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_FILE = "trainer.yml"
ATTENDANCE_FILE = "attendance_log.csv"
COOLDOWN_SECONDS = 30
CONFIDENCE_THRESHOLD = 75  # Increased from 55 for better reliability
FACE_SIZE = (100, 100)      # Standard size for face samples

# Global State
app_mode = 'idle'
reg_name = ""
reg_emp_id = ""
reg_counter = 0
reg_faces = []
reg_status_msg = "idle"
today_log = []
reg_faces_color = []
latest_recognition = {"name": "", "emp_id": "", "time": 0}
user_db = {}

# Detectors
face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
if face_cascade.empty():
    print("WARNING: Face Cascade Classifier failed to load! Check opencv installation.")
recognizer = cv2.face.LBPHFaceRecognizer_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- DATABASE CONFIG ---
def get_db_connection(db_name=None):
    # Support DATABASE_URL (standard on Render) or individual vars
    db_url = os.environ.get('DATABASE_URL')
    
    if db_url:
        result = urlparse(db_url)
        db_from_url = result.path[1:] if result.path else ""
        target_db = db_name if db_name else (db_from_url if db_from_url else 'face')
        
        conf = {
            'host': result.hostname,
            'user': result.username,
            'password': result.password,
            'database': target_db,
            'port': result.port or 3306
        }
    else:
        conf = {
            'host': os.environ.get('DB_HOST', 'localhost'),
            'user': os.environ.get('DB_USER', 'root'), 
            'password': os.environ.get('DB_PASSWORD', ''),
            'database': os.environ.get('DB_NAME', 'face') if not db_name else db_name,
            'port': int(os.environ.get('DB_PORT', 3306))
        }
    
    try:
        conn = mysql.connector.connect(**conf)
        return conn
    except mysql.connector.Error as err:
        print(f"Database Connect Error: {err}")
        return None

def init_db():
    conn = get_db_connection()
    if not conn:
        print("CRITICAL: COULD NOT CONNECT TO MYSQL SERVER.")
        return

    cursor = conn.cursor()
    try:
        # If we didn't specify a DB in URL, we might need to create it
        db_url = os.environ.get('DATABASE_URL')
        if not db_url or 'localhost' in db_url:
            cursor.execute("CREATE DATABASE IF NOT EXISTS face")
    except Exception as e:
        print(f"Error creating DB: {e}")
    conn.close()
    
    conn = get_db_connection() # Connect to the actual database
    if not conn: return
        
    cursor = conn.cursor()
    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            emp_id VARCHAR(50) UNIQUE,
            profile_image LONGBLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Attendance table
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
    # System storage for trainer.yml (Persistence for Render)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_storage (
            key_name VARCHAR(50) PRIMARY KEY,
            data_blob LONGBLOB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def load_resources():
    global user_db, recognizer
    user_db = {}
    
    # 1. Try to load model from Local File first (if exists)
    model_loaded = False
    if os.path.exists(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            model_loaded = True
            print("Model loaded from local file.")
        except: pass
    
    # 2. If not loaded, try to load from Database (Render Persistence)
    if not model_loaded:
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT data_blob FROM system_storage WHERE key_name = 'trainer_model'")
                row = cursor.fetchone()
                if row and row[0]:
                    with open(MODEL_FILE, "wb") as f:
                        f.write(row[0])
                    recognizer.read(MODEL_FILE)
                    model_loaded = True
                    print("Model loaded from database.")
            except Exception as e:
                print(f"Error loading model from DB: {e}")
            finally:
                conn.close()

    # 3. Load user records
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, name, emp_id FROM users")
            for row in cursor.fetchall():
                user_db[row['id']] = {"name": row['name'], "emp_id": row['emp_id']}
            conn.close()
    except Exception as e:
        print(f"Error loading user records: {e}")
    
    if not model_loaded:
        print("No trained model found. Please register users.")

def log_attendance_db(internal_id, user_data):
    global today_log
    name = user_data['name']
    emp_id = user_data.get('emp_id', 'N/A')
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check cooldown locally to avoid excessive DB writes
    for entry in today_log:
        if entry['id'] == emp_id:
             last_time = datetime.datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
             if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
                 return

    today_log.append({"name": name, "id": emp_id, "time": time_str})
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (user_id, name, emp_id, timestamp) VALUES (%s, %s, %s, %s)", 
                      (internal_id, name, emp_id, now))
        conn.commit()
        conn.close()

def save_model_to_db():
    """Saves the trainer.yml file to the database for persistence."""
    if not os.path.exists(MODEL_FILE): return
    try:
        with open(MODEL_FILE, "rb") as f:
            blob = f.read()
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("REPLACE INTO system_storage (key_name, data_blob) VALUES ('trainer_model', %s)", (blob,))
            conn.commit()
            conn.close()
            print("Model backed up to database.")
    except Exception as e:
        print(f"Error backing up model: {e}")

def save_new_user():
    global user_db, reg_faces, reg_faces_color, reg_name, reg_emp_id, recognizer
    if not reg_faces: return False
    try:
        conn = get_db_connection()
        if not conn: return False
        cursor = conn.cursor()
        
        # Save profile image
        img_blob = None
        if reg_faces_color:
            best_frame = reg_faces_color[len(reg_faces_color)//2] 
            success, encoded = cv2.imencode('.jpg', best_frame)
            if success: img_blob = encoded.tobytes()

        try:
            cursor.execute("INSERT INTO users (name, emp_id, profile_image) VALUES (%s, %s, %s)", (reg_name, reg_emp_id, img_blob))
            conn.commit()
            new_internal_id = cursor.lastrowid
        except:
            # If user already exists, update them or just get ID
            cursor.execute("SELECT id FROM users WHERE emp_id = %s", (reg_emp_id,))
            new_internal_id = cursor.fetchone()[0]
        conn.close()

        user_db[new_internal_id] = {"name": reg_name, "emp_id": reg_emp_id}
        
        # Train recognizer
        faces_np = [np.array(f, dtype=np.uint8) for f in reg_faces]
        ids_np = np.array([new_internal_id]*len(reg_faces))

        if os.path.exists(MODEL_FILE):
             recognizer.update(faces_np, ids_np)
        else:
            recognizer.train(faces_np, ids_np)
            
        recognizer.write(MODEL_FILE)
        save_model_to_db() # Persist to DB
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global app_mode, reg_counter, reg_faces, reg_faces_color, reg_status_msg, latest_recognition
    
    data = request.json
    frame_data = data.get('image')
    if not frame_data:
        return jsonify({"success": False, "error": "No image data"})

    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return jsonify({"success": False, "error": "Decode failed"})
    except:
        return jsonify({"success": False, "error": "Invalid image format"})
    
    try:
        response_data = {"success": True, "faces": []}
        
        if app_mode == 'registration':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
            
            for (x,y,w,h) in faces:
                response_data['faces'].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
                if reg_counter < 40 and reg_status_msg == "processing":
                    # Standardize face size for better LBPH performance
                    face_roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                    reg_faces.append(face_roi)
                    reg_faces_color.append(frame[y:y+h, x:x+w])
                    reg_counter += 1
                    if reg_counter >= 40:
                        reg_status_msg = "training"
                        threading.Thread(target=finish_registration).start()
                        
        elif app_mode == 'attendance':
            # Optimization: process at 0.5x resolution for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            small_gray = clahe.apply(small_gray)
            faces_small = face_cascade.detectMultiScale(small_gray, 1.1, 5, minSize=(30, 30))
            
            # Prepare full-res gray for recognition
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_full = clahe.apply(gray_full)
    
            for (x, y, w, h) in faces_small:
                # Scale back to original frame size
                x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
                name = "Unknown"
                emp_id = ""
                conf_val = 999
                
                if user_db:
                    try:
                        # Crop and Resize to match training data size
                        face_roi = cv2.resize(gray_full[y:y+h, x:x+w], FACE_SIZE)
                        id_label, conf = recognizer.predict(face_roi)
                        conf_val = conf
                        
                        # LBPH: Confidence is distance. Lower is better. 0 is perfect.
                        # 75 is a reasonable threshold for variety in lighting.
                        if conf < CONFIDENCE_THRESHOLD:
                            user_data = user_db.get(id_label)
                            if user_data:
                                name = user_data['name']
                                emp_id = user_data['emp_id']
                                log_attendance_db(id_label, user_data)
                                latest_recognition = {"name": name, "emp_id": emp_id, "time": time.time()}
                    except Exception as e:
                        print(f"Prediction Error: {e}")
                
                response_data['faces'].append({
                    "x": x, "y": y, "w": w, "h": h, 
                    "name": name, "emp_id": emp_id, "conf": int(conf_val)
                })

        return jsonify(response_data)

    except Exception as e:
        print(f"Error in process_frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

def finish_registration():
    global reg_status_msg, reg_faces, reg_faces_color, app_mode
    if save_new_user():
        reg_status_msg = "complete"
    else:
        reg_status_msg = "error"
    time.sleep(2)
    reg_status_msg = "idle"
    reg_faces, reg_faces_color = [], []
    app_mode = 'idle'

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
    global app_mode, reg_name, reg_emp_id, reg_counter, reg_faces, reg_faces_color, reg_status_msg
    name, emp_id = request.args.get('name'), request.args.get('emp_id')
    if name and emp_id:
        reg_name, reg_emp_id, reg_counter = name, emp_id, 0
        reg_faces, reg_faces_color = [], []
        reg_status_msg, app_mode = "processing", 'registration'
        return jsonify({"success": True})
    return jsonify({"success": False})

@app.route('/registration_status')
def registration_status():
    return jsonify({"status": reg_status_msg, "progress": reg_counter, "mode": app_mode})

@app.route('/get_attendance')
def get_attendance():
    return jsonify(today_log)

@app.route('/current_recognition')
def current_recognition():
    if time.time() - latest_recognition['time'] < 4.0:
        return jsonify(latest_recognition)
    return jsonify({})

@app.route('/user_image/<emp_id>')
def user_image(emp_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT profile_image FROM users WHERE emp_id = %s", (emp_id,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]: return Response(row[0], mimetype='image/jpeg')
    except: pass
    return "", 404

# Initialize App
with app.app_context():
    init_db()
    load_resources()

if __name__ == '__main__':
    # For local running
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
