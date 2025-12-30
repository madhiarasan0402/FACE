from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import pickle
import numpy as np
import datetime
import threading
import time
import base64
import mysql.connector

app = Flask(__name__)

# --- CONFIGURATION ---
DATA_FILE = "encodings.pickle" 
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
ATTENDANCE_FILE = "attendance_log.csv"
COOLDOWN_SECONDS = 30

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# --- DATABASE CONFIG ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '', # Default XAMPP/WAMP password
    'database': 'face'
}

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
    conn = get_db_connection()
    if not conn:
        print("CRITICAL: COULD NOT CONNECT TO MYSQL SERVER.")
        return

    cursor = conn.cursor()
    try:
        cursor.execute("CREATE DATABASE IF NOT EXISTS face")
    except Exception as e:
        print(f"Error creating DB: {e}")
    conn.close()
    
    conn = get_db_connection('face')
    if not conn: return
        
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(100),
            emp_id VARCHAR(50) UNIQUE,
            profile_image LONGBLOB,
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
    conn.commit()
    conn.close()

def load_resources():
    global user_db, recognizer
    user_db = {}
    if os.path.exists(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            print("Model loaded.")
        except: pass
    
    conn = get_db_connection('face')
    if conn:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name, emp_id FROM users")
        for row in cursor.fetchall():
            user_db[row['id']] = {"name": row['name'], "emp_id": row['emp_id']}
        conn.close()

def log_attendance_db(internal_id, user_data):
    global today_log
    name = user_data['name']
    emp_id = user_data.get('emp_id', 'N/A')
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    for entry in today_log:
        if entry['id'] == emp_id and entry['name'] == name:
             last_time = datetime.datetime.strptime(entry['time'], "%Y-%m-%d %H:%M:%S")
             if (now - last_time).total_seconds() < COOLDOWN_SECONDS:
                 return

    today_log.append({"name": name, "id": emp_id, "time": time_str})
    conn = get_db_connection('face')
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO attendance (user_id, name, emp_id, timestamp) VALUES (%s, %s, %s, %s)", 
                      (internal_id, name, emp_id, now))
        conn.commit()
        conn.close()

def save_model():
    global user_db, reg_faces, reg_faces_color, reg_name, reg_emp_id, recognizer
    if not reg_faces: return False
    try:
        conn = get_db_connection('face')
        if not conn: return False
        cursor = conn.cursor()
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
            cursor.execute("SELECT id FROM users WHERE emp_id = %s", (reg_emp_id,))
            new_internal_id = cursor.fetchone()[0]
        conn.close()

        user_db[new_internal_id] = {"name": reg_name, "emp_id": reg_emp_id}
        faces_np = [np.array(f, dtype=np.uint8) for f in reg_faces]
        ids_np = np.array([new_internal_id]*len(reg_faces))

        if os.path.exists(MODEL_FILE):
             recognizer.update(faces_np, ids_np)
        else:
            recognizer.train(faces_np, ids_np)
        recognizer.write(MODEL_FILE)
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

    # Decode base64 image
    img_data = base64.b64decode(frame_data.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    response_data = {"success": True, "faces": []}
    
    if app_mode == 'registration':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x,y,w,h) in faces:
            response_data['faces'].append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            if reg_counter < 40 and reg_status_msg == "processing":
                reg_faces.append(gray[y:y+h, x:x+w])
                reg_faces_color.append(frame[y:y+h, x:x+w])
                reg_counter += 1
                if reg_counter >= 40:
                    reg_status_msg = "training"
                    threading.Thread(target=finish_registration).start()
                    
    elif app_mode == 'attendance':
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        small_gray = clahe.apply(small_gray)
        faces_small = face_cascade.detectMultiScale(small_gray, 1.1, 4)
        
        for (x, y, w, h) in faces_small:
            x, y, w, h = int(x*2), int(y*2), int(w*2), int(h*2)
            name = "Unknown"
            emp_id = ""
            conf_val = 999
            
            if user_db:
                try:
                    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_full = clahe.apply(gray_full)
                    id_, conf = recognizer.predict(gray_full[y:y+h, x:x+w])
                    conf_val = conf
                    if conf < 55:
                        user_data = user_db.get(id_)
                        if user_data:
                            name = user_data['name']
                            emp_id = user_data['emp_id']
                            log_attendance_db(id_, user_data)
                            latest_recognition = {"name": name, "emp_id": emp_id, "time": time.time()}
                except: pass
            
            response_data['faces'].append({
                "x": x, "y": y, "w": w, "h": h, 
                "name": name, "emp_id": emp_id, "conf": int(conf_val)
            })
            
    return jsonify(response_data)

def finish_registration():
    global reg_status_msg, reg_faces, reg_faces_color, app_mode
    if save_model():
        reg_status_msg = "complete"
        time.sleep(3)
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
        conn = get_db_connection('face')
        cursor = conn.cursor()
        cursor.execute("SELECT profile_image FROM users WHERE emp_id = %s", (emp_id,))
        row = cursor.fetchone()
        conn.close()
        if row and row[0]: return Response(row[0], mimetype='image/jpeg')
    except: pass
    return "", 404

if __name__ == '__main__':
    init_db()
    load_resources()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
