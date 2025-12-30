import cv2
import os
import pickle
import numpy as np
import datetime
import pyttsx3
import time

# Configuration
DATA_FILE = "encodings.pickle" # We will reuse this name but store LBPH model data differently or separate files
MODEL_FILE = "trainer.yml"
LABELS_FILE = "labels.pickle"
ATTENDANCE_FILE = "attendance_log.csv"
COOLDOWN_SECONDS = 30
CONFIDENCE_THRESHOLD = 50 
# Note: For LBPH, confidence is 'distance'. 
# Lower is better. 0 = perfect match. < 50 is good. > 80 is unknown.

# Initialize Text-to-Speech
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
except Exception as e:
    print(f"Warning: TTS initialization failed: {e}")
    engine = None

def speak(text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    print(f"[VOICE]: {text}")

# Initialize OpenCV Face Detector (Haar Cascade) works out of box
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def load_data():
    """Laws trained model and labels."""
    labels = {}
    if os.path.exists(MODEL_FILE):
        try:
            recognizer.read(MODEL_FILE)
            print("Model loaded.")
        except Exception:
            print("Model file exists but could not be read. Train first.")
    
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "rb") as f:
            labels = pickle.load(f) # {name: id}
            # Invert to {id: name}
            labels = {v: k for k, v in labels.items()}
            
    return labels

def save_data(faces, ids, label_map):
    """
    Trains the recognizer on ALL data and saves it.
    faces: list of face numpy arrays
    ids: list of integer ids corresponding to faces
    label_map: dict {name: id} to save
    """
    if not faces:
        return

    print("Training model...")
    recognizer.train(faces, np.array(ids))
    recognizer.write(MODEL_FILE)
    
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)
    
    print("Model saved successfully.")

def mark_attendance(name):
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'w') as f:
            f.write("Name,Timestamp\n")
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{timestamp_str}\n")

def get_face_capture(cap):
    """Captures a single face from camera with UI."""
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray) # Apply CLAHE
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw all
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        
        cv2.putText(frame, "Press 'c' to capture, 'q' to cancel", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Registration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                return gray[y:y+h, x:x+w]
            elif len(faces) == 0:
                print("No face detected.")
            else:
                print("Multiple faces detected.")
        if key == ord('q'):
            return None



def register_new_user(cap, current_label_map):
    """
    Walks user through capturing multiple photos for training.
    Uses Tkinter for input to avoid freezing the OpenCV window.
    """
    # Create a hidden root window for the dialog
    import tkinter as tk
    from tkinter import simpledialog
    
    root = tk.Tk()
    root.withdraw() # Hide the main window
    root.attributes('-topmost', True) # Make sure popup is on top
    
    speak("Do you want to register a new user?")
    # We can just prompt for name immediately
    name = simpledialog.askstring("Register Face", "Enter Name for new user (Click Cancel to skip):")
    root.destroy()
    
    if not name: 
        speak("Registration cancelled.")
        return current_label_map
    
    if name in current_label_map:
        user_id = current_label_map[name]
    else:
        # Generate new ID
        try:
            user_id = max(current_label_map.values()) + 1 if current_label_map else 1
        except:
            user_id = 1
        current_label_map[name] = user_id
        
    print(f"Capturing faces for {name}. Look at the camera.")
    speak("Please look at the camera. I need to take 10 photos.")
    
    collected_faces = []
    collected_ids = []
    
    count = 0
    while count < 20: # Take 20 samples
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray) # Apply CLAHE
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            # Save the face region
            collected_faces.append(gray[y:y+h, x:x+w])
            collected_ids.append(user_id)
            count += 1
            cv2.waitKey(100) # Small delay
            
        cv2.putText(frame, f"Capturing: {count}/20", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Attendance System', frame)
        cv2.waitKey(1)
        
        if count >= 20:
             break

    speak("Capture complete. Model updating.")
    
    # We need to re-train the model with ALL data. 
    recognizer.update(collected_faces, np.array(collected_ids))
    recognizer.write(MODEL_FILE)
    
    # Save labels
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(current_label_map, f)
        
    speak("Registration successful.")
    return current_label_map


def main():
    import tkinter as tk
    from tkinter import messagebox

    # Load labels
    id_to_name = load_data() 
    # Create name_to_id map
    name_to_id = {v: k for k, v in id_to_name.items()}

    # Try using DirectShow (CAP_DSHOW) which is often more stable on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
         print("Camera 0 failed. Trying default backend...")
         cap = cv2.VideoCapture(0)
         
    if not cap.isOpened():
        print("ERROR: Could not open video source. Please check your camera permissions.")
        speak("Error. I cannot see the camera.")
        return
        
    # ASK USER ON STARTUP
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    if messagebox.askyesno("Attendance System", "Do you want to add a new user now?"):
        name_to_id = register_new_user(cap, name_to_id)
        id_to_name = {v: k for k, v in name_to_id.items()}
    root.destroy()
    
    last_attendance = {} # name: time

    print("System Ready.")
    if not id_to_name:
        speak("No users registered. Press 'a' to add a user.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Optimization: Resize for detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        small_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        small_gray = clahe.apply(small_gray)

        faces_small = face_cascade.detectMultiScale(small_gray, 1.2, 5)
        
        faces = []
        for (x, y, w, h) in faces_small:
            faces.append((x*2, y*2, w*2, h*2))

        for (x, y, w, h) in faces:
            # Recognize
            if id_to_name: # Only if we have a model
                try:
                    # Use full res gray with CLAHE
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = clahe.apply(gray)
                    id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    # Check confidence
                    # Confidence is distance: 0 is match, 100+ is mismatch
                    if confidence < 75: 
                        name = id_to_name.get(id_, "Unknown")
                        conf_text = f" {round(100 - confidence)}%"
                        
                        # Attendance
                        now = datetime.datetime.now()
                        if name in last_attendance:
                             delta = (now - last_attendance[name]).total_seconds()
                        else:
                             delta = 999
                             
                        if delta > COOLDOWN_SECONDS:
                            mark_attendance(name)
                            last_attendance[name] = now
                            speak(f"Thank you {name}")
                            
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        conf_text = f" {round(100 - confidence)}%"
                        color = (0, 0, 255)
                except Exception as e:
                    name = "Unknown"
                    color = (0, 0, 255)
            else:
                 name = "Unknown"
                 color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(frame, "'a': Add User | 'q': Quit", (10, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
        cv2.imshow('Attendance System', frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('a'):
            name_to_id = register_new_user(cap, name_to_id)
            # Rebuild reverse map
            id_to_name = {v: k for k, v in name_to_id.items()}

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
