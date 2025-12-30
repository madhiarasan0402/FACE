# Face Recognition Attendance System

A standalone Python application for real-time attendance tracking using facial recognition.

## Features
- Real-time face detection and recognition.
- Automatic attendance logging with timestamps.
- Text-to-speech feedback ("Thank you, [Name]").
- In-app enrollment of new faces.
- Duplicate entry prevention (30-second cooldown).
- CSV export of attendance records.

## Prerequisites

1.  **Python 3.8+**
2.  **C++ Build Tools**:
    - The `dlib` library (dependency of `face_recognition`) requires C++ compilers.
    - On Windows, install **Visual Studio Community** with "Desktop development with C++".
    - Alternatively, try installing a pre-built wheel for dlib if compilation fails.

## Installation

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `dlib` fails to install, ensure you have CMake and C++ build tools installed.*

## Usage

1.  Run the application:
    ```bash
    python attendance_app.py
    ```

2.  **First Run / Enrollment**:
    - If no faces are registered, the app will prompt you to press 'a'.
    - Press `a` on your keyboard to register a new face.
    - Look at the camera.
    - Enter the name in the console window when prompted.

3.  **Attendance**:
    - Just walk in front of the camera.
    - If recognized, it will say "Thank you" and mark "Present" on screen.
    - Attendance is saved to `attendance_log.csv`.

4.  **Controls**:
    - `q`: Quit application.
    - `a`: Add/Register a new face.

## Data Storage
- **Face Encodings**: Stored in `encodings.pickle`.
- **Logs**: stored in `attendance_log.csv`.
