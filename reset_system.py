import mysql.connector
import os

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root', 
    'password': '',
    'database': 'face'
}

def reset_system():
    # 1. Clear Database
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Disable FK checks to allow truncation
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        print("Truncating 'attendance' table...")
        cursor.execute("TRUNCATE TABLE attendance")
        
        print("Truncating 'users' table...")
        cursor.execute("TRUNCATE TABLE users")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        conn.commit()
        conn.close()
        print("Database cleared.")
    except Exception as e:
        print(f"Database Error: {e}")

    # 2. Delete Files
    files_to_delete = [
        "trainer.yml",
        "encodings.pickle",
        "labels.pickle",
        "attendance_log.csv"
    ]
    
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        else:
            print(f"{f} not found.")

if __name__ == "__main__":
    reset_system()
