import mysql.connector
import os
from urllib.parse import urlparse

def get_db_connection():
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        result = urlparse(db_url)
        conf = {
            'host': result.hostname,
            'user': result.username,
            'password': result.password,
            'database': result.path[1:],
            'port': result.port or 3306
        }
    else:
        conf = {
            'host': 'localhost',
            'user': 'root', 
            'password': '',
            'database': 'face'
        }
    return mysql.connector.connect(**conf)

def reset_system():
    # 1. Clear Database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Disable FK checks to allow truncation
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables = ['attendance', 'users', 'system_storage']
        for table in tables:
            try:
                print(f"Truncating '{table}' table...")
                cursor.execute(f"TRUNCATE TABLE {table}")
            except:
                print(f"Table '{table}' not found or could not be truncated.")
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        
        conn.commit()
        conn.close()
        print("Database cleared.")
    except Exception as e:
        print(f"Database Error: {e}")

    # 2. Delete Files
    files_to_delete = [
        "trainer.yml",
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
