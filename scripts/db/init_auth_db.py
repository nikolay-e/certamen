import os
import sys
from pathlib import Path

import psycopg2

DB_HOST = os.getenv("CERTAMEN_DB_HOST", "localhost")
DB_PORT = int(os.getenv("CERTAMEN_DB_PORT", "5432"))
DB_NAME = os.getenv("CERTAMEN_DB_NAME", "certamen")
DB_USER = os.getenv("CERTAMEN_DB_USER", "certamen")
DB_PASSWORD = os.getenv("CERTAMEN_DB_PASSWORD", "")


def init_database():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        conn.autocommit = True
        cursor = conn.cursor()

        sql_file = Path(__file__).parent / "init_auth_db.sql"
        with open(sql_file) as f:
            sql_script = f.read()

        cursor.execute(sql_script)
        print("Database initialized successfully!")

        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"Current users count: {user_count}")

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f"User: {DB_USER}")
    init_database()
