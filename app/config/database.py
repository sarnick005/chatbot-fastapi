import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from fastapi import HTTPException, status

# Load environment variables
load_dotenv()

# Database configuration
db_config = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME"),
}


def create_connection():
    """Create a MySQL database connection."""
    try:
        connection = mysql.connector.connect(**db_config)
        if connection.is_connected():
            return connection
    except Error as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}",
        )


async def create_tables(cursor):
    """Create the necessary tables if they don't exist."""
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS intents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tag VARCHAR(255) UNIQUE NOT NULL
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patterns (
            id INT AUTO_INCREMENT PRIMARY KEY,
            intent_id INT,
            pattern TEXT NOT NULL,
            FOREIGN KEY (intent_id) REFERENCES intents (id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            id INT AUTO_INCREMENT PRIMARY KEY,
            intent_id INT,
            response TEXT NOT NULL,
            FOREIGN KEY (intent_id) REFERENCES intents (id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_input TEXT NOT NULL,
            bot_response TEXT,
            intent_tag VARCHAR(255),
            feedback_score INT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
