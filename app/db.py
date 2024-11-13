import json
import mysql.connector
import os


def create_tables(cursor):
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

    # Create the user_feedback table
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


def import_data(json_data, db_config):
    """Import data from JSON to MySQL database."""
    try:
        # Connect to MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Create tables
        create_tables(cursor)

        # Clear existing data
        cursor.execute("DELETE FROM responses")
        cursor.execute("DELETE FROM patterns")
        cursor.execute("DELETE FROM intents")

        # Import data
        for intent in json_data["intents"]:
            # Insert intent
            cursor.execute("INSERT INTO intents (tag) VALUES (%s)", (intent["tag"],))
            intent_id = cursor.lastrowid

            # Insert patterns
            for pattern in intent["patterns"]:
                cursor.execute(
                    "INSERT INTO patterns (intent_id, pattern) VALUES (%s, %s)",
                    (intent_id, pattern),
                )

            # Insert responses
            for response in intent["responses"]:
                cursor.execute(
                    "INSERT INTO responses (intent_id, response) VALUES (%s, %s)",
                    (intent_id, response),
                )

        # Commit changes
        conn.commit()
        print("Data imported successfully!")

        # Print summary
        cursor.execute("SELECT COUNT(*) FROM intents")
        intents_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM patterns")
        patterns_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM responses")
        responses_count = cursor.fetchone()[0]

        print(f"\nImport Summary:")
        print(f"Intents: {intents_count}")
        print(f"Patterns: {patterns_count}")
        print(f"Responses: {responses_count}")

    except Exception as e:
        print(f"Error importing data: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


def main():
    # Check if JSON file exists
    json_file = "intents.json"
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found!")
        return

    # MySQL database configuration
    db_config = {
        "user": "your_username",  # replace with your MySQL username
        "password": "your_password",  # replace with your MySQL password
        "host": "localhost",  # or your MySQL host
        "database": "your_database",  # replace with your MySQL database name
    }

    # Read JSON data
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        print("JSON data loaded successfully!")

        # Import data to database
        import_data(json_data, db_config)

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()

