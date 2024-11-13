from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict
import numpy as np
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import os
from typing import List
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

app = FastAPI(title="Self Learning Chatbot API")

# Database configuration from environment variables
db_config = {
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME"),
}


# Pydantic models for request/response validation
class ChatInput(BaseModel):
    user_input: str  # Changed from message to user_input
    feedback_score: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent_tag: Optional[str] = None


class FeedbackInput(BaseModel):
    chat_id: int
    feedback_score: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # Fix for the warning
    status: str
    model_accuracy: float


# Pydantic models
class Pattern(BaseModel):
    pattern: str


class Response(BaseModel):
    response: str


class Intent(BaseModel):
    tag: str
    patterns: List[str]
    responses: List[str]


class IntentsData(BaseModel):
    intents: List[Intent]


class ImportSummary(BaseModel):
    status: str
    new_intents_count: int
    new_patterns_count: int
    new_responses_count: int
    updated_intents_count: int
    message: str


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


class SelfLearningChatbot:
    def __init__(self, db_config, confidence_threshold=0.4):
        """Initialize the self-learning chatbot with MySQL database."""
        self.db_config = db_config
        self.confidence_threshold = confidence_threshold
        self.conversation_history = []
        self._init_database()
        self._prepare_training_data()
        self._train_model()

    def _create_connection(self):
        """Create a MySQL database connection."""
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                print("Connected to DB")
                return connection
        except Error as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database connection error: {str(e)}",
            )

    def _init_database(self):
        """Initialize the database with required tables."""
        connection = self._create_connection()
        if connection:
            cursor = connection.cursor()
            # Create tables if they don't exist
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_input TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    confidence FLOAT,
                    intent_tag VARCHAR(255),
                    feedback_score INT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    intent_id INT,
                    pattern TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            connection.commit()
            cursor.close()
            connection.close()

    def _prepare_training_data(self):
        """Prepare training data from both original and learned patterns."""
        self.X = []
        self.y = []

        connection = self._create_connection()
        if connection:
            cursor = connection.cursor()

            # Get original patterns
            cursor.execute(
                """
                SELECT p.pattern, i.tag 
                FROM patterns p 
                JOIN intents i ON p.intent_id = i.id
            """
            )
            for pattern, tag in cursor.fetchall():
                self.X.append(pattern.lower())
                self.y.append(tag)

            # Get learned patterns
            cursor.execute(
                """
                SELECT lp.pattern, i.tag 
                FROM learned_patterns lp 
                JOIN intents i ON lp.intent_id = i.id
            """
            )
            for pattern, tag in cursor.fetchall():
                self.X.append(pattern.lower())
                self.y.append(tag)

            cursor.close()
            connection.close()

    def _train_model(self):
        """Train the model with available data."""
        if not self.X or not self.y:
            self.accuracy = 0
            raise ValueError("No training data available")

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.pipeline.fit(X_train, y_train)
        self.accuracy = self.pipeline.score(X_test, y_test)

    def predict_intent(self, user_input: str) -> tuple:
        """Predict intent with confidence score."""
        processed_input = user_input.lower()
        predicted_tag = self.pipeline.predict([processed_input])[0]
        confidence = float(np.max(self.pipeline.predict_proba([processed_input])))

        return (
            predicted_tag if confidence >= self.confidence_threshold else None,
            confidence,
        )

    def get_response(self, tag: str) -> str:
        """Get response for the predicted tag."""
        connection = self._create_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT response 
                FROM responses r
                JOIN intents i ON r.intent_id = i.id
                WHERE i.tag = %s
            """,
                (tag,),
            )

            responses = cursor.fetchall()
            cursor.close()
            connection.close()

            if responses:
                return str(np.random.choice([r[0] for r in responses]))
        return "I'm still learning about that."

    def store_interaction(
        self,
        user_input: str,
        response: str,
        confidence: float,
        intent_tag: Optional[str],
        feedback_score: Optional[int],
    ) -> int:
        """Store chat interaction in database."""
        connection = self._create_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO chat_history 
                (user_input, bot_response, confidence, intent_tag, feedback_score)
                VALUES (%s, %s, %s, %s, %s)
            """,
                (user_input, response, confidence, intent_tag, feedback_score),
            )

            chat_id = cursor.lastrowid
            connection.commit()
            cursor.close()
            connection.close()
            return chat_id

    def update_feedback(self, chat_id: int, feedback_score: int):
        """Update feedback score for a chat interaction."""
        connection = self._create_connection()
        if connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE chat_history 
                SET feedback_score = %s 
                WHERE id = %s
            """,
                (feedback_score, chat_id),
            )

            connection.commit()
            cursor.close()
            connection.close()


# Create global chatbot instance
chatbot = SelfLearningChatbot(db_config)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_input: ChatInput):
    """Handle chat interactions."""
    try:
        predicted_tag, confidence = chatbot.predict_intent(chat_input.user_input)

        if predicted_tag is None:
            response = (
                "I'm not quite sure about that yet. Could you rephrase or teach me?"
            )
            confidence = 0.0
        else:
            response = chatbot.get_response(predicted_tag)

        # Store interaction
        chat_id = chatbot.store_interaction(
            chat_input.user_input,
            response,
            confidence,
            predicted_tag,
            chat_input.feedback_score,
        )
        print(chat_id)
        print(f"User:{chat_input.user_input}")
        print(response)

        return ChatResponse(
            response=response, confidence=confidence, intent_tag=predicted_tag
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.post("/feedback")
async def feedback_endpoint(feedback: FeedbackInput):
    """Handle feedback updates."""
    try:
        chatbot.update_feedback(feedback.chat_id, feedback.feedback_score)
        return {"status": "success", "message": "Feedback updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_accuracy=chatbot.accuracy)


@app.post("/import-intents", response_model=ImportSummary)
async def import_intents(data: IntentsData):
    """Import intents data from JSON to database without deleting existing data."""
    try:
        # Get database configuration from environment variables
        db_config = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME"),
        }

        # Connect to MySQL database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Create tables if they don't exist
        await create_tables(cursor)

        # Initialize counters
        new_intents_count = 0
        new_patterns_count = 0
        new_responses_count = 0
        updated_intents_count = 0

        # Import data
        for intent in data.intents:
            # Check if intent already exists
            cursor.execute("SELECT id FROM intents WHERE tag = %s", (intent.tag,))
            existing_intent = cursor.fetchone()

            if existing_intent:
                # Intent exists, update patterns and responses
                intent_id = existing_intent["id"]
                updated_intents_count += 1

                # Get existing patterns
                cursor.execute(
                    "SELECT pattern FROM patterns WHERE intent_id = %s", (intent_id,)
                )
                existing_patterns = {row["pattern"] for row in cursor.fetchall()}

                # Add new patterns
                for pattern in intent.patterns:
                    if pattern not in existing_patterns:
                        cursor.execute(
                            "INSERT INTO patterns (intent_id, pattern) VALUES (%s, %s)",
                            (intent_id, pattern),
                        )
                        new_patterns_count += 1

                # Get existing responses
                cursor.execute(
                    "SELECT response FROM responses WHERE intent_id = %s", (intent_id,)
                )
                existing_responses = {row["response"] for row in cursor.fetchall()}

                # Add new responses
                for response in intent.responses:
                    if response not in existing_responses:
                        cursor.execute(
                            "INSERT INTO responses (intent_id, response) VALUES (%s, %s)",
                            (intent_id, response),
                        )
                        new_responses_count += 1
            else:
                # Insert new intent
                cursor.execute("INSERT INTO intents (tag) VALUES (%s)", (intent.tag,))
                intent_id = cursor.lastrowid
                new_intents_count += 1

                # Insert all patterns
                for pattern in intent.patterns:
                    cursor.execute(
                        "INSERT INTO patterns (intent_id, pattern) VALUES (%s, %s)",
                        (intent_id, pattern),
                    )
                    new_patterns_count += 1

                # Insert all responses
                for response in intent.responses:
                    cursor.execute(
                        "INSERT INTO responses (intent_id, response) VALUES (%s, %s)",
                        (intent_id, response),
                    )
                    new_responses_count += 1

        # Commit changes
        conn.commit()

        return ImportSummary(
            status="success",
            new_intents_count=new_intents_count,
            new_patterns_count=new_patterns_count,
            new_responses_count=new_responses_count,
            updated_intents_count=updated_intents_count,
            message="Data imported successfully!",
        )

    except mysql.connector.Error as e:
        if "conn" in locals() and conn.is_connected():
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )
    except Exception as e:
        if "conn" in locals() and conn.is_connected():
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error importing data: {str(e)}",
        )
    finally:
        if "conn" in locals() and conn.is_connected():
            cursor.close()
            conn.close()


# Get intents route remains the same
@app.get("/get-intents", response_model=IntentsData)
async def get_intents():
    """Retrieve all intents data from database."""
    try:
        db_config = {
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME"),
        }

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        intents_list = []

        # Get all intents
        cursor.execute("SELECT * FROM intents")
        intents = cursor.fetchall()

        for intent in intents:
            # Get patterns for this intent
            cursor.execute(
                "SELECT pattern FROM patterns WHERE intent_id = %s", (intent["id"],)
            )
            patterns = [row["pattern"] for row in cursor.fetchall()]

            # Get responses for this intent
            cursor.execute(
                "SELECT response FROM responses WHERE intent_id = %s", (intent["id"],)
            )
            responses = [row["response"] for row in cursor.fetchall()]

            intents_list.append(
                {"tag": intent["tag"], "patterns": patterns, "responses": responses}
            )

        return IntentsData(intents=intents_list)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving intents: {str(e)}",
        )
    finally:
        if "conn" in locals() and conn.is_connected():
            cursor.close()
            conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
