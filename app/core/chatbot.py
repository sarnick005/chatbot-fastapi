import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple
from app.config.database import create_connection
import mysql.connector  # Import mysql.connector for database connections
from mysql.connector import Error  # Import Error to handle connection errors
from fastapi import HTTPException, status


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
