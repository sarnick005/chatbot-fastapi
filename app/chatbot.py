from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from app.database import db


class SelfLearningChatbot:
    def __init__(self, confidence_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.conversation_history = []
        self._prepare_training_data()
        self._train_model()

    def _prepare_training_data(self):
        self.X = []
        self.y = []
        for intent in db.intents.find():
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                self.X.append(pattern.lower())
                self.y.append(tag)

    def _train_model(self):
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

    def predict_intent(self, user_input: str):
        processed_input = user_input.lower()
        predicted_tag = self.pipeline.predict([processed_input])[0]
        confidence = float(np.max(self.pipeline.predict_proba([processed_input])))
        return (
            predicted_tag if confidence >= self.confidence_threshold else None,
            confidence,
        )

    def get_response(self, tag: str):
        intent = db.intents.find_one({"tag": tag})
        if intent and intent["responses"]:
            return np.random.choice(intent["responses"])
        return "I'm still learning about that."

    def store_interaction(
        self, user_input, response, confidence, intent_tag, feedback_score
    ):
        chat_id = db.chat_history.insert_one(
            {
                "user_input": user_input,
                "bot_response": response,
                "confidence": confidence,
                "intent_tag": intent_tag,
                "feedback_score": feedback_score,
                "timestamp": datetime.now(),
            }
        ).inserted_id
        return str(chat_id)

    def update_feedback(self, chat_id, feedback_score):
        db.chat_history.update_one(
            {"_id": chat_id}, {"$set": {"feedback_score": feedback_score}}
        )
