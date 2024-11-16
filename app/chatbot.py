from datetime import datetime
import numpy as np
from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from app.database import db


class SelfLearningChatbot:
    def __init__(self, confidence_threshold=0.4, retraining_threshold=100):
        self.confidence_threshold = confidence_threshold
        self.retraining_threshold = retraining_threshold
        self.conversation_history = []
        self.interactions_since_retrain = 0
        self._prepare_training_data()
        self._train_model()

    def _prepare_training_data(self):
        """Prepare training data from both intents and high-confidence historical interactions"""
        self.X = []
        self.y = []

        # Get patterns from predefined intents
        for intent in db.intents.find():
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                self.X.append(pattern.lower())
                self.y.append(tag)

        # Get historical interactions with high feedback scores
        historical_data = db.chat_history.find(
            {
                "feedback_score": {
                    "$gte": 4
                },  # Only use interactions with good feedback
                "confidence": {"$gte": 0.8},  # Only use high-confidence predictions
            }
        )

        for interaction in historical_data:
            self.X.append(interaction["user_input"].lower())
            self.y.append(interaction["intent_tag"])

    def _train_model(self):
        if not self.X or not self.y:
            self.accuracy = 0
            raise ValueError("No training data available")

        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=100, random_state=42, class_weight="balanced"
                    ),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        self.pipeline.fit(X_train, y_train)
        self.accuracy = self.pipeline.score(X_test, y_test)

    def predict_intent(self, user_input: str):
        """Predict intent with confidence score"""
        processed_input = user_input.lower()
        predicted_tag = self.pipeline.predict([processed_input])[0]
        confidence = float(np.max(self.pipeline.predict_proba([processed_input])))
        return (
            predicted_tag if confidence >= self.confidence_threshold else None,
            confidence,
        )

    def get_response(self, tag: str):
        """Get response based on predicted intent tag"""
        intent = db.intents.find_one({"tag": tag})
        if intent and intent["responses"]:
            return np.random.choice(intent["responses"])
        return "I'm still learning about that."

    def store_interaction(
        self,
        user_input: str,
        response: str,
        confidence: float,
        intent_tag: str,
        feedback_score: int,
    ) -> str:
        """Store chat interaction with metadata"""
        try:
            chat_id = db.chat_history.insert_one(
                {
                    "user_input": user_input,
                    "bot_response": response,
                    "confidence": confidence,
                    "intent_tag": intent_tag,
                    "feedback_score": feedback_score,
                    "timestamp": datetime.now(),
                    "model_accuracy": self.accuracy,
                    "model_version": self._get_model_version(),
                }
            ).inserted_id

            self.interactions_since_retrain += 1

            # Check if we should retrain the model
            if self.interactions_since_retrain >= self.retraining_threshold:
                self._retrain_model()

            return str(chat_id)

        except Exception as e:
            print(f"Error storing interaction: {e}")
            return None

    def update_feedback(self, chat_id: str, feedback_score: int) -> bool:
        """Update feedback score for a chat interaction"""
        try:
            # Validate chat_id format
            if not ObjectId.is_valid(chat_id):
                return False

            result = db.chat_history.update_one(
                {"_id": ObjectId(chat_id)},
                {
                    "$set": {
                        "feedback_score": feedback_score,
                        "feedback_timestamp": datetime.now(),
                    }
                },
            )

            # If feedback is very positive, consider for model improvement
            if feedback_score >= 4:
                interaction = db.chat_history.find_one({"_id": ObjectId(chat_id)})
                if interaction and interaction.get("confidence", 0) >= 0.8:
                    self._update_training_data(
                        interaction["user_input"], interaction["intent_tag"]
                    )

            return result.modified_count > 0

        except Exception as e:
            print(f"Error updating feedback: {e}")
            return False

    def _retrain_model(self):
        """Retrain the model with updated data"""
        self._prepare_training_data()
        self._train_model()
        self.interactions_since_retrain = 0

        # Store model metrics
        db.model_versions.insert_one(
            {
                "version": self._get_model_version(),
                "accuracy": self.accuracy,
                "training_samples": len(self.X),
                "timestamp": datetime.now(),
            }
        )

    def _update_training_data(self, user_input: str, intent_tag: str):
        """Add new training data point"""
        self.X.append(user_input.lower())
        self.y.append(intent_tag)

    def _get_model_version(self) -> str:
        """Generate a version identifier for the model"""
        return f"{datetime.now().strftime('%Y%m%d')}_{len(self.X)}"

    def get_performance_metrics(self):
        """Get model performance metrics"""
        return {
            "accuracy": self.accuracy,
            "training_samples": len(self.X),
            "interactions_since_retrain": self.interactions_since_retrain,
            "version": self._get_model_version(),
        }
