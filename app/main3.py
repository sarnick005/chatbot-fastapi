from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from pathlib import Path
import numpy as np
import json
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load environment variables
load_dotenv()

app = FastAPI(title="Self-Learning Chatbot API")

# MongoDB configuration
client = MongoClient(os.getenv("MONGO_URI"))
db = client.get_database(os.getenv("DB_NAME"))
try:
    intents_collection = db.intents
    print("Connected with MongoDB")
except:
    print("Error connecting with MongoDB")
    exit()


def load_data_from_json():
    """Load intents data from 'intents.json'."""
    intents_file = Path("./intents.json")
    if not intents_file.exists():
        raise FileNotFoundError("intents.json file not found.")

    with open(intents_file, "r") as file:
        intents_data = json.load(file)
    return intents_data


@app.on_event("startup")
def initialize_data():
    """Check if data exists in MongoDB, if not, load from JSON and insert."""
    if intents_collection.count_documents({}) == 0:
        print("No data found in MongoDB. Loading data from intents.json...")
        try:
            intents_data = load_data_from_json()
            intents_collection.insert_many(intents_data)
            print("Data successfully loaded into MongoDB.")
        except Exception as e:
            print(f"Error loading data: {e}")


# Pydantic models for request/response validation
class ChatInput(BaseModel):
    user_input: str
    feedback_score: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent_tag: Optional[str] = None


class FeedbackInput(BaseModel):
    chat_id: str
    feedback_score: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_accuracy: float


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


# Define a response model for a single intent
class IntentOutput(BaseModel):
    tag: str
    patterns: List[str]
    responses: List[str]


# Define a response model for a list of intents
class IntentsResponse(BaseModel):
    intents: List[IntentOutput]


class SelfLearningChatbot:
    def __init__(self, confidence_threshold=0.4):
        """Initialize the chatbot and train the model."""
        self.confidence_threshold = confidence_threshold
        self.conversation_history = []
        self._prepare_training_data()
        self._train_model()

    def _prepare_training_data(self):
        """Prepare training data from both original and learned patterns."""
        self.X = []
        self.y = []

        # Fetch intents and patterns from MongoDB
        for intent in db.intents.find():
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                self.X.append(pattern.lower())
                self.y.append(tag)

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
        intent = db.intents.find_one({"tag": tag})
        if intent and intent["responses"]:
            return np.random.choice(intent["responses"])
        return "I'm still learning about that."

    def store_interaction(
        self,
        user_input: str,
        response: str,
        confidence: float,
        intent_tag: Optional[str],
        feedback_score: Optional[int],
    ) -> str:
        """Store chat interaction in MongoDB."""
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

    def update_feedback(self, chat_id: str, feedback_score: int):
        """Update feedback score for a chat interaction."""
        db.chat_history.update_one(
            {"_id": chat_id}, {"$set": {"feedback_score": feedback_score}}
        )


# Create global chatbot instance
chatbot = SelfLearningChatbot()


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
    """Import intents data from JSON to MongoDB without deleting existing data."""
    try:
        new_intents_count = 0
        new_patterns_count = 0
        new_responses_count = 0
        updated_intents_count = 0

        for intent_data in data.intents:
            existing_intent = db.intents.find_one({"tag": intent_data.tag})

            if existing_intent:
                updated_intents_count += 1
                new_patterns = [
                    p
                    for p in intent_data.patterns
                    if p not in existing_intent["patterns"]
                ]
                new_responses = [
                    r
                    for r in intent_data.responses
                    if r not in existing_intent["responses"]
                ]

                db.intents.update_one(
                    {"tag": intent_data.tag},
                    {
                        "$addToSet": {
                            "patterns": {"$each": new_patterns},
                            "responses": {"$each": new_responses},
                        }
                    },
                )
                new_patterns_count += len(new_patterns)
                new_responses_count += len(new_responses)
            else:
                db.intents.insert_one(
                    {
                        "tag": intent_data.tag,
                        "patterns": intent_data.patterns,
                        "responses": intent_data.responses,
                    }
                )
                new_intents_count += 1
                new_patterns_count += len(intent_data.patterns)
                new_responses_count += len(intent_data.responses)

        return ImportSummary(
            status="success",
            new_intents_count=new_intents_count,
            new_patterns_count=new_patterns_count,
            new_responses_count=new_responses_count,
            updated_intents_count=updated_intents_count,
            message="Data imported successfully!",
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@app.get("/get-intents", response_model=IntentsResponse)
async def get_intents():
    """Retrieve all intents from MongoDB."""
    try:
        intents_data = list(db.intents.find())
        intents = [
            IntentOutput(
                tag=intent["tag"],
                patterns=intent["patterns"],
                responses=intent["responses"],
            )
            for intent in intents_data
        ]
        return IntentsResponse(intents=intents)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
