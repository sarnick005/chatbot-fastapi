from pymongo import MongoClient
from pathlib import Path
import json
from app.config import MONGO_URI, DB_NAME

try:
    # Establish MongoDB connection
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

intents_collection = db.intents
chat_history_collection = db.chat_history


def load_data_from_json():
    """Load intents data from 'intents.json'."""
    intents_file = Path("./intents.json")
    if not intents_file.exists():
        raise FileNotFoundError("intents.json file not found.")
    with open(intents_file, "r") as file:
        return json.load(file)


def initialize_database():
    """Check and initialize the database with intents data."""
    if intents_collection.count_documents({}) == 0:
        print("No data found in MongoDB. Loading data from intents.json...")
        intents_data = load_data_from_json()
        intents_collection.insert_many(intents_data)
        print("Data successfully loaded into MongoDB.")
