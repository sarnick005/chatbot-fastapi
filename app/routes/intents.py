from fastapi import APIRouter, HTTPException, status
from app.models import IntentsData, ImportSummary, IntentsResponse, IntentOutput
from app.database import intents_collection

router = APIRouter()


@router.post("/import", response_model=ImportSummary)
async def import_intents(data: IntentsData):
    """Import intents data from JSON to MongoDB without deleting existing data."""
    try:
        new_intents_count = 0
        new_patterns_count = 0
        new_responses_count = 0
        updated_intents_count = 0

        for intent_data in data.intents:
            existing_intent = intents_collection.find_one({"tag": intent_data.tag})

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

                intents_collection.update_one(
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
                intents_collection.insert_one(
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


@router.get("/", response_model=IntentsResponse)
async def get_all_intents():
    """Retrieve all intents from MongoDB."""
    try:
        intents_data = list(intents_collection.find())
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
