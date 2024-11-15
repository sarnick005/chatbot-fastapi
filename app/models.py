from pydantic import BaseModel, ConfigDict
from typing import List, Optional


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


class IntentOutput(BaseModel):
    tag: str
    patterns: List[str]
    responses: List[str]


class IntentsResponse(BaseModel):
    intents: List[IntentOutput]
