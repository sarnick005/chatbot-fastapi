from pydantic import BaseModel, ConfigDict
from typing import Optional, List


class ChatInput(BaseModel):
    user_input: str
    feedback_score: Optional[int] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    intent_tag: Optional[str] = None


class FeedbackInput(BaseModel):
    chat_id: int
    feedback_score: int


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_accuracy: float
