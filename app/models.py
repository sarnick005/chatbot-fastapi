from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any, List
from datetime import datetime


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


class ModelMetrics(BaseModel):
    """Metrics for model performance monitoring"""

    accuracy: float
    total_samples: int
    version: str
    last_trained: datetime
    intent_distribution: Dict[str, int]
    confusion_matrix: Optional[List[List[float]]] = None
    training_duration: Optional[float] = None
    performance_by_intent: Dict[str, Dict[str, float]]
    model_config = ConfigDict(protected_namespaces=())


class ChatHistoryResponse(BaseModel):
    """Response model for chat history entries"""

    user_id: str
    timestamp: datetime
    user_input: str
    bot_response: str
    intent_tag: Optional[str] = None
    confidence_score: float
    feedback_score: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    model_version: str
    processing_time: Optional[float] = None
    model_config = ConfigDict(protected_namespaces=())
