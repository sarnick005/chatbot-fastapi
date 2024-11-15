from fastapi import APIRouter
from app.models import HealthResponse
from app.chatbot import SelfLearningChatbot

router = APIRouter()

chatbot = SelfLearningChatbot()


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Check the health of the API and the chatbot model."""
    return HealthResponse(status="healthy", model_accuracy=chatbot.accuracy)
