from fastapi import APIRouter, HTTPException, status
from app.models import FeedbackInput
from app.chatbot import SelfLearningChatbot

router = APIRouter()

chatbot = SelfLearningChatbot()


@router.post("/")
async def update_feedback(feedback: FeedbackInput):
    """Update feedback for a specific chat interaction."""
    try:
        chatbot.update_feedback(feedback.chat_id, feedback.feedback_score)
        return {"status": "success", "message": "Feedback updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
