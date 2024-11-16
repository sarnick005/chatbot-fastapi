# app/routes/feedback.py
from fastapi import APIRouter, HTTPException, status
from app.models import FeedbackInput
from app.chatbot import SelfLearningChatbot

router = APIRouter()
chatbot = SelfLearningChatbot()


@router.post("/")
async def feedback(feedback_input: FeedbackInput):
    """Handle feedback for chat interactions"""
    try:
        success = chatbot.update_feedback(feedback_input.chat_id, feedback_input.score)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat interaction not found",
            )
        return {"status": "success", "message": "Feedback recorded successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
