from fastapi import APIRouter, HTTPException, status
from app.schemas.chat import ChatInput, ChatResponse, FeedbackInput, HealthResponse
from app.core.chatbot import SelfLearningChatbot
from app.config.database import db_config

router = APIRouter()
chatbot = SelfLearningChatbot(db_config=db_config)


@router.post("/chat", response_model=ChatResponse)
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


@router.post("/feedback")
async def feedback_endpoint(feedback: FeedbackInput):
    """Handle feedback updates."""
    try:
        chatbot.update_feedback(feedback.chat_id, feedback.feedback_score)
        return {"status": "success", "message": "Feedback updated successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="healthy", model_accuracy=chatbot.accuracy)
