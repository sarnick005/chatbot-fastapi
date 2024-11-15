from fastapi import APIRouter, HTTPException, status
from app.models import ChatInput, ChatResponse
from app.chatbot import SelfLearningChatbot

router = APIRouter()

chatbot = SelfLearningChatbot()


@router.post("/", response_model=ChatResponse)
async def chat(chat_input: ChatInput):
    try:
        predicted_tag, confidence = chatbot.predict_intent(chat_input.user_input)
        if not predicted_tag:
            response = "I'm not sure about that. Can you teach me?"
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
