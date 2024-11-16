# app/routes/metrics.py
from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
from datetime import datetime
from app.models import ModelMetrics, ChatHistoryResponse
from app.chatbot import SelfLearningChatbot
from app.database import db

router = APIRouter()
chatbot = SelfLearningChatbot()


@router.get("/model", response_model=ModelMetrics)
async def get_model_metrics():
    """Get current model performance metrics"""
    try:
        return chatbot.get_performance_metrics()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/chat-history", response_model=List[ChatHistoryResponse])
async def get_chat_history(
    user_id: str,
    limit: int = Query(20, ge=1, le=100),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
):
    """Get chat history for a specific user"""
    try:
        query = {"user_id": user_id}
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date

        return list(
            db.chat_history.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.get("/model-versions")
async def get_model_versions(limit: int = Query(10, ge=1, le=50)):
    """Get history of model versions and their performance"""
    try:
        return list(
            db.model_versions.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/retrain")
async def force_retrain():
    """Force model retraining"""
    try:
        chatbot._retrain_model()
        return {
            "status": "success",
            "new_accuracy": chatbot.accuracy,
            "training_samples": len(chatbot.X),
            "version": chatbot._get_model_version(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
