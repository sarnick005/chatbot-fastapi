# app/routes/health.py
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from app.database import db

router = APIRouter()


@router.get("/")
async def health_check():
    """Check system health status"""
    try:
        # Check database connection
        db_status = "healthy" if db.command("ping")["ok"] else "unhealthy"

        return {
            "status": "healthy",
            "database": db_status,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }
