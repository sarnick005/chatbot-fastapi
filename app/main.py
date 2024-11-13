from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import chat, intent
import os
from dotenv import load_dotenv


load_dotenv()

cors_origins = os.getenv("CORS_ORIGINS", "").split(",")

app = FastAPI(title="Self Learning Chatbot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Include routers
app.include_router(chat.router, tags=["chat"])
app.include_router(intent.router, tags=["intents"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
