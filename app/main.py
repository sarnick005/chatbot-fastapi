from fastapi import FastAPI
from app.routes import chat, feedback, intents, health
from app.database import initialize_database
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Self-Learning Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-express-server.com",
    ],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

# Initialize database and load data on startup
@app.on_event("startup")
def on_startup():
    initialize_database()


# Include routes
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
app.include_router(intents.router, prefix="/intents", tags=["Intents"])
app.include_router(health.router, prefix="/health", tags=["Health"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
