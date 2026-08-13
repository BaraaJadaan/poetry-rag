import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import generate_response_stream

load_dotenv()

app = FastAPI()

allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

class ChatRequest(BaseModel):
    messages: list
    use_openrouter: bool

@app.get("/config")
def get_config():
    is_cloud = os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true"
    return {"cloud_deployment": is_cloud}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    is_cloud = os.getenv("CLOUD_DEPLOYMENT", "false").lower() == "true"
    if is_cloud:
        req.use_openrouter = True

    # Strip any accidental surrounding quotes from the .env value
    api_key = os.getenv("opentouter_api", "").strip().strip("'\"")
    return StreamingResponse(
        generate_response_stream(req.messages, req.use_openrouter, api_key),
        media_type="text/event-stream"
    )

# Ensure the frontend directory exists before mounting
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
