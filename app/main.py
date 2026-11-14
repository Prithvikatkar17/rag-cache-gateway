from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.redis_client import redis_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs the moment the server turns on
    await redis_manager.connect()
    yield  # The server pauses here and handles user traffic
    # This runs the moment you turn the server off
    await redis_manager.close()

app = FastAPI(title="RAG Cache Gateway", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"ok": True}