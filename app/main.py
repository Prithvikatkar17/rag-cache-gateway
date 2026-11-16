from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.db.redis_client import redis_manager
from app.core.embedding import embedding_model  # <--- NEW IMPORT

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_model.load()          # <--- NEW: Load the 300MB model into memory BEFORE traffic hits
    await redis_manager.connect()
    yield
    await redis_manager.close()

app = FastAPI(title="RAG Cache Gateway", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"ok": True}