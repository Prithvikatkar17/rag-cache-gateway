
import asyncio
import hashlib
import time
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI
from pydantic import BaseModel

client: redis.Redis | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = redis.Redis(host="localhost", port=6379, decode_responses=False)
    await client.ping()
    print("Connected to Redis")
    yield
    await client.aclose()
    print("Redis connection closed")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    status: str
    execution_time_ms: float
    response_text: str

async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(1.2)
    return f"Answer to: {prompt}"

def make_key(prompt: str) -> str:
    return "exact:" + hashlib.sha256(prompt.encode()).hexdigest()

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.perf_counter()
    key = make_key(req.prompt)

    # Check Redis cache
    cached_result = await client.get(key)

    if cached_result is not None:
        status = "CACHE_HIT"
        text = cached_result.decode()
    else:
        status = "CACHE_MISS"
        text = await fake_llm(req.prompt)
        await client.set(key, text, ex=300)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "status": status,
        "execution_time_ms": round(elapsed_ms, 3),
        "response_text": text,
    }
