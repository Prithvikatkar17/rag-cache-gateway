import asyncio
import time
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
store = {}

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    status: str
    execution_time_ms: float
    response_text: str

async def fake_llm(prompt: str) -> str:
    await asyncio.sleep(1.2)
    return f"Answer to: {prompt}"

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    start = time.perf_counter()

    if(req.prompt in store):
        status = "CACHE_HIT"
        text = store[req.prompt]
    else:
        status = "CACHE_MISS"
        text = await fake_llm(req.prompt)
        store[req.prompt] = text

    # TODO 1: check whether req.prompt is already in store
    #         if yes: set status to "CACHE_HIT" and text to the stored answer
    # TODO 2: if not: set status to "CACHE_MISS", get the text from
    #         fake_llm (await it), and save it in store

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "status" : status,
        "execution_time_ms" : round(elapsed_ms ,2),
        "response_text" : text

    }
    # TODO 3: return a dict with the three fields of QueryResponse
    #         (round elapsed_ms to 2 decimals)