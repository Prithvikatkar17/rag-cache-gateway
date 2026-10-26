# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/health")
# async def health():
#     return {"ok": True}


from fastapi import FastAPI
from pydantic import BaseModel

# app = FastAPI()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("STARTING UP: connect to Redis here")
    yield
    print("SHUTTING DOWN: close Redis here")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    prompt: str

class QueryResponse(BaseModel):
    status: str
    response_text: str

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    return {"status": "OK", "response_text": f"You asked: {req.prompt}"}