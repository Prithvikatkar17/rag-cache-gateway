import asyncio
import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

INDEX = "idx:semcache"
PREFIX = "semcache:"
THRESHOLD = 0.90                    # similarity (your Part 1 choice)
MAX_DISTANCE = 1 - THRESHOLD        # the same rule, in Redis's units

model = SentenceTransformer("all-MiniLM-L6-v2")

def to_bytes(text: str) -> bytes:
    # Converts the AI's math array into raw bytes so Redis can store it
    return model.encode(text).astype(np.float32).tobytes()

def txt(x):
    # Cleans up the data coming back from Redis (turns bytes back into strings)
    return x.decode() if isinstance(x, bytes) else x

async def create_index(client):
    try:
        await client.ft(INDEX).dropindex(delete_documents=True)
    except Exception:
        pass                         # no index yet, nothing to drop
        
    schema = (
        TextField("prompt"),
        VectorField("embedding", "HNSW",
                    {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
    )
    await client.ft(INDEX).create_index(
        schema,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
    )

async def add(client, n: int, prompt: str, answer: str):
    # TODO 1: save a hash under the key f"{PREFIX}{n}" with three fields
    key = f"{PREFIX}{n}"
    await client.hset(key, mapping={
        "prompt": prompt,
        "answer": answer,
        "embedding": to_bytes(prompt)
    })

async def lookup(client, prompt: str):
    # KNN 1 means "find the 1 closest neighbor"
    q = (Query("*=>[KNN 1 @embedding $vec AS distance]")
         .sort_by("distance")
         .return_fields("prompt", "answer", "distance")
         .dialect(2))
         
    res = await client.ft(INDEX).search(q, query_params={"vec": to_bytes(prompt)})
    if not res.docs:
        return None
        
    doc = res.docs[0]
    distance = float(txt(doc.distance))
    
    # TODO 2: Calculate similarity and return the appropriate dictionary
    similarity = 1 - distance
    if distance <= MAX_DISTANCE:
        return {
            "hit": True, 
            "similarity": round(similarity, 3),
            "matched": txt(doc.prompt), 
            "answer": txt(doc.answer)
        }
    else:
        return {
            "hit": False, 
            "similarity": round(similarity, 3),
            "matched": txt(doc.prompt)
        }

async def main():
    # Make sure you have a local Redis server running with the RediSearch module installed!
    client = redis.Redis(host="localhost", port=6379, decode_responses=False)
    await create_index(client)
    
    await add(client, 1, "How do I reset my password?", "Click Forgot password on the login page.")
    await add(client, 2, "Is my data safe?", "Yes, it is encrypted at rest and in transit.")

    tests = [
        "How do I reset my password?",
        "how do i reset my password",
        "How can I reset my password?",
        "How do I reset my passwrod?",
        "What are the steps to change my password?",
        "How do I reset my router?",
        "How is my data protected?",
    ]
    for t in tests:
        print(t, "->", await lookup(client, t))
        
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())