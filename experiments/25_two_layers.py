import asyncio
import hashlib
import time
import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

INDEX = "idx:semcache"
PREFIX = "semcache:"
EXACT_PREFIX = "exact:"
THRESHOLD = 0.90
MAX_DISTANCE = 1 - THRESHOLD
TTL = 300                          # seconds, for both layers

model = SentenceTransformer("all-MiniLM-L6-v2")

def to_bytes(text):
    return model.encode(text).astype(np.float32).tobytes()

def txt(x):
    return x.decode() if isinstance(x, bytes) else x

def sha(text):
    # Turns a long prompt into a short, unique string (hash)
    return hashlib.sha256(text.encode()).hexdigest()

async def create_index(client):
    try:
        await client.ft(INDEX).dropindex(delete_documents=True)
    except Exception:
        pass
    schema = (
        TextField("prompt"),
        VectorField("embedding", "HNSW",
                    {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
    )
    await client.ft(INDEX).create_index(
        schema,
        definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
    )

async def full_pipeline(prompt):
    # Simulates an expensive, slow API call to an LLM like OpenAI or Gemini
    await asyncio.sleep(1.2)       
    return f"[Generated answer] for: {prompt}"

async def semantic_lookup(client, prompt):
    q = (Query("*=>[KNN 1 @embedding $vec AS distance]")
         .sort_by("distance").return_fields("prompt", "answer", "distance").dialect(2))
    res = await client.ft(INDEX).search(q, query_params={"vec": to_bytes(prompt)})
    if not res.docs:
        return None
    doc = res.docs[0]
    distance = float(txt(doc.distance))
    similarity = 1 - distance
    
    if distance <= MAX_DISTANCE:
        return {"hit": True, "similarity": round(similarity, 3), "answer": txt(doc.answer)}
    return {"hit": False, "similarity": round(similarity, 3)}

async def save_semantic(client, prompt, answer):
    key = PREFIX + sha(prompt)[:16]
    await client.hset(key, mapping={"prompt": prompt, "answer": answer,
                                    "embedding": to_bytes(prompt)})
    await client.expire(key, TTL)  # Automatically delete after 300 seconds

async def handle_query(client, prompt):
    start = time.perf_counter()
    exact_key = EXACT_PREFIX + sha(prompt)
    status, answer, sim = None, None, None

    # TODO 1 (Layer 1): Check the exact cache first
    cached = await client.get(exact_key)
    
    if cached is not None:
        status = "EXACT_CACHE_HIT"
        answer = txt(cached)
    else:
        # TODO 2 (Layer 2): If not an exact match, check for semantic similarity
        hit = await semantic_lookup(client, prompt)
        
        if hit is not None and hit["hit"]:
            status = "SEMANTIC_CACHE_HIT"
            answer = hit["answer"]
            sim = hit["similarity"]
        else:
            # TODO 3 (Miss): Both caches failed. Call the AI.
            sim = hit["similarity"] if hit else None
            answer = await full_pipeline(prompt)
            
            # Save it in Layer 1 (Exact match)
            await client.set(exact_key, answer, ex=TTL)
            
            # Save it in Layer 2 (Semantic match)
            await save_semantic(client, prompt, answer)
            
            status = "CACHE_MISS"

    # Calculate how many milliseconds this whole process took
    ms = (time.perf_counter() - start) * 1000
    return {"status": status, "ms": round(ms, 1), "nearest_similarity": sim, "answer": answer}

async def main():
    client = redis.Redis(host="localhost", port=6379, decode_responses=False)
    await create_index(client)                         
    
    # Clear out old test data
    async for key in client.scan_iter(f"{EXACT_PREFIX}*"):
        await client.delete(key)                       

    prompts = [
        "How do I reset my password?",                   # 1. Miss (calls AI)
        "How do I reset my password?",                   # 2. Exact Hit
        "how do i reset my password",                    # 3. Semantic Hit (case difference)
        "What are the steps to change my password?",     # 4. Semantic Hit (meaning match)
        "How do I reset my router?",                     # 5. Miss (different meaning)
        "how do i reset my password",                    # 6. Exact Hit (now cached exactly)
    ]
    
    for n, p in enumerate(prompts, start=1):
        r = await handle_query(client, p)
        print(f"{n}. {r['status']:<20} {r['ms']:>8} ms  nearest={r['nearest_similarity']}  | {p}")
        
    await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())