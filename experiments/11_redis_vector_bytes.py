import numpy as np
import asyncio
import redis.asyncio as redis

async def main():
    client = redis.Redis(host="localhost" ,port=6379 ,decode_responses=False)

    vec = np.array([0.1 , 0.2 ,0.3] , dtype=np.float32)
    raw = vec.tobytes()

    print( raw ,len(raw))

    await client.hset("vec:1", mapping={"label": "demo", "embedding": raw})
    data = await client.hgetall("vec:1")
    print(data)

    label = data[b"label"].decode() 
    print(label)
    embedding_bytes = data[b"embedding"]

    embedding = np.frombuffer(embedding_bytes, dtype=np.float32) 
    print(embedding) 
    await client.aclose()

asyncio.run(main())
