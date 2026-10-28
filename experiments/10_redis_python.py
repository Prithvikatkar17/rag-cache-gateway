import asyncio
import redis.asyncio as redis

async def main():
    client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=False
    )

    print(await client.ping())

    await client.set("greeting" , "hello")

    value = await client.get("greeting")

    print(value ,type(value))
    print(value.decode)

    print(await client.get("missing"))

    await client.hset(
        "cache:2",
        mapping={
            "prompt": "reset password?",
            "answer": "Go to settings."
        }
    )
    data = await client.hgetall("cache:2")
    print(data)

    await client.set("x" , "y" ,ex=30)
    print(await client.ttl("x"))


asyncio.run(main())