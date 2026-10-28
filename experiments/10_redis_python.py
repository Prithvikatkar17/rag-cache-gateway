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


asyncio.run(main())