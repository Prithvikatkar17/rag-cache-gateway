import asyncio
import time

from app.core.embedding import embedding_model


async def ticker(stop, ticks):
    while not stop.is_set():
        await asyncio.sleep(0.05)
        ticks.append(1)


async def measure(label, work):
    stop, ticks = asyncio.Event(), []
    task = asyncio.create_task(ticker(stop, ticks))
    start = time.perf_counter()
    await work()
    elapsed = time.perf_counter() - start
    stop.set()
    await task
    print(f"{label}: {elapsed:.2f} s, ticker ran {len(ticks)} times")


async def blocking():
    for _ in range(20):
        embedding_model.embed("How do I reset my password?")


async def threaded():
    for _ in range(20):
        await embedding_model.aembed("How do I reset my password?")


async def main():
    embedding_model.load()
    vec = embedding_model.embed("How do I reset my password?")
    print(vec.shape, vec.dtype, len(embedding_model.to_bytes(vec)), "bytes")
    await measure("blocking (embed)", blocking)
    await measure("threaded (aembed)", threaded)


asyncio.run(main())