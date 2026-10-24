import asyncio
import time

async def fetch(name, seconds):
    print(f"{name}: start")
    await asyncio.sleep(seconds)      # pretend to wait for Redis or an LLM
    print(f"{name}: done")
    return name

async def main():
    start = time.perf_counter()
    a = await fetch("A", 1)
    b = await fetch("B", 1)
    print(f"one after another: {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    results = await asyncio.gather(fetch("C", 1), fetch("D", 1))
    print(results, f"together: {time.perf_counter() - start:.1f}s")

asyncio.run(main())