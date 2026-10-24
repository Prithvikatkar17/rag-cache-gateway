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





async def get_answer():
    return "hi"  # Return a value

async def main1():
    x = get_answer()       # No await: returns a coroutine object
    print(x)

    y = await get_answer() # Await: gets the actual result
    print(y)               # Output: hi

asyncio.run(main1())       # Run the main coroutine


import time

async def slow(name):
    time.sleep(1)           # Blocks the entire event loop
    print(name, "done")

async def main2():
    start = time.perf_counter()

    # Tasks run one after another because sleep blocks
    await asyncio.gather(slow("A"), slow("B"))

    print(f"{time.perf_counter() - start:.1f}s")  # Total time ≈ 2s

asyncio.run(main2())         # Start the event loop
