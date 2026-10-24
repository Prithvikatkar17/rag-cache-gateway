
import asyncio
import time

store = {}

async def fake_redis_get(key):
    await asyncio.sleep(0.01)
    return store.get(key)  # Get cached answer

async def fake_redis_set(key, value):
    await asyncio.sleep(0.01)
    store[key] = value  # Save answer in cache

async def fake_llm(prompt):
    await asyncio.sleep(1.2)  # Simulate LLM response time
    return f"Answer to: {prompt}"


async def answer(prompt):
    var = await fake_redis_get(prompt)  # Check cache first

    if var:
        return var  # Return cached answer

    result = await fake_llm(prompt)  # Generate new answer
    await fake_redis_set(prompt, result)  # Cache the answer
    return result


async def main():
    # Test cache: first call is slow, second is fast
    for _ in range(2):
        start = time.perf_counter()
        result = await answer("hello")
        print(result, f"{time.perf_counter() - start:.2f}s")

    # Run three independent prompts concurrently
    start = time.perf_counter()
    results = await asyncio.gather(
        answer("a"),
        answer("b"),
        answer("c")
    )
    print(results, f"Together: {time.perf_counter() - start:.2f}s")


asyncio.run(main())
