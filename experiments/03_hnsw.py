import numpy as np
import time
import faiss

rng = np.random.default_rng(0)
N , D = 50_000 , 384

def make(n):
    x = rng.standard_normal((n,D)).astype("float32")
    x /= np.linalg.norm(x ,axis=1 ,keepdims=True)
    return x

data = make(N)
queries = make(100)

t = time.perf_counter()
exact = [int(np.argmax(data @ q)) for q in queries]
brute_ms = (time.perf_counter() - t) / len(queries) * 1000
print(f"brute force: {brute_ms:.2f} ms per query")


index = faiss.IndexHNSWFlat(D, 16, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 100
t = time.perf_counter()
index.add(data)
print(f"build time: {time.perf_counter() - t:.1f} s")


for ef in [1, 4, 16, 64, 256]:
    index.hnsw.efSearch = ef
    t = time.perf_counter()
    found = [int(index.search(q[None, :], 1)[1][0][0]) for q in queries]
    ms = (time.perf_counter() - t) / len(queries) * 1000
    recall = np.mean([f == e for f, e in zip(found, exact)])
    print(f"efSearch={ef:<4} {ms:.3f} ms per query | found the true best: {recall:.0%}")

scores, positions = index.search(queries[:1], 3)
print(scores)
print(positions)
