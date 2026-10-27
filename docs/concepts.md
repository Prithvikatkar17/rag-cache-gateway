## 1. Embeddings
**In one line:** An embedding turns a sentence into a list of numbers (a vector) so that sentences with similar meaning end up close together, and unrelated sentences end up far apart.

**How it works:** An embedding model takes a sentence and outputs a fixed-length vector (384 numbers for all-MiniLM-L6-v2), regardless of sentence length. It was trained to pull related sentences close together and push unrelated ones apart. Each sentence becomes a point in 384-dimensional space. To compare two sentences I use cosine similarity: dot(a, b) / (|a| * |b|). The dot product multiplies the vectors position by position and sums; dividing by both lengths removes size so only direction matters. 1.0 means the same direction, 0 means unrelated. No single number means anything alone; the meaning is spread across all 384.

**Where it's used in this project:** (planned) embedding the question for the semantic cache lookup, and embedding the question to find similar document chunks.

**What I tried:** I compared sentence pairs with cosine similarity:
- SSH paraphrases: 0.48
- SSH vs pizza (unrelated): 0.13
- "reset my password" vs "steps to change my password": 0.823
- "reset my password" vs "reset my router": 0.53
- "I love this phone" vs "I hate this phone": 0.813

**What surprised me / what I got wrong at first:** I expected love/hate to score low because the meanings are opposite, but it scored 0.813, almost the same as a true paraphrase (0.823). Embeddings mostly capture what a sentence is about, and "love" and "hate" appear in the same kinds of sentences, so the model puts them close together. I also expected paraphrases to score high, but the SSH pair scored only 0.48. Any similarity cutoff I choose has to come from my own test pairs, not a number I copied.

**Interview question I can now answer:** *"What's the risk of a semantic cache, and how would you reduce it?"*
The main risk is a false hit: opposite or different questions can have very similar embeddings, so the cache returns a confidently wrong answer. A threshold alone can't fix it, because opposite pairs (0.813) can score as high as real paraphrases (0.823), while a high threshold misses real paraphrases (0.48). To reduce the risk I would use a conservative threshold, add a second check such as a cross-encoder that reads both sentences together, and avoid caching high-stakes questions.


## 2. Cosine similarity

**In one line:** Cosine similarity measures how closely two vectors point in the same direction. It returns 1 for the same direction, 0 for unrelated, and -1 for opposite.

**How it works:**
- Formula: cosine(a, b) = dot(a, b) / (|a| * |b|)
- Dot product: multiply the two vectors position by position, then add up the results. Example: [1, 2, 3] . [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
- |a| is the vector's length: square each number, add them, take the square root
- Dividing by both lengths removes the effect of size, so only direction counts. [3, 0] and [5, 0] have different lengths but point the same way, so they score exactly 1.0
- Why direction and not length: embedding models put the meaning in the direction. Two sentences on the same topic can give vectors of different lengths, and cosine ignores that
- Normalized vectors: if every vector has length 1, the division does nothing and cosine is just the dot product. I checked this, and `np.linalg.norm` of the model's vector printed 1.0, so the model's vectors are already normalized
- `stored_vecs @ q` takes the dot product of every stored row with q in one step, which is why it gave me valid cosine scores (stored_vecs is (5, 384), q is (384,), and the result is 5 scores)

**Similarity vs distance:**
- similarity: 1 means identical
- distance = 1 - similarity: 0 means identical
- Redis returns distance, not similarity. Comparing one as if it were the other flips the logic, and the best match would look like the worst. Example from my run: similarity 0.713 means distance 0.287

**Brute-force search:**
- Compare the query against every stored vector and take the highest score
- It is correct, but the cost grows with the number of stored vectors: 1 million vectors means 1 million comparisons per query
- That's why vector indexes (HNSW) exist, which is the next concept
- It also always returns a "best" match, even a bad one (0.396 for the router sentence). A cutoff is needed to say "nothing here is close enough", and it has to come from my own data

**Where it's used in this project:** (planned) comparing a new question with saved questions for the cache, and with document chunks for retrieval. Redis will compute it for me, using distance.

**What I tried:**
- Part A (by hand): [3,0] vs [5,0] = 1.0, [1,0] vs [0,1] = 0.0, [1,0] vs [-1,0] = -1.0
- Part B: norm of an embedding = 1.0
- Part C: query "I forgot my password, what should I do?" against 5 stored sentences:
  - 0.713 How do I change my password?
  - 0.691 Steps to recover a forgotten login
  - 0.396 How do I reset my router?
  - 0.228 How can I SSH into a server?
  - 0.098 What is the best pizza recipe?

**What surprised me / what I got wrong at first:**
- "Steps to recover a forgotten login" shares almost no words with my query but scored 0.691, almost the same as the top match. A keyword search would have missed it
- "Reset my router" shares the word "reset" but scored only 0.396, so embeddings go by meaning, not shared words
- (add your own: I mistyped/misread the Part A result at first, and the first value should have been 1.0)

**Interview question I can now answer:** *"Why cosine similarity instead of Euclidean distance for embeddings, and why does it get slow at scale?"*
Embeddings carry meaning in their direction, so cosine ignores differences in vector length. With normalized vectors it reduces to a single dot product, which is cheap. But comparing a query against every stored vector is O(n) per query, so large collections need an approximate index like HNSW.


## 3. Vector search and HNSW

**In one line:** Vector search finds the stored vectors closest to a query vector. HNSW is an index that does this approximately but much faster than checking every vector.

**How it works:**
- Brute force compares the query with every stored vector. It is exact, but the cost grows with the number of vectors (N vectors = N comparisons per query)
- HNSW (Hierarchical Navigable Small World) stores the vectors as a graph, where each vector is linked to a few nearby vectors
- Search is a greedy walk: start somewhere, look at the neighbors, move to the one closest to the query, and repeat until no neighbor is closer
- A single flat graph needs too many small hops, so HNSW uses layers, like a road network: the top layer has few nodes with long links (highways), the lower layers have more nodes with shorter links (local streets), and the bottom layer has every node. A search starts at the top, finds the closest node, drops down a layer from there, and repeats, so each layer gets it closer and the final walk is short. The number of steps grows roughly with log(N) instead of N
- It is approximate: the greedy walk can stop at a good match that is not the best one

**Settings:**
- M: links per node (build time). More links means a better graph but more memory
- efConstruction: candidates considered when linking a new node (build time). Higher gives a better graph and a slower build
- efSearch: candidates tracked during a query (query time). Higher gives better accuracy and slower queries
- Cost of HNSW: the graph lives in RAM, and building it makes inserts slower than a plain list append

**Where it's used in this project:** (planned) Redis will keep HNSW indexes for the document vectors and for the cached questions. I'll set M, efConstruction and efSearch when I create the index.

**What I tried:** FAISS HNSW index on 50,000 random vectors of length 384, 100 queries. Brute force took 2.22 ms per query. "Recall" is the fraction of queries where HNSW found the true best match.

| efSearch | ms per query | recall |
|---|---|---|
| 1 | 0.036 | 1% |
| 4 | 0.083 | 3% |
| 16 | 0.207 | 9% |
| 64 | 0.630 | 17% |
| 256 | 1.457 | 41% |

Then I ran it on clustered vectors ():
(paste your own table here)

**What surprised me / what I got wrong at first:**
- With random vectors HNSW was bad: even at efSearch=256 it found the true best only 41% of the time and was only about 1.5x faster than brute force
- The reason: random vectors in 384 dimensions are almost all perpendicular to each other (the best match scored only about 0.18), so every vector is roughly equally far from the query and the greedy walk has no downhill direction to follow
- Real text embeddings have structure (similar topics cluster together), and with clustered data the recall was much higher at small efSearch. HNSW depends on structure in the data
- Brute force with NumPy was faster than I expected (2 ms for 50,000 vectors), so HNSW matters most at large scale. For a small collection, brute force is competitive

**Interview question I can now answer:** *"Why use an approximate index like HNSW instead of exact search, and what do you give up?"*
Exact search compares the query with every vector, so cost grows linearly with data size. HNSW walks a layered graph and touches only a small fraction of the vectors, so cost grows roughly with log(N). What you give up is a guarantee: it can miss the true nearest neighbor. efSearch controls the balance, since a higher value is more accurate and slower. In a semantic cache a miss means one extra LLM call, which is safer than a false hit that returns a wrong answer.



## 4. Python patterns used in the project

- **Class and `self`:** A class bundles data and the functions that use it. `__init__` sets up the starting data, and `self` is the object itself, which is how its functions reach its data (`self.store`, `self.hits`). The project creates one shared object (`redis_manager = RedisManager()`) so every file uses the same Redis connection.
- **try/except:** Run code that might fail, and if it raises an error, jump to `except` instead of crashing. The project uses it for "create if missing": try to read the index, and if that fails because it doesn't exist, create it.
- **Exact cache limit:** I built a mini exact cache that turns each prompt into a SHA-256 fingerprint and stores the answer under it. "Hello" missed even though "hello" was saved, because a one-letter change gives a completely different fingerprint. The exact cache only catches character-for-character repeats, which is why Layer 2 (semantic) is needed.


## 5. Async Python

- **`await`:** Runs an async function and pauses this function until it finishes, so Python can run other work while it waits. If I forget `await`, I get a coroutine object (an unstarted to-do item) instead of the real result. With Redis, `client.get(key)` without `await` returns that object and not my data.
- **`gather`:** Starts several async calls together and waits for all of them. In my test, three fake LLM calls took 1.25 s together instead of about 3.6 s one after another, because all three waited at the same time. This is why a server can handle many requests that are waiting on Redis or an LLM.
- **Blocking:** `time.sleep` freezes the whole program, not just one function, so nothing else gets a turn. Only waits that use `await` (like `asyncio.sleep` or an async Redis call) let other work run. `model.encode` is regular blocking code, so while it runs the server can't serve anyone else.
- **Cache hit vs miss, in miniature:** first `answer("hello")` took 1.24 s (cache lookup + LLM + save), the second took 0.02 s (lookup only).
- **Limit I noticed:** if identical requests arrive at the same moment, all of them miss the cache and all call the LLM (a cache stampede). Not fixing it in this project, but I can explain it.



## 6. FastAPI

- **Three pieces:** FastAPI defines the routes (`@app.post("/query")`), uvicorn is the server that runs the app, and Pydantic defines the shape of the data (`class QueryRequest(BaseModel)`). FastAPI uses my type hints to read and check the request body.
- **Validation:** If the request body doesn't match the model (missing field, wrong type), FastAPI returns 422 and my function never runs. `response_model` checks what I return too. A bad return value gives a 500, which is what happened when I returned a coroutine and a tuple.
- **Reading errors:** In a long traceback, ignore the `site-packages` lines and read the last few lines. Mine said exactly which fields had which wrong types. Status codes in the log: 200 means success, 404 means no such route (what I got for `/`), 422 means bad input, 500 means my code failed.
- **lifespan:** Code before `yield` runs once at startup, and code after runs once at shutdown. The project will connect to Redis and create the indexes at startup, and close the connection at shutdown.
- **My test:** miss on the first call (about 1200 ms), hit on repeats (about 0.01 ms), and "Hello" missed again. The `store` dictionary lives in server memory, so it resets on every `--reload`. Redis will fix that.



## 7. Redis and Docker

- **Image vs container:** An image is the packaged, read-only program (`redis/redis-stack`). A container is a running copy of it. I can delete and recreate containers from the same image.
- **Why `volumes`:** Redis saves its data inside the container. The compose file mounts a Docker volume at `/data`, so the data survives when the container is removed.
- **Ports:** `6379` is Redis itself (my Python code connects here) and `8001` is the RedisInsight web page. The format `6379:6379` connects a port on my PC to a port in the container.
- **Why Redis Stack:** Plain Redis has no text search or vector search. Redis Stack adds both, which I need for BM25 and HNSW.