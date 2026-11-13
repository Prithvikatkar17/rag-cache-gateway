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


### Redis commands (typed by hand in RedisInsight CLI)

- **Strings:** `SET key value` stores a value, `GET key` reads it, and a missing key returns `(nil)`, which is Redis's version of Python's `None` and means a cache miss. Setting the same key again overwrites the old value. Keys are case-sensitive (`GET Greeting` missed when I had set `greeting`), which is the same reason "Hello" missed in my exact cache. `EX 30` makes a key delete itself after 30 seconds, and `TTL key` shows the seconds left (`-1` means no expiry, `-2` means the key is gone). Expiry matters for a cache because stored answers shouldn't live forever if the documents change.
- **Hashes:** `HSET key field value ...` stores several named fields under one key, and `HGET`/`HGETALL` read one or all of them. The semantic cache needs a hash and not a plain string, because each entry has several parts (prompt, answer, and later the embedding). Key names like `cache:1` use a prefix, and a search index can be told to watch every key starting with `cache:`, which is how it finds my data.
- **Gotcha:** `docker compose up -d` failed with "failed to connect to the docker API" when Docker Desktop wasn't open. The engine only runs while Docker Desktop is running, so I open it and wait for "Engine running" first. My data survived because it's in the volume.
- **My test:** after `SET q:1 "first" EX 100` and then `SET q:1 "second"`, `TTL q:1` returned -1 (so a plain `SET` removes the expiry).

- **Python and bytes:** I use `redis.asyncio`, so every Redis call needs `await` (without it I'd get a coroutine, not data). Redis returns bytes (`b'hello'`), and I turn them into text with `.decode()`. A hash comes back as a dict with bytes keys and values (`data[b"prompt"]`). Forgetting the `()` in `.decode()` prints the method itself instead of the text.
- **Why `decode_responses=False`:** a vector is stored as raw bytes (float32, 4 bytes per number, so 384 numbers = 1536 bytes). Automatic decoding would try to read those bytes as text and break them. I decode the text fields myself. Round trip: `vec.tobytes()` to store, `np.frombuffer(raw, dtype=np.float32)` to read back (3 floats = 12 bytes).

- **API with Redis:** `lifespan` creates the Redis client at startup (before `yield`) and closes it at shutdown (after `yield`). The cache key is `"exact:" + sha256(prompt)`, the answer is stored with `ex=300` (a plain SET would remove the expiry), and a hit returns bytes, so I `.decode()` before responding.
- **Why it survived a restart:** the old dictionary lived inside the Python process and disappeared when the server stopped. Redis runs in its own container, so the answer was still there. My hit after the restart took about 4 ms (a network round trip) versus about 1200 ms for a miss.


## 8. BM25 and keyword search

- **Keyword search:** split text into lowercase words and count how often the query words appear in each document.
- **Where plain counting fails:** (1) repetition beats relevance: for "reset password", a document repeating "password" three times beat the documents that actually explain resetting a password, since both words appeared only once in them; (2) common words dominate: for "the router", 7 of the top score of 10 came from "the". BM25 fixes these with diminishing returns for repeats, rarity weighting, and a length adjustment.

- **IDF (rarity):** ln(1 + (N - n + 0.5)/(n + 0.5)), where n is how many documents contain the word. "router" (in 1 of 4 docs) got 1.204 and "password" (in 3 of 4) got 0.357, so rare words count more. This fixed "the" dominating. With only 4 documents it can't show the full effect.
- **TF part (diminishing returns):** tf × (k1+1) / (tf + k1 × length_factor). With k1 = 1.5 one mention is worth 1.0, three mentions 1.67, ten mentions only 2.17, with a hard cap of k1 + 1 = 2.5. This fixed repetition beating relevance: for "reset password", the documents with both words (0.870, 0.839) now beat the ones repeating one word.
- **Length factor:** 1 - b + b × len(doc)/avgdl. A longer-than-average document is penalized (doc 1, with 30 words against an average of 15, has a factor of 1.75). With b = 0 the penalty disappears, and with k1 = 0 repeats stop counting.

- **BM25 vs embeddings (my test, 6 docs, 3 queries):** For `ERR_CONN_RESET` both found the right document (BM25 2.63 vs 0.65, embeddings 0.80 vs 0.37). For `recovering lost credentials` and `slow internet keeps disconnecting`, BM25 scored 0 for every document because no words were shared ("recovering" vs "recover", "disconnecting" vs "dropped connections"), while embeddings found the right document (0.76 and 0.49). I expected embeddings to struggle with the rare token, but they didn't.
- **Takeaway:** BM25 gives a clear "no match" (score 0) and is precise on exact strings, but it cannot handle paraphrases. Embeddings handle meaning but always return a best guess, even a bad one. The cases where embeddings are weak are near-identical identifiers (error codes, versions), which my small test didn't reach.

## 9. RRF (Reciprocal Rank Fusion)

- **Why scores can't be added directly:** BM25 scores and cosine scores are on different scales (BM25 was around 12, cosine under 1), and BM25 scores aren't comparable even between libraries. In my test, adding raw scores gave exactly the BM25 ranking, so the embeddings had no real vote.
- **Min-max normalization is fragile:** scaling each list to 0..1 first helped, but (1) one outlier (BM25 for A at 40) made B's fused score fall from 1.20 to 0.54 though B's own scores didn't change, and (2) a list where every score is equal (BM25 found nothing) gives a division by zero. RRF avoids both by using only ranks, not scores.


- **Formula:** RRF score(doc) = sum over each list that contains it of 1 / (k + rank), with rank starting at 1 and k = 60. Missing from a list means no contribution from it. Example: C was 2nd in BM25 and 1st in embeddings, so 1/62 + 1/61 = 0.03252.
- **Why it works:** it uses only ranks, so score scales don't matter, outliers can't distort it, and an empty list (BM25 found nothing) just adds nothing. Documents that appear in both lists beat documents in one: B (3rd and 4th, in both) scored 0.0315 against D (2nd in only the embedding list) at 0.0161.
- **Trade-off:** RRF throws away how large the scores were. D had a strong embedding score (0.78) but still ranked 4th. Also, `enumerate(..., start=1)` matters: starting at 0 would make first place rank 0.

- **What `k` does:** it sets how much top positions are worth versus lower ones. With the same lists (bm = A..E, em = F, G, H, D, E): at k = 1, A (1st in one list) scored 0.5 and beat D (4th in both) at 0.4; at k = 60, D scored 0.031 and beat A at 0.016; at k = 1000 the scores only reflect how many lists contain the document. Default 60 favors agreement between the lists. I won't tune it before I have real data.
- **Empty list:** BM25 only returns documents with a score above 0, otherwise unmatched documents would earn rank points. When BM25 found nothing ("recovering lost credentials"), the fused list was exactly the embedding ranking (scores 1/61, 1/62, 1/63), so RRF falls back to the other method. On my 6 documents RRF never changed a top answer, because the lists agreed or one was empty. Its value shows when they disagree.

## 10. RAG

- **Three steps:** retrieve the most relevant chunks (embeddings + BM25 + RRF), augment the prompt by pasting those chunks in with the question, then generate with the LLM. The LLM doesn't change; it's only given the right reading material. In the project, this is the expensive path that runs on a cache miss.
- **Prompt shape:** instructions, then numbered context (best chunk first, with a fallback line when nothing was found), then the question.
- **Grounding test (invented company, so the model can't know the answer):** with the instruction, it answered "14 days" when the context had it, and replied with the exact sentence "I don't know based on the provided documents." when the context was unrelated or empty. Without the instruction: with unrelated context it said the context doesn't mention refunds and stopped (fine), but with no context at all it answered from general knowledge (refund windows for Amazon, Swiggy, IRCTC) and never mentioned AcmeCloud.
- **Why the instruction line matters:** it forbids outside knowledge (the empty-context case shows what happens without it), it makes the refusal a fixed sentence my code can detect, and it keeps behavior consistent across models and cases.

- **Why chunk:** an embedding of a whole document is a vague blur, while a short focused chunk matches questions about its topic, and only a few chunks fit in a prompt. Retrieval returns chunks, not documents.
- **Overlap:** each chunk repeats the last few words of the previous one so a fact on a boundary isn't split. With size 15 and no overlap, my refund sentence was cut ("Refunds are available within" / "14 days of purchase"), and neither chunk alone answered "how many days". To guarantee a piece of L words stays whole, overlap ≥ L − 1 (tested: 8-word sentence, overlap 7, intact at every size I tried). Overlap must be smaller than size.
- **Trade-offs:** chunks too small (size 8 gave 11 fragments) lose context; chunks too big (one 200-word chunk) blur the embedding in a large document; more overlap means more chunks and more duplicated text to embed and store (5 chunks at overlap 0, 8 at overlap 7).

- **The pipeline:** chunk documents (size + overlap), embed the chunks and index them for BM25, then for each question run BM25 and vector search, merge the two ranked lists with RRF, take the top 3 chunks and build the grounded prompt. This is the project's cache-miss path.
- **What I learned from the test:** (1) BM25 without stopword removal and stemming is noisy: for "How many days do I have to ask for a refund?", common words ("many", "have", "to") pulled in unrelated troubleshooting and plans chunks, and "refund" didn't match "Refunds"/"refundable", so the main policy chunk missed the top 3. Redis does stopwords and stemming itself. (2) Retrieval always returns top_k chunks, even for an unrelated question (pizza), and the fused RRF score looked the same for irrelevant and relevant results (0.03105 vs 0.03175), so RRF scores can't signal "nothing relevant"; the grounding instruction does that job. (3) After adding stopwords and stemming: (fill in your result).(3) After adding stopwords and a crude stemmer to the tokenizer: for the refund question, both refund chunks ranked first in BM25 and in embeddings, and the chunk with "within 14 days of purchase" was first in the prompt. For the pizza question BM25 matched nothing (an empty list), so the fused list came from embeddings alone, with a top score of 0.01639 (= 1/61, the most a single list can give). Still, an empty BM25 list can't be used as a "no answer" rule, because paraphrased questions also have empty BM25 lists. The grounding instruction is what makes the model say "I don't know".

## 11. Semantic cache

- **How lookup works:** embed the new question, find the nearest saved question (dot product against every saved embedding, then argmax), and return its saved answer only if the similarity is at least the threshold; otherwise it's a miss and the full pipeline runs.
- **What my measurements said:** real paraphrases scored 0.67 to 0.83, but traps scored 0.70 to 0.84 (refund window vs refund arrival 0.841, Starter vs Pro plan 0.840, love/hate 0.813), so no threshold separates paraphrases from traps: every threshold that blocks the traps (0.85+) also blocks all my real paraphrases. Near-duplicates (different capitalization, punctuation, a reworded "How can I..." ) scored 0.93 to 1.0; a typo dropped to 0.70. So the semantic layer mostly catches near-duplicates, not loose paraphrases.
- **My threshold:** 0.90, inside the empty gap between the highest trap (0.841) and the lowest near-duplicate (0.933), on the safe side because a missed hit costs one extra LLM call and a wrong hit returns a wrong answer with full confidence. It's based on 11 pairs, so I'll log near-misses (0.80 to 0.95) in the real project and revisit. In Redis, 0.90 similarity = 0.10 cosine distance.

- **Redis version:** the cache entries are hashes (`prompt`, `answer`, `embedding` as float32 bytes: 384 × 4 = 1,536 bytes) under keys with a prefix, and the index has a `VECTOR` field (HNSW, FLOAT32, DIM 384, COSINE). A lookup is a KNN query (`*=>[KNN 1 @embedding $vec AS distance]`) with the query vector passed as bytes in `query_params`. Needs `dialect(2)`.
- **Distance rule:** Redis returns cosine distance = 1 - similarity, so a 0.90 similarity threshold means distance <= 0.10. My Redis results matched my NumPy results (0.994, 0.992, 0.702, 0.823, 0.53, 0.667), which confirms the conversion. Redis always returns the nearest neighbour, even a bad one; my code applies the threshold to decide hit or miss. Gotcha: in current redis-py the import is `redis.commands.search.index_definition` (older versions used `indexDefinition`).

- **Layer order and why:** exact first (a hash and one GET, no embedding), then semantic (embed the question, then a KNN query and a threshold check), then the full pipeline. Each layer protects the more expensive one after it. A miss saves into both layers with a TTL (`set(..., ex=TTL)` for exact; `hset` followed by `expire` for semantic, since HSET has no EX).
- **My measurements:** miss about 1.3 to 1.5 s (1.2 s is the mock sleep, the rest is overhead, including embedding the question twice), semantic hit 53 to 83 ms (most of it embedding the question on the CPU), exact hit 2.4 ms. The semantic hit on a lowercase variant scored 0.994, and the loose paraphrase (0.823) and a different question (0.53) correctly missed. `model.encode` blocks the event loop like `time.sleep` does.
- **Improvement:** (fill in after the rerun) saving the answer under the exact key on a semantic hit made the repeat call become ___ at ___ ms. Downside: the new key gets a fresh TTL, so an answer can live up to about twice its original TTL.

## 12. Load testing

- **httpx:** the Python library for sending HTTP requests from code (like curl). `httpx.Client(timeout=30)` reuses one connection and gives up after 30 s; `post(url, json={...})` sends a JSON body; `raise_for_status()` turns error statuses into exceptions so a failing server can't produce quietly wrong numbers. A stopped server gives `httpx.ConnectError`, the same "nothing is listening on that port" as the Redis connection error.
- **Server time vs client time:** the server's `execution_time_ms` only covers the function body, not parsing, validation, serialization or the network. My measurements: miss 1218 ms server / 1231 ms client (about 12 ms gap), hit about 1.1 ms server / about 3.8 ms client (about 2.7 ms gap, around 70% of what the client sees). The client time is what a user experiences, and what Locust measures, so it's the number to report. These were localhost numbers; a real network adds more.

- **Locust:** a `HttpUser` is a simulated person; `@task` is what it does repeatedly; `wait_time = between(0.5, 2)` is its pause between requests. Users are added at the spawn rate until the target is reached. `catch_response=True` lets me mark a response as a failure based on its content, not just the HTTP status.
- **My run (10 users, 5 prompts, about 72 s, Redis exact cache):** 531 requests, 0 failures, 7.4 RPS (about 10 users / 1.25 s average wait; the RPS is capped by user wait time, not by the server). Median 6 ms, 95th percentile 28 ms, 99th percentile 1200 ms, average 32 ms, max 1331 ms.
- **Why the average misleads:** about 98% of requests were fast hits and about 10 (roughly 2%) were 1.2 s misses, so the average (32 ms) describes no real request, and it sits above the 95th percentile (28 ms). Percentiles show the shape. The extra misses (about 10 for 5 prompts) were a cache stampede at startup: all 10 users asked at once before the first answer was cached.