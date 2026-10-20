# Concepts I learned

## 1. Embeddings
**In one line:** An embedding turns a sentence into a list of numbers (a vector) so that sentences with similar meaning end up close together, and unrelated sentences end up far apart.

**How it works:** An embedding model takes a sentence as input and outputs a fixed-length vector (384 numbers for all-MiniLM-L6-v2), no matter how long the sentence is. The model was trained on huge numbers of sentence pairs to pull related sentences close together and push unrelated ones apart. Each sentence becomes a point in a 384-dimensional space, and I compare two sentences by measuring how close their points are using cosine similarity (1.0 means same direction, around 0 means unrelated). No single number means anything on its own; the meaning is spread across all 384 together.

**Where it's used in this project:**
- Layer 2 semantic cache: embed the new question and find the closest saved question (`core/embedding.py`, `api/router.py`)
- Document retrieval: embed the question and find the closest document chunks (`core/search_fusion.py`)
- Redis stores these vectors and searches them with an HNSW index (`db/redis_client.py`)

**What I tried:** I compared sentence pairs with cosine similarity:
- SSH paraphrases: 0.48 (clearly the same question, but a low score)
- SSH vs pizza (unrelated): 0.13
- "reset my password" vs "steps to change my password": 0.823
- "reset my password" vs "reset my router" (different topic): 0.53
- "I love this phone" vs "I hate this phone": 0.813

**What surprised me / what I got wrong at first:** I expected love/hate to score low because the meanings are opposite, but it scored 0.813, almost the same as a true paraphrase (0.823). Embeddings mostly capture what a sentence is about (topic and context), and "love" and "hate" appear in the same kinds of sentences, so the model puts them close together. Negation and opposite words are often blurred. I also expected paraphrases to score above 0.8, but the SSH pair scored only 0.48, so real paraphrases can score low. The project's 0.88 threshold is a choice, not a rule.

**Interview question I can now answer:** *"What's the risk of a semantic cache, and how would you reduce it?"*
The main risk is a false hit: two questions with opposite or different meanings can have very similar embeddings, so the cache returns a confidently wrong answer. A similarity threshold alone can't fix this, because opposite pairs (0.813) can score as high as true paraphrases (0.823), while raising the threshold loses real paraphrase hits (0.48). To reduce the risk, I would use a high threshold, add a second check such as a cross-encoder that compares both sentences together, and set expiry times or avoid caching high-stakes questions.