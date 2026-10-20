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