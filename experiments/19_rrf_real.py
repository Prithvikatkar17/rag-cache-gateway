import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Paste your rrf(rankings, k=60) function from 18_rrf.py here
def rrf(rankings, k=60):
    """rankings is a list of ranked lists, best first."""
    scores = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, start=1):
            contribution = 1 / (k + position)
            scores[doc] = scores.get(doc, 0.0) + contribution
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

# ---------- Experiment A: what does k control? ----------
bm = ["A", "B", "C", "D", "E"]
em = ["F", "G", "H", "D", "E"]
for k in [1, 60, 1000]:
    print(k, [(d, round(s, 5)) for d, s in rrf([bm, em], k)])

# ---------- Experiment B: your Day 5 data ----------
docs = [
    "How do I reset my password?",
    "Steps to recover a forgotten login",
    "Error ERR_CONN_RESET means the server closed the connection unexpectedly",
    "How to change the WiFi name on your router",
    "Troubleshooting network timeouts and dropped connections",
    "Best pizza recipe with a crispy crust",
]
queries = [
    "ERR_CONN_RESET",
    "recovering lost credentials",
    "slow internet keeps disconnecting",
]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75)
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = model.encode(docs)

for q in queries:
    scores_bm = bm25.get_scores(tokenize(q))
    scores_emb = doc_vecs @ model.encode(q)

    # TODO 1: bm25_ranking = document positions with a BM25 score above 0, best first.
    bm25_ranking = [int(i) for i in np.argsort(scores_bm)[::-1] if scores_bm[i] > 0]
    
    # TODO 2: emb_ranking = ALL document positions, best first.
    emb_ranking = [int(i) for i in np.argsort(scores_emb)[::-1]]

    fused = rrf([bm25_ranking, emb_ranking])
    print(f"\n=== {q}")
    print("BM25 list:", bm25_ranking)
    print("Emb list: ", emb_ranking)
    for i, s in fused[:3]:
        print(f"{s:.5f} | {docs[i]}")