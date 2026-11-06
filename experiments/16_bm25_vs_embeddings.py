import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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

def show(title, scores):
    print(f"{title}:")
    # Get the indices of the top 3 scores in descending order
    top_indices = np.argsort(scores)[::-1][:3]
    
    # Iterate and print the score alongside the document
    for idx in top_indices:
        print(f"{scores[idx]:.4f} | {docs[idx]}")

for q in queries:
    print(f"\n=== {q}")
    show("BM25", bm25.get_scores(tokenize(q)))
    show("Embeddings", doc_vecs @ model.encode(q))