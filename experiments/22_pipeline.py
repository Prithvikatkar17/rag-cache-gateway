import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# paste here: chunk_words, rrf, INSTRUCTIONS, build_prompt
def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    # TODO 1: step = how far the window moves each time (size minus overlap)
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        # TODO 2: take `size` words beginning at `start` (a list slice),
        #         join them with spaces, and add the result to `chunks`
        chunks.append(" ".join(words[start:start + size]))
        # TODO 3: if this window already reaches the end of the text
        #         (start + size >= len(words)), stop the loop with `break`
        if start + size >= len(words):
            break
    return chunks

def rrf(rankings, k=60):
    """rankings is a list of ranked lists, best first."""
    scores = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, start=1):
            # TODO 1: contribution = 1 / (k + position)
            contribution = 1 / (k + position)
            
            # TODO 2: add the contribution to scores[doc].
            #         A doc not seen yet should start at 0
            scores[doc] = scores.get(doc, 0.0) + contribution
            
    # TODO 3: return a list of (doc, score) pairs, highest score first
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

INSTRUCTIONS = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly:
"I don't know based on the provided documents."
Do not use outside knowledge."""


def build_prompt(question: str, chunks: list[str]) -> str:
    # TODO 1: make context_text.
    if not chunks:
        context_text = "(no relevant documents found)"
    else:
        context_text = "\n".join(f"[{n}] {text}" for n, text in enumerate(chunks, start=1))

    # TODO 2: return an f-string with the three sections
    return f"{INSTRUCTIONS}\n\nContext:\n{context_text}\n\nQuestion: {question}"

DOCS = {
    "plans": "AcmeCloud offers three plans: Starter, Pro and Enterprise. Starter costs 5 dollars a month and includes 5 GB of storage and email support. Pro costs 20 dollars a month and includes 100 GB of storage and chat support. Enterprise pricing is custom and includes a dedicated account manager and a 99.9 percent uptime guarantee.",
    "refunds": "Refunds are available within 14 days of purchase. To request one, email billing@acmecloud.example with your order number. Refunds are issued to the original payment method within 5 to 7 business days. Annual plans are refunded in full during the first 14 days and are not refundable after that.",
    "security": "All data stored on AcmeCloud is encrypted at rest with AES-256 and in transit with TLS 1.3. Servers are hosted in Mumbai and Frankfurt. Two-factor authentication is available on every plan. Security audits are performed every year by an independent firm and the reports are available to Enterprise customers.",
    "troubleshooting": "If you see error 504 gateway timeout, wait one minute and retry, because the server may be restarting. Error 429 means you have sent too many requests and must slow down. If uploads fail, check that the file is smaller than 2 GB. For any other problem contact support through the chat widget.",
    "account": "To reset your password, click Forgot password on the login page and follow the email link. To change your email address, open Settings and choose Profile. To delete your account, contact support; deletion removes all stored files after 30 days.",
}
CHUNK_SIZE, OVERLAP = 30, 8

# ---- indexing: done once, at startup ----
chunks, sources = [], []
for name, text in DOCS.items():
    for c in chunk_words(text, CHUNK_SIZE, OVERLAP):
        chunks.append(c)
        sources.append(name)
print(len(chunks), "chunks")
print(any("daysand" in c for c in chunks))

STOPWORDS = {"is", "the", "a", "an", "to", "of", "do", "i", "my", "have", "how", "many",
             "what", "for", "and", "in", "on", "if", "that", "are", "you", "with", "this",
             "it", "or", "be", "can"}

def stem(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") else word

def tokenize(text):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]

bm25 = BM25Okapi([tokenize(c) for c in chunks])
model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vecs = model.encode(chunks)

# ---- retrieval: done for every question ----
def retrieve(question, top_k=3):
    scores_bm = bm25.get_scores(tokenize(question))
    scores_emb = chunk_vecs @ model.encode(question)
    # TODO 1: BM25 ranking (only chunks with score > 0)
    bm25_ranking = [
        i for i in np.argsort(scores_bm)[::-1]
        if scores_bm[i] > 0
    ]

    # TODO 2: Embedding ranking (all chunks)
    emb_ranking = np.argsort(scores_emb)[::-1].tolist()

    # TODO 3: Reciprocal Rank Fusion
    fused = rrf([bm25_ranking, emb_ranking])

    return fused[:top_k], bm25_ranking, emb_ranking

questions = [
    "How many days do I have to ask for a refund?",
    "Is my data safe?",
    "error 504 gateway timeout",
    "What is the best pizza recipe?",
]

for q in questions:
    top, bm_r, em_r = retrieve(q)
    print("\n" + "=" * 60)
    print("QUESTION:", q)
    for i, s in top:
        b = bm_r.index(i) + 1 if i in bm_r else "-"
        e = em_r.index(i) + 1
        print(f"  {s:.5f} | {sources[i]:<16} | BM25 rank {b} | emb rank {e} | {chunks[i][:50]}...")

# the exact prompt the LLM would get for the first question
top, _, _ = retrieve(questions[0])
print("\n" + "=" * 60)
print(build_prompt(questions[0], [chunks[i] for i, _ in top]))