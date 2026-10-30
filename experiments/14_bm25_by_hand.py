
import re
import math

docs = [
    "how to reset your password in the settings menu",
    "reset the router by holding the reset button on the back of the router for ten seconds then the router restarts and the lights blink and the reset is complete",
    "password rules: use a long password and never share your password",
    "contact support if the password reset email does not arrive",
]

def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

toks = [tokenize(d) for d in docs]
N = len(docs)
avgdl = sum(len(t) for t in toks) / N
k1, b = 1.5, 0.75

def idf(word: str) -> float:
    # TODO 1
    n = sum(1 for doc in toks if word in doc)
    return math.log(1 + (N - n + 0.5) / (n + 0.5))

def bm25(query: str, i: int) -> float:
    doc = toks[i]
    total = 0.0

    for word in tokenize(query):
        tf = doc.count(word)

        if tf == 0:
            continue

        # TODO 2
        length_factor = 1 - b + b * len(doc) / avgdl

        # TODO 3
        term_score = idf(word) * tf * (k1 + 1) / (tf + k1 * length_factor)

        total += term_score

    return total

def search(query: str):
    print(f"\nQuery: {query}")
    scored = sorted(((bm25(query, i), i) for i in range(N)), reverse=True)

    for s, i in scored:
        if s > 0:
            print(f"{s:.3f} | doc {i} | {docs[i][:60]}")

print("idf router:", round(idf("router"), 3), " idf password:", round(idf("password"), 3))
search("reset password")
search("the router")
search("password")

