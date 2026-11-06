import re
from rank_bm25 import BM25Okapi

docs = [
    "how to reset your password in the settings menu",
    "reset the router by holding the reset button on the back of the router for ten seconds then the router restarts and the lights blink and the reset is complete",
    "password rules: use a long password and never share your password",
    "contact support if the password reset email does not arrive",
]

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

bm25 = BM25Okapi([tokenize(d) for d in docs], k1=1.5, b=0.75)

for q in ["reset password", "the router", "password"]:
    scores = bm25.get_scores(tokenize(q))
    print(q, [round(float(s), 3) for s in scores])