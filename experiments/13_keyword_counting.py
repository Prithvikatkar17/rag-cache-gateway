
import re

docs = [
    "how to reset your password in the settings menu",
    "reset the router by holding the reset button on the back of the router for ten seconds then the router restarts and the lights blink and the reset is complete",
    "password rules: use a long password and never share your password",
    "contact support if the password reset email does not arrive",
]

def tokenize(text: str) -> list[str]:
    # TODO 1
    return re.findall(r"[a-z0-9]+", text.lower())

def score(query: str, doc: str) -> int:
    doc_words = tokenize(doc)
    total = 0

    for word in tokenize(query):
        # TODO 2
        total += doc_words.count(word)

    return total

def search(query: str):
    print(f"\nQuery: {query}")

    scored = [(score(query, d), i) for i, d in enumerate(docs)]

    # TODO 3
    scored.sort(reverse=True)

    for s, i in scored:
        if s > 0:
            print(f"{s} | {i} | {docs[i]}")

search("reset password")
search("the router")
search("password")

