import numpy as np
from sentence_transformers import SentenceTransformer

# Load a lightweight, fast sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

class SemanticCache:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.vecs = np.empty((0, 384), dtype=np.float32)   # one row per question

    def add(self, question, answer):
        # TODO 1: append the question and the answer to their lists, and add
        #         the question's embedding as a new row of self.vecs
        self.questions.append(question)
        self.answers.append(answer)
        self.vecs = np.vstack([self.vecs, model.encode(question)])

    def lookup(self, question, threshold):
        if len(self.questions) == 0:
            return None
        
        q = model.encode(question)
        sims = self.vecs @ q                    # similarity to every saved question
        best = int(np.argmax(sims))             # position of the closest one
        
        # TODO 2: if sims[best] >= threshold, return a dict. Otherwise return None.
        if sims[best] >= threshold:
            return {
                "answer": self.answers[best],
                "matched_question": self.questions[best],
                "similarity": float(sims[best])
            }
        return None

# ---------- Experiment 1: the cache in action ----------
cache = SemanticCache()
cache.add("How do I reset my password?", "Click Forgot password on the login page.")
cache.add("Is my data safe?", "Yes, it is encrypted at rest and in transit.")

print("--- Experiment 1 ---")
for q in ["How do I reset my password?",
          "What are the steps to change my password?",
          "What is the best pizza recipe?"]:
    print(q, "->", cache.lookup(q, threshold=0.7))


# ---------- Experiment 2: where does a threshold work? ----------
pairs = [
    # same question, different words
    ("How do I reset my password?", "What are the steps to change my password?", "same"),
    ("How do I reset my password?", "I forgot my password, what should I do?", "same"),
    ("How many days do I have to ask for a refund?", "What is the refund window?", "same"),
    ("Is my data safe?", "How is my data protected?", "same"),
    ("How do I delete my account?", "I want to close my account permanently", "same"),
    
    # different questions that look alike
    ("I love this phone", "I hate this phone", "different"),
    ("How do I reset my password?", "How do I reset my router?", "different"),
    ("How many days do I have to ask for a refund?", "How many days until my refund arrives?", "different"),
    ("Does the Starter plan include chat support?", "Does the Pro plan include chat support?", "different"),
    ("What is the price of the Pro plan?", "What is the price of the Starter plan?", "different"),
    ("How do I delete my account?", "How do I delete a file?", "different"),
]

def sim(a, b):
    # TODO 3: embed both sentences and return their cosine similarity
    vec_a = model.encode(a)
    vec_b = model.encode(b)
    return float(vec_a @ vec_b)

print("\n--- Experiment 2 ---")
rows = sorted(((sim(a, b), label, a, b) for a, b, label in pairs), reverse=True)
for s, label, a, b in rows:
    print(f"{s:.3f} | {label:<9} | {a}  <->  {b}")

print("\nthreshold | real hits | WRONG hits")
for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    # TODO 4: Calculate true positives (good) and false positives (bad)
    good = sum(1 for s, label, a, b in rows if label == "same" and s >= t)
    bad = sum(1 for s, label, a, b in rows if label == "different" and s >= t)
    print(f"{t:<9} | {good:<9} | {bad}")


base = "How do I reset my password?"
variants = [
    "how do i reset my password",
    "How do I reset my password",
    "How can I reset my password?",
    "How do you reset a password?",
    "How do I reset my passwrod?",
    "  How do I reset my password?  ",
]
print("\n--- near-duplicates of:", base)
for v in variants:
    print(f"{sim(base, v):.3f} | {v!r}")