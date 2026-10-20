from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I securely access a remote server?",
    "How can I SSH into a machine?",
    "What is the best pizza recipe?",
]
vecs = model.encode(sentences)

print(vecs.shape)     # what do you see?
print(vecs[0][:5])    # first 5 numbers of sentence 1

def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cos(vecs[0], vecs[1]))   # similar meaning
print(cos(vecs[0], vecs[2]))   # unrelated

pairs = [
    ("I love this phone", "I hate this phone"),
    ("How do I reset my password?", "What are the steps to change my password?"),
    ("How do I reset my password?", "How do I reset my router?"),
]
for a, b in pairs:
    va, vb = model.encode([a, b])
    print(round(float(cos(va, vb)), 3), "|", a, "|", b)