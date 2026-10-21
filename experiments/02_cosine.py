import numpy as np 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine(a,b):
    a , b = np.array(a ,dtype=float) , np.array(b,dtype=float)
    return np.dot(a,b)/((np.linalg.norm(a))*(np.linalg.norm(b)))

print(cosine([3,0] ,[0,5]))
print(cosine([1,0] ,[0,1]))
print(cosine([1,0] ,[-1,0]))


vec = model.encode("How do i reset my password")
print(np.linalg.norm(vec))

stored = [
    "How do I change my password?",
    "What is the best pizza recipe?",
    "How can I SSH into a server?",
    "How do I reset my router?",
    "Steps to recover a forgotten login",
]

stored_vecs = model.encode(stored)

query = "I forgot my password , What should i do?"
q = model.encode(query)

scores = stored_vecs @ q
for text, s in sorted(zip(stored, scores), key=lambda x: -x[1]):
    print(round(float(s), 3), "|", text)


best = float(scores.max())
print("similarity:", round(best, 3), "-> distance:", round(1 - best, 3))