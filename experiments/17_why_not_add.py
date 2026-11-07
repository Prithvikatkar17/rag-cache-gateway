docs = ['A', 'B', 'C', 'D','E']
bm25 = {"A": 12.4, "B": 11.9, "C": 3.1, "D": 0.0, "E": 0.0}
emb  = {"A": 0.31, "B": 0.29, "C": 0.82, "D": 0.78, "E": 0.12}

def ranking(scores):
    return sorted(scores , key=scores.get ,reverse=True)

print("bm25 ranking : ", ranking(bm25))
print("embedding ranking : ", ranking(emb))

raw = {d: bm25[d] + emb[d] for d in docs}
print("raw sum:", ranking(raw), {d: round(raw[d], 2) for d in docs})

def minmax(scores):
    lo , hi = min(scores.values()) ,max(scores.values())
    return {d : (s-lo)/(hi -lo) for d , s in scores.items()}

b , e = minmax(bm25) , minmax(emb)
norm = {d :b[d] + e[d] for d in docs}
print("min-max sum:", ranking(norm), {d: round(norm[d], 2) for d in docs})

bm25_outlier = dict(bm25, A=40.0)       # copy of bm25 with A changed to 40
b2 = minmax(bm25_outlier)
norm2 = {d: b2[d] + e[d] for d in docs}
print("min-max, one BM25 outlier:", ranking(norm2), {d: round(norm2[d], 2) for d in docs})


zero = {d: 0.0 for d in docs}
# TODO 3: call minmax(zero) inside try/except ZeroDivisionError
try:
    minmax(zero)
except ZeroDivisionError:
    print("Attempt 4 failed: Division by zero because the highest and lowest scores are the same.")