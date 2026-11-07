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

bm25_list = ["A", "C", "B"]
emb_list = ["C", "D", "A", "B", "E"]

for doc, score in rrf([bm25_list, emb_list]):
    print(f"{doc}: {score:.5f}")