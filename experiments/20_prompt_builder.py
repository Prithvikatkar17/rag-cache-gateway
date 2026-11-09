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


question = "How many days do I have to ask for a refund?"

answer_chunk = "Refunds are available within 14 days of purchase. To request one, email billing@acmecloud.example with your order number."
unrelated_chunk = "AcmeCloud servers are hosted in Mumbai and Frankfurt."

cases = {
    "1: context has the answer": [unrelated_chunk, answer_chunk],
    "2: context is unrelated":   [unrelated_chunk],
    "3: no context at all":      [],
}

for name, chunks in cases.items():
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(build_prompt(question, chunks))
    print()