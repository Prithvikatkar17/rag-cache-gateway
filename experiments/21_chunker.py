TEXT = (
    "AcmeCloud offers three plans: Starter, Pro and Enterprise. "
    "Starter includes 5 GB of storage and email support. "
    "Pro includes 100 GB of storage and chat support. "
    "Refunds are available within 14 days of purchase. To request one, email billing@acmecloud.example with your order number. "
    "Enterprise customers receive a dedicated account manager. "
    "AcmeCloud servers are hosted in Mumbai and Frankfurt, and all data is encrypted at rest."
)

KEY = "Refunds are available within 14 days of purchase."

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

def intact(chunks):
    return any(KEY in c for c in chunks)

for size, overlap in [(15, 0), (15, 5), (15, 7), (8, 2), (200, 0)]:
    chunks = chunk_words(TEXT, size, overlap)
    print(f"size={size:<4} overlap={overlap:<2} chunks={len(chunks):<3} key sentence intact: {intact(chunks)}")

print("\n--- size 15, overlap 0 ---")
for i, c in enumerate(chunk_words(TEXT, 15, 0), start=1):
    print(i, c)

print("\n--- size 15, overlap 5 ---")
for i, c in enumerate(chunk_words(TEXT, 15, 5), start=1):
    print(i, c)