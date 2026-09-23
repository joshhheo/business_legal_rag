from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

files = ["test_rag/termination.txt", "test_rag/payment.txt"]

all_chunks = []

for file in files:
    with open(file, "r") as f:
        text = f.read()

    chunks = text.split(".")

    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            all_chunks.append(chunk)

embeddings = model.encode(all_chunks)

print(embeddings)
