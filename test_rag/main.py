from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

question = "How can a contract be terminated?"

# encode expects 2D array
question_embedding = model.encode([question])

similarities = cosine_similarity(question_embedding, embeddings)

print(similarities)

similarities = similarities[0]

# arg sort is ascending by default
# arg sort saves initial index positions of each value
# index position represents chunk identity
# 2 slices because reverse order before indexing
sorted_similarities = similarities.argsort()[::-1][:2]

for index in sorted_similarities:
    print(all_chunks[index])

