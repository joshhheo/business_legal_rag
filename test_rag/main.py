from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from google import genai

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

# returns 2D list
similarities = cosine_similarity(question_embedding, embeddings)

# arg sort is ascending by default
# arg sort saves initial index positions of each value
# index position represents chunk identity
sorted_similarities = similarities[0].argsort()[::-1]

top_matches = sorted_similarities[:2]

retrieved_chunks = []

for index in top_matches:
    retrieved_chunks.append(all_chunks[index])

load_dotenv()

api_key = os.getenv("gemini_API_key")

client = genai.Client(api_key=api_key)

context = retrieved_chunks

prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print(response.text)
