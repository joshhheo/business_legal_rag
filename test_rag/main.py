from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
from google import genai
import re


with open("test_rag/microsoft_annual_stock_awards.txt", "r") as f:
    text = f.read()

# splits at gaps (not line breaks)
paragraphs = re.split(r"\n\s*\n", text)

all_chunks = []

for paragraph in paragraphs:
    paragraph = paragraph.strip()

    if paragraph:
        all_chunks.append(paragraph)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

chunk_embeddings = model.encode(all_chunks)

question = input()

if not question.strip():
    print("question is blank")
    exit()

# encode expects 2D array
question_embedding = model.encode([question])

# returns 2D list
similarities = cosine_similarity(question_embedding, chunk_embeddings)

# arg sort is ascending by default
# arg sort saves initial index positions of each value
# index position represents chunk identity
sorted_similarities = similarities[0].argsort()[::-1]

similarity_floor = 0.5

max_chunks = 4

retrieved_chunks = []

for index in sorted_similarities:
    # test similarity values
    print(similarities[0][index])
    print(all_chunks[index])
    print()

    if similarities[0][index] > similarity_floor:
        retrieved_chunks.append(all_chunks[index])


if len(retrieved_chunks) > max_chunks:
    print("question too vague")
    exit()

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

Be detailed, direct, and clear.
"""

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt
)

print(response.text)
