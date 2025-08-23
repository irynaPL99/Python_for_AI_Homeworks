"""task 4.2
Получить эмбеддинги двух разных текстов и сравнить их.
Реализовать простой поиск похожих текстов.
"""
import os
import faiss
from dotenv import load_dotenv
from google import genai
import numpy as np
from pathlib import Path

# Go one folder up and load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load API key from .env file
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)


def get_embedding(text):
    """
    Get embedding (vector) for one text.
    :param text: input text (string)
    :return: vector (numpy array)
    """
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(response.embeddings[0].values)


# Example texts about Python and jobs
texts_to_index = [
    "Python is easy to learn.",
    "Python is used for data science.",
    "Many web developers use Python.",
    "Python is popular for machine learning.",
    "Software engineers write code in Python.",
    "Python is used for automation.",
    "Data analysts often work with Python.",
    "Python is good for beginners in programming.",
    "Scientists use Python for research.",
    "AI engineers use Python every day.",
    "I like programming in Python.",
    "My hobby is to write Python code.",
    "I want to get a job as a Python developer.",
    "Python helps people work with big data.",
    "I enjoy learning Python in my free time."
]

# Get embeddings for all texts
embeddings_list = [get_embedding(text) for text in texts_to_index]
embeddings_array = np.array(embeddings_list)

# Build FAISS index
dimension = embeddings_array.shape[1]      # vector size
index = faiss.IndexFlatL2(dimension)       # L2 = Euclidean distance
index.add(embeddings_array)                # add vectors to index


def semantic_search(query, index, texts, k=3):
    """
    Find most similar texts to query.
    :param query: text to search (string)
    :param index: FAISS index
    :param texts: list of texts in database
    :param k: number of results
    :return: list of k texts
    """
    query_embedding = get_embedding(query).reshape(1, -1)
    D, I = index.search(query_embedding, k)
    results = [texts[i] for i in I[0]]
    return results


# Example: search by query
search_query = "Jobs where Python is needed"
search_results = semantic_search(search_query, index, texts_to_index, k=4)

print("\n--- Semantic Search Results ---")
print(f"Query: '{search_query}'")
print("Found texts:")
for result in search_results:
    print(f"- {result}")
