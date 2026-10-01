import os
import chromadb

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "RAG")

CHROMA_PATH = os.path.join(
    DATA_DIR,
    "chroma_db"
)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# --------------------------------------------------
# 3. Get the company policy collection
# --------------------------------------------------

collection = client.get_collection(
    name="company_policy"
)


# --------------------------------------------------
# 4. Retrieve relevant documents
# --------------------------------------------------

def retrieve(query, k=3, threshold=1.0):

    # Convert the query into an embedding
    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    # Search Chroma
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    retrieved_results = []

    # Process retrieved documents
    for i in range(len(results["documents"][0])):

        distance = results["distances"][0][i]

        # Reject results that are not relevant enough
        if distance >= threshold:
            continue

        retrieved_results.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_id": results["metadatas"][0][i]["chunk_id"],
            "distance": distance
        })

    return retrieved_results