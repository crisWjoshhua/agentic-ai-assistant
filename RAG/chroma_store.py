import chromadb
import pickle
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Load the embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# 2. Create a persistent Chroma database
# --------------------------------------------------

client = chromadb.PersistentClient(
    path="RAG/chroma_db"
)


# --------------------------------------------------
# 3. Create or get a collection
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="company_policy"
)


# --------------------------------------------------
# 4. Load our existing chunks and metadata
# --------------------------------------------------

with open("RAG/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 5. Prepare data for Chroma
# --------------------------------------------------

documents = []
metadatas = []
ids = []


for chunk in chunks:

    documents.append(chunk["text"])

    metadatas.append({
        "source": chunk["source"],
        "page": chunk["page"],
        "chunk_id": chunk["chunk_id"]
    })

    ids.append(
        f"chunk_{chunk['chunk_id']}"
    )


# --------------------------------------------------
# 6. Generate embeddings
# --------------------------------------------------

texts = [chunk["text"] for chunk in chunks]

embeddings = embedding_model.encode(texts)

embeddings = embeddings.tolist()


# --------------------------------------------------
# 7. Add everything to Chroma
# --------------------------------------------------

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)


# --------------------------------------------------
# 8. Verify
# --------------------------------------------------

print(
    "Documents stored in Chroma:",
    collection.count()
)

print("Chroma ingestion completed!")