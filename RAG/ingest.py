import fitz
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import pickle


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# EXTRACT TEXT FROM PDF
# --------------------------------------------------

def text_extract(doc_path):

    pdf_path = list(Path(doc_path).glob("*.pdf"))[0]

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):

        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number
        })

    return pages


# --------------------------------------------------
# CHUNK TEXT
# --------------------------------------------------

def chunk_text(text, chunksize=500, overlap=100):

    paragraphs = text.split("\n\n")

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        # If the paragraph itself is larger than
        # the chunk size
        if len(paragraph) > chunksize:

            # Save whatever we have collected
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Split the large paragraph with overlap
            start = 0

            while start < len(paragraph):

                end = start + chunksize

                chunk = paragraph[start:end]

                chunks.append(chunk)

                start += chunksize - overlap

        else:

            # Try to add paragraph to current chunk
            if len(current_chunk) + len(paragraph) <= chunksize:

                if current_chunk:
                    current_chunk += "\n\n" + paragraph

                else:
                    current_chunk = paragraph

            else:

                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk)

                current_chunk = paragraph

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# --------------------------------------------------
# INGESTION
# --------------------------------------------------

pages = text_extract("Document")

print("Number of pages:", len(pages))


all_chunks = []

chunk_id = 0


for page in pages:

    page_chunks = chunk_text(page["text"])

    for chunk in page_chunks:

        all_chunks.append({
            "text": chunk,
            "source": "Company_policy.pdf",
            "page": page["page"],
            "chunk_id": chunk_id
        })

        chunk_id += 1


print("Number of chunks:", len(all_chunks))


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

texts = [
    chunk["text"]
    for chunk in all_chunks
]

embeddings = embedding_model.encode(texts)

print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# CREATE FAISS INDEX
# --------------------------------------------------

index = faiss.IndexFlatL2(384)

index.add(
    np.array(embeddings).astype("float32")
)

print("Number of vectors:", index.ntotal)


# --------------------------------------------------
# SAVE FAISS INDEX
# --------------------------------------------------

faiss.write_index(
    index,
    "RAG/faiss_index.bin"
)


# --------------------------------------------------
# SAVE CHUNKS + METADATA
# --------------------------------------------------

with open("RAG/chunks.pkl", "wb") as f:

    pickle.dump(all_chunks, f)


print("Chunks and metadata saved")

print("Ingestion Completed")