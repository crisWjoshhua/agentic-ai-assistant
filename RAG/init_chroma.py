import os
import fitz
import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer


# ==================================================
# CONFIGURATION
# ==================================================

DATA_DIR = os.getenv("DATA_DIR", "RAG")

CHROMA_PATH = os.path.join(
    DATA_DIR,
    "chroma_db"
)

PDF_PATH = Path("Document/Company_policy.pdf")


# ==================================================
# EMBEDDING MODEL
# ==================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# EXTRACT TEXT FROM PDF
# ==================================================

def text_extract(pdf_path):

    doc = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        doc,
        start=1
    ):

        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number
        })

    return pages


# ==================================================
# CHUNK TEXT
# ==================================================

def chunk_text(
    text,
    chunksize=500,
    overlap=100
):

    paragraphs = text.split("\n\n")

    paragraphs = [
        p.strip()
        for p in paragraphs
        if p.strip()
    ]

    chunks = []

    current_chunk = ""

    for paragraph in paragraphs:

        if len(paragraph) > chunksize:

            if current_chunk:

                chunks.append(
                    current_chunk
                )

                current_chunk = ""

            start = 0

            while start < len(paragraph):

                end = start + chunksize

                chunk = paragraph[start:end]

                chunks.append(chunk)

                start += chunksize - overlap

        else:

            if (
                len(current_chunk)
                + len(paragraph)
                <= chunksize
            ):

                if current_chunk:

                    current_chunk += (
                        "\n\n" + paragraph
                    )

                else:

                    current_chunk = paragraph

            else:

                if current_chunk:

                    chunks.append(
                        current_chunk
                    )

                current_chunk = paragraph

    if current_chunk:

        chunks.append(
            current_chunk
        )

    return chunks


# ==================================================
# INITIALIZE CHROMA
# ==================================================

def initialize_chroma():

    print("Initializing ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_or_create_collection(
        name="company_policy"
    )

    # --------------------------------------------------
    # Avoid duplicate ingestion
    # --------------------------------------------------

    if collection.count() > 0:

        print(
            "ChromaDB already contains",
            collection.count(),
            "documents."
        )

        print(
            "Skipping ingestion."
        )

        return

    # --------------------------------------------------
    # Extract PDF
    # --------------------------------------------------

    print(
        "Reading:",
        PDF_PATH
    )

    pages = text_extract(
        PDF_PATH
    )

    print(
        "Number of pages:",
        len(pages)
    )

    # --------------------------------------------------
    # Create chunks
    # --------------------------------------------------

    all_chunks = []

    chunk_id = 0

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for chunk in page_chunks:

            all_chunks.append({
                "text": chunk,
                "source": "Company_policy.pdf",
                "page": page["page"],
                "chunk_id": chunk_id
            })

            chunk_id += 1

    print(
        "Number of chunks:",
        len(all_chunks)
    )

    # --------------------------------------------------
    # Prepare Chroma data
    # --------------------------------------------------

    documents = [
        chunk["text"]
        for chunk in all_chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in all_chunks
    ]

    ids = [
        f"chunk_{chunk['chunk_id']}"
        for chunk in all_chunks
    ]

    # --------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------

    print(
        "Generating embeddings..."
    )

    embeddings = embedding_model.encode(
        documents
    ).tolist()

    # --------------------------------------------------
    # Store in Chroma
    # --------------------------------------------------

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(
        "Documents stored in Chroma:",
        collection.count()
    )

    print(
        "ChromaDB initialization completed!"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    initialize_chroma()