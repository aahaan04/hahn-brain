"""Chunk hahn_content.json, embed with OpenAI, and store in ChromaDB collection 'hahn'."""

import json
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)  # let .env win over any stale OPENAI_API_KEY in the environment

EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500       # words
CHUNK_OVERLAP = 50     # words
BATCH_SIZE = 50

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into word chunks of `size` with `overlap` words shared between chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + size >= len(words):
            break
    return chunks


def embed_batch(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def main():
    with open("hahn_content.json", "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Build the full list of chunks with their source URLs.
    documents = []
    metadatas = []
    ids = []
    for page in pages:
        url = page["url"]
        for i, chunk in enumerate(chunk_text(page["content"])):
            documents.append(chunk)
            metadatas.append({"url": url})
            ids.append(f"{url}#{i}")

    print(f"Total chunks: {len(documents)}")

    chroma_client = chromadb.PersistentClient(path="chroma_db")
    # Recreate the collection from scratch for a clean re-run.
    try:
        chroma_client.delete_collection("hahn")
    except Exception:
        pass
    collection = chroma_client.create_collection("hahn")

    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]
        batch_ids = ids[start:end]
        embeddings = embed_batch(batch_docs)
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids,
        )
        print(f"Embedded and stored chunks {start}-{min(end, len(documents))}")

    print(f"\nDone. Collection 'hahn' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
