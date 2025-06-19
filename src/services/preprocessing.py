from typing import Union

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.services.chroma import client
from src.services.embedding import embedding_fn


def generate_docs_embeddings(docs: Document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_documents([docs])
    print(f"chunks{chunks}")

    for i, chunk in enumerate(chunks):
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = i + 1

    embeddings = embedding_fn.embed_documents([chunk.page_content for chunk in chunks])

    print(f"Total Chunks Created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i} (Page {chunk.metadata['page']}): {chunk.page_content[:200]}...")
    return chunks, embeddings


def generate_text_embeddings(docs: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    chunks = text_splitter.split_text(docs)
    embeddings = embedding_fn.embed_documents(texts=chunks)

    print(f"Total Chunks Created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):
        print(f"Chunk {i}: {chunk[:100]}...\n")
    print(f"chunks{chunks}")
    return chunks, embeddings


def save_embeddings(name: str, docs: Union[str, Document]):
    collection = client.get_or_create_collection("scandlearn")
    print(collection)

    if isinstance(docs, Document):
        chunks, embeddings = generate_docs_embeddings(docs)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            page_number = chunk.metadata.get("page", i + 1)
            unique_id = f"{name}_{i}"
            print(f" Storing Chunk {i} with Page {page_number}")
            collection.upsert(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[{
                    "document": name, "text": chunk.page_content, "page": page_number}],
                documents=[chunk.page_content]
            )
    elif isinstance(docs, str):
        chunks, embeddings = generate_text_embeddings(docs)
        if len(chunks) != len(embeddings):
            print(f"Warning: Chunks count {len(chunks)} doesn't match embeddings count {len(embeddings)}")
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            unique_id = f"{name}_{i}"
            print(f" Storing Chunk {i} with Page {i + 1}")
            collection.upsert(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[{"document": name, "text": chunk, "page": i + 1}],
                documents=[chunk]
            )
    return True
