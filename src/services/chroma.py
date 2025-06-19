import os

import chromadb
from dotenv import load_dotenv
from src.services.embedding import embedding_fn

load_dotenv()

client = chromadb.HttpClient(
    host=str(os.getenv("CHROMA_HOST")),
    port=os.getenv("CHROMA_PORT"),
    ssl=False
)


def create_collection_(collection_name: str):
    collections = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn
    )
    return collections


def list_collections():
    collections = client.list_collections()
    return collections


def delete_collection(collection_name: str):
    client.delete_collection(collection_name)
    return True


def check_collection(collection_name: str):
    try:
        collections = client.get_collection(collection_name)
        count = collections.count()
        print(f"Collection '{collection_name}' has {count} documents")
    except Exception as e:
        print(f"Error checking collection '{collection_name}': {e}")


def view_chroma_contents(collection_name: str):
    collection = client.get_collection(collection_name)
    results = collection.get(include=["documents", "metadatas"])
    print(f"\nTotal entries in '{collection_name}': {len(results['ids'])}\n")
    for i in range(len(results["ids"])):
        print(f"ID: {results['ids'][i]}")
        print(f"Metadata: {results['metadatas'][i]}")
        print(f"Document:\n{results['documents'][i]}...\n")
        print("-" * 200)


def print_document_chunks_by_name(collection_name: str, document_name: str):
    try:
        collection = client.get_collection(collection_name)

        results = collection.get(
            where={"document": document_name},
            include=["documents", "metadatas", "embeddings"]
        )

        if not results["ids"]:
            print(f"No documents found for '{document_name}' in collection '{collection_name}'")
            return

        print(f"\nFound {len(results['ids'])} chunks for document '{document_name}':")
        print("=" * 100)

        for i, (doc_id, document, metadata) in enumerate(zip(
                results["ids"],
                results["documents"],
                results["metadatas"]
        )):
            page = metadata.get("page", "N/A")
            print(f"CHUNK {i + 1}/{len(results['ids'])} | ID: {doc_id} | Page: {page}")
            # print(f"Metadata: {metadata}")
            print(f"Content:\n{document}")
            print("-" * 100)

    except Exception as e:
        print(f"Error retrieving document chunks: {e}")