from typing import List, Dict
import os
import threading
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorDB:
    def __init__(self, collection_name: str, embedding_length: int, working_dir: str = None, documents=None, embedding_function=None, dont_add_if_collection_exist=False):
        """
        Initialize ChromaDB without LangChain.
        Args:
            collection_name: Name of the collection
            embedding_length: For API compatibility
            working_dir: Directory for persistent storage
            documents: List of dicts with 'content' and 'metadata'
            embedding_function: Callable that returns embeddings for a list of texts
        """
        self.persist_directory = os.path.join(working_dir, "chroma_db")
        self.collection_name = collection_name
        self.embedding_length = embedding_length

        # Default embedding function: use Chroma's SentenceTransformer
        if embedding_function is None:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        else:
            self.embedding_function = embedding_function

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        if self.collection_exists(collection_name):
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        else:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )

        # Add initial documents if provided
        if documents is not None and not dont_add_if_collection_exist:
            self.add_documents(documents)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_collection_info(self, collection_name: str) -> dict:
        """Get information about a collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection.name,
                "embedding_function": collection.embedding_function,
                "num_documents": collection.count()
            }
        except Exception:
            return {}

    def reset_collection(self, collection_name: str, embedding_length: int = None) -> None:
        """Reset/recreate a collection"""
        print(f"Resetting collection: {collection_name}")
        if self.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Created new collection: {collection_name}")
        self.collection_name = collection_name

    def delete_collection(self) -> None:
        """Delete the current collection"""
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def add_documents(self, documents: List[Dict]) -> List[str]:
        """
        Add documents to the vector store.
        Each document should be a dict with 'content' and 'metadata'.
        Returns list of document IDs.
        """
        ids = []
        contents = []
        metadatas = []
        for doc in documents:
            doc_id = doc.get("id", None)
            if doc_id is None:
                doc_id = f"doc_{len(ids)}"
            ids.append(doc_id)
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        return ids

    def search(self, query: str, limit: int = 5, filter: dict = None) -> List[Dict]:
        """
        Search for documents similar to the query.
        Returns list of dicts with id, score, metadata, and content.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter
        )
        # Chroma returns results as dicts with keys: ids, documents, metadatas, distances
        formatted_results = []
        # Check if results contain at least one result set
        if results.get("ids") and len(results["ids"]) > 0:
                for i in range(len(results["ids"][0])):
                    cosine_similarity = 1 - results["distances"][0][i]
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "score": cosine_similarity,
                        "metadata": results["metadatas"][0][i],
                        "content": results["documents"][0][i]
                    })
        return formatted_results