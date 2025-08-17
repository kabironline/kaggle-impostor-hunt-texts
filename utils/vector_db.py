from typing import List, Dict
import os
import threading
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def get_google_embeddings():
    """Get thread-local instance of Google embeddings"""
    thread_local = threading.local()
    if not hasattr(thread_local, 'google'):
        thread_local.google = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
        )
    return thread_local.google

class VectorDB:
    def __init__(self, collection_name: str, embedding_length: int, google_api_key: str, working_dir: str = None, documents= None):
        """Initialize ChromaDB with LangChain integration
        
        Args:
            collection_name: Name of the collection
            embedding_length: Not directly used by ChromaDB but kept for API compatibility
            open_ai_api_key: OpenAI API key for embeddings
            qdrant_ai_api_key: Not used with ChromaDB but kept for API compatibility
        """
        # Create persistent directory path - next to this file
        self.persist_directory = os.path.join(working_dir, "chroma_db")
        
        # Set up embedding function
        self.embedding_model = "gemini-embedding-001"
        self.embeddings = get_google_embeddings()

        # Store parameters
        self.collection_name = collection_name
        self.embedding_length = embedding_length  # For compatibility with existing code

        if documents is not None:
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
        elif self.collection_exists(collection_name):
            # Initialize Chroma
            self.db = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Create new empty collection
            self.db = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists"""
        try:
            # Try to initialize a client and access the collection
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_directory)
            client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def reset_collection(self, collection_name: str, embedding_length: int = None) -> None:
        """Reset/recreate a collection"""
        print(f"Resetting collection: {collection_name}")
        
        # Delete collection if it exists
        if self.collection_exists(collection_name):
            from chromadb import PersistentClient
            client = PersistentClient(path=self.persist_directory)
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
            
        # Create a new collection
        self.db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Created new collection: {collection_name}")
        
        # Update the collection name
        self.collection_name = collection_name

    def delete_collection(self) -> None:
        """Delete the current collection"""
        from chromadb import PersistentClient
        client = PersistentClient(path=self.persist_directory)
        client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def add_documents(self, documents) -> List[str]:
        """Add documents to the vector store
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs
        """
        return self.db.add_documents(documents)
    
    def search(self, query: str, limit: int = 5, filter: dict = None, score_threshold: float = 0.15) -> List[Dict]:
        """Search for documents similar to the query
        
        Args:
            query: The search query
            limit: Maximum number of results
            filter: Metadata filter criteria
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with id, score, and payload
        """
        # Convert filter from Qdrant style to Chroma style if needed
        chroma_filter = self._convert_filter(filter) if filter else None
        
        # Get results from ChromaDB
        results = self.db.similarity_search_with_score(
            query=query,
            k=limit,
            filter=chroma_filter
        )
        
        # Format results to match the expected structure
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "id": doc.metadata.get("id", ""),
                "score": 1 - score,  # Convert distance to similarity
                "payload": doc.metadata,
                "content": doc.page_content
            })

        return formatted_results
    
    def _convert_filter(self, qdrant_filter: dict) -> dict:
        """Convert Qdrant style filter to ChromaDB filter format"""
        chroma_filter = {}
        
        for key, value in qdrant_filter.items():
            if isinstance(value, str) or isinstance(value, int) or isinstance(value, bool) or isinstance(value, float):
                # Simple equality filter
                chroma_filter[key] = value
            elif isinstance(value, dict):
                # Range filter - needs conversion
                # This is a simplified implementation - may need expansion based on your specific filters
                if "gte" in value:
                    chroma_filter[key] = {"$gte": value["gte"]}
                if "lte" in value:
                    chroma_filter[key] = {"$lte": value["lte"]}
                # Add other operators as needed
        
        return chroma_filter
