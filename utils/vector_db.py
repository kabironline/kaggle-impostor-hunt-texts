from typing import List, Dict
import os
import numpy as np
import tiktoken
from chromadb.utils import embedding_functions
from chromadb.config import Settings
import chromadb


class VectorDB:
    def __init__(self, collection_name: str, embedding_length: int, working_dir: str = None, documents=None, embedding_function=None, dont_add_if_collection_exist=False):
        """
        Initialize ChromaDB without LangChain.
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
        try:
            self.client.get_collection(collection_name)
            return True
        except Exception:
            return False

    def get_collection_info(self, collection_name: str) -> dict:
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
        self.client.delete_collection(self.collection_name)
        print(f"Deleted collection: {self.collection_name}")

    def add_documents(self, documents: List[Dict]) -> List[str]:
        ids, contents, metadatas = [], [], []
        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{idx}")
            ids.append(doc_id)
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))
        self.collection.add(documents=contents, metadatas=metadatas, ids=ids)
        return ids

    def search(self, query: str, limit: int = 5, filter: dict = None) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter
        )
        formatted_results = []
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

    # ---------------------------
    # Late Chunking Method
    # ---------------------------
    def add_documents_with_late_chunking(self, documents: List[Dict], chunk_size: int = 1500, chunk_overlap: int = 200, max_context: int = 8192):
        """
        Add documents using Late Chunking.
        - Computes global embedding (whole document, truncated to max_context)
        - Splits into chunks and computes chunk embeddings
        - Combines global + local embeddings via mean pooling
        """

        encoding = tiktoken.get_encoding("cl100k_base")

        ids, contents, metadatas = [], [], []

        for idx, doc in enumerate(documents):
            doc_id = doc.get("id", f"doc_{idx}")
            text = doc["content"]
            metadata = doc.get("metadata", {})

            # ---- Global Embedding ----
            tokens = encoding.encode(text)
            if len(tokens) > max_context:
                tokens = tokens[:max_context]
            truncated_text = encoding.decode(tokens)
            global_emb = self.embedding_function([truncated_text])[0]

            # ---- Local Chunk Embeddings ----
            chunks = []
            start = 0
            while start < len(tokens):
                end = min(start + chunk_size, len(tokens))
                chunk_text = encoding.decode(tokens[start:end])
                chunks.append(chunk_text)
                start += (chunk_size - chunk_overlap)

            for c_idx, chunk in enumerate(chunks):
                local_emb = self.embedding_function([chunk])[0]

                # Mean pooling global + local
                combined_emb = np.mean([global_emb, local_emb], axis=0).tolist()

                ids.append(f"{doc_id}_chunk{c_idx}")
                contents.append(chunk)
                meta = metadata.copy()
                meta.update({"parent_id": doc_id, "chunk_index": c_idx})
                metadatas.append(meta)

                # Insert with combined embedding
                self.collection.add(
                    documents=[chunk],
                    metadatas=[meta],
                    ids=[f"{doc_id}_chunk{c_idx}"],
                    embeddings=[combined_emb]
                )

        return ids
