import hashlib
from typing import List
from langchain.schema import Document
from config import Config

class EmbeddingService:
    def __init__(self, es_client, ollama_client):
        self.es_client = es_client
        self.ollama_client = ollama_client
        

    def index_exists(self, index_name: str) -> bool:
        return self.es_client.indices.exists(index=index_name)

    def create_index(self, index_name: str, embedding_dim:int):
        self.es_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": embedding_dim,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "metadata": {"type": "object"},
                    }
                }
            }
        )

    def generate_embeddings_from_chunks(self, text_chunks: List[str], index_name: str, fileId: str, embedding_dim: int):
        """
        Convert text chunks to embeddings and store in Elasticsearch using Ollama.
        :param text_chunks: List of text chunks
        :param index_name: Name of the Elasticsearch index
        :param fileId: Unique identifier for the source file
        """
        
        try:
            print("Checking if index exists...")
            if not self.index_exists(index_name):
                print("Index does not exist. Creating index...")
                self.create_index(index_name, embedding_dim)
            else:
                print("Index already exists.")

            print(f"Preparing {len(text_chunks)} documents for file {fileId}...")
            documents = [Document(page_content=chunk) for chunk in text_chunks]
            print("Documents prepared.")

            for i, doc in enumerate(documents):
                print(f"\nProcessing chunk {i + 1}/{len(documents)}")

                try:
                    print("Calling Ollama embeddings API...")
                    response = self.ollama_client.embeddings(
                        model=Config.MODEL,  # update to your model
                        prompt=doc.page_content  # <-- fix: pass raw string, not Document object
                    )
                    print("Received response from Ollama.")
                    embedding = response.get("embedding")
                except Exception as embed_err:
                    print(f"Embedding failed for chunk {i}: {embed_err}")
                    continue

                if not embedding:
                    print(f"Empty embedding for chunk {i}, skipping...")
                    continue

                # Create deterministic document ID
                chunk_id = f"{fileId}_{i}"
                doc_id = hashlib.sha256(chunk_id.encode()).hexdigest()
                print(f"Generated doc_id: {doc_id}")

                # Prepare document for indexing
                document = {
                    "text": doc.page_content,
                    "embedding": embedding,
                    "metadata": {
                        "source": index_name,
                        "fileId": fileId,
                        "chunkId": chunk_id,
                        "deleted": False
                    }
                }

                print("Indexing document into Elasticsearch...")
                self.es_client.index(index=index_name, id=doc_id, document=document)
                print("Document indexed successfully.")

                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(text_chunks)} chunks")

            print("Refreshing Elasticsearch index...")
            self.es_client.indices.refresh(index=index_name)
            print(f"Successfully added {len(text_chunks)} chunks from {fileId} to index '{index_name}'")

        except Exception as e:
            print(f"Error during embedding and indexing: {e}")
            raise
