import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import ElasticsearchStore
# from dotenv import load_dotenv
from constants import contextualize_q_system_prompt, universal_assistant_prompt
from text_extractor import convert_to_text
from config import Config
from sentence_transformers import SentenceTransformer
from EmbeddingService import EmbeddingService

# Load environment variables
# load_dotenv()

class OllamaElasticsearchManager:
    def __init__(self, 
                 es_url=Config.ELASTICURL,
                 es_username=None,
                 es_password=None,
                 ollama_host=Config.RUNPOD,
                 llm_model = Config.MODEL,
                 embedding_model='all-MiniLM-L6-v2',
                #  llm_model="llama3.2:3b",
                 temperature=0,
                 chunk_size=1000,
                 chunk_overlap=200,
                 index_name=None):
        """
        Initialize Ollama-powered Elasticsearch RAG Manager
        
        :param es_url: Elasticsearch cluster URL
        :param es_username: Elasticsearch username
        :param es_password: Elasticsearch password
        :param ollama_host: Ollama server URL
        :param llm_model: Ollama model for LLM & model for embeddings
        :param temperature: LLM temperature
        :param chunk_size: Text chunk size for splitting
        :param chunk_overlap: Overlap between chunks
        :param index_name: Default Elasticsearch index name
        """
        
        # Initialize Elasticsearch client
        self.es_url = es_url
        self.index_name = index_name
        
        # Setup Elasticsearch connection
        if es_username and es_password:
            self.es_client = Elasticsearch(
                [es_url],
                basic_auth=(es_username, es_password),
                verify_certs=False
            )
        else:
            self.es_client = Elasticsearch([es_url])
        
        # Test Elasticsearch connection
        try:
            info = self.es_client.info()
            print(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            print(f"Failed to connect to Elasticsearch: {e}")
            raise
        
        # Initialize Ollama client
        self.ollama_host = ollama_host
        self.llm_model = llm_model
        self.llm_model = llm_model
        
        # Initialize Embedding model
        self.embedding_model= SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            self.ollama_client = ollama.Client(host=ollama_host)
            # Test Ollama connection by getting embedding dimension
            test_embedding = self.ollama_client.embeddings(
                model=llm_model, 
                prompt="test"
            )["embedding"]
            self.embedding_dim = len(test_embedding)
            print(f"Connected to Ollama. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            raise
        
        # Initialize LangChain components with Ollama
        self.ollama_embeddings = OllamaEmbeddings(
            base_url=ollama_host,
            model=llm_model
        )
        
        self.llm = Ollama(
            base_url=ollama_host,
            model=llm_model,
            temperature=temperature
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Initialize prompts
        self.contextualize_q_prompt = self.create_contextualize_prompt()
        self.qa_prompt = self.create_qa_prompt()
        
        # Initialize session storage
        self.session = {}
        
        # Initialize vector store and chains if index exists
        if self.index_name and self.index_exists(self.index_name):
            self.load_embeddings()
            self.initialize_chains()

    def create_contextualize_prompt(self):
        """Create contextualize prompt for history-aware retrieval"""
        return ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def create_qa_prompt(self):
        """Create QA prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", universal_assistant_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists in Elasticsearch"""
        try:
            return self.es_client.indices.exists(index=index_name)
        except Exception as e:
            print(f"Error checking index existence: {e}")
            return False

    def create_index(self, index_name: str):
        """
        Create an Elasticsearch index with proper mappings for vector search
        
        :param index_name: Name of the index to create
        """
        if self.index_exists(index_name):
            print(f"Index '{index_name}' already exists")
            return

        # Define index mapping for vector search
        mapping = {
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "fileId": {"type": "keyword"},
                            "chunkId": {"type": "keyword"},
                            "deleted": {"type": "boolean"}
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        }

        try:
            self.es_client.indices.create(index=index_name, body=mapping)
            print(f"Index '{index_name}' created successfully")
        except Exception as e:
            print(f"Error creating index: {e}")
            raise

    def delete_index(self, index_name: str):
        """Delete an Elasticsearch index"""
        try:
            if self.index_exists(index_name):
                self.es_client.indices.delete(index=index_name)
                print(f"Index '{index_name}' deleted successfully")
            else:
                print(f"Index '{index_name}' does not exist")
        except Exception as e:
            print(f"Error deleting index: {e}")
            raise

    def list_indices(self):
        """List all indices in Elasticsearch"""
        try:
            indices = self.es_client.cat.indices(format="json")
            index_names = [idx['index'] for idx in indices if not idx['index'].startswith('.')]
            print(f"Available indices: {index_names}")
            return index_names
        except Exception as e:
            print(f"Error listing indices: {e}")
            return []

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        try:
            embedding = self.ollama_client.embeddings(
                model=self.llm_model,
                prompt=text
            )["embedding"]
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    def generate_embeddings_from_chunks(self, text_chunks: List[str], index_name: str, fileId: str):
        """
        Convert text chunks to embeddings and store in Elasticsearch using Ollama
        
        :param text_chunks: List of text chunks
        :param index_name: Name of the Elasticsearch index
        :param fileId: Unique identifier for the source file
        """
        print(f"generate embeddings:::")
        try:
            embd_client = EmbeddingService(
                es_client=self.es_client,
                ollama_client=self.ollama_client
            )
            
            embed = self.ollama_client.embeddings(model="llama3.2:3b", prompt="a")["embedding"]
            embedding_dim = len(embed)
            print(f"embedding_dim value :::: {embedding_dim}")
            print(f"embedding_dim type :::: {type(embedding_dim)}")
            return embd_client.generate_embeddings_from_chunks(text_chunks, index_name, fileId, embedding_dim)
            # Ensure index exists
            # if not self.index_exists(index_name):
            #     self.create_index(index_name)

            # print(f"Processing {len(text_chunks)} chunks for file {fileId}...")
            
            # # Process chunks in batches for better performance
            # # for i, chunk in enumerate(text_chunks):
            #     # Generate embedding using Ollama
            # documents = Document(page_content=text_chunks)
            # es_client = Elasticsearch(hosts=[Config.ELASTICURL]) 
            # # embedding =  self.embedding_model.encode(chunk)
            # es_store = ElasticsearchStore.from_documents(
            #         documents=documents,
            #         embedding=self.es_client,
            #         es_client=es_client,
            #         index_name=self.index_name
            # )
                
            # doc_id = hashlib.sha256(chunk.encode()).hexdigest()  # deterministic ID
            # es.index(
            #         index=index_name,
            #         id=doc_id,
            #         document={"text": doc["text"], 
            #                   "source": doc["source"], 
            #                   "embedding": embedding},
            #     )
            # vectorstore = ElasticsearchStore.from_documents(
            #         documents=documents,
            #         embedding=self.ollama_embeddings,
            #         es_client=self.es_client,
            #         index_name=self.index_name  # choose a name
            #     )
                
                
            # retriever = vectorstore.as_retriever(
            #         search_type="similarity",
            #         search_kwargs={"k": 3}  # Fewer chunks for focused extraction
            #     )
            # embedding = self.generate_embedding(chunk)
                
            # if not embedding:
            #     print(f"Failed to generate embedding for chunk {i}")
            #     continue
                
                # Create deterministic ID
                # chunk_id = f"{fileId}_{i}"
                # doc_id = hashlib.sha256(chunk_id.encode()).hexdigest()
                
                # # Prepare document
                # document = {
                #     "text": chunk,
                #     "embedding": embedding,
                #     "metadata": {
                #         "source": index_name,
                #         "fileId": fileId,
                #         "chunkId": chunk_id,
                #         "deleted": False
                #     }
                # }
                
                # # Index document
                # self.es_client.index(
                #     index=index_name,
                #     id=doc_id,
                #     document=document
                # )
                
                # if (i + 1) % 10 == 0:
                #     print(f"Processed {i + 1}/{len(text_chunks)} chunks")
            
            # Refresh index to make documents searchable
            # self.es_client.indices.refresh(index=index_name)
            # print(f"Successfully added {len(text_chunks)} chunks from {fileId} to index {index_name}")
        
        except Exception as e:
            print(f"Error converting chunks to embeddings: {e}")
            raise

    def generate_text_chunks(self, file_path: str, fileId: str, index_name: str):
        """
        Extract text from file, split into chunks, and generate embeddings
        
        :param file_path: Path to the source file
        :param fileId: Unique identifier for the file
        :param index_name: Elasticsearch index name
        """
        print("generate text chunk")
        try:
            text = convert_to_text(file_path)
            print(f"generate text chunk === {text}")
            chunks = self.text_splitter.split_text(text)
            self.generate_embeddings_from_chunks(
                text_chunks=chunks, 
                index_name=index_name, 
                fileId=fileId
            )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            raise

    def vector_search(self, query: str, index_name: str, k: int = 5, 
                     include_deleted: bool = False) -> List[Dict]:
        """
        Perform vector similarity search using Ollama embeddings
        
        :param query: Search query
        :param index_name: Elasticsearch index name
        :param k: Number of results to return
        :param include_deleted: Whether to include soft-deleted documents
        :return: List of search results
        """
        try:
            # Generate query embedding using Ollama
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Build filter for deleted documents
            filter_clause = []
            if not include_deleted:
                filter_clause.append({"term": {"metadata.deleted": False}})
            
            # Perform kNN search
            search_body = {
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": k,
                    "num_candidates": min(k * 10, 100)
                },
                "_source": ["text", "metadata"]
            }
            
            # Add filter if needed
            if filter_clause:
                search_body["knn"]["filter"] = {"bool": {"must": filter_clause}}
            
            response = self.es_client.search(index=index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'content': hit['_source']['text'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'id': hit['_id']
                })
            
            return results
        
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []

    def hybrid_search(self, query: str, index_name: str, k: int = 5, 
                     include_deleted: bool = False) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search
        
        :param query: Search query
        :param index_name: Elasticsearch index name
        :param k: Number of results to return
        :param include_deleted: Whether to include soft-deleted documents
        :return: List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                print("Failed to generate query embedding")
                return []
            
            # Build filter clause
            filter_clause = []
            if not include_deleted:
                filter_clause.append({"term": {"metadata.deleted": False}})
            
            # Hybrid search combining kNN and text search
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "knn": {
                                    "embedding": {
                                        "vector": query_embedding,
                                        "k": k
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["text"],
                                    "type": "best_fields",
                                    "boost": 0.3
                                }
                            }
                        ],
                        "filter": filter_clause
                    }
                },
                "size": k,
                "_source": ["text", "metadata"]
            }
            
            response = self.es_client.search(index=index_name, body=search_body)
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'content': hit['_source']['text'],
                    'metadata': hit['_source']['metadata'],
                    'score': hit['_score'],
                    'id': hit['_id']
                })
            
            return results
        
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            return []

    def load_embeddings(self):
        """Load embeddings from Elasticsearch index (setup for retrieval)"""
        if not self.index_name:
            raise ValueError("Index name not specified")
        
        # Create a custom retriever for Ollama-based search
        class OllamaElasticsearchRetriever:
            def __init__(self, es_manager, index_name, k=5):
                self.es_manager = es_manager
                self.index_name = index_name
                self.k = k
            
            def get_relevant_documents(self, query):
                results = self.es_manager.vector_search(query, self.index_name, self.k)
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result['content'],
                        metadata=result['metadata']
                    )
                    documents.append(doc)
                return documents
        
        self.retriever = OllamaElasticsearchRetriever(self, self.index_name)

    def initialize_chains(self):
        """Initialize RAG chains"""
        if not hasattr(self, 'retriever'):
            raise ValueError("Retriever not initialized. Call load_embeddings() first.")
        
        # Create a simple retrieval chain without LangChain's complex chain operations
        # since we're using Ollama directly
        self.qa_chain_initialized = True

    def ask_question_simple(self, query: str, session_id: str = None) -> str:
        """
        Simple question answering using Ollama and Elasticsearch
        
        :param query: User query
        :param session_id: Session identifier (optional)
        :return: Answer string
        """
        try:
            # Get relevant documents
            if not hasattr(self, 'retriever'):
                self.load_embeddings()
            
            relevant_docs = self.retriever.get_relevant_documents(query)
            
            # Prepare context from relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs[:3]])
            
            # Create prompt
            prompt = f"""Based on the following context, answer the question. If you don't know the answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response using Ollama
            response = self.ollama_client.generate(
                model=self.llm_model,
                prompt=prompt
            )
            
            answer = response['response']
            
            # Store in session if provided
            if session_id:
                if session_id not in self.session:
                    self.session[session_id] = []
                self.session[session_id].append({
                    'query': query,
                    'answer': answer,
                    'context_docs': len(relevant_docs)
                })
            
            return answer
        
        except Exception as e:
            print(f"Error processing question: {e}")
            return "I'm sorry, I encountered an error while processing your question."

    def soft_delete_document(self, file_id: str, index_name: str, deleted: bool = True):
        """
        Soft delete a document by updating the deleted flag in metadata
        
        :param file_id: The ID of the file to soft delete
        :param index_name: Name of the index
        :param deleted: Boolean flag indicating deletion status
        :return: Number of documents updated
        """
        try:
            # Update query to mark documents as deleted/undeleted
            update_body = {
                "script": {
                    "source": "ctx._source.metadata.deleted = params.deleted",
                    "params": {"deleted": deleted}
                },
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"metadata.fileId": file_id}},
                            {"term": {"metadata.deleted": not deleted}}
                        ]
                    }
                }
            }
            
            response = self.es_client.update_by_query(
                index=index_name,
                body=update_body,
                refresh=True
            )
            
            updated_count = response.get('updated', 0)
            action = 'deleted' if deleted else 'restored'
            print(f"Successfully {action} {updated_count} documents for file_id: {file_id}")
            return updated_count
        
        except Exception as e:
            print(f"Error during soft delete: {e}")
            raise

    def hard_delete_document(self, file_id: str, index_name: str):
        """
        Permanently delete documents from Elasticsearch
        
        :param file_id: The ID of the file to delete
        :param index_name: Name of the index
        :return: Number of documents deleted
        """
        try:
            delete_body = {
                "query": {
                    "term": {"metadata.fileId": file_id}
                }
            }
            
            response = self.es_client.delete_by_query(
                index=index_name,
                body=delete_body,
                refresh=True
            )
            
            deleted_count = response.get('deleted', 0)
            print(f"Successfully hard deleted {deleted_count} documents for file_id: {file_id}")
            return deleted_count
        
        except Exception as e:
            print(f"Error during hard delete: {e}")
            raise

    def get_document_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics about documents in the index
        
        :param index_name: Name of the index
        :return: Dictionary with document statistics
        """
        try:
            # Get total document count
            total_docs = self.es_client.count(index=index_name)['count']
            
            # Get count of non-deleted documents
            active_docs = self.es_client.count(
                index=index_name,
                body={"query": {"term": {"metadata.deleted": False}}}
            )['count']
            
            # Get count of deleted documents
            deleted_docs = self.es_client.count(
                index=index_name,
                body={"query": {"term": {"metadata.deleted": True}}}
            )['count']
            
            # Get unique file count
            file_agg = self.es_client.search(
                index=index_name,
                body={
                    "size": 0,
                    "aggs": {
                        "unique_files": {
                            "cardinality": {
                                "field": "metadata.fileId"
                            }
                        }
                    }
                }
            )
            unique_files = file_agg['aggregations']['unique_files']['value']
            
            stats = {
                "total_documents": total_docs,
                "active_documents": active_docs,
                "deleted_documents": deleted_docs,
                "unique_files": unique_files
            }
            
            print(f"Index '{index_name}' statistics: {stats}")
            return stats
        
        except Exception as e:
            print(f"Error getting document stats: {e}")
            return {}

    def clear_session(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.session:
            del self.session[session_id]
            print(f"Cleared session: {session_id}")

    def get_session_history(self, session_id: str) -> List:
        """Get conversation history for a session"""
        return self.session.get(session_id, [])

# Example usage
# if __name__ == "__main__":
#     # Example configuration matching your setup
#     es_manager = OllamaElasticsearchManager(
#         es_url=Config.ELASTICURL,
#         es_username=Config.ELASTICUSER,
#         es_password=Config.ELASTICPASS,
#         ollama_host=Config.RUNPOD,
#         llm_model=Config.MODEL,
#         index_name="rag_docs_ollama"
#     )
    
#     # Create index
#     es_manager.create_index("rag_docs_ollama")
    
#     # Example: Add some documents
#     documents = [
#         "The weather forecast for tomorrow is cloudy.",
#         "I had pancakes for breakfast.",
#         "The stock market is down 500 points today.",
#         "Building an exciting new project with LangChain."
#     ]
    
#     # Add documents to index
#     es_manager.generate_embeddings_from_chunks(
#         text_chunks=documents,
#         index_name="rag_docs_ollama",
#         fileId="example_docs"
#     )
    
#     # Initialize for querying
#     es_manager.load_embeddings()
#     es_manager.initialize_chains()
    
#     # Ask a question
#     answer = es_manager.ask_question_simple("What is the weather like tomorrow?", "session_001")
#     print(f"Answer: {answer}")