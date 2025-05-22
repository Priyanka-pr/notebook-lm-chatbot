def set_soft_delete(self, fileId, collection_name, deleted=True):
    """
    Set or remove soft delete for all chunks related to a file based on fileId.
    If `deleted` is True, it will mark the chunks as deleted.
    If `deleted` is False, it will restore (remove soft delete) the chunks.

    :param fileId: ID of the file to modify the soft delete state.
    :param collection_name: ChromaDB collection where the file's chunks are stored.
    :param deleted: If True, soft delete chunks; if False, restore them.
    """
    try:
        # Retrieve the collection
        collection = self.persistent_client.get_collection(collection_name)

        # Perform the soft delete or restore operation based on the `deleted` argument
        collection.update(
            where={"metadata.fileId": fileId},
            set={"metadata.deleted": deleted}  # Set the deleted state to the passed value
        )
        
        if deleted:
            print(f"Soft deleted all chunks for file with fileId: {fileId}")
        else:
            print(f"Restored all chunks for file with fileId: {fileId}")
    
    except Exception as e:
        print(f"Error during soft delete operation for file {fileId}: {e}")



def hard_delete_file(self, fileId, collection_name):
    """
    Hard delete all chunks related to a file based on fileId.

    :param fileId: ID of the file to be hard deleted.
    :param collection_name: ChromaDB collection where the file's chunks are stored.
    """
    try:
        # Retrieve the collection
        collection = self.persistent_client.get_collection(collection_name)

        # Hard delete all chunks related to the given fileId
        collection.delete(where={"metadata.fileId": fileId})  # Delete based on fileId
        
        print(f"Hard deleted all chunks for file with fileId: {fileId}")
    
    except Exception as e:
        print(f"Error during hard delete of file {fileId}: {e}")






def soft_delete_document(self,file_id,collection_name, deleted):
    """
    Soft delete a document by marking all its chunks as deleted in the metadata.
    Does not remove the actual embeddings from ChromaDB.
    
    :param file_id: The ID of the file to soft delete
    :return: Number of chunks marked as deleted
    """
    try:
        # Get the Chroma collection
        chroma_collection = self.persistent_client.get_collection(name=collection_name)
        
        # Get all documents with matching file_id and current deletion status
        # Using ChromaDB's operator format for where clause
        where_clause = {
            "$and": [
                {"fileId": {"$eq": file_id}},
                {"deleted": {"$eq": not deleted}}
            ]
        }
        
        results = chroma_collection.get(
            where=where_clause
        )
        
        if not results or not results['ids']:
            print(f"No documents found with file_id: {file_id}")
            return 0
        print(deleted, 'this is dleted')
        # Update metadata for all chunks of this file
        for doc_id in results['ids']:
            current_metadata = chroma_collection.get(
                ids=[doc_id]
            )['metadatas'][0]
            
            # Update the deleted flag
            current_metadata['deleted'] = deleted
            
            # Update the document metadata
            chroma_collection.update(
                ids=[doc_id],
                metadatas=[current_metadata]
            )
        
        print(f"Soft deleted {len(results['ids'])} chunks for file_id: {file_id}")
        return len(results['ids'])
    
    except Exception as e:
        print(f"Error during soft delete: {e}")
        return 0
    

    

from dataclasses import dataclass, field
from typing import List, Dict, Union, Any, Optional

from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.documents import Document

from .log import logger

from .vectordb import ChromaDBManager
from .rag_system_prompts.qa_system_prompts import qa_system_prompt
from .rag_system_prompts.contextualize_system_prompts import contextualize_q_system_prompt

from .utils.errors import ChatbotError,InitializationError,EmbeddingError,ChainError


# Type aliases
ChatMessage = Union[HumanMessage, AIMessage]
ChatHistory = List[ChatMessage]


@dataclass
class Chatbot:
    collection_name: str
    persist_directory: str
    llm_model_name: str = "gpt-4"
    embedding_model_name: str = "text-embedding-3-large"
    temperature: float = 0.0
    chunk_size: int = 1000
    chunk_overlap: int = 200
    seed: int = 121

    # Initialize store attributes with proper type annotations
    store: Dict[str, Any] = field(default_factory=dict)
    store_conversation: Dict[str, Any] = field(default_factory=dict)
    rag_chat_history: ChatHistory = field(default_factory=list)

    # Initialize as None and type-annotate components
    vectorstore: Optional[Chroma] = None
    retriever: Any = None
    history_aware_retriever: Any = None
    question_answer_chain: Any = None

    def __post_init__(self):
        """Initialize after dataclass initialization"""
        self.llm = ChatOpenAI(
            model=self.llm_model_name,
            temperature=self.temperature,
            seed=self.seed
        )
        self.contextualize_q_prompt = self.create_contextualize_prompt()
        self.qa_prompt = self.create_qa_prompt()

        try:
            self.chroma_manager = ChromaDBManager(
                self.collection_name,
                self.persist_directory,
                embedding_model_name=self.embedding_model_name
            )
        except Exception as e:
            raise InitializationError(f"Failed to initialize ChromaDBManager: {str(e)}")

    def create_contextualize_prompt(self) -> ChatPromptTemplate:
        """Create the contextualization prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def create_qa_prompt(self) -> ChatPromptTemplate:
        """Create the QA prompt template"""
        return ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def _create_history_aware_retriever(self) -> Any:
        """Create a history-aware retriever chain"""
        if not self.retriever:
            raise ChainError("Retriever not initialized")
        return create_history_aware_retriever(
            self.llm,
            self.retriever,
            self.contextualize_q_prompt
        )

    def _create_question_answer_chain(self) -> Any:
        """Create a question-answer chain"""
        return create_stuff_documents_chain(self.llm, self.qa_prompt)

    def add_embeddings(self, documents: List[Document], collection_name: Optional[str] = None) -> None:
        """Add embeddings to the vector store"""
        try:
            actual_collection = collection_name or self.collection_name
            self.chroma_manager.generate_embeddings(documents, actual_collection)
            logger.info(f"Successfully added embeddings to collection: {actual_collection}")
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to add embeddings: {str(e)}")

    def load_embeddings(self, business_id: str) -> None:
        """Load embeddings for a specific business"""
        try:
            self.vectorstore = self.chroma_manager.load_saved_embeddings(business_id)
            if not self.vectorstore:
                raise EmbeddingError("Failed to load embeddings: vectorstore is None")

            logger.info("Creating Retriever for Loaded Embeddings...")
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={'k': 5, 'fetch_k': 50, 'lambda_mult': 0.5}
            )
            logger.info("Successfully loaded embeddings and created retriever")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to load embeddings: {str(e)}")

    def initialize_chains(self) -> None:
        """Initialize the retriever and QA chains"""
        try:
            logger.info("Creating History Aware Retriever Chain...")
            self.history_aware_retriever = self._create_history_aware_retriever()

            logger.info("Creating Question Answer Chain...")
            self.question_answer_chain = self._create_question_answer_chain()

            logger.info("Successfully initialized all chains")
        except Exception as e:
            logger.error(f"Failed to initialize chains: {str(e)}")
            raise ChainError(f"Failed to initialize chains: {str(e)}")

    def initialize_bot(self, business_id: str) -> None:
        """Initialize the chatbot for a specific business"""
        try:
            self.load_embeddings(business_id)
            self.initialize_chains()
            logger.info(f"Successfully initialized bot for business: {business_id}")
        except Exception as e:
            logger.error(f"Failed to initialize bot: {str(e)}")
            raise InitializationError(f"Failed to initialize bot: {str(e)}")

    def get_rag_response(self, query: str, chat_history: ChatHistory) -> Dict[str, Any]:
        """Generate a response using the RAG pipeline"""
        try:
            logger.info("Starting history chain processing...")
            if not self.history_aware_retriever:
                raise ChainError("History aware retriever not initialized")

            history_chain_response = self.history_aware_retriever.invoke({
                'input': query,
                'chat_history': chat_history
            })
            logger.info("History chain processing completed")

            logger.info("Starting QA chain processing...")
            if not self.question_answer_chain:
                raise ChainError("Question answer chain not initialized")

            qa_chain_response = self.question_answer_chain.invoke({
                'input': query,
                'context': history_chain_response,
                'chat_history': chat_history
            })
            logger.info("QA chain processing completed")

            return {
                'input': query,
                'answer': qa_chain_response,
                'chat_history': chat_history,
                'type': 'rag'
            }
        except: 
            pass
    

