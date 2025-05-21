import os
import chromadb
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from constants import contextualize_q_system_prompt, universal_assistant_prompt
from text_extractor import convert_to_text
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Access the value of the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
llm_model = ''

class ChromaDBManager:
    def __init__(self, persist_directory=None, llm_model_name="gpt-4o", embedding_model_name="text-embedding-3-large",
                 temperature=0, chunk_size=1000, chunk_overlap=200, seed=121, collection_name=None):
        """
        Initialize a persistent ChromaDB client with the given path.

        :param path: Directory to store the embeddings database.
        """
        # self.client = chromadb.PersistentClient(path=path)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.persistent_client = chromadb.PersistentClient(self.persist_directory)
        self.openai_ef = OpenAIEmbeddings(model=self.embedding_model_name, show_progress_bar=True, skip_empty=True)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # Initialize Language Model
        self.llm = ChatOpenAI(model=llm_model_name, temperature=temperature, seed=seed)
        self.prompt_template = """
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: 
            {question}

            Helpful Answer:
        """
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        self.contextualize_q_prompt = self.create_contextualize_prompt()
        self.qa_prompt = self.create_qa_prompt()


        #initialize rag chains
        self.load_embeddings()
        self.initialize_chains()
        
        #
        self.session = {}

        # Initialize total_chunks attribute
        # self.total_chunks = 0

    def initialize_chains(self):
        self.history_aware_retriever = self._create_history_aware_retriever()
        self.question_answer_chain = self._create_question_answer_chain()
        self.rag_chain = self._create_rag_chain()

    def load_embeddings(self):
        '''
        '''
        persist_directory = self.persist_directory
        collection_name = self.collection_name
        self.vectorstore = self.load_saved_embeddings(collection_name, persist_directory)
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5, 'fetch_k': 50, 'lambda_mult': 0.5, 'filter': {'deleted': False}}) #by default search_type="mmr"

    def create_contextualize_prompt(self):
        '''
        '''
        return ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def _create_history_aware_retriever(self):
        return create_history_aware_retriever(self.llm, self.retriever, self.contextualize_q_prompt)

    def create_qa_prompt(self):
        '''
        '''
        return ChatPromptTemplate.from_messages(
            [
                ("system", universal_assistant_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    def _create_question_answer_chain(self):
        return create_stuff_documents_chain(self.llm, self.qa_prompt)

    def _create_rag_chain(self):
        return create_retrieval_chain(self.history_aware_retriever, self.question_answer_chain)

    def create_collection(self, collection_name=None):
        """
        Create a new collection in the database.

        :param collection_name: Name of the collection to be created.
        """
        # self.client.create_collection(name=collection_name)
        collection_metadata = {"hnsw:space": "cosine"}
        if collection_name:
            self.persistent_client.get_or_create_collection(name=collection_name
                                                            )
            print(f"Collection '{collection_name}' created.")

    def list_collections(self):
        """
        List all collections in the database.

        :return: A list of collection names.
        """
        collections = self.persistent_client.list_collections()
        colls = [collection.name for collection in collections]
        print(f"Collections list: {colls}")
        # return colls

    def get_chroma(self, collection_name=None, persist_directory=None):
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.openai_ef,
            persist_directory=self.persist_directory,
            client=self.persistent_client,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def generate_embeddings(self, documents, collection_name=None):
        """
        """
        print(f"Adding documents to collection: {collection_name}")
        if documents and collection_name:
            chroma_obj = self.get_chroma(collection_name, persist_directory=self.persist_directory)
            chroma_obj.from_documents(documents=documents,
                                       embedding=self.openai_ef,
                                       collection_name=self.collection_name,
                                       persist_directory=self.persist_directory)

    # def generate_embeddings_from_chunks(self, documents, collection_name=None):
    # def generate_embeddings_from_chunks(self, documents, collection_name):
    def generate_embeddings_from_chunks(self, text_chunks, collection_name, fileId):
        """
        Convert text chunks to embeddings and store in ChromaDB
        
        :param text_chunks: List of text chunks
        :param collection_name: Name of the ChromaDB collection
        :return: Chroma vector store with embedded documents
        """
        try:

            # Update total chunks
            #self.total_chunks += len(text_chunks)
            #print(f"Total chunks processed: {self.total_chunks}")

            # Convert text chunks to Document objects
            # Generate simple IDs for chunks
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "source": collection_name,
                        "fileId": fileId,
                        "chunkId": f"{fileId}_{i}",  # Simple ID: fileId + index
                        "deleted": False  # Soft delete flag
                    }
                )
                for i, chunk in enumerate(text_chunks)
            ]
            chroma_obj = self.get_chroma(collection_name, persist_directory=self.persist_directory)
            
            # Create Chroma vector store
            vector_store = chroma_obj.from_documents(
                documents=documents,
                embedding=self.openai_ef,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )
            
            print(f"Successfully converted {len(text_chunks)} of {fileId} chunks to embeddings")
            return vector_store
        
        except Exception as e:
            print(f"Error converting chunks to embeddings: {e}")
            return None




    def genrate_text_chunks(self,dir, fileId) : 
        text= convert_to_text(dir)
        chunks = self.text_splitter.split_text(text)
        self.generate_embeddings_from_chunks(text_chunks=chunks, collection_name=self.collection_name, fileId=fileId)

    def load_saved_embeddings(self, collection_name=None, persist_directory=None):
        '''
        :param persist_directory:
        :return:
        '''
        chroma_obj = self.get_chroma(collection_name, persist_directory=self.persist_directory)
        return chroma_obj

    def delete_collection(self, collection_name):
        """
        Delete a collection from the database.

        :param collection_name: Name of the collection to delete.
        """
        self.persistent_client.delete_collection(name=collection_name)
        print(f"Collection '{collection_name}' deleted.")


    def create_qa_chain(self, collection_name):
        """
        Create a Question-Answering chain
        
        :param collection_name: Name of the ChromaDB collection
        :return: RetrievalQA chain
        """
        # Load vector store
        vector_store =  self.get_chroma(collection_name, persist_directory=self.persist_directory)
        
        if not vector_store:
            print("Could not load vector store.")
            return None
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_kwargs={'k': 3}  # Retrieve top 3 most relevant documents
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={'prompt': self.prompt}
        )
        
        return qa_chain  
    
    def ask_question1(self, collection_name, question):
        """
        Ask a question to the document collection
        
        :param collection_name: Name of the ChromaDB collection
        :param question: Question to ask
        :return: Answer and source documents
        """
        try:
            # Create QA chain
            qa_chain = self.create_qa_chain(collection_name)
            
            if not qa_chain:
                return None
            
            # q = self.generate_openai_embeddings(question)
            
            # Run the query
            result = qa_chain.invoke({"query": question})
            return {
                "answer": result['result'],
                "source_documents": result['source_documents']
            }
        
        except Exception as e:
            print(f"Error processing question: {e}")
            return None
    
    def soft_delete_document(self, file_id, collection_name, deleted):
        """
        Soft delete a document by marking all its chunks as deleted in the metadata.
        Uses batch update for better performance.
        
        :param file_id: The ID of the file to soft delete
        :param collection_name: Name of the collection
        :param deleted: Boolean flag indicating whether to mark as deleted or not
        :return: Number of chunks updated
        """
        try:
            # Get the Chroma collection
            chroma_collection = self.persistent_client.get_collection(name=collection_name)
            
            # Get all documents with matching file_id and current deletion status
            where_clause = {
                "$and": [
                    {"fileId": {"$eq": file_id}},
                    {"deleted": {"$eq": not deleted}}
                ]
            }
            
            # Get all matching documents in a single query
            results = chroma_collection.get(
                where=where_clause
            )
            
            if not results or not results['ids']:
                print(f"No documents found with file_id: {file_id}")
                return 0
                
            # Create updated metadata for all documents at once
            updated_metadata = [{**meta, "deleted": deleted} for meta in results['metadatas']]
            
            # Batch update all documents
            chroma_collection.update(
                ids=results['ids'],
                metadatas=updated_metadata
            )
            
            action = 'deleted' if deleted else 'restored'
            print(f"Successfully {action} {len(results['ids'])} chunks for file_id: {file_id}")
            return len(results['ids'])
        
        except Exception as e:
            print(f"Error during soft delete: {e}")
            raise

    def hard_delete_document(self, file_id, collection_name):
        """
        Permanently delete a document and all its chunks from ChromaDB.
        This operation cannot be undone.
        
        :param file_id: The ID of the file to delete
        :param collection_name: Name of the collection
        :return: Number of chunks deleted
        """
        try:
            # Get the Chroma collection
            chroma_collection = self.persistent_client.get_collection(name=collection_name)
            
            # Get all documents with matching file_id
            where_clause = {
                "fileId": {"$eq": file_id}
            }
            
            # Get all matching document IDs
            results = chroma_collection.get(
                where=where_clause
            )
            
            if not results or not results['ids']:
                print(f"No documents found with file_id: {file_id}")
                return 0
            
            # Delete all chunks in a single operation
            chroma_collection.delete(
                ids=results['ids']
            )
            
            print(f"Successfully hard deleted {len(results['ids'])} chunks for file_id: {file_id}")
            return len(results['ids'])
        
        except Exception as e:
            print(f"Error during hard delete: {e}")
            raise


    def ask_question(self, query, session_id):  
        try:
            if session_id not in self.session:
                self.session[session_id] = []  # Initialize an empty session
            chat_history = self.session[session_id]   

            history_aware_retriever_respone = self.history_aware_retriever.invoke({'input': query, 'chat_history': chat_history})
            qa_chain_response =  self.question_answer_chain.invoke({'input': query, 'context': history_aware_retriever_respone, 'chat_history': chat_history})
            rag_chain_response = self.rag_chain.invoke({'input': query, 'context': qa_chain_response, 'chat_history': chat_history})
            # print('this is history',history_aware_retriever_respone,'\n\n\n')
            # print('this is chain response', qa_chain_response,'\n\n\n')
            # print('this is rag chain',rag_chain_response,'\n\n\n')
            answer = rag_chain_response['answer']
            self.session[session_id].append(HumanMessage(content=query))
            self.session[session_id].append(AIMessage(content=answer))
            return rag_chain_response
        
        except Exception as e:
            print(f"Error processing question: {e}")
            return None  
        
