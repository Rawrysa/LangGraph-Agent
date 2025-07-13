"""
retriever.py
------------
Handles document ingestion, chunking, vector database creation, and semantic retrieval using FAISS and HuggingFace embeddings.
- Supports .txt, .pdf, and .docx files.
- Adds metadata to each chunk for traceability.
- Provides methods for saving documents, creating indexes, and retrieving information.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader
from langgraph.config import get_stream_writer
from logger import setup_logger

class Retriever:
    """
    Handles document ingestion, chunking, vector database creation, and semantic retrieval.
    """
    def __init__(self, file_name: str = None, index_name: str = None):
        """
        Initialize the Retriever. If a file_name is provided, loads or creates the vector DB retriever for that file.
        """
        self.logger = setup_logger(__name__, 'retriever.log')

        if file_name and index_name:
            self.retriever = self.get_vector_db_retriever(index_name, file_name)

        self.logger.info('Retriever initialized with vector database.')

    def _create_faiss_index(self, index_name: str, file_name: str):
        """
        Internal helper to create a FAISS index for a given document file, with metadata.
        - Splits the document into overlapping chunks.
        - Adds metadata (file name, type, path) to each chunk.
        - Saves the FAISS index for future retrieval.
        """
        try:
            persist_path = os.path.join('./data', "faiss_index")

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            index_name = index_name.lower()
            doc_path = os.path.join("./docs", f"{file_name}")
            file_extension = file_name.split('.')[-1].lower()

            if file_extension == 'txt':
                doc_loader = TextLoader(doc_path, encoding="utf-8")
                documents = doc_loader.load()

            else:
                if file_extension == 'pdf':
                    doc_loader = UnstructuredPDFLoader(doc_path)
                else:
                    doc_loader = UnstructuredWordDocumentLoader(doc_path)
                documents = doc_loader.load()

            if not documents:
                self.logger.error('No documents found in the specified path.')
                raise ValueError("No documents found in the specified path.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=300,
                separators=["\n\n", "\n", " "]
            )

            texts = text_splitter.split_documents(documents)

            for chunk in texts:
                if not hasattr(chunk, 'metadata') or chunk.metadata is None:
                    chunk.metadata = {}
                chunk.metadata.update({
                    'source_file': file_name,
                    'file_extension': file_extension,
                    'doc_path': doc_path
                })

            index_file = os.path.join(persist_path, f"{index_name}.faiss")

            if os.path.exists(index_file):
                self.logger.info('Updating existing FAISS index for %s.', file_name)
                vectorstore = FAISS.load_local(persist_path, embedding_model, index_name=index_name, allow_dangerous_deserialization=True)
                vectorstore.add_documents(texts)
            else:
                self.logger.info('Creating new FAISS index for %s.', file_name)
                vectorstore = FAISS.from_documents(texts, embedding_model)

            vectorstore.save_local(persist_path, index_name=f"{index_name}")
            self.logger.info('FAISS index created/updated and saved for %s.', file_name)

        except Exception as e:
            self.logger.error('Error creating FAISS index for %s: %s', file_name, e, exc_info=True)
            raise RuntimeError(f"Failed to create FAISS index for {file_name}: {e}")

    def get_vector_db_retriever(self, index_name: str, file_name: str):
        """
        Loads a vector database retriever using FAISS and HuggingFace embeddings.
        Only loads an existing index; does not create a new one on the fly.
        Returns a retriever object for semantic search.
        """
        try:
            self.logger.info('Initializing vector DB retriever.')

            persist_path = os.path.join('./data', "faiss_index")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            index_name = index_name.lower()
            index_file = os.path.join(persist_path, f"{index_name}.faiss")

            if os.path.exists(index_file):
                self.logger.info('Loading existing FAISS index.')
                vectorstore = FAISS.load_local(persist_path, embedding_model, index_name=index_name, allow_dangerous_deserialization=True)
                return vectorstore.as_retriever(search_kwargs={"k": 6, "search_filter": {"file_extension": file_name.split('.')[-1].lower(), "source_file": file_name}})
            
            else:
                self.logger.error('FAISS index not found for %s. Please save the document first.', file_name)
                raise FileNotFoundError(f"FAISS index not found for {file_name}. Please save the document first.")
            
        except Exception as e:
            self.logger.error('Error initializing vector DB retriever: %s', e, exc_info=True)
            raise RuntimeError(f"Failed to initialize vector DB retriever: {e}")

    def get_info(self, query):
        """
        Retrieve information based on the query using the vector database.
        Returns the most relevant chunks and their metadata.
        """
        self.logger.info('Retrieving information for query: %s', query)
        get_stream_writer()({"get info": f"😒 Retrieving information for query: {query}"})
        return self.retriever.invoke(query)
    
    def save_doc(self, file):
        """
        Save a document to the local file system and create a FAISS index for it.
        If the document already exists, does not overwrite it.
        """
        self.logger.info('Saving document: %s', file.name)

        persist_path = os.path.join('./docs')
        doc_path = os.path.join(persist_path, file.name)

        if not os.path.exists(doc_path):

            with open(doc_path, "wb") as f:
                f.write(file.getbuffer())

            self.logger.info('Document saved at: %s', doc_path)
            
            self._create_faiss_index("docs", file.name)
            return doc_path
        
        else:
            self.logger.warning('Document already exists at: %s', doc_path)
            
            return doc_path
