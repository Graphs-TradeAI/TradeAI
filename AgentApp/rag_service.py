import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAG', 'chroma_db')
        self._embeddings = None
        self._vectorstore = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info("Loading RAG Embeddings (HuggingFace)...")
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return self._embeddings

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            if os.path.exists(self.persist_directory):
                logger.info("Loading RAG ChromaDB from %s...", self.persist_directory)
                from langchain_chroma import Chroma
                self._vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                logger.warning("RAG ChromaDB directory not found.")
        return self._vectorstore

    def get_context(self, query: str, k: int = 3) -> str:
        vstore = self.vectorstore
        if not vstore:
            return ""
        
        try:
            results = vstore.similarity_search(query, k=k)
            context = "\n---\n".join([doc.page_content for doc in results])
            return context
        except Exception as e:
            logger.error("Error retrieving context from RAG: %s", e)
            return ""
