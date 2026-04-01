import os
import logging
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.persist_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAG', 'chroma_db')
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self._initialize_db()

    def _initialize_db(self):
        try:
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info("RAG ChromaDB initialized from %s", self.persist_directory)
            else:
                logger.warning("RAG ChromaDB directory not found at %s", self.persist_directory)
        except Exception as e:
            logger.error("Error initializing ChromaDB: %s", e)

    def get_context(self, query: str, k: int = 3) -> str:
        if not self.vectorstore:
            return ""
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            context = "\n---\n".join([doc.page_content for doc in results])
            return context
        except Exception as e:
            logger.error("Error retrieving context from RAG: %s", e)
            return ""
