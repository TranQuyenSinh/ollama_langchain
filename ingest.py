import os
import time
from dotenv import load_dotenv
from langchain_chroma import Chroma
from uuid import uuid4
from spliter import Spliter
from models import Models

load_dotenv()

model = Models()
spliter = Spliter()

class Ingest:
    def __init__(self):
        self.vectorstore = Chroma(
            embedding_function=model.embeddings_ollama,
            persist_directory='./db',
        )

    def ingest_doc(self, filepath):
        # Skip non-PDF files
        if not filepath.lower().endswith('.pdf'):
            return
        # Đọc file pdf và tách văn bản thành các chunk
        documents = spliter.split_pdf(filepath)
        # Thêm uuid cho từng văn bản
        uuids = [str(uuid4()) for _ in range(len(documents))]
        # Thêm văn bản vào vector store
        self.vectorstore.add_documents(documents=documents,ids=uuids)

    def remove_doc(self, filepath):
        # tìm id của văn bản trong vector store
        ids = self.vectorstore.get(where={"source": filepath})['ids']
        if len(ids): 
            self.vectorstore.delete(ids)
            ids = self.vectorstore.get(where={"source": filepath})['ids']