from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Spliter:
    def __init__(self):
        self.chunk_size = 2000
        self.chunk_overlap = 200

    def split_pdf(self, filepath):
        # Đọc pdf
        loader = PyPDFLoader(filepath)
        loaded_documents = loader.load()
        # Tách văn bản (chunk)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap, 
            separators=["\n\n", "\n", " ", ""]
        )
        documents = text_splitter.split_documents(loaded_documents)
        return documents