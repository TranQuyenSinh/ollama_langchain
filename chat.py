from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from models import Models

top_k = 2
model = Models()
# prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("ai", """
#                 Bạn là một chatbot tư vấn pháp luật dựa trên tài liệu được cung cấp. 
#                 Trả lời một cách ngắn gọn, tập trung vào trọng tâm câu hỏi
#                 Hãy trả lời câu hỏi dựa trên toàn bộ nội dung trên, không bỏ sót bất kỳ phần nào. Nếu nội dung dài, vui lòng tiếp tục trả lời cho đến hết.
#                 """),
#                 ("human", "Sử dụng câu hỏi {input} để tìm kiếm thông tin trong tài liệu. Chỉ sử dụng ngữ cảnh {context} để trả lời câu hỏi"),
#             ]
#         )bao nhiêu tuổi thì được ứng cử
prompt = ChatPromptTemplate.from_messages([
    ("system", "Bạn là một chatbot hữu ích có thể trả lời câu hỏi dựa trên tài liệu được cung cấp. Chỉ sử dụng ngữ cảnh được cung cấp để trả lời câu hỏi. QUAN TRỌNG: Nếu bạn không chắc chắn về câu trả lời, hãy nói 'Tôi không biết' và không trả lời bừa bãi."),
    ("human", "Câu hỏi: {input}\nNgữ cảnh: {context}")
])

class Chat:
    def __init__(self):
        self.vectorstore = Chroma(
            embedding_function=model.embeddings_ollama,
            persist_directory='./db',
        )
        self.memory = ConversationBufferMemory(llm=model.model_ollama, memory_key="chat_history", return_messages=True)

    def ask(self, query):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        combine_doc_chain = create_stuff_documents_chain(
            model.model_ollama, 
            prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_doc_chain)
        result = retrieval_chain.invoke({
            "input": query,
            "chat_history": self.memory.load_memory_variables({}).get("chat_history", [])
        })

        # Save chat history
        self.memory.save_context({"input": query}, {"output": result['answer']})
        return result['answer']