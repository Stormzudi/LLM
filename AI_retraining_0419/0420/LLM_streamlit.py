import os
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama as OllamaLLM
import streamlit as st

# 加载本地PDF文档
loader = PyPDFLoader("123.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


# 创建 OllamaEmbeddings 和向量存储
embeddings = OllamaEmbeddings(model="gemma:2b")
docsearch = Chroma.from_documents(texts, embeddings)

# 创建问答链
qa_chain = RetrievalQA.from_chain_type(
    llm=OllamaLLM(model="gemma:2b"),
    chain_type="stuff",
    retriever=docsearch.as_retriever()
)


# Streamlit 应用程序
def main():
    st.title("Local PDF Document QA")

    # 对话历史记录
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 用户输入查询
    query = st.text_input("Enter your question about the document:")

    if query:
        # 执行问答
        result = qa_chain.run(query)

        # 将查询和答案添加到对话历史记录中
        st.session_state.chat_history.append({"query": query, "answer": result})

    # 显示对话历史记录
    for chat in st.session_state.chat_history:
        st.write(f"Question: {chat['query']}")
        st.write(f"Answer: {chat['answer']}")
        st.write("---")


if __name__ == "__main__":
    main()