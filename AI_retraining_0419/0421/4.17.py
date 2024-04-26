from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama as OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, get_buffer_string
from langchain_core.prompts import format_document, ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents.base import Document
from operator import itemgetter

_template = """给定以下对话历史和一个后续问题,请将后续问题转述为一个独立的问题,使用原始语言。

对话历史:

{chat_history}

后续问题: {question}

独立问题:"""

CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template(_template)

template = """基于以下背景信息回答问题:

{context}

问题: {question}

"""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="gemma:2b")

vectorstore = FAISS.from_texts(
    ["Langchain是一个用于开发LLM应用的开源框架。它提供了许多有用的工具和组件,如提示模板、记忆体系、索引等,帮助开发者更轻松地构建LLM应用。"],
    embedding=OllamaEmbeddings(model="gemma:2b")
)

retriever = vectorstore.as_retriever()

_inputs = RunnableParallel(
    chat_history=RunnablePassthrough(),
    question=(
        {"question": RunnablePassthrough(), "chat_history": lambda x: get_buffer_string(x["chat_history"])}
        | CONDENSE_QUESTION_PROMPT
        | model
        | StrOutputParser()
    ),
)

_context = {
    "context": itemgetter("question") | retriever | (lambda docs: format_document(Document(page_content="\n".join(doc.page_content for doc in docs), metadata={"context": "\n".join(doc.page_content for doc in docs), "question": None}), prompt)),
    "question": lambda x: x["question"],
}

conversational_qa_chain = _inputs | _context | prompt | model

chat_history = [
    HumanMessage(content="Langchain是什么?"),
    AIMessage(content="Langchain是一个用于开发大语言模型应用的开源框架。"),
    HumanMessage(content="那么Langchain有哪些主要功能?"),
]

print(conversational_qa_chain.invoke({"question": "Langchain有哪些特点?", "chat_history": chat_history}))