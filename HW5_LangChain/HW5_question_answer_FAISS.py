"""
RAG-способ (эмбеддинги + поиск по кусочкам)

Надёжнее: файл режем на части, строим векторный индекс,
ищем релевантные куски и только их даём модели.
"""
# RAG Q&A over a small file (embeddings + FAISS retriever)
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os

# Go one folder up and load .env
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 1) Load env
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2) LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# 3) Load file
loader = TextLoader("sample.txt", encoding="utf-8")
docs = loader.load()

# 4) Split text into small chunks
#    Small chunks -> better search
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
splits = splitter.split_documents(docs)

# 5) Build embeddings + FAISS
emb = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
vectorstore = FAISS.from_documents(splits, emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # search top 3 chunks

# 6) Make prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the context. "
     "If you don't know, say 'I don't know'.\n\nContext:\n{context}"),
    ("human", "{question}")
])

# 7) Make the chain: LLM + retriever
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, doc_chain)

# 8) Ask a question
query = "What are some examples of AI use?"
#query = "Where is Hamburg?"

# 9) Run chain
#out = rag_chain.invoke({"question": query})
out = rag_chain.invoke({"input": query, "question": query})

# 10) Show result
print("Query:", query)
print("Answer:", out["answer"])