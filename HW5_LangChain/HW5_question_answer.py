# Simple Q&A over one small file (no vectors, no embeddings)
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
import os

# Go one folder up and load .env
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# 1) Load env
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2) LLM
# Model is fast and cheap; good for small tasks
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# 3) Load small text file
loader = TextLoader("sample.txt", encoding="utf-8")
docs = loader.load()  # -> list of Document

# 4) Prompt (A2 English, short, force use of context only)
prompt = ChatPromptTemplate.from_template(
    "Use the context to answer the question. If you do not know, say 'I don't know'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

# 5) Build chain that stuffs docs into the prompt
chain = create_stuff_documents_chain(llm, prompt)

# 6) Ask a question
question = "What is the main idea of the text?"
#question = "Where is Hamburg?"
result = chain.invoke({"context": docs, "question": question})

# 7) Print answer
#print(result)
print("Question:", question)
print("Answer:", result)