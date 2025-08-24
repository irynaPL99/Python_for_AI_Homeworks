"""
1.Создайте простую цепочку LangChain для суммаризации текста из URL веб-страницы.
2.Зарегистрируйтесь на PromptHub и изучите его интерфейс.
Найдите и опишите 3 интересных промпта.
3.(Дополнительно) Создайте цепочку "вопрос-ответ по документам"
для небольшого текстового файла.
"""
# Function to make a chain. It puts documents together and sends them to LLM.
from langchain.chains.combine_documents import create_stuff_documents_chain

# Class to make prompt templates for chat.
from langchain_core.prompts import ChatPromptTemplate

# Class to use Google generative model.
from langchain_google_genai import ChatGoogleGenerativeAI

# Class to load documents (here – web pages).
from langchain_community.document_loaders import WebBaseLoader

# Module to work with environment variables (from .env file).
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Module to work with the operating system (for example, to get environment variables).
import os

# Go one folder up and load .env
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Load environment variables from .env file.
# We use this to get secret keys without writing them in the code.
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Start Google generative model with model "gemini-2.0-flash" and give API key.
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Make a loader to download content from a web page.
loader = WebBaseLoader("https://germanjob.info/kak_naiti_delo_zhizni")

# Load document from the web page.
# In "docs" we will have the text from the page.
docs = loader.load()

# Make a template for the prompt.
# Here {context} will be replaced with the document text.
prompt = ChatPromptTemplate.from_template("Write a short summary of this text: {context}")

# Make a chain. It puts the document and prompt together and sends them to LLM.
chain = create_stuff_documents_chain(llm, prompt)

# Run the chain with the document as "context".
# The function invoke sends input and gives back the result.
result = chain.invoke({"context": docs})

# Show the result (the short summary).
print(result)
