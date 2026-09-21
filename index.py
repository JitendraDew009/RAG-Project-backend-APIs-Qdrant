from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path =Path(__file__).parent/"Nutri_Ninja_Thesis_Jitendra_Dewangan.pdf"

# Laod this file in pyhton program
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# print(docs[12])
# Chunking
# pip install -qU langchain-text-splitters

## Split the docs into smaller chunks
text_splitter =RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400 # so you don't lose context (basically you take chuck of previous part part so docs have context what all is about.)
)

chunks = text_splitter.split_documents(documents = docs)

# Next step is to create Vector embedding
# pip install -qU langchain-openai
# pip install -qU langchain-qdrant

# Vector Embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embedding_model,
    url = "http://localhost:6333",
    collection_name = "Building_RAG"

)

print("Index of documents done...")