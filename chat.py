import os
import warnings

from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from google.genai.types import AutomaticFunctionCallingConfig  

# 1. Block internal SDK logs from reaching stdout
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# 2. Block system alerts and standard UserWarnings 
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

# Vector Embedding
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
     embedding= embedding_model,
     url = "http://localhost:6333",
     collection_name = "Building_RAG",
)

# take the user input
user_query = input("Hi! 👤, Ask Something: ")

# Relevant chunks from the vector db
search_results = vector_db.similarity_search(query= user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
                         for result in search_results])

SYSTEM_PROMPT = """
"You are an expert academic assistant. Use the following retrieved thesis context "
    "to answer the question. Retrive from the PDF file along with page_contents and pange number.If you don't know the answer, say you don't know.\n\n"

    You should only answer the used based on the following context and navigate the user to open the right page number to know more.
    "Context:\n{context}"

    """

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}")
])

# Setup LLM and invoke
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    
    )
messages = prompt_template.format_messages(context=context, question=user_query)

print("\n🤖 Thinking...")
response = llm.invoke(messages)

# Clean, Human-Readable Output Formatting
print("\n" + "="*50)
print("🎯 ANSWER FROM THESIS")
print("="*50)
# .text extracts ONLY the clean text, ignoring signatures and raw metadata arrays
print(response.text.strip()) 
print("="*50 + "\n")