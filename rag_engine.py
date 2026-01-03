import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# 1. The Ingestion System (Unchanged)
def process_pdf_to_vector_db(file_path):
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")
        
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vector_db/faiss_index")
    return len(chunks)

# 2. The Modern "LCEL" Retrieval System (No 'chains' import needed!) 🚀
def answer_query(user_question):
    # Check if DB exists
    if not os.path.exists("vector_db/faiss_index"):
        return "⚠️ Memory not found. Please upload a PDF first."

    # Load the Database
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local("vector_db/faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever()
    
    # Define the "Brain" (LLM)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Define the Prompt Template (Instructions for the AI)
    template = """You are a Financial Analyst assistant. 
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know. 
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Build the Chain (The Modern Way)
    # This connects: Retriever -> Prompt -> LLM -> String Output
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Run it
    return rag_chain.invoke(user_question)