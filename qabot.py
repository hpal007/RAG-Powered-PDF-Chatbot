from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def document_loader(file):
    try:
        filename = file.name
        file_size = file.size if hasattr(file, 'size') else 'unknown'
        print(f"Uploaded file name: {filename}, size: {file_size}")

        # Determine file type and load accordingly
        if file.name.lower().endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
        elif file.name.lower().endswith(".txt"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            loader = TextLoader(tmp_file_path)
            documents = loader.load()
        else:
            logging.warning(f"Unsupported file type: {file.name}")
            return None

        logging.info(f"Successfully loaded {len(documents)} documents from {filename}")
        return documents
    except Exception as e:
        logging.error(f"Error loading file {file.name}: {e}")
        return None

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

## Vector db
def vector_database(chunks):
    if chunks:
        embedding_model = huggingface_embedding()
        vectordb = Chroma.from_documents(chunks, embedding_model)
        logging.info(f"Successfully created vector database with {len(chunks)} chunks.")
        return vectordb
    return None


def huggingface_embedding():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    logging.info("Successfully initialized HuggingFaceEmbeddings.")
    return embedding_model


## Retriever
def retriever(file):
    if file:
        splits = document_loader(file)
        if splits:
            chunks = text_splitter(splits)
            vectordb = vector_database(chunks)
            if vectordb:
                retriever = vectordb.as_retriever()
                logging.info("Successfully created retriever.")
                return retriever
            else:
                logging.warning("Vector database not created.")
                return None
        else:
            logging.warning("Documents not loaded.")
            return None
    else:
        logging.warning("No file provided for retrieval.")
        return None


## QA Chain
def retriever_qa(file, query):
    if file:
        retriever_obj = retriever(file)
        if retriever_obj:
            try:
                qa = RetrievalQA.from_chain_type(llm=ChatOllama(model="gemma3:4b"), 
                                        chain_type="stuff", 
                                        retriever=retriever_obj, 
                                        return_source_documents=True)
                response = qa.invoke(query)
                logging.info("Successfully invoked QA chain.")
                return response
            except Exception as e:
                logging.error(f"An error occurred during question answering: {str(e)}")
                return {"result": f"An error occurred during question answering: {str(e)}"}
        else:
            logging.warning("Retriever not initialized.")
            return {"result": "No documents loaded or retriever set up."}
    else:
        logging.warning("No file provided for question answering.")
        return {"result": "No documents loaded or retriever set up."}