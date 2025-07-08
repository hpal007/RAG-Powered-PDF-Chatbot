# RAG-Powered PDF & Text Chatbot

A simple yet powerful chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about the content of your uploaded PDF and text files.

![Chatbot in action](chatbot_demo.gif)

## 🌟 Features

- **Interactive Chat Interface**: A user-friendly interface built with Streamlit.
- **File Upload**: Supports both PDF (`.pdf`) and Text (`.txt`) files.
- **RAG-Powered Q&A**: Leverages a RAG pipeline to provide context-aware answers based on the document's content.
- **View Sources**: Displays the specific text chunks from the source document that were used to generate the answer.
- **Chat History**: Persists your conversation history locally and displays it on reload.
- **Clear Conversation**: Easily start a new session by clearing the chat history.

## ⚙️ How It Works

This application follows a standard Retrieval-Augmented Generation (RAG) architecture:

1. **File Ingestion & Processing**: When you upload a file, it is loaded and split into smaller, manageable chunks of text.
2. **Vector Embeddings**: Each text chunk is converted into a numerical representation (embedding) using a sentence-transformer model.
3. **Vector Store**: These embeddings are stored in a FAISS vector store, which allows for efficient similarity searches.
4. **Retrieval**: When you ask a question, your query is also embedded, and the vector store is searched to find the most relevant text chunks from the document.
5. **Generation**: The retrieved chunks are combined with your original question and sent as context to a Large Language Model (LLM), which generates a coherent and contextually accurate answer.

## 🛠️ Setup & Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd RAG-Powered-PDF-Chatbot
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3. **Install the required dependencies:**
    *(Note: A `requirements.txt` file should be created for this project. It would typically contain packages like `streamlit`, `langchain`, `openai`, `faiss-cpu`, `pypdf`, `python-dotenv`)*

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up Environment Variables:**
    If you are using Ollama with the `gemma3:4b` model, you typically do not need an external API key. However, ensure that Ollama is installed and running locally. If your setup requires any environment variables (e.g., for custom configurations), create a `.env` file in the root directory and add them as needed.

    Example `.env` file (if required):

    ```bash
    # Example: Set custom port for Ollama (optional)
    OLLAMA_HOST="http://localhost:11434"
    ```

    The application logic in `qabot.py` would need to be configured to load this variable.

## ▶️ How to Run

Once the setup is complete, run the Streamlit application from your terminal:

```bash
streamlit run main.py
```

This will start the application and open it in your default web browser.

1. Use the file uploader to select a PDF or TXT file.
2. Once the file is uploaded, ask a question in the chat input box.
3. The chatbot will provide an answer. You can expand the "Source Documents" section to see the context used for the answer.

## 💻 Technologies Used

- **Frontend**: Streamlit
- **RAG & LLM Orchestration**: LangChain
- **Vector Store**: Chroma
- **File Processing**: `pypdf` for PDFs.
- **LLM**: Ollama(`gemma3:4b`)
