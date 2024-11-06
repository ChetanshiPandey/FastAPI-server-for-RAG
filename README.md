FastAPI server implementation for document retrieval and embedding, enabling efficient Retrieval-Augmented Generation (RAG) with optimized performance.

This repository provides a FastAPI server tailored for Retrieval-Augmented Generation (RAG), facilitating the ingestion and querying of documents. Using ChromaDB as a vector database, the server supports multiple file formats, including PDF, DOC, DOCX, and TXT. Document embeddings are generated with the sentence-transformers/all-MiniLM-L6-v2 model, optimized for CPU-based operations. The server ensures efficient, non-blocking API endpoints to handle multiple requests concurrently.

Features

Document Storage & Retrieval: Supports uploading and searching documents across various formats with ChromaDB.
High-Quality Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 for meaningful text embeddings.
Asynchronous API: FastAPI’s non-blocking design allows for efficient handling of concurrent requests.
Technology Stack

## Supported File Formats
- **PDF**: Text is extracted using `PyMuPDF`.
- **DOCX**: Text is extracted using `python-docx`.
- **TXT**: Plain text files are supported.

FastAPI: Framework for building high-performance APIs.
ChromaDB: Vector database for storing and querying document embeddings.
Sentence-Transformers: Model for creating text embeddings.
Python: Core programming language.
Uvicorn: ASGI server for FastAPI application deployment.
Core Libraries and Tools

FastAPI: High-performance framework for API development.
Uvicorn: ASGI server optimized for FastAPI.
ChromaDB: Vector storage solution for managing embeddings.
Sentence-Transformers: Library for creating embeddings with transformer models.
Langchain: For handling various document formats.
Python Standard Libraries: Includes uuid for unique ID generation and logging for server monitoring.
Getting Started

Prerequisites
Python 3.8+
pip for installing dependencies
Installation
Clone the Repository
git clone https://github.com/<username>/fastapi-document-server.git
cd fastapi-document-server
Install Dependencies
pip install -r requirements.txt
Start the Server
uvicorn main:app --reload
Access the server at http://127.0.0.1:8000.
API Endpoints

1. /ingest/ [POST]
Endpoint to upload documents for storage and later retrieval.

Request: Multipart form-data containing files.
Example Files: document1.txt, document2.pdf
Sample Response:
{ "status": "Documents successfully ingested." }
2. /query/ [GET]
Search through stored documents using a text query.

Parameter: query (str) - Search text.
Example URL: http://127.0.0.1:8000/query/?query=Explain FastAPI
Sample Response:
{
  "results": [
    {
      "filename": "document1.txt",
      "score": 0.821,
      "text": "Introduction to FastAPI: FastAPI is a modern, high-performance web framework..."
    }
  ]
}
3. /database/ [GET]
Retrieve metadata and text of all stored documents.

Sample Response:
{
  "documents": [
    { "filename": "document1.txt", "text": "Content of the document goes here..." },
    { "filename": "document2.pdf", "text": "Sample content of another document..." }
  ]
}
Running the Server

Launch the Server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Access the server at http://localhost:8000.
Testing Endpoints
Use Postman, Thunder Client, or a web browser to test endpoints.
Usage Examples

Uploading Documents
Upload documents via a POST request to /ingest/.

Endpoint URL: http://localhost:8000/ingest/
Method: POST with files in form-data.
Searching Documents
Send a GET request to /query/ with a search query.

Endpoint URL: http://localhost:8000/query/?query=<your_search_text>
Contributing

Contributions are welcome! Feel free to submit a Pull Request.


License

This project is licensed under the MIT License.

Acknowledgements

FastAPI
ChromaDB
Sentence-Transformers
Langchain
