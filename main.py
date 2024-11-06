from fastapi import FastAPI, UploadFile, File
import uvicorn
from chromadb import Client as ChromaDatabase
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
from typing import List
import logging
import uuid

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
log_manager = logging.getLogger(__name__)

# Load SentenceTransformer model (CPU)
try:
    embedding_generator = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    log_manager.info("SentenceTransformer model loaded successfully.")
except Exception as model_loading_exception:
    log_manager.error(f"Failed to load model: {str(model_loading_exception)}")
    raise model_loading_exception

# Configure ChromaDB client for persistence
try:
    chroma_database = ChromaDatabase()
    doc_collection = chroma_database.get_or_create_collection(name="text_documents")
    log_manager.info("ChromaDB client initialized and collection created.")
except Exception as db_init_exception:
    log_manager.error(f"ChromaDB initialization error: {str(db_init_exception)}")
    raise db_init_exception

@app.post("/ingest/", response_class=JSONResponse)
async def add_documents(files: List[UploadFile] = File(...)):
    """ Endpoint to ingest files for document retrieval """
    doc_data = []
    embedding_list = []
    document_ids = []
    
    try:
        # Process files and prepare for ingestion
        for file in files:
            try:
                file_content = await file.read()
                decoded_content = file_content.decode('utf-8')
                unique_doc_id = str(uuid.uuid4())
                document_record = {"text": decoded_content, "metadata": {'filename': file.filename}}
                doc_data.append(document_record)
                document_ids.append(unique_doc_id)
                log_manager.info(f"File '{file.filename}' processed successfully.")

            except UnicodeDecodeError:
                log_manager.error(f"Failed to decode '{file.filename}'. Unsupported encoding.")
                return JSONResponse(content={"error": f"Cannot decode '{file.filename}'."}, status_code=400)
            except Exception as file_processing_error:
                log_manager.error(f"File processing error for '{file.filename}': {str(file_processing_error)}")
                return JSONResponse(content={"error": f"File error: {str(file_processing_error)}"}, status_code=500)

        # Generate embeddings for documents
        try:
            embedding_list = [embedding_generator.encode(doc["text"]).tolist() for doc in doc_data]
            log_manager.info("Embeddings generated successfully.")
        except Exception as embedding_error:
            log_manager.error(f"Error generating embeddings: {str(embedding_error)}")
            return JSONResponse(content={"error": f"Embedding error: {str(embedding_error)}"}, status_code=500)

        # Add documents to ChromaDB
        try:
            doc_collection.add(ids=document_ids, documents=[doc["text"] for doc in doc_data], 
                               metadatas=[doc["metadata"] for doc in doc_data], embeddings=embedding_list)
            log_manager.info("Documents successfully added to ChromaDB.")
        except Exception as storage_error:
            log_manager.error(f"Error storing documents in database: {str(storage_error)}")
            return JSONResponse(content={"error": f"Database error: {str(storage_error)}"}, status_code=500)

        return JSONResponse(content={"status": "Documents ingested successfully"})

    except Exception as ingestion_error:
        log_manager.error(f"Unexpected error during ingestion: {str(ingestion_error)}")
        return JSONResponse(content={"error": f"Server Error: {str(ingestion_error)}"}, status_code=500)

@app.get("/query/", response_class=JSONResponse)
async def search_documents(query_text: str):
    """ Endpoint to retrieve documents based on a query """
    try:
        # Generate embedding for the query
        query_embedding = embedding_generator.encode(query_text).tolist()
        log_manager.info("Query embedding generated successfully.")
        
        # Query ChromaDB
        query_results = doc_collection.query(query_embeddings=[query_embedding], n_results=5)
        search_response = [
            {
                "filename": doc_metadata.get('filename', 'unknown') if isinstance(doc_metadata, dict) else 'unknown',
                "score": doc_score,
                "text": retrieved_doc
            }
            for doc_metadata, doc_score, retrieved_doc in zip(query_results['metadatas'], query_results['distances'], query_results['documents'])
        ]
        log_manager.info("Query executed successfully.")
        return JSONResponse(content={"results": search_response})
    
    except Exception as query_exception:
        log_manager.error(f"Error during query execution: {str(query_exception)}")
        return JSONResponse(content={"error": f"Server Error: {str(query_exception)}"}, status_code=500)

@app.get("/database/", response_class=JSONResponse)
async def view_documents():
    """ Endpoint to view all stored documents """
    try:
        all_documents = doc_collection.get()
        doc_response = [
            {
                "filename": doc_meta.get('filename', 'unknown') if isinstance(doc_meta, dict) else 'unknown',
                "text": doc_text
            }
            for doc_meta, doc_text in zip(all_documents['metadatas'], all_documents['documents'])
        ]
        log_manager.info("Successfully retrieved all documents.")
        return JSONResponse(content={"documents": doc_response})
    except Exception as retrieval_exception:
        log_manager.error(f"Error retrieving documents: {str(retrieval_exception)}")
        return JSONResponse(content={"error": f"Server Error: {str(retrieval_exception)}"}, status_code=500)

if __name__ == "__main__":
    # Run the FastAPI app with live-reload enabled
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)