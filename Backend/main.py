import requests
import faiss
import numpy as np
import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup, Comment
from transformers import AutoTokenizer, AutoModel
import torch

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
CHUNK_SIZE = 500  # Max tokens per chunk
TOP_K = 10        # Number of top results to return
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Models ---

class SearchRequest(BaseModel):
    """Pydantic model for the search request payload."""
    url: str
    query: str

class ChunkResult(BaseModel):
    """Pydantic model for a single search result."""
    chunk_content: str
    score: float
    token_start: int
    token_end: int

class SearchResponse(BaseModel):
    """Pydantic model for the overall search response."""
    results: List[ChunkResult]

# --- Global Components (Loaded Once) ---
# NOTE: This simulation uses in-memory model and index, which is re-created per run.
# For a real application, the model would be loaded once, and the index would be persistent (e.g., Milvus/Weaviate).
try:
    logging.info(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Sentence Transformer model: {e}")
    # Use a dummy function if model loading fails (e.g., during testing without internet)
    def dummy_encode(texts):
        logging.warning("Using dummy encoder. Semantic search will fail.")
        # Return random 384-dim vectors
        return np.random.rand(len(texts), 384).astype('float32')
    model = None
    tokenizer = None
    # Patch the function if model is null
    def encode_texts(texts):
        return dummy_encode(texts)
else:
    # Function to perform mean pooling for generating embeddings
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Function to encode texts
    def encode_texts(texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts using the loaded model."""
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform pooling. In this case, mean pooling.
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        # Convert to numpy and normalize
        return sentence_embeddings.cpu().numpy().astype('float32')


# --- Core Logic Functions ---

def fetch_and_clean_html(url: str) -> Optional[str]:
    """Fetches HTML and extracts clean text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements (scripts, styles, comments)
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.extract()
            
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Get the main text content, often from the body
        text = soup.body.get_text(separator=' ', strip=True)
        return text

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")
    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing HTML content: {e}")

def chunk_text_by_tokens(text: str) -> List[dict]:
    """Tokenizes text and splits it into chunks of up to CHUNK_SIZE tokens."""
    if not tokenizer:
        return [{"content": text[:1000], "start": 0, "end": len(text)}] # Fallback if tokenizer failed to load

    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)
    
    chunks_data = []
    current_chunk_tokens = []
    
    # Iterate through tokens and split into chunks
    for token in tokens:
        current_chunk_tokens.append(token)
        if len(current_chunk_tokens) >= CHUNK_SIZE:
            # Reconstruct the text from the tokens of the current chunk
            chunk_content = tokenizer.convert_tokens_to_string(current_chunk_tokens)
            
            # Simple simulation of token index range
            start_index = len(chunks_data) * CHUNK_SIZE
            end_index = start_index + len(current_chunk_tokens)
            
            chunks_data.append({
                "content": chunk_content,
                "start": start_index,
                "end": end_index
            })
            current_chunk_tokens = []

    # Add the last remaining chunk
    if current_chunk_tokens:
        chunk_content = tokenizer.convert_tokens_to_string(current_chunk_tokens)
        start_index = len(chunks_data) * CHUNK_SIZE
        end_index = start_index + len(current_chunk_tokens)
        chunks_data.append({
            "content": chunk_content,
            "start": start_index,
            "end": end_index
        })

    logging.info(f"Text chunked into {len(chunks_data)} segments.")
    return chunks_data

def semantic_search(chunks_data: List[dict], query: str) -> List[ChunkResult]:
    """
    Simulates semantic search using Faiss on the chunk embeddings.
    This replaces a call to an external vector database (Milvus/Weaviate/Pinecone).
    """
    if not chunks_data:
        return []
    
    # 1. Prepare data and embed all chunks
    chunk_contents = [data['content'] for data in chunks_data]
    try:
        logging.info("Generating embeddings for chunks...")
        chunk_embeddings = encode_texts(chunk_contents)
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        # Fallback to simple keyword search if embedding fails
        return keyword_search(chunks_data, query)

    dimension = chunk_embeddings.shape[1]
    
    # 2. Build the In-Memory Faiss Index (Simulating Vector DB Indexing)
    index = faiss.IndexFlatIP(dimension) # Use Inner Product for cosine similarity (vectors are normalized)
    index.add(chunk_embeddings)
    logging.info(f"Faiss index created with {index.ntotal} vectors of dimension {dimension}.")

    # 3. Embed the query
    query_embedding = encode_texts([query])
    # Faiss requires the query vector to be normalized for IP to approximate Cosine Similarity
    faiss.normalize_L2(query_embedding)

    # 4. Search the index
    D, I = index.search(query_embedding, TOP_K) # D is distances/scores, I is indices
    
    # 5. Format results
    results: List[ChunkResult] = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: # Faiss returns -1 if fewer than TOP_K results are found
            continue
            
        original_chunk = chunks_data[idx]
        
        results.append(ChunkResult(
            chunk_content=original_chunk['content'],
            score=float(score),
            token_start=original_chunk['start'],
            token_end=original_chunk['end']
        ))
        
    logging.info(f"Semantic search complete. Found {len(results)} matches.")
    return results

def keyword_search(chunks_data: List[dict], query: str) -> List[ChunkResult]:
    """A simple fallback keyword search to ensure functionality if model fails."""
    logging.warning("Falling back to simple keyword search.")
    matches = []
    query_lower = query.lower()
    
    for i, chunk in enumerate(chunks_data):
        if query_lower in chunk['content'].lower():
            # Assign a dummy score based on index for ordering stability
            matches.append(ChunkResult(
                chunk_content=chunk['content'],
                score=1.0 - (i / len(chunks_data) * 0.1), # High score, slightly varied
                token_start=chunk['start'],
                token_end=chunk['end']
            ))

    # Sort by the dummy score (highest first) and return top K
    matches.sort(key=lambda x: x.score, reverse=True)
    return matches[:TOP_K]


# --- FastAPI Application Setup ---

app = FastAPI(
    title="RAG Search Backend",
    description="Backend for fetching, chunking, and performing semantic search on web content."
)

# CORS middleware setup: Allow frontend running on a different port to connect
origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "*" # Allows all origins for local development simplicity
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Simple health check endpoint."""
    return {"message": "RAG Search Backend is running."}


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    """
    Main endpoint to process a URL and query, perform RAG-style search, 
    and return the top results.
    """
    logging.info(f"Received request for URL: {request.url} with Query: '{request.query}'")
    
    # 1. Fetch and Clean HTML
    clean_text = fetch_and_clean_html(request.url)
    if not clean_text:
        raise HTTPException(status_code=404, detail="Could not retrieve or clean content from the URL.")
    
    # 2. Tokenize and Chunk Content
    chunks_data = chunk_text_by_tokens(clean_text)
    if not chunks_data:
        raise HTTPException(status_code=500, detail="Failed to chunk the processed text.")
    
    # 3. Perform Semantic Search (using Faiss simulation)
    results = semantic_search(chunks_data, request.query)
    
    return SearchResponse(results=results)

# --- How to Run Locally ---
# To run this file:
# 1. Install dependencies: pip install "fastapi[all]" uvicorn requests beautifulsoup4 sentence-transformers faiss-cpu numpy
# 2. Run the server: uvicorn main:app --reload