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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CHUNK_SIZE = 500
TOP_K = 10
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

try:
    logging.info(f"Loading tokenizer and model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading Sentence Transformer model: {e}")
    def dummy_encode(texts):
        logging.warning("Using dummy encoder. Semantic search will fail.")
        return np.random.rand(len(texts), 384).astype('float32')
    model = None
    tokenizer = None
    def encode_texts(texts):
        return dummy_encode(texts)
else:
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_texts(texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of texts using the loaded model."""
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings.cpu().numpy().astype('float32')



def fetch_and_clean_html(url: str) -> Optional[str]:
    """Fetches HTML and extracts clean text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.extract()
            
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

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

    tokens = tokenizer.tokenize(text)
    
    chunks_data = []
    current_chunk_tokens = []
    
    for token in tokens:
        current_chunk_tokens.append(token)
        if len(current_chunk_tokens) >= CHUNK_SIZE:
            chunk_content = tokenizer.convert_tokens_to_string(current_chunk_tokens)
            
            start_index = len(chunks_data) * CHUNK_SIZE
            end_index = start_index + len(current_chunk_tokens)
            
            chunks_data.append({
                "content": chunk_content,
                "start": start_index,
                "end": end_index
            })
            current_chunk_tokens = []

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
    
    chunk_contents = [data['content'] for data in chunks_data]
    try:
        logging.info("Generating embeddings for chunks...")
        chunk_embeddings = encode_texts(chunk_contents)
    except Exception as e:
        logging.error(f"Embedding generation failed: {e}")
        return keyword_search(chunks_data, query)

    dimension = chunk_embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    index.add(chunk_embeddings)
    logging.info(f"Faiss index created with {index.ntotal} vectors of dimension {dimension}.")

    query_embedding = encode_texts([query])
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, TOP_K)
    
    results: List[ChunkResult] = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
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
            matches.append(ChunkResult(
                chunk_content=chunk['content'],
                score=1.0 - (i / len(chunks_data) * 0.1), 
                token_start=chunk['start'],
                token_end=chunk['end']
            ))

    matches.sort(key=lambda x: x.score, reverse=True)
    return matches[:TOP_K]



app = FastAPI(
    title="RAG Search Backend",
    description="Backend for fetching, chunking, and performing semantic search on web content."
)

origins = [
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "*" 
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
    
    clean_text = fetch_and_clean_html(request.url)
    if not clean_text:
        raise HTTPException(status_code=404, detail="Could not retrieve or clean content from the URL.")
    
    chunks_data = chunk_text_by_tokens(clean_text)
    if not chunks_data:
        raise HTTPException(status_code=500, detail="Failed to chunk the processed text.")
    
    results = semantic_search(chunks_data, request.query)
    
    return SearchResponse(results=results)
