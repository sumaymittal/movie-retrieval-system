import os
import sys
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add the cli directory to path so we can import the existing logic
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cli"))

# Try loading .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies
from lib.llm_utils import get_gemini_client, get_rag_nl_prompt, query_gemini

app = FastAPI(title="Movie Retrieval System API")

print("Loading dataset and initializing models...")
movies = load_movies()
hybrid_search = HybridSearch(movies)

try:
    gemini_client = get_gemini_client()
    print("Models loaded successfully.")
except AssertionError:
    print("WARNING: GEMINI_API_KEY not found. RAG functionality will be disabled.")
    gemini_client = None

@app.get("/api/search")
def search_movies(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)):
    # Perform RRF search
    results = hybrid_search.rrf_search(q, k=60, limit=limit)
    
    formatted_results = []
    for doc_id, res in results:
        formatted_results.append({
            "id": doc_id,
            "title": res.get("title", ""),
            "description": res.get("description", ""),
            "score": round(res.get("rrf", 0), 4),
        })
    return {"query": q, "results": formatted_results}

@app.get("/api/ask")
def ask_question(q: str = Query(..., min_length=1)):
    if not gemini_client:
        return {
            "query": q,
            "answer": "Error: GEMINI_API_KEY is not configured on the server. AI features are disabled.",
            "sources": []
        }

    # Retrieve top 5 context docs for RAG
    results = hybrid_search.rrf_search(q, k=60, limit=5)
    sys_prompt, contents = get_rag_nl_prompt(q, results)
    
    try:
        rag_response = query_gemini(gemini_client, sys_prompt, contents)
    except Exception as e:
        rag_response = f"Error generating answer: {str(e)}"
        
    formatted_sources = [{"title": r["title"]} for _, r in results]
    
    return {
        "query": q,
        "answer": rag_response,
        "sources": formatted_sources
    }

# Mount static files for the frontend
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static"), html=True), name="static")
