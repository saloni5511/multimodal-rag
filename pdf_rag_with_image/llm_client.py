import json
import os
import numpy as np
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Fix asyncio event loop issues
try:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        raise RuntimeError("Event loop is closed")
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)



# ---------- Constants ----------
TOP_K = 5
NORMALIZE = True

# ---------- Data structures ----------
@dataclass
class DocChunk:
    text: str
    metadata: Dict[str, Any]

class GeminiLLM:
    """Unified wrapper for both Gemini LLM and embeddings functionality."""
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.3, normalize: bool = NORMALIZE):
        # Load API key - check multiple possible locations
        possible_paths = [
            "../gemini_api_key.json",  # From rag_pipeline directory
            "gemini_api_key.json",     # From main directory
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "gemini_api_key.json")  # Absolute path
        ]
        
        api_key_file = None
        for path in possible_paths:
            if os.path.exists(path):
                api_key_file = path
                break
        
        if not api_key_file:
            raise FileNotFoundError(f"API key file not found. Tried: {possible_paths}")
        
        print(f"Using API key from: {api_key_file}")
        with open(api_key_file, 'r') as f:
            api_data = json.load(f)
            api_key = api_data.get('api_key')
        
        if not api_key:
            raise ValueError("API key not found in gemini_api_key.json")
        
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Initialize both LLM and embeddings
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.normalize = normalize
    
    def generate(self, prompt: str) -> str:
        """Generate text response for a given prompt."""
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.embeddings.embed_documents(texts)
        embs = np.array(embeddings, dtype=np.float32)
        
        if self.normalize:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return embs

