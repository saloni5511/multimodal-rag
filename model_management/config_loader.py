"""Configuration loader for the RAG system."""

import json
import os

class Config:
    """Configuration class for the RAG system."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # config.json is now in the same directory as config_loader.py
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        with open(config_path, 'r') as f:
            self._config = json.load(f)
    
    # Chunking settings
    @property
    def chunk_size(self) -> int:
        return self._config["chunking"]["chunk_size"]
    
    @property
    def overlap(self) -> int:
        return self._config["chunking"]["overlap"]
    
    # Retrieval settings
    @property
    def top_k(self) -> int:
        return self._config["retrieval"]["top_k"]
    
    @property
    def normalize(self) -> bool:
        return self._config["retrieval"]["normalize"]
    
    # LLM settings
    @property
    def use_claude(self) -> bool:
        return self._config["llm"]["use_claude"]
    
    @property
    def aws_region(self) -> str:
        return self._config["llm"]["aws_region"]
    
    # Gemini settings
    @property
    def gemini_model(self) -> str:
        return self._config["llm"]["gemini"]["model"]
    
    @property
    def gemini_embedding_model(self) -> str:
        return self._config["llm"]["gemini"]["embedding_model"]
    
    # Claude settings
    @property
    def claude_main_model(self) -> str:
        return self._config["llm"]["claude"]["main_model"]
    
    @property
    def claude_image_extraction_model(self) -> str:
        return self._config["llm"]["claude"]["image_extraction_model"]
    
    # Cohere settings
    @property
    def cohere_embedding_model(self) -> str:
        return self._config["llm"]["cohere"]["embedding_model"]

# Global config instance
config = Config()