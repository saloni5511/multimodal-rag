import json
import os
import numpy as np
import boto3
from typing import Dict, Any, List
from dataclasses import dataclass
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from botocore.config import Config
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_management.config_loader import config

# ---------- Data structures ----------
@dataclass
class DocChunk:
    text: str
    metadata: Dict[str, Any]

class GeminiLLM:
    """Unified wrapper for both Gemini LLM and embeddings functionality."""
    def __init__(self, model: str = None, temperature: float = 0.3, normalize: bool = None):
        # Use config values if none provided
        if model is None:
            model = config.gemini_model
        if normalize is None:
            normalize = config.normalize
            
        # Load API key
        api_key_file = "gemini_api_key.json"
        with open(api_key_file, 'r') as f:
            api_data = json.load(f)
            api_key = api_data.get('api_key')
        
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Initialize both LLM and embeddings
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.gemini_embedding_model)
        self.normalize = normalize
    
    def generate(self, prompt: str, image_data: str = None) -> str:
        """Generate text response for a given prompt."""
        if image_data:
            print("Warning: Image analysis not yet supported with Gemini, using text-only")
        
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.embeddings.embed_documents(texts)
        embs = np.array(embeddings, dtype=np.float32)
        
        if self.normalize:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return embs


class BedrockLLM:
    """Unified wrapper for AWS Bedrock with Claude Sonnet 4 and Bedrock embeddings."""
    def __init__(self, model: str = None, temperature: float = 0.3, normalize: bool = None):
        # Use config values if none provided
        if model is None:
            model = config.claude_main_model
        if normalize is None:
            normalize = config.normalize
            
        # Store model and temperature for direct API calls
        self.model = model
        self.temperature = temperature
        
        # Configure AWS session with region from config
        aws_config = Config(region_name=config.aws_region)
        
        # Initialize Bedrock LLM and embeddings
        self.llm = ChatBedrockConverse(
            model=model,
            temperature=temperature,
            config=aws_config
        )
        self.embeddings = BedrockEmbeddings(
            model_id=config.cohere_embedding_model,
            config=aws_config
        )
        self.normalize = normalize
        print(f"Using AWS Bedrock in region: {config.aws_region}")
    
    def generate(self, prompt: str, image_data: str = None) -> str:
        """Generate text response for a given prompt, optionally with image data."""
        if image_data and image_data.startswith("data:image"):
            # Extract base64 data from data URL
            base64_data = image_data.split(",")[1]
            
            # Create message with image for Claude
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                ]
            }
            
            # Use direct Bedrock API call for multimodal
            client = boto3.client('bedrock-runtime', region_name=config.aws_region)
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "temperature": 0.3,
                "messages": [message]
            }
            
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
        else:
            # Text-only prompt
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        embeddings = self.embeddings.embed_documents(texts)
        embs = np.array(embeddings, dtype=np.float32)
        
        if self.normalize:
            embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        return embs


class LLMClient:
    """Unified client that switches between Gemini and Bedrock based on config."""
    def __init__(self, temperature: float = 0.3, normalize: bool = None):
        if normalize is None:
            normalize = config.normalize
            
        if config.use_claude:
            self.client = BedrockLLM(temperature=temperature, normalize=normalize)
            print("Using AWS Bedrock with Claude Sonnet 4")
        else:
            self.client = GeminiLLM(temperature=temperature, normalize=normalize)
            print("Using Gemini API")
    
    def generate(self, prompt: str, image_data: str = None) -> str:
        """Generate text response for a given prompt, optionally with image data."""
        return self.client.generate(prompt, image_data)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        return self.client.encode(texts)
