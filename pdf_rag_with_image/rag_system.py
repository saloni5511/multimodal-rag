from typing import List, Dict, Any, Tuple
import re
import base64
import os
import sys
from .llm_client import LLMClient
from .ocr import ImageAnalyzer
from model_management.config_loader import config
from langchain_community.vectorstores import FAISS
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_management.prompts import get_final_response_prompt

class RAGSystem:
    """RAG system using md_faiss_index for retrieval and LLM for generation."""
    
    def __init__(self, index_path: str = None):
        """Initialize RAG system with FAISS index and LLM client."""
        print("Initializing RAG system...")
        
        # Initialize LLM client and image analyzer
        self.llm_client = LLMClient()
        self.image_analyzer = ImageAnalyzer(self.llm_client)
        
        # Determine the correct index path
        if index_path is None:
            import os
            possible_paths = [
                "./md_faiss_index",      # From main directory
                "md_faiss_index",        # From main directory
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "md_faiss_index")  # Absolute path
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss")):
                    index_path = path
                    break
            
            if index_path is None:
                raise FileNotFoundError(f"FAISS index not found. Tried: {possible_paths}")
        
        # Load the FAISS index using standard LangChain functionality
        print(f"Loading FAISS index from {index_path}...")
        self.vectorstore = FAISS.load_local(
            index_path, 
            self.llm_client.client.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        print(" RAG system initialized successfully!")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Retrieve relevant chunks from the FAISS index."""
        print(f"Retrieving top {top_k} chunks for query: '{query}'")
        
        # Use standard LangChain FAISS similarity_search_with_score
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Convert to our format and sort by score in descending order (best scores first)
        retrieved_chunks = []
        for doc, score in docs_and_scores:
            retrieved_chunks.append((float(score), doc.page_content, doc.metadata))
        
        # Sort by score in descending order (higher scores first)
        retrieved_chunks.sort(key=lambda x: x[0], reverse=True)
        
        print(f"Found {len(retrieved_chunks)} relevant chunks")
        for i, (score, text, metadata) in enumerate(retrieved_chunks):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            print(f"  {i+1}. Score: {score:.4f} | Source: {source} | Page: {page}")
        
        return retrieved_chunks
    
    def query(self, question: str, top_k: int = None, retrieved_chunks: List[Tuple[float, str, Dict[str, Any]]] = None) -> str:
        """Perform RAG query: retrieve chunks and generate answer."""
        print(f"\n=== RAG Query Pipeline ===")
        print(f"📝 User Question: {question}")
        
        # Step 1: Retrieve relevant chunks (if not provided)
        
        top_k = config.top_k
        print(f"🔍 Step 1: Extracting {top_k} most relevant MD chunks...")
        retrieved_chunks = self.retrieve(question, top_k)
        print(f"✅ Extracted {len(retrieved_chunks)} chunks from knowledge base")
    
        
        if not retrieved_chunks:
            print("❌ No relevant chunks found")
            return "No relevant information found in the knowledge base."
        
        # Step 2: Analyze images in chunks
        print(f"🖼️ Step 2: Analyzing images in extracted chunks...")
        filtered_chunks, image_stats = self.image_analyzer.process_chunks_with_images(retrieved_chunks, question)
        print(f"✅ Image analysis complete - processed {len(filtered_chunks)} chunks")
        
        # Store image stats and processed chunks for UI access
        self._last_image_stats = image_stats
        self._last_processed_chunks = filtered_chunks
        
        # Step 3: Prepare final context
        print(f"📝 Step 3: Preparing final context for generative LLM...")
        context_parts = []
        for i, (score, text, metadata) in enumerate(filtered_chunks, 1):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            page_info = f"Page {page}" if page else "Header"
            print(f"   Chunk {i}: {source} ({page_info}) - {len(text)} chars")
            context_parts.append(f"[{source} - {page_info}] {text}")
        
        context = "\n\n".join(context_parts)
        print(f"✅ Final context prepared: {len(context)} characters")
        
        # Step 4: Create prompt and generate answer using centralized prompt
        prompt = get_final_response_prompt(question, context)
        
        print("🤖 Step 4: Sending to generative LLM (Claude Sonnet 4)...")
        print(f"   Prompt length: {len(prompt)} characters")
        answer = self.llm_client.generate(prompt)
        
        print(f"✅ Generated response: {len(answer)} characters")
        print(f"=== RAG Pipeline Complete ===\n")
        
        # Process HTML to convert image paths to base64
        processed_answer = self.process_html_images(answer)
        
        return processed_answer
    
    def convert_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 data URL."""
        try:
            print(f"🔍 Converting image path: {image_path}")
            
            # Handle relative paths from different locations
            possible_paths = [
                image_path,
                os.path.join("..", image_path.lstrip("../")),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path.lstrip("../"))
            ]
            
            # If path starts with ../, try to resolve it from the project root
            if image_path.startswith("../"):
                project_root = os.path.dirname(os.path.dirname(__file__))
                clean_path = image_path.lstrip("../")
                full_path = os.path.join(project_root, clean_path)
                possible_paths.insert(0, full_path)
                
                # Also search in PDF subdirectories if the direct path doesn't exist
                if "extracted_from_pdf/images/" in clean_path:
                    # Extract the filename from the path
                    filename = os.path.basename(clean_path)
                    images_dir = os.path.join(project_root, "extracted_from_pdf", "images")
                    if os.path.exists(images_dir):
                        # Search in all PDF subdirectories
                        for pdf_dir in os.listdir(images_dir):
                            pdf_path = os.path.join(images_dir, pdf_dir)
                            if os.path.isdir(pdf_path):
                                potential_path = os.path.join(pdf_path, filename)
                                possible_paths.append(potential_path)
            
            print(f"   Trying paths: {possible_paths}")
            
            actual_path = None
            for path in possible_paths:
                print(f"   Checking: {path} - Exists: {os.path.exists(path)}")
                if os.path.exists(path):
                    actual_path = path
                    break
            
            if not actual_path:
                print(f"   ❌ Image not found in any path")
                return image_path  # Return original path if not found
            
            print(f"   ✅ Found image at: {actual_path}")
            with open(actual_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                base64_url = f"data:image/png;base64,{encoded}"
                print(f"   ✅ Converted to base64: {len(base64_url)} chars")
                return base64_url
        except Exception as e:
            print(f"   ❌ Error converting image {image_path}: {e}")
            return image_path  # Return original path on error
    
    def process_html_images(self, html_content: str) -> str:
        """Convert relative image paths in HTML to base64 data URLs."""
        print(f"Processing HTML for images... (length: {len(html_content)})")
        
        # Pattern to match img src attributes with relative paths - more flexible
        img_patterns = [
            r'<img\s+([^>]*?)src="([^"]*?\.png)"([^>]*?)>',  # Original pattern
            r'<img\s+src="([^"]*?\.png)"([^>]*?)>',          # src first
            r'<img\s+([^>]*?)src=\'([^\']*?\.png)\'([^>]*?)>', # single quotes
            r'<img\s+([^>]*?)src="([^"]*?\.jpg)"([^>]*?)>',  # JPG files
            r'<img\s+([^>]*?)src="([^"]*?\.jpeg)"([^>]*?)>', # JPEG files
        ]
        
        def replace_img_src(match):
            if len(match.groups()) == 3:
                before_src = match.group(1)
                img_path = match.group(2)
                after_src = match.group(3)
            else:
                # Handle 2-group pattern
                img_path = match.group(1)
                before_src = ""
                after_src = match.group(2)
            
            print(f"Found image: {img_path}")
            
            # Convert to base64 if it's a relative path
            if img_path.startswith("../"):
                print(f"Converting to base64: {img_path}")
                base64_url = self.convert_image_to_base64(img_path)
                result = f'<img {before_src}src="{base64_url}"{after_src}>'
                print(f"Conversion successful: {len(base64_url)} chars")
                return result
            
            return match.group(0)  # Return unchanged if not relative path
        
        # Try each pattern
        processed = html_content
        for pattern in img_patterns:
            processed = re.sub(pattern, replace_img_src, processed)
        
        print(f"Image processing complete. Found data:image: {'data:image' in processed}")
        return processed
    
    def process_chunks_with_images(self, retrieved_chunks: List[Tuple[float, str, Dict[str, Any]]], query: str) -> Tuple[List[Tuple[float, str, Dict[str, Any]]], Dict[str, Any]]:
        """Delegate image processing to ImageAnalyzer."""
        return self.image_analyzer.process_chunks_with_images(retrieved_chunks, query)
    
    def get_last_image_stats(self) -> Dict[str, int]:
        """Get image analysis statistics from the last query."""
        return getattr(self, '_last_image_stats', {"total_found": 0, "processed": 0, "kept": 0, "removed": 0})
    
    def get_last_processed_chunks(self) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Get the processed chunks (after image filtering) from the last query."""
        return getattr(self, '_last_processed_chunks', [])
    
    def generate_final_response(self, question: str, filtered_chunks: List[Tuple[float, str, Dict[str, Any]]]) -> str:
        """Generate final response using processed chunks."""
        print(f"📝 Preparing final context for generative LLM...")
        context_parts = []
        for i, (score, text, metadata) in enumerate(filtered_chunks, 1):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            page_info = f"Page {page}" if page else "Header"
            print(f"   Chunk {i}: {source} ({page_info}) - {len(text)} chars")
            context_parts.append(f"[{source} - {page_info}] {text}")
        
        context = "\n\n".join(context_parts)
        print(f"✅ Final context prepared: {len(context)} characters")
        
        # Create prompt and generate answer using centralized prompt
        prompt = get_final_response_prompt(question, context)
        
        print("🤖 Sending to generative LLM (Claude Sonnet 4)...")
        print(f"   Prompt length: {len(prompt)} characters")
        answer = self.llm_client.generate(prompt)
        
        print(f"✅ Generated response: {len(answer)} characters")
        
        # Process HTML to convert image paths to base64
        processed_answer = self.process_html_images(answer)
        
        # Store processed chunks for UI access
        self._last_processed_chunks = filtered_chunks
        
        return processed_answer
   

def main():
    rag = RAGSystem()
    rag.query("Unified view and review experience in Home view?")
  
if __name__ == "__main__":
    main()