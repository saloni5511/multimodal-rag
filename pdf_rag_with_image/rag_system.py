from typing import List, Dict, Any, Tuple
import re
import base64
import os
from pathlib import Path
from .llm_client import GeminiLLM
from langchain_community.vectorstores import FAISS

class RAGSystem:
    """RAG system using md_faiss_index for retrieval and GeminiLLM for generation."""
    
    def __init__(self, index_path: str = None):
        """Initialize RAG system with FAISS index and GeminiLLM."""
        print("Initializing RAG system...")
        
        # Initialize GeminiLLM for both embeddings and generation
        self.gemini_llm = GeminiLLM()
        
        # Determine the correct index path
        if index_path is None:
            import os
            possible_paths = [
                "../md_faiss_index",     # From rag_pipeline directory
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
            self.gemini_llm.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        print(" RAG system initialized successfully!")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
        """Retrieve relevant chunks from the FAISS index."""
        print(f"Retrieving top {top_k} chunks for query: '{query}'")
        
        # Use standard LangChain FAISS similarity_search_with_score
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Convert to our format
        hits = []
        for doc, score in docs_and_scores:
            hits.append((float(score), doc.page_content, doc.metadata))
        
        print(f"Found {len(hits)} relevant chunks")
        for i, (score, text, metadata) in enumerate(hits):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            print(f"  {i+1}. Score: {score:.4f} | Source: {source} | Page: {page}")
        
        return hits
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Perform RAG query: retrieve chunks and generate answer."""
        print(f"\n=== RAG Query ===")
        print(f"Question: {question}")
        
        # Step 1: Retrieve relevant chunks
        hits = self.retrieve(question, top_k)
        
        if not hits:
            return "No relevant information found in the knowledge base."
        
        # Step 2: Prepare context from retrieved chunks
        context_parts = []
        for score, text, metadata in hits:
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', '')
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[{source}{page_info}] {text}")
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Create prompt and generate answer
        prompt = f""" Don not say Here is an HTML response to answer the query, just write html content.
    You are an AI assistant tasked with creating informative HTML content based on a given query and context. Your response should be well-structured, informative, and visually appealing.

    Query: {question}

    Context: {context}

    Instructions:
    1. Create an HTML response that answers the query using information from the context.
    2. Use appropriate HTML tags to structure your response (e.g., <h1>, <h2>, <p>, <ul>, <li>).
    3. If you reference any images from the context, use the exact file path provided in the context.
    4. Use the following format for images: <img src="[IMAGE_PATH]" alt="[DESCRIPTION]" style="max-width: 100%; height: auto;">
    5. Provide a brief caption for each image using a <p> tag with a class "caption".
    6. Use inline CSS to style your HTML for better readability.
    7. If no relevant information is found in the context, state that clearly.
    8. Do not invent or assume any information not present in the context.
    9. transformed the image path into this path "../extracted_from_pdf/images/page_8_img_0_0bc3a9d8.png", "../extracted_from_pdf/images/page_6_img_1_0bc3a9d8.png"
    Your response should be complete, well-formatted HTML that can be directly rendered in a Jupyter notebook.

    Example: 
     image_path =  "../extracted_from_pdf/images/page_8_img_0_0bc3a9d8.png"

    html_content = f'''
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: auto;">
    <h1 style="color: #4285F4;">Google Image Search</h1>
    
    <p style="font-size: 16px; line-height: 1.5;">
    Google Image Search is a powerful tool that allows users to search the web for image content. 
    It uses advanced algorithms to analyze and categorize images, making it easier for users to 
    find specific visual content quickly and efficiently.
    </p>
    
    <h2 style="color: #34A853;">Key Features:</h2>
    <ul style="font-size: 16px; line-height: 1.5;">
        <li>Visual search capabilities</li>
        <li>Filter options (size, color, type, etc.)</li>
        <li>Reverse image search</li>
        <li>Integration with Google Lens</li>
    </ul>
    
    <p style="font-size: 16px; line-height: 1.5;">
    Below is an image demonstrating the Google Image Search interface:
    </p>
    
    <img src="image_path" alt="Google Image Search Interface" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">
    
    <p style="font-size: 14px; color: #666; font-style: italic;">
    Image: Example of Google Image Search results page
    </p>
</div>
'''
    """
        
        print("\nGenerating answer...")
        answer = self.gemini_llm.generate(prompt)
        
        print(f"\n=== Answer ===")
        print(answer)
        
        # Process HTML to convert image paths to base64
        processed_answer = self.process_html_images(answer)
        
        return processed_answer
    
    def convert_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 data URL."""
        try:
            # Handle relative paths from different locations
            possible_paths = [
                image_path,
                os.path.join("..", image_path.lstrip("../")),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), image_path.lstrip("../"))
            ]
            
            actual_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    actual_path = path
                    break
            
            if not actual_path:
                return image_path  # Return original path if not found
            
            with open(actual_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"Error converting image {image_path}: {e}")
            return image_path  # Return original path on error
    
    def process_html_images(self, html_content: str) -> str:
        """Convert relative image paths in HTML to base64 data URLs."""
        print(f"Processing HTML for images... (length: {len(html_content)})")
        
        # Pattern to match img src attributes with relative paths - more flexible
        img_patterns = [
            r'<img\s+([^>]*?)src="([^"]*?\.png)"([^>]*?)>',  # Original pattern
            r'<img\s+src="([^"]*?\.png)"([^>]*?)>',          # src first
            r'<img\s+([^>]*?)src=\'([^\']*?\.png)\'([^>]*?)>', # single quotes
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
    
   

def main():
    rag = RAGSystem()
    rag.query("Unified view and review experience in Home view?")
  
if __name__ == "__main__":
    main()