import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .llm_client import LLMClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model_management.config_loader import config
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class MarkdownIndexBuilder:
    """Class for building FAISS index from markdown documents."""
    
    def __init__(self):
        """Initialize the markdown index builder."""
        self.llm_client = LLMClient()
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = config.chunk_size
        if overlap is None:
            overlap = config.overlap
            
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunks.append(" ".join(words[start:end]))
            start = end - overlap if end - overlap > start else end
        return chunks

    def read_markdown_files(self, content_dir: str = "./extracted_from_pdf/extracted_content", specific_files: List[str] = None) -> List[Dict[str, Any]]:
        """Read specific markdown files or all markdown files from the extracted content directory."""
        content_path = Path(content_dir)
        
        if not content_path.exists():
            raise FileNotFoundError(f"Content directory not found: {content_dir}")
        
        if specific_files:
            # Read only specific markdown files
            markdown_files = []
            for filename in specific_files:
                md_file = content_path / f"{filename}.md"
                if md_file.exists():
                    markdown_files.append(md_file)
                else:
                    print(f"Warning: {md_file} not found")
        else:
            # Read all markdown files
            markdown_files = list(content_path.glob("*.md"))
        
        if not markdown_files:
            if specific_files:
                raise FileNotFoundError(f"No specified markdown files found in {content_dir}")
            else:
                raise FileNotFoundError(f"No markdown files found in {content_dir}")
        
        documents = []
        for md_file in markdown_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents.append({
                'filename': md_file.name,
                'content': content,
                'path': str(md_file)
            })
        
        print(f"Found {len(documents)} markdown files")
        return documents

    def process_markdown_content(self, documents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
        """Process markdown documents into chunks with metadata."""
        all_texts = []
        all_metadatas = []
        
        for doc in documents:
            content = doc['content']
            filename = doc['filename']
            
            # Split content by pages if available
            page_pattern = r'## Page (\d+)'
            pages = re.split(page_pattern, content)
            
            current_page = None
            chunk_id = 0
            
            for i, section in enumerate(pages):
                if i == 0:
                    # Header content before first page
                    if section.strip():
                        chunks = self.chunk_text(section.strip())
                        for chunk in chunks:
                            all_texts.append(chunk)
                            all_metadatas.append({
                                'source': filename,
                                'page': None,
                                'chunk_id': chunk_id,
                                'type': 'header'
                            })
                            chunk_id += 1
                elif i % 2 == 1:
                    # Page number
                    current_page = int(section) if section.isdigit() else None
                else:
                    # Page content
                    if section.strip():
                        # Keep image references in the text
                        clean_content = section.strip()
                        clean_content = re.sub(r'\s+', ' ', clean_content).strip()
                        
                        if clean_content:
                            chunks = self.chunk_text(clean_content)
                            for chunk in chunks:
                                all_texts.append(chunk)
                                all_metadatas.append({
                                    'source': filename,
                                    'page': current_page,
                                    'chunk_id': chunk_id,
                                    'type': 'content'
                                })
                                chunk_id += 1
        
        print(f"Created {len(all_texts)} text chunks")
        return all_texts, all_metadatas

    def build_index(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> FAISS:
        """Build FAISS index using LangChain's built-in functionality."""
        # Create documents from texts and metadata
        documents = [Document(page_content=text, metadata=meta or {}) 
                    for text, meta in zip(texts, metadatas or [{}] * len(texts))]
        
        # Use standard LangChain FAISS.from_documents
        vectorstore = FAISS.from_documents(documents, self.llm_client.client.embeddings)
        
        return vectorstore

    def build_md_index(self, content_dir: str = "./extracted_from_pdf/extracted_content", 
                       save_path: str = "./md_faiss_index", specific_files: List[str] = None) -> FAISS:
        """Build FAISS index from markdown documents."""
        print("=== Building Markdown Index ===")
        
        if specific_files:
            print(f"Processing specific files: {specific_files}")
            # For specific files, we need to rebuild the entire index with all files
            # because FAISS doesn't support efficient incremental updates
            documents = self.read_markdown_files(content_dir)  # Read all files
        else:
            # Read all markdown files
            documents = self.read_markdown_files(content_dir)
        
        # Process into chunks
        texts, metadatas = self.process_markdown_content(documents)
        
        # Build FAISS index
        print("Creating FAISS index...")
        vectorstore = self.build_index(texts, metadatas)
        
        # Save index using standard LangChain method
        print(f"Saving index to {save_path}...")
        vectorstore.save_local(save_path)
        
        print("✅ Markdown index built and saved successfully!")
        return vectorstore

    def load_md_index(self, save_path: str = "./md_faiss_index") -> FAISS:
        """Load existing markdown FAISS index."""
        # Use standard LangChain FAISS.load_local
        vectorstore = FAISS.load_local(
            save_path, 
            self.llm_client.client.embeddings, 
            allow_dangerous_deserialization=True
        )
        
        print(f"✅ Loaded markdown index from {save_path}")
        return vectorstore

def main():
    """Main function to build the index."""
    builder = MarkdownIndexBuilder()
    builder.build_md_index()

if __name__ == "__main__":
    main()