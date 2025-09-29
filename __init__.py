"""
PDF RAG Chatbot - A RAG-based PDF chatbot with image rendering capabilities.

This package provides tools for:
- Converting PDFs to markdown with image extraction  
- Building FAISS vector indices for document search
- Running a Streamlit chatbot interface with HTML rendering
- Command-line tools for PDF processing and querying
"""

__version__ = "1.0.0"
__author__ = "Vivek Singh"
__email__ = "vivekkgp97@gmail.com"

# Import main components
try:
    from pdf_rag_with_image.rag_system import RAGSystem
    from pdf_rag_with_image.pdf_processor import PDFToMarkdown
    from pdf_rag_with_image.index_builder import MarkdownIndexBuilder
    
    __all__ = ["RAGSystem", "PDFToMarkdown", "MarkdownIndexBuilder"]
except ImportError:
    # Handle import errors gracefully during package building
    __all__ = []