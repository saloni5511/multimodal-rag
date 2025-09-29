#!/usr/bin/env python3
"""
CLI interface for PDF-to-Markdown RAG system.
Provides options to convert PDFs, build index, and ask questions.
"""

import sys
import argparse
from pathlib import Path

# Add the pdf_rag_with_image directory to the path
package_dir = Path(__file__).parent / "pdf_rag_with_image"
sys.path.insert(0, str(package_dir))

from pdf_processor import PDFToMarkdown
from index_builder import MarkdownIndexBuilder
from rag_system import RAGSystem


def convert_pdf(pdf_path: str = None):
    """Convert PDF to markdown."""
    converter = PDFToMarkdown()
    
    if pdf_path:
        if not Path(pdf_path).exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return
        
        print(f"Converting PDF: {pdf_path}")
        converter.process_pdf(pdf_path)
    else:
        print("Converting all PDFs in sample_pdf directory...")
        converter.process_sample_pdfs()


def build_index():
    """Build FAISS index from markdown files."""
    print("Building index from markdown files...")
    builder = MarkdownIndexBuilder()
    builder.build_md_index()


def ask_question(question: str, top_k: int = 5):
    """Ask a question using the RAG system."""
    print(f"Asking question: {question}")
    rag = RAGSystem()
    answer = rag.query(question, top_k)
    print(f"\nAnswer: {answer}")
    return answer


def interactive_mode():
    """Run in interactive question-answering mode."""
    print("\n=== Interactive RAG System ===")
    print("Ask questions about your documents. Type 'quit' to exit.\n")
    
    try:
        rag = RAGSystem()
        
        while True:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            print("\nSearching for answer...")
            answer = rag.query(question)
            print("\n" + "="*50 + "\n")
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error initializing RAG system: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s convert                          # Convert all PDFs in sample_pdf/
  %(prog)s convert path/to/file.pdf         # Convert specific PDF
  %(prog)s index                            # Build search index
  %(prog)s ask "What is this document about?"  # Ask a question
  %(prog)s interactive                      # Start interactive mode
  %(prog)s pipeline                         # Run full pipeline (convert + index)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert PDF to Markdown')
    convert_parser.add_argument('pdf_path', nargs='?', help='Path to PDF file (optional)')
    
    # Index command
    subparsers.add_parser('index', help='Build FAISS index from markdown files')
    
    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('question', help='Question to ask')
    ask_parser.add_argument('--top-k', type=int, default=5, help='Number of results to retrieve')
    
    # Interactive command
    subparsers.add_parser('interactive', help='Start interactive Q&A mode')
    
    # Pipeline command (convert + index)
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full pipeline: convert PDFs and build index')
    pipeline_parser.add_argument('pdf_path', nargs='?', help='Path to PDF file (optional)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'convert':
            convert_pdf(args.pdf_path)
            
        elif args.command == 'index':
            build_index()
            
        elif args.command == 'ask':
            ask_question(args.question, args.top_k)
            
        elif args.command == 'interactive':
            interactive_mode()
            
        elif args.command == 'pipeline':
            print("=== Running Full Pipeline ===")
            print("Step 1: Converting PDF(s) to Markdown...")
            convert_pdf(args.pdf_path)
            print("\nStep 2: Building search index...")
            build_index()
            print("\n✅ Pipeline complete!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()