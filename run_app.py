#!/usr/bin/env python3
"""Simple Streamlit RAG App."""

import streamlit as st
import json
import os
import time
import re
from pathlib import Path
from pdf_rag_with_image.rag_system import RAGSystem
from pdf_rag_with_image.pdf_processor import PDFToMarkdown
from pdf_rag_with_image.index_builder import MarkdownIndexBuilder

# Page config
st.set_page_config(
    page_title="Multi-Modal PDF Chatbot",
    page_icon="💬",
    layout="wide"
)

# Constants
SAMPLE_PDF_DIR = Path("sample_pdf")
PDF_STATUS_FILE = "pdf_status.json"

# Ensure sample_pdf directory exists
SAMPLE_PDF_DIR.mkdir(exist_ok=True)

def load_pdf_status():
    """Load PDF status from JSON file."""
    if os.path.exists(PDF_STATUS_FILE):
        with open(PDF_STATUS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_pdf_status(status_dict):
    """Save PDF status to JSON file."""
    with open(PDF_STATUS_FILE, 'w') as f:
        json.dump(status_dict, f, indent=2)

def get_pdf_files():
    """Get all PDF files in sample_pdf directory."""
    return list(SAMPLE_PDF_DIR.glob("*.pdf"))

def update_pdf_status():
    """Update PDF status based on current files."""
    status = load_pdf_status()
    current_pdfs = {pdf.name for pdf in get_pdf_files()}
    
    # Add new PDFs as not embedded
    for pdf_name in current_pdfs:
        if pdf_name not in status:
            status[pdf_name] = {"embed": False}
    
    # Remove deleted PDFs
    status = {k: v for k, v in status.items() if k in current_pdfs}
    
    save_pdf_status(status)
    return status

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system."""
    return RAGSystem()

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = load_rag_system()
        # Don't stop here - let the main interface handle the None case

def main():
    # Initialize session state
    initialize_session_state()
    
    st.title("Multi-Modal PDF Chatbot")
    
    # Sidebar for PDF management
    with st.sidebar:
        st.header("📚 PDF Management")
        
        # PDF Upload
        uploaded_file = st.file_uploader(
            "Upload PDF", 
            type="pdf", 
            help="Upload a PDF file to add to your knowledge base"
        )
        
        if uploaded_file is not None:
            # Use session state to prevent duplicate processing
            if 'last_uploaded_file' not in st.session_state:
                st.session_state.last_uploaded_file = None
            
            # Check if this is a new file upload
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_uploaded_file != file_key:
                pdf_path = SAMPLE_PDF_DIR / uploaded_file.name
                
                # Save uploaded file
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Update status immediately to show the new PDF
                status = load_pdf_status()
                status[uploaded_file.name] = {"embed": False}
                save_pdf_status(status)
                
                # Update session state
                st.session_state.last_uploaded_file = file_key
                
                print(f"📄 Uploaded: {uploaded_file.name} → embed: False")
                st.success(f"✅ Uploaded: {uploaded_file.name}")
                st.rerun()
        
        st.divider()
        
        # PDF Status Display
        st.subheader("📄 PDF Status")
        pdf_status = update_pdf_status()
        
        if not pdf_status:
            st.info("No PDFs found. Upload a PDF to get started!")
        else:
            for pdf_name, status in pdf_status.items():
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"📄 {pdf_name}")
                        if status["embed"]:
                            st.write("✅ Embedded")
                        else:
                            st.write("❌ Not embedded")
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{pdf_name}", help="Delete PDF"):
                            os.remove(SAMPLE_PDF_DIR / pdf_name)
                            st.rerun()
                
                st.divider()
        
        # Embedding Controls
        st.subheader("🔧 Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Build Embedding", help="Process PDFs and create embeddings"):
                with st.spinner("Building embeddings..."):
                    # Get current status
                    status = load_pdf_status()
                    
                    # Find PDFs that need embedding (embed: False)
                    pdfs_to_embed = [pdf_name for pdf_name, pdf_status in status.items() 
                                   if not pdf_status.get("embed", False)]
                    
                    if not pdfs_to_embed:
                        st.info("All PDFs are already embedded!")
                        return
                    
                    print(f"🔄 Building embeddings for {len(pdfs_to_embed)} PDFs: {pdfs_to_embed}")
                    
                    # Process only PDFs that need embedding
                    processor = PDFToMarkdown()
                    processor.process_sample_pdfs(pdfs_to_embed)
                    
                    # Build index - rebuild entire index to include new PDFs
                    # (FAISS requires full rebuild for consistency)
                    builder = MarkdownIndexBuilder()
                    builder.build_md_index()
                    
                    # Update status only for processed PDFs
                    for pdf_name in pdfs_to_embed:
                        if pdf_name in status:
                            status[pdf_name]["embed"] = True
                    save_pdf_status(status)
                    
                    # Clear cache and reload RAG system  
                    st.cache_resource.clear()
                    st.session_state.rag_system = load_rag_system()
                    
                    print(f"✅ Embeddings built successfully for {len(pdfs_to_embed)} PDFs!")
                    
                    # Force UI refresh to show chat interface
                    st.success(f"✅ Embedding built successfully for {len(pdfs_to_embed)} PDFs! Chat is now ready.")
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            st.empty()  # Empty column for layout balance
        
    
    # Main chat interface
    
    # Check if RAG system is available
    if st.session_state.rag_system is None:
        st.error("❌ RAG System not available. Please build embeddings first.")
        
        # Add build embedding button in main area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Build Embeddings Now", help="Build embeddings to enable chat", key="main_build_btn"):
                with st.spinner("Building embeddings..."):
                    # Get current status
                    status = load_pdf_status()
                    
                    # Find PDFs that need embedding (embed: False)
                    pdfs_to_embed = [pdf_name for pdf_name, pdf_status in status.items() 
                                   if not pdf_status.get("embed", False)]
                    
                    if not pdfs_to_embed:
                        st.info("All PDFs are already embedded!")
                        # Try to reload RAG system
                        st.session_state.rag_system = load_rag_system()
                        st.rerun()
                        return
                    
                    print(f"🔄 Building embeddings for {len(pdfs_to_embed)} PDFs: {pdfs_to_embed}")
                    
                    # Process only PDFs that need embedding
                    processor = PDFToMarkdown()
                    processor.process_sample_pdfs(pdfs_to_embed)
                    
                    # Build index
                    builder = MarkdownIndexBuilder()
                    builder.build_md_index()
                    
                    # Update status only for processed PDFs
                    for pdf_name in pdfs_to_embed:
                        if pdf_name in status:
                            status[pdf_name]["embed"] = True
                    save_pdf_status(status)
                    
                    print(f"✅ Embeddings built successfully for {len(pdfs_to_embed)} PDFs!")
                    
                    # Clear cache and reload RAG system
                    st.cache_resource.clear()
                    st.session_state.rag_system = load_rag_system()
                    
                    # Force UI refresh to show chat interface
                    st.success(f"✅ Embedding built successfully for {len(pdfs_to_embed)} PDFs! Chat is now ready.")
                    time.sleep(1)
                    st.rerun()
        
        st.stop()  # Stop execution here if no RAG system
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Render HTML response
                st.components.v1.html(message["content"], height=400, scrolling=True)
            else:
                st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the PDF..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            # Create placeholders for streaming updates in correct order
            extracting_placeholder = st.empty()
            image_processing_placeholder = st.empty()
            generating_placeholder = st.empty()
            response_placeholder = st.empty()
            
            # Show extracting status
            extracting_placeholder.info("📤 Extracting relevant chunks...")
            
            # Get relevant chunks first
            retrieved_chunks = st.session_state.rag_system.retrieve(prompt)
            
            if retrieved_chunks:
                # Show chunk information with image analysis results
                chunk_info = []
                total_images = 0
                
                # Process chunks to get image statistics
                for (score, text, metadata) in retrieved_chunks:
                    source = metadata.get('source', 'Unknown')
                    page = metadata.get('page', 'N/A')
                    
                    # Count images in this chunk
                    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                    images_in_chunk = len(re.findall(img_pattern, text))
                    total_images += images_in_chunk
                    
                    chunk_info.append(f"📄 **{source}** (Page {page}) - Score: {score:.3f} - {images_in_chunk} images")
                
                extracting_placeholder.success(f"✅ Extracted {len(retrieved_chunks)} relevant chunks with {total_images} images:\n\n" + "\n".join(chunk_info))
                
                # Step 2: Image Analysis
                image_processing_placeholder.info("🖼️ Analyzing images...")
                
                # Process chunks with images (separate from final generation)
                filtered_chunks, image_stats = st.session_state.rag_system.process_chunks_with_images(retrieved_chunks, prompt)
                
                # Store image stats for UI access
                st.session_state.rag_system._last_image_stats = image_stats
                
                # Show image analysis results for all chunks
                chunk_details = []
                for chunk in image_stats.get("chunk_details", []):
                    chunk_details.append(
                        f"📄 {chunk['source']} (Page {chunk['page']}): {chunk['images_removed']} removed"
                    )
                
                if chunk_details:
                    chunk_info = "\n".join(chunk_details)
                    image_analysis_info = f"""🖼️ **Image Processing Results:**
{chunk_info}

**Summary:** {image_stats['kept']} kept, {image_stats['removed']} removed from {image_stats['total_found']} found"""
                else:
                    image_analysis_info = f"🖼️ Images: {image_stats['kept']} kept, {image_stats['removed']} removed from {image_stats['total_found']} found"
                    
                image_processing_placeholder.success(image_analysis_info)
                
                # Step 3: Generate Answer and show final response immediately after image processing
                generating_placeholder.info("🤖 Generating answer from updated chunks...")
                
                # Generate final response using processed chunks
                response = st.session_state.rag_system.generate_final_response(prompt, filtered_chunks)
                
                # Show final answer immediately after image processing
                generating_placeholder.success("✅ Answer generated from updated chunks:")
                
                # Display the final response
                st.session_state.messages.append({"role": "assistant", "content": response})
                with response_placeholder.container():
                    st.components.v1.html(response, height=400, scrolling=True)
                
                # Show final processed chunks below the answer
                processed_chunks = st.session_state.rag_system.get_last_processed_chunks()
                if processed_chunks:
                    st.subheader("📝 Final Chunks Sent to LLM")
                    for i, (score, text, metadata) in enumerate(processed_chunks, 1):
                        source = metadata.get('source', 'Unknown')
                        page = metadata.get('page', 'N/A')
                        
                        with st.expander(f"Chunk {i}: {source} (Page {page}) - Score: {score:.3f}"):
                            st.text_area(
                                "Processed Content:",
                                text,
                                height=200,
                                key=f"chunk_{i}"
                            )
            else:
                extracting_placeholder.warning("⚠️ No relevant chunks found")
                response_placeholder.write("I couldn't find relevant information in the knowledge base to answer your question.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()