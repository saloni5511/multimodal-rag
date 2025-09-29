import streamlit as st
import sys
import asyncio
from pathlib import Path

# Import from the proper package
from pdf_rag_with_image.rag_system import RAGSystem

# Fix for Streamlit async issues
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Page config
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="💬",
    layout="centered"
)

@st.cache_resource
def load_rag_system():
    """Load and cache the RAG system."""
    try:
        return RAGSystem()
    except Exception as e:
        st.error(f"Failed to load RAG system: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = load_rag_system()
        if st.session_state.rag_system is None:
            st.stop()

def main():
    # Initialize session state
    initialize_session_state()
    
    st.title("💬 PDF Chatbot")
    
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
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_system.query(prompt)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                    # Render the HTML response
                    st.components.v1.html(response, height=400, scrolling=True)
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()