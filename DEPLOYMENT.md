# PDF RAG with Image - Deployment Guide

A complete PDF RAG (Retrieval-Augmented Generation) system that processes PDFs, extracts images, and provides an intelligent chatbot interface.

## Features

- 📄 **PDF Processing**: Convert PDFs to searchable markdown with image extraction
- 🔍 **Vector Search**: FAISS-powered semantic search for accurate retrieval
- 🤖 **LLM Integration**: Google Gemini for intelligent responses
- 🖼️ **Image Support**: Automatic image extraction and embedding in responses
- 💬 **Chat Interface**: User-friendly Streamlit chatbot interface
- 🐳 **Docker Support**: Easy containerized deployment
- 📦 **Pip Installable**: Simple installation via pip

## Installation Options

### Option 1: Pip Installation (Recommended for End Users)

```bash
pip install pdf-rag-with-image
```

### Option 2: Docker Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag_image

# Using Docker Compose
docker-compose up -d
```

### Option 3: Local Development

```bash
# Clone and install for development
git clone <repository-url>
cd rag_image
pip install -e .
```

## Setup Requirements

### 1. Google API Key

Create a `gemini_api_key.json` file in your project directory:

```json
{
    "api_key": "your-google-gemini-api-key-here"
}
```

**Get your API key from**: https://makersuite.google.com/app/apikey

### 2. Directory Structure

Create the following directories:
```
your-project/
├── gemini_api_key.json
├── sample_pdf/              # Place your PDF files here
├── extracted_from_pdf/      # Auto-created during processing
└── md_faiss_index/          # Auto-created during indexing
```

## Usage

### Command Line Interface (CLI)

After installation, use the `pdf-rag` command:

```bash
# Convert all PDFs in sample_pdf/ directory
pdf-rag convert

# Convert a specific PDF
pdf-rag convert path/to/document.pdf

# Build search index from converted files
pdf-rag index

# Ask questions via CLI
pdf-rag ask "What is this document about?"

# Interactive chat mode
pdf-rag interactive

# Run complete pipeline (convert + index)
pdf-rag pipeline
```

### Web Interface (Streamlit)

Launch the web chatbot interface:

```bash
pdf-rag-ui
```

This opens a browser window with an intuitive chat interface.

### Python API

Use programmatically in your Python code:

```python
from pdf_rag_with_image import PDFToMarkdown, MarkdownIndexBuilder, RAGSystem

# Convert PDF
converter = PDFToMarkdown()
converter.process_pdf("document.pdf")

# Build index
builder = MarkdownIndexBuilder()
builder.build_md_index()

# Query system
rag = RAGSystem()
response = rag.query("What are the key findings?")
print(response)
```

## Deployment Options

### 1. Local Deployment

**For Personal Use:**
```bash
pip install pdf-rag-with-image
pdf-rag-ui  # Launches on localhost:8501
```

### 2. Docker Deployment

**For Production/Team Use:**

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

**Custom Docker Setup:**
```dockerfile
# Use the provided Dockerfile
docker build -t pdf-rag .
docker run -p 8501:8501 -v /path/to/data:/app/data pdf-rag
```

### 3. Cloud Deployment

**Deploy to cloud platforms:**

- **Streamlit Cloud**: Connect your GitHub repo with Streamlit Cloud
- **AWS/GCP**: Use container services with the Docker image
- **Heroku**: Deploy using the provided Dockerfile

## Configuration

### Environment Variables

Set these for advanced configuration:

```bash
export GEMINI_API_KEY="your-api-key"
export PDF_INPUT_DIR="custom_pdf_directory"
export FAISS_INDEX_PATH="custom_index_path"
```

### Performance Tuning

- **Memory**: Minimum 4GB RAM recommended
- **Storage**: 500MB per 100 documents
- **CPU**: Multi-core recommended for large document sets

## Troubleshooting

### Common Issues

1. **API Key Not Found**:
   ```bash
   # Ensure gemini_api_key.json exists and contains valid API key
   echo '{"api_key": "your-key"}' > gemini_api_key.json
   ```

2. **Import Errors**:
   ```bash
   # Reinstall with force
   pip install --force-reinstall pdf-rag-with-image
   ```

3. **Docker Issues**:
   ```bash
   # Ensure Docker daemon is running
   docker --version
   sudo systemctl start docker
   ```

### Logging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Sharing with Friends

### Easy Installation for End Users

Your friends can install in one command:

```bash
pip install pdf-rag-with-image
```

Then provide them with:
1. A sample `gemini_api_key.json` file (with their own API key)
2. Instructions to run `pdf-rag-ui` for the web interface

### Quick Start Script

Create a setup script for easy sharing:

```bash
#!/bin/bash
echo "Installing PDF RAG system..."
pip install pdf-rag-with-image

echo "Please add your Gemini API key:"
echo '{"api_key": "REPLACE_WITH_YOUR_KEY"}' > gemini_api_key.json

echo "Setup complete! Run 'pdf-rag-ui' to start"
```

## Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Python with LangChain and FAISS
- **LLM**: Google Gemini 2.5 Flash
- **Storage**: Local filesystem with FAISS vectorstore
- **Processing**: PyMuPDF for PDF extraction

## Development

### Contributing

```bash
git clone <repo>
cd rag_image
pip install -e .
# Make changes and test
python -m pytest
```

### Building from Source

```bash
python -m build
pip install dist/pdf_rag_with_image-*.whl
```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review logs for error details
- Ensure API keys and file paths are correct

## License

[Add your license information here]

---

**Generated with [Claude Code](https://claude.ai/code)**