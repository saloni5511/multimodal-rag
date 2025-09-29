# PDF RAG with Image 🤖📚🖼️

A powerful Retrieval-Augmented Generation (RAG) chatbot that processes PDF documents and provides intelligent answers with image rendering capabilities.

## ✨ Features

- 📄 **PDF Processing**: Extract text and images from PDF documents
- 🔍 **Vector Search**: FAISS-based semantic search for relevant content
- 💬 **Interactive Chat**: Streamlit web interface with chat functionality  
- 🖼️ **Image Rendering**: Display embedded images in responses using base64 encoding
- 🎨 **HTML Responses**: Rich formatting with images, tables, and styling
- 🚀 **Multiple Interfaces**: Web UI, CLI, and Python API
- 🐳 **Docker Ready**: Easy deployment with Docker and docker-compose

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pdf-rag-with-image
   ```

2. **Add your Google Gemini API key**
   ```json
   # Create gemini_api_key.json
   {
     "api_key": "your-google-gemini-api-key-here"
   }
   ```

3. **Add PDF documents**
   ```bash
   # Place your PDF files in the sample_pdf directory
   mkdir -p sample_pdf
   cp your-documents.pdf sample_pdf/
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

5. **Access the web interface**
   Open http://localhost:8501 in your browser

### Option 2: Pip Installation

1. **Install the package**
   ```bash
   pip install pdf-rag-with-image
   ```

2. **Setup API key and data directories**
   ```bash
   # Create working directory
   mkdir my-rag-system && cd my-rag-system
   
   # Create API key file
   echo '{"api_key": "your-api-key"}' > gemini_api_key.json
   
   # Create directories
   mkdir -p sample_pdf extracted_from_pdf/images md_faiss_index
   ```

3. **Process PDFs and build index**
   ```bash
   # Add PDFs to sample_pdf directory
   pdf-rag pipeline
   ```

4. **Start the web interface**
   ```bash
   pdf-rag-ui
   ```

### Option 3: Local Development

1. **Clone and setup**
   ```bash
   git clone <your-repo-url>
   cd pdf-rag-with-image
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   python main_ui.py
   ```

## 📖 Usage

### Web Interface

1. **Access**: http://localhost:8501
2. **Chat**: Type questions in the chat input
3. **View**: See responses with embedded images and formatting

### Command Line Interface

```bash
# Convert PDFs to markdown
pdf-rag convert

# Build search index  
pdf-rag index

# Ask questions
pdf-rag ask "What is this document about?"

# Interactive mode
pdf-rag interactive

# Full pipeline (convert + index)
pdf-rag pipeline
```

### Python API

```python
from pdf_rag_with_image import RAGSystem, PDFToMarkdown

# Initialize system
rag = RAGSystem()

# Ask questions
answer = rag.query("What are the key features?")
print(answer)
```

## 🏗️ Architecture

```
├── pdf-rag-with-image/        # Main package - Core RAG components
│   │   ├── pdf_processor.py   # PDF to Markdown conversion
│   │   ├── index_builder.py   # FAISS index creation
│   │   ├── rag_system.py      # Main RAG system
│   │   └── llm_client.py      # Gemini LLM client
│   ├── frontend/              # Streamlit web interface
│   │   └── streamlit_app.py   # Chat interface
│   ├── cli.py                 # Command line interface
│   └── main_ui.py             # UI launcher
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── requirements.txt           # Python dependencies
```

## 🛠️ Configuration

### Environment Variables

```bash
# Optional: Set custom paths
export GOOGLE_API_KEY="your-api-key"
export PYTHONPATH="/app"
```

### API Key Setup

Create `gemini_api_key.json` in your working directory:
```json
{
  "api_key": "your-google-gemini-api-key-here"
}
```

Get your API key from: [Google AI Studio](https://makersuite.google.com/app/apikey)

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t pdf-rag-with-image .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/sample_pdf:/app/sample_pdf \
  -v $(pwd)/gemini_api_key.json:/app/gemini_api_key.json:ro \
  pdf-rag-with-image
```

### Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📚 Dependencies

- **streamlit**: Web interface
- **PyMuPDF**: PDF processing
- **langchain**: LLM framework
- **langchain-google-genai**: Google Gemini integration
- **faiss-cpu**: Vector similarity search
- **numpy**: Numerical computations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `gemini_api_key.json` exists and contains valid Google Gemini API key
   - Check file permissions

2. **Images Not Rendering**
   - Verify PDF extraction completed successfully
   - Check that images exist in `extracted_from_pdf/images/`

3. **Docker Issues**
   - Ensure Docker and docker-compose are installed
   - Check port 8501 is not in use

4. **Memory Issues**
   - Large PDFs may require more memory
   - Consider processing smaller batches

### Support

- 🐛 [Report bugs](https://github.com/yourusername/pdf-rag-chatbot/issues)
- 💬 [Discussions](https://github.com/yourusername/pdf-rag-chatbot/discussions)
- 📧 Contact: your.email@example.com

## 🎯 Roadmap

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Multiple LLM providers (OpenAI, Claude)
- [ ] Advanced search filters
- [ ] Document management interface
- [ ] API endpoints for integration
- [ ] Batch processing capabilities

---

**Made with ❤️ by Vivek Singh**