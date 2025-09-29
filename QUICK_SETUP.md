# Quick Setup Guide for Friends 🚀

This guide helps you quickly get the PDF RAG system running on your machine.

## Option 1: Using Docker (Recommended) 🐳

### Prerequisites
- Docker installed on your machine
- A Google Gemini API key

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/vivek-slngh/pdf-rag-with-image.git
   cd pdf-rag-with-image
   ```

2. **Create API key file:**
   ```bash
   echo '{"api_key": "YOUR_GEMINI_API_KEY_HERE"}' > gemini_api_key.json
   ```

3. **Add your PDF files:**
   ```bash
   mkdir -p sample_pdf
   # Copy your PDF files to the sample_pdf directory
   ```

4. **Build the index:**
   ```bash
   python pdf_rag_with_image/index_builder.py
   ```

5. **Run with Docker:**
   ```bash
   docker-compose up
   ```

6. **Access the app:**
   Open http://localhost:8501 in your browser

## Option 2: Using pip install 📦

### Prerequisites
- Python 3.8 or higher
- A Google Gemini API key

### Steps
1. **Install the package:**
   ```bash
   pip install pdf-rag-with-image
   ```

2. **Create a project directory:**
   ```bash
   mkdir my-pdf-rag
   cd my-pdf-rag
   ```

3. **Create API key file:**
   ```bash
   echo '{"api_key": "YOUR_GEMINI_API_KEY_HERE"}' > gemini_api_key.json
   ```

4. **Add your PDF files:**
   ```bash
   mkdir sample_pdf
   # Copy your PDF files to the sample_pdf directory
   ```

5. **Build the index:**
   ```bash
   python -m pdf_rag_with_image.index_builder
   ```

6. **Run the application:**
   ```bash
   pdf-rag-ui
   ```

7. **Access the app:**
   Open http://localhost:8501 in your browser

## Getting a Gemini API Key 🔑

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the generated key
4. Use it in your `gemini_api_key.json` file

## Troubleshooting 🔧

### Common Issues:

1. **"API key not found" error:**
   - Make sure `gemini_api_key.json` exists in your project directory
   - Verify the JSON format is correct: `{"api_key": "your-key-here"}`

2. **Docker issues:**
   - Make sure Docker is running
   - Try `docker-compose down` then `docker-compose up` again

3. **No PDFs showing up:**
   - Make sure PDF files are in the `sample_pdf` directory
   - Check file permissions

4. **Port already in use:**
   - Change the port in `docker-compose.yml` from `8501:8501` to `8502:8501`
   - Access via http://localhost:8502

## Features ✨

- Upload and process PDF documents
- Ask questions about your PDFs
- View images from PDFs in responses  
- Semantic search across document content
- Clean, responsive web interface

## Need Help? 🤝

- Check the [full documentation](README.md)
- [Report issues](https://github.com/vivek-slngh/pdf-rag-with-image/issues)
- Contact: vivekkgp97@gmail.com