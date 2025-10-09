# Multi-Modal PDF RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that processes PDF documents with intelligent image filtering and provides contextual answers through a Streamlit interface.

## Overview

This system combines text and image processing to create a comprehensive knowledge base from PDF documents. It uses Claude Sonnet 4 for intelligent image analysis and filtering, ensuring only relevant visual content is included in responses.

## System Architecture

```
PDF Upload → Text Extraction → Image Processing → Vector Indexing → Query Processing → Response Generation
```

## UI Flow

### 1. PDF Management (Sidebar)

#### Upload Process
1. **Upload PDF**: Users upload PDF files through the file uploader
2. **Status Update**: System immediately shows PDF status as "❌ Not embedded"
3. **File Storage**: PDF is saved to `sample_pdf/` directory
4. **Status Tracking**: `pdf_status.json` updated with `{"filename.pdf": {"embed": false}}`

#### Build Embeddings
1. **Trigger**: Click "🚀 Build Embedding" button
2. **Processing**: System processes only PDFs with `embed: false` status
3. **Steps**:
   - Extract text and images from PDFs using PyMuPDF
   - Convert to markdown format with image references
   - Create vector embeddings using Cohere model
   - Build FAISS index for similarity search
   - Update status to `embed: true`

#### PDF Status Display
- **Green ✅ Embedded**: PDF is processed and ready for queries
- **Red ❌ Not embedded**: PDF needs processing
- **Delete 🗑️**: Remove PDF and clear from index

### 2. Chat Interface (Main Area)

#### Query Processing Flow

When a user submits a question, the system follows this sequential process:

##### Step 1: Extract Chunks
```
📤 Extracting relevant chunks...
```
- Retrieves top 5 most relevant text chunks from FAISS vector store
- Sorts chunks by similarity score (highest first)
- Shows chunk details:
  ```
  ✅ Extracted 5 relevant chunks with 10 images:
  📄 Document.pdf (Page 9) - Score: 1.149 - 2 images
  📄 Document.pdf (Page 10) - Score: 0.876 - 2 images
  ...
  ```

##### Step 2: Image Analysis
```
🖼️ Analyzing images...
```
- Processes each chunk using Claude Sonnet 4
- Analyzes image relevance to user query
- Filters out irrelevant images
- Adds detailed summaries for kept images
- Shows processing results:
  ```
  🖼️ Image Processing Results:
  📄 Document.pdf (Page 9): 0 removed
  📄 Document.pdf (Page 10): 2 removed
  
  Summary: 4 kept, 6 removed from 10 found
  ```

##### Step 3: Generate Answer
```
🤖 Generating answer from updated chunks...
```
- Sends filtered chunks to Claude Sonnet 4
- Generates HTML response with:
  - Structured content using proper HTML tags
  - Embedded images (converted to base64)
  - Detailed captions based on image summaries
  - Styled formatting for readability

##### Step 4: Display Results
- **HTML Response**: Rich formatted answer with embedded images
- **Final Chunks**: Expandable sections showing processed content sent to LLM

### 3. Response Display Features

#### Rich HTML Rendering
- **Images**: Automatically converted from relative paths to base64
- **Styling**: Inline CSS for professional appearance
- **Structure**: Proper headers, paragraphs, lists, and captions
- **Responsiveness**: Images scale to fit container

#### Debug Information
- **Chunk Details**: View exact content sent to final LLM
- **Image Statistics**: See how many images were filtered per chunk
- **Processing Flow**: Clear visual feedback for each step

## Technical Components

### Configuration
- **config.json**: Centralized settings for chunking, retrieval, and model configuration
- **Chunking**: 600 characters with 90 character overlap
- **Retrieval**: Top 5 chunks with normalized embeddings
- **Models**: Claude Sonnet 4 for generation, Cohere for embeddings

### Image Processing
- **Extraction**: Images saved to `extracted_from_pdf/images/{pdf_name}/`
- **Analysis**: Claude Sonnet 4 evaluates relevance to user query
- **Filtering**: Only essential images kept with detailed summaries
- **Rendering**: Base64 conversion for Streamlit HTML display

### Vector Store
- **FAISS**: Efficient similarity search
- **Embeddings**: Cohere English v3 model via AWS Bedrock
- **Indexing**: Complete rebuild when new PDFs added
- **Storage**: Persistent index in `md_faiss_index/`

## Code Architecture

### High-Level Query Processing

The core query processing is handled by the `RAGSystem.query()` method:

```python
def query(self, question: str, top_k: int = None, retrieved_chunks: List[Tuple[float, str, Dict[str, Any]]] = None) -> str:
    """Perform RAG query: retrieve chunks and generate answer."""
    
    # Step 1: Retrieve relevant chunks from FAISS index
    retrieved_chunks = self.retrieve(question, top_k)
    
    # Step 2: Analyze and filter images using Claude Sonnet 4
    filtered_chunks, image_stats = self.image_analyzer.process_chunks_with_images(retrieved_chunks, question)
    
    # Step 3: Prepare context from filtered chunks
    context_parts = []
    for score, text, metadata in filtered_chunks:
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', '')
        page_info = f"Page {page}" if page else "Header"
        context_parts.append(f"[{source} - {page_info}] {text}")
    
    context = "\n\n".join(context_parts)
    
    # Step 4: Generate final HTML response using centralized prompt
    prompt = get_final_response_prompt(question, context)
    answer = self.llm_client.generate(prompt)
    
    # Step 5: Convert relative image paths to base64 for rendering
    processed_answer = self.process_html_images(answer)
    
    return processed_answer
```

### Key Components

#### 1. Vector Retrieval (`retrieve()`)
```python
def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict[str, Any]]]:
    # FAISS similarity search with Cohere embeddings
    docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
    # Sort by relevance score (descending)
    retrieved_chunks.sort(key=lambda x: x[0], reverse=True)
```

#### 2. Image Analysis (`ImageAnalyzer.process_chunks_with_images()`)
```python
def process_chunks_with_images(self, retrieved_chunks, query):
    for chunk in retrieved_chunks:
        # Use Claude Sonnet 4 to analyze image relevance
        processed_text = self.process_chunk_with_llm(chunk_text, query)
        # Track statistics: found, kept, removed
    return filtered_chunks, image_stats
```

#### 3. HTML Processing (`process_html_images()`)
```python
def process_html_images(self, html_content: str) -> str:
    # Find image tags with relative paths
    # Convert ../extracted_from_pdf/images/... to base64
    # Search in PDF subdirectories if needed
    return processed_html_with_base64_images
```

### Data Flow

```
User Query
    ↓
FAISS Vector Search (Cohere embeddings)
    ↓
Retrieved Chunks [(score, text, metadata), ...]
    ↓
Claude Sonnet 4 Image Analysis
    ↓
Filtered Chunks + Image Statistics
    ↓
Context Preparation
    ↓
Claude Sonnet 4 Final Response Generation
    ↓
HTML Processing (Base64 Image Conversion)
    ↓
Final HTML Response
```

## Key Features

1. **Intelligent Image Filtering**: Only relevant images included in responses
2. **Real-time Processing Feedback**: Clear status updates throughout the pipeline
3. **Dual Model Architecture**: Claude for reasoning, Cohere for embeddings
4. **Rich HTML Responses**: Professional formatting with embedded visuals
5. **Session Management**: Prevents duplicate uploads and processing
6. **Comprehensive Debugging**: Full visibility into processing decisions

## Dependencies

- **Streamlit**: Web interface
- **PyMuPDF**: PDF processing
- **FAISS**: Vector similarity search
- **LangChain**: LLM orchestration
- **AWS Bedrock**: Claude and Cohere model access
- **Base64**: Image encoding for web display

## Usage

1. Start the application: `streamlit run run_app.py`
2. Upload PDF documents using the sidebar
3. Click "Build Embedding" to process documents
4. Ask questions in the chat interface
5. View rich responses with filtered images and detailed explanations

The system provides complete transparency into its decision-making process while delivering high-quality, contextually relevant responses with appropriate visual content.

---

**Made by Saloni Rai**