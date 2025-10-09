"""
Image analysis and OCR functionality for RAG system.
Handles LLM-based image processing and filtering.
"""
import re
import json
from typing import List, Dict, Any, Tuple
from .llm_client import LLMClient
from model_management.prompts import get_image_analysis_prompt


class ImageAnalyzer:
    """Handles image analysis and filtering using LLM."""
    
    def __init__(self, llm_client: LLMClient = None):
        """Initialize with LLM client."""
        self.llm_client = llm_client or LLMClient()
    
    def process_chunk_with_llm(self, chunk_text: str, query: str) -> str:
        """Use LLM to process entire chunk and handle all images at once."""
        try:
            # Count images in original chunk
            img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            original_images = re.findall(img_pattern, chunk_text)
            print(f"        🤖 Processing chunk with {len(original_images)} images using LLM...")
            
            # Use centralized prompt
            analysis_prompt = get_image_analysis_prompt(query, chunk_text)
            
            # Use Claude Sonnet 4 to process the chunk
            print(f"        📤 Sending prompt to Claude Sonnet 4 (length: {len(analysis_prompt)} chars)...")
            response = self.llm_client.generate(analysis_prompt)
            print(f"        📥 Claude response received: {len(response)} chars")
            print(f"        📝 Full Claude response: {response}")
            
            # Parse JSON response
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            result = json.loads(clean_response)
            updated_chunk = result.get("updated_chunk", chunk_text)
            
            # Count images in processed chunk
            processed_images = re.findall(img_pattern, updated_chunk)
            print(f"        ✅ Chunk processed: {len(original_images)} → {len(processed_images)} images")
            print(f"        📊 Images removed: {len(original_images) - len(processed_images)}")
            
            return updated_chunk
                
        except Exception as e:
            print(f"        ❌ Error processing chunk: {e}")
            print(f"        ⚠️ Returning original chunk with all {len(original_images)} images due to processing failure")
            # Return original chunk if processing fails
            return chunk_text
    
    def process_chunks_with_images(self, retrieved_chunks: List[Tuple[float, str, Dict[str, Any]]], query: str) -> Tuple[List[Tuple[float, str, Dict[str, Any]]], Dict[str, Any]]:
        """Process chunks to analyze images and enhance content based on relevance to query."""
        filtered_chunks = []
        total_images_found = 0
        images_processed = 0
        images_kept = 0
        images_removed = 0
        chunk_stats = []
        
        for chunk_idx, (score, text, metadata) in enumerate(retrieved_chunks, 1):
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            print(f"   📄 Processing Chunk {chunk_idx}: {source} (Page {page})")
            
            # Find image references in the text
            img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
            images = re.findall(img_pattern, text)
            
            chunk_images_found = len(images)
            
            if images:
                print(f"      🖼️ Found {len(images)} images in chunk")
                total_images_found += len(images)
                
                # Use LLM to process the entire chunk at once
                processed_text = self.process_chunk_with_llm(text, query)
                
                # Count how many images remain after processing
                processed_images = re.findall(img_pattern, processed_text)
                chunk_images_kept = len(processed_images)
                chunk_images_removed = chunk_images_found - chunk_images_kept
                
                # If no images were removed and we had original images, it likely means processing failed
                if chunk_images_removed == 0 and chunk_images_found > 0 and processed_text == text:
                    print(f"      ⚠️ LLM Processing failed: {chunk_images_kept} kept (no filtering applied)")
                else:
                    print(f"      ✅ LLM Processing: {chunk_images_kept} kept, {chunk_images_removed} removed")
                
                images_kept += chunk_images_kept
                images_removed += chunk_images_removed
                images_processed += chunk_images_found
            else:
                print(f"      📝 No images found in chunk")
                processed_text = text
                chunk_images_kept = 0
                chunk_images_removed = 0
            
            # Clean up any extra whitespace  
            processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text).strip()
            filtered_chunks.append((score, processed_text, metadata))
            
            # Store chunk statistics
            chunk_stats.append({
                "chunk_idx": chunk_idx,
                "source": source,
                "page": page,
                "images_found": chunk_images_found,
                "images_kept": chunk_images_kept,
                "images_removed": chunk_images_removed
            })
        
        # Summary
        print(f"📊 Image Analysis Summary:")
        print(f"   Total images found: {total_images_found}")
        print(f"   Images processed: {images_processed}")
        print(f"   Images kept (relevant): {images_kept}")
        print(f"   Images removed (irrelevant): {images_removed}")
        
        image_stats = {
            "total_found": total_images_found,
            "processed": images_processed,
            "kept": images_kept,
            "removed": images_removed,
            "chunk_details": chunk_stats
        }
        
        return filtered_chunks, image_stats