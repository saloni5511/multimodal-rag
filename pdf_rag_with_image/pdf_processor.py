import fitz  # PyMuPDF
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import re


class PDFToMarkdown:
    def __init__(self, output_dir: str = "extracted_from_pdf/extracted_content", images_dir: str = "extracted_from_pdf/images", sample_pdf_dir: str = "sample_pdf"):
        self.output_dir = Path(output_dir)
        self.images_dir = Path(images_dir)
        self.sample_pdf_dir = Path(sample_pdf_dir)
        
        # Create directories if they don't exist (including parent directories)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sample_pdf_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir.absolute()}")
        print(f"Images directory: {self.images_dir.absolute()}")
        print(f"Sample PDF directory: {self.sample_pdf_dir.absolute()}")
    
    def extract_text_and_images(self, pdf_path: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract text blocks and images from PDF."""
        doc = fitz.open(pdf_path)
        text_blocks = []
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text blocks with their positions
            blocks = page.get_text("dict")
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"]
                        text += "\n"
                    
                    if text.strip():
                        text_blocks.append({
                            "page": page_num + 1,
                            "text": text.strip(),
                            "bbox": block["bbox"]
                        })
            
            # Extract images
            image_list = page.get_images()
            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    
                    # Generate unique filename
                    img_hash = hashlib.md5(img_data).hexdigest()[:8]
                    img_filename = f"page_{page_num + 1}_img_{img_index}_{img_hash}.png"
                    img_path = self.images_dir / img_filename
                    
                    # Save image
                    with open(img_path, "wb") as f:
                        f.write(img_data)
                    
                    images.append({
                        "page": page_num + 1,
                        "filename": img_filename,
                        "path": str(img_path),
                        "bbox": None  # Skip bbox for now to avoid the error
                    })
                
                pix = None
        
        doc.close()
        return text_blocks, images
    
    def convert_to_markdown(self, text_blocks: List[Dict], images: List[Dict], pdf_name: str) -> str:
        """Convert extracted content to markdown format."""
        markdown_content = []
        markdown_content.append(f"# {pdf_name}\n")
        
        # Group content by page
        pages = {}
        for block in text_blocks:
            page = block["page"]
            if page not in pages:
                pages[page] = {"text": [], "images": []}
            pages[page]["text"].append(block)
        
        for img in images:
            page = img["page"]
            if page not in pages:
                pages[page] = {"text": [], "images": []}
            pages[page]["images"].append(img)
        
        # Generate markdown page by page
        for page_num in sorted(pages.keys()):
            markdown_content.append(f"\n## Page {page_num}\n")
            
            page_data = pages[page_num]
            
            # Add text blocks
            for text_block in page_data["text"]:
                text = text_block["text"]
                
                # Basic formatting detection
                if self._is_heading(text):
                    markdown_content.append(f"### {text}\n")
                elif self._is_list_item(text):
                    markdown_content.append(f"- {text}\n")
                else:
                    markdown_content.append(f"{text}\n")
            
            # Add images
            for img in page_data["images"]:
                # Use relative path from extracted_content to images
                img_url = f"../images/{img['filename']}"
                markdown_content.append(f"\n![Image from page {page_num}]({img_url})\n")
        
        return "\n".join(markdown_content)
    
    def _is_heading(self, text: str) -> bool:
        """Detect if text is likely a heading."""
        return (len(text) < 100 and 
                text.isupper() or 
                re.match(r'^\d+\.?\s+[A-Z]', text) or
                text.endswith(':'))
    
    def _is_list_item(self, text: str) -> bool:
        """Detect if text is a list item."""
        return re.match(r'^\s*[•\-\*]\s+', text) or re.match(r'^\s*\d+[\.\)]\s+', text)
    
    def process_pdf(self, pdf_path: str) -> str:
        """Main method to process PDF and generate markdown."""
        pdf_name = Path(pdf_path).stem
        
        # Extract content
        text_blocks, images = self.extract_text_and_images(pdf_path)
        
        # Convert to markdown
        markdown_content = self.convert_to_markdown(text_blocks, images, pdf_name)
        
        # Save markdown file
        md_filename = f"{pdf_name}.md"
        md_path = self.output_dir / md_filename
        
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Processed PDF: {pdf_path}")
        print(f"Markdown saved: {md_path}")
        print(f"Images saved in: {self.images_dir}")
        print(f"Extracted {len(images)} images")
        
        return str(md_path)
    
    def process_sample_pdfs(self) -> List[str]:
        """Process all PDF files in the sample_pdf directory."""
        pdf_files = list(self.sample_pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.sample_pdf_dir}")
            return []
        
        processed_files = []
        print(f"Found {len(pdf_files)} PDF files to process:")
        
        for pdf_file in pdf_files:
            print(f"\nProcessing: {pdf_file.name}")
            try:
                md_path = self.process_pdf(str(pdf_file))
                processed_files.append(md_path)
                print(f"✓ Successfully processed: {pdf_file.name}")
            except Exception as e:
                print(f"✗ Error processing {pdf_file.name}: {str(e)}")
        
        print(f"\nProcessing complete! {len(processed_files)} files converted successfully.")
        return processed_files
    
    def get_sample_pdf_files(self) -> List[Path]:
        """Get list of PDF files in sample_pdf directory."""
        return list(self.sample_pdf_dir.glob("*.pdf"))


# Example usage
if __name__ == "__main__":
    # Initialize converter with new directory structure
    converter = PDFToMarkdown(
        output_dir="extracted_from_pdf/extracted_content",
        images_dir="extracted_from_pdf/images", 
        sample_pdf_dir="sample_pdf"
    )
    
    print("\n=== PDF to Markdown Converter ===")
    
    # Check for PDF files in sample_pdf directory
    pdf_files = converter.get_sample_pdf_files()
    
    if pdf_files:
        print(f"\nFound PDF files in sample_pdf directory:")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"  {i}. {pdf_file.name}")
        
        # Process all PDFs in sample_pdf directory
        print("\nProcessing all PDF files...")
        processed = converter.process_sample_pdfs()
        
        if processed:
            print(f"\n✓ Successfully converted {len(processed)} PDF(s) to Markdown!")
            print("\nGenerated files:")
            for md_file in processed:
                print(f"  → {md_file}")
        else:
            print("\n✗ No files were successfully processed.")
    else:
        print("\nNo PDF files found in sample_pdf directory.")
        print("Place your PDF files in the 'sample_pdf' folder and run again.")
    
    print("\n=== Usage Options ===")
    print("1. Place PDFs in 'sample_pdf/' folder and run this script")
    print("2. Use programmatically: converter.process_pdf('path/to/file.pdf')")
    print("3. Process all sample PDFs: converter.process_sample_pdfs()")