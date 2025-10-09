"""
Centralized prompt management for the RAG system.
Contains all prompts used for image analysis and final response generation.
"""

def get_image_analysis_prompt(query: str, chunk_text: str) -> str:
    """
    Get the prompt for analyzing and filtering images in chunks based on relevance to the query.
    
    Args:
        query: User's question/query
        chunk_text: The text chunk containing images to analyze
        
    Returns:
        Formatted prompt for image analysis
    """
    return f"""You are processing a text chunk to answer the user query: "{query}"

Original chunk: {chunk_text}

CRITICAL TASK: Be EXTREMELY SELECTIVE. COMPLETELY REMOVE image URLs that are not essential.

KEEP IMAGE (include the ![...](...) markdown) ONLY if it shows:
- The EXACT interface/screen the user needs to see for their query
- SPECIFIC buttons, forms, or fields the user must interact with
- PRECISE step-by-step visual instructions for the queried task
- ESSENTIAL diagrams that directly explain the concept being asked
- CRITICAL screenshots that show the exact process mentioned in the query

COMPLETELY REMOVE IMAGE URL (delete the entire ![...](...) line) for EVERYTHING else including:
- General interfaces not directly needed for the query
- Overview screens, dashboards, or home pages
- Navigation elements, menus, headers, footers
- Logos, branding, decorative elements
- Examples that don't match the exact query
- Supporting or contextual visuals
- Generic screenshots that don't show the specific action
- Images that are "nice to have" but not "must have"

DEFAULT DECISION: When in doubt, COMPLETELY REMOVE the image URL. Only keep if absolutely essential.

EXAMPLES:
- Query "How to book a ticket?" → KEEP: booking form, payment screen → COMPLETELY REMOVE: homepage, menu, logo
- Query "What is refund policy?" → KEEP: refund policy text/table → COMPLETELY REMOVE: booking screens, general terms
- Query "How to login?" → KEEP: login form → COMPLETELY REMOVE: after-login dashboard, general app interface

IMPORTANT: For irrelevant images, DO NOT include them in the output at all. Delete the entire ![...](...) line.

CRITICAL: For the few relevant images you keep, you MUST provide BOTH the image URL AND a detailed summary that will help the final response generation:
![alt text](image_url)
[Image Summary: Detailed description of what the image shows and specifically how it helps answer the user's query. Include key elements, text, buttons, forms, or interface elements visible in the image that are relevant to the question.]

The image summary is ESSENTIAL because:
1. It provides context for the final response generation
2. It helps the AI understand what the image contains without seeing it
3. It enables better integration of visual information into the text response
4. It ensures the final response can reference specific image content meaningfully

Return JSON:
{{"updated_chunk": "the processed chunk text with ONLY absolutely essential image URLs and their detailed summaries. All irrelevant image URLs must be completely deleted from the text."}}"""


def get_final_response_prompt(question: str, context: str) -> str:
    """
    Get the prompt for generating the final HTML response based on the processed context.
    
    Args:
        question: User's question
        context: Processed context with filtered images
        
    Returns:
        Formatted prompt for final response generation
    """
    return f"""You are an AI assistant tasked with creating informative HTML content based on a given query and context. Your response should be well-structured, informative, and visually appealing.

IMPORTANT: Return ONLY the HTML content without any markdown code blocks, backticks, or ```html tags. Start directly with HTML tags.

Query: {question}

Context: {context}

Instructions:
1. Create an HTML response that answers the query using information from the context.
2. Use appropriate HTML tags to structure your response (e.g., <h1>, <h2>, <p>, <ul>, <li>).
3. CRITICAL: When you find images in the context, pay attention to both the image URL and the [Image Summary] that follows it.
4. Use the image summaries to provide rich, detailed descriptions in your response, even if the image isn't displayed.
5. If you reference any images from the context, use the exact file path provided in the context.
6. Use the following format for images: <img src="[IMAGE_PATH]" alt="[DESCRIPTION]" style="max-width: 100%; height: auto;">
7. Create detailed captions based on the image summaries using a <p> tag with class "caption".
8. INTEGRATE image information: Use the image summaries to enhance your textual explanations.
9. Use inline CSS to style your HTML for better readability.
10. If no relevant information is found in the context, state that clearly.
11. Do not invent or assume any information not present in the context or image summaries.
12. Transform image paths to this format: "../extracted_from_pdf/images/page_X_img_Y_hash.png"
13. DO NOT include any "Additional Context", "Content Analysis Summary", or similar meta-information sections.
14. Focus ONLY on directly answering the user's question with the relevant information from the context.

IMPORTANT: The context may contain [Image Summary: ...] sections that describe what's in the images. Use this information to:
- Provide detailed explanations of interface elements
- Describe buttons, forms, and controls mentioned in the summaries
- Reference specific visual elements that help answer the user's question
- Create comprehensive responses that combine text and visual information

Your response should be complete, well-formatted HTML that directly answers the question without any additional meta-sections.

Example: 
 image_path =  "../extracted_from_pdf/images/page_8_img_0_0bc3a9d8.png"

html_content = f'''
<div style="font-family: Arial, sans-serif; max-width: 800px; margin: auto;">
<h1 style="color: #4285F4;">Google Image Search</h1>

<p style="font-size: 16px; line-height: 1.5;">
Google Image Search is a powerful tool that allows users to search the web for image content. 
It uses advanced algorithms to analyze and categorize images, making it easier for users to 
find specific visual content quickly and efficiently.
</p>

<h2 style="color: #34A853;">Key Features:</h2>
<ul style="font-size: 16px; line-height: 1.5;">
    <li>Visual search capabilities</li>
    <li>Filter options (size, color, type, etc.)</li>
    <li>Reverse image search</li>
    <li>Integration with Google Lens</li>
</ul>

<p style="font-size: 16px; line-height: 1.5;">
Below is an image demonstrating the Google Image Search interface:
</p>

<img src="image_path" alt="Google Image Search Interface" style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px;">

<p style="font-size: 14px; color: #666; font-style: italic;">
Image: Example of Google Image Search results page
</p>
</div>
'''
"""