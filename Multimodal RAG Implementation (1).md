Here's how to handle PDF documents with mixed content (text, images, diagrams) for embeddings:

## Step 1: PDF Content Extraction

**Use Azure Document Intelligence (formerly Form Recognizer)**:

```python
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

# Initialize client
client = DocumentIntelligenceClient(
    endpoint="your-endpoint", 
    credential=AzureKeyCredential("your-key")
)

# Extract content with layout analysis
with open("document.pdf", "rb") as f:
    poller = client.begin_analyze_document(
        "prebuilt-layout", 
        document=f,
        output_content_format="markdown"  # Preserves structure
    )
result = poller.result()

```

This extracts:

-   Text content with structure preserved
-   Tables as structured data
-   Images and their positions
-   Diagrams and figures

## Step 2: Content Processing Strategy

**Chunking Strategy for Mixed Content**:

```python
def process_mixed_content(document_result):
    chunks = []
    
    for page in document_result.pages:
        # Text chunks
        text_content = extract_text_from_page(page)
        if text_content:
            chunks.append({
                'content': text_content,
                'type': 'text',
                'page': page.page_number
            })
        
        # Image chunks with OCR
        for figure in page.figures:
            image_data = extract_image(figure)
            ocr_text = perform_ocr(image_data)
            chunks.append({
                'content': f"Image description: {ocr_text}",
                'image_data': image_data,
                'type': 'image',
                'page': page.page_number
            })
    
    return chunks

```

## Step 3: Multi-Modal Embedding Generation

**Hybrid Embedding Approach**:

```python
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from openai import AzureOpenAI

# Text embeddings
def get_text_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

# Image embeddings using Azure AI Vision
def get_image_embedding(image_data):
    vision_client = ImageAnalysisClient(
        endpoint="your-vision-endpoint",
        credential=AzureKeyCredential("your-key")
    )
    
    # Get image features and description
    result = vision_client.analyze(
        image_data=image_data,
        visual_features=[
            VisualFeatures.CAPTION,
            VisualFeatures.DENSE_CAPTIONS,
            VisualFeatures.OBJECTS,
            VisualFeatures.TAGS
        ]
    )
    
    # Create text description from visual analysis
    description = f"Image contains: {result.caption.text}. Objects: {', '.join([obj.name for obj in result.objects])}"
    
    # Embed the description
    return get_text_embedding(description)

```

## Step 4: Enhanced Processing for Diagrams

**For technical diagrams and charts**:

```python
def process_diagram(image_data):
    # Use GPT-4V for diagram understanding
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this diagram/chart in detail, including any text, data, relationships, and technical content:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    )
    
    diagram_description = response.choices[0].message.content
    return get_text_embedding(diagram_description)

```

## Step 5: Azure AI Search Integration

**Store in vector database**:

```python
def store_in_search_index(chunks):
    for chunk in chunks:
        document = {
            "id": f"{chunk['page']}_{chunk['type']}_{hash(chunk['content'])}",
            "content": chunk['content'],
            "content_type": chunk['type'],
            "page_number": chunk['page'],
            "embedding": chunk['embedding'],
            "source_file": "document.pdf"
        }
        
        search_client.upload_documents([document])

```

## **Step 6: Multimodal Document Retrieval**

### Query Processing and Embedding

```python
def process_query(query, query_type="text"):
    """Process different types of queries"""
    
    if query_type == "text":
        # Text query embedding
        query_embedding = get_text_embedding(query)
        
    elif query_type == "image":
        # Image query - convert to description then embed
        image_description = process_diagram(query)  # query is image data
        query_embedding = get_text_embedding(image_description)
        
    elif query_type == "multimodal":
        # Combined text + image query
        text_part, image_part = query['text'], query['image']
        text_embedding = get_text_embedding(text_part)
        image_description = process_diagram(image_part)
        image_embedding = get_text_embedding(image_description)
        
        # Combine embeddings (weighted average or concatenation)
        query_embedding = combine_embeddings(text_embedding, image_embedding)
    
    return query_embedding

```

### Hybrid Search Implementation

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

def retrieve_multimodal_documents(query, query_type="text", top_k=10):
    """Retrieve relevant multimodal content"""
    
    # Process query to get embedding
    query_embedding = process_query(query, query_type)
    
    # Create vector query
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    
    # Hybrid search with both vector and text search
    search_results = search_client.search(
        search_text=query if query_type == "text" else "",
        vector_queries=[vector_query],
        select=["id", "content", "content_type", "page_number", "source_file"],
        top=top_k
    )
    
    return list(search_results)

```

### Content-Type Aware Retrieval

```python
def retrieve_by_content_type(query, content_types=["text", "image"], top_k=10):
    """Retrieve specific content types"""
    
    query_embedding = process_query(query)
    
    # Filter by content type
    filter_expr = " or ".join([f"content_type eq '{ct}'" for ct in content_types])
    
    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    
    search_results = search_client.search(
        vector_queries=[vector_query],
        filter=filter_expr,
        select=["id", "content", "content_type", "page_number", "source_file"],
        top=top_k
    )
    
    return list(search_results)

```

### Multi-Step Retrieval for Complex Queries

```python
def multi_step_retrieval(complex_query, top_k=20):
    """Handle complex queries requiring multiple retrieval steps"""
    
    # Step 1: Initial broad retrieval
    initial_results = retrieve_multimodal_documents(complex_query, top_k=top_k*2)
    
    # Step 2: Re-rank based on content type diversity
    ranked_results = rerank_by_diversity(initial_results)
    
    # Step 3: Context-aware filtering
    filtered_results = filter_by_context(ranked_results, complex_query)
    
    return filtered_results[:top_k]

def rerank_by_diversity(results):
    """Ensure diverse content types in results"""
    text_results = [r for r in results if r['content_type'] == 'text']
    image_results = [r for r in results if r['content_type'] == 'image']
    
    # Interleave results to ensure diversity
    diverse_results = []
    max_len = max(len(text_results), len(image_results))
    
    for i in range(max_len):
        if i < len(text_results):
            diverse_results.append(text_results[i])
        if i < len(image_results):
            diverse_results.append(image_results[i])
    
    return diverse_results

```

### RAG Generation with Multimodal Context

```python
def generate_answer_with_multimodal_context(query, retrieved_docs):
    """Generate answer using multimodal retrieved context"""
    
    # Prepare context from different content types
    text_context = []
    image_context = []
    
    for doc in retrieved_docs:
        if doc['content_type'] == 'text':
            text_context.append(f"Page {doc['page_number']}: {doc['content']}")
        elif doc['content_type'] == 'image':
            image_context.append(f"Image from page {doc['page_number']}: {doc['content']}")
    
    # Create comprehensive context
    full_context = "TEXT CONTENT:\n" + "\n".join(text_context)
    if image_context:
        full_context += "\n\nIMAGE CONTENT:\n" + "\n".join(image_context)
    
    # Generate answer with GPT-4
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions using both text and image content from documents. Reference specific pages when possible."
            },
            {
                "role": "user",
                "content": f"Context: {full_context}\n\nQuestion: {query}\n\nAnswer:"
            }
        ]
    )
    
    return response.choices[0].message.content

```

### Complete Retrieval Workflow

```python
def complete_multimodal_rag(user_query, query_type="text"):
    """End-to-end multimodal RAG pipeline"""
    
    # Step 1: Retrieve relevant multimodal documents
    retrieved_docs = retrieve_multimodal_documents(
        query=user_query, 
        query_type=query_type, 
        top_k=10
    )
    
    # Step 2: Post-process and rank results
    processed_docs = multi_step_retrieval(user_query, top_k=5)
    
    # Step 3: Generate answer with multimodal context
    answer = generate_answer_with_multimodal_context(user_query, processed_docs)
    
    # Step 4: Return structured response
    return {
        "answer": answer,
        "sources": [
            {
                "content": doc['content'][:200] + "...",
                "type": doc['content_type'],
                "page": doc['page_number'],
                "file": doc['source_file']
            }
            for doc in processed_docs
        ]
    }

# Example usage
result = complete_multimodal_rag("What does the architecture diagram show about the system components?")
print(result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(f"- {source['type']} from page {source['page']}: {source['content']}")

```

## Advanced Retrieval Features

### Similarity Thresholding

```python
def retrieve_with_confidence_threshold(query, threshold=0.7):
    """Only return results above confidence threshold"""
    results = retrieve_multimodal_documents(query, top_k=20)
    filtered_results = [r for r in results if r.get('@search.score', 0) > threshold]
    return filtered_results

```

### Cross-Modal Retrieval

```python
def cross_modal_retrieval(text_query):
    """Find images that match text query and vice versa"""
    # Find images that relate to text query
    image_results = retrieve_by_content_type(text_query, content_types=["image"])
    
    # Find text that relates to the query
    text_results = retrieve_by_content_type(text_query, content_types=["text"])
    
    return {
        "related_images": image_results,
        "related_text": text_results
    }

```

This comprehensive retrieval system handles various query types and ensures you get the most relevant multimodal content from your PDF documents, whether users are asking about text, diagrams, images, or complex relationships between different content types.

## Complete Workflow

1.  **Extract**: Use Document Intelligence for layout-aware extraction
2.  **Separate**: Split text, images, and diagrams into chunks
3.  **Enhance**: Use GPT-4V for complex diagram descriptions
4.  **Embed**: Generate embeddings for all content types
5.  **Store**: Index in Azure AI Search with metadata
6.  **Query**: Use hybrid search (vector + keyword) for retrieval

This approach ensures you capture the full semantic meaning of your PDF documents, including visual elements that traditional text-only processing would miss.

===================================================
Here's how to automatically determine the query type from user input:

## **Step 1: Input Type Detection**

```python
import base64
import mimetypes
from PIL import Image
import io

def detect_input_type(user_input):
    """Detect if input is text, image, or mixed"""
    
    if isinstance(user_input, dict):
        # Structured input with multiple components
        return analyze_structured_input(user_input)
    
    elif isinstance(user_input, str):
        # Check if it's base64 encoded image
        if is_base64_image(user_input):
            return "image"
        # Check if it contains image references
        elif contains_image_references(user_input):
            return "multimodal"
        else:
            return "text"
    
    elif isinstance(user_input, bytes):
        # Binary data - likely an image
        return "image"
    
    else:
        return "text"

def is_base64_image(string):
    """Check if string is base64 encoded image"""
    try:
        if string.startswith('data:image'):
            return True
        # Try to decode as base64 and check if it's an image
        decoded = base64.b64decode(string)
        Image.open(io.BytesIO(decoded))
        return True
    except:
        return False

def contains_image_references(text):
    """Check if text contains references to visual elements"""
    visual_keywords = [
        'image', 'diagram', 'chart', 'graph', 'figure', 'picture', 
        'screenshot', 'visualization', 'plot', 'illustration'
    ]
    return any(keyword in text.lower() for keyword in visual_keywords)

```

## **Step 2: Content-Based Query Classification**

```python
from openai import AzureOpenAI

def classify_text_query_intent(text_query):
    """Classify what type of content the text query is asking about"""
    
    classification_prompt = f"""
    Classify this query into one of these categories:
    - "text_seeking": Looking for textual information, explanations, definitions
    - "visual_seeking": Looking for diagrams, charts, images, visual elements
    - "mixed_seeking": Looking for both text and visual information
    - "diagram_analysis": Specifically asking to analyze or understand a diagram/chart
    
    Query: "{text_query}"
    
    Respond with only the category name.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": classification_prompt}],
        max_tokens=20
    )
    
    return response.choices[0].message.content.strip()

def analyze_query_keywords(text_query):
    """Analyze keywords to determine content type preference"""
    
    text_indicators = [
        'explain', 'define', 'what is', 'describe', 'tell me about',
        'summary', 'details', 'information', 'text', 'content'
    ]
    
    visual_indicators = [
        'show', 'diagram', 'chart', 'graph', 'image', 'picture',
        'visualization', 'figure', 'illustration', 'screenshot',
        'architecture', 'flowchart', 'layout', 'design'
    ]
    
    analysis_indicators = [
        'analyze', 'interpret', 'understand', 'breakdown',
        'components', 'structure', 'relationships', 'connections'
    ]
    
    text_score = sum(1 for indicator in text_indicators if indicator in text_query.lower())
    visual_score = sum(1 for indicator in visual_indicators if indicator in text_query.lower())
    analysis_score = sum(1 for indicator in analysis_indicators if indicator in text_query.lower())
    
    return {
        'text_score': text_score,
        'visual_score': visual_score,
        'analysis_score': analysis_score,
        'dominant_type': max([
            ('text_seeking', text_score),
            ('visual_seeking', visual_score),
            ('analysis_seeking', analysis_score)
        ], key=lambda x: x[1])[0]
    }

```

## **Step 3: Image Content Classification**

```python
def classify_image_content(image_data):
    """Determine if image contains diagram, chart, or general image"""
    
    # Use GPT-4V to classify image content
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    classification_prompt = """
    Classify this image into one of these categories:
    - "diagram": Technical diagrams, flowcharts, architecture diagrams, process flows
    - "chart": Graphs, plots, data visualizations, charts, tables
    - "document": Text-heavy images, scanned documents, forms
    - "general": General images, photos, illustrations
    - "mixed": Contains multiple types of content
    
    Respond with only the category name.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": classification_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=20
    )
    
    return response.choices[0].message.content.strip()

```

## **Step 4: Comprehensive Query Type Determiner**

```python
def determine_query_type(user_input):
    """Main function to determine comprehensive query type"""
    
    # Step 1: Basic input type detection
    basic_type = detect_input_type(user_input)
    
    if basic_type == "text":
        # Analyze text query intent
        intent = classify_text_query_intent(user_input)
        keyword_analysis = analyze_query_keywords(user_input)
        
        query_classification = {
            'input_type': 'text',
            'content_seeking': intent,
            'keyword_analysis': keyword_analysis,
            'retrieval_strategy': determine_retrieval_strategy(intent, keyword_analysis)
        }
        
    elif basic_type == "image":
        # Classify image content
        if isinstance(user_input, str) and user_input.startswith('data:image'):
            image_data = base64.b64decode(user_input.split(',')[1])
        else:
            image_data = user_input
            
        image_type = classify_image_content(image_data)
        
        query_classification = {
            'input_type': 'image',
            'image_content_type': image_type,
            'retrieval_strategy': 'image_similarity_search'
        }
        
    elif basic_type == "multimodal":
        # Handle mixed input
        query_classification = {
            'input_type': 'multimodal',
            'retrieval_strategy': 'hybrid_multimodal_search'
        }
    
    return query_classification

def determine_retrieval_strategy(intent, keyword_analysis):
    """Determine the best retrieval strategy based on analysis"""
    
    if intent == "visual_seeking" or keyword_analysis['visual_score'] > keyword_analysis['text_score']:
        return "visual_content_priority"
    elif intent == "text_seeking":
        return "text_content_priority"
    elif intent == "diagram_analysis":
        return "diagram_analysis_focus"
    else:
        return "balanced_multimodal"

```

## **Step 5: Query Processing Pipeline**

```python
def process_user_query(user_input):
    """Complete query processing pipeline"""
    
    # Determine query type
    query_classification = determine_query_type(user_input)
    
    # Process based on classification
    if query_classification['input_type'] == 'text':
        processed_query = {
            'query': user_input,
            'type': 'text',
            'strategy': query_classification['retrieval_strategy'],
            'content_preference': query_classification['content_seeking']
        }
        
    elif query_classification['input_type'] == 'image':
        # Convert image to searchable description
        image_description = process_diagram(user_input)
        processed_query = {
            'query': image_description,
            'type': 'image',
            'original_image': user_input,
            'image_type': query_classification['image_content_type'],
            'strategy': 'image_similarity_search'
        }
        
    elif query_classification['input_type'] == 'multimodal':
        processed_query = {
            'query': user_input,
            'type': 'multimodal',
            'strategy': 'hybrid_multimodal_search'
        }
    
    return processed_query

# Enhanced retrieval function
def enhanced_multimodal_retrieval(user_input):
    """Retrieval with automatic query type detection"""
    
    # Process and classify query
    processed_query = process_user_query(user_input)
    
    # Retrieve based on strategy
    if processed_query['strategy'] == 'visual_content_priority':
        results = retrieve_by_content_type(
            processed_query['query'], 
            content_types=["image", "text"], 
            top_k=10
        )
        
    elif processed_query['strategy'] == 'text_content_priority':
        results = retrieve_by_content_type(
            processed_query['query'], 
            content_types=["text", "image"], 
            top_k=10
        )
        
    elif processed_query['strategy'] == 'diagram_analysis_focus':
        # Focus on diagrams and related text
        results = retrieve_multimodal_documents(
            processed_query['query'], 
            query_type="text", 
            top_k=10
        )
        # Filter for diagram-heavy content
        results = [r for r in results if 'diagram' in r.get('content', '').lower() 
                  or r.get('content_type') == 'image']
        
    elif processed_query['strategy'] == 'image_similarity_search':
        results = retrieve_multimodal_documents(
            processed_query['query'], 
            query_type="image", 
            top_k=10
        )
        
    else:  # balanced_multimodal
        results = retrieve_multimodal_documents(
            processed_query['query'], 
            query_type="multimodal", 
            top_k=10
        )
    
    return {
        'results': results,
        'query_analysis': processed_query,
        'total_results': len(results)
    }

```

## **Example Usage**

```python
# Example queries and their automatic classification

# Text query seeking visual content
query1 = "Show me the architecture diagram from the system design document"
result1 = enhanced_multimodal_retrieval(query1)
# Will classify as: text input, visual_seeking, visual_content_priority strategy

# Image upload for analysis
with open("diagram.png", "rb") as f:
    image_data = f.read()
result2 = enhanced_multimodal_retrieval(image_data)
# Will classify as: image input, diagram type, image_similarity_search strategy

# Mixed query
query3 = "What does this flowchart show about the user authentication process?"
result3 = enhanced_multimodal_retrieval(query3)
# Will classify as: text input, analysis_seeking, diagram_analysis_focus strategy

print(f"Query 1 strategy: {result1['query_analysis']['strategy']}")
print(f"Query 2 strategy: {result2['query_analysis']['strategy']}")
print(f"Query 3 strategy: {result3['query_analysis']['strategy']}")

```

This system automatically determines the query type and optimizes retrieval accordingly, ensuring users get the most relevant multimodal content regardless of how they phrase their questions or what type of input they provide.

