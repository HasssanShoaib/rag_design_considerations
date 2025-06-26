Azure OpenAI's specific multimodal embedding capabilities:

**Azure OpenAI Service does NOT currently offer a dedicated multimodal embedding model.**

The embedding models available in Azure OpenAI are:

-   `text-embedding-ada-002` (Version 2), `text-embedding-3-large`, `text-embedding-3-small`

These are **text-only** embedding models.

## For Multimodal Embeddings in Azure, you need to use:

**Azure AI Vision Multimodal Embeddings API** - This is separate from Azure OpenAI:

-   Generates vector representations of images that are compatible with text search over the same vector space
-   Can be used with Azure AI Search to vectorize images alongside OCR for text recognition

## Recommended Architecture:

1.  **For text content**: Use Azure OpenAI `text-embedding-3-large` or `text-embedding-3-small`
2.  **For images/multimodal content**: Use Azure AI Vision Multimodal Embeddings API
3.  **Integration**: Both can be integrated with Azure AI Search vector database

This hybrid approach gives you the best of both worlds - Azure OpenAI's excellent text embeddings combined with Azure's multimodal capabilities for images and other content types.

