
from __future__ import annotations

from dotenv import load_dotenv
from dataclasses import dataclass

@dataclass
class Settings:
    """
    Comprehensive configuration settings for the RAG pipeline.
    
    This class centralizes all configurable parameters, allowing easy tuning
    of the system's behavior without modifying the core logic.
    """
    
    # =========================
    # Qdrant Vector Database Configuration
    # =========================
    qdrant_url: str = "http://localhost:6333"
    """
    Qdrant server URL. 
    - Default: Local development instance
    - Production: Use your Qdrant cloud URL or server address
    - Alternative: Can be overridden via environment variable QDRANT_URL
    """
    
    collection: str = "rag_chunks"
    """
    Collection name for storing document chunks and vectors.
    - Naming convention: Use descriptive names like 'company_docs', 'research_papers'
    - Multiple collections: Can create separate collections for different document types
    - Cleanup: Old collections can be dropped and recreated for fresh indexing
    """
    
    # =========================
    # Embedding Model Configuration
    # =========================
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    """
    HuggingFace sentence transformer model for generating embeddings.
    
    Model Options & Trade-offs:
    - all-MiniLM-L6-v2: 384 dimensions, fast, good quality, balanced choice
    - all-MiniLM-L12-v2: 768 dimensions, slower, higher quality, better for complex queries
    - all-mpnet-base-v2: 768 dimensions, excellent quality, slower inference
    - paraphrase-multilingual-MiniLM-L12-v2: 768 dimensions, multilingual support
    
    Dimension Impact:
    - Lower dimensions (384): Faster search, less memory, slightly lower accuracy
    - Higher dimensions (768+): Better accuracy, slower search, more memory usage
    
    Performance Considerations:
    - L6 models: ~2-3x faster than L12 models
    - L12 models: ~10-15% better semantic understanding
    - Base models: Good balance between speed and quality
    """
    
    # =========================
    # Document Chunking Configuration
    # =========================
    chunk_size: int = 700
    """
    Maximum number of characters per document chunk.
    
    Chunk Size Trade-offs:
    - Small chunks (200-500): Better precision, more granular retrieval, higher storage overhead
    - Medium chunks (500-1000): Balanced precision and context, recommended for most use cases
    - Large chunks (1000+): Better context preservation, lower precision, fewer chunks to manage
    
    Optimal Sizing Guidelines:
    - Technical documents: 500-800 characters (preserve technical context)
    - General text: 700-1000 characters (good balance)
    - Conversational text: 300-600 characters (preserve dialogue flow)
    - Code/structured data: 200-500 characters (preserve logical units)
    
    Impact on Retrieval:
    - Smaller chunks: Higher recall, lower precision, more relevant snippets
    - Larger chunks: Lower recall, higher precision, more complete context
    """
    
    chunk_overlap: int = 120
    """
    Number of characters to overlap between consecutive chunks.
    
    Overlap Strategy:
    - No overlap (0): Clean separation, may miss context at boundaries
    - Small overlap (50-150): Preserves context, minimal redundancy
    - Large overlap (200+): Maximum context preservation, higher storage cost
    
    Optimal Overlap Guidelines:
    - Technical content: 100-200 characters (preserve technical terms)
    - General text: 100-150 characters (good balance)
    - Conversational: 50-100 characters (preserve dialogue context)
    - Code: 50-100 characters (preserve function boundaries)
    
    Storage Impact:
    - 0% overlap: Base storage requirement
    - 20% overlap: ~20% increase in storage
    - 50% overlap: ~50% increase in storage
    """
    vector_size: int = 1536
    # =========================
    # Hybrid Search Configuration
    # =========================
    top_n_semantic: int = 30
    """
    Number of top semantic search candidates to retrieve initially.
    
    Semantic Search Candidates:
    - Low values (10-20): Fast retrieval, may miss relevant results
    - Medium values (30-50): Good balance between speed and recall
    - High values (100+): Maximum recall, slower performance
    
    Performance Impact:
    - Retrieval time: Linear increase with candidate count
    - Memory usage: Linear increase with candidate count
    - Quality: Diminishing returns beyond 50-100 candidates
    
    Tuning Guidelines:
    - Small collections (<1000 docs): 20-30 candidates
    - Medium collections (1000-10000 docs): 30-50 candidates
    - Large collections (10000+ docs): 50-100 candidates
    """
    
    top_n_text: int = 100
    """
    Maximum number of text-based matches to consider for hybrid fusion.
    
    Text Search Scope:
    - Low values (50): Fast text filtering, may miss relevant matches
    - Medium values (100): Good balance between speed and coverage
    - High values (200+): Maximum text coverage, slower performance
    
    Hybrid Search Strategy:
    - Text search acts as a pre-filter for semantic results
    - Higher values improve the quality of text-semantic fusion
    - Optimal value depends on collection size and query complexity
    """
    
    final_k: int = 6
    # k: int = 6
    """
    Final number of results to return after all processing steps.
    
    Result Count Considerations:
    - User experience: 3-5 results for simple queries, 5-10 for complex ones
    - Context window: Align with LLM context limits (e.g., 6-8 chunks for GPT-3.5)
    - Diversity: Higher values allow MMR to select more diverse results
    
    LLM Integration:
    - GPT-3.5: 6-8 chunks typically fit in context
    - GPT-4: 8-12 chunks can be processed
    - Claude: 6-10 chunks work well
    """
    
    alpha: float = 0.75
    """
    Weight for semantic similarity in hybrid score fusion (0.0 to 1.0).
    
    Alpha Parameter Behavior:
    - alpha = 0.0: Pure text-based ranking (BM25, keyword matching)
    - alpha = 0.5: Equal weight for semantic and text relevance
    - alpha = 0.75: Semantic similarity prioritized (current setting)
    - alpha = 1.0: Pure semantic ranking (cosine similarity only)
    
    Use Case Recommendations:
    - Technical queries: 0.7-0.9 (semantic understanding important)
    - Factual queries: 0.5-0.7 (balanced approach)
    - Keyword searches: 0.3-0.5 (text matching more important)
    - Conversational queries: 0.6-0.8 (semantic context matters)
    
    Tuning Strategy:
    - Start with 0.75 for general use
    - Increase if semantic results seem irrelevant
    - Decrease if text matching is too weak
    """
    
    text_boost: float = 0.20
    """
    Additional score boost for results that match both semantic and text criteria.
    
    Text Boost Mechanism:
    - Applied additively to fused scores
    - Encourages results that satisfy both search strategies
    - Helps surface highly relevant content that matches multiple criteria
    
    Boost Value Guidelines:
    - Low boost (0.1-0.2): Subtle preference for hybrid matches
    - Medium boost (0.2-0.4): Strong preference for hybrid matches
    - High boost (0.5+): Heavy preference, may dominate ranking
    
    Optimal Settings:
    - General use: 0.15-0.25
    - Technical content: 0.20-0.30
    - Factual queries: 0.10-0.20
    """
    
    # =========================
    # MMR (Maximal Marginal Relevance) Configuration
    # =========================
    use_mmr: bool = True
    """
    Whether to use MMR for result diversification and redundancy reduction.
    
    MMR Benefits:
    - Reduces redundant results with similar content
    - Improves coverage of different aspects of the query
    - Better user experience with diverse information
    
    MMR Trade-offs:
    - Slightly slower than simple top-K selection
    - May reduce absolute relevance scores
    - Better for exploratory queries, worse for specific fact retrieval
    
    Alternatives:
    - False: Simple top-K selection (faster, may have redundancy)
    - True: MMR diversification (slower, better diversity)
    """
    
    mmr_lambda: float = 0.6
    """
    MMR diversification parameter balancing relevance vs. diversity (0.0 to 1.0).
    
    Lambda Parameter Behavior:
    - lambda = 0.0: Pure diversity (ignore relevance, maximize difference)
    - lambda = 0.5: Balanced relevance and diversity
    - lambda = 0.6: Slight preference for relevance (current setting)
    - lambda = 1.0: Pure relevance (ignore diversity, top-K selection)
    
    Use Case Recommendations:
    - Research queries: 0.4-0.6 (diverse perspectives important)
    - Factual queries: 0.7-0.9 (relevance more important)
    - Exploratory queries: 0.3-0.5 (diversity valuable)
    - Specific searches: 0.8-1.0 (precision over diversity)
    
    Tuning Guidelines:
    - Start with 0.6 for general use
    - Decrease if results seem too similar
    - Increase if results seem too diverse
    """
    
    # =========================
    # LLM Configuration (Optional)
    # =========================
    lm_base_env: str = "OPENAI_BASE_URL"
    """
    Environment variable name for LLM service base URL.
    
    Supported Services:
    - OpenAI: https://api.openai.com/v1
    - LM Studio: http://localhost:1234/v1
    - Ollama: http://localhost:11434/v1
    - Custom API: Your endpoint URL
    
    Configuration Examples:
    - OpenAI: OPENAI_BASE_URL=https://api.openai.com/v1
    - LM Studio: OPENAI_BASE_URL=http://localhost:1234/v1
    - Azure OpenAI: OPENAI_BASE_URL=https://your-resource.openai.azure.com
    """
    
    lm_key_env: str = "OPENAI_API_KEY"
    """
    Environment variable name for LLM service API key.
    
    Security Notes:
    - Never hardcode API keys in source code
    - Use environment variables or secure secret management
    - Rotate keys regularly for production systems
    
    Configuration Examples:
    - OpenAI: OPENAI_API_KEY=sk-...
    - LM Studio: OPENAI_API_KEY=lm-studio (can be any value)
    - Azure: OPENAI_API_KEY=your-azure-key
    """
    
    lm_model_env: str = "LMSTUDIO_MODEL"
    """
    Environment variable name for the specific LLM model to use.
    
    Model Selection:
    - OpenAI: gpt-3.5-turbo, gpt-4, gpt-4-turbo
    - LM Studio: Any model name you've loaded
    - Ollama: llama2, codellama, mistral, etc.
    - Custom: Your model identifier
    
    Configuration Examples:
    - OpenAI: LMSTUDIO_MODEL=gpt-3.5-turbo
    - LM Studio: LMSTUDIO_MODEL=llama-2-7b-chat
    - Ollama: LMSTUDIO_MODEL=llama2:7b
    """

load_dotenv()