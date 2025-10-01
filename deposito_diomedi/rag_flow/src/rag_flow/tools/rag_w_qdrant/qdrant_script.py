from .config import Settings
# from config import Settings

import numpy as np
from typing import List, Any, Tuple
from langchain.schema import Document


from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """
    Create or recreate a Qdrant collection optimized for RAG (Retrieval-Augmented Generation).
    
    This function sets up a vector database collection with optimal configuration for
    semantic search, including HNSW indexing, payload indexing, and quantization.
    
    Args:
        client: Qdrant client instance for database operations
        settings: Configuration object containing collection parameters
        vector_size: Dimension of the embedding vectors (e.g., 384 for MiniLM-L6)
        
    Collection Architecture:
    - Vector storage: Dense vectors for semantic similarity search
    - Payload storage: Metadata and text content for retrieval
    - Indexing: HNSW for approximate nearest neighbor search
    - Quantization: Scalar quantization for memory optimization
        
    Distance Metric Selection:
    - Cosine distance: Normalized similarity, good for semantic embeddings
    - Alternatives: Euclidean (L2), Manhattan (L1), Dot product
    - Cosine preferred for normalized embeddings (sentence-transformers)
        
    HNSW Index Configuration:
    - m=32: Average connections per node (higher = better quality, more memory)
    - ef_construct=256: Search depth during construction (higher = better quality, slower build)
    - Trade-offs: Higher values improve recall but increase memory and build time
        
    Optimizer Configuration:
    - default_segment_number=2: Parallel processing segments
    - Benefits: Faster indexing, better resource utilization
    - Considerations: More segments = more memory overhead
        
    Quantization Strategy:
    - Scalar quantization: Reduces vector precision from float32 to int8
    - Memory savings: ~4x reduction in vector storage
    - Quality impact: Minimal impact on search accuracy
    - always_ram=False: Vectors stored on disk, loaded to RAM as needed
        
    Payload Indexing Strategy:
    - Text index: Full-text search capabilities (BM25 scoring)
    - Keyword indices: Fast exact matching and filtering
    - Performance: Significantly faster than unindexed field searches
        
    Collection Lifecycle:
    - recreate_collection: Drops existing collection and creates new one
    - Use case: Development/testing, major schema changes
    - Production: Consider using create_collection + update_collection_info
        
    Performance Considerations:
    - Build time: HNSW construction scales with collection size
    - Memory usage: Vectors loaded to RAM during search
    - Storage: Quantized vectors + payload data
    - Query latency: HNSW provides sub-millisecond search times
        
    Scaling Guidelines:
    - Small collections (<100K vectors): Current settings optimal
    - Medium collections (100K-1M vectors): Increase m to 48-64
    - Large collections (1M+ vectors): Consider multiple collections or sharding
    """
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
            ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2  # parallelismo/segmentazione iniziale
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
        ),
    )

    # Indice full-text sul campo 'text' per filtri MatchText
    client.create_payload_index(
        collection_name=settings.collection,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Indici keyword per filtri esatti / velocità nei filtri
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )


def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "trustability": doc.metadata.get("trustability", "untrusted"),
            "filename": doc.metadata.get("filename", ""),
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings):
    """
    Insert or update document chunks in Qdrant vector database.
    
    Converts document chunks to vector embeddings and stores them in the
    Qdrant collection with associated metadata and payload information.
    
    Parameters
    ----------
    client : QdrantClient
        Qdrant database client for vector operations
    settings : Settings
        Configuration object containing collection parameters
    chunks : List[Document]
        Document chunks to be embedded and stored
    embeddings : Union[HuggingFaceEmbeddings, AzureOpenAIEmbeddings]
        Embedding model for vector generation (supports both HF and Azure)
        
    Notes
    -----
    Updated to accept both HuggingFaceEmbeddings and AzureOpenAIEmbeddings
    for improved flexibility with different embedding providers.
    """
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    points = build_points(chunks, vecs)
    client.upsert(collection_name=settings.collection, points=points, wait=True)

# =========================
# Ricerca: semantica / testuale / ibrida
# =========================

def qdrant_semantic_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings,
    limit: int,
    with_vectors: bool = False
):
    """
    Perform semantic search using vector similarity in Qdrant database.
    
    Converts the query to embedding vector and searches for the most similar
    document chunks using HNSW approximate nearest neighbor search.
    
    Parameters
    ----------
    client : QdrantClient
        Qdrant database client for vector operations
    settings : Settings
        Configuration object containing collection parameters
    query : str
        User query string to search for
    embeddings : Union[HuggingFaceEmbeddings, AzureOpenAIEmbeddings]
        Embedding model for query vectorization (supports both HF and Azure)
    limit : int
        Maximum number of results to return
    with_vectors : bool, optional
        Whether to include vector data in results (default: False)
        
    Returns
    -------
    List[ScoredPoint]
        Ranked list of similar document chunks with scores
        
    Notes
    -----
    Updated to accept both HuggingFaceEmbeddings and AzureOpenAIEmbeddings
    for improved flexibility with different embedding providers.
    """
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(
            hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
            exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
        ),
    )
    return res.points

def qdrant_text_prefilter_ids(
    client: QdrantClient,
    settings: Settings,
    query: str,
    max_hits: int
) -> List[int]:
    """
    Usa l'indice full-text su 'text' per prefiltrare i punti che contengono parole chiave.
    Non restituisce uno score BM25: otteniamo un sottoinsieme di id da usare come boost.
    """
    # Scroll con filtro MatchText per ottenere id dei match testuali
    # (nota: scroll è paginato; qui prendiamo solo i primi max_hits per semplicità)
    matched_ids: List[int] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=settings.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=query))]
            ),
            limit=min(256, max_hits - len(matched_ids)),
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        matched_ids.extend([p.id for p in points])
        if not next_page or len(matched_ids) >= max_hits:
            break
    return matched_ids

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]],
    k: int,
    lambda_mult: float
) -> List[int]:
    """
    Select diverse results using Maximal Marginal Relevance (MMR) algorithm.
    
    MMR balances relevance to the query with diversity among selected results,
    reducing redundancy and improving information coverage. This is particularly
    useful for RAG systems where diverse context provides better generation.
    
    Args:
        query_vec: Query embedding vector for relevance calculation
        candidates_vecs: List of candidate document embedding vectors
        k: Number of results to select
        lambda_mult: MMR parameter balancing relevance vs. diversity (0.0 to 1.0)
        
    Returns:
        List[int]: Indices of selected candidates in order of selection
        
    MMR Algorithm Overview:
    
    The algorithm iteratively selects candidates that maximize the MMR score:
    
    MMR_score(i) = λ × Relevance(i, query) - (1-λ) × max_similarity(i, selected)
    
    Where:
    - λ (lambda_mult): Weight for relevance vs. diversity
    - Relevance(i, query): Cosine similarity between candidate i and query
    - max_similarity(i, selected): Maximum similarity between candidate i and already selected items
        
    Algorithm Steps:
    
    1. INITIALIZATION:
       - Calculate relevance scores for all candidates vs. query
       - Select the highest-scoring candidate as the first result
       - Initialize selected and remaining candidate sets
        
    2. ITERATIVE SELECTION:
       - For each remaining position, calculate MMR score for all candidates
       - MMR score balances query relevance with diversity from selected items
       - Select candidate with highest MMR score
       - Update selected and remaining sets
        
    3. TERMINATION:
       - Continue until k candidates selected or no more candidates available
       - Return indices in selection order
        
    Mathematical Foundation:
    
    Cosine Similarity:
    - cos(a,b) = (a·b) / (||a|| × ||b||)
    - Range: [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    - Normalized vectors typically have values in [0, 1] range
        
    MMR Score Calculation:
    - Relevance term: λ × cos(query, candidate)
    - Diversity term: (1-λ) × max(cos(candidate, selected_i))
    - Higher relevance increases score, higher similarity to selected decreases score
        
    Lambda Parameter Behavior:
    
    λ = 0.0 (Pure Diversity):
    - Only diversity matters, relevance ignored
    - Results may be irrelevant to query
    - Useful for exploratory search
        
    λ = 0.5 (Balanced):
    - Equal weight for relevance and diversity
    - Good compromise for general use
    - Moderate redundancy reduction
        
    λ = 0.6 (Current Setting):
    - Slight preference for relevance
    - Good diversity while maintaining relevance
    - Recommended for most RAG applications
        
    λ = 1.0 (Pure Relevance):
    - Only relevance matters, diversity ignored
    - Equivalent to simple top-K selection
    - May have redundant results
        
    Performance Characteristics:
    
    Time Complexity:
    - O(k × n) where k = results to select, n = total candidates
    - Each iteration processes all remaining candidates
    - Quadratic complexity in worst case (k ≈ n)
        
    Space Complexity:
    - O(n) for storing vectors and similarity scores
    - O(k) for selected indices
    - O(n) for remaining candidate set
        
    Memory Usage:
    - Vector storage: All candidate vectors loaded in memory
    - Similarity cache: Relevance scores computed once
    - Selection state: Small overhead for tracking
        
    Quality Metrics:
    
    Relevance Preservation:
    - Higher lambda values preserve more relevance
    - Lower lambda values may sacrifice relevance for diversity
    - Optimal balance depends on use case
        
    Diversity Improvement:
    - MMR significantly reduces redundancy compared to top-K
    - Diversity increases as lambda decreases
    - Measurable improvement in information coverage
        
    User Experience:
    - Less repetitive results
    - Better coverage of different aspects
    - More informative context for LLM generation
        
    Use Case Recommendations:
    
    Research & Exploration:
    - λ = 0.3-0.5: Maximize diversity for comprehensive understanding
    - Higher k values: More diverse perspectives
        
    Factual Queries:
    - λ = 0.7-0.9: Prioritize relevance for accurate information
    - Lower k values: Focus on most relevant results
        
    Technical Documentation:
    - λ = 0.5-0.7: Balance relevance with diverse technical perspectives
    - Moderate k values: Comprehensive technical coverage
        
    Conversational AI:
    - λ = 0.6-0.8: Good relevance with some diversity
    - Higher k values: Rich context for generation
        
    Tuning Guidelines:
    
    For Maximum Diversity:
    - Decrease lambda to 0.3-0.5
    - Increase k to 8-12 results
    - Monitor relevance quality
        
    For Maximum Relevance:
    - Increase lambda to 0.8-1.0
    - Decrease k to 3-6 results
    - Accept some redundancy
        
    For Balanced Results:
    - Use lambda = 0.6-0.7 (current setting)
    - Moderate k values (6-8)
    - Good compromise for most applications
        
    Implementation Notes:
    
    Numerical Stability:
    - Small epsilon (1e-12) added to prevent division by zero
    - Cosine similarity handles normalized vectors robustly
    - Float precision sufficient for similarity calculations
        
    Edge Cases:
    - Empty candidate list: Returns empty result
    - k > candidates: Returns all candidates
    - Single candidate: Returns that candidate regardless of lambda
        
    Optimization Opportunities:
    - Vector similarity could be pre-computed and cached
    - Parallel processing for large candidate sets
    - Early termination for very low diversity scores
    """
    
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)

    def cos(a, b):
        na = (a @ a) ** 0.5 + 1e-12
        nb = (b @ b) ** 0.5 + 1e-12
        return float((a @ b) / (na * nb))

    sims = [cos(v, q) for v in V]
    selected: List[int] = []
    remaining = set(range(len(V)))

    while len(selected) < min(k, len(V)):
        if not selected:
            # pick the highest similarity first
            best = max(remaining, key=lambda i: sims[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx = None
        best_score = -1e9
        for i in remaining:
            max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
            score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def hybrid_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings
):
    """
    Perform hybrid search combining semantic similarity and text-based matching.
    
    This function implements a sophisticated retrieval strategy that leverages both
    semantic understanding and traditional text search to provide high-quality,
    relevant results with minimal redundancy.
    
    Args:
        client: Qdrant client for database operations
        settings: Configuration object containing search parameters
        query: User's search query string
        embeddings: Embedding model for semantic search (HuggingFace or Azure OpenAI)
        
    Returns:
        List[ScoredPoint]: Ranked list of relevant document chunks
        
    Hybrid Search Strategy Overview:
    
    1. SEMANTIC SEARCH (Vector Similarity):
       - Converts query to embedding vector
       - Performs approximate nearest neighbor search using HNSW index
       - Retrieves top_n_semantic candidates based on cosine similarity
       - Provides semantic understanding of query intent
        
    2. TEXT-BASED PREFILTERING:
       - Uses full-text search capabilities (BM25 scoring)
       - Identifies documents containing query keywords/phrases
       - Creates a set of text-relevant document IDs
       - Acts as a relevance filter for semantic results
        
    3. SCORE FUSION & NORMALIZATION:
       - Normalizes semantic scores to [0,1] range for fair comparison
       - Applies alpha weight to balance semantic vs. text relevance
       - Adds text_boost for results matching both criteria
       - Creates unified relevance scoring
        
    4. RESULT DIVERSIFICATION (Optional MMR):
       - Applies Maximal Marginal Relevance to reduce redundancy
       - Balances relevance with diversity using mmr_lambda parameter
       - Selects final_k results from top candidates
        
    Algorithm Flow:
    
    Phase 1: Semantic Retrieval
    - Query embedding generation
    - HNSW-based vector search
    - Score normalization for fusion
        
    Phase 2: Text Matching
    - Full-text search with MatchText filter
    - ID collection for hybrid scoring
    - Performance optimization with pagination
        
    Phase 3: Score Fusion
    - Linear combination of semantic and text scores
    - Boost application for hybrid matches
    - Ranking by fused scores
        
    Phase 4: Result Selection
    - Top-N selection or MMR diversification
    - Final result ordering and return
        
    Performance Characteristics:
    
    Time Complexity:
    - Semantic search: O(log n) with HNSW index
    - Text search: O(m) where m is text matches
    - Score fusion: O(k) where k is semantic candidates
    - MMR: O(k²) for diversity computation
        
    Memory Usage:
    - Vector storage: Quantized vectors in memory
    - Score storage: Temporary arrays for fusion
    - Result storage: Final selected points
        
    Quality Metrics:
    
    Recall (Completeness):
    - Semantic search: High recall for conceptual queries
    - Text search: High recall for keyword queries
    - Hybrid approach: Combines strengths of both
        
    Precision (Relevance):
    - Score fusion: Balances multiple relevance signals
    - Text boost: Rewards multi-criteria matches
    - MMR: Reduces redundant results
        
    Diversity:
    - MMR algorithm: Maximizes information coverage
    - Lambda parameter: Controls diversity vs. relevance trade-off
    - Result variety: Better user experience
        
    Tuning Guidelines:
    
    For High Precision:
    - Increase alpha (0.8-0.9): Prioritize semantic similarity
    - Increase text_boost (0.3-0.5): Reward text matches
    - Decrease mmr_lambda (0.7-0.9): Prioritize relevance
        
    For High Recall:
    - Increase top_n_semantic (50-100): More candidates
    - Increase top_n_text (150-200): More text matches
    - Decrease alpha (0.5-0.7): Balance search strategies
        
    For High Diversity:
    - Enable MMR (use_mmr=True)
    - Decrease mmr_lambda (0.3-0.6): Prioritize diversity
    - Increase final_k (8-12): More diverse results
        
    Use Case Optimizations:
    
    Technical Documentation:
    - High alpha (0.8-0.9): Semantic understanding critical
    - High text_boost (0.3-0.4): Technical terms important
    - MMR enabled: Diverse technical perspectives
        
    General Knowledge:
    - Balanced alpha (0.6-0.8): Both strategies valuable
    - Moderate text_boost (0.2-0.3): Balanced approach
    - MMR enabled: Comprehensive coverage
        
    Factual Queries:
    - High alpha (0.7-0.9): Semantic context important
    - Low text_boost (0.1-0.2): Facts over style
    - MMR optional: Precision over diversity
        
    Notes
    -----
    Updated to support both HuggingFaceEmbeddings and AzureOpenAIEmbeddings
    for improved flexibility with different embedding providers.
    """
    # (1) semantica
    sem = qdrant_semantic_search(
        client, settings, query, embeddings,
        limit=settings.top_n_semantic, with_vectors=True
    )
    if not sem:
        return []

    # (2) full-text prefilter (id)
    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))

    # Normalizzazione score semantici per fusione
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):  # robusto al caso smin==smax
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    # (3) fusione con boost testuale
    fused: List[Tuple[int, float, Any]] = []  # (idx, fused_score, point)
    for idx, p in enumerate(sem):
        base = norm(p.score)                    # [0..1]
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost         # boost additivo
        fused.append((idx, fuse, p))

    # ordina per fused_score desc
    fused.sort(key=lambda t: t[1], reverse=True)

    # MMR opzionale per diversificare i top-K
    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        # prendiamo i primi N dopo fusione (es. 30) e poi MMR per final_k
        N = min(len(fused), max(settings.final_k * 5, settings.final_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    # altrimenti, prendi i primi final_k dopo fusione
    return [p for _, _, p in fused[:settings.final_k]]