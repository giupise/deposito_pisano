import warnings
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .azure_connections import get_azure_embedding_model
from .utils import Settings, split_documents
# from azure_connections import get_azure_embedding_model
# from utils import Settings, split_documents

warnings.filterwarnings("ignore", category=UserWarning)


def build_faiss_vectorstore(
    chunks: List[Document], embeddings: get_azure_embedding_model, persist_dir: str
) -> FAISS:
    """
    Build a FAISS vector store from document chunks and persist to disk.
    
    Creates a FAISS IndexFlatL2 vector store from document chunks using Azure OpenAI
    embeddings. The resulting index is automatically saved to the specified directory
    for future retrieval and reuse.
    
    Parameters
    ----------
    chunks : List[Document]
        List of document chunks to be indexed in the vector store
    embeddings : get_azure_embedding_model
        Azure OpenAI embedding model for vectorizing document content
    persist_dir : str
        Directory path where the FAISS index will be saved
        
    Returns
    -------
    FAISS
        Configured FAISS vector store with indexed document chunks
        
    Notes
    -----
    The function creates the persist directory if it doesn't exist and
    automatically saves the vector store using FAISS's save_local method.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    return vs


def load_or_build_vectorstore(
    settings: Settings, embeddings: get_azure_embedding_model, docs: List[Document]
) -> FAISS:
    """
    Load existing FAISS index or build new one if none exists.
    
    Attempts to load a persisted FAISS vector store from the configured directory.
    If no existing index is found, creates a new one from the provided documents
    after splitting them into appropriate chunks.
    
    Parameters
    ----------
    settings : Settings
        Configuration object containing persistence directory and chunking parameters
    embeddings : get_azure_embedding_model
        Azure OpenAI embedding model for document vectorization
    docs : List[Document]
        List of documents to be processed and indexed
        
    Returns
    -------
    FAISS
        FAISS vector store either loaded from disk or newly created
        
    Notes
    -----
    Currently always builds a new vector store regardless of existing indices.
    Documents are split into chunks using the settings configuration before indexing.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)


def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configure and create a retriever from the FAISS vector store.
    
    Creates a retriever with configurable search strategy (MMR or similarity).
    MMR (Maximal Marginal Relevance) provides less redundant and more diverse
    results compared to plain similarity search.
    
    Parameters
    ----------
    vector_store : FAISS
        FAISS vector store containing indexed documents
    settings : Settings
        Configuration object containing retrieval parameters (search_type, k, fetch_k, etc.)
        
    Returns
    -------
    VectorStoreRetriever
        Configured retriever for document retrieval from the vector store
        
    Notes
    -----
    Supports two search types:
    - "mmr": Maximal Marginal Relevance for diverse results
    - "similarity": Standard cosine similarity search
    
    MMR parameters include k (final results), fetch_k (candidates), and lambda_mult (diversity).
    """
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.k,
                "fetch_k": settings.fetch_k,
                "lambda_mult": settings.mmr_lambda,
            },
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )
