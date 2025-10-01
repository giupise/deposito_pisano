from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (CSVLoader, PyMuPDFLoader,
                                                  UnstructuredImageLoader, 
                                                  UnstructuredMarkdownLoader,
                                                  TextLoader)

load_dotenv()


@dataclass
class Settings:
    """
    Configuration settings for the RAG system components.
    
    This dataclass contains all configurable parameters for FAISS vector store
    persistence, text chunking, retrieval strategies, and Azure OpenAI integration.
    Optimized for technical aeronautic documents processing.
    
    Attributes
    ----------
    persist_dir : str
        Directory path for FAISS index persistence (default: "./faiss_db/default_aeronautics")
    chunk_size : int
        Size of text chunks for document splitting, optimized for technical docs (default: 1000)
    chunk_overlap : int
        Overlap between consecutive chunks for better continuity (default: 200)
    search_type : str
        Retrieval search strategy: "mmr" or "similarity" (default: "mmr")
    k : int
        Number of documents to retrieve for better coverage (default: 6)
    fetch_k : int
        Number of initial candidates for MMR algorithm (default: 20)
    mmr_lambda : float
        MMR lambda parameter balancing relevance vs diversity (default: 0.7)
    hf_model_name : str
        HuggingFace embedding model identifier (default: "sentence-transformers/all-MiniLM-L6-v2")
    lmstudio_model_env : str
        Model name for LM Studio environment (default: "gpt-4o")
    azure_embedding_model : str
        Azure OpenAI embedding model name (from env AZURE_EMBEDDING_MODEL)
    api_version : tuple
        Azure API version tuple (from env AZURE_API_VERSION)
    azure_endpoint : tuple
        Azure OpenAI endpoint tuple (from env AZURE_OPENAI_ENDPOINT)
    api_key : str
        Azure OpenAI API key (from env AZURE_OPENAI_API_KEY)
    """
    # Persistenza FAISS
    persist_dir: str = "./faiss_db/default_aeronautics"
    # Text splitting - Parametri ottimizzati per documenti tecnici
    chunk_size: int = 1000        #  Chunks più grandi per più contesto
    chunk_overlap: int = 200      #  Overlap maggiore per continuità
    # Retriever (MMR)
    search_type: str = "mmr"  # "mmr" o "similarity"
    k: int = 6               #  Più risultati per migliore coverage
    fetch_k: int = 20        # candidati iniziali (per MMR)
    mmr_lambda: float = 0.7  #  Più pertinenza, meno diversificazione
    # Embedding
    hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # LM Studio (OpenAI-compatible)
    lmstudio_model_env: str = "gpt-4o"  # nome del modello in LM Studio, via env var
    azure_embedding_model: str = os.getenv(
        "AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"
    )
    api_version = (os.getenv("AZURE_API_VERSION"),)
    azure_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT"),)
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    def set_persist_dir_from_query(self, query: str) -> None:
        """
        Set persistence directory dynamically based on query content.
        
        Creates a sanitized directory name from the input query to organize
        FAISS indices by topic or domain. Useful for maintaining separate
        knowledge bases for different subjects.
        
        Parameters
        ----------
        query : str
            Input query string to derive directory name from
            
        Returns
        -------
        str
            Sanitized persistence directory path
            
        Notes
        -----
        The query is sanitized by removing special characters, replacing spaces
        with underscores, and limiting length to avoid filesystem issues.
        """
        # Pulisci la query per renderla un nome di cartella valido
        clean_query = self._sanitize_query_for_filename(query)
        self.persist_dir = f"./faiss_db/{clean_query}"
        return self.persist_dir

    def _sanitize_query_for_filename(self, query: str) -> str:
        """
        Convert query string into valid filename/directory name.
        
        Sanitizes input string by removing special characters, normalizing
        spaces, and applying length constraints for filesystem compatibility.
        
        Parameters
        ----------
        query : str
            Raw query string to be sanitized
            
        Returns
        -------
        str
            Sanitized string suitable for filesystem usage
            
        Notes
        -----
        - Removes non-alphanumeric characters except spaces and hyphens
        - Replaces spaces with underscores
        - Limits length to 30 characters
        - Removes consecutive underscores
        - Returns "default" if result is empty
        """
        # Rimuovi caratteri speciali e sostituisci spazi con underscore
        clean = re.sub(r"[^\w\s-]", "", query.lower())
        clean = re.sub(r"\s+", "_", clean)
        # Limita la lunghezza per evitare nomi troppo lunghi
        clean = clean[:30]
        # Rimuovi underscore multipli
        clean = re.sub(r"_+", "_", clean).strip("_")
        return clean if clean else "default"


def load_documents(file_paths: List[str]) -> List[Document]:
    """
    Load documents from various file formats into LangChain Document objects.
    
    Supports multiple file formats including PDF, CSV, Markdown, text files,
    and images. Each file is loaded using the appropriate LangChain loader
    with content preview logging for debugging purposes.
    
    Parameters
    ----------
    file_paths : List[str]
        List of file paths to be loaded as documents
        
    Returns
    -------
    List[Document]
        List of LangChain Document objects containing loaded content and metadata
        
    Supported Formats
    ----------------
    - PDF: Using PyMuPDFLoader for robust PDF text extraction
    - CSV: Using CSVLoader for structured data loading
    - Markdown: Using UnstructuredMarkdownLoader for .md files
    - Text: Using TextLoader for plain text files
    - Images: Using UnstructuredImageLoader for image formats (png, jpg, jpeg, bmp, gif, tiff)
    
    Notes
    -----
    - Unsupported file types are skipped with warning messages
    - Each document's content is previewed in logs (first 100 characters)
    - Total document count and individual file statistics are logged
    """
    print(f" LOAD_DOCUMENTS: Caricamento di {len(file_paths)} file(s)")
    documents = []
    for file_path in file_paths:
        print(f"📄 Caricamento file: {file_path}")
        ext = file_path.split(".")[-1].lower()
        if ext == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif ext == "csv":
            loader = CSVLoader(file_path)
        elif ext == "md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif ext == "txt":
            loader = TextLoader(file_path)
        elif ext in ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]:
            loader = UnstructuredImageLoader(file_path)
        else:
            print(f"❌ Tipo file non supportato: {file_path}")
            continue
        docs = loader.load()
        print(f"    Caricati {len(docs)} documento/i da {file_path}")
        for i, doc in enumerate(docs):
            content_preview = doc.page_content[:100].replace("\n", " ")
            print(
                f"      Doc {i+1}: {len(doc.page_content)} caratteri - '{content_preview}...'"
            )
        documents.extend(docs)

    print(f"📚 LOAD_DOCUMENTS: Totale {len(documents)} documenti caricati")
    return documents


def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Apply robust document splitting for optimal retrieval performance.
    
    Splits documents into smaller chunks using RecursiveCharacterTextSplitter
    with hierarchical separators optimized for technical documentation.
    Maintains semantic coherence through configurable overlap.
    
    Parameters
    ----------
    docs : List[Document]
        List of documents to be split into chunks
    settings : Settings
        Configuration object containing chunk_size and chunk_overlap parameters
        
    Returns
    -------
    List[Document]
        List of document chunks with preserved metadata and optimized content boundaries
        
    Notes
    -----
    Uses hierarchical separator strategy:
    1. Markdown headers (#, ##, ###)
    2. Paragraph breaks (\n\n, \n)
    3. Sentence endings (., ?, !, ;, :)
    4. Clause separators (, )
    5. Word boundaries ( )
    6. Aggressive character-level fallback
    
    Chunk size and overlap are optimized for technical documents to ensure
    sufficient context while maintaining computational efficiency.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "#", "##", "###",
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",
                # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Format retrieved documents for LLM prompt with source citations.
    
    Combines multiple document chunks into a single formatted string with
    source attributions for transparency and traceability in RAG responses.
    Each document is prefixed with its source information.
    
    Parameters
    ----------
    docs : List[Document]
        List of retrieved Document objects to format
        
    Returns
    -------
    str
        Formatted string containing all document contents with source citations
        
    Format
    ------
    Each document is formatted as:
    [source:filename/path] document_content
    
    Documents are separated by double newlines for clear delineation.
    Source information is extracted from document metadata or uses default naming.
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def scan_docs_folder(docs_dir: str = "docs") -> List[str]:
    """
    Recursively scan directory for supported document formats.
    
    Searches through the specified directory and its subdirectories to find
    all files with supported extensions for document loading and processing.
    
    Parameters
    ----------
    docs_dir : str, optional
        Directory path to scan for documents (default: "docs")
        
    Returns
    -------
    List[str]
        List of absolute file paths for all supported documents found
        
    Supported Extensions
    -------------------
    - Documents: .pdf, .csv, .md, .txt
    - Images: .png, .jpg, .jpeg, .bmp, .gif, .tiff
    
    Notes
    -----
    - Performs recursive search through all subdirectories
    - Returns empty list if directory doesn't exist
    - Logs the total number of files found
    - Case-insensitive extension matching
    """
    supported_extensions = {
        ".pdf",
        ".csv",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".gif",
        ".tiff",
        ".txt"
    }
    file_paths = []

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"❌ Cartella {docs_dir} non trovata")
        return []

    # Scansione ricorsiva
    for file_path in docs_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            file_paths.append(str(file_path))

    print(f"Trovati {len(file_paths)} file nella cartella {docs_dir}")
    return file_paths


def clean_web_content(text: str) -> str:
    """
    Clean web-scraped content by removing unwanted UI elements and noise.
    
    Applies comprehensive text cleaning to web-scraped content, removing
    navigation elements, advertisements, legal notices, and formatting
    artifacts while preserving meaningful textual information.
    
    Parameters
    ----------
    text : str
        Raw web content text to be cleaned
        
    Returns
    -------
    str
        Cleaned text with unwanted elements removed and normalized formatting
        
    Cleaning Operations
    ------------------
    1. Normalize whitespace and control characters
    2. Remove common UI patterns (cookies, navigation, social media)
    3. Remove legal notices and copyright information
    4. Filter out URLs and email addresses
    5. Remove timestamps and date patterns
    6. Clean special character sequences
    7. Filter short/meaningless lines
    8. Remove all-caps navigation text
    
    Notes
    -----
    - Preserves accented characters and Unicode text
    - Maintains sentence structure and punctuation
    - Filters content shorter than 20 characters per line
    - Returns empty string for null/empty input
    """
    if not text:
        return ""

    # Rimuovi caratteri di controllo e spazi multipli
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Rimuovi pattern comuni di navigazione e UI
    patterns_to_remove = [
        r"Cookie Policy|Privacy Policy|Note Legali|Termini e Condizioni",
        r"Accetta tutti i cookie|Gestisci cookie|Rifiuta cookie",
        r"Iscriviti alla newsletter|Seguici su|Condividi su",
        r"Copyright.*?\d{4}|All rights reserved|Tutti i diritti riservati",
        r"Menu|Navbar|Header|Footer|Sidebar",
        r"Caricamento in corso|Loading|Attendere prego",
        r"Clicca qui|Click here|Leggi tutto|Read more",
        r"Ti potrebbe interessare|Articoli correlati|Notizie correlate",
        r"I più visti|Più letti|Trending|Popular",
        r"Pubblicità|Advertisement|Sponsor|Promo",
        r"PODCAST|RUBRICHE|SONDAGGI|LE ULTIME EDIZIONI",
        r"Ascolta i Podcast.*?|Vedi tutti.*?|Scopri di più.*?",
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Rimuovi URL e email
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Rimuovi numeri isolati (spesso date, ore, contatori)
    text = re.sub(r"\b\d{1,2}[:.]\d{2}\b", "", text)  # Orari
    text = re.sub(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", "", text)  # Date

    # Rimuovi caratteri speciali ripetuti
    text = re.sub(r'[^\w\s.,!?;:()\-"\'àèéìíîòóùúçñü]+', " ", text, flags=re.UNICODE)

    # Rimuovi linee molto corte (probabilmente navigazione)
    lines = text.split(".")
    meaningful_lines = []
    for line in lines:
        line = line.strip()
        if len(line) > 20 and not re.match(r"^[A-Z\s]+$", line):  # Non solo maiuscole
            meaningful_lines.append(line)

    text = ". ".join(meaningful_lines)

    # Pulizia finale
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text
