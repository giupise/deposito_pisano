#!/usr/bin/env python3
"""
COMPLETE RAG TOOL
=================

Tool CrewAI che mantiene TUTTE le funzionalità del sistema RAG originale.
Nessuna funzionalità rimossa, solo adattato con decoratore @tool.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import mimetypes

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from openai import AzureOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader
)

# LangChain Core components for prompt/chain construction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Qdrant vector database client and models
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
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

# CrewAI
# CrewAI: import the tool decorator with graceful fallback for versions that don't export it
try:
    from crewai_tools import tool  # type: ignore
except Exception:
    # Minimal no-op decorator fallback to allow running tests and direct calls
    def tool(name: str):  # type: ignore
        def _wrap(fn):
            setattr(fn, "tool_name", name)
            return fn
        return _wrap

# Alternative: Create a simple tool wrapper without crewai_tools dependency
class SimpleTool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

# =========================
# Configurazione - IDENTICA ALL'ORIGINALE
# =========================

load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

@dataclass
class Settings:
    # Paths
    docs_folder: str = r"C:\Users\An291pz\deposito_pisano\culinary-project\docs"
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag_documents"
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = "AZURE_OPENAI_ENDPOINT"
    azure_openai_api_key: str = "AZURE_OPENAI_API_KEY" 
    azure_openai_api_version: str = "AZURE_OPENAI_API_VERSION"
    azure_embeddings_deployment: str = "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
    azure_chat_deployment: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    
    # Chunking parameters
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval parameters
    top_n_semantic: int = 20
    top_n_text: int = 50
    final_k: int = 5
    alpha: float = 0.7
    text_boost: float = 0.25
    
    # MMR parameters
    use_mmr: bool = True
    mmr_lambda: float = 0.7

SETTINGS = Settings()

# =========================
# Caricamento documenti - IDENTICO ALL'ORIGINALE
# =========================

def load_document_by_extension(file_path: Path) -> List[Document]:
    """Carica un documento basandosi sulla sua estensione"""
    try:
        extension = file_path.suffix.lower()
        
        if extension == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
            
        elif extension == '.pdf':
            loader = PyPDFLoader(str(file_path))
            
        elif extension in ['.docx', '.doc']:
            loader = Docx2txtLoader(str(file_path))
            
        elif extension == '.csv':
            loader = CSVLoader(str(file_path))
            
        elif extension in ['.md', '.markdown']:
            loader = TextLoader(str(file_path), encoding='utf-8')
            
        else:
            print(f"⚠️ Tipo di file non supportato: {extension} per {file_path.name}")
            return []
        
        documents = loader.load()
        
        # Aggiungi metadati personalizzati
        for doc in documents:
            doc.metadata.update({
                'source_file': file_path.name,
                'file_path': str(file_path),
                'file_extension': extension,
                'file_size': file_path.stat().st_size if file_path.exists() else 0
            })
        
        return documents
        
    except Exception as e:
        print(f"❌ Errore nel caricamento di {file_path.name}: {e}")
        return []

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Carica tutti i documenti supportati da una cartella"""
    docs_path = Path(folder_path)
    
    if not docs_path.exists():
        print(f"❌ La cartella '{folder_path}' non esiste")
        print(f"💡 Creo la cartella '{folder_path}' - aggiungi i tuoi documenti lì")
        docs_path.mkdir(parents=True, exist_ok=True)
        return []
    
    supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.csv', '.md', '.markdown'}
    all_documents = []
    
    print(f" Cercando documenti in: {docs_path.absolute()}")
    
    # Trova tutti i file supportati
    found_files = []
    for extension in supported_extensions:
        found_files.extend(docs_path.glob(f"*{extension}"))
        found_files.extend(docs_path.glob(f"**/*{extension}"))  # Include sottocartelle
    
    if not found_files:
        print(f"⚠️ Nessun documento trovato in '{folder_path}'")
        print(f"💡 Formati supportati: {', '.join(supported_extensions)}")
        return []
    
    print(f"📄 Trovati {len(found_files)} file:")
    
    for file_path in found_files:
        print(f"   📄 {file_path.name} ({file_path.suffix})")
        documents = load_document_by_extension(file_path)
        all_documents.extend(documents)
    
    print(f" Caricati {len(all_documents)} documenti totali")
    return all_documents

# =========================
# Componenti di base - IDENTICI ALL'ORIGINALE
# =========================

def check_azure_config(settings: Settings) -> Dict[str, str]:
    """Verifica e ritorna la configurazione Azure OpenAI"""
    config = {}
    required_vars = [
        settings.azure_openai_endpoint,
        settings.azure_openai_api_key,
        settings.azure_openai_api_version,
        settings.azure_embeddings_deployment
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if not value:
            raise RuntimeError(f"Variabile d'ambiente mancante: {var}")
        config[var] = value
    
    # Chat deployment è opzionale
    chat_deployment = os.getenv(settings.azure_chat_deployment)
    if chat_deployment:
        config[settings.azure_chat_deployment] = chat_deployment
    
    return config

def get_embeddings(settings: Settings):
    """Inizializza il client per gli embeddings Azure OpenAI"""
    config = check_azure_config(settings)
    
    try:
        # Create a custom embeddings class using OpenAI client directly
        class CustomAzureEmbeddings:
            def __init__(self, azure_endpoint, api_key, api_version, azure_deployment):
                self.client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
                self.azure_deployment = azure_deployment
            
            def embed_query(self, text):
                response = self.client.embeddings.create(
                    model=self.azure_deployment,
                    input=text
                )
                return response.data[0].embedding
            
            def embed_documents(self, texts):
                response = self.client.embeddings.create(
                    model=self.azure_deployment,
                    input=texts
                )
                return [data.embedding for data in response.data]
        
        embeddings = CustomAzureEmbeddings(
            azure_endpoint=config[settings.azure_openai_endpoint],
            api_key=config[settings.azure_openai_api_key],
            api_version=config[settings.azure_openai_api_version],
            azure_deployment=config[settings.azure_embeddings_deployment]
        )
        
        # Test di connessione
        print("🔌 Testando connessione embeddings...")
        test_embed = embeddings.embed_query("test")
        print(f" Embeddings configurati correttamente (dimensione: {len(test_embed)})")
        return embeddings
        
    except Exception as e:
        print(f"❌ Errore nella configurazione degli embeddings: {e}")
        print("\nControlla le seguenti variabili d'ambiente nel file .env:")
        for var in [settings.azure_openai_endpoint, settings.azure_openai_api_key, 
                   settings.azure_openai_api_version, settings.azure_embeddings_deployment]:
            print(f"  {var} = {os.getenv(var, 'NON IMPOSTATA')}")
        raise

def _probe_embedding_dimension(embeddings) -> int:
    """Determina la dimensione dei vettori di embedding"""
    try:
        vec = embeddings.embed_query("dimension probe")
        return len(vec)
    except Exception as exc:
        raise RuntimeError(f"Failed to probe embedding dimension: {exc}")

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Divide i documenti in chunk più piccoli"""
    if not docs:
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️ Documenti divisi in {len(chunks)} chunk")
    return chunks

# =========================
# Qdrant - IDENTICO ALL'ORIGINALE
# =========================

def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Inizializza il client Qdrant"""
    try:
        client = QdrantClient(url=settings.qdrant_url)
        # Test di connessione
        collections = client.get_collections()
        print(f"🗄️ Connesso a Qdrant su {settings.qdrant_url}")
        return client
    except Exception as e:
        print(f"❌ Errore di connessione a Qdrant: {e}")
        print("💡 Assicurati che Qdrant sia in esecuzione su http://localhost:6333")
        print("💡 Avvia con: docker run -p 6333:6333 qdrant/qdrant")
        raise

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """Ricrea la collection per RAG con configurazioni ottimali"""
    print(f"🗄️ Creando collection '{settings.collection}' (dimensione vettori: {vector_size})...")
    
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,
            ef_construct=256
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)
        ),
    )

    # Indici per ricerca
    client.create_payload_index(
        collection_name=settings.collection,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    for key in ["source_file", "file_extension", "chunk_id"]:
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )
    
    print(" Collection e indici creati")

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    """Costruisce i punti per l'inserimento in Qdrant"""
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "source_file": doc.metadata.get("source_file", "unknown"),
            "file_path": doc.metadata.get("file_path", ""),
            "file_extension": doc.metadata.get("file_extension", ""),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings: Any):
    """Carica i chunk nella collection Qdrant"""
    if not chunks:
        print("⚠️ Nessun chunk da inserire")
        return
        
    print(f"🧮 Generando embeddings per {len(chunks)} chunk...")
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    
    print("📤 Inserendo dati in Qdrant...")
    points = build_points(chunks, vecs)
    client.upsert(collection_name=settings.collection, points=points, wait=True)
    print(" Dati inseriti con successo")

# =========================
# Ricerca - IDENTICA ALL'ORIGINALE
# =========================

def qdrant_semantic_search(client: QdrantClient, settings: Settings, query: str, embeddings: Any, limit: int):
    """Ricerca semantica usando vettori"""
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=True,
        search_params=SearchParams(hnsw_ef=256, exact=False),
    )
    return res.points

def qdrant_text_prefilter_ids(client: QdrantClient, settings: Settings, query: str, max_hits: int) -> List[int]:
    """Ottiene gli ID dei documenti che corrispondono alla ricerca testuale"""
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

def mmr_select(query_vec: List[float], candidates_vecs: List[List[float]], k: int, lambda_mult: float) -> List[int]:
    """Implementazione MMR per diversificare i risultati"""
    import numpy as np
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

def hybrid_search(client: QdrantClient, settings: Settings, query: str, embeddings: Any):
    """Ricerca ibrida che combina semantica e testuale"""
    sem = qdrant_semantic_search(client, settings, query, embeddings, settings.top_n_semantic)
    if not sem:
        return []

    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))

    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    fused: List[Tuple[int, float, Any]] = []
    for idx, p in enumerate(sem):
        base = norm(p.score)
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost
        fused.append((idx, fuse, p))

    fused.sort(key=lambda t: t[1], reverse=True)

    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        N = min(len(fused), max(settings.final_k * 3, settings.final_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    return [p for _, _, p in fused[:settings.final_k]]

def format_docs_for_display(points: Iterable[Any]) -> str:
    """Formatta i documenti per la visualizzazione"""
    blocks = []
    for i, p in enumerate(points, 1):
        pay = p.payload or {}
        src = pay.get("source_file", "unknown")
        text = pay.get("text", "").strip()
        
        # Tronca il testo se troppo lungo
        if len(text) > 300:
            text = text[:300] + "..."
            
        blocks.append(f"{i}. 📄 {src} (Score: {p.score:.3f})\n   {text}")
    return "\n\n".join(blocks)

# =========================
# VARIABILI GLOBALI - SINGLETON PATTERN
# =========================

_embeddings = None
_client = None
_system_initialized = False

def initialize_rag_system():
    """Inizializza il sistema RAG una sola volta"""
    global _embeddings, _client, _system_initialized
    
    if _system_initialized:
        return _embeddings, _client
    
    try:
        print(" Inizializzazione sistema RAG per tool...")
        
        # Inizializza componenti usando funzioni originali
        _embeddings = get_embeddings(SETTINGS)
        _client = get_qdrant_client(SETTINGS)
        
        # Carica documenti
        docs = load_documents_from_folder(SETTINGS.docs_folder)
        
        if docs:
            # Processa documenti
            chunks = split_documents(docs, SETTINGS)
            
            if chunks:
                # Setup Qdrant
                vector_size = _probe_embedding_dimension(_embeddings)
                recreate_collection_for_rag(_client, SETTINGS, vector_size)
                upsert_chunks(_client, SETTINGS, chunks, _embeddings)
        
        _system_initialized = True
        print(" Sistema RAG inizializzato per tool")
        
        return _embeddings, _client
        
    except Exception as e:
        print(f"❌ Errore nell'inizializzazione del sistema RAG: {e}")
        return None, None

# =========================
# TOOL CON DECORATORE - Input via parametro invece di input()
# =========================

@tool("document_search")
def document_search(query: str) -> str:
    """
    Cerca informazioni nei documenti della knowledge base locale usando ricerca ibrida avanzata.
    
    Questo tool mantiene TUTTE le funzionalità del sistema RAG originale:
    - Ricerca semantica con embeddings Azure OpenAI
    - Ricerca testuale con match esatti su Qdrant
    - Algoritmo MMR per diversificare risultati e ridurre ridondanza
    - Ricerca ibrida che combina semantica + testuale con pesi configurabili
    - Supporto completo per documenti .txt, .pdf, .docx, .csv, .md, .markdown
    - Configurazione Qdrant avanzata (HNSW, quantizzazione, indici)
    - Chunking con parametri configurabili (size, overlap, separatori)
    - Gestione completa metadati documenti
    
    La domanda dell'utente viene passata come parametro dal flow della crew,
    sostituendo l'input() interattivo del sistema originale.
    
    Args:
        query: La domanda dell'utente da processare (arriva dal flow CrewAI)
        
    Returns:
        Informazioni trovate nei documenti con la stessa formattazione del sistema originale
    """
    
    if not query or not query.strip():
        return "Errore: query di ricerca vuota"
    
    # Inizializza sistema RAG se necessario (mantiene singleton pattern)
    embeddings, client = initialize_rag_system()
    
    if not embeddings or not client:
        return (
            "Sistema RAG non disponibile. "
            "Verificare la configurazione Azure OpenAI e che Qdrant sia in esecuzione."
        )
    
    try:
        print(f" Ricerca ibrida per: '{query}'")
        
        # Usa IDENTICA funzione di ricerca ibrida del file originale
        results = hybrid_search(client, SETTINGS, query, embeddings)
        
        if not results:
            return (
                f"❌ Nessun documento rilevante trovato.\n"
                f"💡 Prova con parole chiave diverse o aggiungi più documenti alla cartella 'docs/'"
            )
        
        print(f" Trovati {len(results)} documenti rilevanti")
        
        # Usa IDENTICA formattazione del file originale
        formatted_results = format_docs_for_display(results)
        
        return formatted_results
        
    except Exception as e:
        return f"❌ Errore durante la ricerca: {e}"

# Create a simple tool wrapper for CrewAI
document_search_tool = SimpleTool(
    name="document_search",
    description="Ricerca ibrida nella knowledge base locale (Qdrant + Azure embeddings)",
    func=document_search
)

# =========================
# MAIN
# =========================

def main():
    """Test del tool completo"""
    result = document_search("Come si prepara la pasta al pomodoro")
    print(result)

if __name__ == "__main__":
    main()