from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain Core components for prompt/chain construction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import init_chat_model

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
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

# =========================
# Configurazione
# =========================

# Load .env co-located with this file (e.g., 26-09/.env)
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

@dataclass
class Settings:
    qdrant_url: str = "http://localhost:6333"
    collection: str = "rag_chunks"
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: str = "AZURE_OPENAI_ENDPOINT"
    azure_openai_api_key: str = "AZURE_OPENAI_API_KEY" 
    azure_openai_api_version: str = "AZURE_OPENAI_API_VERSION"
    azure_embeddings_deployment: str = "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"
    azure_chat_deployment: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"
    
    # Chunking parameters
    chunk_size: int = 700
    chunk_overlap: int = 120
    
    # Retrieval parameters
    top_n_semantic: int = 30
    top_n_text: int = 100
    final_k: int = 6
    alpha: float = 0.75
    text_boost: float = 0.20
    
    # MMR parameters
    use_mmr: bool = True
    mmr_lambda: float = 0.6

SETTINGS = Settings()

# =========================
# Componenti di base
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
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config[settings.azure_openai_endpoint],
            api_key=config[settings.azure_openai_api_key],
            api_version=config[settings.azure_openai_api_version],
            azure_deployment=config[settings.azure_embeddings_deployment],
            model="text-embedding-ada-002"  # Specifica il modello
        )
        
        # Test di connessione
        print("Testing embeddings connection...")
        test_embed = embeddings.embed_query("test")
        print(f"✓ Embeddings configurati correttamente (dimensione: {len(test_embed)})")
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

def get_llm(settings: Settings):
    """Inizializza il modello di linguaggio"""
    try:
        config = check_azure_config(settings)
        chat_deployment = os.getenv(settings.azure_chat_deployment)
        
        if not chat_deployment:
            print("❌ Chat deployment non configurato - saltando la generazione")
            return None
            
        llm = AzureChatOpenAI(
            azure_endpoint=config[settings.azure_openai_endpoint],
            api_key=config[settings.azure_openai_api_key],
            api_version=config[settings.azure_openai_api_version],
            azure_deployment=chat_deployment,
            temperature=0.1
        )
        
        # Test di connessione
        print("Testing LLM connection...")
        test_response = llm.invoke("Rispondi con 'OK' se funziono correttamente.")
        print(f"✓ LLM configurato correttamente: {test_response.content}")
        return llm
        
    except Exception as e:
        print(f"❌ Errore nella configurazione dell'LLM: {e}")
        print("Continuando senza LLM - verrà mostrato solo il contenuto recuperato")
        return None

def simulate_corpus() -> List[Document]:
    """Crea un corpus di documenti di esempio"""
    docs = [
        Document(
            page_content=(
                "LangChain è un framework per costruire applicazioni con Large Language Models. "
                "Fornisce chains, agents, template di prompt, memoria e molte integrazioni. "
                "È particolarmente utile per creare applicazioni RAG (Retrieval-Augmented Generation)."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md", "title": "Intro LangChain", "lang": "it"}
        ),
        Document(
            page_content=(
                "FAISS è una libreria per la ricerca efficiente di similarità su vettori densi. "
                "Supporta sia la ricerca esatta che quella approssimata dei vicini più prossimi su larga scala. "
                "È sviluppata da Meta AI Research ed è ampiamente utilizzata in produzione."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md", "title": "FAISS Overview", "lang": "it"}
        ),
        Document(
            page_content=(
                "I sentence-transformers come all-MiniLM-L6-v2 producono embeddings di 384 dimensioni "
                "per ricerca semantica, clustering e generazione aumentata da recupero. "
                "Sono modelli leggeri e veloci, ideali per applicazioni in tempo reale."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md", "title": "MiniLM Embeddings", "lang": "it"}
        ),
        Document(
            page_content=(
                "Una tipica pipeline RAG include indicizzazione (carica, dividi, embed, salva), "
                "recupero e generazione. Il recupero seleziona i chunk più rilevanti, "
                "poi l'LLM risponde basandosi su quei chunk per fornire informazioni accurate."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md", "title": "RAG Pipeline", "lang": "it"}
        ),
        Document(
            page_content=(
                "La Maximal Marginal Relevance (MMR) bilancia rilevanza e diversità "
                "per ridurre la ridondanza e migliorare la copertura di aspetti distinti "
                "nei chunk recuperati. È essenziale per evitare informazioni duplicate."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md", "title": "MMR Retrieval", "lang": "it"}
        ),
        Document(
            page_content=(
                "Qdrant è un database vettoriale open-source ottimizzato per applicazioni AI. "
                "Supporta ricerca ibrida, filtri sui metadati, quantizzazione e clustering. "
                "È scritto in Rust per massime performance e può gestire miliardi di vettori."
            ),
            metadata={"id": "doc6", "source": "qdrant-intro.md", "title": "Qdrant Database", "lang": "it"}
        ),
    ]
    return docs

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """Divide i documenti in chunk più piccoli"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✓ Documenti divisi in {len(chunks)} chunk")
    return chunks

# =========================
# Qdrant: creazione collection + indici
# =========================

def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Inizializza il client Qdrant"""
    try:
        client = QdrantClient(url=settings.qdrant_url)
        # Test di connessione
        collections = client.get_collections()
        print(f"✓ Connesso a Qdrant su {settings.qdrant_url}")
        return client
    except Exception as e:
        print(f"❌ Errore di connessione a Qdrant: {e}")
        print("Assicurati che Qdrant sia in esecuzione su http://localhost:6333")
        raise

def recreate_collection_for_rag(client: QdrantClient, settings: Settings, vector_size: int):
    """Ricrea la collection per RAG con configurazioni ottimali"""
    print(f"Creando collection '{settings.collection}' con vettori di dimensione {vector_size}...")
    
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,             # grado medio del grafo HNSW
            ef_construct=256  # ampiezza lista candidati in fase costruzione
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2  # parallelismo/segmentazione iniziale
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)
        ),
    )

    # Indice full-text sul campo 'text' per filtri MatchText
    client.create_payload_index(
        collection_name=settings.collection,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Indici keyword per filtri esatti
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=settings.collection,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )
    
    print("✓ Collection e indici creati con successo")

# =========================
# Ingest: chunk -> embed -> upsert
# =========================

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    """Costruisce i punti per l'inserimento in Qdrant"""
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "it"),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks(client: QdrantClient, settings: Settings, chunks: List[Document], embeddings: Any):
    """Carica i chunk nella collection Qdrant"""
    print(f"Generando embeddings per {len(chunks)} chunk...")
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    
    print("Costruendo punti per l'inserimento...")
    points = build_points(chunks, vecs)
    
    print("Inserendo punti in Qdrant...")
    client.upsert(collection_name=settings.collection, points=points, wait=True)
    print("✓ Chunk inseriti con successo")

# =========================
# Ricerca: semantica / testuale / ibrida
# =========================

def qdrant_semantic_search(
    client: QdrantClient,
    settings: Settings,
    query: str,
    embeddings: Any,
    limit: int,
    with_vectors: bool = False
):
    """Ricerca semantica usando vettori"""
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=settings.collection,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(
            hnsw_ef=256,  # ampiezza lista in fase di ricerca
            exact=False   # usa HNSW per ricerca approssimata
        ),
    )
    return res.points

def qdrant_text_prefilter_ids(
    client: QdrantClient,
    settings: Settings,
    query: str,
    max_hits: int
) -> List[int]:
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

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]],
    k: int,
    lambda_mult: float
) -> List[int]:
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
            # Prendi il più simile per primo
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
    embeddings: Any
):
    """Ricerca ibrida che combina semantica e testuale"""
    # (1) Ricerca semantica
    sem = qdrant_semantic_search(
        client, settings, query, embeddings,
        limit=settings.top_n_semantic, with_vectors=True
    )
    if not sem:
        return []

    # (2) Ricerca testuale (prefilter)
    text_ids = set(qdrant_text_prefilter_ids(client, settings, query, settings.top_n_text))

    # Normalizzazione score semantici per fusione
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    # (3) Fusione con boost testuale
    fused: List[Tuple[int, float, Any]] = []
    for idx, p in enumerate(sem):
        base = norm(p.score)
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost
        fused.append((idx, fuse, p))

    # Ordina per score fuso discendente
    fused.sort(key=lambda t: t[1], reverse=True)

    # MMR opzionale per diversificare
    if settings.use_mmr:
        qv = embeddings.embed_query(query)
        N = min(len(fused), max(settings.final_k * 5, settings.final_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    # Altrimenti, prendi i primi final_k dopo fusione
    return [p for _, _, p in fused[:settings.final_k]]

# =========================
# Prompt/Chain per generazione con citazioni
# =========================

def format_docs_for_prompt(points: Iterable[Any]) -> str:
    """Formatta i documenti per il prompt"""
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)

def build_rag_chain(llm):
    """Costruisce la chain RAG per generare risposte"""
    system_prompt = (
        "Sei un assistente tecnico esperto. Rispondi in italiano, in modo conciso e accurato. "
        "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO fornito. "
        "Se l'informazione richiesta non è presente nel contesto, dichiara chiaramente: "
        "'L'informazione richiesta non è presente nel contesto fornito.' "
        "Cita sempre le fonti nel formato [source:NOME_FILE]."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "CONTENUTO:\n{context}\n\n"
         "Istruzioni:\n"
         "1) Basa la risposta SOLO sul contenuto fornito.\n"
         "2) Includi sempre citazioni [source:NOME_FILE].\n"
         "3) Non inventare informazioni non presenti nel contenuto.")
    ])

    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# =========================
# Main end-to-end demo
# =========================

def main():
    """Funzione principale del sistema RAG"""
    print(" Avvio sistema RAG")
    print("=" * 50)
    
    s = SETTINGS
    
    try:
        # 1) Inizializzazione componenti
        print("Inizializzando componenti...")
        embeddings = get_embeddings(s)
        llm = get_llm(s)
        client = get_qdrant_client(s)
        
        # 2) Preparazione dati
        print("\nPreparando i dati...")
        docs = simulate_corpus()
        chunks = split_documents(docs, s)
        
        # 3) Creazione collection
        print(f"\nConfigurazione collection...")
        vector_size = _probe_embedding_dimension(embeddings)
        recreate_collection_for_rag(client, s, vector_size)
        
        # 4) Inserimento dati
        print(f"\nInserimento dati...")
        upsert_chunks(client, s, chunks, embeddings)
        
        # 5) Test con query multiple
        questions = [
            "Cos'è una pipeline RAG e quali sono le sue fasi principali?",
            "A cosa serve FAISS e che caratteristiche offre per la ricerca?",
            "Che cos'è MMR e perché è utile per ridurre la ridondanza?",
            "Qual è la dimensione degli embedding di all-MiniLM-L6-v2?",
            "Cosa distingue Qdrant dagli altri database vettoriali?",
        ]
        
        print(f"\n Testando con {len(questions)} domande")
        print("=" * 80)
        
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Q: {q}")
            print("-" * 60)
            
            # Ricerca ibrida
            hits = hybrid_search(client, s, q, embeddings)
            
            if not hits:
                print("❌ Nessun risultato trovato.")
                continue
            
            # Debug info
            print(f"✓ Trovati {len(hits)} risultati:")
            for j, p in enumerate(hits, 1):
                src = p.payload.get('source', 'unknown')
                print(f"  {j}. [score:{p.score:.3f}] {src}")
            
            # Generazione risposta
            if llm:
                try:
                    print(f"\n💭 Generando risposta...")
                    ctx = format_docs_for_prompt(hits)
                    chain = build_rag_chain(llm)
                    
                    # Passa sia question che context
                    answer = chain.invoke({"question": q, "context": ctx})
                    print(f"\n Risposta:\n{answer}")
                    
                except Exception as e:
                    print(f"❌ Errore nella generazione: {e}")
                    print("\n📄 Contenuto recuperato:")
                    print(format_docs_for_prompt(hits))
            else:
                print("\n📄 Contenuto recuperato:")
                print(format_docs_for_prompt(hits))
            
            print("\n" + "=" * 80)
    
    except Exception as e:
        print(f"❌ Errore critico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()