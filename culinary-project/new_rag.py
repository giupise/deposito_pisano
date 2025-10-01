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

evaluations = []

# =========================
# Configurazione
# =========================

# Load .env co-located with this file
load_dotenv(dotenv_path=Path(__file__).with_name('.env'))

@dataclass
class Settings:
    # Paths
    docs_folder: str = "docs"
    
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
# Caricamento documenti da cartella
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
# Componenti di base (invariati)
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
            model="text-embedding-ada-002"
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

def get_llm(settings: Settings):
    """Inizializza il modello di linguaggio"""
    try:
        config = check_azure_config(settings)
        chat_deployment = os.getenv(settings.azure_chat_deployment)
        
        if not chat_deployment:
            print("⚠️ Chat deployment non configurato - saltando la generazione")
            return None
            
        llm = AzureChatOpenAI(
            azure_endpoint=config[settings.azure_openai_endpoint],
            api_key=config[settings.azure_openai_api_key],
            api_version=config[settings.azure_openai_api_version],
            azure_deployment=chat_deployment,
            temperature=0.1
        )
        
        # Test di connessione
        print("🔌 Testando connessione LLM...")
        test_response = llm.invoke("Rispondi con 'OK' se funziono correttamente.")
        print(f" LLM configurato correttamente")
        return llm
        
    except Exception as e:
        print(f"⚠️ LLM non disponibile: {e}")
        print("📖 Modalità solo recupero documenti attivata")
        return None

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
# Qdrant: creazione collection + indici (invariati)
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
# Ricerca (invariata ma semplificata per output)
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

# =========================
# Generazione risposta
# =========================

def format_docs_for_prompt(points: Iterable[Any]) -> str:
    """Formatta i documenti per il prompt"""
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source_file", "unknown")
        text = pay.get("text", "").strip()
        blocks.append(f"[Fonte: {src}]\n{text}")
    return "\n\n" + "="*50 + "\n\n".join(blocks)

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

def build_rag_chain(llm):
    """Costruisce la chain RAG per generare risposte"""
    system_prompt = (
        "Sei un assistente esperto che risponde a domande basandosi sui documenti forniti. "
        "Rispondi in italiano, in modo accurato e dettagliato. "
        "Usa SOLO le informazioni presenti nei documenti forniti. "
        "Se l'informazione non è presente nei documenti, dichiaralo chiaramente. "
        "Cita sempre le fonti usando [Fonte: NOME_FILE]."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", 
         "DOMANDA: {question}\n\n"
         "DOCUMENTI DISPONIBILI:\n{context}\n\n"
         "Fornisci una risposta completa basata sui documenti, citando le fonti.")
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
# Interfaccia interattiva
# =========================

def print_header():
    """Stampa l'header del sistema"""
    print("=" * 70)
    print("🤖 SISTEMA RAG INTERATTIVO")
    print(" Legge documenti dalla cartella 'docs/'")
    print("💬 Rispondi alle tue domande basandosi sui documenti")
    print("=" * 70)

def print_help():
    """Stampa i comandi disponibili"""
    print("\n💡 COMANDI DISPONIBILI:")
    print("  'help' - Mostra questo messaggio")
    print("  'reload' - Ricarica i documenti dalla cartella")
    print("  'stats' - Mostra statistiche sui documenti")
    print("  'exit' o 'quit' - Esci dal programma")
    print("  Oppure fai una domanda sui tuoi documenti!\n")

def print_stats(client: QdrantClient, settings: Settings):
    """Stampa statistiche sui documenti caricati"""
    try:
        info = client.get_collection(settings.collection)
        print(f"\n📊 STATISTICHE DOCUMENTI:")
        print(f"   📄 Chunk totali: {info.points_count}")
        print(f"   📏 Dimensione vettori: {info.config.params.vectors.size}")
        print(f"   🗄️ Collection: {settings.collection}")
        
        # Conta i file unici
        points, _ = client.scroll(
            collection_name=settings.collection,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        
        files = set()
        for point in points:
            source = point.payload.get('source_file')
            if source:
                files.add(source)
        
        print(f"    File unici: {len(files)}")
        if files:
            print(f"    File caricati: {', '.join(sorted(files))}")
        print()
        
    except Exception as e:
        print(f"❌ Errore nel recuperare statistiche: {e}\n")

def interactive_mode(client: QdrantClient, settings: Settings, embeddings: Any, llm: Any):
    """Modalità interattiva per fare domande"""
    print_header()
    print_help()
    
    while True:
        try:
            # Input utente
            query = input("🤔 La tua domanda (o 'help' per aiuto): ").strip()
            
            if not query:
                continue
                
            # Comandi speciali
            if query.lower() in ['exit', 'quit']:
                print("👋 Arrivederci!")
                break
                
            elif query.lower() == 'help':
                print_help()
                continue
                
            elif query.lower() == 'stats':
                print_stats(client, settings)
                continue
                
            elif query.lower() == 'reload':
                print(" Ricaricamento documenti...")
                return 'reload'  # Segnala di ricaricare
            
            # Ricerca documenti
            print(f"\n🔍 Cercando informazioni per: '{query}'")
            hits = hybrid_search(client, settings, query, embeddings)
            
            if not hits:
                print("❌ Nessun documento rilevante trovato.")
                print("💡 Prova con parole chiave diverse o aggiungi più documenti alla cartella 'docs/'\n")
                continue
            
            print(f" Trovati {len(hits)} documenti rilevanti")
            
            # Generazione risposta
            if llm:
                try:
                    print("🧠 Generando risposta...")
                    ctx = format_docs_for_prompt(hits)
                    chain = build_rag_chain(llm)
                    answer = chain.invoke({"question": query, "context": ctx})
                    
                    print(f"\n🤖 RISPOSTA:")
                    print("=" * 50)
                    print(answer)
                    print("=" * 50)
                    
                except Exception as e:
                    print(f"❌ Errore nella generazione: {e}")
                    print("\n📄 Ecco i documenti più rilevanti trovati:")
                    print(format_docs_for_display(hits))
            else:
                print("\n📄 DOCUMENTI PIÙ RILEVANTI:")
                print("=" * 50)
                print(format_docs_for_display(hits))
                print("=" * 50)
            
            print()  # Riga vuota per separare le sessioni
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrotto dall'utente. Arrivederci!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            print("💡 Prova di nuovo o digita 'help' per aiuto\n")

# =========================
# Interfaccia interattiva (PATCHED)
# =========================

def interactive_mode(client: QdrantClient, settings: Settings, embeddings: Any, llm: Any):
    """Modalità interattiva per fare domande"""
    print_header()
    print_help()
    
    while True:
        try:
            # Input utente
            query = input("🤔 La tua domanda (o 'help' per aiuto): ").strip()
            
            if not query:
                continue
                
            # Comandi speciali
            if query.lower() in ['exit', 'quit']:
                print("👋 Arrivederci!")
                break
                
            elif query.lower() == 'help':
                print_help()
                continue
                
            elif query.lower() == 'stats':
                print_stats(client, settings)
                continue
                
            elif query.lower() == 'reload':
                print(" Ricaricamento documenti...")
                return 'reload'  # Segnala di ricaricare
            
            # Ricerca documenti
            print(f"\n🔍 Cercando informazioni per: '{query}'")
            hits = hybrid_search(client, settings, query, embeddings)
            
            if not hits:
                print("❌ Nessun documento rilevante trovato.")
                print("💡 Prova con parole chiave diverse o aggiungi più documenti alla cartella 'docs/'\n")
                continue
            
            print(f" Trovati {len(hits)} documenti rilevanti")
            
            # Generazione risposta
            if llm:
                try:
                    print("🧠 Generando risposta...")
                    ctx = format_docs_for_prompt(hits)
                    chain = build_rag_chain(llm)
                    answer = chain.invoke({"question": query, "context": ctx})
                    
                    print(f"\n🤖 RISPOSTA:")
                    print("=" * 50)
                    print(answer)
                    print("=" * 50)

                    # =========================
                    # NEW: Salva interazione per Ragas
                    # =========================
                    evaluations.append({
                        "question": query,
                        "answer": answer,
                        "contexts": [p.payload.get("text", "") for p in hits],
                        # opzionale: "ground_truth": "risposta attesa"
                    })
                    
                except Exception as e:
                    print(f"❌ Errore nella generazione: {e}")
                    print("\n📄 Ecco i documenti più rilevanti trovati:")
                    print(format_docs_for_display(hits))
            else:
                print("\n📄 DOCUMENTI PIÙ RILEVANTI:")
                print("=" * 50)
                print(format_docs_for_display(hits))
                print("=" * 50)
            
            print()  # Riga vuota per separare le sessioni
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrotto dall'utente. Arrivederci!")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")
            print("💡 Prova di nuovo o digita 'help' per aiuto\n")


# =========================
# NEW: Funzione valutazione Ragas
# =========================
def evaluate_with_ragas():
    """Valuta le performance del RAG usando Ragas"""
    global evaluations
    if not evaluations:
        print("⚠️ Nessuna interazione salvata, niente da valutare.")
        return
    
    print(" Valutazione con Ragas in corso...")
    
    # Trasforma in dataset HuggingFace
    dataset = Dataset.from_list(evaluations)

    # Esegui valutazione
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )

    # Mostra risultati
    print("\n=== RISULTATI RAGAS ===")
    print(result)

    # Salva su CSV per analisi successive
    df_results = result.to_pandas()
    df_results.to_csv("ragas_eval.csv", index=False)
    print(" Risultati salvati in ragas_eval.csv")


# =========================
# Main function (PATCHED)
# =========================

def main():
    """Funzione principale del sistema RAG interattivo"""
    s = SETTINGS
    
    while True:  # Loop per permettere il reload
        try:
            print("Inizializzazione sistema RAG...")
            
            # 1) Inizializzazione componenti
            embeddings = get_embeddings(s)
            llm = get_llm(s)
            client = get_qdrant_client(s)
            
            # 2) Caricamento documenti
            print(f"\n📁 Caricamento documenti da '{s.docs_folder}'...")
            docs = load_documents_from_folder(s.docs_folder)
            
            if not docs:
                print(f"\n⚠️ Nessun documento trovato nella cartella '{s.docs_folder}'")
                print("💡 Aggiungi documenti (.txt, .pdf, .docx, .csv, .md) alla cartella e riprova")
                
                docs_path = Path(s.docs_folder)
                docs_path.mkdir(exist_ok=True)
                
                example_doc = docs_path / "esempio.txt"
                if not example_doc.exists():
                    example_doc.write_text(
                        "Questo è un documento di esempio per testare il sistema RAG.\n\n"
                        "Il sistema RAG (Retrieval-Augmented Generation) combina la ricerca di documenti "
                        "con la generazione di testi per fornire risposte accurate basate su fonti specifiche.\n\n"
                        "Puoi aggiungere i tuoi documenti nella cartella 'docs' e fare domande su di essi!"
                    )
                    print(f"📄 Creato documento di esempio: {example_doc}")
                    docs = load_documents_from_folder(s.docs_folder)
                
                if not docs:
                    input("\n⏸️ Premi INVIO dopo aver aggiunto documenti...")
                    continue
            
            # 3) Chunking
            chunks = split_documents(docs, s)
            
            # 4) Setup Qdrant
            vector_size = _probe_embedding_dimension(embeddings)
            recreate_collection_for_rag(client, s, vector_size)
            upsert_chunks(client, s, chunks, embeddings)
            
            # 5) Modalità interattiva
            result = interactive_mode(client, s, embeddings, llm)

            # =========================
            # NEW: valutazione Ragas al termine
            # =========================
            evaluate_with_ragas()
            
            if result != 'reload':
                break  # Exit se non è stato richiesto reload
                
        except KeyboardInterrupt:
            print("\n\n👋 Programma interrotto. Arrivederci!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Errore critico: {e}")
            import traceback
            traceback.print_exc()
            
            retry = input("\n🔄 Vuoi riprovare? (s/n): ").strip().lower()
            if retry not in ['s', 'si', 'sì', 'y', 'yes']:
                break

if __name__ == "__main__":
    main()
