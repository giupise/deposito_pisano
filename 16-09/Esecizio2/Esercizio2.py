from __future__ import annotations

import os
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
from dotenv import load_dotenv

# LangChain core / schema
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Azure OpenAI (LangChain)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Vector store + text splitter + loaders
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ddgs import DDGS
import ssl
import urllib3


# Disabilita warnings SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# setup
@dataclass
class Settings:
    # Dati & persistenza
    data_dir: str = "documents"         # cartella con i documenti reali
    persist_dir: str = "faiss_index_example"

    # Text splitting
    chunk_size: int = 1000              # Aumentato per chunk più grandi
    chunk_overlap: int = 200            # Aumentato overlap

    # Retriever (MMR)
    search_type: str = "mmr"            # "mmr" o "similarity"
    k: int = 4                          # Aumentato numero risultati
    fetch_k: int = 10                   # Aumentato candidati iniziali
    mmr_lambda: float = 0.5             # Bilanciato tra diversificazione e pertinenza

    # Env per Azure OpenAI Embeddings
    azure_embeddings_endpoint_env: str = "AZURE_EMBEDDINGS_ENDPOINT"
    azure_embeddings_key_env: str = "AZURE_EMBEDDINGS_KEY"
    azure_embeddings_api_version_env: str = "AZURE_EMBEDDINGS_API_VERSION"
    azure_embeddings_deployment_env: str = "AZURE_EMBEDDINGS_DEPLOYMENT"

    # Env per Azure OpenAI Chat
    azure_chat_endpoint_env: str = "AZURE_CHAT_ENDPOINT"
    azure_chat_key_env: str = "AZURE_CHAT_KEY"
    azure_chat_api_version_env: str = "AZURE_CHAT_API_VERSION"
    azure_chat_deployment_env: str = "AZURE_CHAT_DEPLOYMENT"

    # Web search fallback
    enable_web_search: bool = True
    web_search_keywords: List[str] = None  # es: ["Azure OpenAI", "LangChain", "FAISS"]
    web_search_max_results: int = 3
    web_search_region: str = "it-it"  # regione per risultati in italiano
    
    def __post_init__(self):
        if self.web_search_keywords is None:
            self.web_search_keywords = ["Azure OpenAI", "Microsoft AI", "LangChain RAG"]


SETTINGS = Settings()

# Funzioni helper 

from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
try:
    from langchain_community.document_loaders import PyMuPDFLoader
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# carica un singolo file
def _load_single_file(path: Path) -> List[Document]:
    ext = path.suffix.lower()
    docs: List[Document] = []
    
    print(f"[DEBUG] Tentativo di caricare: {path}")
    
    try:
        if ext in {".txt", ".md", ".markdown"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
        elif ext == ".docx":
            loader = Docx2txtLoader(str(path))
            docs = loader.load()
        elif ext == ".pdf":
            if HAS_PYMUPDF:
                loader = PyMuPDFLoader(str(path))
                docs = loader.load()
            else:
                loader = PyPDFLoader(str(path))
                docs = loader.load()
        else:
            print(f"[SKIP] Tipo file non supportato: {path.name}")
            return []
    except Exception as e:
        print(f"[WARN] Errore nel leggere {path.name}: {e}")
        return []

    # DEBUG: Mostra contenuto caricato
    if docs:
        print(f"[DEBUG] File {path.name} caricato con successo:")
        print(f"        Numero di documenti: {len(docs)}")
        if docs[0].page_content:
            print(f"        Primi 200 caratteri: {docs[0].page_content[:200]}...")

    for d in docs:
        d.metadata = d.metadata or {}
        d.metadata.setdefault("source", path.name)
        d.metadata.setdefault("filepath", str(path.resolve()))
    
    return docs

# carica tutti i documenti in una cartella
def load_documents_from_dir(data_dir: str) -> List[Document]:
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"La cartella dati non esiste: {base.resolve()}")

    patterns = ["**/*.txt", "**/*.md", "**/*.markdown", "**/*.pdf", "**/*.docx"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(base.rglob(pat))

    if not files:
        raise RuntimeError(f"Nessun file trovato in {base.resolve()} (txt, md, pdf, docx)")

    # DEBUG: Mostra i file trovati
    print(f"\n[DEBUG] File trovati in {base.resolve()}:")
    for f in files:
        print(f"  - {f}")

    all_docs: List[Document] = []
    for p in sorted(files):
        all_docs.extend(_load_single_file(p))

    print(f"\n[INFO] Documenti caricati: {len(all_docs)} da {len(files)} file")
    
    # DEBUG: Mostra un riepilogo del contenuto
    if all_docs:
        print("\n[DEBUG] Riepilogo contenuto documenti:")
        for i, doc in enumerate(all_docs[:3]):  # Mostra solo i primi 3
            print(f"\nDocumento {i+1}:")
            print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            print(f"  Contenuto (primi 150 char): {doc.page_content[:150]}...")
    
    return all_docs

# determina determinare la dimensione dei vettori di embedding generati dal modello corrente
def _current_embedding_dim(embeddings: Embeddings) -> int:
    """
    Calcola la dimensione dell'embedding corrente con una query di prova.
    """
    vec = embeddings.embed_query("dimension probe")
    print(f"[DEBUG] Dimensione embedding: {len(vec)}")
    return len(vec)



# Funzioni per componenti AI 

def get_embeddings(settings: Settings) -> Embeddings:
    """
    Restituisce embeddings da Azure OpenAI
    """
    endpoint = os.getenv(settings.azure_embeddings_endpoint_env)
    api_key = os.getenv(settings.azure_embeddings_key_env)
    api_version = os.getenv(settings.azure_embeddings_api_version_env)
    deployment = os.getenv(settings.azure_embeddings_deployment_env)

    missing = [name for name, val in [
        (settings.azure_embeddings_endpoint_env, endpoint),
        (settings.azure_embeddings_key_env, api_key),
        (settings.azure_embeddings_api_version_env, api_version),
        (settings.azure_embeddings_deployment_env, deployment),
    ] if not val]
    if missing:
        raise RuntimeError("Variabili d'ambiente embeddings mancanti: " + ", ".join(missing))

    print(f"\n[DEBUG] Configurazione embeddings:")
    print(f"  Endpoint: {endpoint}")
    print(f"  API Version: {api_version}")
    print(f"  Deployment: {deployment}")

    return AzureOpenAIEmbeddings(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        api_version=api_version,
        azure_deployment=deployment,
    )


def get_llm_from_azure_chat(settings: Settings) -> AzureChatOpenAI:
    """
    Restituisce un chat model Azure OpenAI
    """
    endpoint = os.getenv(settings.azure_chat_endpoint_env)
    api_key = os.getenv(settings.azure_chat_key_env)
    api_ver = os.getenv(settings.azure_chat_api_version_env)
    chat_deployment = os.getenv(settings.azure_chat_deployment_env)

    missing = [n for n, v in [
        (settings.azure_chat_endpoint_env, endpoint),
        (settings.azure_chat_key_env, api_key),
        (settings.azure_chat_api_version_env, api_ver),
        (settings.azure_chat_deployment_env, chat_deployment),
    ] if not v]
    if missing:
        raise RuntimeError("Variabili d'ambiente chat mancanti: " + ", ".join(missing))

    print(f"\n[DEBUG] Configurazione chat LLM:")
    print(f"  Endpoint: {endpoint}")
    print(f"  API Version: {api_ver}")
    print(f"  Deployment: {chat_deployment}")

    return AzureChatOpenAI(
        azure_endpoint=endpoint.rstrip("/"),
        api_key=api_key,
        api_version=api_ver,
        azure_deployment=chat_deployment,
        temperature=0,
        timeout=60,
    )


# Funzione di ricerca web
def search_web_fallback(query: str, keywords: List[str], settings: Settings) -> List[Document]:
    """
    Cerca informazioni su web quando non trovate nel contesto locale.
    """
    print(f"\n[DEBUG] Ricerca web fallback per: '{query}'")
    
    web_docs = []
    
    try:
        with DDGS(verify=False) as ddgs:
            # Prima prova la query originale
            try:
                print(f"[DEBUG] Cercando query originale: {query}")
                results = ddgs.text(
                    query,
                    region=settings.web_search_region,
                    safesearch='off',
                    max_results=settings.web_search_max_results
                )
                
                for r in results:
                    if r.get('body') and r.get('title'):  # Solo risultati validi
                        content = f"{r.get('title', '')}\n{r.get('body', '')}"
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": f"Web: {r.get('link', 'N/A')}",
                                "title": r.get('title', 'N/A'),
                                "type": "web_search"
                            }
                        )
                        web_docs.append(doc)
                        print(f"[DEBUG] Trovato: {r.get('title', 'N/A')}")
                        
            except Exception as e:
                print(f"[WARN] Errore ricerca query originale: {e}")
            
            # Se non trova risultati utili, prova con keywords in inglese
            if len(web_docs) < 2:
                # Query specifiche per embedding models
                specific_queries = [
                    "embedding models Azure OpenAI",
                    "text-embedding-ada-002 models",
                    "OpenAI embeddings list"
                ]
                
                for sq in specific_queries:
                    try:
                        print(f"[DEBUG] Cercando: {sq}")
                        results = ddgs.text(
                            sq,
                            region='wt-wt',  # worldwide per risultati in inglese
                            safesearch='off',
                            max_results=2
                        )
                        
                        for r in results:
                            if r.get('body') and r.get('title'):
                                content = f"{r.get('title', '')}\n{r.get('body', '')}"
                                doc = Document(
                                    page_content=content,
                                    metadata={
                                        "source": f"Web: {r.get('link', 'N/A')}",
                                        "title": r.get('title', 'N/A'),
                                        "type": "web_search"
                                    }
                                )
                                web_docs.append(doc)
                                print(f"[DEBUG] Trovato: {r.get('title', 'N/A')}")
                                
                    except Exception as e:
                        print(f"[WARN] Errore ricerca per '{sq}': {e}")
                        continue
                    
    except Exception as e:
        print(f"[ERROR] Errore generale ricerca web: {e}")
        
    print(f"[DEBUG] Trovati {len(web_docs)} risultati web utili")
    return web_docs

# Funzioni per l'indicizzazione 

# divide i documenti in chunk
def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    """
    Splitting robusto per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""
        ],
    )
    chunks = splitter.split_documents(docs)
    
    print(f"\n[DEBUG] Splitting documenti:")
    print(f"  Documenti originali: {len(docs)}")
    print(f"  Chunks creati: {len(chunks)}")
    print(f"  Chunk size: {settings.chunk_size}")
    print(f"  Overlap: {settings.chunk_overlap}")
    
    if chunks:
        print(f"\n[DEBUG] Esempio primo chunk:")
        print(f"  Source: {chunks[0].metadata.get('source', 'N/A')}")
        print(f"  Contenuto: {chunks[0].page_content[:200]}...")
    
    return chunks

# crea l'indice FAISS
def build_faiss_vectorstore(chunks: List[Document], embeddings: Embeddings, persist_dir: str) -> FAISS:
    """
    Costruisce da zero un FAISS index e lo salva su disco.
    """
    print(f"\n[DEBUG] Costruzione FAISS vectorstore con {len(chunks)} chunks...")
    
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(persist_dir)
    
    print(f"[DEBUG] Vectorstore salvato in: {persist_dir}")
    print(f"[DEBUG] Dimensione indice: {vs.index.ntotal} vettori")
    
    return vs

# carica o ricostruisce l'indice 
def load_or_build_vectorstore(settings: Settings, embeddings: Embeddings, docs: List[Document]) -> FAISS:
    """
    Carica un indice FAISS se esiste e la dimensione coincide.
    Altrimenti ricostruisce l'indice e lo salva.
    """
    persist_path = Path(settings.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"

    cur_dim = _current_embedding_dim(embeddings)

    if index_file.exists() and meta_file.exists():
        try:
            print(f"\n[DEBUG] Tentativo di caricare indice esistente da {settings.persist_dir}")
            vs = FAISS.load_local(
                settings.persist_dir,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            index_dim = vs.index.d
            print(f"[DEBUG] Indice caricato: {vs.index.ntotal} vettori, dimensione {index_dim}")
            
            if index_dim != cur_dim:
                print(f"[INFO] Dimension mismatch (index={index_dim}, current={cur_dim}). Rebuilding index...")
                shutil.rmtree(persist_path, ignore_errors=True)
                persist_path.mkdir(parents=True, exist_ok=True)
                chunks = split_documents(docs, settings)
                return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)
            return vs
        except Exception as e:
            print(f"[WARN] Load indice fallito: {e}. Ricostruzione...")
            shutil.rmtree(persist_path, ignore_errors=True)
            persist_path.mkdir(parents=True, exist_ok=True)
            chunks = split_documents(docs, settings)
            return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)

    # Nessun indice: costruisci da zero
    print(f"\n[DEBUG] Nessun indice trovato. Costruzione nuovo indice...")
    persist_path.mkdir(parents=True, exist_ok=True)
    chunks = split_documents(docs, settings)
    return build_faiss_vectorstore(chunks, embeddings, settings.persist_dir)

# Funzioni per retrieval e generazione

# configura il sistema di ricerca
def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever.
    """
    print(f"\n[DEBUG] Configurazione retriever:")
    print(f"  Search type: {settings.search_type}")
    print(f"  k: {settings.k}")
    if settings.search_type == "mmr":
        print(f"  fetch_k: {settings.fetch_k}")
        print(f"  lambda: {settings.mmr_lambda}")
    
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )

# formatta i documenti per il prompt
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni.
    """
    print(f"\n[DEBUG] Formattazione {len(docs)} documenti per il prompt")
    
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
        print(f"[DEBUG] Doc {i}: {src} - {d.page_content[:100]}...")
    
    formatted = "\n\n".join(lines)
    print(f"[DEBUG] Contesto totale: {len(formatted)} caratteri")
    return formatted

#costruisce la catena rag completa
def build_rag_chain(llm, retriever):
    """
    Catena RAG: retrieval -> prompt -> LLM
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# esegue la query includendo il fallback
def rag_answer_with_fallback(question: str, chain, retriever, settings: Settings, llm) -> str:
    """
    Risponde usando RAG, con fallback su ricerca web se necessario.
    """
    # Prima prova con documenti locali
    answer = chain.invoke(question)
    
    # Se non trova risposta e web search è abilitato
    if "Non è presente nel contesto fornito" in answer and settings.enable_web_search:
        print("\n[INFO] Risposta non trovata nei documenti locali. Ricerca su web...")
        
        # Cerca sul web
        web_docs = search_web_fallback(question, settings.web_search_keywords, settings)
        
        if web_docs:
            # Formatta documenti web
            web_context = format_docs_for_prompt(web_docs)
            
            # Crea nuovo prompt con contesto web
            system_prompt = (
                "Sei un assistente esperto. Rispondi in italiano. "
                "Usa il CONTENUTO fornito nel contesto, che include risultati da ricerca web. "
                "Specifica sempre quando le informazioni provengono dal web. "
                "Includi citazioni tra parentesi quadre nel formato [source:...]. "
                "Sii conciso, accurato e tecnicamente corretto."
            )
            
            prompt_web = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human",
                 "Domanda:\n{question}\n\n"
                 "Contesto (da ricerca web):\n{context}\n\n"
                 "Istruzioni:\n"
                 "1) Rispondi basandoti sulle informazioni del contesto web.\n"
                 "2) Specifica che le informazioni provengono da ricerca web.\n"
                 "3) Cita sempre le fonti nel formato [source:Web: URL].")
            ])
            
            # Crea catena per risposta web usando lo stesso LLM
            web_chain = prompt_web | llm | StrOutputParser()
            
            # Invoca con il contesto web
            answer = web_chain.invoke({
                "question": question,
                "context": web_context
            })
            
    return answer


# funzione per debug
def debug_retriever(retriever, question: str):
    """Funzione di debug per vedere cosa recupera il retriever"""
    print(f"\n[DEBUG] ========== TEST RETRIEVER ==========")
    print(f"[DEBUG] Query: '{question}'")
    
    docs = retriever.get_relevant_documents(question)
    print(f"[DEBUG] Documenti recuperati: {len(docs)}")
    
    for i, doc in enumerate(docs):
        print(f"\n[DEBUG] --- Documento {i+1} ---")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Content: {doc.page_content[:300]}...")
        if hasattr(doc, 'score'):
            print(f"  Score: {doc.score}")
    
    print(f"[DEBUG] ========== FINE TEST ==========\n")
    return docs


# main 
def main():

    settings = SETTINGS

    print(f"\n{'='*60}")
    print(f"RAG con Azure OpenAI e FAISS - DEBUG MODE")
    print(f"{'='*60}")
    print(f"\nConfigurazione:")
    print(f"  Data dir: {settings.data_dir}")
    print(f"  Persist dir: {settings.persist_dir}")
    print(f"  Search type: {settings.search_type}")
    print(f"  k: {settings.k}")
    print(f"  fetch_k: {settings.fetch_k}")
    print(f"  lambda: {settings.mmr_lambda}")

    try:
        # 1) Carica documenti
        docs = load_documents_from_dir(settings.data_dir)

        # 2) Inizializza i componenti AI LLM & Embeddings: legge le variabili d'ambiente, crea il client per l'embeddings e per il chat model
        embeddings = get_embeddings(settings)
        llm = get_llm_from_azure_chat(settings)

        # 3) Configura keywords per ricerca web se necessario
        settings.enable_web_search = True
        settings.web_search_keywords = ["Azure OpenAI", "embedding models", "text-embedding"]
        settings.web_search_max_results = 5
        settings.web_search_region = "wt-wt"  # worldwide per migliori risultati

        # 4) Indicizzazione e Retriever: divide i documenti in chunk, genera embedding per ogni chunk, costruisce l'indice FAISS e salva su disco 
        vector_store = load_or_build_vectorstore(settings, embeddings, docs)
        retriever = make_retriever(vector_store, settings)

        # 5) Costruzione della catena RAG: crea il template del prompt con istruzioni specifiche e costruisce la catena: retriever → formattazione → prompt → LLM → output
        chain = build_rag_chain(llm, retriever)

        # 6) Test immediato
        print("\n[DEBUG] Test di base del sistema...")
        test_query = "test query"
        test_docs = debug_retriever(retriever, test_query)

        # 7) Loop interattivo
        print("\n✅ Indicizzazione pronta.")
        print("Fai una domanda e premi invio.")
        print("Comandi:")
        print("  :reindex - ricostruisce l'indice")
        print("  :debug <query> - mostra solo i documenti recuperati")
        print("  :quit - esci\n")

        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nUscita.")
                break

            if not q:
                continue
            if q.lower() in {":quit", ":exit"}:
                print("Uscita.")
                break
            if q.lower() == ":reindex":
                print("Ricostruzione indice...")
                shutil.rmtree(Path(settings.persist_dir), ignore_errors=True)
                docs = load_documents_from_dir(settings.data_dir)
                vector_store = load_or_build_vectorstore(settings, embeddings, docs)
                retriever = make_retriever(vector_store, settings)
                chain = build_rag_chain(llm, retriever)
                print("Indice ricostruito.")
                continue
            if q.lower().startswith(":debug "):
                debug_query = q[7:]
                debug_retriever(retriever, debug_query)
                continue

            try:
                # Debug: mostra cosa recupera               
                ans = rag_answer_with_fallback(q, chain, retriever, settings, llm)
                print("\n--- Risposta ---")
                print(ans)
                print("---------------\n")
            except Exception as e:
                print(f"[ERRORE] {e}\n")
                import traceback
                traceback.print_exc()

    except Exception as e:
        print(f"\n[ERRORE FATALE] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()