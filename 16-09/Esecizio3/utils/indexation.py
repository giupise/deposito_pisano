

# Funzioni per l'indicizzazione 

from .settings import Settings
from typing import List
from ddgs import DDGS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from typing import List
import shutil

# determina determinare la dimensione dei vettori di embedding generati dal modello corrente
def _current_embedding_dim(embeddings: Embeddings) -> int:
    """
    Calcola la dimensione dell'embedding corrente con una query di prova.
    """
    vec = embeddings.embed_query("dimension probe")
    print(f"[DEBUG] Dimensione embedding: {len(vec)}")
    return len(vec)


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
