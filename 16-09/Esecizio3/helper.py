# Funzioni helper 
from langchain.schema import Document
from typing import List
from pathlib import Path
from langchain_core.embeddings import Embeddings

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


