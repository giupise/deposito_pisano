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
    # Persistenza FAISS
    persist_dir: str = "./faiss_db/default"
    # Text splitting
    chunk_size: int = 800
    chunk_overlap: int = 350
    # Retriever (MMR)
    search_type: str = "mmr"  # "mmr" o "similarity"
    k: int = 4  # risultati finali
    fetch_k: int = 20  # candidati iniziali (per MMR)
    mmr_lambda: float = 0.6  # 0 = diversificazione massima, 1 = pertinenza massima
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
        Imposta persist_dir dinamicamente basandosi sulla query.
        """
        # Pulisci la query per renderla un nome di cartella valido
        clean_query = self._sanitize_query_for_filename(query)
        self.persist_dir = f"./faiss_db/{clean_query}"

    def _sanitize_query_for_filename(self, query: str) -> str:
        """
        Converte una query in un nome di file/cartella valido.
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
    Carica documenti da file PDF, CSV e immagini, restituendo una lista di Document.
    """
    print(f"🔍 LOAD_DOCUMENTS: Caricamento di {len(file_paths)} file(s)")
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
        print(f"   ✅ Caricati {len(docs)} documento/i da {file_path}")
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
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ": ",
            ", ",
            " ",
            "",  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)


def scan_docs_folder(docs_dir: str = "docs") -> List[str]:
    """
    Scansiona la cartella docs e restituisce tutti i file supportati.
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

    print(f"📂 Trovati {len(file_paths)} file nella cartella {docs_dir}")
    return file_paths


def clean_web_content(text: str) -> str:
    """
    Pulisce il contenuto web da elementi indesiderati.
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
