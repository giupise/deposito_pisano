# setup

from dataclasses import dataclass
from typing import List

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
