from __future__ import annotations

import os
import warnings

from langchain.chat_models import init_chat_model
from langchain_openai import AzureOpenAIEmbeddings

from utils import Settings

warnings.filterwarnings("ignore", category=UserWarning)

SETTINGS = Settings()


def get_azure_embedding_model(settings: Settings = SETTINGS) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        model=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
        # chunk_size=settings.chunk_size,
        validate_base_url=True,
    )


def get_llm_from_lmstudio():
    """
    Inizializza un ChatModel puntando a LM Studio (OpenAI-compatible).
    Richiede:
      - OPENAI_BASE_URL (es. http://localhost:1234/v1)
      - OPENAI_API_KEY (placeholder qualsiasi, es. "not-needed")
      - LMSTUDIO_MODEL (nome del modello caricato in LM Studio)
    """
    base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
    model_name = os.getenv("AZURE_MODEL")

    if not base_url or not api_key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_API_KEY devono essere impostate per Azure OpenAI"
        )
    if not model_name:
        raise RuntimeError(
            f"Imposta la variabile AZURE_MODEL con il nome del modello caricato in Azure"
        )

    # model_provider="openai" perché l'endpoint è OpenAI-compatible
    return init_chat_model(
        model_name,
        api_key=api_key,
        api_version=api_version,
        model_provider="azure_openai",
    )
