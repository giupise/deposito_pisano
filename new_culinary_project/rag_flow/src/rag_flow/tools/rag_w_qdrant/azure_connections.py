from __future__ import annotations

import os
import warnings

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings

from .utils import Settings

warnings.filterwarnings("ignore", category=UserWarning)

SETTINGS = Settings()


def get_azure_embedding_model(settings: Settings = SETTINGS) -> AzureOpenAIEmbeddings:
    """
    Initialize Azure OpenAI embedding model for document vectorization.
    
    Creates and configures an AzureOpenAIEmbeddings instance using environment
    variables for Azure OpenAI service connection. This model is used for
    converting text documents into high-dimensional vector representations.
    
    Parameters
    ----------
    settings : Settings, optional
        Configuration settings object (default: SETTINGS global instance)
        
    Returns
    -------
    AzureOpenAIEmbeddings
        Configured Azure OpenAI embedding model ready for text vectorization
        
    Environment Variables
    --------------------
    AZURE_EMBEDDING_MODEL : str
        Azure deployment name for embedding model (default: "text-embedding-ada-002")
    AZURE_OPENAI_API_KEY : str
        API key for Azure OpenAI service authentication
    AZURE_OPENAI_ENDPOINT : str
        Azure OpenAI service endpoint URL
    AZURE_API_VERSION : str
        API version for Azure OpenAI service (default: "2023-05-15")
    """
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        model=os.getenv("AZURE_EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
        # chunk_size=settings.chunk_size,
        validate_base_url=True,
    )

def get_llm():
    """
    Initialize a ChatModel pointing to Azure OpenAI service.
    
    Creates a chat model instance configured for Azure OpenAI service using
    environment variables. This model is used for text generation and
    conversational AI capabilities within the RAG system.
    
    Returns
    -------
    ChatModel
        Configured Azure OpenAI chat model instance
        
    Raises
    ------
    RuntimeError
        If required environment variables are missing (AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_KEY, or AZURE_MODEL)
        
    Environment Variables
    ---------------------
    AZURE_OPENAI_ENDPOINT : str
        Azure OpenAI service endpoint URL (required)
    AZURE_OPENAI_API_KEY : str
        API key for Azure OpenAI service authentication (required)
    AZURE_API_VERSION : str
        API version for Azure OpenAI service (default: "2024-12-01-preview")
    AZURE_MODEL : str
        Name of the Azure OpenAI model to use (required)
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


    return AzureChatOpenAI(
        openai_api_version=api_version,
        azure_deployment=model_name,
        azure_endpoint=base_url,
        openai_api_key=api_key,
        validate_base_url=False,
        openai_api_type="azure",)
