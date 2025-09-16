
from langchain_core.embeddings import Embeddings
import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from .settings import Settings

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