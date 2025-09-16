# Funzione di ricerca web

from typing import List
from langchain.schema import Document
from ddgs import DDGS
from .settings import Settings

def search_web_fallback(query: str, keywords: List[str], settings: Settings) -> List[Document]:
    """
    Cerca informazioni su web quando non trovate nel contesto locale.
    """
    print(f"\n[DEBUG] Ricerca web fallback per: '{query}'")
    
    web_docs = []
    
    try:
        # disabilitare ssl 
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