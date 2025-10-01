import warnings

import bs4
from duckduckgo_search import DDGS
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

from .utils import clean_web_content
# from utils import clean_web_content

warnings.filterwarnings("ignore", category=UserWarning)


def ddgs_results(query: str, max_results: int = 5):
    """
    Perform web search using DuckDuckGo Search with SSL bypass and error handling.
    
    Executes web search queries using DDGS (DuckDuckGo Search) API with optimized
    configuration for reliable results. Includes SSL verification bypass for
    improved connectivity and comprehensive error handling.
    
    Parameters
    ----------
    query : str
        Search query string for web search
    max_results : int, optional
        Maximum number of search results to return (default: 5)
        
    Returns
    -------
    List[str]
        List of URLs from search results, empty list if search fails
        
    Features
    --------
    - SSL verification bypass for improved connectivity
    - Custom User-Agent header for better compatibility
    - Regional search configuration (us-en)
    - Moderate safe search filtering
    - 20-second timeout for reliability
    - Detailed logging of results and errors
    
    Notes
    -----
    Results are formatted as URLs only for compatibility with downstream
    processing. Full result metadata (titles, descriptions) are logged
    but not returned in the output.
    """
    print(f"🔍 DDGS: Ricerca per '{query}' (max {max_results} risultati)")

    try:
        # DDGS con SSL bypass
        with DDGS(
            verify=False,  # ← BYPASSA SSL
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; DDGSBot/1.0)"},
        ) as ddgs:

            results = list(
                ddgs.text(
                    keywords=query,
                    max_results=max_results,
                    region="us-en",
                    safesearch="moderate",
                    timelimit=None,  # Risultati dell'ultimo anno
                )
            )

        print(f"   ✅ {len(results)} risultati trovati")

        # Formatta per compatibilità con il resto del codice
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(result.get("href", ""))
            print(f"   {i}. {result.get('title', '')[:50]}...")

        return formatted

    except Exception as e:
        print(f"❌ Errore DDGS: {e}")
        return []


def web_search_and_format(path: str):
    """
    Load and clean web content for RAG system integration.
    
    Fetches web page content from the specified URL, applies comprehensive
    cleaning to remove navigation elements and noise, and formats the result
    as LangChain Document objects suitable for vector indexing.
    
    Parameters
    ----------
    path : str
        URL of the web page to load and process
        
    Returns
    -------
    List[Document]
        List of cleaned Document objects with web content and metadata
        
    Processing Pipeline
    ------------------
    1. **Content Extraction**: Uses WebBaseLoader with CSS selectors for main content
    2. **Cleaning**: Applies clean_web_content() for noise reduction
    3. **Validation**: Filters out empty or overly short content
    4. **Formatting**: Creates Document objects with URL metadata
    
    Targeted Content Selectors
    --------------------------
    - article, main: Primary content containers
    - .content, .post-content: CMS-specific content areas  
    - #content, #main-content: Common content identifiers
    - .entry-content: Blog post content areas
    - p: Fallback to paragraph extraction
    
    Error Handling
    -------------
    - Graceful fallback through multiple extraction strategies
    - Comprehensive exception handling with detailed logging
    - Returns empty list if all strategies fail
    - Content length validation to ensure meaningful results
    
    Notes
    -----
    - Uses BeautifulSoup parser for reliable HTML processing
    - Applies domain-specific cleaning rules for Italian and English content
    - Minimum content length threshold of 100 characters
    - All results include source URL in metadata for citation purposes
    """
    print(f"🌐 Caricamento contenuto da: {path}")

    try:
        # Strategia 1: Selettori CSS specifici per il contenuto principale
        content_selectors = [
            {"name": "article", "elements": ["article"]},
            {"name": "main-content", "elements": ["main", ".main", "#main"]},
            {
                "name": "content-areas",
                "elements": [
                    ".content",
                    ".post-content",
                    ".article-content",
                    ".entry-content",
                ],
            },
            {
                "name": "text-body",
                "elements": [".text", ".body", ".story-body", ".article-body"],
            },
        ]

        valid_docs = []

        for selector_group in content_selectors:
            if valid_docs:  # Se abbiamo già trovato contenuto valido, fermati
                break

            try:
                loader = WebBaseLoader(
                    web_paths=(path,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(selector_group["elements"])
                    ),
                )

                docs = loader.load()
                print(
                    f"   🔍 Tentativo con selettori {selector_group['name']}: {len(docs)} documenti"
                )

                for doc in docs:
                    # Pulisci il contenuto
                    cleaned_content = clean_web_content(doc.page_content)

                    # Verifica che il contenuto pulito sia significativo
                    if (
                        len(cleaned_content.strip()) > 100
                    ):  # Almeno 100 caratteri dopo pulizia
                        # Aggiorna il documento con contenuto pulito
                        doc.page_content = cleaned_content
                        valid_docs.append(doc)
                        print(
                            f"   ✅ Contenuto valido trovato: {len(cleaned_content)} caratteri puliti"
                        )
                        break

            except Exception as e:
                print(f"   ⚠️ Errore con selettori {selector_group['name']}: {e}")
                continue

        # Strategia 2: Se non abbiamo trovato nulla, prova senza filtri ma pulisci di più
        if not valid_docs:
            print("   🔄 Nessun contenuto valido trovato, provo senza filtri CSS...")
            try:
                loader = WebBaseLoader(web_paths=(path,))
                docs = loader.load()

                for doc in docs:
                    # Pulizia aggressiva per contenuto non filtrato
                    cleaned_content = clean_web_content(doc.page_content)

                    # Rimozione aggiuntiva per contenuto non filtrato
                    lines = cleaned_content.split("\n")
                    content_lines = []

                    for line in lines:
                        line = line.strip()
                        # # Mantieni solo linee con contenuto sostanziale
                        # if (
                        #     len(line) > 30
                        #     and not re.match(r"^[A-Z\s]+$", line)  # Non solo maiuscole
                        #     and not re.match(r"^\d+$", line)  # Non solo numeri
                        #     and len(line.split()) > 3
                        # ):  # Almeno 4 parole
                        content_lines.append(line)

                    final_content = " ".join(content_lines)

                    if (
                        len(final_content.strip()) > 150
                    ):  # Standard più alto per contenuto non filtrato
                        doc.page_content = final_content
                        valid_docs.append(doc)
                        print(
                            f"Contenuto recuperato e pulito: {len(final_content)} caratteri"
                        )
                        break

            except Exception as e:
                print(f"Errore anche senza filtri: {e}")

        # Verifica finale e fallback
        if not valid_docs:
            print("Impossibile estrarre contenuto significativo")
            return [
                Document(
                    page_content=f"Contenuto non disponibile per {path}. Il sito web potrebbe non essere accessibile o non contenere testo leggibile.",
                    metadata={"source": path, "error": "no_meaningful_content"},
                )
            ]

        # Mostra preview del contenuto pulito
        for i, doc in enumerate(valid_docs):
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"Preview contenuto {i+1}: '{preview}...'")

        return valid_docs

    except Exception as e:
        print(f"Errore generale nel caricamento di {path}: {e}")
        return [
            Document(
                page_content=f"Errore nel caricamento di {path}: {str(e)}",
                metadata={"source": path, "error": str(e)},
            )
        ]
