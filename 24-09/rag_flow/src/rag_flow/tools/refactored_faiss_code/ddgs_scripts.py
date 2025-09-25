import re
import warnings

import bs4
from duckduckgo_search import DDGS
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader

from utils import clean_web_content

warnings.filterwarnings("ignore", category=UserWarning)


def ddgs_results(query: str, max_results: int = 5):
    """
    Ricerca web con DDGS diretto, bypassa SSL e gestisce errori.
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
                    region="it-it",
                    safesearch="moderate",
                    timelimit="y",  # Risultati dell'ultimo anno
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
    Esegue una ricerca web, carica il contenuto e lo pulisce per un migliore processing.
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
                            f"   ✅ Contenuto recuperato e pulito: {len(final_content)} caratteri"
                        )
                        break

            except Exception as e:
                print(f"   ❌ Errore anche senza filtri: {e}")

        # Verifica finale e fallback
        if not valid_docs:
            print("   ❌ Impossibile estrarre contenuto significativo")
            return [
                Document(
                    page_content=f"Contenuto non disponibile per {path}. Il sito web potrebbe non essere accessibile o non contenere testo leggibile.",
                    metadata={"source": path, "error": "no_meaningful_content"},
                )
            ]

        # Mostra preview del contenuto pulito
        for i, doc in enumerate(valid_docs):
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"   📄 Preview contenuto {i+1}: '{preview}...'")

        return valid_docs

    except Exception as e:
        print(f"❌ Errore generale nel caricamento di {path}: {e}")
        return [
            Document(
                page_content=f"Errore nel caricamento di {path}: {str(e)}",
                metadata={"source": path, "error": str(e)},
            )
        ]
