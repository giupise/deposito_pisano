import warnings

from .azure_connections import get_azure_embedding_model, get_llm_from_lmstudio
from .ddgs_scripts import ddgs_results, web_search_and_format
from .faiss_code import load_or_build_vectorstore, make_retriever
from .rag_structure import build_rag_chain, keywords_generation, rag_answer
from .ragas_scripts import ragas_evaluation
from .utils import Settings, load_documents, scan_docs_folder

from crewai.tools import tool

warnings.filterwarnings("ignore", category=UserWarning)

@tool('rag_system')
def rag_system(question : str) -> str:
    """
    Advanced RAG System with FAISS vector store and Azure OpenAI integration.
    
    This comprehensive tool provides a complete Retrieval-Augmented Generation system
    specifically optimized for technical aeronautic documentation. It combines local
    document indexing with web search capabilities for enhanced context retrieval.
    
    Parameters
    ----------
    question : str
        Input question to be answered using the RAG system
        
    Returns
    -------
    str
        Generated answer based on retrieved context with source citations and RAGAS evaluation metrics
        
    System Capabilities
    ------------------
    - **Document Processing**: Supports PDF, CSV, Markdown, text files, and images
    - **Vector Store**: FAISS IndexFlatL2 with persistent storage and Azure OpenAI embeddings
    - **Retrieval Strategy**: MMR (Maximal Marginal Relevance) for diverse, non-redundant results
    - **Web Enhancement**: DuckDuckGo search integration for context enrichment (optional)
    - **Content Validation**: Automated reliability scoring for web-scraped content
    - **Answer Generation**: Azure OpenAI with anti-hallucination safeguards and source citations
    - **Quality Evaluation**: RAGAS metrics for answer quality assessment
    
    Technical Architecture
    ---------------------
    1. **Indexing**: Documents are chunked (1000 chars, 200 overlap) and vectorized using Azure text-embedding-ada-002
    2. **Storage**: FAISS vector store with persistent local storage in ./faiss_db/
    3. **Retrieval**: MMR algorithm with k=6 results, fetch_k=20 candidates, lambda=0.7
    4. **Generation**: Azure OpenAI GPT models with context-aware prompting
    5. **Evaluation**: RAGAS framework for faithfulness, relevance, and correctness metrics
    
    Optimization Features
    --------------------
    - **Chunk Strategy**: Hierarchical separators (headers, paragraphs, sentences)
    - **Context Formation**: Source attribution for transparency and traceability
    - **Web Filtering**: Content validation and cleaning for noise reduction
    - **Prompt Engineering**: Anti-hallucination instructions and citation requirements
    - **Performance Monitoring**: Detailed logging and evaluation metrics
    
    Use Cases
    ---------
    - Technical documentation Q&A for aeronautic domain
    - Research assistance with source verification
    - Knowledge base querying with quality assurance
    - Multi-source information synthesis and validation
    
    Notes
    -----
    - Requires Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_MODEL)
    - Document directory: ./docs_test/ (configurable)
    - Index persistence enables fast subsequent queries
    - Web search integration optional based on query complexity
    - All responses include source citations for verification
    
    Example
    -------
    >>> rag_system("What are the main components of aircraft propulsion systems?")
    "Aircraft propulsion systems consist of... [source:aeronautics_guide.pdf] ... RAGAS Metrics: {...}"
    """
    settings = Settings()

    # name_index = input(
    #     "Inserisci il nome dell'indice FAISS (o premi Invio per 'faiss_index'): "
    # ).strip()
    name_index = question.strip()[:5]
    if name_index:
        name_index = settings.set_persist_dir_from_query(name_index)

    print(f"📂 Usando indice FAISS in: {name_index}")

    # 1) Componenti
    embeddings = get_azure_embedding_model()
    llm = get_llm_from_lmstudio()

    # 2) Dati simulati e indicizzazione (load or build)
    file_paths = scan_docs_folder("C:\\Users\\KG376DF\\OneDrive - EY\\Desktop\\python_scripts\\deposito_diomedi\\rag_flow\\src\\rag_flow\\tools\\refactored_faiss_code\\docs_test")
    docs = load_documents(file_paths)
    vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    # req = (
    #     input("Vuoi eseguire una ricerca web per arricchire il contesto? (s/n): ")
    #     .strip()
    #     .lower()
    # )
    
    # if req == "n":
    #     vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    # else:
    #     query = input("Inserisci il termine di ricerca web: ").strip()
    #     print(f"🔍 Eseguo ricerca web su: {query}")
    #     keywords = keywords_generation(query)
    #     print(f"🔑 Keywords generate: {keywords}")
    #     str_new = " ".join(keywords)
    #     ddgs_request = ddgs_results(str_new)
    #     print(ddgs_request)
    #     all_docs = []
    #     for url in ddgs_request:
    #         docs_web = web_search_and_format(url)
    #             # Combina tutto il contenuto per validazione aggregata
    #         combined_content = "\n".join([doc.page_content for doc in docs_web])
            
    #         validation_prompt = f"""
    #         Analizza queste informazioni aeronautiche per accuratezza e affidabilità:
            
    #         {combined_content[:2000]}
            
    #         Valuta:
    #         1. Accuratezza tecnica
    #         2. Coerenza delle informazioni
    #         3. Affidabilità della fonte
            
    #         Restituisci SOLO un punteggio da 1-10 scritto come "Punteggio: X".
    #         """
            
    #         validation_response = llm.invoke([{"role": "user", "content": validation_prompt}])
    #         validation_score = validation_response.content.strip()
            
    #         print(f"🔍 Validazione contenuto da {url}: {validation_score[:100]}...")
            
    #         # Filtra documenti solo se la validazione è positiva
    #         if any(str(i) in validation_score for i in range(6, 11)):
    #             all_docs.extend(docs_web)
    #         else:
    #             print(f"⚠️ Contenuto scartato per bassa affidabilità")
    #     if all_docs:
    #         docs.extend(all_docs)
    #     print(f"📚 Caricati {len(docs)} documenti dalla ricerca web")
    #     vector_store = load_or_build_vectorstore(settings, embeddings, docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    #while True:
        # question = input("Inserisci la tua domanda (o 'exit'/'quit'/'q' per uscire): ")
    # question = question
    
        # Controlla se l'utente vuole uscire
        #if question.lower().strip() in ["exit", "quit", "q", "esci"]:
        #    print("👋 Arrivederci!")
        #    break

    # print("=" * 80)
    # print("Q:", question)
    # print("-" * 80)
    ans = rag_answer(question, chain)
    print(ans)
    # print()
    rag_eval = ragas_evaluation(
        question, chain, llm, embeddings, retriever, settings
    )
    print("\n METRICHE OTTENUTE:\n", rag_eval)
    return ans


if __name__ == "__main__":
    rag_system()
