from __future__ import annotations
import shutil
from pathlib import Path
from dotenv import load_dotenv
import urllib3

# Disabilita warnings SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Carica variabili d'ambiente
load_dotenv()

# Import delle configurazioni
from utils.settings import Settings

# Import delle funzioni helper
from helper import load_documents_from_dir  # assumendo che sia in helper.py

# Import dei moduli utils
from utils.AIComponent import get_embeddings, get_llm_from_azure_chat
from utils.indexation import load_or_build_vectorstore
from utils.retr_gen import make_retriever, build_rag_chain
from utils.retr_gen import rag_answer_with_fallback  
from utils.debug import debug_retriever

# Import opzionale per RAGAS
try:
    from utils.regas_eval import run_ragas_evaluation
    RAGAS_AVAILABLE = True
    print("[INFO] RAGAS caricato correttamente")
except ImportError as e:
    print(f"[WARN] Errore caricamento RAGAS: {e}")
    RAGAS_AVAILABLE = False
except Exception as e:
    print(f"[ERROR] Errore generico RAGAS: {e}")
    RAGAS_AVAILABLE = False


# Inizializza settings
SETTINGS = Settings()


def main():
    settings = SETTINGS

    print(f"\n{'='*60}")
    print(f"RAG con Azure OpenAI e FAISS - DEBUG MODE")
    print(f"{'='*60}")
    print(f"\nConfigurazione:")
    print(f"  Data dir: {settings.data_dir}")
    print(f"  Persist dir: {settings.persist_dir}")
    print(f"  Search type: {settings.search_type}")
    print(f"  k: {settings.k}")
    print(f"  fetch_k: {settings.fetch_k}")
    print(f"  lambda: {settings.mmr_lambda}")

    try:
        # 1) Carica documenti
        docs = load_documents_from_dir(settings.data_dir)

        # 2) Inizializza i componenti AI
        print("\n[INFO] Inizializzazione componenti AI...")
        embeddings = get_embeddings(settings)
        llm = get_llm_from_azure_chat(settings)

        # 3) Configura ricerca web
        settings.enable_web_search = True
        settings.web_search_keywords = ["Azure OpenAI", "embedding models", "text-embedding"]
        settings.web_search_max_results = 5
        settings.web_search_region = "wt-wt"  # worldwide

        # 4) Indicizzazione e Retriever
        print("\n[INFO] Costruzione indice FAISS...")
        vector_store = load_or_build_vectorstore(settings, embeddings, docs)
        retriever = make_retriever(vector_store, settings)

        # 5) Costruzione catena RAG
        chain = build_rag_chain(llm, retriever)
        
        # Lista per memorizzare le interazioni per la valutazione
        eval_runs = []

        # 6) Test iniziale del sistema
        print("\n[DEBUG] Test di base del sistema...")
        test_query = "test query"
        test_docs = debug_retriever(retriever, test_query)

        # 7) Loop interattivo
        print("\n Indicizzazione pronta.")
        print("\nFai una domanda e premi invio.")
        print("\nComandi disponibili:")
        print("  :reindex         - ricostruisce l'indice")
        print("  :debug <query>   - mostra solo i documenti recuperati")
        print("  :eval            - valuta le interazioni con RAGAS")
        print("  :clear           - pulisce la cronologia valutazioni")
        print("  :quit            - esci\n")

        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Uscita.")
                break

            if not q:
                continue
                
            # Gestione comandi speciali
            if q.lower() in {":quit", ":exit"}:
                print("👋 Uscita.")
                break
                
            if q.lower() == ":clear":
                eval_runs.clear()
                print(" Cronologia valutazioni cancellata.")
                continue
                
            if q.lower() == ":reindex":
                print(" Ricostruzione indice...")
                shutil.rmtree(Path(settings.persist_dir), ignore_errors=True)
                docs = load_documents_from_dir(settings.data_dir)
                vector_store = load_or_build_vectorstore(settings, embeddings, docs)
                retriever = make_retriever(vector_store, settings)
                chain = build_rag_chain(llm, retriever)
                print(" Indice ricostruito.")
                continue
                
            if q.lower().startswith(":debug "):
                debug_query = q[7:]
                debug_retriever(retriever, debug_query)
                continue          
        
            if q.lower() == ":check":
                print(f"\n[DEBUG] Numero di interazioni salvate: {len(eval_runs)}")
                if eval_runs:
                    print("[DEBUG] Ultima interazione:")
                    last = eval_runs[-1]
                    print(f"  Domanda: {last['question']}")
                    print(f"  Risposta: {last['answer'][:100]}...")
                    print(f"  Contesti: {len(last['contexts'])} documenti")
                continue
                
            if q.lower() == ":eval":
                if not RAGAS_AVAILABLE:
                    print("[EVAL] RAGAS non disponibile. Installa con: pip install ragas")
                    continue
                    
                if not eval_runs:
                    print("[EVAL] Nessuna interazione da valutare. Fai prima alcune domande.")
                    continue
                    
                try:
                    print(f"[EVAL] Avvio valutazione RAGAS su {len(eval_runs)} interazioni...")
                    res = run_ragas_evaluation(eval_runs, llm, embeddings)
                    
                    if res:
                        means = res["means"]
                        print("\n=== RAGAS Evaluation Results (macro-avg) ===")
                        for metric, value in means.items():
                            print(f"{metric:25s}: {value:.3f}")
                        print("=" * 45)
                        print(f"\nReport dettagliato salvato in:")
                        print(f"   CSV: {res['csv_path']}")
                        print(f"  📄 JSON: {res['json_path']}")
                    else:
                        print("[EVAL] Valutazione fallita.")
                        
                except Exception as e:
                    print(f"[EVAL][ERRORE] {e}")
                    import traceback
                    traceback.print_exc()
                continue

            # Gestione query normale
            try:
                print("\n[INFO] Elaborazione query...")
                
                # Ottieni risposta con possibile fallback web
                result = rag_answer_with_fallback(q, chain, retriever, settings, llm)
                
                # Estrai componenti dalla risposta
                if isinstance(result, dict):
                    ans = result.get("answer", "")
                    contexts = result.get("contexts", [])
                    used_fallback = result.get("used_fallback", False)
                else:
                    # Compatibilità se rag_answer_with_fallback ritorna solo stringa
                    ans = str(result)
                    # Recupera manualmente i contexts
                    docs_retrieved = retriever.get_relevant_documents(q)
                    contexts = [doc.page_content for doc in docs_retrieved]
                    used_fallback = False

                # Salva per valutazione RAGAS
                eval_runs.append({
                    "question": q,
                    "answer": ans,
                    "contexts": contexts if isinstance(contexts, list) else [contexts],
                    "used_fallback": used_fallback,
                })
                
                # Mostra risposta
                print("\n--- Risposta ---")
                print(ans)
                if used_fallback:
                    print("\n[INFO] Risposta generata usando ricerca web")
                print("---------------\n")
                
            except Exception as e:
                print(f"\n[ERRORE] {e}")
                import traceback
                traceback.print_exc()
                print()

    except Exception as e:
        print(f"\n[ERRORE FATALE] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()