# utils/regas_eval.py
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any

# Import per RAGAS 0.3.4
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness
)

def run_ragas_evaluation(eval_runs: List[Dict], llm, embeddings) -> Dict[str, Any]:
    """
    Valuta le risposte RAG usando RAGAS metrics v0.3.4
    
    Metriche utilizzate:
    - context_precision: primi k chunk sono pertinenti alla domanda (retrieval "pulito")
    - context_recall: copri i pezzi necessari
    - faithfulness: la risposta sta dentro il contesto (meno allucinazioni)
    - answer_relevancy: risposta on-topic
    - answer_correctness: risposta corretta vs ground_truth (se fornita)
    """
    if not eval_runs:
        print("[EVAL] Nessuna interazione da valutare")
        return None
    
    # Prepara i dati nel formato richiesto da RAGAS
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for run in eval_runs:
        data["question"].append(run["question"])
        data["answer"].append(run["answer"])
        
        # contexts deve essere una lista di stringhe
        if isinstance(run["contexts"], list):
            data["contexts"].append(run["contexts"])
        else:
            data["contexts"].append([str(run["contexts"])])
        
        # Ground truth necessaria per context_precision, context_recall e answer_correctness
        # Se non disponibile, usa la risposta come placeholder
        data["ground_truth"].append(run.get("ground_truth", run["answer"]))
    
    # Crea il dataset
    dataset = Dataset.from_dict(data)
    
    print(f"[EVAL] Valutando {len(eval_runs)} interazioni...")
    
    # Controlla se abbiamo ground truth reali
    has_real_ground_truth = any(run.get("ground_truth") and 
                               run.get("ground_truth") != run["answer"] 
                               for run in eval_runs)
    
    if has_real_ground_truth:
        # Usa tutte le metriche se abbiamo ground truth
        metrics_to_use = [
            context_precision,    # Richiede ground_truth
            context_recall,       # Richiede ground_truth
            faithfulness,         # Non richiede ground_truth
            answer_relevancy,     # Non richiede ground_truth
            answer_correctness    # Richiede ground_truth
        ]
        print("[EVAL] Usando tutte le metriche (ground truth disponibile)")
    else:
        # Usa solo metriche che non richiedono ground truth
        metrics_to_use = [
            faithfulness,         # Non richiede ground_truth
            answer_relevancy,     # Non richiede ground_truth
        ]
        print("[EVAL] Usando solo faithfulness e answer_relevancy (no ground truth)")
    
    try:
        # Valuta con RAGAS 0.3.4
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            llm=llm,
            embeddings=embeddings,
        )
        
    except Exception as e:
        print(f"[EVAL] Errore durante la valutazione RAGAS: {e}")
        
        # Se fallisce, prova con le sole metriche base
        if has_real_ground_truth:
            print("[EVAL] Riprovo senza metriche che richiedono ground_truth...")
            metrics_minimal = [faithfulness, answer_relevancy]
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=metrics_minimal,
                    llm=llm,
                    embeddings=embeddings,
                )
            except Exception as e2:
                print(f"[EVAL] Fallito anche con metriche minime: {e2}")
                return None
        else:
            return None
    
    # Estrai il DataFrame dei risultati
    df = result.to_pandas()
    
    # Calcola le medie per ogni metrica
    means = {}
    for col in df.columns:
        if col not in ['question', 'answer', 'contexts', 'ground_truth']:
            try:
                means[col] = df[col].mean()
            except:
                pass
    
    # Salva i risultati
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV con tutti i dettagli
    csv_path = f"ragas_eval_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # JSON con il riepilogo
    json_path = f"ragas_eval_summary_{timestamp}.json"
    summary = {
        "timestamp": timestamp,
        "num_samples": len(eval_runs),
        "has_ground_truth": has_real_ground_truth,
        "metrics_used": [m.name for m in metrics_to_use],
        "means": means,
        "web_search_usage": sum(1 for r in eval_runs if r.get("used_fallback", False)),
        "interpretation": {
            "faithfulness": "% risposte fedeli al contesto (no allucinazioni)",
            "answer_relevancy": "% risposte pertinenti alla domanda",
            "context_precision": "% chunk recuperati sono rilevanti" if "context_precision" in means else "N/A",
            "context_recall": "% informazioni necessarie recuperate" if "context_recall" in means else "N/A", 
            "answer_correctness": "% risposte corrette vs ground truth" if "answer_correctness" in means else "N/A"
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[EVAL] Valutazione completata!")
    print(f"[EVAL] Risultati salvati in: {csv_path} e {json_path}")
    
    # Mostra interpretazione delle metriche
    print("\n[EVAL] Interpretazione risultati:")
    for metric, value in means.items():
        interp = summary["interpretation"].get(metric, "")
        print(f"  - {metric}: {value:.3f} ({interp})")
    
    return {
        "means": means,
        "csv_path": csv_path,
        "json_path": json_path,
        "dataframe": df,
        "summary": summary
    }


def prepare_evaluation_with_ground_truth(question: str, ground_truth: str) -> dict:
    """
    Helper per preparare domande con ground truth per valutazione
    
    Esempio uso:
    eval_data = prepare_evaluation_with_ground_truth(
        "Cos'è Azure OpenAI?",
        "Azure OpenAI è un servizio cloud di Microsoft che fornisce accesso ai modelli GPT"
    )
    """
    return {
        "question": question,
        "ground_truth": ground_truth
    }