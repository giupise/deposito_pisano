from typing import List

from ragas import EvaluationDataset, evaluate
from ragas.metrics import \
    answer_correctness  # usa questa solo se hai ground_truth
from ragas.metrics import \
    answer_relevancy, AnswerRelevancy  # pertinenza della risposta vs domanda
from ragas.metrics import \
    context_precision  # "precision@k" sui chunk recuperati
from ragas.metrics import context_recall  # copertura dei chunk rilevanti
from ragas.metrics import faithfulness  # ancoraggio della risposta al contesto

from .rag_structure import get_contexts_for_question
from .utils import Settings

def format_contexts_for_chain(contexts: List[str]) -> str:
    """Formatta una lista di contexts per la chain."""
    return "\n\n".join([f"[source:unknown] {context}" for context in contexts])

def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Build RAGAS evaluation dataset from RAG pipeline execution.
    
    Executes the complete RAG pipeline for each question to generate the
    evaluation dataset required by RAGAS framework. Each dataset entry contains
    question, retrieved contexts, generated answer, and optional ground truth.
    
    Parameters
    ----------
    questions : List[str]
        List of questions to evaluate through the RAG pipeline
    retriever : VectorStoreRetriever
        Configured retriever for context extraction
    chain : RunnableSequence
        RAG chain for answer generation
    k : int
        Number of context chunks to retrieve per question
    ground_truth : dict[str, str], optional
        Dictionary mapping questions to their ground truth answers
        
    Returns
    -------
    List[dict]
        List of evaluation entries, each containing:
        - question: Input question
        - contexts: Retrieved context chunks
        - answer: Generated RAG answer
        - ground_truth: Reference answer (if provided)
        
    Dataset Structure
    -----------------
    Each entry follows RAGAS expected format::
    
        {
            'question': str,
            'contexts': List[str], 
            'answer': str,
            'ground_truth': str (optional)
        }
    
    Notes
    -----
    - Ground truth is optional but enables answer_correctness evaluation
    - Context extraction uses the configured retrieval strategy
    - Answer generation follows the complete RAG chain
    - Dataset format is compatible with RAGAS EvaluationDataset
    """
    dataset = []
    for q in questions:
        contexts = get_contexts_for_question(retriever, q, k)
        # answer = chain.invoke(q) SCOMMENTA PER FAISS
        ctx = format_contexts_for_chain(contexts)  # Formatta per la chain
        answer = chain.invoke({"question": q, "context": ctx})

        row = {
            # chiavi richieste da molte metriche Ragas
            "user_input": q,
            "retrieved_contexts": contexts,
            "response": answer,
        }
        if ground_truth and q in ground_truth:
            row["reference"] = ground_truth[q]

        dataset.append(row)
    return dataset


def ragas_evaluation(
    question: str, chain, llm, embeddings, retriever, settings: Settings
):
    questions = [question]
    try:
        dataset = build_ragas_dataset(
            questions=questions, retriever=retriever, chain=chain, k=settings.k
        )
    except Exception as e:
        dataset = build_ragas_dataset(
            questions=questions, retriever=retriever, chain=chain, k=settings.final_k
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    ar = AnswerRelevancy(strictness=1)
    # 7) Scegli le metriche
    metrics = [
        #     context_precision,
        #    context_recall,
        faithfulness,
        ar,
    ]
    # Aggiungi correctness solo se tutte le righe hanno ground_truth
    if all("ground_truth" in row for row in dataset):
        metrics.append(answer_correctness)

    # 8) Esegui la valutazione con il TUO LLM e le TUE embeddings
    ragas_result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=llm,  # passa l'istanza LangChain del tuo LLM (LM Studio)
        embeddings=embeddings,  # o riusa 'embeddings' creato sopra
    )

    df = ragas_result.to_pandas()
    cols = ["user_input", "response", "faithfulness", "answer_relevancy"]
    
    # Formatta le metriche per una visualizzazione più chiara
    print("\n" + "="*60)
    print(" METRICHE RAGAS - VALUTAZIONE QUALITÀ RAG")
    print("="*60)
    
    for idx, row in df.iterrows():
        print(f"\n DOMANDA: {row['user_input'][:50]}...")
        print(f" RISPOSTA: {row['response'][:100]}...")
        print(f" FAITHFULNESS: {row['faithfulness']:.2f} ({' Eccellente' if row['faithfulness'] > 0.8 else '⚠️ Da migliorare' if row['faithfulness'] > 0.5 else '❌ Insufficiente'})")
        print(f" ANSWER RELEVANCY: {row['answer_relevancy']:.2f} ({' Eccellente' if row['answer_relevancy'] > 0.8 else '⚠️ Da migliorare' if row['answer_relevancy'] > 0.5 else '❌ Insufficiente'})")
        print("-" * 60)
    
    print(f"\n RIEPILOGO METRICHE:")
    print(f"   • Faithfulness Media: {df['faithfulness'].mean():.2f}")
    print(f"   • Answer Relevancy Media: {df['answer_relevancy'].mean():.2f}")
    print(f"   • Qualità Generale: {'Eccellente' if df['faithfulness'].mean() > 0.8 and df['answer_relevancy'].mean() > 0.8 else '🟡 Buona' if df['faithfulness'].mean() > 0.6 and df['answer_relevancy'].mean() > 0.6 else ' Da migliorare'}")
    print("="*60)
    
    return df[cols].round(2)