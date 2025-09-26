from typing import List

from ragas import EvaluationDataset, evaluate
from ragas.metrics import \
    answer_correctness  # usa questa solo se hai ground_truth
from ragas.metrics import \
    answer_relevancy  # pertinenza della risposta vs domanda
from ragas.metrics import \
    context_precision  # "precision@k" sui chunk recuperati
from ragas.metrics import context_recall  # copertura dei chunk rilevanti
from ragas.metrics import faithfulness  # ancoraggio della risposta al contesto

from .rag_structure import get_contexts_for_question
from .utils import Settings


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
        answer = chain.invoke(q)

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
    dataset = build_ragas_dataset(
        questions=questions, retriever=retriever, chain=chain, k=settings.k
    )

    evaluation_dataset = EvaluationDataset.from_list(dataset)

    # 7) Scegli le metriche
    metrics = [
        #     context_precision,
        #    context_recall,
        faithfulness,
        answer_relevancy,
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
    cols = ["faithfulness", "answer_relevancy"]
    return df[cols].round().to_string(index=False)
