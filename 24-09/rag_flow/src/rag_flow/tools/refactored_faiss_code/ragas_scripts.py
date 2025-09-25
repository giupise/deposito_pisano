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

from rag_structure import get_contexts_for_question
from utils import Settings


def build_ragas_dataset(
    questions: List[str],
    retriever,
    chain,
    k: int,
    ground_truth: dict[str, str] | None = None,
):
    """
    Esegue la pipeline RAG per ogni domanda e costruisce il dataset per Ragas.
    Ogni riga contiene: question, contexts, answer, (opzionale) ground_truth.
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
