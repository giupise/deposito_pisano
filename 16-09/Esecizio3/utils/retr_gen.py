# Funzioni per retrieval e generazione


from __future__ import annotations
from typing import List
from typing import Tuple


# LangChain core / schema
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Vector store + text splitter + loaders
from langchain_community.vectorstores import FAISS


# import funzioni helper
from helper import *
from .settings import Settings
from .searchFunction import search_web_fallback

def rag_answer_and_capture(
    question: str,
    chain,
    retriever,
) -> Tuple[str, list]:
    """
    Esegue retrieval esplicito per catturare i contesti (testo puro) e poi invoca la chain.
    Restituisce (answer, contexts_texts).
    """
    docs = retriever.get_relevant_documents(question)
    contexts = [d.page_content for d in docs]
    answer = chain.invoke(question)
    return answer, contexts


# configura il sistema di ricerca
def make_retriever(vector_store: FAISS, settings: Settings):
    """
    Configura il retriever.
    """
    print(f"\n[DEBUG] Configurazione retriever:")
    print(f"  Search type: {settings.search_type}")
    print(f"  k: {settings.k}")
    if settings.search_type == "mmr":
        print(f"  fetch_k: {settings.fetch_k}")
        print(f"  lambda: {settings.mmr_lambda}")
    
    if settings.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": settings.k, "fetch_k": settings.fetch_k, "lambda_mult": settings.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.k},
        )

# formatta i documenti per il prompt
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni.
    """
    print(f"\n[DEBUG] Formattazione {len(docs)} documenti per il prompt")
    
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
        print(f"[DEBUG] Doc {i}: {src} - {d.page_content[:100]}...")
    
    formatted = "\n\n".join(lines)
    print(f"[DEBUG] Contesto totale: {len(formatted)} caratteri")
    return formatted

#costruisce la catena rag completa
def build_rag_chain(llm, retriever):
    """
    Catena RAG: retrieval -> prompt -> LLM
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])

    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# esegue la query includendo il fallback
def rag_answer_with_fallback(question: str, chain, retriever, settings: Settings, llm):
    """
    Risponde usando RAG, con fallback su ricerca web se necessario.
    Restituisce un dizionario: {"answer": str, "contexts": List[str], "used_fallback": bool}
    """
    # Primo tentativo sui documenti locali (catturo i contesti)
    local_docs = retriever.get_relevant_documents(question)
    local_contexts = [d.page_content for d in local_docs]
    answer = chain.invoke(question)

    if "Non è presente nel contesto fornito" in answer and settings.enable_web_search:
        print("\n[INFO] Risposta non trovata nei documenti locali. Ricerca su web...")

        web_docs = search_web_fallback(question, settings.web_search_keywords, settings)
        if web_docs:
            web_context = format_docs_for_prompt(web_docs)

            system_prompt = (
                "Sei un assistente esperto. Rispondi in italiano. "
                "Usa il CONTENUTO fornito nel contesto, che include risultati da ricerca web. "
                "Specifica sempre quando le informazioni provengono dal web. "
                "Includi citazioni tra parentesi quadre nel formato [source:...]. "
                "Sii conciso, accurato e tecnicamente corretto."
            )

            prompt_web = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human",
                 "Domanda:\n{question}\n\n"
                 "Contesto (da ricerca web):\n{context}\n\n"
                 "Istruzioni:\n"
                 "1) Rispondi basandoti sulle informazioni del contesto web.\n"
                 "2) Specifica che le informazioni provengono da ricerca web.\n"
                 "3) Cita sempre le fonti nel formato [source:Web: URL].")
            ])

            web_chain = prompt_web | llm | StrOutputParser()
            answer = web_chain.invoke({
                "question": question,
                "context": web_context
            })

            # CONTEXTS per RAGAS quando si usa web: usa direttamente i testi dei doc web
            web_contexts = [d.page_content for d in web_docs]
            return {"answer": answer, "contexts": web_contexts, "used_fallback": True}

    # caso standard: locali
    return {"answer": answer, "contexts": local_contexts, "used_fallback": False}

