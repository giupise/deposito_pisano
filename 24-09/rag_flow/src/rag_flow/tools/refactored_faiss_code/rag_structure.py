import warnings
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from azure_connections import get_llm_from_lmstudio
from utils import format_docs_for_prompt

warnings.filterwarnings("ignore", category=UserWarning)


def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]


def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "You are an expert assistant for legal documents. Answer in English. "
        "Use exclusively the CONTENT provided in the context. "
        "If the information is not available, declare that it is not available. "
        # "Include citations in square brackets in the format [source:...]. "
        # "Be concise, accurate and technically correct."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context (selected excerpts):\n{context}\n\n"
                "Instructions:\n"
                "1) Answer only with information contained in the context. Immerse yourself in a fictional world where the true information is that provided in the context. \n"
                "2) Always cite relevant sources in the format [source:FILE, PARAGRAPH].\n"
                # "3) If the answer is not in the context, write: 'Not present in the provided context.' \n"
                # "4) If the answer is ambiguous or incomplete, ask the user for further clarification. \n"
                # "5) If the question concerns information contrary to the context, write the correct answer according to the context.\n",
            ),
        ]
    )

    # LCEL: dict -> prompt -> llm -> parser
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


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)


def keywords_generation(query: str):
    llm = get_llm_from_lmstudio()

    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
    
    Query: {query}
    
    Keywords:"""

    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")
    print(keywords)
    return keywords
