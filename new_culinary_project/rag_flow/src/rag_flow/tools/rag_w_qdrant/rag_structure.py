import warnings
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .azure_connections import get_llm


warnings.filterwarnings("ignore", category=UserWarning)


def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """
    Retrieve top-k document chunks as context for a given question.
    
    Uses the configured retriever to find the most relevant document chunks
    for the input question, returning only the text content for context formation.
    
    Parameters
    ----------
    retriever : VectorStoreRetriever
        Configured retriever from FAISS vector store
    question : str
        Input question to retrieve relevant context for
    k : int
        Maximum number of document chunks to retrieve
        
    Returns
    -------
    List[str]
        List of text content from the top-k most relevant document chunks
        
    Notes
    -----
    The retriever uses the configured search strategy (MMR or similarity)
    and returns document content without metadata for prompt construction.
    """
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]




def build_rag_chain(llm):
    system_prompt = (
        "Sei un assistente tecnico. Rispondi in italiano, conciso e accurato. "
        "Usa ESCLUSIVAMENTE le informazioni presenti nel CONTENUTO. "
        "Se non è presente, dichiara: 'Non è presente nel contesto fornito.' "
        "Cita sempre le fonti nel formato [source:FILE]."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "CONTENUTO:\n{context}\n\n"
         "Istruzioni:\n"
         "1) Risposta basata solo sul contenuto.\n"
         "2) Includi citazioni [source:...].\n"
         "3) Niente invenzioni.")
    ])

    chain = (
        {
            "context": RunnablePassthrough(),  # stringa già formattata
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Execute RAG chain to generate answer for a single question.
    
    Invokes the complete RAG chain with the provided question, handling
    document retrieval, context formatting, and response generation in a
    single streamlined operation.
    
    Parameters
    ----------
    question : str
        Input question to be answered using the RAG system
    chain : RunnableSequence
        Configured RAG chain from build_rag_chain()
        
    Returns
    -------
    str
        Generated answer based on retrieved context with source citations
        
    Notes
    -----
    The chain automatically handles:
    - Document retrieval based on question similarity
    - Context formatting with source attributions
    - Prompt construction and LLM invocation
    - Response parsing and formatting
    """
    return chain.invoke(question)


def keywords_generation(query: str):
    """
    Generate web search keywords from a user query using LLM.
    
    Uses the configured language model to extract and expand relevant
    keywords from the input query for enhanced web search capabilities.
    Useful for enriching RAG context with web-sourced information.
    
    Parameters
    ----------
    query : str
        Input query to extract keywords from
        
    Returns
    -------
    List[str]
        List of generated keywords suitable for web search
        
    Notes
    -----
    - Uses Azure OpenAI model for keyword generation
    - Keywords are comma-separated in the LLM response
    - Results are split and returned as a list
    - Intended for use with web search tools like DuckDuckGo
    """
    llm = get_llm()

    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
    
    Query: {query}
    
    Keywords:"""

    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")
    print(keywords)
    return keywords
