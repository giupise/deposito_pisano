import warnings
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .azure_connections import get_llm_from_lmstudio
from .utils import format_docs_for_prompt

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


def build_rag_chain(llm, retriever):
    """
    Build a complete RAG chain with retrieval, prompting, and generation components.
    
    Constructs a LangChain Expression Language (LCEL) chain that combines
    document retrieval, context formatting, prompt construction, and LLM generation
    with anti-hallucination safeguards and source citation requirements.
    
    Parameters
    ----------
    llm : ChatModel
        Language model instance for text generation
    retriever : VectorStoreRetriever
        Configured retriever for document retrieval
        
    Returns
    -------
    RunnableSequence
        Complete RAG chain ready for question answering
        
    Chain Components
    ---------------
    1. Context retrieval using the configured retriever
    2. Document formatting with source citations
    3. Prompt construction with system and human messages
    4. LLM generation with strict adherence to provided context
    5. String output parsing for clean response format
    
    Anti-Hallucination Features
    --------------------------
    - System prompt enforces context-only responses
    - Requires explicit source citations in [source:FILE, PARAGRAPH] format
    - Instructs model to declare unavailable information rather than guess
    - Emphasizes technical accuracy and precision
    
    Notes
    -----
    The chain uses LCEL syntax for composable and efficient execution.
    Context formatting includes source attributions for transparency.
    """
    system_prompt = (
        "Sei un assistente AI esperto e preciso. "
        "Usa esclusivamente i CONTENUTI forniti nel contesto. "
        "Se le informazioni non sono disponibili, dichiara che non sono disponibili. "
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
                "Istruzioni:\n"
                "1) Rispondi solo con informazioni contenute nel contesto. \n"
                "2) Cita sempre le fonti rilevanti nel formato [source:FILE, PARAGRAPH].\n"
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
    llm = get_llm_from_lmstudio()

    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
    
    Query: {query}
    
    Keywords:"""

    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")
    print(keywords)
    return keywords
