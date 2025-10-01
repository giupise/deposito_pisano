import streamlit as st
import warnings
import shutil
from pathlib import Path

# Import moduli del RAG system
from .azure_connections import get_azure_embedding_model, get_llm
from .ddgs_scripts import ddgs_results, web_search_and_format
from .faiss_code import load_or_build_vectorstore, make_retriever
from .rag_structure import build_rag_chain, keywords_generation, rag_answer
from .ragas_scripts import ragas_evaluation
from .utils import Settings, load_documents, scan_docs_folder
# from azure_connections import get_azure_embedding_model, get_llm
# from ddgs_scripts import ddgs_results, web_search_and_format
# from faiss_code import load_or_build_vectorstore, make_retriever
# from rag_structure import build_rag_chain, keywords_generation, rag_answer
# from ragas_scripts import ragas_evaluation
# from utils import Settings, load_documents, scan_docs_folder

warnings.filterwarnings("ignore", category=UserWarning)

def initialize_models():
    """Initialize Azure models using .env configuration"""
    try:
        embeddings = get_azure_embedding_model()
        llm = get_llm()
        settings = Settings()
        return embeddings, llm, settings
    except Exception as e:
        st.error(f"❌ Error initializing models: {e}")
        st.info("💡 Make sure your .env file contains: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_MODEL, AZURE_EMBEDDING_MODEL")
        return None, None, None


def main_menu():
    """Main menu to choose between local data and web data"""
    st.title("🚀 RAG Assistant")
    st.markdown("Choose your preferred mode:")
    
    # Initialize models from .env
    if "models_initialized" not in st.session_state:
        with st.spinner("Initializing Azure models from .env..."):
            embeddings, llm, settings = initialize_models()
            if embeddings and llm and settings:
                st.session_state.embeddings = embeddings
                st.session_state.llm = llm
                st.session_state.settings = settings
                st.session_state.models_initialized = True
                st.success("✅ Models initialized successfully!")
            else:
                st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Local Documents")
        st.markdown("Use your local documents as context for chat")
        if st.button("📄 Start Local Chat", use_container_width=True):
            st.session_state.app_mode = "local"
            st.rerun()
    
    with col2:
        st.markdown("### 🌐 Web Search")
        st.markdown("Use web content as context for chat")
        if st.button("🔍 Start Web Search", use_container_width=True):
            st.session_state.app_mode = "web"
            st.rerun()
    
    # Show current mode if one is selected
    if "app_mode" in st.session_state:
        st.info(f"Current mode: **{st.session_state.app_mode.title()}**")
        if st.button("🔄 Change Mode"):
            for key in ['app_mode', 'vectorstore', 'retriever', 'rag_chain', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


def local_documents_page():
    """Page for local documents RAG"""
    st.title("📁 Local Documents RAG")
    
    # Show current configuration in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Using .env configuration")
        
        if st.button("🏠 Back to Menu"):
            for key in ['app_mode', 'vectorstore', 'retriever', 'rag_chain', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Document Information")
        if "vectorstore" in st.session_state:
            st.success("✅ Vector store ready")
        
        # Index naming
        st.markdown("### 🏷️ Index Configuration")
        index_name = st.text_input("FAISS Index Name", value="default", help="Nome per l'indice FAISS")
        if st.button("🔄 Update Index Name"):
            if index_name:
                st.session_state.settings.set_persist_dir_from_query(index_name)
                st.info(f"Index path: {st.session_state.settings.persist_dir}")
                # Reset vectorstore per ricreare con nuovo path
                if "vectorstore" in st.session_state:
                    del st.session_state["vectorstore"]
                    del st.session_state["retriever"]
                    del st.session_state["rag_chain"]
    
    # Document upload and processing
    if "vectorstore" not in st.session_state:
        st.subheader("📄 Upload Documents")
        st.markdown("Upload your documents to create the knowledge base:")
        
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['pdf', 'csv', 'md', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'],
            accept_multiple_files=True
        )
        
        # Option to use docs folder
        if st.checkbox("Use documents from 'docs' folder"):
            if st.button("📂 Scan docs folder"):
                with st.spinner("Scanning docs folder..."):
                    try:
                        file_paths = scan_docs_folder("docs")
                        if file_paths:
                            st.success(f"Found {len(file_paths)} files in docs folder")
                            docs = load_documents(file_paths)
                            st.session_state.local_docs = docs
                            st.info(f"Loaded {len(docs)} documents")
                        else:
                            st.warning("No files found in docs folder")
                    except Exception as e:
                        st.error(f"Error scanning docs folder: {e}")
        
        if uploaded_files or st.session_state.get('local_docs'):
            if st.button("🔨 Build Vector Store"):
                with st.spinner("Processing documents and building vector store..."):
                    try:
                        # Use uploaded files or docs folder
                        temp_dir = None
                        if uploaded_files:
                            # Save uploaded files temporarily and load them
                            temp_dir = Path("temp_uploads")
                            temp_dir.mkdir(exist_ok=True)
                            
                            file_paths = []
                            for uploaded_file in uploaded_files:
                                temp_path = temp_dir / uploaded_file.name
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                file_paths.append(str(temp_path))
                            
                            docs = load_documents(file_paths)
                        else:
                            docs = st.session_state.local_docs
                        
                        # Build vector store
                        vectorstore = load_or_build_vectorstore(
                            st.session_state.settings,
                            st.session_state.embeddings,
                            docs
                        )
                        
                        # Create retriever and chain
                        retriever = make_retriever(vectorstore, st.session_state.settings)
                        rag_chain = build_rag_chain(st.session_state.llm, retriever)
                        
                        # Clean up temporary files

                        shutil.rmtree(temp_dir)
                        st.info("🗑️ Temporary files cleaned up")
                        
                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = retriever
                        st.session_state.rag_chain = rag_chain
                        st.session_state.messages = []
                        
                        st.success("✅ Vector store built successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error building vector store: {e}")
    
    # Chat interface
    if "vectorstore" in st.session_state:
        st.subheader("💬 Chat with your documents")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Show message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metrics" in message:
                    with st.expander("📊 RAGAS Metrics"):
                        st.write(message["metrics"])
        
        # User input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get RAG answer
                        answer = rag_answer(prompt, st.session_state.rag_chain)
                        st.markdown(answer)
                        
                        # Get RAGAS evaluation
                        with st.spinner("Evaluating response..."):
                            metrics = ragas_evaluation(
                                prompt,
                                st.session_state.rag_chain,
                                st.session_state.llm,
                                st.session_state.embeddings,
                                st.session_state.retriever,
                                st.session_state.settings
                            )
                        
                        # Add AI response to history with metrics
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "metrics": metrics
                        })
                        
                        # Show metrics
                        with st.expander("📊 RAGAS Metrics"):
                            st.write(metrics)
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

def web_search_page():
    """Page for web search RAG"""
    st.title("🌐 Web Search RAG")
    
    # Show current configuration in sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Using .env configuration")
        
        if st.button("🏠 Back to Menu"):
            for key in ['app_mode', 'vectorstore', 'retriever', 'rag_chain', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔍 Search Information")
        if "vectorstore" in st.session_state:
            st.success("✅ Web content indexed")
    
    # Web search setup
    if "vectorstore" not in st.session_state:
        st.subheader("🔍 Web Search Setup")
        st.markdown("Enter a search query to find and index web content:")
        
        search_query = st.text_input("Search Query", placeholder="Enter your search terms...")
        
        if search_query and st.button("🌐 Search and Index Web Content"):
            with st.spinner("Searching web and building vector store..."):
                try:
                    # Update settings with search query
                    st.session_state.settings.set_persist_dir_from_query(search_query)
                    st.info(f"Using index: {st.session_state.settings.persist_dir}")
                    
                    # Generate keywords
                    keywords = keywords_generation(search_query)
                    st.success(f"Generated keywords: {', '.join(keywords)}")
                    
                    # Search web
                    search_terms = " ".join(keywords)
                    urls = ddgs_results(search_terms)
                    st.info(f"Found {len(urls)} URLs")
                    
                    # Load web content
                    all_docs = []
                    progress_bar = st.progress(0)
                    
                    for i, url in enumerate(urls):
                        try:
                            docs = web_search_and_format(url)
                            all_docs.extend(docs)
                            progress_bar.progress((i + 1) / len(urls))
                        except Exception as e:
                            st.warning(f"Failed to load {url}: {e}")
                    
                    if all_docs:
                        st.success(f"Loaded {len(all_docs)} documents from web search")
                        
                        # Build vector store
                        vectorstore = load_or_build_vectorstore(
                            st.session_state.settings,
                            st.session_state.embeddings,
                            all_docs
                        )
                        
                        # Create retriever and chain
                        retriever = make_retriever(vectorstore, st.session_state.settings)
                        rag_chain = build_rag_chain(st.session_state.llm, retriever)
                        
                        # Store in session state
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = retriever
                        st.session_state.rag_chain = rag_chain
                        st.session_state.messages = []
                        st.session_state.search_query = search_query
                        
                        st.success("✅ Web content indexed successfully!")
                        st.rerun()
                    else:
                        st.error("No content could be loaded from web search")
                        
                except Exception as e:
                    st.error(f"Error during web search: {e}")
    
    # Chat interface
    if "vectorstore" in st.session_state:
        st.subheader(f"💬 Chat about: {st.session_state.get('search_query', 'Web content')}")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Show message history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metrics" in message:
                    with st.expander("📊 RAGAS Metrics"):
                        st.write(message["metrics"])
        
        # User input
        if prompt := st.chat_input("Ask a question about the web content..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Get RAG answer
                        answer = rag_answer(prompt, st.session_state.rag_chain)
                        st.markdown(answer)
                        
                        # Get RAGAS evaluation
                        with st.spinner("Evaluating response..."):
                            metrics = ragas_evaluation(
                                prompt,
                                st.session_state.rag_chain,
                                st.session_state.llm,
                                st.session_state.embeddings,
                                st.session_state.retriever,
                                st.session_state.settings
                            )
                        
                        # Add AI response to history with metrics
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "metrics": metrics
                        })
                        
                        # Show metrics
                        with st.expander("📊 RAGAS Metrics"):
                            st.write(metrics)
                        
                    except Exception as e:
                        error_msg = f"Error generating response: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

def main():
    st.set_page_config(
        page_title="RAG Assistant", 
        page_icon="🚀",
        layout="wide"
    )
    
    # Check if app mode is selected
    if not st.session_state.get('app_mode'):
        main_menu()
    else:
        # Route to the appropriate page based on selected mode
        if st.session_state.app_mode == "local":
            local_documents_page()
        elif st.session_state.app_mode == "web":
            web_search_page()

if __name__ == "__main__":
    main()
    
    