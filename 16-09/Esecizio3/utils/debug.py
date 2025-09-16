
# funzione per debug
def debug_retriever(retriever, question: str):
    """Funzione di debug per vedere cosa recupera il retriever"""
    print(f"\n[DEBUG] ========== TEST RETRIEVER ==========")
    print(f"[DEBUG] Query: '{question}'")
    
    docs = retriever.get_relevant_documents(question)
    print(f"[DEBUG] Documenti recuperati: {len(docs)}")
    
    for i, doc in enumerate(docs):
        print(f"\n[DEBUG] --- Documento {i+1} ---")
        print(f"  Source: {doc.metadata.get('source', 'N/A')}")
        print(f"  Content: {doc.page_content[:300]}...")
        if hasattr(doc, 'score'):
            print(f"  Score: {doc.score}")
    
    print(f"[DEBUG] ========== FINE TEST ==========\n")
    return docs