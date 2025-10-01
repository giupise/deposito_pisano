#!/usr/bin/env python3
"""
Simplified Culinary Flow - Direct RAG Implementation
Bypasses CrewAI complexity and uses RAG system directly
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.culinary_flow.tools.rag_tool import document_search

# Load environment variables
load_dotenv()

def simple_culinary_flow(question: str) -> str:
    """
    Simplified culinary flow that uses RAG directly
    """
    print(f" Processing question: {question}")
    
    # Step 1: Check if question is culinary (simple keyword check)
    culinary_keywords = [
        'ricetta', 'pasta', 'pizza', 'cucina', 'cucinare', 'ingredienti',
        'preparazione', 'cottura', 'forno', 'pentola', 'chef', 'gastronomia',
        'alimentare', 'nutrizione', 'bevande', 'vino', 'caffè', 'tè',
        'dolce', 'salato', 'spezie', 'erbe', 'vegetariano', 'vegano'
    ]
    
    question_lower = question.lower()
    is_culinary = any(keyword in question_lower for keyword in culinary_keywords)
    
    if not is_culinary:
        return "❌ Questa domanda non sembra riguardare il mondo culinario. Prova a chiedere qualcosa su ricette, ingredienti, tecniche di cucina, ecc."
    
    print(" Domanda culinaria rilevata")
    
    # Step 2: Search in RAG knowledge base
    print(" Cercando nella knowledge base...")
    try:
        rag_results = document_search(question)
        print(" Ricerca RAG completata")
        
        if rag_results and "❌" not in rag_results:
            # Step 3: Format the response
            response = f"""
================================================================================
🍽️ RISPOSTA CULINARIA
================================================================================

Domanda: {question}

Risposta basata sulla knowledge base:

{rag_results}

================================================================================
 Risposta generata dalla knowledge base locale
================================================================================
"""
            return response
        else:
            return f"""
================================================================================
🍽️ RISPOSTA CULINARIA
================================================================================

Domanda: {question}

❌ Non sono riuscito a trovare informazioni specifiche nella knowledge base per questa domanda.

💡 Suggerimenti:
- Prova a riformulare la domanda
- Usa termini più specifici
- Chiedi informazioni su ricette, ingredienti o tecniche di cucina

================================================================================
"""
    except Exception as e:
        return f"""
================================================================================
🍽️ RISPOSTA CULINARIA
================================================================================

Domanda: {question}

❌ Errore durante la ricerca: {e}

================================================================================
"""

def main():
    """Main function for testing"""
    print("🍽️ Simplified Culinary Flow")
    print("=" * 50)
    
    # Test with a sample question
    question = "ricetta pasta al pesto"
    result = simple_culinary_flow(question)
    print(result)

if __name__ == "__main__":
    main()
