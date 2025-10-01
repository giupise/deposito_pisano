#!/usr/bin/env python3
"""
Interactive Culinary Flow - Direct RAG Implementation
Interactive version that can be used as a replacement for the complex CrewAI flow
"""

import os
import sys
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.culinary_flow.tools.rag_tool import document_search

# Load environment variables
load_dotenv()

def interactive_culinary_flow():
    """
    Interactive culinary flow that uses RAG directly
    """
    print("🍽️ Interactive Culinary Flow")
    print("=" * 50)
    print("Chiedi qualsiasi domanda culinaria e riceverai una risposta basata sulla knowledge base!")
    print("Digita 'quit' o 'exit' per uscire.")
    print("=" * 50)
    
    while True:
        try:
            # Get user input
            question = input("\n La tua domanda culinaria: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'q', 'esci']:
                print("\n👋 Arrivederci! Grazie per aver usato il sistema culinario.")
                break
            
            if not question:
                print("❌ Per favore, inserisci una domanda.")
                continue
            
            # Process the question
            result = process_culinary_question(question)
            print("\n" + result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Arrivederci! Grazie per aver usato il sistema culinario.")
            break
        except Exception as e:
            print(f"\n❌ Errore: {e}")

def process_culinary_question(question: str) -> str:
    """
    Process a culinary question using RAG
    """
    print(f"\n Elaborando: {question}")
    
    # Step 1: Check if question is culinary (simple keyword check)
    culinary_keywords = [
        'ricetta', 'pasta', 'pizza', 'cucina', 'cucinare', 'ingredienti',
        'preparazione', 'cottura', 'forno', 'pentola', 'chef', 'gastronomia',
        'alimentare', 'nutrizione', 'bevande', 'vino', 'caffè', 'tè',
        'dolce', 'salato', 'spezie', 'erbe', 'vegetariano', 'vegano',
        'pesto', 'carbonara', 'risotto', 'lasagne', 'tiramisu', 'pollo',
        'pesce', 'carne', 'verdura', 'frutta', 'pane', 'dolci'
    ]
    
    question_lower = question.lower()
    is_culinary = any(keyword in question_lower for keyword in culinary_keywords)
    
    if not is_culinary:
        return """
❌ Questa domanda non sembra riguardare il mondo culinario.

💡 Prova a chiedere qualcosa su:
- Ricette (es: "ricetta pasta al pesto")
- Ingredienti (es: "come usare il basilico")
- Tecniche di cucina (es: "come cuocere la pasta")
- Piatti specifici (es: "come fare la pizza")
- Bevande (es: "come preparare il caffè")
"""
    
    print(" Domanda culinaria rilevata")
    
    # Step 2: Search in RAG knowledge base
    print(" Cercando nella knowledge base...")
    try:
        rag_results = document_search(question)
        print(" Ricerca completata")
        
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
 Informazioni trovate nella knowledge base locale
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
- Assicurati che la domanda riguardi il mondo culinario

================================================================================
"""
    except Exception as e:
        return f"""
================================================================================
🍽️ RISPOSTA CULINARIA
================================================================================

Domanda: {question}

❌ Errore durante la ricerca: {e}

💡 Prova a:
- Riformulare la domanda
- Verificare che la connessione alla knowledge base sia attiva
- Contattare l'amministratore se il problema persiste

================================================================================
"""

def main():
    """Main function"""
    interactive_culinary_flow()

if __name__ == "__main__":
    main()
