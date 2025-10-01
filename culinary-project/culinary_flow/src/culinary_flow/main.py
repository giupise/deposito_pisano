#!/usr/bin/env python3
"""
MAIN CULINARY FLOW ORCHESTRATOR
===============================

Orchestratore principale del sistema CrewAI culinario che gestisce il flusso completo:
1. Valutazione topic culinario
2. Ricerca RAG se culinario
3. Ricerca web se RAG insufficiente
4. Risposta finale

Autore: Sistema CrewAI Culinario
Versione: 1.0.0
"""

from pydantic import BaseModel
from crewai.flow import Flow, listen, start

from src.culinary_flow.crews.evaluate_topic_crew.evaluate_topic_crew import EvaluateTopicCrew
from src.culinary_flow.crews.culinary_rag_expert_crew.culinary_rag_expert_crew import CulinaryRAGExpertCrew
from src.culinary_flow.crews.research_crew.research_crew import WebCrew 
from src.culinary_flow.crews.answerer_crew.answerer_crew import AnswererCrew
import os 

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"


class CulinaryState(BaseModel):
    """Stato del flusso culinario"""
    question: str = ""
    is_culinary: bool = False
    confidence_score: float = 0.0
    classification_reasoning: str = ""
    
    # RAG results
    rag_status: str = ""  # SUFFICIENTE/PARZIALE/INSUFFICIENTE
    rag_content: dict = {}
    rag_confidence: float = 0.0
    
    # Web research results
    web_content: str = ""
    web_sources: list = []
    
    # Final answer
    final_answer: str = ""
    answer_sources: list = []


class CulinaryFlow(Flow[CulinaryState]):
    """
    Flusso principale del sistema culinario CrewAI
    
    Workflow:
    1. @start: Riceve domanda utente
    2. evaluate_topic: Valuta se culinaria
    3. Se non culinaria -> risposta negativa
    4. Se culinaria -> ricerca RAG
    5. Se RAG insufficiente -> ricerca web
    6. Risposta finale combinata
    """

    @start()
    def receive_user_question(self):
        """Punto di ingresso - riceve la domanda dell'utente"""
        print("Ricevendo domanda dall'utente...")
        
        # In un'applicazione reale, questo potrebbe venire da:
        # - Input CLI
        # - API request
        # - UI web
        # Per ora, chiediamo interattivamente
        
        if not self.state.question:
            self.state.question = input("Inserisci la tua domanda culinaria: ").strip()
        
        print(f"Domanda ricevuta: '{self.state.question}'")

    @listen(receive_user_question)
    def evaluate_topic(self):
        """Valuta se la domanda è culinaria usando evaluate_topic_crew"""
        print("Valutando se la domanda è culinaria...")
        
        try:
            result = (
                EvaluateTopicCrew()
                .crew()
                .kickoff(inputs={"question": self.state.question})
            )
            
            # Parse del risultato della crew di valutazione
            # Assumiamo che torni un JSON o structured output
            evaluation_data = self._parse_evaluation_result(result.raw)
            
            self.state.is_culinary = evaluation_data.get("is_culinary", False)
            self.state.confidence_score = evaluation_data.get("confidence_score", 0.0)
            self.state.classification_reasoning = evaluation_data.get("classification_reasoning", "")
            
            print(f"Valutazione completata: {'CULINARIA' if self.state.is_culinary else 'NON CULINARIA'}")
            print(f"Confidenza: {self.state.confidence_score:.2f}")
            
        except Exception as e:
            print(f"Errore nella valutazione topic: {e}")
            # Fallback: assumiamo sia culinaria per continuare il test
            self.state.is_culinary = True
            self.state.confidence_score = 0.5

    @listen(evaluate_topic)
    def handle_non_culinary_question(self):
        """Gestisce domande non culinarie con risposta negativa"""
        if not self.state.is_culinary:
            print("Domanda non culinaria - generando risposta negativa...")
            
            try:
                result = (
                    AnswererCrew()
                    .crew()
                    .kickoff(inputs={
                        "question": self.state.question,
                        "is_culinary": False,
                        "classification_reasoning": self.state.classification_reasoning,
                        "rag_content": {},
                        "web_content": ""
                    })
                )
                
                self.state.final_answer = result.raw
                print("Risposta negativa generata")
                
            except Exception as e:
                print(f"Errore nella risposta negativa: {e}")
                self.state.final_answer = (
                    "Mi dispiace, ma sono specializzato in questioni culinarie. "
                    "La tua domanda non sembra riguardare cucina, ricette o gastronomia. "
                    "Puoi farmi una domanda su ricette, ingredienti, tecniche di cucina?"
                )

    @listen(evaluate_topic)
    def search_rag_knowledge(self):
        """Cerca nella knowledge base RAG se la domanda è culinaria"""
        if self.state.is_culinary:
            print("Domanda culinaria - cercando nella knowledge base...")
            
            try:
                result = (
                    CulinaryRAGExpertCrew()
                    .crew()
                    .kickoff(inputs={"question": self.state.question})
                )
                
                # Parse del risultato RAG
                rag_data = self._parse_rag_result(result.raw)
                
                self.state.rag_status = rag_data.get("status", "INSUFFICIENTE")
                self.state.rag_content = rag_data.get("content_found", {})
                self.state.rag_confidence = rag_data.get("confidence_score", 0.0)
                
                print(f"Ricerca RAG completata - Status: {self.state.rag_status}")
                print(f"Confidenza RAG: {self.state.rag_confidence:.2f}")
                
            except Exception as e:
                print(f"Errore nella ricerca RAG: {e}")
                # Fallback
                self.state.rag_status = "INSUFFICIENTE"
                self.state.rag_content = {}
                self.state.rag_confidence = 0.0

    @listen(search_rag_knowledge)
    def generate_rag_only_answer(self):
        """Genera risposta basata solo su RAG se sufficienti"""
        if (self.state.is_culinary and 
            self.state.rag_status == "SUFFICIENTE" and 
            self.state.rag_confidence >= 0.7):
            
            print("Informazioni RAG sufficienti - generando risposta finale...")
            
            try:
                result = (
                    AnswererCrew()
                    .crew()
                    .kickoff(inputs={
                        "question": self.state.question,
                        "is_culinary": True,
                        "rag_content": self.state.rag_content,
                        "web_content": "",
                        "answer_type": "rag_only"
                    })
                )
                
                self.state.final_answer = result.raw
                self.state.answer_sources = ["Knowledge Base RAG"]
                
                print("Risposta RAG generata con successo")
                
            except Exception as e:
                print(f"Errore nella risposta RAG: {e}")
                # Continua al web search come fallback
                self.state.rag_status = "INSUFFICIENTE"

    @listen(search_rag_knowledge)
    def search_web_information(self):
        """Cerca informazioni web se RAG insufficiente"""
        if (self.state.is_culinary and 
            self.state.rag_status in ["INSUFFICIENTE", "PARZIALE"] and
            not self.state.final_answer):  # Solo se non abbiamo già una risposta RAG
            
            print("Informazioni RAG insufficienti - cercando sul web...")
            
            try:
                result = (
                    WebCrew()  # Aggiornato per matchare il nome della tua crew
                    .crew()
                    .kickoff(inputs={
                        "question": self.state.question,
                        "rag_gaps": self.state.rag_content.get("gaps_identified", []),
                        "search_terms": self.state.rag_content.get("suggested_search_terms", [])
                    })
                )
                
                # Parse risultato web research
                web_data = self._parse_web_result(result.raw)
                
                self.state.web_content = web_data.get("content", "")
                self.state.web_sources = web_data.get("sources", [])
                
                print("Ricerca web completata")
                print(f"Fonti web trovate: {len(self.state.web_sources)}")
                
            except Exception as e:
                print(f"Errore nella ricerca web: {e}")
                self.state.web_content = ""
                self.state.web_sources = []

    @listen(search_web_information)
    def generate_combined_answer(self):
        """Genera risposta finale combinando RAG + Web"""
        if (self.state.is_culinary and 
            not self.state.final_answer and  # Solo se non abbiamo già una risposta
            (self.state.rag_content or self.state.web_content)):
            
            print("Generando risposta combinata RAG + Web...")
            
            try:
                result = (
                    AnswererCrew()
                    .crew()
                    .kickoff(inputs={
                        "question": self.state.question,
                        "is_culinary": True,
                        "rag_content": self.state.rag_content,
                        "web_content": self.state.web_content,
                        "answer_type": "combined"
                    })
                )
                
                self.state.final_answer = result.raw
                
                # Combina fonti
                sources = []
                if self.state.rag_content:
                    sources.extend(self.state.rag_content.get("sources", []))
                if self.state.web_sources:
                    sources.extend(self.state.web_sources)
                self.state.answer_sources = sources
                
                print("Risposta combinata generata con successo")
                
            except Exception as e:
                print(f"Errore nella risposta combinata: {e}")
                self.state.final_answer = (
                    "Mi dispiace, c'è stato un errore nella generazione della risposta. "
                    "Riprova con una domanda più specifica."
                )

    @listen(generate_rag_only_answer)
    @listen(generate_combined_answer)
    @listen(handle_non_culinary_question)
    def display_final_result(self):
        """Mostra il risultato finale all'utente"""
        print("\n" + "="*80)
        print("RISPOSTA FINALE")
        print("="*80)
        print(f"\nDomanda: {self.state.question}")
        print(f"\nRisposta:\n{self.state.final_answer}")
        
        if self.state.answer_sources:
            print(f"\nFonti consultate:")
            for i, source in enumerate(self.state.answer_sources, 1):
                print(f"  {i}. {source}")
        
        print("\n" + "="*80)

    # ===========================
    # UTILITY METHODS
    # ===========================

    def _parse_evaluation_result(self, raw_result: str) -> dict:
        """Parse del risultato della crew di evaluation"""
        try:
            import json
            import re
            
            # Cerca pattern JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, raw_result)
            
            if matches:
                return json.loads(matches[-1])
            
            # Fallback parsing
            is_culinary = any(word in raw_result.lower() for word in ["culinaria", "culinario", "true", "sì"])
            return {
                "is_culinary": is_culinary,
                "confidence_score": 0.5,
                "classification_reasoning": raw_result[:200] + "..."
            }
            
        except Exception:
            return {"is_culinary": True, "confidence_score": 0.5}

    def _parse_rag_result(self, raw_result: str) -> dict:
        """Parse del risultato della crew RAG"""
        try:
            import json
            import re
            
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, raw_result, re.DOTALL)
            
            if matches:
                return json.loads(matches[-1])
            
            # Fallback
            status = "PARZIALE"
            if "SUFFICIENTE" in raw_result:
                status = "SUFFICIENTE"
            elif "INSUFFICIENTE" in raw_result:
                status = "INSUFFICIENTE"
                
            return {
                "status": status,
                "confidence_score": 0.5,
                "content_found": {"summary": raw_result[:300]}
            }
            
        except Exception:
            return {"status": "INSUFFICIENTE", "confidence_score": 0.0}

    def _parse_web_result(self, raw_result: str) -> dict:
        """Parse del risultato della crew web research"""
        try:
            # Estrai contenuto e fonti dal risultato web
            sources = []
            if "http" in raw_result:
                import re
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', raw_result)
                sources = urls
            
            return {
                "content": raw_result,
                "sources": sources
            }
            
        except Exception:
            return {"content": raw_result, "sources": []}


def kickoff():
    """Avvia il flusso culinario"""
    culinary_flow = CulinaryFlow()
    culinary_flow.kickoff()


def kickoff_with_question(question: str):
    """Avvia il flusso con una domanda specifica"""
    culinary_flow = CulinaryFlow()
    culinary_flow.state.question = question
    culinary_flow.kickoff()


def plot():
    """Visualizza il grafico del flusso"""
    culinary_flow = CulinaryFlow()
    culinary_flow.plot()


if __name__ == "__main__":
    kickoff()