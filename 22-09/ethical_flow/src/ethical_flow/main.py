from crewai.flow import Flow, start, router, listen
import json

from ethical_flow.crews.ethical_reviewer_crew.ethical_reviewer_crew import EthicalReviewerCrew
from ethical_flow.crews.answerer_crew.answerer_crew import AnswererCrew

import os
from pathlib import Path
from dotenv import load_dotenv
import ssl
import urllib3

# PRIMA di tutto: configurazione SSL e disabilitazione warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configurazione SSL
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"

# Crea contesto SSL non verificato
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests PRIMA di importare crewai
import requests

original_request = requests.request
def no_ssl_request(*args, **kwargs):
    kwargs['verify'] = False
    return original_request(*args, **kwargs)
requests.request = no_ssl_request

original_session_request = requests.Session.request
def no_ssl_session_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_session_request(self, *args, **kwargs)
requests.Session.request = no_ssl_session_request


os.environ["OTEL_SDK_DISABLED"] = "true"

# Trova il .env partendo dal file corrente e risalendo
def find_and_load_env():
    current = Path(__file__).resolve()
    
    # Prova in diverse posizioni
    possible_locations = [
        current.parent,  # stessa directory di main.py
        current.parent.parent,  # ethical_flow/
        current.parent.parent.parent,  # root del progetto
    ]
    
    for location in possible_locations:
        env_file = location / '.env'
        if env_file.exists():
            print(f"✅ Trovato .env in: {env_file}")
            load_dotenv(env_file)
            return True
    
    print("❌ .env non trovato in nessuna posizione!")
    return False

# Carica il .env
find_and_load_env()

# Verifica
if os.getenv('OPENAI_API_KEY'):
    print("✅ OPENAI_API_KEY caricata")
elif os.getenv('AZURE_API_KEY'):
    print("✅ AZURE_API_KEY caricata")
else:
    print("⚠️ Nessuna API key trovata nel .env")


class EthicalFlow(Flow):
    def __init__(self):
        super().__init__()
        self.input_queue = []

    @start("retry")
    def start(self):
        return "starting"

    @listen(start)
    def ethical_review(self):
        """Step iniziale: revisione etica"""
        ethical_reviewer_crew = EthicalReviewerCrew().crew()
        result = ethical_reviewer_crew.kickoff(inputs={"user_input": self.state["user_input"]})
        return str(result)
    
    @router("ethical_review")
    def decide_route(self, result: str):
        """Smista il flusso in base al risultato JSON dell'ethical reviewer"""
        try:
            import re, json
            result_clean = str(result).strip()
            result_clean = result_clean.replace('{{', '{').replace('}}', '}')
            result_clean = result_clean.replace('\\"', '"')

            json_match = re.search(r'\{[^{}]*"is_ethical"[^{}]*\}', result_clean, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(result_clean)

        except Exception as e:
            print(f"❌ Errore parsing JSON: {e}")
            return "invalid_output"

        if parsed.get("is_ethical", False):
            print("✅ Domanda etica - proseguo verso answer_step")
            return "ethical"
        else:
            self.state["motivation"] = parsed.get("reasoning", "N/A")
            print(f"❌ Domanda non etica: {self.state['motivation']}")
            return "retry"

    @listen("ethical")
    def answer_step(self):
        """Se etico → passa all'answerer crew"""
        answerer_crew = AnswererCrew().crew()
        answer = answerer_crew.kickoff(inputs={"user_input": self.state["user_input"]})
        return f"💡 Risposta: {answer}"

    @listen("invalid_output")
    def invalid_output_step(self):
        return "⚠️ Errore: l'agente etico non ha restituito JSON valido."
    
    @listen("retry")
    def retry_step(self):
        """Qui non rifacciamo review: solo chiediamo nuovo input"""
        domanda = input("👉 Inserisci una nuova domanda: ")
        self.state["user_input"] = domanda
        return "starting"   # fa ripartire da start → ethical_review



  # --- Utility functions ---
def kickoff():
    flow = EthicalFlow()
    domanda = input("👉 Inserisci la tua domanda: ")
    flow.state["user_input"] = domanda
    result = flow.kickoff()
    print(result)


def plot():
    flow = EthicalFlow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
