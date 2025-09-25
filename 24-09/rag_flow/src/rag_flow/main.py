import os
from pathlib import Path
from dotenv import load_dotenv
import ssl
import urllib3
import requests
import re
import json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"

ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests per ignorare SSL
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


def find_and_load_env():
    current = Path(__file__).resolve()
    possible_locations = [
        current.parent,
        current.parent.parent,
        current.parent.parent.parent,
    ]
    for location in possible_locations:
        env_file = location / ".env"
        if env_file.exists():
            print(f"✅ Trovato .env in: {env_file}")
            load_dotenv(env_file, override=True)
            return True
    print("❌ .env non trovato!")
    return False

find_and_load_env()


AZURE_KEY = os.getenv("AZURE_API_KEY")
AZURE_BASE = os.getenv("AZURE_API_BASE", "").rstrip("/") + "/"
AZURE_VERSION = os.getenv("AZURE_API_VERSION", "2024-02-01")
DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")

if AZURE_KEY and AZURE_BASE:
    os.environ["LITELLM_PROVIDER"] = "azure"
    os.environ["LITELLM_MODEL"] = f"azure/{DEPLOYMENT}"
    os.environ["LITELLM_API_BASE"] = AZURE_BASE
    os.environ["LITELLM_API_VERSION"] = AZURE_VERSION
    os.environ["LITELLM_API_KEY"] = AZURE_KEY
    os.environ["OPENAI_API_KEY"] = AZURE_KEY

    print("✅ Azure configurato")
    print(f"   Endpoint: {AZURE_BASE}")
    print(f"   Version : {AZURE_VERSION}")
    print(f"   Deploy  : {DEPLOYMENT}")
    print(f"   LiteLLM Model: azure/{DEPLOYMENT}")


from crewai.flow import Flow, start, router, listen
from ethical_flow.crews.ethical_reviewer_crew.ethical_reviewer_crew import EthicalReviewerCrew
from ethical_flow.crews.outline_crew.outline_crew import OutlineCrew
from ethical_flow.crews.research_crew.research_crew import ResearchCrew


def safe_json_extract(text: str):
    """
    Prova ad estrarre il primo blocco JSON valido da una stringa.
    """
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return {}

def clean_agent_output(text: str) -> str:
    """
    Ripulisce l'output degli agenti rimuovendo blocchi markdown, JSON grezzi
    e prefissi che non appartengono al contenuto finale del report.
    """
    # Rimuovi blocchi di codice tipo ```json ... ```
    text = re.sub(r"```[\s\S]*?```", "", text)

    # Rimuovi blocchi JSON isolati { ... } che contengono solo meta-informazioni
    text = re.sub(r'^\s*\{.*?\}\s*$', '', text, flags=re.DOTALL | re.MULTILINE)

    # Rimuovi linee che iniziano con "approved", "bias_score", "concerns"
    text = re.sub(r'^\s*"?approved"?:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*"?bias_score"?:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*"?concerns"?:.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Rimuovi prefissi comuni
    text = re.sub(r"(Final Answer:|Answer:)", "", text, flags=re.IGNORECASE)

    # Collassa linee vuote multiple
    text = re.sub(r"\n\s*\n", "\n\n", text)

    return text.strip()



class EthicalFlow(Flow):
    def __init__(self):
        super().__init__()
        self.outline_points = []
        self.research_content = {}

    @start("retry")
    def start(self):
        print("🚀 Avvio del flusso di valutazione etica...")
        return "starting"

    @listen(start)
    def ethical_review(self):
        print("⚖️ Fase 1: Ethical Review - Controllo etico della domanda")
        ethical_reviewer_crew = EthicalReviewerCrew().crew()
        result = ethical_reviewer_crew.kickoff(inputs={"user_input": self.state["user_input"]})
        return str(result)

    @router("ethical_review")
    def decide_ethical_route(self, result: str):
        parsed = safe_json_extract(result)

        if parsed.get("is_ethical", True):  # default = True
            print("✅ Domanda etica - proseguo verso outline_creation")
            return "ethical_approved"
        else:
            self.state["motivation"] = parsed.get("reasoning", "N/A")
            print(f"❌ Domanda non etica: {self.state['motivation']}")
            return "retry"

    @listen("ethical_approved")
    def outline_creation(self):
        print("📝 Fase 2: Outline Creation - Creazione scaletta punti")
        outline_crew = OutlineCrew().crew()
        result = outline_crew.kickoff(inputs={"user_input": self.state['user_input']})

        try:
            json_match = re.search(r'\{[^{}]*"outline"[^{}]*\}', str(result), re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                outline_sections = parsed.get("outline", [])
                self.outline_points = [s.get("section", "") for s in outline_sections]
                print(f"📋 Scaletta creata: {len(self.outline_points)} punti")
                return str(result)
        except Exception as e:
            print(f"⚠️ Errore parsing outline: {e}")

        lines = str(result).split("\n")
        self.outline_points = [line.strip() for line in lines if line.strip() and not line.startswith("📝")][:5]
        return str(result)

    @listen("outline_creation")
    def bias_review(self):
        print("🔍 Fase 3: Bias Review - Controllo bias e accuratezza")
        bias_prompt = f"""
        Analizza questa scaletta per bias, accuratezza e rischi:
        Punti: {self.outline_points}
        Topic originale: {self.state['user_input']}
        Restituisci un JSON con:
        {{"approved": true/false, "concerns": ["concern1"], "recommendations": ["rec1"]}}
        """
        bias_reviewer = EthicalReviewerCrew().crew()
        result = bias_reviewer.kickoff(inputs={"user_input": bias_prompt})
        return str(result)

    @router("bias_review")
    def decide_bias_route(self, result: str):
        parsed = safe_json_extract(result)

        if parsed.get("approved", True):  # default = True
            print("✅ Bias Review superato - proseguo verso research")
        else:
            print("⚠️ Bias rilevati - proseguo comunque verso research")

        return "bias_approved"

    @listen("bias_approved")
    def research_phase(self):
        print("🔬 Fase 4: Research Phase - Ricerca dettagliata")
        research_crew = ResearchCrew().crew()
        for i, point in enumerate(self.outline_points[:3]):
            print(f"🔍 Ricerca per punto {i+1}: {point}")
            research_result = research_crew.kickoff(inputs={
                "research_point": point,
                "main_topic": self.state['user_input']
            })
            self.research_content[point] = clean_agent_output(str(research_result))
        return f"Ricerca completata per {len(self.research_content)} punti"

    @listen("research_phase")
    def export_report(self):
        print("📄 Fase 5: Export - Creazione report finale")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        report_path = output_dir / "report.md"

        report_content = f"# Report di Analisi\n\n## Topic: {self.state['user_input']}\n\n## Scaletta Approvata\n"
        for i, point in enumerate(self.outline_points, 1):
            report_content += f"\n{i}. {point}"
        report_content += "\n\n## Ricerca Dettagliata\n"

        for point, research in self.research_content.items():
            cleaned = clean_agent_output(str(research))
            if cleaned:
                report_content += f"\n## {point}\n\n{cleaned}\n\n---\n"

        report_content += "\n\n*Report generato automaticamente dal sistema EthicalFlow*"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"✅ Report salvato in: {report_path}")
        return f"📄 Report completato e salvato in {report_path}"

    @listen("invalid_output")
    def invalid_output_step(self):
        return "⚠️ Errore: output non valido dal sistema di revisione."

    @listen("retry")
    def retry_step(self):
        domanda = input("👉 Inserisci una nuova domanda (deve essere etica): ")
        self.state["user_input"] = domanda
        return "starting"


def kickoff():
    print("🎯 Avvio EthicalFlow con nuovo sistema completo")
    print("=" * 50)
    flow = EthicalFlow()
    domanda = input("👉 Inserisci la tua domanda per l'analisi completa: ")
    flow.state["user_input"] = domanda
    result = flow.kickoff()
    print("\n" + "=" * 50)
    print("🏁 Flusso completato!")
    print(result)

def plot():
    flow = EthicalFlow()
    flow.plot()

if __name__ == "__main__":
    kickoff()
