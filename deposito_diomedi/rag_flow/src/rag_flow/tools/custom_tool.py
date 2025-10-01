from typing import Type
import os
import yaml
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool


class TrustedWebSearchInput(BaseModel):
    search_query: str = Field(..., description="Search query for trusted web sources")

class TrustedWebSearch(BaseTool):
    name: str = "Trusted Web Search"
    description: str = "Search web using only trusted domains"
    args_schema: Type[BaseModel] = TrustedWebSearchInput
    serper_tool: SerperDevTool = None
    trusted_domains: list = None
    
    def _load_trusted_domains(self) -> list:
        """Carica domini trusted dal file YAML di configurazione"""
        # Percorso relativo al file YAML dalla posizione corrente
        yaml_path = Path(__file__).parent.parent / "crews" / "web_crew" / "config" / "domains.yaml"
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                domains = config.get('trusted_domains', [])
                # print(f"Caricati {len(domains)} domini trusted da {yaml_path}")
                return domains
        except FileNotFoundError:
            print(f"File {yaml_path} non trovato, uso domini di default")
            return self._get_default_domains()
        except yaml.YAMLError as e:
            print(f"Errore parsing YAML {yaml_path}: {e}, uso domini di default")
            return self._get_default_domains()
        except Exception as e:
            print(f"Errore caricamento domini: {e}, uso domini di default")
            return self._get_default_domains()
    
    def _get_default_domains(self) -> list:
        """Domini di fallback se il caricamento YAML fallisce"""
        return [
            'wikipedia.org', 'github.com', 'stackoverflow.com', 
            'python.org', 'nature.com', 'gov.it'
        ]
    
    def __init__(self, api_key: str, n_results: int = 10):
        super().__init__()
        # Imposta la variabile d'ambiente che SerperDevTool si aspetta
        import os
        os.environ["SERPER_API_KEY"] = api_key
        
        # Inizializza SerperDevTool con parametri completi
        self.serper_tool = SerperDevTool(n_results=n_results)
        # Carica domini trusted dal file YAML
        self.trusted_domains = self._load_trusted_domains()
    
    def _is_trusted_domain(self, url: str) -> bool:
        """Verifica se un URL appartiene a un dominio trusted"""
        return any(domain in url for domain in self.trusted_domains)
    
    def _process_organic_results(self, organic_results: list) -> list:
        """Processa risultati organici filtrandoli per domini trusted"""
        trusted_organic = []
        for result in organic_results:
            link = result.get("link", "")
            if self._is_trusted_domain(link):
                trusted_result = {
                    "title": result.get("title", ""),
                    "link": link,
                    "snippet": result.get("snippet", ""),
                    "position": result.get("position", ""),
                }
                
                # Aggiungi sitelinks se presenti e trusted
                if "sitelinks" in result:
                    trusted_sitelinks = []
                    for sitelink in result["sitelinks"]:
                        if self._is_trusted_domain(sitelink.get("link", "")):
                            trusted_sitelinks.append({
                                "title": sitelink.get("title", ""),
                                "link": sitelink.get("link", "")
                            })
                    if trusted_sitelinks:
                        trusted_result["sitelinks"] = trusted_sitelinks
                
                trusted_organic.append(trusted_result)
        return trusted_organic
    
    def _process_people_also_ask(self, paa_results: list) -> list:
        """Processa People Also Ask filtrandoli per domini trusted"""
        trusted_paa = []
        for result in paa_results:
            link = result.get("link", "")
            if not link or self._is_trusted_domain(link):
                trusted_paa.append({
                    "question": result.get("question", ""),
                    "snippet": result.get("snippet", ""),
                    "title": result.get("title", ""),
                    "link": link
                })
        return trusted_paa
    
    def _process_knowledge_graph(self, kg: dict) -> dict:
        """Processa Knowledge Graph se da fonte trusted"""
        kg_website = kg.get("website", "")
        if kg_website and self._is_trusted_domain(kg_website):
            return {
                "title": kg.get("title", ""),
                "type": kg.get("type", ""),
                "website": kg_website,
                "description": kg.get("description", ""),
                "descriptionSource": kg.get("descriptionSource", ""),
                "descriptionLink": kg.get("descriptionLink", ""),
                "attributes": kg.get("attributes", {})
            }
        return None
    
    def _process_related_searches(self, related_results: list) -> list:
        """Processa ricerche correlate (sempre incluse)"""
        return [{"query": result.get("query", "")} for result in related_results]
    
    def _format_output(self, trusted_data: dict, search_params: dict) -> str:
        """Formatta l'output finale con tutte le informazioni trusted"""
        output_lines = []
        
        # Header con parametri di ricerca
        total_trusted = (
            len(trusted_data.get("organic", [])) +
            (1 if trusted_data.get("knowledgeGraph") else 0)
        )
        
        output_lines.append(f"TRUSTED SEARCH RESULTS")
        output_lines.append(f"Query: {search_params.get('q', '')}")
        output_lines.append(f"Total trusted sources found: {total_trusted}")
        output_lines.append("=" * 60)
        output_lines.append("")
        
        # Knowledge Graph (prioritario)
        if "knowledgeGraph" in trusted_data:
            kg = trusted_data["knowledgeGraph"]
            output_lines.append("KNOWLEDGE GRAPH")
            output_lines.append(f"**{kg.get('title', '')}** ({kg.get('type', '')})")
            output_lines.append(f"Source: {kg.get('website', '')}")
            output_lines.append(f"{kg.get('description', '')}")
            if kg.get('attributes'):
                output_lines.append("Key Attributes:")
                for key, value in kg.get('attributes', {}).items():
                    output_lines.append(f"   • {key}: {value}")
            output_lines.append("")
        
        # Risultati organici
        if "organic" in trusted_data:
            output_lines.append("ORGANIC RESULTS")
            for i, result in enumerate(trusted_data["organic"], 1):
                output_lines.append(f"{i}. **{result.get('title', '')}**")
                output_lines.append(f" {result.get('link', '')}")
                output_lines.append(f"{result.get('snippet', '')}")
                output_lines.append(f"Position: {result.get('position', 'N/A')}")
                
                # Sitelinks se presenti
                if result.get('sitelinks'):
                    output_lines.append("   🔗 Related links:")
                    for sitelink in result['sitelinks']:
                        output_lines.append(f"      • {sitelink.get('title', '')}: {sitelink.get('link', '')}")
                output_lines.append("")
        
        # People Also Ask
        if "peopleAlsoAsk" in trusted_data:
            output_lines.append("PEOPLE ALSO ASK")
            for i, paa in enumerate(trusted_data["peopleAlsoAsk"], 1):
                output_lines.append(f"{i}. Q: {paa.get('question', '')}")
                if paa.get('snippet'):
                    output_lines.append(f"   A: {paa.get('snippet', '')}")
                if paa.get('link'):
                    output_lines.append(f"  {paa.get('link', '')}")
                output_lines.append("")
        
        # Ricerche correlate
        if "relatedSearches" in trusted_data:
            output_lines.append("RELATED SEARCHES")
            for search in trusted_data["relatedSearches"]:
                output_lines.append(f"   • {search.get('query', '')}")
            output_lines.append("")
        
        return "\n".join(output_lines)
    
    def _run(self, search_query: str) -> str:
        # Ricerca completa con SerperDevTool
        results = self.serper_tool._run(search_query=search_query)
        
        # Estrai dati trusted da tutte le sezioni
        trusted_data = {}
        
        # Processa risultati organici
        if "organic" in results:
            trusted_organic = self._process_organic_results(results["organic"])
            if trusted_organic:
                trusted_data["organic"] = trusted_organic
        
        # Processa Knowledge Graph
        if "knowledgeGraph" in results:
            trusted_kg = self._process_knowledge_graph(results["knowledgeGraph"])
            if trusted_kg:
                trusted_data["knowledgeGraph"] = trusted_kg
        
        # Processa People Also Ask
        if "peopleAlsoAsk" in results:
            trusted_paa = self._process_people_also_ask(results["peopleAlsoAsk"])
            if trusted_paa:
                trusted_data["peopleAlsoAsk"] = trusted_paa
        
        # Processa ricerche correlate (sempre incluse se presenti)
        if "relatedSearches" in results:
            trusted_data["relatedSearches"] = self._process_related_searches(results["relatedSearches"])
        
        # Formatta output finale
        if trusted_data:
            return self._format_output(trusted_data, results.get("searchParameters", {}))
        else:
            total_results = len(results.get("organic", []))
            return f"NO TRUSTED SOURCES FOUND\nTotal results available: {total_results}\nConsider expanding trusted domains list."