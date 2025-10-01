# Architettura CrewAI - RAG Flow

## 📋 Panoramica

Questo progetto implementa un'architettura **CrewAI Flow** avanzata che combina orchestrazione di flussi, agenti specializzati e strumenti personalizzati per creare un sistema di domande e risposte intelligente sull'aeronautica.

---

## 🌊 CrewAI Flow Architecture

### Concetti Fondamentali

**CrewAI Flow** è un pattern di orchestrazione che permette di:
- Definire sequenze di operazioni complesse
- Gestire stati condivisi tra diverse fasi
- Implementare routing condizionale basato sui risultati
- Coordinare l'esecuzione di multiple Crew

### Pattern Flow

```python
class AeronauticRagFlow(Flow[AeronauticRagState]):
    @start()                    # 🚀 Entry point del flow
    @listen(method)             # 👂 Listener per eventi
    @router(condition)          # 🔀 Routing condizionale
```

---

## 🏗️ Struttura dell'Architettura

### Gerarchia Componenti

```
CrewAI Flow System
├── 🌊 Flow Layer (Orchestrazione)
│   ├── State Management (Stato condiviso)
│   ├── Event Listeners (Reattività)
│   └── Conditional Routing (Logica di branching)
├── 👥 Crew Layer (Agenti specializzati)
│   ├── RAG Expert Crew
│   ├── Document Generation Crew
│   └── Web Research Crew
├── 🔧 Tool Layer (Strumenti)
│   ├── RAG System Tool
│   ├── Web Search Tools
│   └── Content Validation Tools
└── 📚 Data Layer (Conoscenza)
    ├── Vector Stores (FAISS)
    ├── Document Collections
    └── Web Content Cache
```

---

## 🌊 Flow Orchestration

### Definizione del Flow

```python
@dataclass
class AeronauticRagState(BaseModel):
    """Stato condiviso attraverso tutto il flow"""
    question_input: str = ""
    rag_result: str = ""
```

### Metodi del Flow

#### **1. Start Method**
```python
@start()
def starting_procedure(self):
    """
    🚀 Entry point - inizializza il sistema
    - Setup logging
    - Configurazione iniziale
    - State initialization
    """
    print("Starting the Aeronautic RAG Flow")
    return "ready"
```

#### **2. Listener Methods**
```python
@listen(starting_procedure)
def generate_question(self):
    """
    👂 Ascolta il completion di starting_procedure
    - Input utente interattivo
    - Validazione formato domanda
    - State update
    """
    question = input("Enter your question about aeronautics:")
    self.state.question_input = question
    return question
```

#### **3. Router Methods**
```python
@router("success")
def rag_analysis(self):
    """
    🔀 Routing condizionale basato su analisi LLM
    - Esecuzione solo se question_analysis ritorna "success"
    - Delegazione a Crew specializzata
    - Result processing
    """
```

### Flow Execution Pattern

1. **Sequential Execution**: Ogni step attende il precedente
2. **State Persistence**: Lo stato è condiviso tra tutti i metodi
3. **Event-Driven**: Listeners reagiscono al completamento
4. **Conditional Branching**: Router permettono logica complessa

---

## 👥 Crew Architecture

### Crew Specialization Pattern

Il sistema implementa **3 crew specializzate**, ciascuna con agenti dedicati e responsabilità specifiche:

#### **1. AeronauticRagCrew** (`crews/poem_crew/`)
- **Agente**: `rag_expert`
- **Ruolo**: RAG Expert Agent specializzato in aeronautica
- **Responsabilità**: 
  - Utilizzare il sistema RAG per recuperare informazioni contestuali
  - Rispondere a domande aeronautiche basandosi su documenti indicizzati
  - Validare la presenza di informazioni nel contesto disponibile
- **Tool**: `rag_system` (FAISS + Azure OpenAI)
- **Output**: Risposte tecniche accurate basate su retrieval

#### **2. WebCrew** (`crews/web_crew/`)
- **Agente**: `web_analyst`
- **Ruolo**: Web Analyst Agent per ricerca e analisi web
- **Responsabilità**:
  - Eseguire ricerche web su topic aeronautici
  - Analizzare risultati di ricerca per rilevanza
  - Estrarre insights chiave da grandi volumi di dati web
  - Fornire summary concisi e strutturati
- **Tool**: `SerperDevTool` (Google Search API)
- **Output**: Summary analitici di contenuti web

#### **3. DocCrew** (`crews/doc_crew/`)
- **Agente**: `doc_redactor`
- **Ruolo**: Document Redactor Agent per generazione documenti
- **Responsabilità**:
  - Creare documenti Markdown strutturati
  - Integrare informazioni da RAG Expert e Web Analyst
  - Garantire chiarezza, coerenza e formattazione professionale
  - Produrre output finale user-ready
- **Tool**: Nessun tool esterno (processing interno)
- **Output**: Documenti Markdown formattati e completi

### Crew Structure

```python
# 1. AeronauticRagCrew - Specializzata in retrieval contestuale
@CrewBase
class AeronauticRagCrew:
    """Crew per elaborazione RAG aeronautica"""
    
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @agent
    def rag_expert(self) -> Agent:
        """🤖 Agente esperto in sistemi RAG"""
        return Agent(
            config=self.agents_config["rag_expert"],
            tools=[rag_system],  # 🔧 RAG tool integration
            verbose=True
        )
    
    @task
    def rag_response_task(self) -> Task:
        """📋 Task di elaborazione RAG"""
        return Task(
            config=self.tasks_config["rag_response_task"],
            agent=self.rag_expert,
        )

# 2. WebCrew - Specializzata in ricerca web
@CrewBase  
class WebCrew:
    """Crew per ricerca e analisi web"""
    
    @agent
    def web_analyst(self) -> Agent:
        """🌐 Agente analista web"""
        return Agent(
            config=self.agents_config["web_analyst"],
            tools=[SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))],
            verbose=True
        )
    
    @task
    def web_analysis_task(self) -> Task:
        """📊 Task di analisi web"""
        return Task(
            config=self.tasks_config["web_analysis_task"],
            agent=self.web_analyst,
        )

# 3. DocCrew - Specializzata in generazione documenti
@CrewBase
class DocCrew:
    """Crew per generazione documenti strutturati"""
    
    @agent
    def doc_redactor(self) -> Agent:
        """📝 Agente redattore documenti"""
        return Agent(
            config=self.agents_config["doc_redactor"],
            verbose=True
        )
    
    @task
    def document_creation_task(self) -> Task:
        """📄 Task di creazione documenti"""
        return Task(
            config=self.tasks_config["document_creation_task"],
            agent=self.doc_redactor,
            output_file="output/generated_document.md"
        )
    
    @crew
    def crew(self) -> Crew:
        """👥 Assembly della crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
```

### Agent Configuration (YAML)

```yaml
# crews/poem_crew/config/agents.yaml
rag_expert:
  role: >
    RAG Expert Agent specializzato in aeronautica
  goal: >
    Use the RAG system to answer {question} based on provided context.
    MUST use RAG system to retrieve relevant information and generate accurate response.
  backstory: >
    Specialized agent that uses Retrieval-Augmented Generation (RAG) system
    to provide answers based on given context. Always uses RAG system instead of
    answering questions without context.
  llm: azure/gpt-4o

# crews/web_crew/config/agents.yaml  
web_analyst:
  role: >
    Web Analyst Agent per ricerca e analisi web
  goal: >
    Analyze web search results and extract relevant information about {question}.
    Summarize key points and insights from web search results.
    Check relevance to {question} before including in summary.
  backstory: >
    Specialized agent that analyzes web search results to extract relevant information.
    Keen eye for detail, quickly identifies important insights from large data volumes.
    Always provides clear and concise summaries.
  llm: azure/gpt-4o

# crews/doc_crew/config/agents.yaml
doc_redactor:
  role: >
    Document Redactor Agent
  goal: >
    Generate document in .md format about {paper} provided by rag_expert and web_analyst.
  backstory: >
    Specialized agent that creates well-structured markdown documents
    based on {paper} from rag_expert and web_analyst agents.
    Always formats in markdown ensuring clarity and coherence.
  llm: azure/gpt-4o
```

### Task Configuration (YAML)

```yaml
# crews/poem_crew/config/tasks.yaml
rag_response_task:
  description: >
    Use the RAG system to answer the {question} based on the provided context.
    You MUST use the RAG system to retrieve relevant information and generate
    accurate answers.
  expected_output: >
    A comprehensive and accurate answer to the question based on retrieved
    context from the RAG system, formatted in clear and professional language.
  agent: rag_expert

# crews/web_crew/config/tasks.yaml
web_analysis_task:
  description: >
    Search the web for information related to {question} and analyze the results.
    Extract the most relevant information and provide a structured summary.
  expected_output: >
    A structured summary of web search results with key insights and relevant
    information about {question}, properly formatted and organized.
  agent: web_analyst

# crews/doc_crew/config/tasks.yaml  
document_creation_task:
  description: >
    Create a comprehensive markdown document about {paper} combining information
    from RAG expert and web analyst findings.
  expected_output: >
    A well-structured markdown document that integrates information from both
    RAG system and web research, formatted professionally with clear sections.
  agent: doc_redactor
  output_file: "output/generated_document.md"
```

### Crew Integration nel Flow

```python
@listen("success")
def rag_analysis(self):
    """Integrazione Crew nel Flow"""
    result = (
        AeronauticRagCrew()                    # 🏭 Istanziazione Crew
        .crew()                               # 👥 Creazione crew assembly
        .kickoff(inputs={                     # 🚀 Esecuzione con input
            "question": self.state.question_input,
            "response": self.state.rag_result
        })
    )
    self.state.rag_result = result.raw        # 💾 State update
```

---

## 🔧 Tool System Architecture

### Tool Definition Pattern

I **Tool** in CrewAI sono funzioni Python decorate che possono essere:
- Assegnate agli agenti
- Utilizzate per interagire con sistemi esterni  
- Composte per creare funzionalità complesse

### Tool Structure

```python
@tool('rag_system')
def rag_system(question: str) -> str:
    """
    🔧 Tool principale per il sistema RAG
    
    Capabilities:
    - Document loading and indexing
    - Vector similarity search
    - Web search integration  
    - Content validation
    - Response generation
    """
    # Tool implementation
    return processed_answer
```

### Tool Integration

#### **1. Agent-Tool Assignment**
```python
@agent  
def rag_expert_agent(self) -> Agent:
    return Agent(
        config=self.agents_config["rag_expert_agent"],
        tools=[rag_system, web_search_tool],  # 🔧 Multiple tools
        tool_sharing=True                      # 🤝 Tool sharing tra agenti
    )
```

#### **2. Tool Execution Flow**
```
Agent riceve Task → Analizza requirements → Seleziona Tool → Esegue Tool → Processa risultato
```

#### **3. Tool Chaining**
I tool possono essere concatenati per operazioni complesse:
```python
# Esempio di tool chaining nel RAG system
def rag_system(question: str) -> str:
    docs = load_documents()           # 📚 Document loading
    embeddings = create_embeddings()  # 🔤 Vectorization  
    results = search_similar()        # 🔍 Similarity search
    web_content = web_search()        # 🌐 Web enhancement
    answer = generate_response()      # 💭 LLM generation
    return answer
```

---

## 🔄 Flow Execution Patterns

### Event-Driven Architecture

#### **1. Linear Flow**
```python
@start() → @listen(start) → @listen(step2) → @listen(step3)
```

#### **2. Conditional Flow** 
```python
@start() → @listen(start) → @router("condition") → [@listen("success"), @listen("failure")]
```

#### **3. Parallel Processing**
```python
@start() → [@listen(start), @listen(start)] → @listen([parallel_task1, parallel_task2])
```

### State Management

```python
class AeronauticRagState(BaseModel):
    """Stato globale del Flow"""
    question_input: str = ""          # 📝 Input utente
    rag_result: str = ""             # 🤖 Risultato elaborazione
    
    # State è accessibile da tutti i metodi del Flow
    def any_flow_method(self):
        self.state.question_input = "new value"  # ✍️ State update
        current_value = self.state.rag_result    # 👁️ State read
```

### Flow Communication Patterns

#### **1. Method Return Values**
```python
@start()
def step1(self):
    return "success"  # 📤 Ritorna valore per routing

@router("success") 
def step2(self):     # 📥 Riceve solo se step1 ritorna "success"
    pass
```

#### **2. State Sharing**
```python
@start()
def step1(self):
    self.state.data = "shared_value"  # 💾 Salva nello state

@listen(step1)
def step2(self):
    value = self.state.data          # 📖 Legge dallo state
```

---

## 🚀 Esecuzione e Deployment

### Setup Ambiente

#### **1. Installazione Dipendenze**
```bash
# Clona il repository
git clone <repository-url>
cd rag_flow

# Installa dipendenze con uv
uv sync

# Oppure con pip
pip install -r requirements.txt
```

#### **2. Configurazione Environment**
```bash
# Crea file .env
cp .env.example .env

# Configura le variabili:
AZURE_OPENAI_ENDPOINT=https://your-endpoint.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_API_VERSION=2024-12-01-preview
MODEL=azure/gpt-4o
SERPER_API_KEY=your-serper-key
```

#### **3. Struttura Configurazione**
```
rag_flow/
├── .env                    # 🔐 Variabili ambiente
├── pyproject.toml         # 📦 Dipendenze progetto
└── src/rag_flow/
    ├── main.py           # 🌊 Flow definition
    └── crews/            # 👥 Crew configurations
        └── */config/
            ├── agents.yaml
            └── tasks.yaml
```

### Comandi di Esecuzione

#### **1. Esecuzione Interattiva**
```bash
# Attiva environment virtuale
source .venv/bin/activate  # Linux/Mac
# oppure
.venv\Scripts\activate     # Windows

# Esegui il flow
crewai run

# Output interattivo:
# Starting the Aeronautic RAG Flow
# Enter your question about aeronautics: [USER INPUT]
# Analyzing question...
# [FLOW EXECUTION]
```

#### **2. Esecuzione con Debugging**
```bash
# Modalità verbose per debugging
crewai run --verbose

# Trace completo dell'esecuzione
crewai run --debug
```

#### **3. Esecuzione Programmatica**
```python
# In Python script
from rag_flow.main import AeronauticRagFlow

# Istanzia e esegui flow
flow = AeronauticRagFlow()
result = flow.kickoff()
print(result)
```

### Flow Monitoring

#### **1. Logging Structure**
```
🌊 Flow: AeronauticRagFlow
ID: [unique-flow-id]
├── ✅ Completed: starting_procedure
├── ✅ Completed: generate_question  
├── ✅ Completed: question_analysis
└── 🔄 Running: rag_analysis
    └── 🚀 Crew: AeronauticRagCrew
        └── 📋 Task: rag_response_task
            └── 🤖 Agent: RAG Expert Agent
                └── 🔧 Tool: rag_system
```

#### **2. Error Handling**
```python
# Flow con error handling
try:
    result = flow.kickoff()
except FlowExecutionError as e:
    print(f"Flow failed at step: {e.step}")
    print(f"Error: {e.message}")
except CrewExecutionError as e:
    print(f"Crew failed: {e.crew_name}")
    print(f"Agent: {e.agent_name}")
```

### Performance Optimization

#### **1. Parallel Execution**
```python
# Per crew indipendenti
@listen(start_step)
async def parallel_crew_1(self): ...

@listen(start_step)  
async def parallel_crew_2(self): ...
```

#### **2. Caching**
```python
# Cache per risultati costosi
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_rag_operation(query: str): ...
```

#### **3. Resource Management**
```python
# Configurazione limiti
crew = Crew(
    agents=agents,
    tasks=tasks,
    max_execution_time=300,    # 5 minuti timeout
    max_retry=3                # Retry automatici
)
```

---

## 🎯 Best Practices

### Flow Design

1. **Single Responsibility**: Ogni step del flow ha una responsabilità specifica
2. **State Minimization**: Mantieni lo state essenziale e clean
3. **Error Boundaries**: Implementa graceful degradation
4. **Monitoring**: Logging strutturato per debugging

### Crew Organization

1. **Domain Separation**: Una crew per dominio/funzionalità
2. **Agent Specialization**: Agenti specializzati con tool specifici
3. **Configuration Externalization**: YAML per flessibilità
4. **Resource Isolation**: Evita condivisione non necessaria

### Tool Development

1. **Single Purpose**: Un tool = una funzionalità
2. **Type Safety**: Annotazioni di tipo per parametri/return
3. **Error Handling**: Gestione robusta degli errori
4. **Documentation**: Docstring dettagliate per gli agenti

---

## 🔍 RAG System Integration

Il sistema RAG è integrato come **tool specializzato** che fornisce:

- **Document Retrieval**: FAISS vector search per documenti locali
- **Web Enhancement**: DuckDuckGo search per contenuti aggiornati  
- **Content Validation**: LLM-based quality assessment
- **Response Generation**: Context-aware answer synthesis

Questa integrazione permette agli agenti CrewAI di accedere a conoscenza estesa mantenendo l'architettura modulare e scalabile.

---

## 📈 Scalability & Extensions

### Horizontal Scaling
- Multiple crew instances
- Distributed tool execution
- Load balancing per web requests

### Vertical Scaling  
- Enhanced LLM models
- Larger vector stores
- More sophisticated routing logic

### Extensibility Points
- Nuove crew specializzate
- Tool aggiuntivi per domini specifici
- Integration con sistemi enterprise

---

*Documentazione CrewAI Architecture*  
*Versione: 2.0*  
*Focus: Flow, Crews & Tools*