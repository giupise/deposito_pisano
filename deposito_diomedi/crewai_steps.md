# Guida CrewAI Flow - Passi per la Creazione

## 1. Creazione del Flow
Genera un flow con il comando e naviga nella cartella:
```bash
crewai create flow guide_creator_flow
cd guide_creator_flow
```

## 2. Configurazione Ambiente
Nel file `.env` inserisci:
- Endpoint del modello
- API Key
- Tutte le variabili globali necessarie per il modello

## 3. Configurazione Agenti
Nel file `agents.yaml` definisci:
- Gli agenti con le loro funzionalità
- Personalità e caratteristiche degli agenti

## 4. Configurazione Task
Nel file `tasks.yaml` definisci:
- Lista dei task da svolgere
- Assegnazione di ogni task all'agente prestabilito

## 5. Aggiunta Crew (Opzionale)
Se necessario, crea una nuova crew per compiti aggiuntivi:
```bash
crewai flow add-crew content-crew
```

## 6. Creazione Tool Personalizzati (Opzionale)
Crea un nuovo tool usando il decoratore `@tool`:

```python
from crewai_tools import tool

@tool("add_two_numbers")
def add_two_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    res = a + b
    return res
```

**Importante:** Il tool deve essere inserito nella definizione dell'agente all'interno del file `nomecrew_crew.py`, dove devi definire:
- Agenti e task relativi alla crew
- Tool prestabiliti
- Input e output

## 7. Installazione Dipendenze
Dalla root principale del progetto:
```bash
crewai install
```
Questo comando creerà il `.venv` del progetto.

## 8. Esecuzione del Flow
1. Entra nel virtual environment e attivalo
2. Dalla root principale del progetto esegui:
```bash
crewai run
```

---

**Note:**
- Assicurati che tutti i file di configurazione siano correttamente popolati prima dell'esecuzione
- I tool personalizzati devono essere importati e assegnati agli agenti appropriati
- Verifica che le variabili d'ambiente siano configurate correttamente


