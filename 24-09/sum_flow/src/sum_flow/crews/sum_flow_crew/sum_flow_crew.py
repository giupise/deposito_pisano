"""
Sum Flow Crew - Crew specializzata per calcoli di somma
Posizionata in crews/sum_flow_crew.py
Usa il formato moderno CrewAI con decorators
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tools.sum_tool import SumTool

@CrewBase
class SumFlowCrew():
    """Sum Flow crew per calcoli di somma con input utente"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        """Inizializza il custom tool."""
        self.sum_tool = SumTool()

    @agent
    def mathematical_calculator(self) -> Agent:
        """
        Agent matematico specializzato per calcoli di somma.
        Usa configurazione da agents.yaml
        """
        return Agent(
            config=self.agents_config['mathematical_calculator'],
            tools=[self.sum_tool],
            verbose=True
        )

    @task
    def sum_calculation_task(self) -> Task:
        """
        Task per eseguire calcoli di somma.
        Usa configurazione da tasks.yaml
        """
        return Task(
            config=self.tasks_config['sum_calculation'],
            agent=self.mathematical_calculator
        )

    @crew
    def crew(self) -> Crew:
        """
        Crea la Sum Flow crew con processo sequenziale.
        
        Returns:
            Crew: Crew configurata per calcoli di somma
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    def execute_sum(self, number1: float, number2: float) -> str:
        """
        Esegue l'operazione di somma utilizzando la crew.
        
        Args:
            number1: Primo numero da sommare
            number2: Secondo numero da sommare
            
        Returns:
            str: Risultato dell'operazione di somma
        """
        # Imposta le variabili per i placeholder nel task YAML
        inputs = {
            'number1': number1,
            'number2': number2
        }
        
        # Esegue la crew con gli input
        result = self.crew().kickoff(inputs=inputs)
        return str(result)

    def get_crew_info(self) -> dict:
        """
        Restituisce informazioni sulla crew.
        
        Returns:
            dict: Informazioni sulla composizione della crew
        """
        return {
            "name": "Sum Flow Crew",
            "agents": ["mathematical_calculator"],
            "tasks": ["sum_calculation_task"],
            "process": "sequential",
            "tools": [self.sum_tool.name],
            "purpose": "Calcolo di somme con validazione e precisione"
        }