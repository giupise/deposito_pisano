from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from rag_flow.tools.refactored_faiss_code.main import rag_system


@CrewBase
class RagFlowCrew():
    """EthicalReviewer crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def areonautic_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['areonautic_expert'],
        )
    
    @agent
    def rag_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_expert'],
        )
    
    @agent
    def doc_redactor(self) -> Agent:
        return Agent(
            config=self.agents_config['doc_redactor'],
        )

    @task
    def evaluate_task(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_task']
        )
    
    @task
    def rag_response_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_response_task'],
            tools=[rag_system]
        )
 
    @task
    def doc_redaction_task(self) -> Task:
        return Task(
            config=self.tasks_config['doc_redaction_task'],
            context=[self.rag_response_task()],
            output_file=["output/redacted_document.md"]
        )

   


    @crew
    def crew(self) -> Crew:
        """Creates the EthicalReviewer crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )