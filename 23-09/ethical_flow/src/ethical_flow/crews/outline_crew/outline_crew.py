from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class OutlineCrew():
    """OutlineCrew for creating structured content outlines"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def outline_creator(self) -> Agent:
        return Agent(
            config=self.agents_config['outline_creator'],
            verbose=True
        )

    @agent  
    def bias_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['bias_checker'],
            verbose=True
        )

    @task
    def create_outline_task(self) -> Task:
        return Task(
            config=self.tasks_config['create_outline_task']
        )

    @task
    def bias_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['bias_review_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the OutlineCrew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )