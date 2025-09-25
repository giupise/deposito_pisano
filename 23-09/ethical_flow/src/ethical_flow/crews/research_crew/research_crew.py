import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class ResearchCrew():
    """ResearchCrew for conducting research and writing content"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Inizializziamo il tool con la API key dal .env
    serper_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[self.serper_tool],   # passa il tool con la key
            verbose=True
        )

    @agent
    def content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task']
        )

    @task
    def content_writing_task(self) -> Task:
        return Task(
            config=self.tasks_config['content_writing_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchCrew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
