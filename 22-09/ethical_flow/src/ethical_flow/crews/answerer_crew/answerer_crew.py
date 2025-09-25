from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List
# import ssl
# ssl.set_default_verify_paths()

@CrewBase
class AnswererCrew:
    """Answerer Crew with research pipeline"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def web_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["web_researcher"],
            tools=[SerperDevTool(), ScrapeWebsiteTool()]
        )

    @agent
    def answerer(self) -> Agent:
        return Agent(
            config=self.agents_config["answerer"],
        )

    @task
    def web_research(self) -> Task:
        return Task(
            config=self.tasks_config["web_research"],
        )

    @task
    def final_answer(self) -> Task:
        return Task(
            config=self.tasks_config["final_answer"],
            context=[self.web_research()]  # Questo task riceve l'output del precedente
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )