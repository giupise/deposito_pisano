from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool


@CrewBase
class EthicalReviewerCrew():
    """EthicalReviewer crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def ethical_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['ethical_reviewer'],
            verbose=True
        )

    @task
    def ethical_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['ethical_review_task']
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