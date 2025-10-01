from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from typing import List
import os
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
    def answerer(self) -> Agent:
        # Set environment variables for CrewAI to use
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        return Agent(
            config=self.agents_config["answerer"]
        )

    @task
    def final_answer(self) -> Task:
        return Task(
            config=self.tasks_config["final_answer"], 
            agent=self.answerer()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )