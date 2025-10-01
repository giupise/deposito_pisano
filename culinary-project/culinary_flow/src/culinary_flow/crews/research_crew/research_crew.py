from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
import os
import ssl
import urllib3

from dotenv import load_dotenv

load_dotenv()

# Disable SSL verification for Serper API
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

web_search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"), n_results=5)

@CrewBase
class WebCrew():
    agents: List[BaseAgent]
    tasks: List[Task]


    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    

    @agent
    def web_analyst(self) -> Agent:
        # Set environment variables for CrewAI to use
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        return Agent(
            config=self.agents_config["web_analyst"],
            tools=[web_search_tool],   # type: ignore[index]
        )

    @task
    def web_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["web_analysis_task"],  # type: ignore[index]
            agent=self.web_analyst()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
