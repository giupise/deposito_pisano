from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

@CrewBase
class EvaluateTopicCrew:
    """Evaluate Topic Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]


    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def topic_classifier(self) -> Agent:
        # Set environment variables for CrewAI to use
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        return Agent(
            config=self.agents_config["topic_classifier"]
        )

    @task
    def classify_topic_task(self) -> Task:
        return Task(
            config=self.tasks_config["classify_topic_task"], 
            agent=self.topic_classifier()
        )

    @crew
    def crew(self) -> Crew:
        """Evaluate the topic Crew"""
       

        return Crew(
            agents=self.agents,  
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
        )
