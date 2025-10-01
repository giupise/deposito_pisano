from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from rag_flow.tools.rag_w_qdrant.main import rag_system


@CrewBase
class AeronauticRagCrew():
    """
    Aeronautic RAG (Retrieval-Augmented Generation) Crew for technical aeronautic questions.
    
    This crew specializes in answering aeronautic-related questions using a RAG system
    that retrieves relevant information from a knowledge base and generates accurate responses.
    
    Attributes
    ----------
    agents : List[BaseAgent]
        List of agents assigned to this crew
    tasks : List[Task]  
        List of tasks to be executed by the crew
    agents_config : str
        Path to the YAML configuration file for agents
    tasks_config : str
        Path to the YAML configuration file for tasks
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def rag_expert(self) -> Agent:
        """
        Create a RAG expert agent specialized in aeronautic knowledge retrieval.
        
        This agent is configured to use RAG (Retrieval-Augmented Generation) system
        to answer technical questions about aeronautics by retrieving relevant information
        from the knowledge base.
        
        Returns
        -------
        Agent
            Configured RAG expert agent with aeronautic domain expertise
        """
        return Agent(
            config=self.agents_config["rag_expert"],  # type: ignore[index]
        )



    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    
    @task
    def rag_response_task(self) -> Task:
        """
        Create a task for generating RAG-based responses to aeronautic questions.
        
        This task uses the RAG system tool to retrieve relevant context from the
        knowledge base and generate comprehensive answers to technical questions
        about aeronautics.
        
        Returns
        -------
        Task
            Configured task for RAG-based question answering with rag_system tool
        """
        return Task(
            config=self.tasks_config["rag_response_task"],  # type: ignore[index]
            tools=[rag_system],  # Usa il tool definito con @tool
        )



    
    @crew
    def crew(self) -> Crew:
        """
        Create and configure the Aeronautic RAG Crew.
        
        Assembles the crew with RAG expert agents and response generation tasks
        for sequential processing of aeronautic questions using RAG methodology.
        
        Returns
        -------
        Crew
            Configured crew with agents, tasks, and sequential processing workflow
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
