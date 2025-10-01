from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from rag_flow.tools.custom_tool import TrustedWebSearch
import os
import ssl
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dotenv import load_dotenv

load_dotenv()

# Configure requests session to bypass SSL issues
class NoSSLHTTPAdapter(HTTPAdapter):
    """Custom HTTP Adapter that disables SSL verification"""
    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = ssl._create_unverified_context()
        return super().init_poolmanager(*args, **kwargs)

# Monkey patch requests to use our custom adapter
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    if not hasattr(self, '_no_ssl_mounted'):
        self.mount('https://', NoSSLHTTPAdapter())
        self._no_ssl_mounted = True
    kwargs.setdefault('verify', False)
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Also patch the main requests module
original_requests_request = requests.request
def patched_requests_request(*args, **kwargs):
    kwargs.setdefault('verify', False)
    return original_requests_request(*args, **kwargs)

requests.request = patched_requests_request

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
# Disable SSL verification for Serper API
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# web_search_tool = SerperDevTool(api_key=os.getenv("SERPER_API_KEY"), n_results=5)
web_search_tool = TrustedWebSearch(api_key=os.getenv("SERPER_API_KEY"), n_results=10)

@CrewBase
class WebCrew():
    """
    Web Research Crew for conducting comprehensive web searches and analysis.
    
    This crew specializes in performing web searches using SerperDev API, analyzing
    search results, and extracting relevant information from web sources to support
    research and documentation tasks.
    
    Attributes
    ----------
    agents : List[BaseAgent]
        List of web analyst agents assigned to this crew
    tasks : List[Task]
        List of web analysis tasks to be executed by the crew
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def web_analyst(self) -> Agent:
        """
        Create a web analysis agent specialized in web research and data extraction.
        
        This agent is configured to perform web searches, analyze search results,
        and extract relevant information from web sources using the SerperDev API.
        
        Returns
        -------
        Agent
            Configured web analyst agent with web search capabilities
        """
        return Agent(
            config=self.agents_config["web_analyst"],  # type: ignore[index]
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def web_analysis_task(self) -> Task:
        """
        Create a task for conducting comprehensive web analysis and research.
        
        This task uses the SerperDev web search tool to find relevant information
        on the web, analyze search results, and provide structured summaries of
        findings for further processing.
        
        Returns
        -------
        Task
            Configured web analysis task with SerperDevTool for web searching
        """
        return Task(
            config=self.tasks_config["web_analysis_task"],  # type: ignore[index]
            tools=[web_search_tool],  # Usa il tool definito con @tool
        )

    @crew
    def crew(self) -> Crew:
        """
        Create and configure the Web Research Crew.
        
        Assembles the crew with web analyst agents and analysis tasks for
        sequential processing of web research workflows using SerperDev API.
        
        Returns
        -------
        Crew
            Configured crew with agents, tasks, and sequential processing workflow
        """
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
