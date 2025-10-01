#!/usr/bin/env python3
"""
CULINARY RAG EXPERT CREW
========================

Crew specializzata nella ricerca di informazioni culinarie dalla knowledge base locale.
Utilizza la struttura standard CrewAI con decoratori @agent, @task, @crew.

Autore: Sistema CrewAI Culinario
Versione: 1.0.0
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os

# Import del custom tool RAG dalla cartella tools
from src.culinary_flow.tools.rag_tool import document_search as document_search_fn, document_search_tool

@CrewBase
class CulinaryRAGExpertCrew:
    """Culinary RAG Expert Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Configurazione YAML files
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def document_searcher(self) -> Agent:
        """Ricercatore Documenti Knowledge Base"""
        # Set environment variables for CrewAI to use
        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        
        return Agent(
            config=self.agents_config["document_searcher"],
            tools=[document_search_tool],
        )

    @agent
    def information_analyzer(self) -> Agent:
        """Analizzatore di Qualità delle Informazioni"""
        return Agent(
            config=self.agents_config["information_analyzer"]
        )

    @agent
    def content_synthesizer(self) -> Agent:
        """Sintetizzatore di Risultati RAG"""
        return Agent(
            config=self.agents_config["content_synthesizer"]
        )

    @task
    def search_knowledge_base_task(self) -> Task:
        """Task per ricerca nella knowledge base usando il tool RAG"""
        return Task(
            config=self.tasks_config["search_knowledge_base_task"],
            agent=self.document_searcher(),
        )

    @task
    def analyze_search_results_task(self) -> Task:
        """Task per analisi qualità dei risultati RAG"""
        return Task(
            config=self.tasks_config["analyze_search_results_task"],
            agent=self.information_analyzer(),
            context=[self.search_knowledge_base_task()],
        )

    @task
    def synthesize_rag_results_task(self) -> Task:
        """Task per sintesi risultati finali"""
        return Task(
            config=self.tasks_config["synthesize_rag_results_task"],
            agent=self.content_synthesizer(),
            context=[self.search_knowledge_base_task(), self.analyze_search_results_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Culinary RAG Expert Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )