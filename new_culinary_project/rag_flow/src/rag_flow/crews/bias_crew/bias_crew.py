from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class BiasCrew():
    """
    CrewAI crew for bias checking and content redaction.
    
    This crew specializes in identifying and mitigating potential biases in generated
    documents, ensuring ethical standards and content accuracy through automated
    analysis and redaction processes.
    
    Attributes
    ----------
    agents : List[BaseAgent]
        Collection of CrewAI agents specialized in bias detection and content analysis
    tasks : List[Task]
        Collection of tasks for bias checking workflow execution
    agents_config : str
        Path to YAML configuration file for agent definitions (config/agents.yaml)
    tasks_config : str
        Path to YAML configuration file for task definitions (config/tasks.yaml)
        
    Methods
    -------
    bias_checker() -> Agent
        Creates and configures the bias detection agent
    bias_check_task() -> Task
        Defines the bias checking task with markdown output generation
    crew() -> Crew
        Assembles and returns the complete bias checking crew
        
    Notes
    -----
    This crew uses YAML configuration files for agent and task definitions,
    enabling flexible configuration management and easy customization of
    bias detection parameters and criteria.
    
    The bias checking process outputs clean, redacted markdown documents
    that maintain content integrity while addressing identified biases.
    """

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


    @agent
    def bias_checker(self) -> Agent:
        """
        Create and configure the bias detection agent.
        
        Instantiates a specialized CrewAI agent responsible for identifying
        potential biases in documents and content. The agent uses YAML
        configuration for flexible parameter management.
        
        Returns
        -------
        Agent
            Configured CrewAI agent with bias detection capabilities
            
        Agent Configuration
        ------------------
        - Source: agents_config['bias_checker'] from YAML configuration
        - Verbose mode: Enabled for detailed execution logging
        - Role: Bias detection and content analysis specialist
        
        Notes
        -----
        The agent configuration is loaded from the YAML file specified in
        agents_config, allowing for easy customization of agent parameters,
        role definitions, and behavioral guidelines without code changes.
        """
        return Agent(
            config=self.agents_config['bias_checker'], # type: ignore[index]
            verbose=True
        )
    
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    

    
    @task
    def bias_check_task(self) -> Task:
        """
        Define the bias checking task with markdown output generation.
        
        Creates a specialized CrewAI task for comprehensive bias analysis
        and content redaction, producing clean markdown documentation
        free from identified biases.
        
        Returns
        -------
        Task
            Configured CrewAI task for bias checking and document redaction
            
        Task Configuration
        ------------------
        - Source: tasks_config['bias_check_task'] from YAML configuration
        - Output file: output/redacted_document.md
        - Verbose mode: Enabled for detailed execution tracking
        - Format: Markdown document with bias-free content
        
        Output Features
        ---------------
        - Automated bias identification and flagging
        - Content redaction maintaining document structure
        - Clean markdown formatting for professional presentation
        - Preservation of original content intent while addressing biases
        
        Notes
        -----
        The task generates a redacted document that removes or addresses
        identified biases while maintaining the core information and
        professional structure of the original content.
        """
        return Task(
            config=self.tasks_config['bias_check_task'],
            output_file="output/redacted_document.md",
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        """
        Assemble and configure the complete bias checking crew.
        
        Creates a fully configured CrewAI crew that orchestrates bias detection
        and content redaction workflows through sequential task execution
        and specialized agent coordination.
        
        Returns
        -------
        Crew
            Complete CrewAI crew configured for bias checking operations
            
        Crew Configuration
        ------------------
        - Agents: Automatically populated from @agent decorated methods
        - Tasks: Automatically populated from @task decorated methods  
        - Process: Sequential execution for systematic bias analysis
        - Verbose mode: Enabled for comprehensive execution monitoring
        
        Execution Flow
        --------------
        1. Document analysis for bias identification
        2. Content evaluation against ethical standards
        3. Bias flagging and categorization
        4. Document redaction and cleaning
        5. Clean markdown output generation
        
        Quality Assurance
        -----------------
        - Systematic bias detection across multiple dimensions
        - Ethical content validation and compliance checking
        - Professional document formatting and structure preservation
        - Comprehensive logging for audit and review purposes
        
        Notes
        -----
        The crew uses sequential processing to ensure thorough analysis
        and systematic bias mitigation while maintaining document quality
        and professional presentation standards.
        """
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
