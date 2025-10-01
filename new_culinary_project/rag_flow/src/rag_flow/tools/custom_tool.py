from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """
    Input schema for MyCustomTool.
    
    This class defines the expected input structure for the custom tool,
    using Pydantic BaseModel for validation and serialization.
    
    Attributes
    ----------
    argument : str
        A required string argument that serves as input for the tool execution.
        Must be provided for the tool to function properly.
    """

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    """
    Custom tool implementation extending CrewAI's BaseTool.
    
    This is a template class demonstrating how to create custom tools for CrewAI agents.
    It provides a structured approach to tool development with proper input validation
    and execution methods.
    
    Attributes
    ----------
    name : str
        Human-readable name of the tool for agent identification
    description : str
        Detailed description explaining the tool's functionality and use cases
    args_schema : Type[BaseModel]
        Pydantic model defining the expected input schema for validation
    """
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        """
        Execute the tool's main functionality.
        
        This method contains the core logic of the tool and is called when
        the tool is invoked by an agent. Override this method with your
        specific implementation.
        
        Parameters
        ----------
        argument : str
            The input argument as defined in the args_schema
            
        Returns
        -------
        str
            The result of the tool execution as a string
        """
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
