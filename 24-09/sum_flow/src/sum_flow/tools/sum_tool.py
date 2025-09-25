"""
Custom Tool realizzato per sommare due numeri
"""

from typing import Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SumToolInput(BaseModel):
    """Input per il tool di somma"""
    number1: float = Field(..., description="Primo addendo")
    number2: float = Field(..., description="Secondo addendo")

class SumTool(BaseTool):
    """Custom tool specializzato per compiere somme"""
    name: str = "sum_calculator"
    description: str = "tool specializzato per compiere somme"
    args_schema: type[BaseModel] = SumToolInput

def sum(self, number1: float, number2: float) -> str:
    """
    Args:
            number1: Primo addendo
            number2: Secondo addendo
            
        Returns:
            str: Risultato formattato della somma
    """
    try:
        result = number1 + number2
        return f"La somma dei numeri {number1} e {number2} è {result}"
    except Exception as e:
        return f"Errore nel calcolo della somma: {str(e)}"
