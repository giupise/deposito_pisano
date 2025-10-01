"""
Sum Flow 
Flow dedicato a raccogliere due numeri dall'utente e sommarli usando un custom tool (sum_tool) di CrewAI

"""

import sys 
from typing import Dict, Any
from crewai.flow.flow import Flow, listen, start
from crews.sum_flow_crew import SumFlowCrew 


class InputValidator:
    """Classe per validare gli input numerici dell'utente."""
    
    @staticmethod
    def get_valid_number(prompt: str) -> float:
        """
        Richiede un numero valido dall'utente con validazione.
        """
        while True:
            try:
                print(f"\n{prompt}")
                user_input = input(">>> ").strip()
                
                if not user_input:
                    print("❌ Input vuoto! Inserisci un numero.")
                    continue
                
                number = float(user_input)
                print(f" Numero valido: {number}")
                return number
                
            except ValueError:
                print(f"❌ '{user_input}' non è un numero valido!")
                print("💡 Esempi validi: 5, 3.14, -2, 0, 10.5")
                continue
            except KeyboardInterrupt:
                print("\n\n⚠️ Operazione interrotta dall'utente.")
                raise

class SumFlowMain(Flow):

    def __init__(self):
        super().__init__()
        self.validator = InputValidator()
        self.sum_crew = SumFlowCrew()

    @start()
    def collect_input(self) -> Dict[str, Any]:
        """
        Raccolta dell'input utente
        
        Returns:
            Dict: Dizionario con i numeri raccolti
        """
        print("Cominciamo a sommare")
        first_number = self.validator.get_valid_number("🔢 Inserisci il PRIMO numero:")
        second_number = self.validator.get_valid_number("🔢 Inserisci il SECONDO numero:")
       
        return {
            "number1": first_number,
            "number2": second_number,
            "status": "numbers_collected"
        }

    @listen(collect_input)
    def prepare_calculation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara il calcolo con le informazioni raccolte.
        
        Args:
            inputs: Dizionario con i numeri dall'input precedente
            
        Returns:
            Dict: Dati preparati per il calcolo
        """
        num1 = inputs["number1"]
        num2 = inputs["number2"]
        
        print(f"\n🔧 PREPARAZIONE CALCOLO")
        print("-" * 30)
        print(f" Operazione: {num1} + {num2}")
           
        # Info crew
        crew_info = self.sum_crew.get_crew_info()
        print(f"👥 Crew: {crew_info['name']}")
        print(f"🔧 Tool: {', '.join(crew_info['tools'])}")
        
        return {
            "number1": num1,
            "number2": num2,
            "crew_ready": True,
            "status": "ready_for_calculation"
        }
    
    @listen(prepare_calculation)
    def execute_crew_calculation(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue il calcolo usando CrewAI crew.
        
        Args:
            inputs: Dati preparati per il calcolo
            
        Returns:
            Dict: Risultato del calcolo
        """
        num1 = inputs["number1"]
        num2 = inputs["number2"]
        
        print(f"\n ESECUZIONE CREWAI")
        print("-" * 25)
        print("CrewAI sta elaborando il calcolo...")
        
        try:
            # Kickoff della crew con inputs
            crew_inputs = {
                'number1': num1,
                'number2': num2
            }
            
            result = self.sum_crew.crew().kickoff(inputs=crew_inputs)
            
            return {
                "number1": num1,
                "number2": num2,
                "result": str(result),
                "success": True,
                "status": "calculation_completed"
            }
            
        except Exception as e:
            return {
                "number1": num1,
                "number2": num2,
                "error": str(e),
                "success": False,
                "status": "calculation_failed"
            }
    
    @listen(execute_crew_calculation)
    def display_final_result(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mostra il risultato finale all'utente.
        
        Args:
            inputs: Risultato del calcolo
            
        Returns:
            Dict: Status finale
        """
        num1 = inputs["number1"]
        num2 = inputs["number2"]
        
        print("\n" + "=" * 60)
        print(" RISULTATO FINALE")
        print("=" * 60)
        print(f"📥 Input utente: {num1} + {num2}")
        
        if inputs["success"]:
            print(f"🤖 CrewAI output: {inputs['result']}")
            print("=" * 60)
            print("✨ Operazione completata con successo!")
        else:
            print(f"❌ Errore: {inputs['error']}")
            print("=" * 60)
            print("🔧 Si è verificato un problema durante il calcolo.")
        
        print("👋 Grazie per aver usato Sum Flow!")
        
        return {
            "flow_completed": True,
            "success": inputs["success"]
        }

def kickoff():
    """
    Entry point dell'applicazione Sum Flow avanzata.
    """
    try:
        print(" Avvio SumFlow con sistema completo")
        print("=" * 50)
        
        # Inizializza il flow
        flow = SumFlowMain()
        
        # Avvia il flow completo
        result = flow.kickoff()
        
        print("\n" + "=" * 50)
        print("🏁 Flusso completato!")
        print(f"Risultato: {result}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Operazione interrotta dall'utente.")
        print("👋 Arrivederci!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Errore critico: {str(e)}")
        sys.exit(1)

def plot():
    """
    Genera il plot del flow per visualizzazione.
    """
    flow = SumFlowMain()
    flow.plot()

if __name__ == "__main__":
    kickoff()