from openai import AzureOpenAI

from dotenv import load_dotenv
import os

# carica variabili dal file .env
load_dotenv()

# leggiamo le variabili dall'.env
endpoint =  os.getenv("ENDPOINT")
model_name =  os.getenv("MODEL_NAME")
deployment =  os.getenv("DEPLOYMENT")
subscription_key = os.getenv("SUBSCRIPTION_KEY")
api_version =  os.getenv("API_VERSION")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

resp = client.embeddings.create(
    input=[
        "Cagliari, capoluogo della Sardegna, è una città ricca di storia e cultura, famosa per il suo affascinante quartiere storico di Castello e le vedute mozzafiato sul Golfo degli Angeli.",
        "La spiaggia del Poetto, con i suoi chilometri di sabbia bianca e mare cristallino, è uno dei luoghi simbolo di Cagliari, amata da residenti e turisti per la sua atmosfera rilassante."
    ],
    model=model_name 
)

# estrazione vettori
vectors = [d.embedding for d in resp.data]
print(f"Numero di vettori/frasi: {len(vectors)}")
print(f"Dimensione embedding: {len(vectors[0])}")