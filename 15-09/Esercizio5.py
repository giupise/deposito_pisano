from openai import AzureOpenAI

endpoint = "https://aiacademygp.cognitiveservices.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2023-05-15"
model_name = "text-embedding-ada-002"
deployment = "text-embedding-ada-002"

subscription_key = ""
api_version = "2024-12-01-preview"

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

