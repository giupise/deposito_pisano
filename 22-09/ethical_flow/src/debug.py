import os
import litellm
from dotenv import load_dotenv

# Carica .env
load_dotenv()

print("=== DEBUG CONFIGURAZIONE AZURE ===\n")

# 1. Verifica variabili d'ambiente
print("1. VARIABILI D'AMBIENTE:")
print(f"   AZURE_API_KEY: {'✓ Presente' if os.getenv('AZURE_API_KEY') else '✗ Mancante'}")
print(f"   AZURE_API_BASE: {os.getenv('AZURE_API_BASE', '✗ Mancante')}")
print(f"   AZURE_API_VERSION: {os.getenv('AZURE_API_VERSION', '✗ Mancante')}")
print(f"   OPENAI_API_KEY: {'✓ Presente' if os.getenv('OPENAI_API_KEY') else '✗ Mancante'}")

# 2. Test con litellm
print("\n2. TEST LITELLM:")

# Abilita debug completo
os.environ['LITELLM_LOG'] = 'DEBUG'
litellm._turn_on_debug()

# Test 1: Con azure/gpt-4o
print("   Test 1 - Tentativo con azure/gpt-4o...")
try:
    response = litellm.completion(
        model="azure/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION", "2024-02-01")
    )
    print("✅ Connessione riuscita!")
    print(f"   Risposta: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Errore: {e}")

# Test 2: Prova con api version diversa
print("\n   Test 2 - Con API version 2024-02-01...")
try:
    response = litellm.completion(
        model="azure/gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version="2024-02-01"
    )
    print("✅ Funziona con API version 2024-02-01!")
except Exception as e:
    print(f"❌ Errore anche con 2024-02-01")

# Test 3: Test diretto con requests
print("\n   Test 3 - Test diretto con requests...")
import requests
headers = {
    "api-key": os.getenv("AZURE_API_KEY"),
    "Content-Type": "application/json"
}
url = f"{os.getenv('AZURE_API_BASE')}/openai/deployments/gpt-4o/chat/completions?api-version={os.getenv('AZURE_API_VERSION')}"
data = {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 10
}
try:
    resp = requests.post(url, headers=headers, json=data)
    print(f"   Status code: {resp.status_code}")
    print(f"   Response: {resp.text[:200]}...")
except Exception as e:
    print(f"❌ Errore requests: {e}")
    
# 3. Prova con parametri espliciti
print("\n3. URL COSTRUITO:")
base = os.getenv('AZURE_API_BASE', '')
if base and not base.endswith('/'):
    base += '/'
url = f"{base}openai/deployments/gpt-4o/chat/completions?api-version={os.getenv('AZURE_API_VERSION', '2024-02-01')}"
print(f"   {url}")

print("\n4. SUGGERIMENTI:")
print("   - Verifica che AZURE_API_BASE NON abbia '/' alla fine")
print("   - Verifica che il deployment 'gpt-4o' esista in Azure Portal")
print("   - Prova ad aggiungere 'AZURE_DEPLOYMENT_NAME=gpt-4o' nel .env")
print("   - Se usi un deployment con nome diverso, cambia 'azure/gpt-4o' con 'azure/tuo-nome-deployment'")