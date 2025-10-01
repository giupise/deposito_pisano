#!/usr/bin/env python3
"""
Test script per validare la configurazione Azure OpenAI
Esegui questo script per verificare se tutte le configurazioni sono corrette
prima di avviare il flusso CrewAI.

Usage: python test_azure_config.py
"""

import os
import sys
from dotenv import load_dotenv

def print_header(title):
    """Stampa un header formattato"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step_num, title):
    """Stampa il numero dello step"""
    print(f"\n{step_num}. {title}")
    print("-" * 40)

def test_environment_variables():
    """Test delle variabili d'ambiente"""
    print_step(1, "CHECKING ENVIRONMENT VARIABLES")
    
    # Lista delle variabili richieste
    required_vars = {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI API Key",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAI Endpoint",
        "AZURE_OPENAI_API_VERSION": "Azure OpenAI API Version",
        "AZURE_OPENAI_CHAT_DEPLOYMENT": "Chat Deployment Name",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT": "Embeddings Deployment Name",
        "SERPER_API_KEY": "Serper API Key (for web search)"
    }
    
    missing_vars = []
    found_vars = {}
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value or value.strip() == "" or value == "key":
            missing_vars.append((var, description))
            print(f"❌ {var}: NOT SET or placeholder")
        else:
            found_vars[var] = value
            # Mostra solo i primi e ultimi caratteri per sicurezza
            if len(value) > 10:
                masked_value = f"{value[:4]}...{value[-4:]}"
            else:
                masked_value = "***"
            print(f" {var}: {masked_value}")
    
    if missing_vars:
        print(f"\n⚠️  Found {len(missing_vars)} missing/invalid variables:")
        for var, desc in missing_vars:
            print(f"   - {var}: {desc}")
        return False, found_vars
    
    print(f"\n🎉 All {len(required_vars)} environment variables are set!")
    return True, found_vars

def test_azure_openai_connection(config):
    """Test della connessione Azure OpenAI"""
    print_step(2, "TESTING AZURE OPENAI CONNECTION")
    
    try:
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=config.get("AZURE_OPENAI_API_KEY"),
            api_version=config.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=config.get("AZURE_OPENAI_ENDPOINT")
        )
        
        print(" Azure OpenAI client initialized successfully")
        return True, client
        
    except ImportError:
        print("❌ OpenAI library not installed. Run: pip install openai")
        return False, None
    except Exception as e:
        print(f"❌ Failed to initialize Azure OpenAI client: {e}")
        return False, None

def test_chat_deployment(client, deployment_name):
    """Test del deployment chat"""
    print_step(3, f"TESTING CHAT DEPLOYMENT: {deployment_name}")
    
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, this is a test!' in Italian."}
            ],
            max_tokens=50,
            temperature=0
        )
        
        answer = response.choices[0].message.content
        print(f" Chat deployment working! Response: {answer}")
        return True
        
    except Exception as e:
        print(f"❌ Chat deployment test failed: {e}")
        print("\n💡 Common issues:")
        print("   - Check deployment name is correct")
        print("   - Verify deployment is active in Azure")
        print("   - Check if you have quota available")
        return False

def test_embeddings_deployment(client, deployment_name):
    """Test del deployment embeddings"""
    print_step(4, f"TESTING EMBEDDINGS DEPLOYMENT: {deployment_name}")
    
    try:
        response = client.embeddings.create(
            model=deployment_name,
            input="This is a test sentence for embeddings."
        )
        
        embedding_length = len(response.data[0].embedding)
        print(f" Embeddings deployment working! Vector length: {embedding_length}")
        return True
        
    except Exception as e:
        print(f"❌ Embeddings deployment test failed: {e}")
        print("\n💡 Common issues:")
        print("   - Check embeddings deployment name is correct")
        print("   - Verify embeddings deployment is active in Azure")
        print("   - Check if you have quota available")
        return False

def test_serper_api(api_key):
    """Test dell'API Serper per web search"""
    print_step(5, "TESTING SERPER API (Web Search)")
    
    if not api_key or api_key == "key":
        print("⚠️  Serper API key not set - web search will not work")
        return False
    
    try:
        import requests
        
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        payload = {
            'q': 'test query',
            'num': 1
        }
        
        response = requests.post(
            'https://google.serper.dev/search',
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print(" Serper API working!")
            return True
        else:
            print(f"❌ Serper API error: {response.status_code}")
            return False
            
    except ImportError:
        print("❌ Requests library not installed. Run: pip install requests")
        return False
    except Exception as e:
        print(f"❌ Serper API test failed: {e}")
        return False

def main():
    """Main function per i test"""
    print_header("🧪 AZURE OPENAI CONFIGURATION TEST")
    print("This script will validate your Azure OpenAI setup for CrewAI")
    
    # Carica variabili d'ambiente
    env_loaded = load_dotenv()
    if env_loaded:
        print(" .env file loaded")
    else:
        print("⚠️  No .env file found - using system environment variables")
    
    # Test 1: Environment Variables
    env_ok, config = test_environment_variables()
    if not env_ok:
        print_header("❌ CONFIGURATION TEST FAILED")
        print("Please fix the missing environment variables in your .env file")
        return False
    
    # Test 2: Azure OpenAI Connection
    conn_ok, client = test_azure_openai_connection(config)
    if not conn_ok:
        print_header("❌ CONNECTION TEST FAILED")
        return False
    
    # Test 3: Chat Deployment
    chat_ok = test_chat_deployment(client, config["AZURE_OPENAI_CHAT_DEPLOYMENT"])
    
    # Test 4: Embeddings Deployment  
    embeddings_ok = test_embeddings_deployment(client, config["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"])
    
    # Test 5: Serper API (optional)
    serper_ok = test_serper_api(config.get("SERPER_API_KEY"))
    
    # Risultati finali
    print_header(" TEST RESULTS SUMMARY")
    
    results = [
        ("Environment Variables", env_ok),
        ("Azure OpenAI Connection", conn_ok), 
        ("Chat Deployment", chat_ok),
        ("Embeddings Deployment", embeddings_ok),
        ("Serper API (Web Search)", serper_ok)
    ]
    
    for test_name, result in results:
        status = " PASS" if result else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    all_critical_passed = env_ok and conn_ok and chat_ok and embeddings_ok
    
    if all_critical_passed:
        print_header("🎉 ALL TESTS PASSED!")
        print("Your Azure OpenAI configuration is ready for CrewAI!")
        if not serper_ok:
            print("⚠️  Note: Web search won't work without Serper API key")
        return True
    else:
        print_header("❌ SOME TESTS FAILED")
        print("Please fix the issues above before running your CrewAI flow")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)