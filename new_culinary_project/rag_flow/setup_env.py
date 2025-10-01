#!/usr/bin/env python3
"""
Environment Setup Script for RAG Flow Project

This script helps you set up the required environment variables for the RAG flow project.
"""

import os

def setup_environment():
    """Set up environment variables for the RAG flow project."""
    
    print(" RAG Flow Environment Setup")
    print("=" * 50)
    
    # Check current environment variables
    print("\n Current Environment Variables:")
    azure_base = os.getenv("AZURE_API_BASE")
    azure_key = os.getenv("AZURE_API_KEY") 
    serper_key = os.getenv("SERPER_API_KEY")
    
    print(f"AZURE_API_BASE: {' Set' if azure_base else '❌ Not set'}")
    print(f"AZURE_API_KEY: {' Set' if azure_key else '❌ Not set'}")
    print(f"SERPER_API_KEY: {' Set' if serper_key else '❌ Not set'}")
    
    print("\n Required Environment Variables:")
    print("1. AZURE_API_BASE - Your Azure OpenAI endpoint")
    print("2. AZURE_API_KEY - Your Azure OpenAI API key")
    print("3. AZURE_API_VERSION - API version (default: 2024-06-01)")
    print("4. MODEL - Model name (default: gpt-4o)")
    print("5. SERPER_API_KEY - Your Serper API key for web search")
    
    print("\n🌐 How to get API keys:")
    print("• Azure OpenAI: https://portal.azure.com")
    print("• Serper API: https://serper.dev")
    
    print("\n💡 To set environment variables:")
    print("Windows PowerShell:")
    print('$env:SERPER_API_KEY="your_serper_api_key_here"')
    print('$env:AZURE_API_BASE="your_azure_endpoint_here"')
    print('$env:AZURE_API_KEY="your_azure_api_key_here"')
    
    print("\nWindows Command Prompt:")
    print('set SERPER_API_KEY=your_serper_api_key_here')
    print('set AZURE_API_BASE=your_azure_endpoint_here')
    print('set AZURE_API_KEY=your_azure_api_key_here')
    
    print("\n🐧 Linux/Mac:")
    print('export SERPER_API_KEY="your_serper_api_key_here"')
    print('export AZURE_API_BASE="your_azure_endpoint_here"')
    print('export AZURE_API_KEY="your_azure_api_key_here"')

if __name__ == "__main__":
    setup_environment()
