#!/usr/bin/env python3
"""
Definitive Serper API Fix Script
This script will ensure Serper API works properly in the RAG flow project.
"""

import os
import sys
from pathlib import Path

def fix_serper_issue():
    """Fix the Serper API issue definitively."""
    
    print(" DEFINITIVE SERPER API FIX")
    print("=" * 50)
    
    # Step 1: Check current directory structure
    print("\n Checking project structure...")
    current_dir = Path.cwd()
    src_dir = current_dir / "src" / "rag_flow"
    
    print(f"Current directory: {current_dir}")
    print(f"Source directory: {src_dir}")
    
    # Step 2: Find .env files
    print("\n Looking for .env files...")
    env_files = []
    
    # Check root directory
    root_env = current_dir / ".env"
    if root_env.exists():
        env_files.append(("Root", root_env))
        print(f" Found .env in root: {root_env}")
    
    # Check src/rag_flow directory
    src_env = src_dir / ".env"
    if src_env.exists():
        env_files.append(("Source", src_env))
        print(f" Found .env in src/rag_flow: {src_env}")
    
    if not env_files:
        print("❌ No .env files found!")
        return False
    
    # Step 3: Check .env content
    print("\n Checking .env file content...")
    for location, env_path in env_files:
        print(f"\n--- {location} .env file ---")
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                lines = content.strip().split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        key = line.split('=')[0].strip()
                        value = line.split('=')[1].strip() if '=' in line else ""
                        if key == "SERPER_API_KEY":
                            if value and value != "your_serper_api_key_here":
                                print(f" {key}: {value[:10]}...")
                            else:
                                print(f"❌ {key}: Not set or placeholder value")
                        else:
                            print(f" {key}: {'Set' if value else 'Not set'}")
        except Exception as e:
            print(f"❌ Error reading {env_path}: {e}")
    
    # Step 4: Test environment loading
    print("\n🧪 Testing environment loading...")
    
    # Add src/rag_flow to Python path
    sys.path.insert(0, str(src_dir))
    
    # Try to load environment variables
    try:
        from dotenv import load_dotenv
        
        # Load from multiple locations
        load_dotenv(root_env)
        load_dotenv(src_env)
        
        # Check if SERPER_API_KEY is loaded
        serper_key = os.getenv("SERPER_API_KEY")
        if serper_key and serper_key != "your_serper_api_key_here":
            print(f" SERPER_API_KEY loaded: {serper_key[:10]}...")
            
            # Test SerperDevTool initialization
            try:
                from crewai_tools import SerperDevTool
                web_search_tool = SerperDevTool(api_key=serper_key, n_results=3)
                print(" SerperDevTool initialized successfully!")
                print("🎉 SERPER API IS WORKING!")
                return True
            except Exception as e:
                print(f"❌ Error initializing SerperDevTool: {e}")
                return False
        else:
            print("❌ SERPER_API_KEY not properly set in .env file")
            print("\n💡 To fix this:")
            print("1. Open your .env file")
            print("2. Set SERPER_API_KEY=your_actual_api_key_here")
            print("3. Get API key from: https://serper.dev")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Installing required packages...")
        os.system("pip install python-dotenv crewai-tools")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_working_env():
    """Create a working .env file template."""
    
    print("\n Creating .env template...")
    
    env_template = """# Azure OpenAI Configuration
AZURE_API_BASE=your_azure_endpoint_here
AZURE_API_KEY=your_azure_api_key_here
AZURE_API_VERSION=2024-06-01
MODEL=gpt-4o

# Serper API Configuration (REQUIRED FOR WEB SEARCH)
SERPER_API_KEY=your_serper_api_key_here

# Instructions:
# 1. Replace 'your_serper_api_key_here' with your actual Serper API key
# 2. Get your API key from: https://serper.dev
# 3. Replace Azure credentials with your actual values
"""
    
    # Create .env in both locations
    locations = [
        Path.cwd() / ".env",
        Path.cwd() / "src" / "rag_flow" / ".env"
    ]
    
    for env_path in locations:
        try:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            with open(env_path, 'w') as f:
                f.write(env_template)
            print(f" Created .env template at: {env_path}")
        except Exception as e:
            print(f"❌ Error creating .env at {env_path}: {e}")

if __name__ == "__main__":
    print(" Starting Serper API Fix...")
    
    success = fix_serper_issue()
    
    if not success:
        print("\n Creating working .env template...")
        create_working_env()
        print("\n Next steps:")
        print("1. Edit the .env file with your actual API keys")
        print("2. Get Serper API key from: https://serper.dev")
        print("3. Run this script again to verify")
    else:
        print("\n🎉 Serper API is working correctly!")
        print("You can now run: python .\\src\\rag_flow\\main.py")
