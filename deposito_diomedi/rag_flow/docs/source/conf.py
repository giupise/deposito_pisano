# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Rag Flow'
copyright = '2025, Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto'
author = 'Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',   # supporta Google e NumPy style
    'numpydoc'
]

numpydoc_show_class_members = False
templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# Mock imports for packages that might not be available during docs build
autodoc_mock_imports = [
    'crewai',
    'crewai.project',
    'crewai.agents.agent_builder.base_agent',
    'crewai_tools',
    'langchain_openai',
    'langchain',
    'langchain.chat_models',
    'langchain_community',
    'langchain_core',
    'langchain_core.output_parsers',
    'langchain_core.runnables',
    'pydantic',
    'ragas',
    'ragas.metrics',
    'faiss',
    'streamlit',
    'litellm',
    'sentence_transformers',
    'numpy',
    'pandas',
    'qdrant_client',
    'qdrant_client.models',
    'qdrant_client.http.models'
]