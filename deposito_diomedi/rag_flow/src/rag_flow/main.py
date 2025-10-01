#!/usr/bin/env python
from pydantic import BaseModel
from crewai.flow import Flow, listen, start, router
from rag_flow.crews.bias_crew.bias_crew import BiasCrew
from rag_flow.crews.poem_crew.rag_crew import AeronauticRagCrew
from rag_flow.crews.web_crew.web_crew import WebCrew
from rag_flow.crews.doc_crew.doc_crew import DocCrew
import os
from langchain_openai import AzureChatOpenAI
from typing import Dict, Any
from opik import configure 
from opik.integrations.crewai import track_crewai 

configure(use_local=True) 

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["SSL_VERIFY"] = "false"
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["OTEL_SDK_DISABLED"] = "true"


# Track CrewAI runs with Opik
track_crewai(project_name="culinary_rag_flow") 

#load_dotenv()  # Carica le variabili d'ambiente dal file .env 
endpoint = os.getenv("AZURE_API_BASE")
key = os.getenv("AZURE_API_KEY")
deployment_name = os.getenv("MODEL")  # nome deployment modello completions
api_version=os.getenv("AZURE_API_VERSION", "2024-06-01")


class AeronauticRagState(BaseModel):
    """
    State model for Aeronautic RAG Flow execution.
    
    This Pydantic model manages the state data throughout the RAG flow execution,
    storing user inputs, intermediate results, and final aggregated outputs from
    different processing stages.
    
    Attributes
    ----------
    question_input : str
        User's input question about aeronautics (default: "")
    rag_result : str
        Result from RAG system analysis using local document knowledge base (default: "")
    web_result : str
        Result from web search and analysis using external sources (default: "")
    all_results : str
        Aggregated results combining RAG and web analysis outputs (default: "")
        
    Notes
    -----
    State persistence enables tracking of data flow between different crew executions
    and allows for comprehensive result aggregation and document generation.
    """
    question_input: str = ""
    rag_result: str = ""
    web_result: str = ""
    all_results: str = ""
    document: str = ""
    final_doc: str = ""
    rag_context: str = ""  # Nuovo campo per ctx

class AeronauticRagFlow(Flow[AeronauticRagState]):
    """
    Multi-stage RAG flow for comprehensive aeronautic question answering.
    
    This CrewAI Flow orchestrates a sophisticated question-answering pipeline that combines
    local knowledge base retrieval, web search capabilities, and document generation.
    The flow includes question validation, dual-source analysis, and result aggregation.
    
    Flow Architecture
    -----------------
    1. **Starting Procedure**: Initialize flow execution
    2. **Question Generation**: Capture user input for aeronautic queries
    3. **Question Analysis**: Validate aeronautic relevance using Azure OpenAI
    4. **RAG Analysis**: Local document-based knowledge retrieval and answer generation
    5. **Web Analysis**: External web search for complementary information
    6. **Result Aggregation**: Combine and synthesize all findings into comprehensive documentation
    7. **Plot Generation**: Visualize the flow execution graph
    
    Crew Integration
    ----------------
    - **AeronauticRagCrew**: Local knowledge base querying with FAISS vector store
    - **WebCrew**: Web search and content analysis using SerperDev API
    - **DocCrew**: Professional markdown document generation and synthesis
    
    State Management
    ----------------
    Uses AeronauticRagState for persistent data flow between stages, enabling
    comprehensive result tracking and aggregation across multiple crew executions.
    
    Routing Logic
    -------------
    Implements intelligent routing based on question relevance:
    - 'success': Question is aeronautic-relevant, proceed with full analysis
    - 'retry': Question lacks aeronautic context, restart question capture
    
    Notes
    -----
    - Requires Azure OpenAI credentials for question validation
    - Integrates with local FAISS knowledge base and external web sources
    - Produces comprehensive markdown documentation with source citations
    - Includes flow visualization capabilities for pipeline monitoring
    """

    @start('retry')
    def starting_procedure(self):
        """
        Initialize the Aeronautic RAG Flow execution.
        
        Entry point for the flow that sets up the initial state and begins
        the question-answering pipeline. Configured to retry on validation failure.
        
        Notes
        -----
        The 'retry' parameter enables automatic restart when question validation
        determines that the input is not relevant to aeronautics.
        """

    @listen(starting_procedure)
    def generate_question(self):
        """
        Capture user input for aeronautic question processing.
        
        Interactive step that prompts the user to enter their aeronautic-related
        question and stores it in the flow state for subsequent processing stages.
        
        State Updates
        -------------
        Updates self.state.question_input with the user's entered question.
        
        Notes
        -----
        This method uses interactive input() which requires console interaction.
        The captured question will be validated for aeronautic relevance in
        the next flow stage.
        """
       

    @router(generate_question)
    def aeronautic_question_analysis(self):
        """
        Validate question relevance to aeronautics using Azure OpenAI.
        
        Analyzes the user's question to determine if it's relevant to aeronautics
        using Azure OpenAI GPT-4o model. Routes the flow based on validation results
        to ensure only aeronautic questions proceed to full analysis.
        
        Returns
        -------
        str
            Routing decision:
            - "success": Question is aeronautic-relevant, proceed with analysis
            - "retry": Question is not aeronautic-relevant, restart question capture
            
        LLM Configuration
        ----------------
        - Model: Azure OpenAI GPT-4o
        - Temperature: 0 (deterministic responses)
        - Max Retries: 2 (robust error handling)
        - API Version: From environment variable AZURE_API_VERSION
        
        Validation Logic
        ---------------
        Uses a system prompt defining the AI as an aeronautics expert and asks
        for binary True/False validation of question relevance. Response parsing
        is case-insensitive and searches for 'true' substring.
        
        Environment Dependencies
        -----------------------
        Requires AZURE_API_BASE, AZURE_API_KEY, MODEL, and AZURE_API_VERSION
        environment variables for Azure OpenAI service connection.
        
        Notes
        -----
        This routing mechanism ensures the RAG system only processes relevant
        queries, improving efficiency and result quality by filtering out
        off-topic questions early in the pipeline.
        """
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",  # or your deployment
        api_version=api_version,  # or your api version
        temperature=0,
        max_retries=2,
        ) 
        messages=[
                {"role": "system", "content": "You are an expert in aeronautics."},
                {"role": "user", "content": f"Is the following question relevant to aeronautics? Question: {self.state.question_input}. Answer only with 'True' or 'False'"}
            ]
        
        res = llm.invoke(messages)
        res = res.content.strip().lower()

        if 'true' in res:
            return "success"
        else:
            print("Question not relevant to aeronautics, please try again.")
            return "retry"


    @router("success")
    def ethic_question_analysis(self):
        """
        Validate question relevance to aeronautics using Azure OpenAI.
        
        Analyzes the user's question to determine if it's relevant to aeronautics
        using Azure OpenAI GPT-4o model. Routes the flow based on validation results
        to ensure only aeronautic questions proceed to full analysis.
        
        Returns
        -------
        str
            Routing decision:
            - "success": Question is aeronautic-relevant, proceed with analysis
            - "retry": Question is not aeronautic-relevant, restart question capture
            
        LLM Configuration
        ----------------
        - Model: Azure OpenAI GPT-4o
        - Temperature: 0 (deterministic responses)
        - Max Retries: 2 (robust error handling)
        - API Version: From environment variable AZURE_API_VERSION
        
        Validation Logic
        ---------------
        Uses a system prompt defining the AI as an aeronautics expert and asks
        for binary True/False validation of question relevance. Response parsing
        is case-insensitive and searches for 'true' substring.
        
        Environment Dependencies
        -----------------------
        Requires AZURE_API_BASE, AZURE_API_KEY, MODEL, and AZURE_API_VERSION
        environment variables for Azure OpenAI service connection.
        
        Notes
        -----
        This routing mechanism ensures the RAG system only processes relevant
        queries, improving efficiency and result quality by filtering out
        off-topic questions early in the pipeline.
        """
        llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",  # or your deployment
        api_version=api_version,  # or your api version
        temperature=0,
        max_retries=2,
        # other params...
        ) 
        messages=[{"role": "system", "content":
                """You are an ethical AI expert specialized in content moderation. 
                Your role is to evaluate if questions are appropriate and ethical.
                Consider a question ETHICAL if it:
                - Seeks legitimate information
                - Has educational or professional purpose
                - Does not promote harm, violence, or illegal activities
                - Does not involve personal attacks or hate speech
                Consider a question UNETHICAL if it:
                - Requests harmful, dangerous, or illegal information
                - Contains hate speech, discrimination, or personal attacks
                - Aims to manipulate, deceive, or cause harm
                - Violates privacy or confidentiality
                Be permissive with legitimate academic, technical, or professional questions."""
                },
                {"role": "user", "content": f"Is the following question ethical or harmful? Question: {self.state.question_input}. Answer only with 'True' or 'False'"}
            ]
        
        res = llm.invoke(messages)
        res = res.content.strip().lower()

        if 'true' in res:
            return "success-ethical"
        else:
            print("Question not ethical, please try again.")
            return "retry"

    @listen("success-ethical")
    def rag_analysis(self):
        """
        Execute RAG-based analysis using local aeronautic knowledge base.
        
        Processes the validated aeronautic question through the AeronauticRagCrew,
        which uses FAISS vector store and Azure OpenAI embeddings to retrieve
        relevant context from local documents and generate comprehensive answers.
        
        Crew Execution
        -------------
        - Crew: AeronauticRagCrew (rag_expert agent with rag_system tool)
        - Input: User's validated aeronautic question
        - Processing: Vector similarity search + context-aware answer generation
        - Output: RAG-generated response with source citations
        
        State Updates
        -------------
        Updates self.state.rag_result with the raw output from the RAG crew execution.
        
        Knowledge Base
        --------------
        Utilizes local FAISS vector store containing aeronautic documentation,
        technical manuals, and domain-specific knowledge for accurate,
        context-grounded responses.
        
        Answer Quality
        --------------
        - Source citations for transparency and verification
        - Anti-hallucination safeguards through context-only responses
        - Technical accuracy through domain-specific knowledge base
        - RAGAS evaluation metrics for quality assessment
        
        Notes
        -----
        This is the primary knowledge retrieval stage that leverages local
        expertise. Results are complemented by web analysis in the next stage
        for comprehensive coverage.
        """
        aero_crew = AeronauticRagCrew().crew()
        result = (
            aero_crew.kickoff(inputs={"question": self.state.question_input,
                             })
        )
        with open("output/last_context.txt", "r", encoding="utf-8") as f:
            CONTEXT = f.read()
        self.state.rag_result = result.raw
        return {
            "aero_crew": aero_crew,
            "rag_context": CONTEXT,
            "rag_result": result.raw,
            "question": self.state.question_input
        }
    
    @listen(rag_analysis)
    def web_analysis(self, payload:Dict[str, Any]):
        """
        Execute web-based analysis to complement local knowledge base.
        
        Processes the aeronautic question through WebCrew to gather external
        information from web sources, providing broader context and current
        information that may not be available in local documentation.
        
        Crew Execution
        -------------
        - Crew: WebCrew (web_analyst agent with SerperDevTool)
        - Input: User's validated aeronautic question
        - Processing: Web search + content extraction + analysis
        - Output: Structured summary of web search findings
        
        State Updates
        -------------
        Updates self.state.web_result with the raw output from the web crew execution.
        
        Search Strategy
        ---------------
        - Uses SerperDevTool for reliable web search results
        - Analyzes and summarizes relevant web content
        - Filters results for aeronautic relevance and quality
        - Provides structured insights from multiple web sources
        
        Content Enhancement
        -------------------
        - Complements local knowledge with current information
        - Provides multiple perspectives on aeronautic topics
        - Includes recent developments and industry updates
        - Validates and cross-references local knowledge base findings
        
        Notes
        -----
        Web analysis results are combined with RAG analysis in the aggregation
        stage to provide comprehensive, multi-source answers with both local
        expertise and current external information.
        """
        web_crew = WebCrew().crew()
        result = (
            web_crew.kickoff(inputs={"question": self.state.question_input,
                             })
        )
        self.state.web_result = result.raw
        payload['web_result'] = result.raw
        payload['web_crew'] = web_crew
        return payload
        # Here you can add more processing of the rag_result if needed
    
    @listen(web_analysis)
    def aggregate_results(self, payload:Dict[str, Any]):
        """
        Aggregate and synthesize results from RAG and web analysis.
        
        Combines outputs from both local knowledge base (RAG) and external web sources
        into a comprehensive aggregated result, then processes this through DocCrew
        for professional document generation and final synthesis.
        
        Aggregation Process
        -------------------
        1. Combines RAG result and web result into structured format
        2. Updates flow state with aggregated content
        3. Passes aggregated results to DocCrew for document generation
        4. Produces final comprehensive markdown documentation
        
        State Updates
        -------------
        Updates self.state.all_results with combined RAG and web analysis outputs
        formatted for document generation processing.
        
        Crew Execution
        --------------
        - Crew: DocCrew (doc_redactor agent)
        - Input: Aggregated results from RAG and web analysis
        - Processing: Document structuring + markdown generation
        - Output: Professional comprehensive documentation with proper formatting
        
        Document Features
        -----------------
        - Structured markdown format for readability
        - Integration of multiple information sources
        - Professional presentation and organization
        - Source attribution and cross-referencing
        - Technical accuracy and completeness
        
        Output Format
        -------------
        Generated document includes:
        - Executive summary of findings
        - Detailed analysis from local knowledge base
        - Complementary insights from web sources
        - Source citations and references
        - Professional formatting and structure
        
        Notes
        -----
        This stage represents the culmination of the multi-source analysis,
        producing a comprehensive, well-structured document that combines
        the best of local expertise and current external information.
        """
        aggregated = f"RAG Result: {self.state.rag_result}\n\nWeb Result: {self.state.web_result}"
        self.state.all_results = aggregated
        doc_crew = DocCrew().crew()
        result = (
            doc_crew
            .kickoff(inputs={"paper": aggregated,
                             })
        )
        self.state.document = result.raw
        payload['doc_context'] = aggregated
        payload['doc_result'] = result.raw
        payload['doc_crew'] = doc_crew
        return payload
    
    @listen(aggregate_results)
    def bias_check(self, payload:Dict[str, Any]):
        """
        Execute bias checking on the generated document.
        
        Processes the final aggregated document through BiasCrew to identify
        and mitigate any potential biases, ensuring the content is accurate,
        balanced, and ethically sound.
        
        Crew Execution
        -------------
        - Crew: BiasCrew (bias_checker agent)
        - Input: Generated document from DocCrew
        - Processing: Bias analysis + content redaction
        - Output: Bias-checked document in markdown format
        
        State Updates
        -------------
        Updates self.state.all_results with the bias-checked document output.
        
        Bias Checking Features
        ----------------------
        - Analyzes content for potential biases in tone, accuracy, and representation
        - Provides a balanced perspective by identifying and mitigating biases
        - Ensures ethical standards are maintained in the documentation
        - Outputs a clean, bias-free markdown document
        
        Output Format
        -------------
        The bias-checked document includes:
        - All original content with identified biases addressed
        - Professional formatting and structure maintained
        - No additional text or explanations outside of the redacted content
        
        Notes
        -----
        This stage enhances the integrity of the final document by ensuring
        that all content is free from biases, promoting accuracy and ethical
        standards in aeronautic documentation.
        """
        bias_crew = BiasCrew().crew()
        result = (
            bias_crew
            .kickoff(inputs={"document": self.state.document,
                             })
        )
        self.state.final_doc = result.raw
        payload['bias_context'] = self.state.document
        payload['bias_result'] = result.raw
        payload['bias_crew'] = bias_crew
        return payload
    
    @listen(bias_check)
    def plot_generation(self, payload:Dict[str, Any]):
        """
        Generate and display flow execution visualization.
        
        Creates a visual representation of the flow execution graph showing
        the complete pipeline from question input through final document generation.
        Useful for monitoring, debugging, and understanding the flow architecture.
        
        Visualization Features
        ----------------------
        - Complete flow graph with all stages and connections
        - Node representations for each processing step
        - Edge connections showing data flow and dependencies
        - Routing decisions and conditional paths
        - State transitions and crew executions
        
        Use Cases
        ---------
        - Pipeline monitoring and debugging
        - Flow architecture documentation
        - Performance analysis and optimization
        - Educational and presentation purposes
        - System maintenance and troubleshooting
        
        Notes
        -----
        This final stage provides visual feedback on the complete flow execution,
        enabling users to understand the processing pipeline and verify correct
        routing and stage execution.
        """
        self.plot()
        return payload




def kickoff():
    """
    Initialize and execute the Aeronautic RAG Flow.
    
    Entry point function that creates an instance of AeronauticRagFlow
    and starts the complete question-answering pipeline execution.
    
    Flow Execution
    --------------
    Triggers the complete multi-stage pipeline:
    1. Question capture and validation
    2. RAG-based local knowledge analysis
    3. Web-based external information gathering
    4. Result aggregation and document generation
    5. Flow visualization and monitoring
    
    Notes
    -----
    This function is the main entry point for interactive execution
    of the aeronautic question-answering system.
    """
    aeronautic_rag_flow = AeronauticRagFlow()
    question = input("Inserisci la tua domanda aeronautica: ")
    aeronautic_rag_flow.state.question_input = question
    aeronautic_rag_flow.kickoff()


def plot():
    """
    Generate and display the flow architecture visualization.
    
    Creates a visual representation of the AeronauticRagFlow pipeline
    without executing the flow, useful for documentation and architecture
    review purposes.
    
    Visualization Output
    --------------------
    - Complete flow graph showing all stages and connections
    - Node representations for each processing step
    - Routing logic and conditional paths
    - State management and data flow
    
    Use Cases
    ---------
    - Architecture documentation and review
    - System design presentations
    - Flow optimization and debugging
    - Educational and training purposes
    
    Notes
    -----
    This function provides flow visualization without execution,
    enabling architecture review and documentation without processing
    actual questions through the pipeline.
    """
    aeronautic_rag_flow = AeronauticRagFlow()
    aeronautic_rag_flow.plot()


if __name__ == "__main__":
    kickoff()
