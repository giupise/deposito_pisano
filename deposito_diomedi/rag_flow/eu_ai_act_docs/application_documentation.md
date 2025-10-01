# Rag Flow — Application Documentation 

## Application Information

<br>**Repository**: [rag_flow](https://github.com/marco-diomedi-ey/deposito_diomedi/tree/main/rag_flow)
<br>**Project Name**: Rag Flow
<br>**Application Owner**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto
<br>**Document Version**: 2025-09-25 v1.0
<br>**Reviewers**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto

## Key Links

* [rag_flow](https://github.com/marco-diomedi-ey/deposito_diomedi/tree/main/rag_flow) project folder in workspace
* [Deployment Pipeline]`crewai run` CLI; entrypoints in `pyproject.toml` scripts
* [API]`rag_flow` runs as a CLI flow (no HTTP API exposed)
* [Cloud Account]`Azure OpenAI` (credentials via env vars)
* [Project Management Board]`N/A`
* [Application Architecture]`architettura_crewai.md` and `crewai_flow.html`

## General Information 

<div style="color: gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1, 2, 3
<!-- info: this section covers the AI Act requirement of a description of the intended purpose, version and provider, relevant versions and updates. In Article 11, 2(d) a datasheet is required which describes all training methodologies and techniques as well as the characteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected, labelling procedures conducted, and data cleaning methodologies deployed. -->
<p></p>
</div>


**Purpose and Intended Use**:
    
* The system orchestrates a multi-stage Retrieval-Augmented Generation (RAG) flow for aeronautic question answering, combining local FAISS-based retrieval with web research and markdown document generation.
* Sector: technical documentation and knowledge assistance in aeronautics.
* Problem: provide accurate, source-grounded answers and structured reports to user questions leveraging local indexed docs and current web insights.
* Target users: engineers, analysts, and documentation specialists.
* KPIs: RAGAS quality metrics (faithfulness, answer relevance, context recall/precision where applicable).
* Ethical and regulatory considerations: transparency through citations, anti-hallucination via context-only generation, clear scope limitation to aeronautics via router validation.
* Prohibited uses: decisions requiring certified compliance or safety-critical control without human review; non-aeronautics topics are filtered by the router.
* Operational environment: executed via CLI on a workstation/server; depends on Azure OpenAI credentials and local FAISS indices; optional web access for research.


## Risk classification

<div style="color: gray">
Prohibited Risk: EU AI Act Chapter II <a href="https://artificialintelligenceact.eu/article/5/" style="color:blue; text-decoration:underline">Article 5</a>
<br>High-Risk: EU AI Act Chapter III, Section 1 <a href="https://artificialintelligenceact.eu/article/6/" style="color:blue; text-decoration:underline">Article 6</a>, <a href="https://artificialintelligenceact.eu/article/7/" style="color:blue; text-decoration:underline">Article 7</a>  
<br>Limited Risk: Chapter IV <a href="https://artificialintelligenceact.eu/article/50/" style="color:blue; text-decoration:underline">Article 50</a>
<p></p>
</div>

<!--info: The AI Act classifies AI systems into four different risk categories. The EU AI Act categorizes AI systems into four risk levels: unacceptable, high, limited, and minimal risk, each with corresponding regulatory requirements.  
Unacceptable risk (Chapter II, Article 5) includes systems that pose a clear threat to safety or fundamental rights (e.g. social scoring, recidivism scoring) and are banned.  
High-risk systems are delineated in Chapter III, Section 1, Articles 6 and 7, including AI used in sensitive domains like healthcare, law enforcement, education, employment, and critical infrastructure. These must meet strict requirements and conduct conformity assessment practices, including risk management, transparency, and human oversight.  
Limited-risk systems, delineated in Chapter IV Article 50, such as chatbots, must meet transparency obligations (e.g. disclosing AI use).  
Minimal-risk systems, like spam filters or AI in video games, face no specific requirements. -->

* Limited/Minimal risk: The application is a documentation assistant and research tool with transparency measures and human-in-the-loop. It does not perform automated actuation or high-stakes decisioning. It enforces topic gating (aeronautics) and provides sources for verification.
   
## Application Functionality 

<div style="color: gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>, paragraph 1, 2, 3
<!-- Info: this section covers the delineation of the general purpose of the system required in Article 1, with a focus on defining what the system should do and how it should work.-->
<p></p>
</div>


* **Instructions for use for deployers**: <div style="color: gray">(EU AI Act <a href="https://artificialintelligenceact.eu/article/13/" style="color:blue; text-decoration:underline">Article 13</a>)</div>
  * Set environment variables: `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`, `MODEL`, optionally `SERPER_API_KEY`.
  * Ensure FAISS indices exist under `src/rag_flow/tools/refactored_faiss_code/faiss_db/*` or configured location.
  * Install deps via `uv sync` or `pip install -e .` then run `crewai run` or `python -m rag_flow.main`.
* **Model Capabilities**:
  * Can: validate aeronautic relevance; retrieve local context via FAISS; run web search summaries; synthesize markdown reports; cite sources.
  * Cannot: make safety-critical decisions; guarantee exhaustiveness of sources; operate without credentials or indices; answer non-aeronautic queries (routed to retry).
  * Languages: prompts in English/Italian supported by Azure GPT-4o; documents primarily markdown text.
* **Input Data Requirements**:
  * Input: free-text question via console prompt.
  * Valid examples: "Explain the role of lift in aircraft wing design"; Invalid: off-topic queries (e.g., cooking recipes) will be rejected.
* **Output Explanation**:
  * Outputs include an aggregated markdown document combining RAG and web analysis. Interpret as draft report with citations to verify.
  * Uncertainty: no calibrated confidence scores; reliability inferred from citation presence and RAGAS metrics where applied.
* **System Architecture Overview**:
  * Flow orchestrated in `rag_flow.main:AeronauticRagFlow` with state `AeronauticRagState` and stages: start → generate_question → router (success/retry) → rag_analysis → web_analysis → aggregate_results → plot.
  * Crews: `AeronauticRagCrew` (RAG), `WebCrew` (Serper search), `DocCrew` (markdown synthesis). Config via `crews/*/config/agents.yaml` and `tasks.yaml`.
  * Tools: `rag_flow.tools.refactored_faiss_code.main:rag_system` (FAISS retrieval, RAG); `SerperDevTool` for web search.

## Models and Datasets

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2 (d)
<p></p>
</div>

<!--All information about models and datasets that are used in the application should be found in their respective dataset or model documentation.  The purpose here is mainly to provide links to those documentation. --> 
<!--In Article 11, 2 (d) a datasheet is required which describes all training methodologies and techniques as well as the charatcteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected labelling procedures conducted and data cleaning methodologies deployed -->

### Models

| Model                              | Link to Single Source of Truth | Description of Application Usage |
|------------------------------------|--------------------------------|----------------------------------|
| Azure OpenAI GPT-4o (chat)         | `AZURE_API_BASE` deployment     | Router validation of aeronautic relevance; generation within crews |
| OpenAI Embeddings (via langchain)  | Configured in tools             | Vectorization for FAISS-based retrieval (within `rag_system`) |

### Datasets

| Dataset/KB                      | Link to Single Source of Truth | Description of Application Usage |
|---------------------------------|--------------------------------|----------------------------------|
| Local FAISS indices             | `src/rag_flow/tools/refactored_faiss_code/faiss_db/*` | Knowledge base for aeronautic retrieval (RAG) |
| Web search results (Serper API) | SerperDev API                   | Complementary, current web information for analysis |

## Deployment
    
* CLI execution on local/server environment with internet access for Azure OpenAI and optional Serper API; local filesystem for FAISS indices.
* Entrypoints: `crewai run`, or `rag_flow.main:kickoff` and `rag_flow.main:plot` scripts defined in `pyproject.toml`.

### Infrastructure and Environment Details

* **Cloud Setup**:
  * Azure OpenAI used for LLM; credentials via env vars.
  * No managed database; FAISS indices stored locally under repo.
  * Standard CPU is sufficient; GPU not required.
* **APIs**:
  * External: Azure OpenAI (key-based), SerperDev (API key) for web search.
  * Internal: no HTTP API; CLI flow.

## Integration with External Systems

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1 (b, c, d, g, h), 2 (a)
  <p></p>
</div>

* **Systems**:
  * Dependencies: `crewai`, `langchain`, `langchain-openai`, `faiss-cpu`, `duckduckgo-search` (optional), `pydantic`, `pandas`, `python-dotenv`, `requests`, `ragas`.
  * Data flow: user input → router (Azure GPT-4o) → RAG crew (`rag_system` + FAISS) → Web crew (Serper) → aggregation → Doc crew (markdown) → `output/*.md`.
  * Error handling: retries in router LLM (max_retries=2); flow restarts on non-aeronautic questions via `@router` returning `retry`.

## Deployment Plan

* **Infrastructure**:
  * Environments: local dev; can be containerized for staging/prod.
  * Scaling: N/A for CLI; schedule jobs if batch usage is desired.
  * Backup: version control FAISS indices and `output` artifacts as needed.
* **Integration Steps**:
  * Configure env vars; prepare FAISS indices; install dependencies; run `crewai run`.
  * Dependencies pinned in `pyproject.toml`; optional `uv.lock` available.
  * Rollback: revert environment or dependency versions; restore previous indices.
* **User Information**: executed from terminal with interactive input prompt.


## Lifecycle Management

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 6
  <p></p>
</div>
    
* Monitoring: console logs from flow execution; optional enrichment via `verbose=True` crews; store outputs for review.
* Versioning: `pyproject.toml` version; track FAISS index versions per folder.
* **Metrics**:
  * Application: N/A
  * Model: RAGAS metrics when evaluating outputs considering faithfulness, answer relevance, context recall/precision where applicable.
  * Infra: CPU/memory nominal for CLI; network latency to Azure.
* **Key Activities**:
  * Review generated markdowns; monitor router false-negative/positive rates.
  * Refresh indices and upgrade dependencies periodically.
* **Documentation Needs**:
  * Keep `architettura_crewai.md` updated; retain generated docs in `output/`.
  * Maintain simple CHANGELOG in repo.
**Maintenance of change logs**: new features, updates, deprecations, removals, bug fixes, security fixes.

### Risk Management System

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/9/" style="color:blue; text-decoration:underline">Article 9</a>
  <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>
  ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>
  <p></p>
</div>
<!--**Instructions:**  A thorough risk management system is mandated by the AI Act, especially for high-risk AI systems. This section documents the  proactive efforts to ensure the AI system operates safely and ethically. In general in this section you should document all the measures undertaken to make sure that a system operates safely on the market. Example: Consider a facial recognition system used for real-time law enforcement in public spaces. This is categorized as high-risk under the EU AI Act. If developers document the risk that the system might misidentify individuals—particularly among minority groups due to biased training data—they can plan for rigorous dataset audits, independent bias testing, and establish human oversight in decision-making. Without documenting this risk, the system might be deployed without safeguards, leading to wrongful detentions and legal liabilities. Systematic documentation ensures these issues are not only identified but addressed before harm occurs.-->


**Risk Assessment Methodology:** qualitative assessment inspired by ISO 31000; track risks in repo issues.

**Identified Risks:** 
* Hallucinations without context; stale or incomplete local KB; external web content quality; credential misconfiguration; off-topic queries.

**Potential Harmful Outcomes:** misleading answers if citations are weak; reliance on outdated docs; privacy risks if sensitive docs are indexed.

**Likelihood and Severity:** moderate likelihood, low-to-moderate severity due to human oversight and citations.

#### Risk Mitigation Measures

**Preventive Measures:** router validation for domain scope; citations-only generation; curated FAISS indices; API key management via env vars; retries in router.

**Protective Measures:** user-in-the-loop review; log and store outputs; disable web step if API unavailable; fallback to local RAG only.

## Testing and Validation (Accuracy, Robustness, Cybersecurity)

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/15/" style="color:blue; text-decoration:underline">Article 15</a>
  <p></p>
</div>

**Testing and Validation Procedures (Accuracy):** manual validation of sample aeronautic queries; compare outputs vs. source docs; use RAGAS scripts in `tools/refactored_faiss_code/ragas_scripts.py` when applicable.

**Performance Metrics:** RAGAS quality metrics (faithfulness, answer relevance, context recall/precision where applicable).

**Validation Results:** baseline runs generate structured markdown with cited sources; router filters off-topic queries effectively in tests.

**Measures for Accuracy:** maintain high-quality indexed docs; periodic index refresh; prompt constraints to use retrieved context.

  
### Accuracy throughout the lifecycle

**Data Quality and Management:** curate and validate documents before indexing; maintain provenance of sources and update schedules.

**Model Selection and Optimisation:** fixed vendor-managed LLM (Azure GPT-4o); tune prompts and retrieval parameters (k, similarity threshold) in `rag_system`.

**Feedback Mechanisms:** collect user feedback on generated docs; review failure cases and update indices/prompts accordingly.

### Robustness 

**Robustness Measures:**
* Input validation via router; graceful degradation if web or FAISS unavailable; retries; deterministic routing temperature.

**Scenario-Based Testing:**
* Test off-topic, ambiguous, and adversarial queries; simulate missing indices; validate fallback behaviors.
    

**Redundancy and Fail-Safes:**
* Fallback to local RAG if web search fails; allow re-run with corrected env vars.
    
**Uncertainty Estimation:**
* Use citation presence/density and agreement between RAG and web outputs as proxies.
    

### Cybersecurity 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2 (h)
  <p></p>
</div>

**Data Security:**
* Store API keys in environment variables; avoid committing secrets; restrict access to FAISS indices.

**Access Control:**
* Limit who can run the flow and who can modify indices; use OS/user-level permissions.

**Incident Response :**
* Revoke/rotate keys on suspicion; remove compromised indices; review logs and regenerate outputs if needed.


These measures include threat modelling, data security, adversarial robustness, secure development practices, access control, and incident response mechanisms.

Post-deployment monitoring, patch management, and forensic logging are crucial to maintaining ongoing cybersecurity compliance.

Documentation of all cybersecurity processes and incidents is mandatory to ensure accountability and regulatory conformity.

  

## Human Oversight 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>;; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2(e)
  <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/14/" style="color:blue; text-decoration:underline">Article 14</a>
  <p></p>
</div>

<!-- info: AI Act Article 11, paragraph 2(e) requirements: assessment of the human oversight measures needed in accordance with Article 14, including the assessment of the technical measures needed to facilitate the integration of the outputs of the AI systems by deployers. -->


**Human-in-the-Loop Mechanisms:** human reviews generated markdown; decides whether to accept, edit, or discard outputs; can re-run with refined queries.

**Override and Intervention Procedures:** stop the CLI at any stage; disable web step by omitting `SERPER_API_KEY`; adjust indices.

**User Instructions and Training:** follow `README.md` and inline console prompts; understand the scope limitation to aeronautics.

**Limitations and Constraints of the System:** relies on index coverage and web API availability; no calibrated confidence; on-topic constraint enforced by router.


## Incident Management
<!-- what happens when things go wrong. This part is particularly important to provide information on how incidents were dealth with and the processes put in place to minimize damage when things go wrong. -->
* **Common Issues**:
  * Missing env vars → set `AZURE_API_BASE`, `AZURE_API_KEY`, `MODEL`, `AZURE_API_VERSION`.
  * No web results → set `SERPER_API_KEY` or skip web step.
  * FAISS index not found → ensure indices under `tools/refactored_faiss_code/faiss_db/*`.
  * Router loops on retry → refine question to aeronautics domain.
* **Support Contact**:
  * Internal maintainers; upstream CrewAI docs and GitHub listed in `README.md`.


### Troubleshooting AI Application Deployment

This section outlines potential issues that can arise during the deployment of an AI application, along with their causes, resolutions, and best practices for mitigation.


#### Infrastructure-Level Issues

##### Insufficient Resources

* **Problem**: Inaccurate resource estimation for production workloads.
  * Unexpected spikes in user traffic can lead to insufficient resources such as compute, memory or storage that can lead to crashes and bad performance

* **Mitigation Strategy**:
  -  Enable autoscaling (e.g., Kubernetes Horizontal Pod Autoscaler).
<!-- - Monitor usage metrics and adjust resource allocation dynamically.
  - Implement rate-limiting for traffic spikes. -->


##### Network Failures

* **Problem**:  network bottlenecks can lead to inaccessible or experiences latency of the application.

* **Mitigation Strategy**:
 
  - Test network connectivity 
<!--  - Use content delivery networks (CDNs) or regional load balancers.
  - Ensure proper failover mechanisms.-->


##### Deployment Pipeline Failures

* **Problem**: pipeline fails to build, test, or deploy because of issues of compatibility between application code and infrastructure, environment variables or credentials misconfiguration.

* **Mitigation Strategy**: 

  - Roll back to the last stable build.
  - Fix pipeline scripts and use containerisation for environment consistency.
  - Enable verbose logging for error diagnostics.


#### Integration Problems

##### API Failures

* **Problem**: External APIs or internal services are unreachable due to network errors or authentication failures.

* **Mitigation Strategy**:

  - Implement retries with exponential backoff.
  - Validate API keys or tokens and refresh as needed.
  - Log and monitor API responses for debugging.

##### Data Format Mismatches

* **Problem**: Crashes or errors due to unexpected data formats such as changes in the schema of external data sources or missing data validation steps.

* **Mitigation Strategy**: 

  - Use schema validation tools (e.g., JSON schema validators).
  - Add versioning to APIs and validate inputs before processing.

#### Data Quality Problems

* **Problem**: Inaccurate or corrupt data leads to poor predictions.
* **Causes**:
  * No data validation or cleaning processes.
  * Inconsistent labelling in training datasets.

* **Mitigation Strategy**: 

- **Resolution**:
  - Automate data quality checks (e.g., Great Expectations framework).
  - Regularly audit and clean production data.


#### Model-Level Issues

##### Performance or Deployment Issues

* **Problem**: Incorrect or inconsistent results due to data drift or inadequate training data for the real world deployment domain. 

* **Mitigation Strategy**:


- **Resolution**:
  - Monitoring for data drift and retraining of the model as needed.
  - Regularly update the model


#### Safety and Security Issues

##### Unauthorised Access

* **Problem**: Sensitive data or APIs are exposed due to misconfigured authentication and authorization.

##### Data Breaches

* **Problem**: User or model data is compromised due to insecure storage or lack of monitoring and logging of data access. 

* **Mitigation Strategy**: 

- **Resolution**:
  - Use secure storage services (e.g., AWS KMS).
  <!-- - Implement auditing for data access and alerts for unusual activity.
  6.1. Delayed or Missing Data -->


#### Monitoring and Logging Failures

##### Missing or Incomplete Logs

* **Problem**: Lack of information to debug issues due to inefficient logging. Critical issues go unnoticed, or too many false positives occur by lack of implementation ofactionable information in alerts. 

* **Mitigation Strategy**: 



- **Resolution**:
  - Fine-tune alerting thresholds and prioritise critical alerts.
  <!-- - Use tools like Prometheus Alertmanager to manage and group alerts. -->


#### Recovery and Rollback

##### Rollback Mechanisms

* **Problem**: New deployment introduces critical errors.

* **Mitigation Strategy**: 


- **Resolution**:
  <!-- - Use blue-green or canary deployments to minimise impact. -->
  - Maintain backups of previous versions and configurations.

##### Disaster Recovery

* **Problem**: Complete system outage or data loss.

* **Mitigation Strategy**:


- **Resolution**:
  - Test and document disaster recovery plans.
  - Use automated backups and verify restore procedures.

### EU Declaration of conformity 

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/47/" style="color:blue; text-decoration:underline">Article 47</a>
  <p></p>
</div>

<!-- when applicable and certifications are available: it requires a systems name as well as the name and address of the provider; a statement that the EU declaration of conformity referred to in Article 47 is issued under the sole responsibility of the provider; a statement that the AI system is in conformity with this Regulation and, if applicable, with any other relevant Union law that provides for the issuing of the EU declaration of conformity referred to in Article 47, Where an AI system involves the processing of personal data;  a statement that that AI system complies with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, reference to the harmonised standards used or any other common specification in relation to which
conformity is declared; the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued; the place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.-->

### Standards applied

* Pydantic for data validation; CrewAI Flow patterns; LangChain integration practices.

## Documentation Metadata

### Template Version
Based on internal application documentation template.

### Documentation Authors
rag_flow engineers, AI/ML team (Owner/Contributors)
