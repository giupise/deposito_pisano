# Rag Flow — Model Documentation 
<!-- info: Replace with model name -->

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>, paragraph 1
    <!-- info:  
    The AI Act requires a description of  
    (a) the intended purpose, version, and provider,  
    (b) a description of how the system interacts with software and hardware,  
    (c) relevant versions and updates,  
    (d) all the forms in which the AI system is put into service
    The overview part should also include:  
    (e) the hardware on which the system is intended to run,  
    (f) whether the system is part of the safety component of a product,  
    (g) a basic description of the user interface, and  
    (h) the instructions for use for the deployers. 
    -->
    <p></p>
</div>

**Model Owner**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino
<br>**Document Version**: 2025-09-25 v1.0
<br>**Reviewers**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino

## Overview 

<div style="color:gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>, paragraph 1

<!-- info: This section enables all stakeholders to have a glimpse into the model selection, design, and development processes.  
You can use this section to provide transparency to users and high-level information to all relevant stakeholders.-->
<p></p>
</div>

### Model Type

**Model Type:** Retrieval-Augmented Generation (RAG) system orchestrated via CrewAI Flow (LLM: Azure OpenAI GPT-4o; Embeddings: Azure text-embedding-ada-002; Vector store: FAISS)

### Model Description 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 1(a)
    <p></p>
</div>

* Description
Answer aeronautics-related questions using local technical documentation and web sources, producing citation-backed answers and a final structured markdown report.
The project implements an aeronautics-focused RAG system named `AeronauticRagFlow`. It orchestrates multiple specialized crews (RAG, Web, Doc) using CrewAI Flow to answer questions: (1) validate query relevance to aeronautics with Azure OpenAI GPT-4o, (2) retrieve context from a local FAISS vector store built from aeronautic documents, optionally enriched by web search, and (3) synthesize results into a structured markdown document. The system emphasizes citation-based, context-grounded answers and evaluates quality with RAGAS metrics. Intended purpose: accurate, transparent Q&A and documentation generation in the aeronautics domain.

### Status 
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Status Date:** 2025-09-25

**Status:** Regularly Updated

### Relevant Links
<!-- info: User studies show document users find quick access to relevant artefacts like papers, model demos, etc..
very useful. -->

Example references:

* Repository: `deposito_diomedi/rag_flow at main · marco-diomedi-ey/deposito_diomedi` 
* Flow overview: `architettura_crewai.md`
* Tool docs: `docs/html/index.html`
* Flow entry points: `rag_flow.main:kickoff`, `rag_flow.main:plot`

### Developers

* **Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino**

### Owner
<!-- info: Remember to reference developers and owners emails. -->
* **Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino**

## Version Details and Artifacts 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1(c)
    <p></p>
</div>

<!-- scope: periscope -->
<!-- info: Provide details about the current model version
and which model version the current model card corresponds to.

For models without version number, use "Not currently tracked"
but be sure to track the release date of the model.
-->


**Current Model Version:** 0.1.0 

**Model Version Release Date:** 2025-09-25

**Model Version at last Model Documentation Update:** 0.1.0

**Artifacts:**

* Vector store: `src/rag_flow/tools/refactored_faiss_code/faiss_db/` (FAISS, persisted locally)
* Tool module: `src/rag_flow/tools/refactored_faiss_code/main.py` (`rag_system`)
* Flow: `src/rag_flow/main.py` (`AeronauticRagFlow`)
* Crew configs: `src/rag_flow/crews/*/config/{agents.yaml,tasks.yaml}`
* Environment: `.env` variables for Azure OpenAI and Serper

## Intended and Known Usage

### Intended Use
<!-- info: This section focuses on the initial purpose and/or reasoning
for creating the model. It is important to define this section as the intended use directly affects the AI Act classification. For example:
A face recognition model for personal photo apps → Limited risk
The same model used for law enforcement → High or unacceptable risk


Example Use Case: A university research team develops a machine learning model to predict the likelihood of hospital readmission among diabetic patients over the age of 65, using data from a regional healthcare network. The model is trained and validated specifically on this elderly population and is intended to support hospital planning and academic research. However, the team does not document the model’s intended use or demographic limitations. A health-tech company later integrates the model into a mobile app aimed at helping diabetes patients of all ages manage their care. The model performs poorly for younger users, frequently overestimating their risk of readmission. This leads to unnecessary anxiety, inappropriate self-care decisions, and false alerts to care providers. The misapplication draws criticism for lacking transparency, and regulators question the ethics of deploying a model outside its original context.   -->

* Description

### Domain(s) of use

* Aeronautics technical Q&A and research support


**Specific tasks performed:**
* Query validation (Azure GPT-4o) for aeronautic relevance
* Context retrieval via FAISS retriever (MMR)
* Optional web search enrichment (SerperDev / DuckDuckGo pipeline)
* RAG answer generation with citations
* RAGAS evaluation of answer quality
* Markdown document synthesis

 **Instructions for use for deployers**:
1) Set environment variables (example `.env`): `AZURE_API_BASE`, `AZURE_API_KEY`, `AZURE_API_VERSION`, `MODEL`, `SERPER_API_KEY`.
2) Install dependencies via `uv sync` or `pip install -r requirements.txt`.
3) Run: `crewai run` (entry: `rag_flow.main:kickoff`). For plotting only: use `rag_flow.main:plot`.
4) Ensure local docs directory is accessible (see `docs_test/` path in tool module).
<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 13</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> 
    <p></p>
</div>

### Out Of Scope Uses

Safety-critical decision-making, real-time flight control, regulatory compliance determinations, or non-aeronautics domains without proper reconfiguration and validation.

### Known Applications 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 1(f)
    <p></p>
</div>

<!-- info: Fill out the following section if the model has any
current known usages.
-->

| **Application**   | **Purpose of Model Usage**                                                 | **[AI Act Risk](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)** |
|-------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Aeronautic Q&A    | Context-grounded Q&A with citations using local docs + web                 | Limited                                                                                      |
| Doc generation    | Produce structured markdown reports from combined sources                   | Limited                                                                                      |

Note, this table may not be exhaustive.  Model users and documentation consumers at large
are highly encouraged to contribute known usages.

## Model Architecture 

<div style="color:gray">
EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(b), 2(c)

Info – AI Act requirements:  
This section should contain a description of the elements of the model and the processes of its training and development.  

Article 11(2)(b) requires the design specifications of a system, model selection, and what the system is designed to optimize for, as well as potential trade-offs.  

Article 11(2)(c) requires a description of the system’s architecture, how software components are built on or feed into each other, and the computational resources needed to develop, train, test, and validate the system.
</div>


<!-- Info: Describing the architecture is fundamental for reproducibility, transparency, and effective maintenance. Without clear records of the model’s layers, activation functions, input/output shapes, and training configurations, it becomes difficult to reproduce results, debug issues, or update the model reliably.  -->


* Architecture Description

`AeronauticRagFlow` orchestrates crews: `AeronauticRagCrew` (RAG via `rag_system` tool), `WebCrew` (SerperDev web search), and `DocCrew` (markdown synthesis). The flow: start → capture question → router validates relevance (Azure GPT-4o) → RAG analysis (FAISS + Azure embeddings + Azure/LMS LLM) → web analysis → aggregation → document generation → optional plot.

* Key components
  - Flow: `src/rag_flow/main.py` (`AeronauticRagFlow` with `@start`, `@listen`, `@router`)
  - RAG Tool: `src/rag_flow/tools/refactored_faiss_code/main.py::rag_system`
  - Vector store: FAISS IndexFlatL2 persisted under `faiss_db/`
  - Embeddings: Azure `text-embedding-ada-002`
  - LLM: Azure GPT-4o (question validation) and LM Studio/OpenAI for generation (per tool helpers)
  - Web: SerperDevTool (Google search) and DuckDuckGo scripts (optional path)
  - Evaluation: RAGAS metrics

* Hyperparameter tuning methodology
  - Not applicable; this is an LLM/RAG system. Retrieval parameters chosen empirically (MMR k=6, fetch_k=20, lambda≈0.7; chunk size≈1000, overlap≈200).

* Training Methodology
  - No model training performed; uses pre-trained embeddings/LLMs and builds a vector index over local documents.

* Training duration
  - N/A
    
* Compute resources used
  - CPU for FAISS indexing/search; calls to Azure OpenAI; optional local LM Studio.

### Data Collection and Preprocessing

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(d)
    <p></p>
</div>

<!--check data documentation to avoid duplicates of information and link it in this sectiion

In Article 11, 2 (d) a datasheet is required which describes all training methodologies and techniques as well as the characteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected labelling procedures conducted and data cleaning methodologies deployed -->

* **Steps Involved**:
  * Data collection: Local aeronautic docs loaded from `docs_test/` (see tool code) and additional folders; optional web content for enrichment.
  * Data cleaning: Basic text extraction/formatting; optional content validation for web sources.
  * Data transformation: Text chunking (~1000 chars, 200 overlap), embedding with Azure embeddings.


       
### Data Splitting 

* **Subset Definitions**:
  * Not applicable (no supervised training). Data indexed for retrieval only.
* **Splitting Methodology**:
  * N/A
* **Proportions**:
  * N/A
* **Reproducibility**:
  * Retrieval is deterministic given fixed index and parameters.
    
**Data Shuffling**:

* Shuffle applied: No 

## Model Training Process 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(c, g), paragraph 3

<!-- AI Act requirements info:  
In Article 11 paragraph 2(c), details about the computational resources needed to develop, train, test, and validate AI systems are required.  

Moreover, in accordance with Article 11 paragraph 2(g), this section must include the validation and testing procedures used, the data involved, and the main metrics adopted to measure accuracy, robustness, and compliance with the requirements laid out in Chapter III, Section 2.  

Paragraph 3 further requires detailed information about the monitoring, functioning, and control of the system, as well as logging of testing, with reports dated and signed by responsible stakeholders.-->
<p></p>
</div>


**Details of Processes**:

* **Initialisation**: Load/create FAISS index; load Azure embeddings and LLM client; configure retriever (MMR).
* **Loss Function**: N/A
* **Optimiser**: N/A
* **Hyperparameters**: Chunk size≈1000, overlap≈200; MMR k=6, fetch_k=20, lambda≈0.7; LLM temperature=0 for validation.
        
## Model Training and Validation 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(g)
    <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 15</a>
    <p></p>
</div>

Objective: Clarify what the model is supposed to achieve. 

* Problem statement: Aeronautics domain Q&A with citation-backed, context-grounded answers and final report generation.
* Business goals: Accuracy, transparency (citations), robustness against hallucinations, usability (markdown output).
* Metrics selected: RAGAS (faithfulness, answer relevance, context recall/precision where applicable).
Rationale: RAG-specific metrics better capture grounding and citation quality than pure accuracy.

* Model predictions on the validation set evaluation description: Evaluated per-question via RAGAS in `rag_system`.

<!--
- Performance metrics (e.g., accuracy, F1 score, RMSE) are monitored.
- Documeting this step is important as it enables to detect errors and performance issues early on: Overfitting can be detected using validation loss trends.
-->

<!--### Performance Metrics

- Evaluation Metrics Used (e.g., Accuracy, Precision, Recall, AUC-ROC)
- Benchmarking Results
- Validation Process
- Real-World Performance
- Stress testing
- Performance across different environments and populations -->

**Hyperparameter Tuning**: N/A
    

        
**Regularisation**: N/A
    

    
**Early Stopping**: N/A
    

## Model Testing and Evaluation

<!--
- Performance metrics (e.g., accuracy, F1 score, RMSE) are monitored.
- Documeting this step is important as it enables to detect errors and performance issues early on: Overfitting can be detected using validation loss trends.
-->

<!-- Example: In medical diagnosis, using accuracy alone can be misleading in imbalanced datasets, potentially missing critical cases like cancer. Metrics like recall (which measures the percentage of actual cancer cases the model correctly identifies. Critical for minimizing missed diagnoses), precision ( to ensure that when the model predicts cancer, it’s actually correct—important to reduce false alarms), F1 score, and AUC-ROC provide a more meaningful assessment by accounting for the real-world impact of false positives and false negatives. Choosing the right metrics ensures models are effective, trustworthy, and aligned with practical goals and consequences.

## Model Validation and Testing
- **Assess the metrics of model performance** 
   - accuracy:
   - precision: 
   - recall:
   - F1 score:

- **Advanced performance metrics**
  - ROC-AUC:
    - trade-off between true positive rate and false positive rate
  - PR- AUC
     - Evaluating precision and recall trade-off
  - Specificity
    - (True Negatives/(True Negatives+False Positives))
  - Log Loss (Cross-Entropy Loss):
    - Penalises incorrect probabilities assigned to classes.


- **Context dependant metrics**: 
  - Regression Metrics: For tasks predicting continuous values
  - Clustering Metrics: for tasks grouping similar data points
  - Ranking Metrics: for tasks predicting rankings (e.g., search engines recommendation systems)
  - NLP processing metrics (e.g., text classification, sequence-to-sequence tasks)


- **Fairness Metrics**:
    
    - Ensure the model treats different groups (e.g., based on gender, race) equitably.
    - Examples: Demographic parity, equal opportunity, and disparate impact.
- **Explainability Metrics**:
    
    - Measure how understandable and interpretable are the model’s decisions.
    - Examples: Feature importance, fidelity (how well explanations match the model), and sparsity (using fewer features for explanations).
    - 
- **Robustness Metrics**:
    
    - Assess how well the model performs under challenging or unexpected conditions.
    - Examples: Adversarial robustness, performance under data drift, and sensitivity to input changes.
 
- Limitations of the performance after the tests
- Simulate deployment scenarios to understand real-world implications.
- Define thresholds for acceptable performance levels.
- Justify the choice of metrics based on the application’s purpose.
   
--> 

 **Performance Metrics**:
    
* RAGAS metrics computed during `rag_system` execution for qualitative evaluation.

 **Confusion Matrix**:
    
* Not applicable.

 **ROC Curve and AUC**:
    
* Not applicable.

 **Feature Importance**:
    
* Not applicable; use citations and retrieved chunks as transparency signals.

 **Robustness Testing**:

* Edge-case queries (ambiguous or off-domain) are routed to retry via the validation router.

 **Comparison to Baselines**:
    
* Informal baseline: direct LLM answer without retrieval (not used by design).

### Model Bias and Fairness Analysis 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2 (f, g), paragraph 3, 4
    <p></p>
</div>

<!-- info: This section aims to cover the AI Act requirements layed out in Article 11 paragraph 2 g that requires the description of the potential discriminatory impacts. 
Paragraph 4 requires the assessment of the appropriateness of the performance metrics.-->  



![](https://lh7-rt.googleusercontent.com/docsz/AD_4nXclauxwg1nWuPj2z0TcgUK9y69AqHzk_-jQ5BJwYeDkjPSOLVddFcHJ6-oOiuZ2p4Rk3VpqyKw9CvU7N1LOqYtpdjN6CV_hhTxTtpNj4auLmqhsaIQ5fRLIPnVpZOnhtR63YNELlg?key=Lv0_1kRp5_LSkJabUJ8gjQ)Implicit Bias, Measurement Bias, Temporal Bias, Selection Bias, Confounding Bias

#### Bias Detection Methods Used
    

**Pre-processing:** Source selection and basic cleaning for local docs; optional content validation on web sources.
    
**In-processing:** N/A (no model training)
    
**Post-processing:** Thresholding via router for domain relevance; citation requirement to mitigate hallucinations.
    


**Results of Bias Testing:**
    
N/A; qualitative checks via RAGAS faithfulness and source citations.

#### Mitigation Measures
    

**Fairness adjustments:** Not applicable (no supervised model training). Emphasis on transparent sources and citations.
    
**Adversarial Debiasing:** Not applicable.
    

#### Retraining approaches

**Fairness Regularization:** N/A
    
**Fair Representation Learning:** N/A
    
### Post-Processing Techniques

**Fairness-Aware Recalibration:** After the model is trained, adjust decision thresholds separately for different demographic groups to reduce disparities in false positive/false negative rates.
    
**Output Perturbation:** Introduce randomness or noise to model predictions to make outcomes more equitable across groups.
    
**Fairness Impact Statement:** Explain trade-offs made to satisfy certain fairness criterias
    

## Model Interpretability and Explainability 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a>; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a>  paragraph 2(e)
    <p></p>
</div>

**Explainability Techniques Used:**
  <!-- for example: Shapley values, LIME, etc. Both SHAP and LIME are explainability techniques that help to understand why a machine learning model made a specific prediction — especially when the model is a complex "black box" like a random forest, gradient boosting, or deep neural net. Shap Uses game theory to assign each feature a value showing how much it contributed to a prediction. Lime builds a simple, interpretable model (like a linear model) near the point of interest to explain the prediction -->
    
  Citations of retrieved documents; display of retrieved chunks; transparency via vector store sources.

**Post-hoc Explanation Models**

* Not applicable; post-hoc explanations replaced by source citation and retrieval context visibility.
    

**Model-Specific Explanation Techniques**

<!-- info: this part is important to delineate why a model makes a decision or to debug and identify if the model is focusing on the right parts of the input. Especially fundamental for models deployed in critical domains such as medical, financial and legal or law enforcement. This section can be useful to draft the user-interface section of the documentation.) -->

* N/A
    

How interpretable is the model’s decision-making process? 
The system emphasizes interpretability through explicit citations of source documents and the ability to inspect retrieved chunks that grounded the answer.
<!--
Some technical tools that can aid transparency include:
- Data Lineage Tools: Track the flow and transformation of data (e.g., Apache Atlas, Pachyderm).
- Explainability Libraries: SHAP, LIME, Captum, TensorFlow Explain.
- Version Control Systems: Git, DVC (Data Version Control) for datasets and models. -->

### EU Declaration of conformity 

<div style="color:gray">
    EU AI Act <a href="https://artificialintelligenceact.eu/article/47/" style="color:blue; text-decoration:underline">Article 47</a>(d)
    <p></p>
</div>

 <!-- when applicable and certifications are available: it requires a systems name as well as the name and address of the provider; a statement that the EU declaration of conformity referred to in Article 47 is issued under the sole responsibility of the provider; a statement that the AI system is in conformity with this Regulation and, if applicable, with any other relevant Union law that provides for the issuing of the EU declaration of conformity referred to in Article 47, Where an AI system involves the processing of personal data;  a statement that that AI system complies with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, reference to the harmonised standards used or any other common specification in relation to which
conformity is declared; the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued; the place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.-->

### Standards applied

<!-- Document here the standards and frameworks used-->
- CrewAI Flow design patterns
- RAG evaluation with RAGAS framework
- Documentation via Sphinx (for tools package) and markdown reports

## Documentation Metadata

### Version
<!-- info: provide version of this document, if applicable (dates might also be useful) -->
2025-09-25 v1.0

### Template Version
<!-- info: link to model documentation template (i.e. could be a GitHub link) -->
N/A

### Documentation Authors
<!-- info: Give documentation authors credit

Select one or more roles per author and reference author's
emails to ease communication and add transparency. -->

* **Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino:** (Owner)
* **Fabio Rizzi, Giulia Pisano, Marco Diomedi, Riccardo Zuanetto, Roberto Gennaro Sciarrino:** (Contributor)
