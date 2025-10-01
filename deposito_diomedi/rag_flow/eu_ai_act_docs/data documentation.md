# Rag Flow — Data Documentation 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/10/" style="color:blue; text-decoration:underline">Article 10</a>
  <br>EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1, 2 (d)
  <!-- info: The AI Act delineates the data governance practices required in Article 10 and requires a description of the intended purpose, version and provider, and relevant versions and updates.  
  In Article 11(2)(d), a datasheet is required which describes all training methodologies and techniques as well as the characteristics of the training dataset, a general description of the dataset, information about its provenance, scope and main characteristics, how the data was obtained and selected, labelling procedures conducted, and data cleaning methodologies deployed. -->
  <p></p>
</div>

**Dataset Owner**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto
<br>**Document Version**: 2025-09-25 v0.1 
<br>**Reviewers**: Fabio Rizzi, Giulia Pisano, Marco Diomedi, Roberto Gennaro Sciarrino, Riccardo Zuanetto

<!-- info: Replace with dataset name -->

## Overview 
The RAG Flow is a multi-stage retrieval-augmented generation (RAG) system. It retrieves context from a local knowledge base built from documents in the `docs/` directory and complements it with web search results. The system uses Azure OpenAI for embeddings and chat completion, FAISS for vector indexing, and CrewAI agents to orchestrate RAG, web research, and document creation.

### Dataset Description 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 1, 2(d)
  <p></p>
  <!-- info: The AI Act requires a description of  all training methodologies and techniques as well as the charatcteristics of the training dataset, general description of the dataset, information about their provenance, scope and main characteristics, how the data was obtained and selected labelling procedures conducted and data cleaning methodologies deployed.-->
</div>

The dataset consists of domain documents (PDF, CSV, Markdown, TXT, and select image formats) located under `docs/` and optionally complemented by publicly available web content fetched during queries. Content covers aeronautics technical documentation, manuals, and related knowledge used to answer user questions. The system indexes these documents into a FAISS vector store using Azure OpenAI embeddings and retrieves the most relevant chunks during inference. Web content, when used, is cleaned to remove UI/boilerplate before inclusion. Primary use is question answering and report generation in `output/redacted_document.md`.

### Status
<!-- scope: telescope -->
<!-- info: Select **one:** -->
**Status Date:** 2025-09-25

**Status:** Regularly Updated 

### Relevant Links
<!-- info: User studies show document users find quick access to relevant artefacts like papers, model demos, etc..
very useful. -->

* Repository: [Rag Flow](https://github.com/marco-diomedi-ey/deposito_diomedi/tree/main/rag_flow)
* Flow entrypoints: `src/rag_flow/main.py` (`kickoff`, `plot`)
* RAG pipeline: `src/rag_flow/tools/refactored_faiss_code/`
* Output doc: `output/redacted_document.md`


### Developers
* *Fabio Rizzi,*
* *Giulia Pisano,* 
* *Marco Diomedi,*
* *Roberto Gennaro Sciarrino,*
* *Riccardo Zuanetto*


### Owner
<!-- info: Remember to reference developers and owners emails. -->
* *Fabio Rizzi,*
* *Giulia Pisano,* 
* *Marco Diomedi,*
* *Roberto Gennaro Sciarrino,*
* *Riccardo Zuanetto*

### Deployer instructions of Use
<!-- info: Important to determine if there are relevant use-cases or if the data is unsuitable for certair applications. -->
* **Instructions for use for deployers**:
  - Ensure environment variables for Azure OpenAI are configured: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_API_VERSION`, `AZURE_EMBEDDING_MODEL`, `AZURE_MODEL`.
  - Populate `docs/` with permissible documents (no personal data; see GDPR section). Run the flow with `crewai run`.
  - Review and validate generated outputs; do not rely on web-sourced content without verification.

<!-- 
How to use the data responsibly. Include restrictions, review process, ethical concerns. -->

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/13/" style="color:blue; text-decoration:underline">Article 13</a>
  <p></p>
</div>

### Version Details
System versioning is code-driven via Git; FAISS indices are persisted under `./faiss_db/<topic>` based on user query (see `Settings.set_persist_dir_from_query`).

## Data Versioning 

(Article 11, paragraph 2(d))

**Data Version Control Tools:**
<!-- Data version control tools are important to track changes in datasets, models, and experiments over time, enabling collaboration, reproducibility, and better model management. This is particularly important to then detect model drifts and debugging or for rollbacks when overwriting on the original data  -->

* Source code via Git; runtime indices persisted under `./faiss_db/` per topic.
* Document set changes are controlled by commits to `docs/` and flow outputs in `output/`.
* No DVC configured; consider DVC or Git-LFS for large artifacts if needed.

### Maintenance of Metadata and Schema Versioning 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 3
  <p></p>
</div>

#### Why

Document structure and retrieval parameters (chunking, search strategy) affect downstream behavior and must be stable and reproducible.

#### How

* Maintain a manifest of source files under `docs/` (path, hash, timestamp).
* Record `Settings` used for chunking/retrieval (chunk_size=1000, chunk_overlap=200, search_type="mmr", k=6, fetch_k=20, mmr_lambda=0.7).
* Persist FAISS indices in topic-specific directories and record their creation time and embedding model version (`AZURE_EMBEDDING_MODEL`).

## Known Usages 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 3
  <p></p>
  <!--info: The AI Act requires delineating a system’s foreseeable unintended outcomes and sources of risks to health and safety, fundamental rights, and discrimination in view of the intended purpose of the AI system;  
  the human oversight measures needed in accordance with Article 14, including the technical measures put in place to facilitate the interpretation of the outputs of AI systems by the deployers;  
  and specifications on input data, as appropriate.-->
</div>

<!-- info: Fill out the following section if the dataset has any
current known usages. This is important to make sure that the dataset is used ethically and legally. A dataset created for classification may not be suitable for regression, or vice versa.
Moreover, labeling quality, data coverage, and structure vary with use case—assuming it can be used for anything is dangerous. For instance:A skin lesion dataset created for classification—labeling images as benign, malignant, or uncertain—is mistakenly used by an insurance company to train a regression model that predicts cancer risk scores. Because the dataset lacks continuous risk-related data such as treatment outcomes, progression timelines, or cost indicators, the model produces unreliable predictions. As a result, high-risk patients may be misclassified as low-risk, leading to denied or delayed insurance claims. This misuse not only puts patients at risk but also exposes the insurer to ethical and legal scrutiny. Hence it is important to define the safe extent of use of a dataset. 
-->
### Model(s)
<!-- scope: telescope -->
<!-- info: Provide a table of known models
that use this dataset.
-->

| **Model**                    | **Model Task**                  | **Purpose of Dataset Usage**            |
|------------------------------|----------------------------------|-----------------------------------------|
| Azure OpenAI Chat (`AZURE_MODEL`) | Generative QA (RAG)             | Answer generation grounded in context   |
| Azure OpenAI Embeddings       | Text Embedding                   | Vectorization for FAISS retrieval       |

Note, this table does not have to be exhaustive. Dataset users and documentation consumers at large
are highly encouraged to contribute known usages.

### Application(s)
<!-- scope: telescope -->
<!-- info: Provide a table of known AI/ML systems
that use this dataset.
-->

| **Application**          | **Brief Description**                                      | **Purpose of Dataset Usage**                 | 
|--------------------------|----------------------------------------------------------------|----------------------------------------------|
| Aeronautic RAG Flow      | Multi-agent RAG + web research + doc redaction pipeline       | QA, analysis, and report generation          |

## Dataset Characteristics

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2(d)
  <p></p>
</div>
<!-- This section reflects the requirements of the AI Act of Article 11, paragraph 2 (d): where relevant, the data requirements in terms of datasheets describing the training methodologies and
techniques and the training data sets used, including a general description of these data sets, information about
their provenance, scope and main characteristics; how the data was obtained and selected; labelling procedures
(e.g. for supervised learning), data cleaning methodologies (e.g. outliers detection). Moreover, in order to comply with GDPR provisions, you need to disclose whether you are handling personal information. -->

**Data Types:** Text (PDF, CSV, MD, TXT), web text (cleaned)
<br>**Size/Volume:** Variable; determined by contents of `docs/`
<br>**Number of Instances/Records:** Variable; derived at runtime via `scan_docs_folder`
<br>**Primary Use Case(s):** Aeronautics QA via RAG; document synthesis
<br>**Associated AI System(s):** `AeronauticRagFlow` (CrewAI Flow)
<br>**Number of Features/Attributes (if applicable):** N/A (textual)
<br>**Label Information (if applicable):** N/A (no supervised labels)
<br>**Geographical Scope:** Not restricted; documents may be global
<br>**Date of Collection:** Based on repository commit timestamps and web query time

## Data Origin and Source
<!-- importanto to define this step to understand also compliance with GDPR.  -->
**Source(s):**
* Local repository documents under `docs/` (PDF/CSV/MD/TXT/images)
* Web search results via WebCrew (SerperDev; content cleaned before use)
<br>**Third-Party Data:** Web content is third-party and subject to website terms; use only excerpts for QA with source citation.
<br>**Ethical Sourcing:** Only non-personal, publicly available materials should be ingested. Do not include personal data or sensitive data. Respect robots.txt/ToS.

## Provenance

The corpus is built on demand for each query from any local files in `docs/` and publicly available web pages discovered via search. Web content is cleaned (UI/boilerplate removed), chunked, embedded with Azure OpenAI, and temporarily indexed in FAISS only to answer the current query. Permanent storage of raw web data is not intended.

### Collection

#### Method(s) Used
<!-- scope: telescope -->
<!-- info: Select **all applicable** methods used to collect data.

Note on crowdsourcing, this covers the case where a crowd labels data
(make sure the reference the [Annotations and Labeling](#annotations-and-labeling)
section), or the case where a crowd is responsible for collecting and
submitting data independently to form a collective dataset.
-->

* Taken from other existing datasets (project `docs/`)
* Scraped or crawled (web pages via search during queries)

#### Methodology Detail(s) 

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> 2 (a), (b), (d)
  <p></p>
</div>
<!-- scope: periscope -->
<!-- info: Provide a description of each collection method used. Use additional notes to capture any other relevant information or
considerations. (Usage Note: Duplicate and complete the following for collection method
type.) -->
**Collection Type**

**Source:** `docs/` repository contents and ad-hoc web pages discovered via search

**Platform:** Local filesystem; web search (SerperDev) and Azure OpenAI for processing

**Is this source considered sensitive or high-risk?** No (personal data excluded by policy)

**Dates of Collection:** Continuous; repository commit history; web at query time

**Update Frequency for collected data:** On demand (when new docs are added or new queries occur)

**Additional Links for this collection:**

See section on [Access, Rention, and Deletion](#access-retention-and-deletion)

**Additional Notes:** Web content is cleaned via `clean_web_content` to remove UI/legal boilerplate before indexing.

#### Source Description(s)
<!-- scope: microscope -->
<!-- info: Provide a description of each upstream source of data.

Use additional notes to capture any other relevant information or
considerations. -->
* **Local Docs:** Structured/unstructured files in `docs/` loaded via format-specific loaders (PDF, CSV, MD, TXT, images)
* **Web Pages:** Public pages identified by web search, cleaned and used as supplemental context

**Additional Notes:** Add here

#### Collection Cadence
<!-- scope: telescope -->
<!-- info: Select **all applicable**: -->
**Static:** Local baseline in `docs/`

**Dynamic:** Updated on demand as new documents are added or new queries fetch web content
    
## Data Pre-Processing 

<div style="color:gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 2 (d, e)
  <p></p>
</div>

### Data Cleaning

* Handling missing data: Not applicable to raw text; loaders skip unsupported files
* Outlier treatment: Not applicable
* Duplicates removal: Implicitly mitigated by splitting and retrieval; consider deduplication if needed
* Error correction: Web boilerplate removed via regex patterns in `clean_web_content`

### Data Transformation

* Text chunking via `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`, hierarchical separators
* Context formatting with explicit source attribution via `format_docs_for_prompt`

### Feature Engineering

* Feature extraction: Azure OpenAI embeddings (`AZURE_EMBEDDING_MODEL`, default `text-embedding-ada-002`)
* Retriever configuration: `search_type="mmr"`with `k=6`, `fetch_k=20` and `mmr_lambda=0.7`

### Dimensionality Reduction

* Not explicitly applied; dimensionality determined by chosen embedding model

### Data Augmentation

* Not used

## Data Annotation and Labeling 

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> 2(d)
  <p></p>
</div>

* Not applicable — the system operates on unlabelled text for retrieval and QA.

## Validation Types

### Method(s) 

Anti-hallucination prompting; context-only answering with mandatory source citations; optional RAG evaluation 

### Breakdown(s)

**Prompt-level constraints** — enforced each query

**Retrieval checks** — top-k relevance via FAISS retriever

### Description(s)
Prompt instructs the model to answer only from provided context and to cite sources; if information is unavailable, it must state so. Retrieval uses MMR to diversify and reduce redundancy.


## Sampling Methods


### Method(s) Used
MMR-based retrieval from FAISS index; top-k = 6 with candidate pool = 20


### Characteristic(s)
Diverse yet relevant contexts via MMR; chunk-based context selection

### Sampling Criteria
Semantic similarity to query with diversity trade-off (`mmr_lambda=0.7`)



### Description(s)
Sampling occurs at inference time based on the query; no static train/val/test split is maintained.

## Dataset Distribution and Licensing 

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> 2(d)
  <p></p>
</div>

* Availability: Local project documents under `docs/`; ad-hoc web content
* Open/public or private dataset: Mixed — local docs (project-controlled), web content (third-party)
* Dataset Documentation Link: This document (in-repo)
* User Rights and Limitations: Respect respective licenses/ToS; use excerpts with attribution; do not redistribute third-party content wholesale

## Access, Retention, and Deletion
<!-- info: Where applicable, collect input from privacy governance team -->
### Access

#### Relevant Links

* Local filesystem: `docs/`, `faiss_db/`, `output/`
* Code paths: `src/rag_flow/`, `src/rag_flow/tools/refactored_faiss_code/`

#### Data Security Classification in and out of scope delineation
<!-- scope: Is there a potentially harmful application iof this data, can you foresee this?  -->
<!-- info: Select **one**: Use your companies data access classification
standards (replace the classifications below accordingly) -->


#### Prerequisite(s)
<!-- scope: microscope -->
<!-- info: Please describe any required training or prerequisites to access
this dataset. -->
Developers require repository access and valid Azure credentials in environment variables. No personal data should be introduced.

### Retention

#### Duration
<!-- scope: periscope -->
<!-- info: Specify the duration for which this dataset can be retained: -->
FAISS indices retained as long as relevant for project; outputs retained per project policy.

#### Reasons for Duration
<!-- scope: periscope -->
<!-- info: Specify the reason for duration for which this dataset can be retained: -->
Support reproducibility, debugging, and incremental updates.

#### Policy Summary
<!-- scope: microscope -->
<!-- info: Summarize the retention policy for this dataset. -->
**Policy:** Project-level policy; avoid storing personal data; clear indices by deleting topic directories under `faiss_db/` when no longer needed.
  
## Data Risk Assessment

**Describe the assessment of data risks**:

* Personal data: Not expected; ingestion policy excludes personal/sensitive data
* Copyright/ToS: Web content must comply with source terms; use minimal excerpts with attribution
* Hallucination risk: Mitigated by context-only prompting and citations; human review recommended for outputs


## Cybersecurity Measures

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/11/" style="color:blue; text-decoration:underline">Article 11</a> ; <a href="https://artificialintelligenceact.eu/annex/4/" style="color:blue; text-decoration:underline">Annex IV</a> paragraph 5
  <p></p>
</div>


### Data Security Measures

#### Data Storage

* **Encryption**: Local developer environment; encryption-at-rest not implemented by code — rely on OS/drive encryption
* **Access Control**: Repository and machine-level access controls
* **Backup**: Per developer or CI environment policies
* **Integrity Monitoring**: Optional — recommend hashing source files and indices
* **Security**: API keys stored in environment variables; do not commit secrets

#### Data Transfer

* **Encryption in Transit**: Azure OpenAI endpoints over TLS; web access via HTTPS
* **Endpoint Security**: Managed by Azure/OpenAI service and local environment
* **API Security**: Azure API key in env vars; rate limits per provider
* **Data Masking**: Not applicable (no personal data intended)

#### Data Processing

* **Secure Environments**: Local dev or controlled runtime
* **Audit Logs**: Console logs for loading and processing steps
* **Data Minimisation**: Only context necessary for answering is retrieved



### Standards Applied
 <!-- info: provide information of the standards applied and certifications in this section-->
Currently no formal certification. Target alignment: EU AI Act (Articles 10, 11, 13, 14), GDPR (no personal data), and good practices from NIST AI RMF.

### Data post-market monitoring

-**Data Drift Detection and Monitoring:** Describe here what type of drift was identified (covariate drift, prior probability drift or concept drift)

<!-- info: This section is particularly important as it enables to understand whether the model is still making accurate predictions, especially if used in critical domains. 
For instance: If yoi train a model to detec and diagnose lung cancer from hospital A, deploying to hopsital B might affect accuracy as the other hospital might use different scanning machines that have different contrasts or resolutions and therefore might affect the distributional differences of the input images and the model might drop accuracy.
 .-->

-**Audit Logs:** Periodically perform manual or semi-automated reviews of data samples and log changes in the data as well as access patterns.

* **Action plans implemented to address identified issues:**.
* Monitor retrieval quality and answer accuracy; adjust `Settings` and document set as needed.



### EU Declaration of conformity

<div style="color: gray">
  EU AI Act <a href="https://artificialintelligenceact.eu/article/47/" style="color:blue; text-decoration:underline">Article 47</a>
  <p></p>
</div>

 <!-- when applicable and certifications are available: it requires a systems name as well as the name and address of the provider; a statement that the EU declaration of conformity referred to in Article 47 is issued under the sole responsibility of the provider; a statement that the AI system is in conformity with this Regulation and, if applicable, with any other relevant Union law that provides for the issuing of the EU declaration of conformity referred to in Article 47, Where an AI system involves the processing of personal data;  a statement that that AI system complies with Regulations (EU) 2016/679 and (EU) 2018/1725 and Directive (EU) 2016/680, reference to the haharmonised standards used or any other common specification in relation to which
conformity is declared; the name and identification number of the notified body, a description of the conformity
assessment procedure performed, and identification of the certificate issued; the place and date of issue of the declaration, the name and function of the person who signed it, as well as an
indication for, or on behalf of whom, that person signed, a signature.-->

### Standards applied
<!-- Document here the standards and frameworks used-->
See "Standards Applied" above.


### Documentation Metadata

### Version
0.1 — 2025-09-25

### Template Version
<!-- info: link to model documentation template (i.e. could be a GitHub link) -->

### Documentation Authors
* Fabio Rizzi
* Giulia Pisano
* Marco Diomedi
* Roberto Gennaro Sciarrino
* Riccardo Zuanetto
