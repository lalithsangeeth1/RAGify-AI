# GenAI Assistant with RAG – Assignment Reference Guide

An interactive, production-ready educational reference sandbox demonstrating the mechanics of **Retrieval-Augmented Generation (RAG)**. This guide provides students with a playground to explore document management, text chunking pipelines, vector embedding matrices, cosine similarity calculations, prompt engineering, and grounded multi-turn conversations.

### 🌐 Live Application Link
You can access the live, interactive deployment of this application here:
👉 **[Live Application Sandbox](https://ragify-ai-184030380121.asia-southeast1.run.app)**

---

## 1. Architectural Diagram

Below is the technical data flow outlining how documents are indexed into the high-dimensionality vector database, and how user queries are resolved through grounded semantic contexts rather than model memory.

```
       [ KNOWLEDGE BASE INGESTION PIPELINE ]
                       
  +-------------------------------------------------+
  |  Raw Documents Store (Flat JSON documents List)  |
  +-------------------------------------------------+
                           │
                           ▼  (Chunking Engine: Word, Char, or Sentence boundaries)
  +-------------------------------------------------+
  |          Document Chunks (Overlap size)         |
  +-------------------------------------------------+
                           │
                           ▼  (embModel: gemini-embedding-2-preview)
  +-------------------------------------------------+
  |           Continuous Vector Databases           |
  |  - In-Memory / SQLite Embeddings Matrix Array   |
  +-------------------------------------------------+


        [ RUNTIME USER CONVERSATION PIPELINE ]

  +-----------------------+
  |  User enters query:   |
  |  "Delete my account"  |
  +-----------------------+
              │
              ▼  (Generate Query Embedding Vector)
  +-------------------------------------------------+
  |      Cosine Similarity Matrix Calculations      |
  |  - Measure angle weights against db chunks       |
  +-------------------------------------------------+
              │
              ▼
  +-------------------------------------------------+
  |           Dynamic Threshold Filtering           |
  |  - At least 1 candidate score >= Threshold      |
  +-------------------------------------------------+
              │
       ┌──────┴──────┐
 [ >= Threshold ] [ < Threshold ]
       │             │
       ▼             ▼
  +-----------+ +-----------------------------------+
  | Retrieve  | | Inject "NO RELEVANT CONTEXT"       |
  | contexts  | | triggers fallback "I do not have   |
  +-----------+ | enough information to answer..." |
       │        +-----------------------------------+
       │                     │
       └──────────┬──────────┘
                  ▼
  +-------------------------------------------------+
  |             Assemble Grounded Prompt            |
  | - System instructions, History, Context, Query  |
  +-------------------------------------------------+
                  │
                  ▼  (llmModel: gemini-3.5-flash with low temperature=0.2)
  +-------------------------------------------------+
  |         Polished User-Facing Reply              |
  +-------------------------------------------------+
```

---

## 2. RAG Workflow Explanation

Standard Large Language Models generate answers based solely on pre-trained computational weights. Consequently, they hallucinate when queried about custom business rules or specific system constraints.

The dynamic Retrieval-Augmented Generation (RAG) workflow implemented in this application bridges this gap:
1. **Repository Ingest:** Internal enterprise guidelines (e.g. password resets, api rates) are maintained as plain-text JSON entries.
2. **Deterministic Partitioning:** Text files are tokenized and split into discrete, small, overlapping chunks to keep semantics localized.
3. **Similarity Retrieval:** Instead of keyword matching (which fails to grasp synonyms or synonyms like "cancel account" vs "termination"), the query is embedded and compared dynamically.
4. **Context Grounding:** Top results are fed into the system-instructions context block, constraining the AI model to behave as an index investigator rather than an opinion generator.

---

## 3. Embedding Strategy

Our application supports hybrid strategies for generating semantic embeddings:

* **Production Live Mode:** Connects directly with Google's state-of-the-art `gemini-embedding-2-preview` model via the `@google/genai` TypeScript SDK. Chunks are mapped into 768-dimensional floating point values representing deep semantic attributes.
* **Local Simulated Fallback Mode:** To prevent system crashes or blocked interfaces when credentials are empty, our backend implements a custom, deterministic vector generation hash. A custom hashing seed computes standard 768-dimensional normalized unit spheres. Because vectors are pre-normalized (`magnitude = 1.0`), the dot product matches cosine similarity metrics perfectly!

---

## 4. Similarity Search Logic

We implement standard mathematical **Cosine Similarity** to index and retrieve relevant content chunks:

$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

### Grounding Threshold Configuration
* **Configurable Boundary ($0.70 - 0.75$):** Prevents hallucinations. If the highest similarities return `< Threshold` bounds (i.e. the user is asking about irrelevant subjects like "What is your refund policy?" or "Recommend a recipe"), the system flags the context as empty.
* **Deterministic Fallback Sentence:** If no context returns above target, the model delivers: *"I do not have enough information to answer that question based on the knowledge base."* This limits speculation.

---

## 5. Prompt Design Reasoning

To keep the LLM completely grounded and deterministic, we enforce low parameter temperature levels (`temperature=0.2`) and assemble a strict hierarchical template:

```text
You are a grounded RAG QA chat assistant for our student assignments.
Guidelines:
1. You MUST answer the user question strictly using ONLY the provided "Context Block" below.
2. If the "Context Block" matches "NO RELEVANT CONTEXT FOUND", or is insufficient to answer, you must respond EXACTLY with the literal sentence: "I do not have enough information to answer that question based on the knowledge base."
3. Keep your answers brief, factual, precise, and completely faithful to the context block. Avoid extra remarks or speculation.
4. If the question is a generic greeting (e.g., 'hello', 'hi', 'hey') and has no contextual question, you can reply with a brief friendly greeting and list what topics you are authorized to discuss.
```

By organizing the instructions as native, system-level declarations, the model adheres strictly to the retrieved context chunks.

---

## 6. Setup Instructions

The application is fully ready to run locally inside standard full-stack Node.js development environments.

### Local Installation
1. Clone the repository and navigate to the project directory:
   ```bash
   npm install
   ```

2. Register your environment secrets. Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY="your_actual_gemini_api_key"
   ```

3. Start the development server (runs full-stack Express router bound in tandem with Vite):
   ```bash
   npm run start
   ```
   Open your browser to [http://localhost:3000](http://localhost:3000) to view the active interactive playground.

### To Build for Production
```bash
npm run build
```

---

## 7. Interactive Interface & Diagnostics Walkthrough

The companion workspace guides you through each processing block of the RAG pipeline with high-fidelity visual control panels.

### Module 01: Core Architecture Workflow
The visual blueprint outlines standard LLM completion pathways contrasted against grounded retrieval loops. It features detailed comparison boxes, semantic match scores, and a checklist of assignment requirements.
![01 Core Architectural Blueprint](docs/screenshots/01_rag_architecture.jpg)

### Module 02: Document Manager Store
An interactive workspace supporting CRUD actions for `docs.json` (Add, Delete, or Restore Defaults) to dynamically alter the database of policies.
![02 Document Manager Store](docs/screenshots/02_document_store.jpg)

### Module 03: Text Chunking Pipeline
Fine-tune character limits, overlap sizes, and parsing granularities (by standard characters, word boundaries, or complete sentence objects) with real-time visual output of the resulting collection.
![03 Text Chunking Pipeline](docs/screenshots/03_chunking_engine.jpg)

### Module 04: Vector Database Index
Inspect high-dimensionality embeddings (768 decimals) generated via Google Gemini models, complete with index status headers and full math tables of unit coordinates.
![04 Vector Store and Embeddings](docs/screenshots/04_vector_database.jpg)

### Module 05: Similarity Search Matrix
Run targeted Cosine Similarity searches on custom phrases over a sliding threshold ($0.0 - 1.0$) to preview the retrieved candidate segments.
![05 Similarity Matrix Search](docs/screenshots/05_similarity_search.jpg)

### Module 06: Grounded Prompt Assembly
Understand the exact system core prompts, context-chunk injections, and conversational history blocks built in memory prior to requesting LLM responses.
![06 Prompt Engineering Panel](docs/screenshots/06_prompt_engineering.jpg)

### Module 07: Live Conversation Terminal
A fully-featured conversational sandbox providing direct terminal interactions. The view includes grounding indicator badges, real-time token tracking, configurable thresholds/temperature sliders, and retrieved context listings.
![07 Playable Conversational Terminal](docs/screenshots/07_conversational_sandbox.jpg)

---

## 8. TRUEAILAB Submission Format

For quick reference during evaluation, here is the official submission format for this assignment:

```text
GitHub Repository:
https://github.com/jukantilalith/trueailab-rag-assignment

Live Application:
https://ragify-ai-184030380121.asia-southeast1.run.app
```

