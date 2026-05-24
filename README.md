# GenAI Assistant with RAG – Assignment Reference Guide

An interactive, production-ready educational reference sandbox demonstrating the mechanics of **Retrieval-Augmented Generation (RAG)**. This guide provides students with a playground to explore document management, text chunking pipelines, vector embedding matrices, cosine similarity calculations, prompt engineering, and grounded multi-turn conversations.

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

## 7. Interactive Interface & Diagnostics

The companion visual workspace features 7 modules:
* **01 Architecture:** Conceptual breakdown comparing raw vs. grounded queries.
* **02 Document Store:** Active document manager allowing full CRUD operations over loaded client policy tables.
* **03 Chunking Engine:** Visualizer to adjust size thresholds, overlaps, and parsing strategies from regular characters, words, or full sentences.
* **04 Vector Database:** Flat index grid displaying normalized values and active mathematical dimensions.
* **05 Similarity Search Matrix:** Run targeted live searches with variable sliding thresholds ($0.0 - 1.0$) and candidate $k$ sizes.
* **06 Prompt Engineering Panel:** Read-only instruction generator showing actual computed text sent to the foundation models.
* **07 Conversational Sandbox Chat:** Low-latency workspace with real-time token diagnostics, grounding state tags, and clear hooks.
