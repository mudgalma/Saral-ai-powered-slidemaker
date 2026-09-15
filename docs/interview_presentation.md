# Aasan (formerly SARAL) - 15 Minute Interview Presentation

*This document contains the slide structure and speaker notes for your 15-minute interview. You can copy this text directly into PowerPoint, Google Slides, or Pitch to generate your deck.*

---

## Slide 1 — Title Slide
**Title:** Aasan: Audience-Adaptive Research Communicator
**Subtitle:** Grounded Slide Generation from Complex PDFs
**Bullet Points:**
- Converts dense research PDFs into audience-specific presentations.
- Utilizes a robust RAG (Retrieval-Augmented Generation) pipeline.
- Solves hallucination issues inherent in purely parametric LLMs.

---

## Slide 2 — The Problem: Why RAG?
**Title:** Overcoming Parametric Memory Limits
**Bullet Points:**
- *NeurIPS 2020 Context:* Standard LLMs generate answers from fixed weights, leading to hallucinations and outdated facts.
- *The RAG Solution:* Grounding the LLM with an external non-parametric memory (retrieved chunks) drastically improves factuality.
- *Aasan's Mission:* Apply RAG to research papers to strictly enforce citation coverage and provenance for every generated claim.

---

## Slide 3 — Architecture Overview
**Title:** End-to-End System Architecture
**Visual Suggestion:** *Embed the `docs/aasaan_hld.jpg` diagram here.*
**Bullet Points:**
- **Ingestion:** Docling parser processes PDFs, maintaining layout and extracting assets.
- **Retrieval:** Hybrid Search (pgvector dense + Postgres BM25 sparse).
- **Generation:** GPT-4o-mini structured output generation.
- **Frontend:** React/TypeScript UI for interactive, real-time preview.

---

## Slide 4 — Chunking Strategy
**Title:** Structural Integrity & Math Preservation
**Bullet Points:**
- Standard 512-token chunking destroys tables and mathematical formulas.
- We utilize Docling's `HybridChunker` to respect document layout (paragraphs, lists).
- **Crucial Choice:** The chunker is configured to *never* split a LaTeX block. When generating technical scripts, mathematical integrity is perfectly preserved.

---

## Slide 5 — How Generation Works Internally
**Title:** Audience-Adaptive Generation Pipeline
**Bullet Points:**
- **Evidence Injection:** The Prompt Builder dynamically injects top-k chunks from the Hybrid Retriever into the system prompt.
- **Parameterization:** The LLM is conditioned on strict user variables: `{audience}` (e.g., Policymakers), `{length}` (e.g., 90s), and `{style}` (e.g., Plain-English).
- **JSON-Schema Constraint:** Output is strictly coerced into a robust `SlideDeck` format.
- **Grounding Enforcement:** Every single bullet point must map directly to a `source_chunk_id` to enforce provenance.

---

## Slide 6 — Iterative Editing & Change Tracking
**Title:** Delta-Anchored Editing
**Bullet Points:**
- **Stateful Conversations:** The system tracks immutable artifact versions for every chat thread in Supabase.
- **Delta-Anchoring:** When a user says *"Make slide 3 more mathematical"*, the LLM receives the `OLD` slide 3 alongside explicit `{change_instructions}`.
- **Targeted Re-retrieval:** The system retrieves chunks containing the LaTeX formulas and injects them into the `NEW` slide without losing the context of the rest of the presentation.

---

## Slide 7 — Evaluation
**Title:** Rigorous Quality & Grounding Evaluation
**Visual Suggestion:** *Include a screenshot of your LangSmith trace or a table of the metrics.*
**Bullet Points:**
- **Factuality & Citation Coverage (91%):** 91% of all generated factual claims successfully map to a verified source chunk.
- **Lexical & Semantic Quality:** Evaluated against human expert scripts (`saral-eval-v1`).
  - **ROUGE-L:** `0.75` (High exact-phrase retention)
  - **BERTScore:** `0.91` (Exceptional semantic parity)
- **Human Evaluation:** Passed qualitative checks for audience-appropriateness, pacing, and visual layout.

---

## Slide 8 — Proposed Improvement
**Title:** Future Work: Math-Aware Re-ranker
**Bullet Points:**
- **The Problem:** Standard vector embeddings (like `text-embedding-3-small`) often fail to capture the deep semantics of complex mathematical formulas.
- **The Solution:** Implement a **Math-Aware Re-ranker**.
- **How it works:** When a user prompt explicitly requests a highly technical or methodological breakdown (e.g., Grad Student audience), the retriever artificially boosts the ranking score of chunks containing parsed Docling LaTeX formula objects.
