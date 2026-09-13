# SARAL Script Buddy

SARAL RECRUITMENT TASK

GENERAL INSTRUCTIONS

● Part A - Programming task (build a working prototype + code)

● Part B - Paper-reading & short presentation (video ≤ 5 mins + slide summary)

You may rely on external literature, APIs, and prebuilt tools (LaTeX, Beamer, pandoc, PyLaTeX,

LLMs, RAG, LangChain etc.). Cite every external resource. Submissions will be checked for

plagiarism (including AI-generated content).

Feel free to write to us (reply-all in the email that you received with task instructions) for

questions & clarifications.

SARAL Chatbot: Audience-Adaptive Script & Bullet

Generator

Goal: Design a chatbot module for SARAL that, given a slide deck or summary, produces

audience-adaptive scripts, bullet points, and tweet-sized abstracts. It should support

iterative editing via conversation (user asks to “make it more visual”, “dumb down #3”, etc.).

Context & motivation:

Users should be able to chat with SARAL, drop in a slide or paper, and say “make a 90-second

script for policymakers” or “create 10 tweet threads and 5 LinkedIn summaries” and iterate until

satisfied.

See RAG literature for retrieval + generative setup. arXiv+1

Part A - Programming Task (required)

Implement a small SARAL Chatbot prototype that:

1. Architecture requirements: Use a retrieval layer (dense or sparse) over the uploaded

paper(s) and a generator (LLM, e.g., open weights or API) to produce scripts and bullets.

The pipeline should:

○ Ingest a paper (PDF/LaTeX) → embed chunks (DPR / sentence-BERT or

OpenAI/other embeddings).

○ Given a user prompt like: "Make a 7-slide talk for graduate students focusing on

methods; include 3 speaker notes per slide and preserve key equations", the

system returns:

■ Slide-level bullets

■ Full speaker script (approx 5–7 minutes)

■ Speaker notes with math preserved in LaTeX inline

○ Provide provenance: for each generated claim/sentence, link to the source

chunk(s) or page numbers.

2. Core features:

● Produce multiple script lengths (30s, 90s, 5min) and styles (technical,

plain-English, press release).

● Generate slide-level bullet points for each audience.

● Provide change-tracking: when the user asks “make #2 less technical,” the

chatbot must show the delta (old vs new) and why it changed.

● Scalable Architecture

● Support safety/style constraints (no offensive content, accessible language).

.

3. Implementation expectations:

● Implement RAG (retrieval-augmented generation) for long decks: chunk slides,

create embeddings, and allow the agent to reference specific slides.

● Provide a minimal UI (chat interface) or a script showcasing the conversation

flows.

● Include conversation logs as evidence that the bot can iterate and refine outputs

● Provide a clean architecture diagram

4. Evaluation:

○ Automatic: factuality proxy by overlap between generated claims and source

sentences (exact-match / semantic similarity); citation coverage (what % of

content has a retrieved provenance).

○ Quality: ROUGE / BERTScore vs. human-authored script for a small test set (3

papers).

○ Human eval: audience appropriateness, factuality, and helpfulness (3 raters).

5. Implementation constraints: You may use existing embedding models: - open ai

embeddings, google ai embeddings, (sentence-BERT) and small LLMs (e.g., 7B open

models) or API-based LLMs. Use small conference papers to avoid excessive cost.

Deliverable: GitHub repo with a modular retrieval + generation pipeline, example runs (inputs

→ outputs), and evaluation.

Starter references: Classic RAG paper for design choices. arXiv+1

Part B - Paper-reading Task (required)

● Read the RAG NeurIPS 2020 paper and one recent slide-generation or summarization

paper; produce a 5-minute video summarizing how RAG helps reduce hallucinations in

this setting (connect to Part A's provenance and citation-coverage requirements), and

propose one concrete improvement for SARAL's use-case - good candidates include

math-aware retrieval (boosting chunks containing target equations), delta-anchored

re-generation to keep change-tracked edits grounded, or an audience-conditioned

re-ranker.

● Deliverable: 5-minute video + 1-page plan describing retrieval index construction, a

chunking strategy that preserves LaTeX math blocks intact, and a prompt template

family parameterised by {audience} (policymakers, grad students, press), {length} (30s /

90s / 5min), {style} (technical / plain-English / press release), and {change_instruction}

for iterative edits - with at least two instantiated examples.

● PS: Make a ppt/presentation based on your understanding for the interview,

strictly it should be done in 15 minutes.

build a react ui in a frontend folder for this project 

https://lovable.dev/projects/c578f000-5fc6-4596-b640-899aba60ecd1
this link background image i like you can use similar make an interface for this in react with production grade code.
good readability, best practices, and file structure.

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/a10e5b4a-f142-4b43-a46c-536d6f01e329).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

This project requires Node `^20.19.0 || >=22.12.0`. With nvm installed, run `nvm use` in
this directory; the included `.nvmrc` selects Node 22. Node 18 cannot run this project's
Vite/Rolldown version.

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
