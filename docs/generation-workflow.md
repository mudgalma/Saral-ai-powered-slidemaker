# Grounded generation workflow

## Contract and flow

`POST /v1/documents/{document_id}/generate` is deliberately a fixed workflow, not an open-ended
agent. It consumes the existing owner-scoped hybrid retrieval result and never broadens its data
source. Its input is `GenerationRequest` (`artifact_type`, `audience`, `length`, `style`, and
`user_instruction`); its output is either a validated `GeneratedArtifact`, a `flagged` response,
or an `unsupported` response when retrieval provides no evidence.

```text
non-slide request: one dense + sparse RRF retrieval
slide-outline request: five deterministic audience-aware hybrid queries
  -> deduplicate + section coverage selection (maximum eight chunks)
  -> EvidencePack(chunk id, text, pages, heading, provenance, linked source visuals)
  -> build prompt
  -> OpenRouter JSON-schema draft (Pydantic validation)
  -> deterministic grounding check
       -> pass: final artifact + citations
       -> fail and first attempt: corrective regeneration
       -> fail again: flagged, no artifact
```

The model receives evidence in delimited blocks and is told to treat it as data. A response must
use only offered chunk IDs in visible Markdown citations (`[chunk-id]`). The checker rejects a
draft that exceeds its word budget, has uncited prose, uses unknown citations, omits a visible
claim citation, or has no meaningful lexical overlap between a claim and its cited evidence.
Citation display metadata is reconstructed from retrieved evidence, never from model-provided page
or heading values.

### Slide retrieval and source visuals

`slide_outline` requests use deterministic query expansion rather than an additional LLM planning
call. This keeps the retrieval plan inspectable and avoids an extra generation charge. The five
focused hybrid queries cover background, objective, method, results, and implications. The
selection stage reserves evidence for every section, gives results two positions, then fills the
remaining two positions by priority, linked-asset availability, and RRF score.

The selected chunks retain their already-stored `asset_ids`, captions, and page metadata. Those
become `EvidencePack.assets` and, only for completed slide outlines, `artifact.visual_assets`.
The browser fetches each visual through SARAL's owner-scoped asset endpoint; no Supabase storage
path or public/signed URL is returned. Visual assets are source-display metadata, not factual
citations, so claims must still cite chunk IDs. Image bytes are not sent to OpenRouter in this
phase; a later multimodal prompt change can use the bounded selected asset set.

## Documentation decisions

1. [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
   supplies the fixed workflow and conditional pass/regenerate/flag routing pattern. This task does
   not need an agent that chooses tools or parallel workers.
2. [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) supplies the
   `TypedDict` state, isolated node functions, fixed edges, conditional edge, and compilation.
3. [Thinking in LangGraph](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
   supplies the discrete-step and minimal-state design: request/evidence/draft/grounding report and
   attempt count are persisted because downstream nodes need them.
4. [Query transformations](https://www.langchain.com/blog/query-transformations) supports focused
   multi-query retrieval when one user request contains several retrieval subproblems. SARAL uses a
   fixed, audience-aware expansion only for slide outlines; it does not add an unbounded LLM query
   rewriter.
5. [Deconstructing RAG](https://www.langchain.com/blog/deconstructing-rag) informs the decomposition
   into query expansion, hybrid retrieval, deduplication, and bounded post-processing. The final
   EvidencePack remains compact even though retrieval considers a broader candidate set.
6. [LangChain knowledge base / semantic search](https://docs.langchain.com/oss/python/langchain/knowledge-base)
   supports treating a retrieved text chunk plus metadata as the retriever handoff. SARAL reuses its
   already indexed Docling chunks instead of adding a duplicate vector store.
7. [Custom RAG with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag) supplies
   the tight grounded prompt, relevance/grounding check, and bounded retry idea. This implementation
   flags after the retry rather than rewrites the question, preserving the stated retrieval boundary.
8. [Lewis et al., 2020](https://arxiv.org/abs/2005.11401) is the theoretical basis for conditioning
   generation on retrieved non-parametric evidence and retaining provenance.

OpenRouter's [OpenAI SDK compatibility guide](https://openrouter.ai/docs/guides/community/openai-sdk),
[embeddings API](https://openrouter.ai/docs/api/api-reference/embeddings/create-embeddings), and
[structured outputs guide](https://openrouter.ai/docs/guides/features/structured-outputs) additionally
inform the provider boundary. SARAL uses the OpenAI Python SDK pointed at OpenRouter's compatible
base URL, requests JSON-schema output, then validates it with Pydantic.

## File-by-file source traceability

| File | Change | Documentation used |
| --- | --- | --- |
| `pyproject.toml` | Adds Python-3.9-compatible `langgraph==0.6.11`. | LangGraph Graph API; the compatible release line is required because this project supports Python 3.9. |
| `src/saral_parser/models.py` | Adds request, evidence, draft, grounding, citation, and response schemas, plus OpenRouter provider configuration. | Thinking in LangGraph state design; Knowledge Base metadata handoff; OpenRouter embeddings and structured outputs. |
| `src/saral_parser/generation.py` | Implements EvidencePack, deterministic slide query expansion/coverage selection, linked source visual propagation, prompt construction, OpenRouter JSON-schema output, Pydantic validation, deterministic checks, and compiled pass/regenerate/flag graph. | Query Transformations; Deconstructing RAG; Workflows and Agents; Graph API; Custom RAG; Original RAG paper; OpenRouter structured outputs. |
| `src/saral_parser/exceptions.py` | Adds the typed generation boundary error. | Thinking in LangGraph discrete failure boundaries. |
| `src/saral_parser/app.py` | Adds the authenticated, document-readiness-gated `/generate` endpoint and tracing. | Knowledge Base retrieval handoff; Custom RAG grounded generation. |
| `tests/test_generation.py` | Tests pass, corrective retry/flag, unsupported claim detection, slide query coverage, and source-visual propagation without a live model. | Query Transformations; Custom RAG’s grading/retry flow; production test requirements. |
| `tests/test_api.py` | Tests the new endpoint’s artifact and citation response with fake dependencies. | Graph API’s explicit inputs/outputs; Custom RAG. |
| `Frontend/src/lib/parser-api.ts` | Adds typed browser API client/request-response contract and authenticated source-asset fetching. | The backend API contract above; Supabase private-storage delivery guidance; no LangGraph implementation is duplicated in the browser. |
| `Frontend/src/components/saral-workspace.tsx` | Removes placeholder answers and renders API-returned grounded artifacts, citations, and selected private source visuals. | Custom RAG’s grounded-answer handoff; Supabase private-storage delivery guidance. |
| `Frontend/src/lib/saral-store.ts` | Persists selected visual metadata with a local assistant message so it can be displayed again in the active browser workspace. | Browser/UI state contract; the backend remains the authority for each private asset. |
| `.env.backend` | Documents OpenRouter key, model, timeout, and output-token configuration. | OpenRouter compatibility/structured-output documentation and production configuration practice. |
| `README.md` | Documents the endpoint, bounded retry, safe flagging, and workflow reference. | All sources above, summarized for operation. |

## Phase 2: conversation, revisions, and versions

Phase 2 is a second bounded LangGraph workflow. It persists only raw conversation messages and
immutable artifact versions, not prompt strings. Every route still invokes the existing
owner-and-document-scoped hybrid retriever; conversation history never becomes evidence.

```text
message -> load state -> intent -> context resolver -> router
  new generation -> evidence query -> retrieval -> artifact generator
  revision       -> #N/current target -> retrieval -> artifact generator
  question       -> QA retrieval -> answer generator
  all complete artifacts -> grounding check -> immutable version -> delta
```

Intent analysis is deterministic and combines explicit UI choices with bounded message heuristics.
This keeps routing inspectable and prevents an ungrounded model call before retrieval. The optional
query transform is deliberately conservative: revisions add the target artifact title to the
retrieval query, while questions and new artifacts use the user’s bounded message. Multi-query and
step-back retrieval remain deferred until retrieval evaluation demonstrates a need.

The `artifact_versions` database function holds a transaction-scoped advisory lock only while it
calculates and inserts the next per-conversation version number. It is `security invoker`, exposed
to `service_role` only, and the tables grant authenticated users read access only under owner RLS.

### Phase 2 file traceability

| File | Change | Documentation used |
| --- | --- | --- |
| `src/saral_parser/conversation.py` | Adds state loading, intent analysis, `#N`/deictic context resolution, three conditional graph branches, retrieval-backed generation, immutable version persistence, and unified deltas. | LangGraph Workflows and Agents; Graph API; Thinking in LangGraph; Query Transformations; Deconstructing RAG; Custom RAG. |
| `src/saral_parser/models.py` | Adds validated conversation requests, raw conversation state, intents, immutable version/response schemas, tweet-thread output, and OpenRouter configuration. | Thinking in LangGraph state design; OpenRouter structured outputs. |
| `src/saral_parser/generation.py` | Allows the Phase 2 revision route to pass a bounded retrieval query and prior artifact as data while preserving the same EvidencePack and grounding workflow. | Custom RAG grounded prompting; Original RAG paper. |
| `src/saral_parser/app.py` | Adds authenticated, document-ready `POST /conversations/messages`. | LangGraph Graph API interfaces; Custom RAG workflow boundary. |
| `src/saral_parser/persistence.py` | Adds owner/document-scoped conversation reads/writes and atomic version-function RPC. | Thinking in LangGraph persistent raw state; Supabase table and RLS guidance. |
| `supabase/schema.sql` | Adds conversations, messages, versions, indexes for foreign keys/query patterns, RLS read policies, least-privilege grants, and the service-role-only advisory-lock version function. | Supabase RLS; Supabase Postgres foreign-key, composite-index, RLS-performance, and advisory-lock guidance. |
| `tests/test_conversation.py` | Covers a new artifact, revision delta, explicit `#N`, missing target, and question routing without live providers. | Workflows and Agents routing; Custom RAG retry/grounding test discipline. |
| `tests/test_api.py` | Covers the conversation endpoint’s routed response. | Graph API explicit input/output contract. |
| `Frontend/src/lib/parser-api.ts` | Adds typed conversation request/response client. | Backend conversation contract; no browser-side orchestration. |
| `Frontend/src/components/saral-workspace.tsx` | Routes ordinary turns through the Phase 2 endpoint and displays version/delta status. | Workflows and Agents routing; Thinking in LangGraph state ownership. |
| `Frontend/src/lib/saral-store.ts` | Migrates legacy local thread IDs to UUIDs required by persisted backend conversation state. | Backend conversation identity contract. |
| `README.md` | Documents the Phase 2 request endpoint and version behavior. | All Phase 2 sources above. |
