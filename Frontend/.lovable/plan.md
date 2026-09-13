# SARAL Recruitment Prototype

## Product
- Build a focused research-communication workspace with threaded conversations saved in the browser.
- Support PDF, LaTeX, and slide uploads in the composer.
- Let users choose audience, output length, and writing style before each request.
- Show source-grounded example outputs, inline citations, retrieval quality, and revision deltas.

## Experience
- Use a three-pane desktop workspace: conversations, chat, and source evidence.
- Collapse to a readable mobile conversation view with an accessible navigation drawer.
- Keep the visual language editorial and research-focused: warm canvas, crisp white surfaces, green grounding cues, and compact typography.

## Technical details
- Use TanStack file routes for stable `/chat/:threadId` conversation URLs.
- Persist per-thread messages and metadata in localStorage.
- Compose the transcript and prompt from AI Elements primitives.
- Keep this delivery frontend-only; the sample retrieval response demonstrates the intended production contract.
