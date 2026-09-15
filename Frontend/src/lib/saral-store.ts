import type { SlideDeck, ParserManifest } from "@/lib/parser-api";

export type SaralMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  kind?: "answer" | "revision";
  visualAssets?: SaralVisualAsset[];
  deck?: SlideDeck;
};

export type SaralVisualAsset = {
  documentId: string;
  assetId: string;
  sourceChunkId: string;
  pageNumbers: number[];
  caption: string | null;
};

export type SaralThread = {
  id: string;
  title: string;
  updatedAt: number;
  messages: SaralMessage[];
  /** Persisted document manifest so the paper link survives page refreshes. */
  source?: ParserManifest;
};

const STORAGE_KEY = "saral-chat-threads-v1";
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const starterThread: SaralThread = {
  id: "00000000-0000-4000-8000-000000000001",
  title: "RAG methods talk",
  updatedAt: Date.now(),
  messages: [
    {
      id: "welcome",
      role: "assistant",
      kind: "answer",
      text: "I’ve indexed **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** — 12 pages, 48 source chunks, including 6 equations.\n\nWhat would you like to create?",
    },
  ],
};

export function loadThreads(): SaralThread[] {
  if (typeof window === "undefined") return [starterThread];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "[]") as SaralThread[];
    if (Array.isArray(parsed) && parsed.length) {
      const threads = parsed.map((thread) =>
        UUID_PATTERN.test(thread.id) ? thread : { ...thread, id: crypto.randomUUID() },
      );
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
      return threads;
    }
  } catch {
    // Ignore malformed browser data and restore the starter workspace.
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify([starterThread]));
  return [starterThread];
}

export function saveThreads(threads: SaralThread[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
}

export function createThread(): SaralThread {
  return {
    id: crypto.randomUUID(),
    title: "Untitled conversation",
    updatedAt: Date.now(),
    messages: [],
  };
}
