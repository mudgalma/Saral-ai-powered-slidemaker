export type SaralMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  kind?: "answer" | "revision";
};

export type SaralThread = {
  id: string;
  title: string;
  updatedAt: number;
  messages: SaralMessage[];
};

const STORAGE_KEY = "saral-chat-threads-v1";

export const starterThread: SaralThread = {
  id: "rag-methods-talk",
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
    if (Array.isArray(parsed) && parsed.length) return parsed;
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
