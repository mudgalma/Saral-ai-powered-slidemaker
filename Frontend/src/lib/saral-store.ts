import { nanoid } from "nanoid";
import { v4 as uuidv4 } from "uuid";
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
      const threads = parsed.map((thread) => ({
        ...thread,
        id: thread.id || uuidv4(),
        messages: thread.messages || [],
      }));
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
      return threads;
    }
  } catch {
    // Ignore malformed browser data and restore the starter workspace.
  }
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify([starterThread]));
  return [starterThread];
}

const getEventTarget = () => {
  if (typeof window === "undefined") return new EventTarget();
  if (!("saralStoreEventTarget" in window)) {
    (window as any).saralStoreEventTarget = new EventTarget();
  }
  return (window as any).saralStoreEventTarget as EventTarget;
};

export const storeEventTarget = getEventTarget();

export function saveThreads(threads: SaralThread[]) {
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
    storeEventTarget.dispatchEvent(new Event("threads_updated"));
  }
}

export function createThread(): SaralThread {
  return {
    id: uuidv4(),
    title: "Untitled conversation",
    updatedAt: Date.now(),
    messages: [],
  };
}
