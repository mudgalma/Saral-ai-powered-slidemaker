import type { FileUIPart } from "ai";

type UploadPart = FileUIPart & { sourceFile?: File };

const maxUploadBytes = 50 * 1024 * 1024;

export type ParserManifest = {
  document_id: string;
  status: "queued" | "processing" | "ready" | "failed";
  input: { original_filename: string; size_bytes: number };
  counts: { pages?: number; pictures?: number; tables?: number; formulas?: number };
  warnings: string[];
  errors: string[];
  chunk_count: number;
  embedding_status: "not_started" | "queued" | "processing" | "ready" | "failed";
};

export type GenerationRequest = {
  artifact_type:
    "answer" | "summary" | "script" | "slide_outline" | "tweet_thread" | "linkedin_post";
  audience: string;
  length: "brief" | "standard" | "extended";
  style: string;
  user_instruction: string;
  slide_count?: number | null;
};

export type SlideDeck = {
  title: string;
  slides: Array<{
    slide_number: number;
    role: string;
    header_takeaway: string;
    bullets: string[];
    speaker_notes: string[];
    spoken_script: string;
    provenance: Array<{ claim: string; citation_ids: string[] }>;
  }>;
};

export type GenerationResponse = {
  document_id: string;
  status: "complete" | "flagged" | "unsupported";
  artifact: null | {
    artifact_type: GenerationRequest["artifact_type"];
    title: string;
    content: string;
    citations: Array<{ chunk_id: string; page_numbers: number[]; heading: string | null }>;
    visual_assets: Array<{
      asset_id: string;
      source_chunk_id: string;
      page_numbers: number[];
      caption: string | null;
    }>;
    deck: SlideDeck | null;
  };
  grounding: { passed: boolean; issues: string[]; cited_chunk_ids: string[] };
  attempts: number;
};

export type ConversationResponse = {
  thread_id: string;
  document_id: string;
  intent: {
    branch: "new_generation" | "revision" | "question";
    artifact_type: GenerationRequest["artifact_type"];
    audience: string;
    length: GenerationRequest["length"];
    style: string;
    is_revision: boolean;
  };
  generation: GenerationResponse;
  version: null | {
    id: string;
    version_number: number;
    parent_version_id: string | null;
    delta: string | null;
  };
};

type AcceptedDocument = {
  document_id: string;
  job_id: string;
  status: "queued";
  status_url: string;
};

const parserApiBaseUrl = (
  import.meta.env["VITE_SARAL_PARSER_API_URL"] ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");
const parserOptions = {
  enable_formula_enrichment: true,
  extract_table_structure: true,
  generate_page_images: true,
  generate_picture_images: true,
};
const chunkOptions = { max_tokens: 512, merge_peers: true };

function authHeaders(): Record<string, string> {
  // A future Supabase Auth UI should set its current short-lived session token here.
  const accessToken = window.sessionStorage.getItem("saral_supabase_access_token");
  return accessToken ? { Authorization: `Bearer ${accessToken}` } : {};
}

export async function uploadPaper(part: UploadPart): Promise<ParserManifest> {
  const file = await toFile(part);
  if (!file.name.toLowerCase().endsWith(".pdf") && file.type !== "application/pdf") {
    throw new Error("SARAL's parser currently accepts PDF papers only.");
  }
  if (file.size === 0) {
    throw new Error("The selected PDF is empty.");
  }
  if (file.size > maxUploadBytes) {
    throw new Error("The selected PDF exceeds the 50 MB upload limit.");
  }
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("options", JSON.stringify(parserOptions));
  formData.append("chunk_options", JSON.stringify(chunkOptions));
  let response: Response;
  try {
    response = await fetch(`${parserApiBaseUrl}/v1/documents`, {
      method: "POST",
      body: formData,
      headers: authHeaders(),
    });
  } catch {
    throw new Error(
      "Could not reach the parser service. Check the parser URL and its allowed frontend origin.",
    );
  }
  const accepted = await readJson<AcceptedDocument>(response);
  return pollDocument(accepted.document_id, file.size);
}

export async function generateArtifact(
  documentId: string,
  request: GenerationRequest,
): Promise<GenerationResponse> {
  let response: Response;
  try {
    response = await fetch(`${parserApiBaseUrl}/v1/documents/${documentId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify(request),
    });
  } catch {
    throw new Error("Could not reach the grounded-generation service.");
  }
  return readJson<GenerationResponse>(response);
}

export async function sendConversationMessage(
  documentId: string,
  threadId: string,
  message: string,
  options: Pick<GenerationRequest, "audience" | "length" | "style">,
): Promise<ConversationResponse> {
  let response: Response;
  try {
    response = await fetch(
      `${parserApiBaseUrl}/v1/documents/${documentId}/conversations/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ thread_id: threadId, message, ...options }),
      },
    );
  } catch {
    throw new Error("Could not reach the document-conversation service.");
  }
  return readJson<ConversationResponse>(response);
}

export async function fetchDocumentAsset(documentId: string, assetId: string): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${parserApiBaseUrl}/v1/documents/${documentId}/assets/${assetId}`, {
      headers: authHeaders(),
    });
  } catch {
    throw new Error("Could not retrieve the selected source visual.");
  }
  if (!response.ok) {
    throw new Error("The selected source visual is unavailable.");
  }
  return response.blob();
}

async function pollDocument(documentId: string, sizeBytes: number): Promise<ParserManifest> {
  const deadline = Date.now() + 60 * 60 * 1000;
  while (Date.now() < deadline) {
    const response = await fetch(`${parserApiBaseUrl}/v1/documents/${documentId}`, {
      headers: authHeaders(),
    });
    const status = await readJson<Omit<ParserManifest, "input"> & { original_filename: string }>(
      response,
    );
    const result: ParserManifest = {
      ...status,
      input: { original_filename: status.original_filename, size_bytes: sizeBytes },
    };
    if (result.status === "ready") return result;
    if (result.status === "failed") throw new Error(result.errors[0] ?? "Paper processing failed.");
    await new Promise((resolve) => window.setTimeout(resolve, 2000));
  }
  throw new Error("Paper processing did not finish within one hour.");
}

async function readJson<T extends object>(response: Response): Promise<T> {
  const payload = (await response.json().catch(() => null)) as T | { detail?: string } | null;
  if (!response.ok) {
    if (response.status === 401) {
      throw new Error("Sign in is required before uploading a paper.");
    }
    const message = payload && "detail" in payload ? payload.detail : "Paper request failed.";
    throw new Error(message ?? "Paper request failed.");
  }
  return payload as T;
}

async function toFile(part: UploadPart): Promise<File> {
  if (part.sourceFile) {
    return part.sourceFile;
  }
  const response = await fetch(part.url);
  if (!response.ok) throw new Error(`Could not read ${part.filename} from the browser attachment.`);
  const blob = await response.blob();
  return new File([blob], part.filename ?? "paper.pdf", { type: part.mediaType || blob.type });
}
