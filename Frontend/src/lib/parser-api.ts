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
