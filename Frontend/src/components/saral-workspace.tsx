"use client";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import {
  PromptInput,
  PromptInputButton,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  type PromptInputMessage,
  usePromptInputAttachments,
} from "@/components/ai-elements/prompt-input";
import { Button } from "@/components/ui/button";
import backgroundArt from "@/assets/background.jpg.asset.json";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  createThread,
  loadThreads,
  saveThreads,
  starterThread,
  type SaralMessage,
  type SaralThread,
} from "@/lib/saral-store";
import { type ParserManifest, uploadPaper } from "@/lib/parser-api";
import { Link, useNavigate } from "@tanstack/react-router";
import {
  BookOpenText,
  Check,
  ChevronLeft,
  Clock3,
  FileText,
  Menu,
  MessageSquareText,
  MoreHorizontal,
  Paperclip,
  Plus,
  Search,
  ShieldCheck,
  Sparkle,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

type SaralWorkspaceProps = { threadId?: string };

const suggestions = [
  "Create a 90-second script for policymakers",
  "Make a 7-slide methods talk for grad students",
  "Write a plain-English thread with citations",
];

function answerFor(prompt: string, revision: boolean): string {
  if (revision) {
    return "### Revised slide 2\n\n**Before**\n- The retriever uses Maximum Inner Product Search over dense passage embeddings.\n\n**After**\n- The system first finds the most relevant passages, then uses them to write a grounded answer. **[p. 4, §3.1]**\n\n**Why it changed**\nI replaced specialist retrieval terminology with a concrete two-step explanation while preserving the original claim and citation.";
  }
  return "## 7-slide talk · Graduate students · 5–7 min\n\n### 1. Why retrieval matters\n- Language models can sound confident without grounding.\n- RAG retrieves evidence before generating each response. **[p. 2, §1]**\n\n**Speaker notes**\n1. Frame hallucination as a knowledge-access problem.\n2. Contrast parametric memory with an external index.\n3. Preview the pipeline: *retrieve → condition → generate*.\n\n### 2. The core method\n- A dense retriever ranks passages using $p_\\eta(z\\mid x)$.\n- The generator predicts tokens conditioned on retrieved evidence: $p_\\theta(y_i\\mid x,z,y_{1:i-1})$. **[p. 4, §3]**\n\n**Speaker script**\nRAG combines two kinds of memory. The model retains learned language knowledge, while the retrieval index provides inspectable, updateable evidence. For each input, relevant passages are selected and supplied to the generator, reducing unsupported claims and making provenance visible.";
}

function parserResultMessage(manifest: ParserManifest): string {
  const { counts, input } = manifest;
  const warnings = manifest.warnings.length
    ? `\n\n**Review notes**\n${manifest.warnings.map((warning) => `- ${warning}`).join("\n")}`
    : "";
  return `## Paper parsed\n\n**${input.original_filename}** is ready for inspection.\n\n- ${counts.pages ?? 0} pages\n- ${counts.pictures ?? 0} figures\n- ${counts.tables ?? 0} tables\n- ${counts.formulas ?? 0} formulas with LaTeX\n- Document ID: \`${manifest.document_id}\`${warnings}`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function AttachmentButton() {
  const attachments = usePromptInputAttachments();
  const paper = attachments.files[0];
  return (
    <div className="flex min-w-0 items-center gap-1">
      <PromptInputButton
        aria-label="Attach a PDF paper"
        onClick={attachments.openFileDialog}
        tooltip="Attach a PDF paper"
      >
        <Paperclip />
      </PromptInputButton>
      {paper && (
        <>
          <span className="max-w-40 truncate text-xs text-foreground" title={paper.filename}>
            {paper.filename}
          </span>
          <PromptInputButton
            aria-label={`Remove ${paper.filename}`}
            onClick={() => attachments.remove(paper.id)}
            tooltip="Remove selected PDF"
          >
            <X />
          </PromptInputButton>
        </>
      )}
    </div>
  );
}

export function SaralWorkspace({ threadId }: SaralWorkspaceProps) {
  const navigate = useNavigate();
  const [threads, setThreads] = useState<SaralThread[]>([starterThread]);
  const [mobileNav, setMobileNav] = useState(false);
  const [audience, setAudience] = useState("Graduate students");
  const [length, setLength] = useState("5 minutes");
  const [style, setStyle] = useState("Technical");
  const [status, setStatus] = useState<"ready" | "submitted">("ready");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [source, setSource] = useState<ParserManifest | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const active = threads.find((thread) => thread.id === threadId);

  useEffect(() => textareaRef.current?.focus(), [threadId, status]);
  useEffect(() => setThreads(loadThreads()), []);

  const persist = useCallback((updater: (current: SaralThread[]) => SaralThread[]) => {
    setThreads((current) => {
      const next = updater(current);
      saveThreads(next);
      return next;
    });
  }, []);

  const handleNew = () => {
    const thread = createThread();
    persist((current) => [thread, ...current]);
    void navigate({ to: "/chat/$threadId", params: { threadId: thread.id } });
    setMobileNav(false);
  };

  const submit = async (message: PromptInputMessage) => {
    const text = message.text.trim();
    const paper = message.files[0];
    if (!text && !paper) return;
    if (message.files.length > 1) return;
    setUploadError(null);
    let currentId = threadId;
    if (!currentId) {
      const thread = createThread();
      currentId = thread.id;
      persist((current) => [thread, ...current]);
      await navigate({ to: "/chat/$threadId", params: { threadId: currentId } });
    }
    const targetId = currentId;
    const userMessage: SaralMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: text || `Upload ${paper?.filename ?? "paper"}`,
    };
    persist((current) =>
      current.map((thread) =>
        thread.id === targetId
          ? {
              ...thread,
              title: thread.messages.length ? thread.title : text.slice(0, 34),
              updatedAt: Date.now(),
              messages: [...thread.messages, userMessage],
            }
          : thread,
      ),
    );
    setStatus("submitted");
    if (paper) {
      try {
        const manifest = await uploadPaper(paper);
        setSource(manifest);
        const parserMessage: SaralMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "answer",
          text: parserResultMessage(manifest),
        };
        persist((current) =>
          current.map((thread) =>
            thread.id === targetId
              ? { ...thread, updatedAt: Date.now(), messages: [...thread.messages, parserMessage] }
              : thread,
          ),
        );
      } catch (error) {
        const messageText = error instanceof Error ? error.message : "Paper upload failed.";
        const failureMessage: SaralMessage = {
          id: crypto.randomUUID(),
          role: "assistant",
          kind: "answer",
          text: `## Could not parse the paper\n\n${messageText}`,
        };
        persist((current) =>
          current.map((thread) =>
            thread.id === targetId
              ? { ...thread, updatedAt: Date.now(), messages: [...thread.messages, failureMessage] }
              : thread,
          ),
        );
      } finally {
        setStatus("ready");
      }
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, 650));
    const revision = /less technical|dumb down|simpl|visual|revise|change/i.test(text);
    const assistantMessage: SaralMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      kind: revision ? "revision" : "answer",
      text: answerFor(text, revision),
    };
    persist((current) =>
      current.map((thread) =>
        thread.id === targetId
          ? {
              ...thread,
              updatedAt: Date.now(),
              messages: [...thread.messages, assistantMessage],
            }
          : thread,
      ),
    );
    setStatus("ready");
  };

  return (
    <TooltipProvider>
      <main className="relative flex h-screen min-h-[680px] overflow-hidden bg-canvas text-foreground">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-cover bg-center opacity-[0.14]"
          style={{ backgroundImage: `url(${backgroundArt.url})` }}
        />
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 bg-gradient-to-b from-canvas/60 via-canvas/75 to-canvas/90"
        />
        {mobileNav && (
          <div
            className="fixed inset-0 z-30 bg-foreground/25 lg:hidden"
            onClick={() => setMobileNav(false)}
          />
        )}
        <aside
          className={`fixed inset-y-0 left-0 z-40 flex w-[276px] flex-col border-r border-sidebar-border bg-sidebar/90 backdrop-blur-sm transition-transform lg:static ${mobileNav ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}
        >
          <div className="flex h-16 items-center gap-3 border-b border-sidebar-border px-5">
            <div className="flex size-8 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <BookOpenText className="size-4" />
            </div>
            <div>
              <div className="font-display text-base font-bold">SARAL</div>
              <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                Research communicator
              </div>
            </div>
          </div>
          <div className="p-4">
            <Button className="w-full justify-start shadow-none" onClick={handleNew}>
              <Plus /> New conversation
            </Button>
          </div>
          <div className="px-4 pb-2 text-[11px] font-bold uppercase tracking-[0.14em] text-muted-foreground">
            Recent
          </div>
          <nav className="flex-1 space-y-1 overflow-y-auto px-3">
            {threads.map((thread) => (
              <div key={thread.id} className="group flex items-center gap-1">
                <Link
                  to="/chat/$threadId"
                  params={{ threadId: thread.id }}
                  onClick={() => setMobileNav(false)}
                  className={`flex min-w-0 flex-1 items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors ${thread.id === threadId ? "bg-sidebar-accent font-semibold text-sidebar-accent-foreground" : "text-sidebar-foreground hover:bg-sidebar-accent/60"}`}
                >
                  <MessageSquareText className="size-4 shrink-0" />
                  <span className="truncate">{thread.title}</span>
                </Link>
                <Button
                  aria-label={`More options for ${thread.title}`}
                  className="size-8 opacity-0 group-hover:opacity-100"
                  size="icon"
                  variant="ghost"
                >
                  <MoreHorizontal />
                </Button>
              </div>
            ))}
          </nav>
          <div className="m-4 rounded-md border border-border bg-card p-3">
            <div className="mb-1 flex items-center gap-2 text-xs font-semibold">
              <ShieldCheck className="size-4 text-primary" /> Grounded mode on
            </div>
            <p className="text-[11px] leading-4 text-muted-foreground">
              Claims are traced to uploaded sources.
            </p>
          </div>
        </aside>

        <section className="relative flex min-w-0 flex-1 flex-col bg-background/85 backdrop-blur-sm">
          <header className="flex h-16 shrink-0 items-center justify-between border-b border-border px-4 sm:px-6">
            <div className="flex min-w-0 items-center gap-3">
              <Button
                aria-label="Open conversations"
                className="lg:hidden"
                onClick={() => setMobileNav(true)}
                size="icon"
                variant="ghost"
              >
                <Menu />
              </Button>
              <div className="min-w-0">
                <h1 className="truncate text-sm font-bold sm:text-base">
                  {active?.title ?? "New research conversation"}
                </h1>
                <p className="text-xs text-muted-foreground">
                  {source
                    ? `${source.counts.pages ?? 0} pages · parser output ready`
                    : "Upload one PDF paper to parse"}
                </p>
              </div>
            </div>
            <Button aria-label="Search conversation" size="icon" variant="ghost">
              <Search />
            </Button>
          </header>

          <div className="flex min-h-0 flex-1">
            <div className="flex min-w-0 flex-1 flex-col">
              <Conversation className="min-h-0">
                <ConversationContent className="mx-auto w-full max-w-3xl gap-6 px-5 py-8 sm:px-8">
                  {!active?.messages.length ? (
                    <div className="flex min-h-[420px] flex-col items-center justify-center text-center">
                      <div className="mb-5 flex size-14 items-center justify-center rounded-md border border-border bg-card text-primary shadow-sm">
                        <BookOpenText className="size-6" />
                      </div>
                      <h2 className="text-2xl font-bold">Inspect a research paper</h2>
                      <p className="mt-2 max-w-md text-sm leading-6 text-muted-foreground">
                        Upload one PDF paper. SARAL validates it, parses text, tables, figures, and
                        formulas, then returns a document ID and inspection-ready outputs.
                      </p>
                      <div className="mt-6 flex max-w-xl flex-wrap justify-center gap-2">
                        {suggestions.map((item) => (
                          <Button
                            key={item}
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              if (textareaRef.current) {
                                textareaRef.current.value = item;
                                textareaRef.current.focus();
                              }
                            }}
                          >
                            {item}
                          </Button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    active.messages.map((message) => (
                      <Message className="animate-rise-in" from={message.role} key={message.id}>
                        {message.role === "assistant" && (
                          <div className="mb-1 flex items-center gap-2 text-xs font-bold text-primary">
                            <div className="flex size-6 items-center justify-center rounded-md bg-primary text-primary-foreground">
                              <Sparkle className="size-3" />
                            </div>
                            SARAL
                          </div>
                        )}
                        <MessageContent
                          className={
                            message.role === "assistant"
                              ? "w-full text-[15px] leading-7"
                              : "bg-primary text-primary-foreground"
                          }
                        >
                          <MessageResponse>{message.text}</MessageResponse>
                        </MessageContent>
                        {message.kind === "revision" && (
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Check className="size-3.5 text-success" /> Change tracked · source
                            grounding preserved
                          </div>
                        )}
                      </Message>
                    ))
                  )}
                  {status === "submitted" && (
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Sparkle className="size-4 text-primary" /> Retrieving evidence and drafting…
                    </div>
                  )}
                </ConversationContent>
                <ConversationScrollButton />
              </Conversation>

              <div className="border-t border-border bg-background px-4 py-4 sm:px-8">
                <div className="mx-auto max-w-3xl">
                  <div className="mb-3 flex gap-2 overflow-x-auto pb-1">
                    <Select value={audience} onValueChange={setAudience}>
                      <SelectTrigger className="h-8 w-[170px] bg-card text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Graduate students">Graduate students</SelectItem>
                        <SelectItem value="Policymakers">Policymakers</SelectItem>
                        <SelectItem value="Press">Press</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={length} onValueChange={setLength}>
                      <SelectTrigger className="h-8 w-[130px] bg-card text-xs">
                        <Clock3 className="mr-1 size-3.5" />
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="30 seconds">30 seconds</SelectItem>
                        <SelectItem value="90 seconds">90 seconds</SelectItem>
                        <SelectItem value="5 minutes">5 minutes</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select value={style} onValueChange={setStyle}>
                      <SelectTrigger className="h-8 w-[130px] bg-card text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Technical">Technical</SelectItem>
                        <SelectItem value="Plain English">Plain English</SelectItem>
                        <SelectItem value="Press release">Press release</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <PromptInput
                    accept="application/pdf,.pdf"
                    maxFiles={1}
                    maxFileSize={50 * 1024 * 1024}
                    onError={(error) => setUploadError(error.message)}
                    onSubmit={submit}
                    className="rounded-lg bg-card shadow-[0_8px_30px_oklch(0.2_0.02_255/0.08)]"
                  >
                    <PromptInputTextarea
                      ref={textareaRef}
                      disabled={status === "submitted"}
                      placeholder="Ask SARAL to create or revise something…"
                      className="min-h-20"
                    />
                    <PromptInputFooter>
                      <PromptInputTools>
                        <AttachmentButton />
                        <span className="hidden text-xs text-muted-foreground sm:inline">
                          One PDF · up to 50 MB
                        </span>
                      </PromptInputTools>
                      <PromptInputSubmit
                        disabled={status === "submitted"}
                        status={status === "submitted" ? "submitted" : "ready"}
                      />
                    </PromptInputFooter>
                  </PromptInput>
                  <p className="mt-2 text-center text-[11px] text-muted-foreground">
                    The parser preserves provenance where available. Review extracted tables,
                    figures, and formulas against the source PDF.
                  </p>
                  {uploadError && (
                    <p className="mt-1 text-center text-[11px] text-destructive" role="alert">
                      {uploadError}
                    </p>
                  )}
                </div>
              </div>
            </div>

            <aside className="relative hidden w-[300px] shrink-0 border-l border-border bg-canvas/70 p-5 backdrop-blur-sm xl:block">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-xs font-bold uppercase tracking-[0.14em] text-muted-foreground">
                  Sources
                </h2>
                <Button aria-label="Collapse sources" size="icon" variant="ghost">
                  <ChevronLeft />
                </Button>
              </div>
              <div className="rounded-md border border-border bg-card p-4 shadow-sm">
                <div className="mb-3 flex items-start gap-3">
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-md bg-secondary text-primary">
                    <FileText className="size-4" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate text-sm font-bold">
                      {source?.input.original_filename ?? "No paper uploaded"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {source
                        ? `${source.counts.pages ?? 0} pages · ${formatBytes(source.input.size_bytes)}`
                        : "Select one PDF to begin"}
                    </p>
                  </div>
                </div>
                <div className="mb-2 h-1.5 overflow-hidden rounded-full bg-muted">
                  <div className={`h-full ${source ? "w-full bg-success" : "w-0 bg-muted"}`} />
                </div>
                <div
                  className={`flex items-center gap-1.5 text-xs font-semibold ${source ? "text-success" : "text-muted-foreground"}`}
                >
                  <Check className="size-3.5" /> {source ? "Parsed" : "Awaiting upload"}
                </div>
              </div>
              <div className="mt-5 rounded-md border border-border bg-card p-4">
                <h3 className="mb-3 text-sm font-bold">Parser output</h3>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Figures</span>
                    <strong>{source?.counts.pictures ?? "—"}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Tables</span>
                    <strong>{source?.counts.tables ?? "—"}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Formula items</span>
                    <strong>{source?.counts.formulas ?? "—"}</strong>
                  </div>
                  {source && (
                    <p className="break-all pt-1 text-[10px] text-muted-foreground">
                      {source.document_id}
                    </p>
                  )}
                </div>
              </div>
            </aside>
          </div>
        </section>
      </main>
    </TooltipProvider>
  );
}
