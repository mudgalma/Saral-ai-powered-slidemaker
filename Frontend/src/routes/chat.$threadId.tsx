import { SaralWorkspace } from "@/components/saral-workspace";
import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/chat/$threadId")({
  head: () => ({
    meta: [
      { title: "Conversation — SARAL" },
      {
        name: "description",
        content: "Refine source-grounded research scripts, bullets, and summaries with SARAL.",
      },
      { property: "og:title", content: "Conversation — SARAL" },
      {
        property: "og:description",
        content: "An audience-adaptive research communication workspace.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: ConversationRoute,
});

function ConversationRoute() {
  const { threadId } = Route.useParams();
  return <SaralWorkspace threadId={threadId} />;
}
