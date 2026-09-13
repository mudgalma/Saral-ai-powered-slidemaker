import { createFileRoute } from "@tanstack/react-router";
import { SaralWorkspace } from "@/components/saral-workspace";

// No head() here: the home route inherits title/description/og/twitter from
// __root.tsx, and ships no og:image so serve-time hosting can inject the
// project's social preview (explicit og:image or latest screenshot).
export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "SARAL — Research Communication Assistant" },
      {
        name: "description",
        content:
          "Create audience-adaptive, source-grounded scripts and slide content from research papers.",
      },
      { property: "og:title", content: "SARAL — Research Communication Assistant" },
      {
        property: "og:description",
        content: "Turn research into clear, cited scripts and slides for any audience.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

// IMPORTANT: Replace this placeholder. See ./README.md for routing conventions.
function Index() {
  return <SaralWorkspace />;
}
