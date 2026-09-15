import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
  type CarouselApi,
} from "@/components/ui/carousel";
import { fetchDocumentAsset, type SlideDeck } from "@/lib/parser-api";
import type { SaralVisualAsset } from "@/lib/saral-store";
import { useEffect, useMemo, useState } from "react";

type SlideDeckPreviewProps = {
  deck: SlideDeck;
  visualAssets: SaralVisualAsset[];
};

/** Match each slide with one non-repeated source visual supported by its provenance. */
function selectSlideVisuals(
  deck: SlideDeck,
  visualAssets: SaralVisualAsset[],
): Array<SaralVisualAsset | undefined> {
  const usedAssetIds = new Set<string>();
  return deck.slides.map((slide) => {
    const citedChunkIds = new Set(
      slide.provenance.flatMap((provenance) => provenance.citation_ids),
    );
    const visual = visualAssets.find(
      (asset) => !usedAssetIds.has(asset.assetId) && citedChunkIds.has(asset.sourceChunkId),
    );
    if (visual) usedAssetIds.add(visual.assetId);
    return visual;
  });
}

function SlideSourceVisual({ asset }: { asset: SaralVisualAsset }) {
  const [url, setUrl] = useState<string | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  useEffect(() => {
    let active = true;
    let objectUrl: string | null = null;
    void fetchDocumentAsset(asset.documentId, asset.assetId)
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob);
        if (active) setUrl(objectUrl);
      })
      .catch(() => {
        if (active) setUnavailable(true);
      });
    return () => {
      active = false;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [asset.assetId, asset.documentId]);

  if (!url) {
    return (
      <div className="flex h-full w-full items-center justify-center text-center text-xs text-muted-foreground">
        {unavailable ? "Source visual unavailable" : "Loading…"}
      </div>
    );
  }
  return (
    <img
      alt={asset.caption ?? "Retrieved source visual"}
      className="h-full w-full object-contain"
      src={url}
    />
  );
}

function SlideCanvas({
  slide,
  visual,
}: {
  slide: SlideDeck["slides"][number];
  visual?: SaralVisualAsset;
}) {
  const citations = [...new Set(slide.provenance.flatMap((item) => item.citation_ids))];
  return (
    // Fixed min-height keeps all carousel slides the same size — content clips
    // gracefully with line-clamp rather than making slides jump in height.
    <article className="flex h-[340px] w-full overflow-hidden rounded-lg border border-border bg-card text-card-foreground shadow-sm">
      <div className={`grid h-full flex-1 ${visual ? "grid-cols-[1.2fr_0.8fr]" : "grid-cols-1"}`}>
        {/* Text column: overflow-hidden clips content at the fixed card boundary */}
        <div className="flex min-w-0 flex-col overflow-hidden p-5 sm:p-7">
          <p className="mb-2 shrink-0 text-[10px] font-bold uppercase tracking-[0.14em] text-primary">
            Slide {slide.slide_number}
          </p>
          <h3 className="font-display shrink-0 text-lg font-bold leading-snug sm:text-xl">
            {slide.header_takeaway}
          </h3>
          <ul className="mt-3 min-h-0 flex-1 space-y-2 overflow-hidden text-xs leading-5 sm:mt-5 sm:text-sm sm:leading-6">
            {slide.bullets.map((bullet, index) => (
              <li className="flex gap-2" key={`${slide.slide_number}-${index}`}>
                <span
                  aria-hidden="true"
                  className="mt-[0.45rem] size-1.5 shrink-0 rounded-full bg-primary"
                />
                <span className="line-clamp-2">{bullet}</span>
              </li>
            ))}
          </ul>
          <p className="mt-3 shrink-0 truncate text-[10px] text-muted-foreground">
            {citations.map((id) => `[${id}]`).join(" ")}
          </p>
        </div>

        {/* Image column — stretches to full card height */}
        {visual && (
          <figure className="flex min-w-0 flex-col border-l border-border bg-muted/30">
            <div className="relative flex-1">
              <div className="absolute inset-0 flex items-center justify-center p-4">
                <SlideSourceVisual asset={visual} />
              </div>
            </div>
            <figcaption className="shrink-0 truncate border-t border-border px-3 py-1.5 text-[9px] text-muted-foreground">
              {visual.caption ?? "Source visual"}
            </figcaption>
          </figure>
        )}
      </div>
    </article>
  );
}

/** Render a generated, grounded slide deck without mutating its persisted artifact version. */
export function SlideDeckPreview({ deck, visualAssets }: SlideDeckPreviewProps) {
  const [api, setApi] = useState<CarouselApi>();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const slideVisuals = useMemo(() => selectSlideVisuals(deck, visualAssets), [deck, visualAssets]);
  const selectedSlide = deck.slides[selectedIndex];

  useEffect(() => {
    if (!api) return;
    const updateSelectedIndex = () => setSelectedIndex(api.selectedScrollSnap());
    updateSelectedIndex();
    api.on("select", updateSelectedIndex);
    api.on("reInit", updateSelectedIndex);
    return () => {
      api.off("select", updateSelectedIndex);
      api.off("reInit", updateSelectedIndex);
    };
  }, [api]);

  if (!selectedSlide) return null;
  return (
    <section className="my-5" aria-label={`${deck.title} slide preview`}>
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-xs font-bold uppercase tracking-[0.14em] text-muted-foreground">
          Slide preview
        </p>
        <p className="text-xs text-muted-foreground">
          {selectedIndex + 1} of {deck.slides.length}
        </p>
      </div>
      <Carousel opts={{ loop: false }} setApi={setApi} className="px-10">
        <CarouselContent className="-ml-0">
          {deck.slides.map((slide, index) => (
            <CarouselItem className="pl-0" key={slide.slide_number}>
              <SlideCanvas slide={slide} visual={slideVisuals[index]} />
            </CarouselItem>
          ))}
        </CarouselContent>
        <CarouselPrevious className="left-0" />
        <CarouselNext className="right-0" />
      </Carousel>
      <div className="mt-3 flex gap-1.5 overflow-x-auto pb-1" aria-label="Select slide">
        {deck.slides.map((slide, index) => (
          <button
            aria-current={index === selectedIndex ? "true" : undefined}
            aria-label={`Go to slide ${slide.slide_number}`}
            className={`h-1.5 w-7 shrink-0 rounded-full transition-colors ${index === selectedIndex ? "bg-primary" : "bg-muted-foreground/30 hover:bg-muted-foreground/60"}`}
            key={slide.slide_number}
            onClick={() => api?.scrollTo(index)}
            type="button"
          />
        ))}
      </div>
      <details className="mt-4 border-t border-border pt-3 text-sm">
        <summary className="cursor-pointer font-semibold">Speaker notes and sources</summary>
        <div className="mt-3 space-y-4 text-muted-foreground">
          <section>
            <h4 className="font-semibold text-foreground">Speaker notes</h4>
            <ul className="mt-1 list-disc space-y-1 pl-5">
              {selectedSlide.speaker_notes.map((note, index) => (
                <li key={`${selectedSlide.slide_number}-note-${index}`}>{note}</li>
              ))}
            </ul>
          </section>
          <section>
            <h4 className="font-semibold text-foreground">Speaker script</h4>
            <p className="mt-1 leading-6">{selectedSlide.spoken_script}</p>
          </section>
          <section>
            <h4 className="font-semibold text-foreground">Claim provenance</h4>
            <ul className="mt-1 space-y-1">
              {selectedSlide.provenance.map((item, index) => (
                <li key={`${selectedSlide.slide_number}-source-${index}`}>
                  {item.claim} {item.citation_ids.map((citationId) => `[${citationId}]`).join(" ")}
                </li>
              ))}
            </ul>
          </section>
        </div>
      </details>
    </section>
  );
}
