import type { Element, ElementContent, Root, Text } from "hast";

// Blocks that lie entirely inside the cited chunk are highlighted whole; text in a block the chunk only
// partly covers (a chunk can start mid-paragraph because of the overlap) is highlighted word-exactly
const BLOCKS = new Set(["p", "li", "pre", "table", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"]);

function markText(node: Text, start: number, end: number): ElementContent[] {
  const s = node.position?.start.offset;
  const e = node.position?.end.offset;
  if (s == null || e == null || e <= start || s >= end) return [node];
  const mark = (value: string): Element => ({
    type: "element",
    tagName: "mark",
    properties: { className: ["hl"] },
    children: [{ type: "text", value }],
  });
  // Source positions map 1:1 onto the text only when it is written literally (no escapes or entities)
  if (e - s !== node.value.length) return [mark(node.value)];
  const a = Math.max(start - s, 0);
  const b = Math.min(end - s, node.value.length);
  const parts: ElementContent[] = [];
  if (a > 0) parts.push({ type: "text", value: node.value.slice(0, a) });
  parts.push(mark(node.value.slice(a, b)));
  if (b < node.value.length) parts.push({ type: "text", value: node.value.slice(b) });
  return parts;
}

// Rehype plugin: highlight the Markdown source range [start, end) using the source positions of the syntax tree
export default function rehypeMarkRange({ start, end }: { start: number; end: number }) {
  const walk = (node: Root | Element) => {
    node.children = (node.children as ElementContent[]).flatMap((child) => {
      if (child.type === "text") return markText(child, start, end);
      if (child.type !== "element") return [child];
      const s = child.position?.start.offset;
      const e = child.position?.end.offset;
      if (s != null && e != null) {
        if (e <= start || s >= end) return [child];
        if (s >= start && e <= end && BLOCKS.has(child.tagName)) {
          const cls = child.properties.className;
          child.properties.className = [...(Array.isArray(cls) ? cls : cls ? [String(cls)] : []), "hl-block"];
          return [child];
        }
      }
      walk(child);
      return [child];
    });
  };
  return (tree: Root) => walk(tree);
}
