// Prepares the answering model's Markdown for rendering with remark-math/rehype-katex, and turns its
// "[3]" citations into links that the chat renders as clickable chips

const CITE_HREF = "#cite-";

// Code (fenced blocks, also an unclosed one while streaming, and inline spans) is never rewritten
const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/;
const MATH = /(\$\$[\s\S]*?\$\$|(?<!\\)\$[^$\n]+?(?<!\\)\$)/;

/** Outside code: \( \) and \[ \] become $ and $$ (the only math delimiters remark-math reads), "$5 and $10"
 * stays money instead of math, and "[3]" citations outside math become citation links. */
export function prepareAnswerMarkdown(md: string): string {
  return md
    .split(CODE)
    .map((part, i) => {
      if (i % 2) return part;
      const text = part
        .replace(/\\\[([\s\S]*?)\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]*?)\\\)/g, (_, m) => `$${m.trim()}$`)
        .replace(/(?<![\\$])\$(?=\d[\d.,]*(?:[\s;:!?)\]]|$))/g, "\\$");
      return text
        .split(MATH)
        .map((t, j) => (j % 2 ? t : t.replace(/\[(\d{1,2})\](?!\()/g, `[[$1]](${CITE_HREF}$1)`)))
        .join("");
    })
    .join("");
}

/** The passage number of a citation link made by prepareAnswerMarkdown, or null for any other link. */
export function citedPassage(href: string | undefined): number | null {
  return href?.startsWith(CITE_HREF) ? Number(href.slice(CITE_HREF.length)) : null;
}
