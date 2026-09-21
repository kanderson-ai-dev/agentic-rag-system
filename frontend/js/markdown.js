// XSS-safe Markdown renderer. Builds DOM nodes with `textContent` / safe tag
// construction only — it never parses untrusted strings through innerHTML, so
// LLM output cannot inject <script>/event handlers. Raw HTML in the answer is
// always treated as literal text, never as markup.
//
// Supported subset: fenced ```code```, inline `code`, headings (#..######),
// unordered (-/*) and ordered (1.) lists, blockquotes (>), paragraphs, and
// inline **bold** / *italic*.

function renderMarkdown(source) {
  const root = document.createDocumentFragment();
  const lines = String(source ?? "").split(/\r?\n/);
  let i = 0;

  const inline = (text) => {
    const frag = document.createDocumentFragment();
    // Bold before italic so `**a *b* c**` doesn't mis-nest. Regex alternates
    // **bold**, *italic*, and inline `code`.
    const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    let last = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
      if (match.index > last) {
        frag.appendChild(document.createTextNode(text.slice(last, match.index)));
      }
      const token = match[0];
      if (token.startsWith("**")) {
        const el = document.createElement("strong");
        el.textContent = token.slice(2, -2);
        frag.appendChild(el);
      } else if (token.startsWith("`")) {
        const el = document.createElement("code");
        el.textContent = token.slice(1, -1);
        frag.appendChild(el);
      } else {
        const el = document.createElement("em");
        el.textContent = token.slice(1, -1);
        frag.appendChild(el);
      }
      last = match.index + token.length;
    }
    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }
    return frag;
  };

  const newEl = (tag) => document.createElement(tag);

  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block: ``` (optionally with a language).
    if (/^\s*```/.test(line)) {
      const codeLines = [];
      i += 1;
      while (i < lines.length && !/^\s*```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1; // skip the closing fence
      const pre = newEl("pre");
      const code = newEl("code");
      code.textContent = codeLines.join("\n");
      pre.appendChild(code);
      root.appendChild(pre);
      continue;
    }

    // Blank line separates blocks.
    if (/^\s*$/.test(line)) {
      i += 1;
      continue;
    }

    // Headings.
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      const el = newEl(`h${heading[1].length}`);
      el.appendChild(inline(heading[2]));
      root.appendChild(el);
      i += 1;
      continue;
    }

    // Unordered list.
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = newEl("ul");
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const li = newEl("li");
        li.appendChild(inline(lines[i].replace(/^\s*[-*]\s+/, "")));
        ul.appendChild(li);
        i += 1;
      }
      root.appendChild(ul);
      continue;
    }

    // Ordered list.
    if (/^\s*\d+\.\s+/.test(line)) {
      const ol = newEl("ol");
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const li = newEl("li");
        li.appendChild(inline(lines[i].replace(/^\s*\d+\.\s+/, "")));
        ol.appendChild(li);
        i += 1;
      }
      root.appendChild(ol);
      continue;
    }

    // Blockquote.
    if (/^\s*>/.test(line)) {
      const quote = newEl("blockquote");
      while (i < lines.length && /^\s*>/.test(lines[i])) {
        const p = newEl("p");
        p.appendChild(inline(lines[i].replace(/^\s*>\s?/, "")));
        quote.appendChild(p);
        i += 1;
      }
      root.appendChild(quote);
      continue;
    }

    // Paragraph (possibly spanning contiguous text lines).
    const paraLines = [line];
    i += 1;
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^\s*(```|#{1,6}\s|[-*]\s|\d+\.\s|>)/.test(lines[i])
    ) {
      paraLines.push(lines[i]);
      i += 1;
    }
    const p = newEl("p");
    p.appendChild(inline(paraLines.join("\n")));
    root.appendChild(p);
  }

  return root;
}

export { renderMarkdown };
