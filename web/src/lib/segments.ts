// Parse an assistant message into renderable segments: narrative text,
// tool calls, and tool results. Mirrors the backend's `intent.rs` parser
// so the same XML-ish format round-trips cleanly.

export type Segment =
  | { kind: 'text'; text: string }
  | { kind: 'tool_call'; name: string; rawBody: string; args: unknown; parseError: string | null }
  | { kind: 'tool_result'; name: string; rawBody: string; result: unknown };

const OPEN_CALL = /<tool name="([^"]+)">/g;
const OPEN_RESULT = /<tool_result name="([^"]+)">/g;
const CLOSE_CALL = '</tool>';
const CLOSE_RESULT = '</tool_result>';

interface Token { kind: 'call' | 'result'; name: string; bodyStart: number; bodyEnd: number; blockStart: number; blockEnd: number; }

export function parseSegments(text: string): Segment[] {
  // First pass: find all tool blocks (both calls and results) in source order.
  // We do this with a simple scan, alternating between looking for "<tool " and "<tool_result ".
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    // Tool_result has the longer prefix so check it first.
    const resAt = text.indexOf('<tool_result name="', i);
    const callAt = text.indexOf('<tool name="', i);
    let next: { idx: number; kind: 'call' | 'result' } | null = null;
    if (resAt !== -1 && (callAt === -1 || resAt < callAt)) {
      next = { idx: resAt, kind: 'result' };
    } else if (callAt !== -1) {
      next = { idx: callAt, kind: 'call' };
    }
    if (!next) break;

    const openPrefix = next.kind === 'call' ? '<tool name="' : '<tool_result name="';
    const closePrefix = next.kind === 'call' ? CLOSE_CALL : CLOSE_RESULT;
    const nameStart = next.idx + openPrefix.length;
    const nameEnd = text.indexOf('"', nameStart);
    if (nameEnd === -1) break;
    const headerEnd = text.indexOf('">', nameEnd);
    if (headerEnd === -1) break;
    const bodyStart = headerEnd + 2;
    const closeAt = text.indexOf(closePrefix, bodyStart);
    if (closeAt === -1) break;
    const blockEnd = closeAt + closePrefix.length;

    tokens.push({
      kind: next.kind,
      name: text.slice(nameStart, nameEnd),
      bodyStart,
      bodyEnd: closeAt,
      blockStart: next.idx,
      blockEnd,
    });
    i = blockEnd;
  }

  // Second pass: weave text and tool blocks into segments.
  const out: Segment[] = [];
  let cursor = 0;
  for (const t of tokens) {
    if (t.blockStart > cursor) {
      const slice = text.slice(cursor, t.blockStart);
      if (slice.trim().length > 0) out.push({ kind: 'text', text: slice });
    }
    const rawBody = text.slice(t.bodyStart, t.bodyEnd).trim();
    if (t.kind === 'call') {
      let args: unknown = null;
      let parseError: string | null = null;
      try { args = rawBody.length ? JSON.parse(rawBody) : null; }
      catch (e) { parseError = e instanceof Error ? e.message : String(e); }
      out.push({ kind: 'tool_call', name: t.name, rawBody, args, parseError });
    } else {
      let result: unknown = rawBody;
      try { result = rawBody.length ? JSON.parse(rawBody) : null; } catch { /* keep raw */ }
      out.push({ kind: 'tool_result', name: t.name, rawBody, result });
    }
    cursor = t.blockEnd;
  }
  if (cursor < text.length) {
    const slice = text.slice(cursor);
    if (slice.trim().length > 0) out.push({ kind: 'text', text: slice });
  }
  return out;
}

/** Convenience: get just the narrative parts joined together. */
export function stripTools(text: string): string {
  return parseSegments(text)
    .filter((s) => s.kind === 'text')
    .map((s) => (s as { kind: 'text'; text: string }).text)
    .join('')
    .trim();
}
