// Renders one assistant message, parsing tool-call / tool-result blocks
// into compact cards so the transcript stays readable.

import { useState } from 'react';
import { parseSegments, type Segment } from '../lib/segments';
import FeedbackButtons from './FeedbackButtons';

export default function AssistantBubble({ text, streaming = false, messageId, onUserVectorChanged }: {
  text: string;
  streaming?: boolean;
  /** When set, render thumbs-up/down that ripple into the user steering vector. */
  messageId?: number;
  onUserVectorChanged?: () => void;
}) {
  const segments = parseSegments(text);
  return (
    <div className="flex items-start gap-2 max-w-[80%]">
      <div className="rounded-lg px-3 py-2 text-sm bg-gray-800 text-gray-100 space-y-2 flex-1">
        {segments.length === 0 && streaming && (
          <span className="inline-block w-1.5 h-3.5 bg-gray-400 animate-pulse align-middle" />
        )}
        {segments.map((s, i) => <SegmentView key={i} seg={s} />)}
        {streaming && segments.length > 0 && (
          <span className="ml-0.5 inline-block w-1.5 h-3.5 bg-gray-400 animate-pulse align-middle" />
        )}
      </div>
      {messageId != null && !streaming && (
        <FeedbackButtons
          kind="chat_message"
          targetId={messageId}
          layout="stack"
          onUserVectorChanged={onUserVectorChanged}
        />
      )}
    </div>
  );
}

function SegmentView({ seg }: { seg: Segment }) {
  if (seg.kind === 'text') {
    return <div className="whitespace-pre-wrap">{seg.text}</div>;
  }
  if (seg.kind === 'tool_call') return <ToolCallCard seg={seg} />;
  return <ToolResultCard seg={seg} />;
}

function ToolCallCard({ seg }: { seg: Extract<Segment, { kind: 'tool_call' }> }) {
  const [open, setOpen] = useState(false);
  return (
    <button
      type="button"
      onClick={() => setOpen((o) => !o)}
      className="block w-full text-left bg-blue-950/50 border border-blue-900/60 rounded px-2 py-1 hover:bg-blue-950/70 transition-colors"
    >
      <div className="flex items-center gap-2 text-xs">
        <span className="text-blue-300">🛠</span>
        <span className="font-medium text-blue-100">{seg.name}</span>
        {seg.parseError && (
          <span className="text-red-300 text-[10px]">bad JSON: {seg.parseError}</span>
        )}
        <span className="ml-auto text-blue-400 text-[10px]">{open ? '▾' : '▸'}</span>
      </div>
      {open && (
        <pre className="mt-1 text-[11px] text-blue-200 whitespace-pre-wrap font-mono overflow-x-auto">
          {seg.parseError ? seg.rawBody : JSON.stringify(seg.args, null, 2)}
        </pre>
      )}
    </button>
  );
}

function ToolResultCard({ seg }: { seg: Extract<Segment, { kind: 'tool_result' }> }) {
  const [open, setOpen] = useState(false);
  const result = seg.result as { ok?: boolean; result?: unknown; code?: string; message?: string };
  const ok = result?.ok === true;
  return (
    <button
      type="button"
      onClick={() => setOpen((o) => !o)}
      className={`block w-full text-left rounded px-2 py-1 transition-colors border ${
        ok
          ? 'bg-emerald-950/40 border-emerald-900/60 hover:bg-emerald-950/60'
          : 'bg-red-950/40 border-red-900/60 hover:bg-red-950/60'
      }`}
    >
      <div className="flex items-center gap-2 text-xs">
        <span className={ok ? 'text-emerald-300' : 'text-red-300'}>{ok ? '✓' : '✗'}</span>
        <span className={`font-medium ${ok ? 'text-emerald-100' : 'text-red-100'}`}>
          {seg.name}
        </span>
        {!ok && result?.code && (
          <span className="text-red-300 text-[10px]">{result.code}</span>
        )}
        <span className={`ml-auto text-[10px] ${ok ? 'text-emerald-400' : 'text-red-400'}`}>
          {open ? '▾' : '▸'}
        </span>
      </div>
      {open && (
        <pre className={`mt-1 text-[11px] whitespace-pre-wrap font-mono overflow-x-auto ${
          ok ? 'text-emerald-200' : 'text-red-200'
        }`}>
          {JSON.stringify(ok ? result.result : { code: result.code, message: result.message }, null, 2)}
        </pre>
      )}
    </button>
  );
}
