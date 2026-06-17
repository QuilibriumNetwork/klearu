// Feedback inspector: shows the user steering vector's running stats,
// lets you tune α (how strongly v is applied at generation time), reset
// the vector, and browse the recent feedback log.

import { useCallback, useEffect, useState } from 'react';
import {
  feedbackList,
  userVectorReset,
  userVectorResetNamed,
  userVectorSetAlpha,
  userVectorSetAlphaNamed,
  userVectorStatus,
  userVectorsList,
  type FeedbackKind,
  type FeedbackRow,
  type UserVectorStatus,
} from '../lib/tauri';

export default function FeedbackPanel() {
  const [status, setStatus] = useState<UserVectorStatus | null>(null);
  const [others, setOthers] = useState<UserVectorStatus[]>([]);
  const [items, setItems] = useState<FeedbackRow[]>([]);
  const [filter, setFilter] = useState<FeedbackKind | 'all'>('all');
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, all, list] = await Promise.all([
        userVectorStatus(),
        userVectorsList(),
        feedbackList(filter === 'all' ? undefined : filter, 200),
      ]);
      setStatus(s);
      // Drop "default" — already shown as the headline card.
      setOthers(all.filter((v) => v.name !== 'default'));
      setItems(list);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, [filter]);

  useEffect(() => { refresh(); }, [refresh]);

  async function changeAlpha(a: number) {
    try { setStatus(await userVectorSetAlpha(a)); }
    catch (e) { setErr(String(e)); }
  }
  async function reset() {
    if (!confirm('Reset the user steering vector? Welford state goes back to zero.')) return;
    try { await userVectorReset(); await refresh(); }
    catch (e) { setErr(String(e)); }
  }

  return (
    <div className="flex flex-1 min-h-0">
      <aside className="w-80 flex flex-col border-r border-gray-700 bg-gray-900 p-3 space-y-3 overflow-y-auto">
        <div>
          <div className="text-xs font-medium text-gray-200 mb-1">User steering vector</div>
          {status ? (
            <div className="text-[11px] text-gray-400 leading-snug space-y-0.5">
              <div>name: <span className="text-gray-200">{status.name}</span></div>
              <div>hidden_size: <span className="text-gray-200 tabular-nums">{status.hidden_size}</span></div>
              <div className="flex gap-3">
                <span>👍 <span className="text-emerald-300 tabular-nums">{status.n_pos}</span></span>
                <span>👎 <span className="text-red-300 tabular-nums">{status.n_neg}</span></span>
              </div>
              <div>‖v‖ = <span className="text-gray-200 tabular-nums">{status.norm.toFixed(4)}</span></div>
            </div>
          ) : <div className="text-[11px] text-gray-500">no LLM loaded</div>}
        </div>

        <div>
          <div className="text-[11px] text-gray-400 mb-1">α (steering strength)</div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={-2} max={2} step={0.1}
              value={status?.alpha ?? 1.0}
              onChange={(e) => changeAlpha(Number(e.target.value))}
              className="flex-1"
            />
            <span className="text-xs tabular-nums w-10 text-right text-gray-300">
              {(status?.alpha ?? 1.0).toFixed(2)}
            </span>
          </div>
          <div className="text-[10px] text-gray-500 mt-1">
            α=0 disables steering · negative inverts the direction
          </div>
        </div>

        <div>
          <div className="text-[11px] text-gray-400 mb-1">v[0..8] preview</div>
          <pre className="text-[10px] text-gray-300 font-mono whitespace-pre-wrap">
            {status ? `[ ${status.head.map((x) => x.toFixed(4)).join(', ')} ]` : '—'}
          </pre>
        </div>

        <button
          onClick={reset}
          className="text-xs px-3 py-1.5 rounded bg-red-900/60 hover:bg-red-900 text-red-100"
        >
          Reset chat vector
        </button>

        {others.length > 0 && (
          <div className="border-t border-gray-800 pt-3 space-y-3">
            <div className="text-xs font-medium text-gray-200">CLIP-space vectors</div>
            <div className="text-[10px] text-gray-500">
              Built from image 👍/👎. Added to every position of the cond CLIP
              embedding before cross-attention.
            </div>
            {others.map((v) => (
              <ClipVectorCard
                key={v.name}
                v={v}
                onAlpha={async (a) => {
                  try { await userVectorSetAlphaNamed(v.name, a); await refresh(); }
                  catch (e) { setErr(String(e)); }
                }}
                onReset={async () => {
                  if (!confirm(`Reset ${v.name}?`)) return;
                  try { await userVectorResetNamed(v.name); await refresh(); }
                  catch (e) { setErr(String(e)); }
                }}
              />
            ))}
          </div>
        )}

        {err && <div className="text-[11px] text-red-300">{err}</div>}
      </aside>

      <section className="flex-1 flex flex-col min-w-0">
        <header className="px-4 py-2 border-b border-gray-700 flex items-center gap-2">
          <span className="text-sm font-medium text-gray-200">Recent feedback</span>
          <div className="ml-auto flex gap-1">
            {(['all', 'chat_message', 'image'] as const).map((k) => (
              <button
                key={k}
                onClick={() => setFilter(k)}
                className={`text-xs px-2 py-0.5 rounded ${
                  filter === k ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:bg-gray-800'
                }`}
              >
                {k === 'all' ? 'All' : k === 'chat_message' ? 'Chat' : 'Image'}
              </button>
            ))}
          </div>
        </header>
        <div className="flex-1 overflow-y-auto">
          {items.length === 0 && (
            <div className="px-4 py-6 text-xs text-gray-500">
              Click ▲ / ▼ on assistant replies or images to seed the user vector.
            </div>
          )}
          {items.map((f) => (
            <div key={f.id} className="px-4 py-2 border-b border-gray-800 flex items-center gap-3">
              <span className={`text-base ${f.rating > 0 ? 'text-emerald-400' : f.rating < 0 ? 'text-red-400' : 'text-gray-500'}`}>
                {f.rating > 0 ? '▲' : f.rating < 0 ? '▼' : '○'}
              </span>
              <span className="text-xs text-gray-400 w-24">{f.target_kind}</span>
              <span className="text-xs text-gray-300 tabular-nums w-12">#{f.target_id}</span>
              <span className="text-[11px] text-gray-500 ml-auto">
                {new Date(f.created_at * 1000).toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function ClipVectorCard({ v, onAlpha, onReset }: {
  v: UserVectorStatus;
  onAlpha: (a: number) => Promise<void>;
  onReset: () => Promise<void>;
}) {
  return (
    <div className="border border-gray-800 rounded p-2 space-y-1.5 bg-gray-950/40">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-medium text-gray-200">{v.name}</span>
        <span className="text-[10px] text-gray-500 ml-auto">d={v.hidden_size}</span>
      </div>
      <div className="text-[10px] text-gray-400 leading-snug">
        👍 <span className="text-emerald-300 tabular-nums">{v.n_pos}</span> ·
        {' '}👎 <span className="text-red-300 tabular-nums">{v.n_neg}</span> ·
        {' '}‖v‖ <span className="text-gray-200 tabular-nums">{v.norm.toFixed(3)}</span>
      </div>
      <div className="flex items-center gap-2">
        <input
          type="range"
          min={-2} max={2} step={0.1}
          value={v.alpha}
          onChange={(e) => onAlpha(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-[10px] tabular-nums w-8 text-right text-gray-300">
          α {v.alpha.toFixed(2)}
        </span>
      </div>
      <button
        onClick={onReset}
        className="w-full text-[10px] py-0.5 rounded bg-red-900/40 hover:bg-red-900/60 text-red-200"
      >
        reset
      </button>
    </div>
  );
}
