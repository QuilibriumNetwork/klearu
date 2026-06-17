// Header chip + popover for incognito mode.
//
// Off: discreet sunglasses chip; click → confirms then starts the session.
// On:  high-contrast banner showing tmp dir + elapsed time; click → confirms
//      then stops (which wipes the tmp dir and swaps DB back to disk).

import { useEffect, useState } from 'react';
import {
  incognitoStart,
  incognitoStatus,
  incognitoStop,
  isTauri,
  type IncognitoStatus,
} from '../lib/tauri';

export default function IncognitoToggle({ onChanged }: { onChanged?: (s: IncognitoStatus) => void }) {
  const [status, setStatus] = useState<IncognitoStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauri()) return;
    incognitoStatus().then((s) => { setStatus(s); }).catch(() => {});
  }, []);

  // Live elapsed counter while active.
  const [now, setNow] = useState<number>(Math.floor(Date.now() / 1000));
  useEffect(() => {
    if (!status?.active) return;
    const t = setInterval(() => setNow(Math.floor(Date.now() / 1000)), 1000);
    return () => clearInterval(t);
  }, [status?.active]);

  if (!isTauri()) return null;

  async function toggle() {
    if (!status) return;
    setBusy(true); setErr(null);
    try {
      const next = status.active
        ? await incognitoStop()
        : await incognitoStart();
      setStatus(next);
      onChanged?.(next);
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e)
        ? (e as { message: string }).message
        : String(e);
      setErr(msg);
    } finally {
      setBusy(false);
    }
  }

  const active = status?.active ?? false;
  const elapsed = active && status?.started_at ? now - status.started_at : 0;
  const elapsedLabel = elapsed > 60
    ? `${Math.floor(elapsed / 60)}m${(elapsed % 60).toString().padStart(2, '0')}s`
    : `${elapsed}s`;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
          active
            ? 'bg-purple-700 text-purple-50 ring-1 ring-purple-400'
            : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
        }`}
        title={active ? 'Incognito session active — click for details' : 'Click for incognito mode'}
      >
        <span className="text-sm leading-none">🕶</span>
        {active ? (
          <span className="tabular-nums">Incognito · {elapsedLabel}</span>
        ) : (
          <span>Incognito</span>
        )}
      </button>

      {open && (
        <div className="absolute right-0 mt-1 w-96 bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-3 z-50">
          <div className="text-sm font-medium text-gray-100 mb-2">
            {active ? 'Incognito session is active' : 'Start an incognito session'}
          </div>
          <div className="text-[11px] text-gray-400 mb-1 mt-1">In-memory only:</div>
          <ul className="text-[11px] text-gray-300 space-y-0.5 mb-3 leading-snug">
            {(status?.disabled_features ?? []).map((f, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-purple-400">·</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
          <div className="text-[11px] text-gray-400 mb-1">Refused (would touch disk):</div>
          <ul className="text-[11px] text-gray-300 space-y-0.5 mb-3 leading-snug">
            {(status?.blocked_features ?? []).map((f, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-red-400">×</span>
                <span>{f}</span>
              </li>
            ))}
          </ul>
          <div className="text-[10px] text-gray-500 mb-3">
            No scratch directory: every byte stays in the SQLite arena and is
            freed when the session stops. Bit-level zeroization of allocator
            pages depends on the OS — quit the app for the strongest guarantee.
          </div>
          {err && <div className="text-[11px] text-red-300 mb-2">{err}</div>}
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setOpen(false)}
              className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-200"
            >
              Cancel
            </button>
            <button
              onClick={() => { toggle(); setOpen(false); }}
              disabled={busy}
              className={`text-xs px-3 py-1 rounded text-gray-100 ${
                active
                  ? 'bg-red-800 hover:bg-red-700'
                  : 'bg-purple-700 hover:bg-purple-600'
              }`}
            >
              {busy ? '…' : (active ? 'Stop incognito' : 'Start incognito')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
