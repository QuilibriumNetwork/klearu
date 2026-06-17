// Header chip for LAN sharing. Shows the HTTPS URL, PIN, and cert
// fingerprint so remote devices can connect and verify.

import { useEffect, useState } from 'react';
import {
  isTauri,
  lanStart,
  lanStatus,
  lanStop,
  type LanStatus,
} from '../lib/tauri';

export default function LanToggle() {
  const [status, setStatus] = useState<LanStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [port, setPort] = useState('3100');
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauri()) return;
    lanStatus().then(setStatus).catch(() => {});
  }, []);

  if (!isTauri()) return null;

  async function toggle() {
    setBusy(true); setErr(null);
    try {
      const next = status?.active
        ? await lanStop()
        : await lanStart(Number(port) || 3100);
      setStatus(next);
    } catch (e) {
      setErr(String(e));
    } finally { setBusy(false); }
  }

  const active = status?.active ?? false;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
          active
            ? 'bg-emerald-700 text-emerald-50 ring-1 ring-emerald-400'
            : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
        }`}
        title={active ? 'LAN sharing active' : 'Share on LAN'}
      >
        <span className="text-sm leading-none">🌐</span>
        <span>{active ? 'LAN on' : 'LAN'}</span>
      </button>

      {open && (
        <div
          className="absolute right-0 mt-1 w-[420px] bg-gray-900 border border-gray-700 rounded-lg shadow-xl p-3 z-50"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="text-sm font-medium text-gray-100 mb-2">
            {active ? 'LAN sharing is active' : 'Share on LAN (HTTPS)'}
          </div>

          {active && status && (
            <>
              <div className="space-y-1.5 mb-3 text-[12px]">
                <div className="text-gray-400">
                  Open on the remote device:
                </div>
                {status.urls.map((url) => (
                  <div key={url} className="font-mono text-emerald-300 select-all break-all">
                    {url}?pin={status.pin}
                  </div>
                ))}
                <div className="mt-2 text-gray-400">
                  PIN: <span className="font-mono text-emerald-200 select-all text-sm">{status.pin}</span>
                </div>
              </div>

              {status.cert_fingerprint && (
                <div className="mb-3">
                  <div className="text-[11px] text-gray-400 mb-0.5">
                    TLS certificate fingerprint (verify in browser):
                  </div>
                  <div className="font-mono text-[10px] text-gray-300 select-all break-all bg-gray-800 rounded px-2 py-1">
                    {status.cert_fingerprint}
                  </div>
                </div>
              )}
            </>
          )}

          {!active && (
            <div className="flex items-center gap-2 mb-3">
              <label className="text-xs text-gray-400">Port:</label>
              <input
                value={port}
                onChange={(e) => setPort(e.target.value)}
                className="w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100"
              />
            </div>
          )}

          {err && <div className="text-[11px] text-red-300 mb-2">{err}</div>}

          <div className="flex justify-end gap-2">
            <button
              onClick={() => setOpen(false)}
              className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-200"
            >
              Close
            </button>
            <button
              onClick={() => { toggle(); }}
              disabled={busy}
              className={`text-xs px-3 py-1 rounded text-gray-100 ${
                active ? 'bg-red-800 hover:bg-red-700' : 'bg-emerald-700 hover:bg-emerald-600'
              }`}
            >
              {busy ? '…' : (active ? 'Stop sharing' : 'Start sharing')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
