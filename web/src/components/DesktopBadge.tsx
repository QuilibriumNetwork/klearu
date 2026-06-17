// Shows a small chip in the header when running inside the Tauri shell.
// Doubles as a smoke test for the IPC bridge — if app_info() returns,
// commands → backend → React deserialization all work.

import { useEffect, useState } from 'react';
import { appInfo, isTauri, type AppInfo } from '../lib/tauri';

export default function DesktopBadge() {
  const [info, setInfo] = useState<AppInfo | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!isTauri()) return;
    appInfo()
      .then((i) => setInfo(i))
      .catch((e) => setErr(String(e)));
  }, []);

  if (!isTauri()) return null;

  if (err) {
    return (
      <span
        title={err}
        className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium bg-red-900/50 text-red-300"
      >
        <span className="inline-block w-2 h-2 rounded-full bg-red-400" />
        Desktop · IPC error
      </span>
    );
  }

  return (
    <span
      title={info ? `data dir: ${info.data_dir}` : 'loading…'}
      className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium bg-blue-900/50 text-blue-300"
    >
      <span className="inline-block w-2 h-2 rounded-full bg-blue-400" />
      Desktop{info ? ` · v${info.version}` : ''}
    </span>
  );
}
