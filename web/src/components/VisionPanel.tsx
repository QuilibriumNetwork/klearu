// Vision classification panel: load a DaViT/ViT checkpoint, drop or pick an
// image, see top-K predictions with confidence bars.

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  visionClassifyBytes,
  visionClassifyPath,
  visionLoad,
  visionStatus,
  visionUnload,
  type VisionKind,
  type VisionPrediction,
  type VisionStatus,
} from '../lib/tauri';

export default function VisionPanel() {
  const [status, setStatus] = useState<VisionStatus | null>(null);
  const [modelDir, setModelDir] = useState('');
  const [forceKind, setForceKind] = useState<VisionKind | ''>('');
  const [loading, setLoading] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);

  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<VisionPrediction[] | null>(null);
  const [classifyErr, setClassifyErr] = useState<string | null>(null);
  const [classifying, setClassifying] = useState(false);
  const [topK, setTopK] = useState(5);

  useEffect(() => { visionStatus().then(setStatus).catch(() => {}); }, []);

  async function pickAndLoad() {
    let dir = modelDir.trim();
    if (!dir) {
      try {
        const { open } = await import('@tauri-apps/plugin-dialog');
        const picked = await open({ directory: true, multiple: false, title: 'Select vision model directory' });
        if (typeof picked === 'string') dir = picked;
        else return;
      } catch (e) { setLoadErr(String(e)); return; }
    }
    setLoadErr(null);
    setLoading(true);
    try {
      const s = await visionLoad(dir, (forceKind || undefined) as VisionKind | undefined);
      setStatus(s);
      setModelDir('');
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setLoadErr(msg);
    } finally { setLoading(false); }
  }

  async function unload() {
    await visionUnload();
    setStatus(await visionStatus());
    setPredictions(null);
  }

  const classifyFile = useCallback(async (file: File) => {
    setClassifyErr(null);
    setPredictions(null);
    setImagePath(null);
    setImageDataUrl(URL.createObjectURL(file));
    setClassifying(true);
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const preds = await visionClassifyBytes(bytes, topK);
      setPredictions(preds);
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setClassifyErr(msg);
    } finally { setClassifying(false); }
  }, [topK]);

  const classifyPath = useCallback(async (path: string) => {
    setClassifyErr(null);
    setPredictions(null);
    setImagePath(path);
    // Use the asset protocol for display.
    const { convertFileSrc } = await import('@tauri-apps/api/core');
    setImageDataUrl(convertFileSrc(path));
    setClassifying(true);
    try {
      const preds = await visionClassifyPath(path, topK);
      setPredictions(preds);
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setClassifyErr(msg);
    } finally { setClassifying(false); }
  }, [topK]);

  async function pickImage() {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const picked = await open({
        multiple: false,
        title: 'Pick an image',
        filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif'] }],
      });
      if (typeof picked === 'string') await classifyPath(picked);
    } catch (e) { setClassifyErr(String(e)); }
  }

  return (
    <div className="flex flex-1 min-h-0">
      <aside className="w-72 flex flex-col border-r border-gray-700 bg-gray-900">
        <EnginePanel
          status={status}
          modelDir={modelDir}
          setModelDir={setModelDir}
          forceKind={forceKind}
          setForceKind={setForceKind}
          onLoad={pickAndLoad}
          onUnload={unload}
          loading={loading}
          err={loadErr}
        />
        <div className="px-3 py-2 text-xs text-gray-400 border-t border-gray-700">
          <label className="flex items-center gap-2">
            <span>Top-K</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value))))}
              className="w-16 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-gray-100"
            />
          </label>
        </div>
        <div className="flex-1" />
      </aside>

      <section className="flex-1 flex flex-col min-w-0">
        <DropZone
          enabled={status?.loaded ?? false}
          onFile={classifyFile}
          onPickFile={pickImage}
          previewSrc={imageDataUrl}
          previewLabel={imagePath ?? undefined}
        />
        <Predictions
          predictions={predictions}
          classifying={classifying}
          error={classifyErr}
        />
      </section>
    </div>
  );
}

// ----------------------------------------------------------------------------

function EnginePanel({ status, modelDir, setModelDir, forceKind, setForceKind, onLoad, onUnload, loading, err }: {
  status: VisionStatus | null;
  modelDir: string;
  setModelDir: (s: string) => void;
  forceKind: VisionKind | '';
  setForceKind: (k: VisionKind | '') => void;
  onLoad: () => Promise<void>;
  onUnload: () => Promise<void>;
  loading: boolean;
  err: string | null;
}) {
  const loaded = !!status?.loaded;
  return (
    <div className="px-3 py-3 border-b border-gray-700">
      <div className="flex items-center gap-2 text-xs mb-2">
        <span className={`inline-block w-2 h-2 rounded-full ${loaded ? 'bg-emerald-400' : 'bg-gray-500'}`} />
        <span className="text-gray-200 font-medium">Vision engine</span>
        {loaded && (
          <button onClick={onUnload} className="ml-auto text-gray-400 hover:text-gray-200 text-[10px] px-1.5 py-0.5 rounded hover:bg-gray-800">
            unload
          </button>
        )}
      </div>
      {loaded ? (
        <div className="text-[11px] text-gray-400 leading-snug truncate" title={status!.model_dir ?? ''}>
          {status!.model_dir}
          <br />
          <span className="text-gray-500">
            {status!.kind} · classes={status!.num_classes} · input={status!.input_size}px
          </span>
        </div>
      ) : (
        <>
          <input
            placeholder="model dir (or pick…)"
            value={modelDir}
            onChange={(e) => setModelDir(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 mb-1"
          />
          <select
            value={forceKind}
            onChange={(e) => setForceKind(e.target.value as VisionKind | '')}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 mb-1"
            title="Override auto-detect"
          >
            <option value="">auto-detect family</option>
            <option value="davit">davit</option>
            <option value="vit">vit</option>
          </select>
          <button
            onClick={onLoad}
            disabled={loading}
            className="w-full text-xs bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-100 rounded py-1"
          >
            {loading ? 'loading…' : 'Load checkpoint'}
          </button>
          {err && <div className="mt-2 text-[11px] text-red-300">{err}</div>}
        </>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------

function DropZone({ enabled, onFile, onPickFile, previewSrc, previewLabel }: {
  enabled: boolean;
  onFile: (f: File) => Promise<void>;
  onPickFile: () => Promise<void>;
  previewSrc: string | null;
  previewLabel?: string;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (enabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={async (e) => {
        e.preventDefault(); setDragging(false);
        if (!enabled) return;
        const file = e.dataTransfer.files?.[0];
        if (file) await onFile(file);
      }}
      className={`flex-1 flex flex-col items-center justify-center px-6 py-6 border-b border-gray-700 transition-colors ${
        dragging ? 'bg-blue-950/30' : 'bg-gray-900/40'
      } ${!enabled ? 'opacity-50' : ''}`}
    >
      {previewSrc ? (
        <>
          <img src={previewSrc} alt={previewLabel} className="max-w-full max-h-[50vh] rounded border border-gray-700" />
          {previewLabel && (
            <div className="text-[11px] text-gray-500 break-all mt-2 max-w-full">{previewLabel}</div>
          )}
          <button
            onClick={() => { if (enabled) onPickFile(); }}
            disabled={!enabled}
            className="mt-3 text-xs px-3 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100"
          >
            Pick another…
          </button>
        </>
      ) : (
        <>
          <div className="text-sm text-gray-400 mb-3">
            {enabled ? 'Drop an image here, or' : 'Load a checkpoint first'}
          </div>
          <button
            onClick={() => { if (enabled) onPickFile(); }}
            disabled={!enabled}
            className="text-xs px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-100"
          >
            Pick an image…
          </button>
          {/* Hidden input used by tests / keyboard fallback. */}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onFile(f);
            }}
          />
        </>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------

function Predictions({ predictions, classifying, error }: {
  predictions: VisionPrediction[] | null;
  classifying: boolean;
  error: string | null;
}) {
  return (
    <div className="px-6 py-4 max-h-[40vh] overflow-y-auto">
      {error && (
        <div className="text-xs px-3 py-2 rounded bg-red-900/40 text-red-200">{error}</div>
      )}
      {classifying && !predictions && (
        <div className="text-sm text-gray-400">classifying…</div>
      )}
      {predictions && predictions.length > 0 && (
        <div className="space-y-1">
          {predictions.map((p, i) => (
            <div key={p.class_index} className="flex items-center gap-3 text-sm">
              <div className="w-6 text-right text-gray-500">#{i + 1}</div>
              <div className="w-40 truncate text-gray-100">{p.label}</div>
              <div className="flex-1 bg-gray-800 rounded h-2 overflow-hidden">
                <div
                  className="bg-emerald-500 h-full"
                  style={{ width: `${(p.probability * 100).toFixed(2)}%` }}
                />
              </div>
              <div className="w-16 text-right text-xs text-gray-400 tabular-nums">
                {(p.probability * 100).toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
