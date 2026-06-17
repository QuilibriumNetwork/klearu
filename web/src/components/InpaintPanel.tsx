// Inpaint panel — pick a source image, paint a mask, hit Generate.
// Works on desktop (Tauri dialog), remote browsers (file input), and
// mobile (responsive layout, touch-friendly controls).

import { useCallback, useEffect, useRef, useState } from 'react';
import { imageSrc, isTauri, settingsGet, settingsSet, type InpaintRequest } from '../lib/tauri';
import MaskCanvas, { type MaskCanvasHandle, type MaskTool } from './MaskCanvas';
import CropEditor, { type CropResult } from './CropEditor';
import { useInpaint } from '../hooks/useInpaint';
import { takeInpaintSource } from '../lib/inpaintHandoff';

function useSessionState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [val, setVal] = useState<T>(() => {
    try {
      const s = sessionStorage.getItem(key);
      return s != null ? JSON.parse(s) : initial;
    } catch { return initial; }
  });
  const set = useCallback((v: T) => {
    setVal(v);
    try { sessionStorage.setItem(key, JSON.stringify(v)); } catch {}
  }, [key]);
  return [val, set];
}

export default function InpaintPanel() {
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [sourceBytes, setSourceBytes] = useState<Uint8Array | null>(null);
  const [brushSize, setBrushSize] = useSessionState('inp_brush', 48);
  const [tool, setTool] = useState<MaskTool>('brush');

  const [prompt, setPrompt] = useSessionState('inp_prompt', '');
  const [neg, setNeg] = useSessionState('inp_neg', '');
  const [steps, setSteps] = useSessionState('inp_steps', 28);
  const [cfg, setCfg] = useSessionState('inp_cfg', 7.5);
  const [strength, setStrength] = useSessionState('inp_strength', 0.85);
  const [seed, setSeed] = useSessionState('inp_seed', '');
  const [targetSize, setTargetSize] = useSessionState('inp_size', 0);
  // Auto-detect default from model type on first load: SDXL (has CLIP-G
  // tokenizer) → 1024, SD 1.5 → 512. Only fires when no saved preference.
  useEffect(() => {
    if (targetSize !== 0) return; // already set by user or sessionStorage
    (async () => {
      const tg = await settingsGet('image.tokenizer_g');
      setTargetSize(tg ? 1024 : 512);
    })();
  }, [targetSize, setTargetSize]);

  const [showControls, setShowControls] = useState(false);
  const [showModelSettings, setShowModelSettings] = useState(false);
  const [checkpoint, setCheckpoint] = useState<string>('');
  const [tokL, setTokL] = useState<string>('');
  const [tokG, setTokG] = useState<string>('');

  // Load model settings on mount (shared with Image tab).
  useEffect(() => {
    (async () => {
      const cp = await settingsGet('image.checkpoint');
      const tl = await settingsGet('image.tokenizer_l');
      const tg = await settingsGet('image.tokenizer_g');
      setCheckpoint(cp ?? '');
      setTokL(tl ?? '');
      setTokG(tg ?? '');
    })();
  }, []);

  // Crop flow: rawImageUrl is the unpicked original; when set, the
  // CropEditor is shown. On confirm, it produces the cropped bytes →
  // sourceBytes/sourceUrl.
  const [rawImageUrl, setRawImageUrl] = useState<string | null>(null);
  const canvasRef = useRef<MaskCanvasHandle>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const inpaint = useInpaint();

  const [resultUrl, setResultUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!inpaint.resultPath) { setResultUrl(null); return; }
    imageSrc(inpaint.resultPath).then(setResultUrl);
  }, [inpaint.resultPath]);

  async function loadFile(file: Blob) {
    // Show the crop editor instead of committing immediately.
    const url = URL.createObjectURL(file);
    setRawImageUrl(url);
  }

  // Consume a generated image handed off from the Image panel's
  // "Use as input" action. Routes through the crop flow like any
  // picked file, so the user can crop/resize before painting a mask.
  useEffect(() => {
    const bytes = takeInpaintSource();
    if (bytes) {
      loadFile(new Blob([bytes as BlobPart], { type: 'image/png' }));
    }
    // Run once on mount — the handoff store is single-shot.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onCropConfirm(result: CropResult) {
    if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    if (rawImageUrl) URL.revokeObjectURL(rawImageUrl);
    setRawImageUrl(null);
    setSourceBytes(result.bytes);
    setSourceUrl(result.objectUrl);
  }

  function onCropCancel() {
    if (rawImageUrl) URL.revokeObjectURL(rawImageUrl);
    setRawImageUrl(null);
  }

  async function pickImage() {
    // Desktop: use the Tauri dialog. Remote/mobile: fall back to a
    // standard file input (works everywhere).
    if (isTauri()) {
      try {
        const { open } = await import('@tauri-apps/plugin-dialog');
        const picked = await open({
          multiple: false,
          title: 'Pick an image to inpaint',
          filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp'] }],
        });
        if (typeof picked === 'string') {
          const { readFile } = await import('@tauri-apps/plugin-fs');
          const rawBytes = await readFile(picked);
          // Resize+crop via canvas just like the File path.
          const blob = new Blob([rawBytes], { type: 'image/png' });
          await loadFile(blob);
          return;
        }
      } catch (e) { console.warn('tauri pick failed, falling back to file input', e); fileInputRef.current?.click(); }
    } else {
      fileInputRef.current?.click();
    }
  }

  async function submit() {
    if (!sourceBytes || !prompt.trim()) return;
    const maskBlob = await canvasRef.current?.exportMask();
    const maskBytes = maskBlob ? new Uint8Array(await maskBlob.arrayBuffer()) : undefined;
    const req: InpaintRequest = {
      prompt,
      negative_prompt: neg.trim() || undefined,
      steps,
      cfg_scale: cfg,
      strength,
      seed: seed.trim() ? Number(seed) : undefined,
      input_bytes: sourceBytes,
      mask_bytes: maskBytes,
    };
    await inpaint.generate(req);
  }

  return (
    <div className="flex flex-col sm:flex-row flex-1 min-h-0 relative">
      {/* Hidden file input for remote/mobile image picking */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) loadFile(f);
          e.target.value = '';
        }}
      />

      {/* Controls sidebar — full overlay on mobile when toggled */}
      <aside className={`${
        showControls ? 'absolute inset-0 z-30' : 'hidden'
      } sm:relative sm:flex sm:w-64 flex-col border-r border-gray-700 bg-gray-900 px-3 py-3 space-y-3 overflow-y-auto`}>
        <div className="flex items-center justify-between">
          <button
            onClick={pickImage}
            className="flex-1 text-xs bg-blue-700 hover:bg-blue-600 text-gray-100 rounded py-1.5"
          >
            {sourceUrl ? 'Replace image…' : 'Pick image…'}
          </button>
          <button
            onClick={() => setShowModelSettings((s) => !s)}
            className="ml-1 text-[10px] px-1.5 py-1 rounded text-gray-400 hover:bg-gray-800"
            title="Model settings (shared with Image tab)"
          >
            ⚙
          </button>
          <button
            onClick={() => setShowControls(false)}
            className="sm:hidden ml-1 text-xs px-1.5 py-1 rounded text-gray-400 hover:bg-gray-800"
          >
            ✕
          </button>
        </div>
        {showModelSettings && (
          <div className="border border-gray-700 rounded p-2 space-y-1.5 text-[11px]">
            <div className="text-gray-400 mb-1">Model (shared with Image tab)</div>
            <ModelField label="checkpoint" value={checkpoint} settingKey="image.checkpoint" onSaved={setCheckpoint} />
            <ModelField label="CLIP-L tok" value={tokL} settingKey="image.tokenizer_l" onSaved={setTokL} />
            <ModelField label="CLIP-G tok" value={tokG} settingKey="image.tokenizer_g" onSaved={setTokG} />
            {!checkpoint && <div className="text-red-300 text-[10px]">checkpoint required</div>}
          </div>
        )}

        <ToolBar
          tool={tool}
          setTool={setTool}
          brushSize={brushSize}
          setBrushSize={setBrushSize}
          onClear={() => canvasRef.current?.clear()}
        />

        <div className="border-t border-gray-700 pt-3 space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-gray-400 w-20">size</span>
            <select
              value={targetSize}
              onChange={(e) => setTargetSize(Number(e.target.value))}
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-100"
            >
              <option value={512}>512×512 (SD 1.5)</option>
              <option value={768}>768×768</option>
              <option value={1024}>1024×1024 (SDXL)</option>
            </select>
          </div>
          <NumberField label="steps" value={steps} setValue={setSteps} min={1} max={200} />
          <NumberField label="cfg" value={cfg} setValue={setCfg} min={0} max={30} step={0.1} />
          <NumberField label="strength" value={strength} setValue={setStrength} min={0} max={1} step={0.05} />
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-gray-400 w-20">seed</span>
            <input
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
              placeholder="random"
              inputMode="numeric"
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-100"
            />
          </div>
        </div>
      </aside>

      {/* Main area */}
      <section className="flex-1 flex flex-col min-w-0">
        {/* Mobile toolbar row */}
        <div className="sm:hidden flex items-center gap-2 border-b border-gray-700 px-3 py-1.5">
          <button
            onClick={() => setShowControls(true)}
            className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300"
          >
            Controls
          </button>
          <button
            onClick={pickImage}
            className="text-xs px-2 py-1 rounded bg-blue-800 hover:bg-blue-700 text-blue-100"
          >
            {sourceUrl ? 'Replace' : 'Pick image'}
          </button>
          <div className="flex gap-1 ml-auto">
            <button
              onClick={() => setTool('brush')}
              className={`text-[10px] px-1.5 py-0.5 rounded ${tool === 'brush' ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400'}`}
            >Brush</button>
            <button
              onClick={() => setTool('eraser')}
              className={`text-[10px] px-1.5 py-0.5 rounded ${tool === 'eraser' ? 'bg-blue-700 text-white' : 'bg-gray-800 text-gray-400'}`}
            >Eraser</button>
          </div>
        </div>

        {/* Prompt + submit */}
        <div className="border-b border-gray-700 px-3 sm:px-4 py-2 sm:py-3 space-y-2">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
            }}
            placeholder="prompt — describe what should appear in the painted area"
            rows={2}
            className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 resize-y min-h-[44px]"
          />
          <div className="flex items-center gap-2">
            <input
              value={neg}
              onChange={(e) => setNeg(e.target.value)}
              placeholder="negative prompt"
              className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100"
            />
            {inpaint.generating ? (
              <button onClick={inpaint.abort} className="text-xs px-3 py-1.5 rounded bg-amber-700 hover:bg-amber-600 text-amber-50 whitespace-nowrap">
                Stop
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!sourceBytes || !prompt.trim()}
                className="text-xs px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-100 whitespace-nowrap"
              >
                Inpaint
              </button>
            )}
          </div>
        </div>

        {/* Canvas + result */}
        <div
          onDragOver={(e) => { e.preventDefault(); }}
          onDrop={async (e) => {
            e.preventDefault();
            const file = e.dataTransfer.files?.[0];
            if (file) await loadFile(file);
          }}
          className="flex-1 overflow-auto px-3 sm:px-4 py-3 sm:py-4"
        >
          {/* Step 1: Crop editor (shown after pick, before mask painting) */}
          {rawImageUrl && (
            <CropEditor
              imageSrc={rawImageUrl}
              targetSize={targetSize}
              onConfirm={onCropConfirm}
              onCancel={onCropCancel}
            />
          )}
          {/* Step 2: Mask painting + result (shown after crop confirm) */}
          {!rawImageUrl && sourceUrl ? (
            <div className="flex flex-col sm:flex-row gap-4 items-start">
              <div className="w-full sm:w-auto">
                <MaskCanvas
                  ref={canvasRef}
                  imageSrc={sourceUrl}
                  brushSize={brushSize}
                  tool={tool}
                />
              </div>
              {resultUrl && (
                <div className="flex flex-col items-start gap-1 w-full sm:w-auto">
                  <div className="text-xs text-gray-400">result</div>
                  <img src={resultUrl} alt="" className="max-w-full max-h-[50vh] sm:max-h-[70vh] rounded border border-gray-700" />
                </div>
              )}
            </div>
          ) : !rawImageUrl ? (
            <div
              className="flex-1 flex flex-col items-center justify-center text-sm text-gray-500 italic min-h-[30vh] border-2 border-dashed border-gray-700 rounded p-6"
              onClick={pickImage}
            >
              <div>Tap to pick an image, or drop one here</div>
            </div>
          ) : null}
        </div>

        {/* Progress */}
        {(inpaint.generating || inpaint.lastError) && (
          <div className="border-t border-gray-700 px-3 py-2 text-xs">
            {inpaint.generating && inpaint.progress && (
              <div>
                <div className="flex items-center gap-2 text-gray-300 mb-1">
                  <span>inpainting · step {inpaint.progress.step}/{inpaint.progress.total || '?'} · {inpaint.progress.elapsedS.toFixed(1)}s</span>
                </div>
                <div className="w-full bg-gray-800 rounded h-1.5 overflow-hidden">
                  <div
                    className="bg-blue-500 h-full transition-all"
                    style={{
                      width: inpaint.progress.total > 0
                        ? `${(inpaint.progress.step / inpaint.progress.total) * 100}%`
                        : '5%',
                    }}
                  />
                </div>
              </div>
            )}
            {inpaint.lastError && (
              <div className="text-red-300 mt-1">{inpaint.lastError}</div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

function ToolBar({ tool, setTool, brushSize, setBrushSize, onClear }: {
  tool: MaskTool;
  setTool: (t: MaskTool) => void;
  brushSize: number;
  setBrushSize: (n: number) => void;
  onClear: () => void;
}) {
  return (
    <div className="space-y-2 border-t border-gray-700 pt-3">
      <div className="flex gap-1">
        <button
          onClick={() => setTool('brush')}
          className={`flex-1 text-xs py-1 rounded ${
            tool === 'brush' ? 'bg-blue-700 text-gray-100' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Brush
        </button>
        <button
          onClick={() => setTool('eraser')}
          className={`flex-1 text-xs py-1 rounded ${
            tool === 'eraser' ? 'bg-blue-700 text-gray-100' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
          }`}
        >
          Eraser
        </button>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-[11px] text-gray-400 w-20">brush</span>
        <input
          type="range"
          min={2}
          max={200}
          value={brushSize}
          onChange={(e) => setBrushSize(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-[11px] text-gray-400 w-8 text-right">{brushSize}</span>
      </div>
      <button
        onClick={onClear}
        className="w-full text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 rounded py-1"
      >
        Clear mask
      </button>
    </div>
  );
}

function ModelField({ label, value, settingKey, onSaved }: {
  label: string;
  value: string;
  settingKey: string;
  onSaved: (v: string) => void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => { setDraft(value); }, [value]);
  return (
    <div className="flex items-center gap-1">
      <span className="text-gray-500 w-16 shrink-0">{label}</span>
      <input
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={async () => {
          if (draft !== value) {
            await settingsSet(settingKey, draft);
            onSaved(draft);
          }
        }}
        className="flex-1 bg-gray-800 border border-gray-700 rounded px-1.5 py-0.5 text-[10px] text-gray-100 min-w-0"
      />
    </div>
  );
}

function NumberField({ label, value, setValue, min, max, step }: {
  label: string;
  value: number;
  setValue: (n: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-gray-400 w-20">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step ?? 1}
        onChange={(e) => setValue(Number(e.target.value))}
        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-xs text-gray-100"
      />
    </div>
  );
}
