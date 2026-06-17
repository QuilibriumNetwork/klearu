// Image generation panel: settings (checkpoint + tokenizer paths), prompt
// composer, live progress, recent-images gallery, and a full-window
// carousel lightbox.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  imageDelete,
  imageSrc,
  settingsGet,
  settingsSet,
  videoGenerate,
  type GenEvent,
  type StoredImage,
  type VideoGenerateRequest,
} from '../lib/tauri';
import { useImageGen } from '../hooks/useImageGen';
import FeedbackButtons from './FeedbackButtons';

const SETTING_KEYS = [
  'image.sd_generate_path',
  'image.checkpoint',
  'image.tokenizer_l',
  'image.tokenizer_g',
  'image.output_dir',
  'image.motion_adapter',
] as const;

type SettingKey = typeof SETTING_KEYS[number];

interface ImageSettings {
  'image.sd_generate_path'?: string;
  'image.checkpoint'?: string;
  'image.tokenizer_l'?: string;
  'image.tokenizer_g'?: string;
  'image.output_dir'?: string;
  'image.motion_adapter'?: string;
}

export default function ImagePanel({ onUseAsInput }: { onUseAsInput?: (img: StoredImage) => void } = {}) {
  const gen = useImageGen();
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<ImageSettings>({});
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [carouselIndex, setCarouselIndex] = useState<number | null>(null);
  // On mobile the sidebar is hidden by default; toggled via a button.
  const [showSidebar, setShowSidebar] = useState(false);

  useEffect(() => {
    (async () => {
      const next: ImageSettings = {};
      for (const k of SETTING_KEYS) {
        const v = await settingsGet(k);
        if (v) next[k] = v;
      }
      setSettings(next);
      setSettingsLoaded(true);
      // A checkpoint is the minimum. SD/SDXL also need a CLIP tokenizer, but
      // HiDream-O1 (a directory model) carries its own — so don't hard-require
      // tokenizer_l here; the backend errors clearly if an SD model lacks it.
      if (!next['image.checkpoint']) {
        setShowSettings(true);
      }
    })();
  }, []);

  const configured = !!settings['image.checkpoint'];

  const handleDelete = useCallback(async (id: number) => {
    await imageDelete(id, true);
    await gen.refreshGallery();
    // If the carousel was showing the deleted image, close it or shift.
    if (carouselIndex != null) {
      const newLen = gen.gallery.length - 1;
      if (newLen <= 0) setCarouselIndex(null);
      else if (carouselIndex >= newLen) setCarouselIndex(newLen - 1);
    }
  }, [gen, carouselIndex]);

  return (
    <div className="flex flex-1 min-h-0 relative">
      {/* Sidebar: gallery — hidden on mobile unless toggled */}
      <aside className={`${
        showSidebar ? 'absolute inset-0 z-30' : 'hidden'
      } sm:relative sm:flex sm:w-64 flex-col border-r border-gray-700 bg-gray-900`}>
        <div className="px-3 py-2 border-b border-gray-700 flex items-center justify-between">
          <span className="text-xs font-medium text-gray-200">Recent</span>
          <div className="flex gap-2">
            <button
              onClick={() => setShowSettings((s) => !s)}
              className={`text-[10px] px-1.5 py-0.5 rounded ${
                showSettings ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:bg-gray-800'
              }`}
              title="Image engine settings"
            >
              ⚙
            </button>
            {/* Close sidebar on mobile */}
            <button
              onClick={() => setShowSidebar(false)}
              className="sm:hidden text-[10px] px-1.5 py-0.5 rounded text-gray-400 hover:bg-gray-800"
            >
              ✕
            </button>
          </div>
        </div>
        <div className="flex-1 overflow-y-auto">
          {gen.gallery.length === 0 && (
            <div className="px-3 py-6 text-xs text-gray-500">No images yet.</div>
          )}
          {gen.gallery.map((img, idx) => (
            <GalleryThumb
              key={img.id}
              image={img}
              onClick={() => { setCarouselIndex(idx); setShowSidebar(false); }}
              onDelete={() => handleDelete(img.id)}
            />
          ))}
        </div>
      </aside>

      {/* Center */}
      <section className="flex-1 flex flex-col min-w-0">
        {showSettings && settingsLoaded && (
          <SettingsBlock
            settings={settings}
            onSet={async (k, v) => {
              await settingsSet(k, v);
              setSettings((prev) => ({ ...prev, [k]: v }));
            }}
            onClose={() => setShowSettings(false)}
          />
        )}
        {/* Mobile gallery toggle */}
        <div className="sm:hidden flex items-center border-b border-gray-700 px-3 py-1.5">
          <button
            onClick={() => setShowSidebar(true)}
            className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-gray-300"
          >
            Gallery ({gen.gallery.length})
          </button>
          <button
            onClick={() => setShowSettings((s) => !s)}
            className="ml-auto text-[10px] px-1.5 py-0.5 rounded text-gray-400 hover:bg-gray-800"
          >
            ⚙ settings
          </button>
        </div>
        <Composer
          configured={configured}
          generating={gen.generating}
          onGenerate={gen.generate}
          onGenerateVideo={gen.generateVideo}
          onAbort={gen.abort}
          onResetAbort={gen.resetAbort}
        />
        <ResultArea
          progress={gen.progress}
          generating={gen.generating}
          lastError={gen.lastError}
          recentLogs={gen.recentLogs}
          gallery={gen.gallery}
          onImageClick={(idx) => setCarouselIndex(idx)}
        />
      </section>

      {/* Full-window carousel lightbox */}
      {carouselIndex != null && gen.gallery.length > 0 && (
        <Carousel
          gallery={gen.gallery}
          index={carouselIndex}
          onIndexChange={setCarouselIndex}
          onClose={() => setCarouselIndex(null)}
          onUseAsInput={onUseAsInput ? (img) => { setCarouselIndex(null); onUseAsInput(img); } : undefined}
        />
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------

function SettingsBlock({ settings, onSet, onClose }: {
  settings: ImageSettings;
  onSet: (k: SettingKey, v: string) => Promise<void>;
  onClose: () => void;
}) {
  return (
    <div className="border-b border-gray-700 bg-gray-900 px-4 py-3 space-y-2">
      <div className="flex items-center mb-1">
        <span className="text-sm font-medium text-gray-200">Diffusion engine</span>
        <button onClick={onClose} className="ml-auto text-xs text-gray-400 hover:text-gray-200">close</button>
      </div>
      <SettingField label="sd_generate binary" placeholder="auto-detect — env or target/debug" k="image.sd_generate_path" value={settings['image.sd_generate_path']} onSet={onSet} pickFile />
      <SettingField label="checkpoint / HiDream dir"       k="image.checkpoint"         value={settings['image.checkpoint']}         onSet={onSet} pickFile pickDir />
      <SettingField label="CLIP-L tokenizer.json"          k="image.tokenizer_l"        value={settings['image.tokenizer_l']}        onSet={onSet} pickFile />
      <SettingField label="CLIP-G tokenizer.json (SDXL)"   k="image.tokenizer_g"        value={settings['image.tokenizer_g']}        onSet={onSet} pickFile />
      <SettingField label="output dir"                     k="image.output_dir"         value={settings['image.output_dir']}         onSet={onSet} pickDir />
      <SettingField label="motion adapter (AnimateDiff)"  k="image.motion_adapter"     value={settings['image.motion_adapter']}     onSet={onSet} pickFile />
    </div>
  );
}

function SettingField({ label, placeholder, k, value, onSet, pickFile, pickDir }: {
  label: string;
  placeholder?: string;
  k: SettingKey;
  value?: string;
  onSet: (k: SettingKey, v: string) => Promise<void>;
  pickFile?: boolean;
  pickDir?: boolean;
}) {
  const [draft, setDraft] = useState(value || '');
  useEffect(() => { setDraft(value || ''); }, [value]);
  async function pick(directory: boolean) {
    const { open } = await import('@tauri-apps/plugin-dialog');
    const picked = await open({ directory, multiple: false, title: label });
    if (typeof picked === 'string') {
      setDraft(picked);
      await onSet(k, picked);
    }
  }
  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-gray-300 w-40 shrink-0">{label}</label>
      <input
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => { if (draft !== (value || '')) onSet(k, draft); }}
        className="flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100"
      />
      {pickFile && (
        <button onClick={() => pick(false)} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100">
          file…
        </button>
      )}
      {pickDir && (
        <button onClick={() => pick(true)} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100">
          dir…
        </button>
      )}
    </div>
  );
}

// ----------------------------------------------------------------------------

function useSessionState<T>(key: string, initial: T): [T, (v: T) => void] {
  const [val, setVal] = useState<T>(() => {
    try {
      const stored = sessionStorage.getItem(key);
      return stored != null ? JSON.parse(stored) : initial;
    } catch { return initial; }
  });
  const set = useCallback((v: T) => {
    setVal(v);
    try { sessionStorage.setItem(key, JSON.stringify(v)); } catch {}
  }, [key]);
  return [val, set];
}

function Composer({ configured, generating, onGenerate, onGenerateVideo, onAbort, onResetAbort }: {
  configured: boolean;
  generating: boolean;
  onGenerate: (req: import('../lib/tauri').GenerateRequest) => Promise<boolean>;
  onGenerateVideo: (req: VideoGenerateRequest) => Promise<void>;
  onAbort: () => Promise<void>;
  onResetAbort: () => void;
}) {
  const [prompt, setPrompt] = useSessionState('img_prompt', '');
  const [neg, setNeg] = useSessionState('img_neg', '');
  const [steps, setSteps] = useSessionState<number>('img_steps', 28);
  const [cfg, setCfg] = useSessionState<number>('img_cfg', 7.5);
  const [seed, setSeed] = useSessionState<string>('img_seed', '');
  const [batch, setBatch] = useSessionState<number>('img_batch', 1);
  const [mode, setMode] = useSessionState<'image' | 'video'>('img_mode', 'image');
  const [numFrames, setNumFrames] = useSessionState<number>('img_frames', 16);
  const [fps, setFps] = useSessionState<number>('img_fps', 8);
  // Reference images for HiDream-O1 reference editing. Held in memory only
  // (Uint8Array bytes + a preview object URL); never persisted — so this works
  // in incognito with no filesystem round-trip.
  const [refImages, setRefImages] = useState<{ bytes: Uint8Array; url: string }[]>([]);
  const refInputRef = useRef<HTMLInputElement | null>(null);

  async function addRefImages(files: FileList | null) {
    if (!files) return;
    const added: { bytes: Uint8Array; url: string }[] = [];
    for (const file of Array.from(files)) {
      const buf = new Uint8Array(await file.arrayBuffer());
      added.push({ bytes: buf, url: URL.createObjectURL(new Blob([buf], { type: file.type || 'image/png' })) });
    }
    setRefImages((prev) => [...prev, ...added]);
  }

  function removeRefImage(idx: number) {
    setRefImages((prev) => {
      const next = prev.slice();
      const [gone] = next.splice(idx, 1);
      if (gone) URL.revokeObjectURL(gone.url);
      return next;
    });
  }

  // Release preview object URLs on unmount.
  useEffect(() => () => { refImages.forEach((r) => URL.revokeObjectURL(r.url)); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function submit() {
    if (!prompt.trim() || generating) return;
    const baseSeed = seed.trim() ? Number(seed) : undefined;
    if (mode === 'video') {
      await onGenerateVideo({
        prompt,
        negative_prompt: neg.trim() || undefined,
        steps,
        cfg_scale: cfg,
        seed: baseSeed,
        num_frames: numFrames,
        fps,
      });
    } else {
      const count = Math.max(1, Math.min(batch, 100));
      const refs = refImages.length ? refImages.map((r) => r.bytes) : undefined;
      onResetAbort(); // clear any prior abort before starting this batch
      for (let i = 0; i < count; i++) {
        const ok = await onGenerate({
          prompt,
          negative_prompt: neg.trim() || undefined,
          steps,
          cfg_scale: cfg,
          seed: baseSeed != null ? baseSeed + i : undefined,
          ref_image_bytes: refs,
        });
        if (!ok) break; // user aborted — stop the whole batch
      }
    }
  }

  return (
    <div className="border-b border-gray-700 px-4 py-3 space-y-2">
      {/* Image / Video mode toggle */}
      <div className="flex bg-gray-800 rounded-lg p-0.5 w-fit">
        <button
          onClick={() => setMode('image')}
          className={`px-3 py-0.5 rounded-md text-xs font-medium ${
            mode === 'image' ? 'bg-gray-600 text-gray-100' : 'text-gray-400'
          }`}
        >Image</button>
        <button
          onClick={() => setMode('video')}
          className={`px-3 py-0.5 rounded-md text-xs font-medium ${
            mode === 'video' ? 'bg-purple-700 text-purple-50' : 'text-gray-400'
          }`}
        >Video</button>
      </div>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); submit(); }
        }}
        disabled={!configured}
        placeholder={configured ? 'prompt — ⌘⏎ to generate' : 'configure checkpoint in ⚙ settings first…'}
        rows={2}
        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 disabled:opacity-60 resize-y min-h-[56px]"
      />
      <input
        value={neg}
        onChange={(e) => setNeg(e.target.value)}
        disabled={!configured}
        placeholder="negative prompt (optional)"
        className="w-full bg-gray-800 border border-gray-700 rounded px-3 py-1.5 text-sm text-gray-100 disabled:opacity-60"
      />
      {mode === 'image' && (
        <div className="flex flex-wrap items-center gap-2">
          <input
            ref={refInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(e) => { addRefImages(e.target.files); e.target.value = ''; }}
          />
          <button
            type="button"
            disabled={!configured}
            onClick={() => refInputRef.current?.click()}
            className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100 disabled:opacity-60"
            title="HiDream-O1 reference editing: condition generation on one or more reference images"
          >+ reference image</button>
          {refImages.map((r, i) => (
            <div key={i} className="relative">
              <img src={r.url} alt={`ref ${i + 1}`} className="h-12 w-12 object-cover rounded border border-gray-600" />
              <button
                type="button"
                onClick={() => removeRefImage(i)}
                className="absolute -top-1.5 -right-1.5 h-4 w-4 rounded-full bg-gray-900 border border-gray-600 text-gray-300 hover:text-white text-[10px] leading-none flex items-center justify-center"
                title="remove"
              >×</button>
            </div>
          ))}
          {refImages.length > 0 && (
            <span className="text-[11px] text-gray-500">reference editing (HiDream-O1)</span>
          )}
        </div>
      )}
      <div className="flex flex-wrap items-end gap-3">
        <Field label="steps">
          <input type="number" value={steps} onChange={(e) => setSteps(Number(e.target.value))} min={1} max={200}
            className="w-16 sm:w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
        </Field>
        <Field label="cfg">
          <input type="number" step="0.1" value={cfg} onChange={(e) => setCfg(Number(e.target.value))} min={0} max={30}
            className="w-16 sm:w-20 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
        </Field>
        <Field label="seed">
          <input value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" inputMode="numeric"
            className="w-20 sm:w-28 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
        </Field>
        {mode === 'image' && (
          <Field label="batch">
            <input type="number" value={batch} onChange={(e) => setBatch(Number(e.target.value))} min={1} max={100}
              className="w-14 sm:w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
          </Field>
        )}
        {mode === 'video' && (
          <>
            <Field label="frames">
              <input type="number" value={numFrames} onChange={(e) => setNumFrames(Number(e.target.value))} min={2} max={64}
                className="w-14 sm:w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
            </Field>
            <Field label="fps">
              <input type="number" value={fps} onChange={(e) => setFps(Number(e.target.value))} min={1} max={60}
                className="w-14 sm:w-16 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100" />
            </Field>
          </>
        )}
        <div className="ml-auto">
          {generating ? (
            <button onClick={onAbort} className="text-sm px-3 py-1.5 rounded bg-amber-700 hover:bg-amber-600 text-amber-50">
              Stop
            </button>
          ) : (
            <button onClick={submit} disabled={!configured || !prompt.trim()}
              className={`text-sm px-3 py-1.5 rounded disabled:bg-gray-800 disabled:text-gray-500 text-gray-100 ${
                mode === 'video' ? 'bg-purple-700 hover:bg-purple-600' : 'bg-blue-700 hover:bg-blue-600'
              }`}>
              {mode === 'video' ? 'Generate Video' : 'Generate'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-wide text-gray-500 mb-0.5">{label}</span>
      {children}
    </div>
  );
}

// ----------------------------------------------------------------------------

function ResultArea({ progress, generating, lastError, recentLogs, gallery, onImageClick }: {
  progress: ReturnType<typeof useImageGen>['progress'];
  generating: boolean;
  lastError: string | null;
  recentLogs: string[];
  gallery: StoredImage[];
  onImageClick: (idx: number) => void;
}) {
  const latest = useMemo(() => gallery[0], [gallery]);
  const previewPath = progress?.outPath || latest?.path;
  const previewIdx = useMemo(() => {
    if (!previewPath) return -1;
    return gallery.findIndex((g) => g.path === previewPath);
  }, [previewPath, gallery]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      {generating && progress && (
        <div className="mb-3">
          <div className="flex items-center gap-2 text-xs text-gray-300 mb-1">
            <span>generating · step {progress.step}/{progress.total || '?'} · {progress.elapsedS.toFixed(1)}s</span>
          </div>
          <div className="w-full bg-gray-800 rounded h-1.5 overflow-hidden">
            <div
              className="bg-blue-500 h-full transition-all"
              style={{
                width: progress.total > 0
                  ? `${(progress.step / progress.total) * 100}%`
                  : '5%',
              }}
            />
          </div>
        </div>
      )}
      {lastError && (
        <div className="mb-3 text-xs px-3 py-2 rounded bg-red-900/40 text-red-200">{lastError}</div>
      )}
      {previewPath ? (
        <ImagePreview
          path={previewPath}
          caption={generating ? 'in progress…' : undefined}
          onClick={previewIdx >= 0 ? () => onImageClick(previewIdx) : undefined}
        />
      ) : (
        <div className="text-sm text-gray-500 italic">
          Configure the engine, enter a prompt, hit Generate. Output appears here.
        </div>
      )}
      {recentLogs.length > 0 && (
        <details className="mt-4">
          <summary className="text-xs text-gray-500 cursor-pointer">subprocess log ({recentLogs.length})</summary>
          <pre className="text-[11px] text-gray-400 font-mono mt-1 whitespace-pre-wrap leading-tight">
            {recentLogs.join('\n')}
          </pre>
        </details>
      )}
    </div>
  );
}

function isVideoPath(path: string): boolean {
  return path.endsWith('.mp4') || path.endsWith('.webm') || path.startsWith('data:video/');
}

function ImagePreview({ path, caption, onClick }: { path: string; caption?: string; onClick?: () => void }) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => { imageSrc(path).then(setSrc); }, [path]);
  const isData = path.startsWith('data:');
  const isVideo = isVideoPath(path);
  return (
    <div className="flex flex-col items-start gap-2">
      {src && isVideo ? (
        <video
          src={src}
          autoPlay
          loop
          muted
          playsInline
          className={`max-w-full max-h-[70vh] rounded border border-gray-700 ${onClick ? 'cursor-pointer hover:ring-2 hover:ring-purple-500/50' : ''}`}
          onClick={onClick}
        />
      ) : src ? (
        <img
          src={src}
          alt=""
          className={`max-w-full max-h-[70vh] rounded border border-gray-700 ${onClick ? 'cursor-pointer hover:ring-2 hover:ring-blue-500/50' : ''}`}
          onClick={onClick}
        />
      ) : null}
      {caption && <div className="text-[11px] text-gray-400">{caption}</div>}
      {!isData && <div className="text-[11px] text-gray-500 break-all">{path}</div>}
    </div>
  );
}

// ----------------------------------------------------------------------------

function GalleryThumb({ image, onClick, onDelete }: {
  image: StoredImage;
  onClick: () => void;
  onDelete: () => Promise<void>;
}) {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => { imageSrc(image.path).then(setSrc); }, [image.path]);
  return (
    <div className="group relative border-b border-gray-800 cursor-pointer" onClick={onClick}>
      {src && <img src={src} alt="" className="w-full h-32 object-cover" />}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity p-2 flex flex-col justify-end">
        <div className="text-[10px] text-gray-200 line-clamp-3">{image.prompt}</div>
        <div className="absolute top-1 left-1">
          <FeedbackButtons kind="image" targetId={image.id} layout="row" size={12} />
        </div>
        <button
          onClick={(e) => { e.preventDefault(); e.stopPropagation(); onDelete(); }}
          className="absolute top-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-red-900/70 hover:bg-red-800 text-red-100"
          title="delete (also removes file)"
        >
          ×
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------
// Carousel lightbox — full-window overlay with arrow nav + autoplay.

function Carousel({ gallery, index, onIndexChange, onClose, onUseAsInput }: {
  gallery: StoredImage[];
  index: number;
  onIndexChange: (idx: number) => void;
  onClose: () => void;
  onUseAsInput?: (img: StoredImage) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [autoplay, setAutoplay] = useState(false);
  const autoRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const image = gallery[index];

  useEffect(() => {
    if (!image) { onClose(); return; }
    imageSrc(image.path).then(setSrc);
  }, [image, onClose]);

  const prev = useCallback(() => {
    onIndexChange((index - 1 + gallery.length) % gallery.length);
  }, [index, gallery.length, onIndexChange]);

  const next = useCallback(() => {
    onIndexChange((index + 1) % gallery.length);
  }, [index, gallery.length, onIndexChange]);

  // Keyboard navigation
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      else if (e.key === 'ArrowLeft') prev();
      else if (e.key === 'ArrowRight') next();
      else if (e.key === ' ') { e.preventDefault(); setAutoplay((a) => !a); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose, prev, next]);

  // Autoplay timer
  useEffect(() => {
    if (autoplay) {
      autoRef.current = setInterval(next, 5000);
    }
    return () => { if (autoRef.current) clearInterval(autoRef.current); };
  }, [autoplay, next]);

  if (!image) return null;

  const isData = image.path.startsWith('data:');

  return (
    <div
      className="fixed inset-0 z-[100] bg-black flex items-center justify-center"
      onClick={onClose}
    >
      {src && isVideoPath(image.path) ? (
        <video
          src={src}
          autoPlay loop muted playsInline controls
          className="absolute inset-0 w-full h-full object-contain"
          onClick={(e) => e.stopPropagation()}
        />
      ) : src ? (
        <img
          src={src}
          alt=""
          className="absolute inset-0 w-full h-full object-contain"
          onClick={(e) => e.stopPropagation()}
        />
      ) : null}

      {/* Overlaid controls — sit on top of the image */}

      {/* Top bar */}
      <div
        className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 py-2 bg-gradient-to-b from-black/70 to-transparent z-10"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="text-sm text-white/80 max-w-[55%] truncate">
          {image.prompt}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-white/50 tabular-nums">
            {index + 1}/{gallery.length}
          </span>
          {onUseAsInput && !isVideoPath(image.path) && (
            <button
              onClick={() => onUseAsInput(image)}
              className="text-xs px-2 py-1 rounded bg-white/10 text-white/70 hover:bg-white/20"
              title="Send to Inpaint as the source image (with or without a mask)"
            >
              ✎ Use as input
            </button>
          )}
          <button
            onClick={() => setAutoplay((a) => !a)}
            className={`text-xs px-2 py-1 rounded ${
              autoplay ? 'bg-blue-600 text-white' : 'bg-white/10 text-white/70 hover:bg-white/20'
            }`}
          >
            {autoplay ? '⏸' : '▶'}
          </button>
          <button
            onClick={onClose}
            className="text-xs px-2 py-1 rounded bg-white/10 text-white/70 hover:bg-white/20"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Arrow buttons — overlaid on the image */}
      {gallery.length > 1 && (
        <>
          <button
            onClick={(e) => { e.stopPropagation(); prev(); }}
            className="absolute left-2 top-1/2 -translate-y-1/2 text-3xl text-white/60 hover:text-white bg-black/30 hover:bg-black/60 rounded-full w-11 h-11 flex items-center justify-center z-10 backdrop-blur-sm"
          >
            ‹
          </button>
          <button
            onClick={(e) => { e.stopPropagation(); next(); }}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-3xl text-white/60 hover:text-white bg-black/30 hover:bg-black/60 rounded-full w-11 h-11 flex items-center justify-center z-10 backdrop-blur-sm"
          >
            ›
          </button>
        </>
      )}

      {/* Bottom info */}
      <div
        className="absolute bottom-0 left-0 right-0 px-4 py-2 flex flex-wrap items-center gap-3 text-xs text-white/40 bg-gradient-to-t from-black/70 to-transparent z-10"
        onClick={(e) => e.stopPropagation()}
      >
        {image.seed != null && <span>seed: {image.seed}</span>}
        {image.steps != null && <span>steps: {image.steps}</span>}
        {image.cfg_scale != null && <span>cfg: {image.cfg_scale}</span>}
        {!isData && <span className="truncate max-w-[40%]">{image.path}</span>}
        <span className="hidden sm:inline ml-auto text-[10px] text-white/25">← → · Space · Esc</span>
      </div>
    </div>
  );
}
