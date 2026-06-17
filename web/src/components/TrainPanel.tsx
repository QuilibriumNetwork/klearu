// Train + datasets panel.
//
// Three sub-views, switched via a small tab bar at the top:
//   - Jobs: list of recent training/distillation runs with live progress
//   - Train: launch an `img_train` run
//   - Distill: launch the SDXL → DaViT → VAE-tokens distillation script
//   - Datasets: register existing directories / manifests as datasets

import { useCallback, useEffect, useState } from 'react';
import {
  datasetDelete,
  datasetList,
  datasetUpsert,
  jobAbort,
  jobDistill,
  jobGet,
  jobList,
  jobLogTail,
  jobTrainImage,
  type DatasetKind,
  type DatasetRow,
  type JobDistillArgs,
  type JobProgress,
  type JobRow,
  type JobTrainImageArgs,
} from '../lib/tauri';

type Sub = 'jobs' | 'train' | 'distill' | 'datasets';

export default function TrainPanel() {
  const [sub, setSub] = useState<Sub>('jobs');
  return (
    <div className="flex flex-1 flex-col min-h-0">
      <nav className="border-b border-gray-700 px-3 py-2 flex gap-1 bg-gray-900">
        {(['jobs', 'train', 'distill', 'datasets'] as Sub[]).map((s) => (
          <button
            key={s}
            onClick={() => setSub(s)}
            className={`text-xs px-3 py-1 rounded-md ${
              sub === s ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {s[0].toUpperCase() + s.slice(1)}
          </button>
        ))}
      </nav>
      <div className="flex-1 min-h-0 overflow-hidden">
        {sub === 'jobs' && <JobsView />}
        {sub === 'train' && <TrainImageForm onLaunched={() => setSub('jobs')} />}
        {sub === 'distill' && <DistillForm onLaunched={() => setSub('jobs')} />}
        {sub === 'datasets' && <DatasetsView />}
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------

function JobsView() {
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<JobRow | null>(null);
  const [log, setLog] = useState<string[]>([]);

  const refresh = useCallback(async () => {
    const list = await jobList(50);
    setJobs(list);
    if (selectedId == null && list.length > 0) setSelectedId(list[0].id);
  }, [selectedId]);

  useEffect(() => { refresh(); }, [refresh]);

  // Poll every 1.5s while there's an active job. When all done, slow down.
  useEffect(() => {
    const any = jobs.some((j) => j.status === 'running');
    const ms = any ? 1500 : 6000;
    const t = setInterval(refresh, ms);
    return () => clearInterval(t);
  }, [jobs, refresh]);

  useEffect(() => {
    if (selectedId == null) { setDetail(null); setLog([]); return; }
    let cancelled = false;
    const tick = async () => {
      try {
        const [d, l] = await Promise.all([jobGet(selectedId), jobLogTail(selectedId, 300)]);
        if (cancelled) return;
        setDetail(d);
        setLog(l);
      } catch {}
    };
    tick();
    const t = setInterval(tick, detail?.status === 'running' ? 1500 : 6000);
    return () => { cancelled = true; clearInterval(t); };
  }, [selectedId, detail?.status]);

  return (
    <div className="flex h-full">
      <aside className="w-72 flex flex-col border-r border-gray-700 bg-gray-900">
        <div className="px-3 py-2 border-b border-gray-700 text-xs font-medium text-gray-200">
          Recent jobs
        </div>
        <div className="flex-1 overflow-y-auto">
          {jobs.length === 0 && (
            <div className="px-3 py-6 text-xs text-gray-500">No jobs yet.</div>
          )}
          {jobs.map((j) => (
            <button
              key={j.id}
              onClick={() => setSelectedId(j.id)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800 ${
                selectedId === j.id ? 'bg-gray-800' : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <StatusChip status={j.status} />
                <span className="text-xs text-gray-300">{j.kind}</span>
              </div>
              <div className="text-sm text-gray-100 truncate">{j.label}</div>
              <div className="text-[10px] text-gray-500">
                {new Date(j.started_at * 1000).toLocaleString()}
              </div>
            </button>
          ))}
        </div>
      </aside>
      <section className="flex-1 flex flex-col min-w-0">
        {detail ? (
          <JobDetail row={detail} log={log} onAbort={async () => {
            await jobAbort(detail.id);
            refresh();
          }} />
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-gray-500">
            Select a job to see details.
          </div>
        )}
      </section>
    </div>
  );
}

function StatusChip({ status }: { status: string }) {
  const map: Record<string, string> = {
    running:   'bg-blue-900/60 text-blue-200',
    completed: 'bg-emerald-900/60 text-emerald-200',
    failed:    'bg-red-900/60 text-red-200',
    aborted:   'bg-amber-900/60 text-amber-200',
  };
  return (
    <span className={`text-[10px] px-1.5 py-0.5 rounded ${map[status] ?? 'bg-gray-800 text-gray-300'}`}>
      {status}
    </span>
  );
}

function JobDetail({ row, log, onAbort }: { row: JobRow; log: string[]; onAbort: () => Promise<void> }) {
  const progress: JobProgress = (() => {
    try { return JSON.parse(row.progress); } catch { return {}; }
  })();
  const pct = progress.step != null && progress.total
    ? Math.min(100, (progress.step / progress.total) * 100)
    : null;
  return (
    <>
      <header className="px-4 py-3 border-b border-gray-700 flex items-center gap-3">
        <StatusChip status={row.status} />
        <div className="flex-1 min-w-0">
          <div className="text-base font-medium text-gray-100 truncate">{row.label}</div>
          <div className="text-[11px] text-gray-500 truncate">{row.command}</div>
        </div>
        {row.status === 'running' && (
          <button
            onClick={onAbort}
            className="text-xs px-2 py-1 rounded bg-amber-700 hover:bg-amber-600 text-amber-50"
          >
            Abort
          </button>
        )}
      </header>
      <div className="px-4 py-3 border-b border-gray-700 grid grid-cols-4 gap-3 text-xs">
        <Metric label="step" value={progress.step != null ? `${progress.step}${progress.total ? '/' + progress.total : ''}` : '—'} />
        <Metric label="loss" value={progress.loss != null ? progress.loss.toFixed(4) : '—'} />
        <Metric label="elapsed" value={progress.elapsed_s != null ? `${progress.elapsed_s.toFixed(1)}s` : '—'} />
        <Metric label="started" value={new Date(row.started_at * 1000).toLocaleTimeString()} />
      </div>
      {pct != null && (
        <div className="px-4 pt-2">
          <div className="w-full bg-gray-800 rounded h-1.5 overflow-hidden">
            <div className="bg-blue-500 h-full transition-all" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <pre className="text-[11px] text-gray-300 font-mono whitespace-pre-wrap leading-tight">
          {log.length === 0 ? '(no log yet)' : log.join('\n')}
        </pre>
      </div>
    </>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-sm text-gray-100 tabular-nums">{value}</div>
    </div>
  );
}

// ----------------------------------------------------------------------------

function TrainImageForm({ onLaunched }: { onLaunched: () => void }) {
  const [form, setForm] = useState<JobTrainImageArgs>({
    manifest: '', out: 'klearu_image.safetensors', steps: 10000, lr: 3e-4, log_every: 100,
  });
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function pick(field: 'manifest' | 'out' | 'resume' | 'bpe' | 'binary_path', isDir = false) {
    const { open, save } = await import('@tauri-apps/plugin-dialog');
    let picked: string | null = null;
    if (field === 'out') {
      const p = await save({ title: 'Output checkpoint', defaultPath: form.out });
      picked = typeof p === 'string' ? p : null;
    } else {
      const p = await open({ multiple: false, directory: isDir, title: field });
      picked = typeof p === 'string' ? p : null;
    }
    if (picked) setForm((f) => ({ ...f, [field]: picked }));
  }

  async function submit() {
    setErr(null);
    if (!form.manifest) { setErr('manifest is required'); return; }
    setSubmitting(true);
    try {
      await jobTrainImage(form);
      onLaunched();
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setErr(msg);
    } finally { setSubmitting(false); }
  }

  return (
    <div className="px-6 py-6 overflow-y-auto h-full">
      <h2 className="text-base font-medium text-gray-100 mb-1">Train klearu-image transformer</h2>
      <p className="text-xs text-gray-500 mb-4">
        Spawns <code className="text-gray-300">img_train</code>. The manifest must be JSONL with
        {' '}<code>caption</code> + <code>image_tokens</code> per line.
      </p>
      <Row label="Label">
        <input value={form.label ?? ''} onChange={(e) => setForm({ ...form, label: e.target.value })}
          placeholder="auto" className={inputClass} />
      </Row>
      <Row label="Manifest (.jsonl)" required>
        <input value={form.manifest} onChange={(e) => setForm({ ...form, manifest: e.target.value })}
          className={inputClass} />
        <button onClick={() => pick('manifest')} className={pickClass}>pick…</button>
      </Row>
      <Row label="Output checkpoint">
        <input value={form.out} onChange={(e) => setForm({ ...form, out: e.target.value })}
          className={inputClass} />
        <button onClick={() => pick('out')} className={pickClass}>pick…</button>
      </Row>
      <Row label="Steps">
        <input type="number" value={form.steps} onChange={(e) => setForm({ ...form, steps: Number(e.target.value) })}
          className={inputClassNarrow} />
      </Row>
      <Row label="Learning rate">
        <input type="number" step="0.0001" value={form.lr} onChange={(e) => setForm({ ...form, lr: Number(e.target.value) })}
          className={inputClassNarrow} />
      </Row>
      <Row label="Log every">
        <input type="number" value={form.log_every} onChange={(e) => setForm({ ...form, log_every: Number(e.target.value) })}
          className={inputClassNarrow} />
      </Row>
      <Row label="Seed">
        <input type="number" value={form.seed ?? 42} onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
          className={inputClassNarrow} />
      </Row>
      <Row label="Resume from (optional)">
        <input value={form.resume ?? ''} onChange={(e) => setForm({ ...form, resume: e.target.value || undefined })}
          className={inputClass} />
        <button onClick={() => pick('resume')} className={pickClass}>pick…</button>
      </Row>
      <Row label="BPE tokenizer.json (optional)">
        <input value={form.bpe ?? ''} onChange={(e) => setForm({ ...form, bpe: e.target.value || undefined })}
          className={inputClass} />
        <button onClick={() => pick('bpe')} className={pickClass}>pick…</button>
      </Row>
      <Row label="img_train binary (override)">
        <input value={form.binary_path ?? ''} onChange={(e) => setForm({ ...form, binary_path: e.target.value || undefined })}
          placeholder="auto-locate via target/{release,debug}" className={inputClass} />
        <button onClick={() => pick('binary_path')} className={pickClass}>pick…</button>
      </Row>

      {err && <div className="mt-3 text-xs px-3 py-2 rounded bg-red-900/40 text-red-200">{err}</div>}
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={submit}
          disabled={submitting}
          className="text-sm px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 text-gray-100"
        >
          {submitting ? 'starting…' : 'Start training'}
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------

function DistillForm({ onLaunched }: { onLaunched: () => void }) {
  const [form, setForm] = useState<JobDistillArgs>({
    prompts: '', sdxl: '', out_dir: '', n_images: 1, steps: 30, guidance: 7.5, seed: 42,
  });
  const [err, setErr] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function pick(field: keyof JobDistillArgs, isDir = false) {
    const { open, save } = await import('@tauri-apps/plugin-dialog');
    let picked: string | null = null;
    if (field === 'out_dir') {
      const p = await open({ directory: true, multiple: false, title: 'Output directory' });
      picked = typeof p === 'string' ? p : null;
    } else {
      const p = await open({ multiple: false, directory: isDir, title: String(field) });
      picked = typeof p === 'string' ? p : null;
    }
    if (picked) setForm((f) => ({ ...f, [field]: picked }) as JobDistillArgs);
    void save; // unused branch placeholder
  }

  async function submit() {
    setErr(null);
    if (!form.prompts || !form.sdxl || !form.out_dir) {
      setErr('prompts, sdxl, and out_dir are required'); return;
    }
    setSubmitting(true);
    try { await jobDistill(form); onLaunched(); }
    catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setErr(msg);
    } finally { setSubmitting(false); }
  }

  return (
    <div className="px-6 py-6 overflow-y-auto h-full">
      <h2 className="text-base font-medium text-gray-100 mb-1">Distill from SDXL</h2>
      <p className="text-xs text-gray-500 mb-4">
        Drives <code className="text-gray-300">scripts/distill_from_sdxl.py</code> end-to-end:
        planner LLM → SDXL → optional DaViT filter → captions.jsonl + image dir.
        Output then feeds <code>scripts/encode_images.py</code> or <code>vqvae_encode.py</code>.
      </p>
      <Row label="Label">
        <input value={form.label ?? ''} onChange={(e) => setForm({ ...form, label: e.target.value })} className={inputClass} />
      </Row>
      <Row label="Prompts file" required>
        <input value={form.prompts} onChange={(e) => setForm({ ...form, prompts: e.target.value })} className={inputClass} />
        <button onClick={() => pick('prompts')} className={pickClass}>pick…</button>
      </Row>
      <Row label="SDXL checkpoint or dir" required>
        <input value={form.sdxl} onChange={(e) => setForm({ ...form, sdxl: e.target.value })} className={inputClass} />
        <button onClick={() => pick('sdxl')} className={pickClass}>pick…</button>
      </Row>
      <Row label="Output dir" required>
        <input value={form.out_dir} onChange={(e) => setForm({ ...form, out_dir: e.target.value })} className={inputClass} />
        <button onClick={() => pick('out_dir', true)} className={pickClass}>pick…</button>
      </Row>
      <Row label="Planner binary (optional)">
        <input value={form.planner_bin ?? ''} onChange={(e) => setForm({ ...form, planner_bin: e.target.value || undefined })}
          placeholder="e.g. target/release/plan_prompts" className={inputClass} />
        <button onClick={() => pick('planner_bin')} className={pickClass}>pick…</button>
      </Row>
      <Row label="DaViT filter (optional)">
        <input value={form.davit_model ?? ''} onChange={(e) => setForm({ ...form, davit_model: e.target.value || undefined })}
          placeholder="timm model id or dir" className={inputClass} />
      </Row>
      <Row label="DaViT min-conf">
        <input type="number" step="0.01" value={form.davit_conf ?? 0.15}
          onChange={(e) => setForm({ ...form, davit_conf: Number(e.target.value) })} className={inputClassNarrow} />
      </Row>
      <div className="grid grid-cols-4 gap-3 mt-2">
        <Field label="images/prompt">
          <input type="number" value={form.n_images ?? 1} onChange={(e) => setForm({ ...form, n_images: Number(e.target.value) })}
            className={inputClassNarrow} />
        </Field>
        <Field label="steps">
          <input type="number" value={form.steps ?? 30} onChange={(e) => setForm({ ...form, steps: Number(e.target.value) })}
            className={inputClassNarrow} />
        </Field>
        <Field label="guidance">
          <input type="number" step="0.1" value={form.guidance ?? 7.5} onChange={(e) => setForm({ ...form, guidance: Number(e.target.value) })}
            className={inputClassNarrow} />
        </Field>
        <Field label="seed">
          <input type="number" value={form.seed ?? 42} onChange={(e) => setForm({ ...form, seed: Number(e.target.value) })}
            className={inputClassNarrow} />
        </Field>
      </div>
      <Row label="Max prompts (cap)">
        <input type="number" value={form.max_prompts ?? 0} onChange={(e) => setForm({ ...form, max_prompts: Number(e.target.value) || undefined })}
          className={inputClassNarrow} placeholder="all" />
      </Row>

      {err && <div className="mt-3 text-xs px-3 py-2 rounded bg-red-900/40 text-red-200">{err}</div>}
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={submit}
          disabled={submitting}
          className="text-sm px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 text-gray-100"
        >
          {submitting ? 'starting…' : 'Start distillation'}
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------

function DatasetsView() {
  const [datasets, setDatasets] = useState<DatasetRow[]>([]);
  const [editing, setEditing] = useState<DatasetRow | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try { setDatasets(await datasetList()); } catch (e) { setErr(String(e)); }
  }, []);
  useEffect(() => { refresh(); }, [refresh]);

  function newDataset(): DatasetRow {
    return {
      id: 0, name: '', description: '', kind: 'images', path: '',
      item_count: 0, created_at: 0, updated_at: 0,
    };
  }

  async function save(d: DatasetRow) {
    setErr(null);
    try {
      await datasetUpsert({
        id: d.id === 0 ? undefined : d.id,
        name: d.name,
        description: d.description,
        kind: d.kind as DatasetKind,
        path: d.path,
      });
      setEditing(null);
      refresh();
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setErr(msg);
    }
  }

  async function remove(id: number) {
    if (!confirm('Delete this dataset record? (files on disk are NOT removed.)')) return;
    await datasetDelete(id);
    refresh();
  }

  return (
    <div className="flex h-full">
      <aside className="w-72 flex flex-col border-r border-gray-700 bg-gray-900">
        <div className="flex items-center px-3 py-2 border-b border-gray-700 text-xs">
          <span className="font-medium text-gray-200">Datasets</span>
          <button
            onClick={() => setEditing(newDataset())}
            className="ml-auto text-gray-400 hover:text-gray-100 px-2 py-0.5 rounded hover:bg-gray-800"
            title="Register a dataset"
          >+</button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {datasets.length === 0 && (
            <div className="px-3 py-6 text-xs text-gray-500">No datasets registered.</div>
          )}
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => setEditing(d)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800 ${
                editing?.id === d.id ? 'bg-gray-800' : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="text-sm text-gray-100 truncate">{d.name}</div>
              <div className="text-[10px] text-gray-500">{d.kind} · {d.item_count} items</div>
            </button>
          ))}
        </div>
        {err && <div className="px-3 py-1 text-[11px] bg-red-900/40 text-red-200">{err}</div>}
      </aside>
      <section className="flex-1 min-w-0">
        {editing ? (
          <DatasetEditor
            initial={editing}
            onCancel={() => setEditing(null)}
            onSave={save}
            onDelete={editing.id !== 0 ? () => remove(editing.id) : undefined}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-gray-500 h-full">
            Select a dataset, or click + to register a new one.
          </div>
        )}
      </section>
    </div>
  );
}

function DatasetEditor({ initial, onCancel, onSave, onDelete }: {
  initial: DatasetRow;
  onCancel: () => void;
  onSave: (d: DatasetRow) => Promise<void>;
  onDelete?: () => void;
}) {
  const [d, setD] = useState(initial);
  return (
    <div className="px-6 py-6">
      <Row label="Name" required>
        <input value={d.name} onChange={(e) => setD({ ...d, name: e.target.value })} className={inputClass} />
      </Row>
      <Row label="Kind">
        <select
          value={d.kind}
          onChange={(e) => setD({ ...d, kind: e.target.value })}
          className={inputClassNarrow}
        >
          <option value="images">images (directory)</option>
          <option value="manifest">manifest (.jsonl)</option>
          <option value="vae-tokens">vae-tokens (.jsonl)</option>
        </select>
      </Row>
      <Row label="Path" required>
        <input value={d.path} onChange={(e) => setD({ ...d, path: e.target.value })} className={inputClass} />
        <button
          onClick={async () => {
            const { open } = await import('@tauri-apps/plugin-dialog');
            const isDir = d.kind === 'images';
            const picked = await open({ multiple: false, directory: isDir, title: 'Pick dataset' });
            if (typeof picked === 'string') setD({ ...d, path: picked });
          }}
          className={pickClass}
        >pick…</button>
      </Row>
      <Row label="Description">
        <textarea
          value={d.description}
          onChange={(e) => setD({ ...d, description: e.target.value })}
          rows={2}
          className={`${inputClass} resize-y min-h-[44px]`}
        />
      </Row>
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={() => onSave(d)}
          disabled={!d.name || !d.path}
          className="text-sm px-3 py-1.5 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 text-gray-100"
        >
          Save
        </button>
        <button onClick={onCancel} className="text-sm px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-100">
          Cancel
        </button>
        {onDelete && (
          <button onClick={onDelete} className="ml-auto text-sm px-3 py-1.5 rounded bg-red-900/60 hover:bg-red-900 text-red-200">
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

// ---- shared form bits ------------------------------------------------------

const inputClass = 'flex-1 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100';
const inputClassNarrow = 'w-32 bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-100';
const pickClass = 'text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100';

function Row({ label, required, children }: { label: string; required?: boolean; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 mb-2">
      <label className="text-xs text-gray-300 w-44 shrink-0">
        {label}{required && <span className="text-red-400"> *</span>}
      </label>
      {children}
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
