// Memory inspector: browse, edit, create, delete entries.
//
// Mirrors the four kinds from the auto-memory format (user, feedback,
// project, reference). The editor is intentionally plain — name +
// one-line description + multi-line markdown body. Later phases will
// add embedding-based search and sync with the on-disk format.

import { useEffect, useState } from 'react';
import {
  memoryList,
  memoryUpsert,
  memoryDelete,
  type MemoryEntry,
  type MemoryKind,
} from '../lib/tauri';

const KINDS: { value: MemoryKind | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'user', label: 'User' },
  { value: 'feedback', label: 'Feedback' },
  { value: 'project', label: 'Project' },
  { value: 'reference', label: 'Reference' },
];

function kindColor(k: MemoryKind): string {
  switch (k) {
    case 'user': return 'bg-blue-900/40 text-blue-300';
    case 'feedback': return 'bg-amber-900/40 text-amber-300';
    case 'project': return 'bg-emerald-900/40 text-emerald-300';
    case 'reference': return 'bg-purple-900/40 text-purple-300';
  }
}

export default function MemoryPanel() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [filter, setFilter] = useState<MemoryKind | 'all'>('all');
  const [selected, setSelected] = useState<MemoryEntry | null>(null);
  const [editing, setEditing] = useState<MemoryEntry | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function refresh() {
    setLoading(true);
    try {
      const list = await memoryList(filter === 'all' ? undefined : filter);
      setEntries(list);
      // Drop selection if it disappeared.
      if (selected && !list.find((e) => e.id === selected.id)) setSelected(null);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { refresh(); }, [filter]);

  async function save(e: MemoryEntry) {
    try {
      const saved = await memoryUpsert({
        id: e.id === 0 ? null : e.id,
        kind: e.kind,
        name: e.name,
        description: e.description,
        body: e.body,
      });
      setEditing(null);
      setSelected(saved);
      await refresh();
    } catch (err) { setErr(String(err)); }
  }

  async function remove(id: number) {
    if (!confirm('Delete this memory entry?')) return;
    try { await memoryDelete(id); await refresh(); }
    catch (err) { setErr(String(err)); }
  }

  function newEntry(): MemoryEntry {
    const now = Math.floor(Date.now() / 1000);
    return {
      id: 0,
      kind: 'user',
      name: '',
      description: '',
      body: '',
      created_at: now,
      updated_at: now,
    };
  }

  return (
    <div className="flex flex-1 min-h-0">
      {/* Sidebar */}
      <aside className="w-72 flex flex-col border-r border-gray-700 bg-gray-900">
        <div className="flex items-center gap-1 px-2 py-2 border-b border-gray-700">
          {KINDS.map((k) => (
            <button
              key={k.value}
              onClick={() => setFilter(k.value)}
              className={`text-xs px-2 py-1 rounded transition-colors ${
                filter === k.value ? 'bg-gray-700 text-gray-100' : 'text-gray-400 hover:bg-gray-800'
              }`}
            >
              {k.label}
            </button>
          ))}
          <button
            onClick={() => setEditing(newEntry())}
            title="New entry"
            className="ml-auto text-gray-400 hover:text-gray-100 px-2 py-1 rounded hover:bg-gray-800"
          >
            +
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {loading && <div className="px-3 py-6 text-xs text-gray-500">loading…</div>}
          {!loading && entries.length === 0 && (
            <div className="px-3 py-6 text-xs text-gray-500">No memories yet. Click + to add one.</div>
          )}
          {entries.map((e) => (
            <button
              key={e.id}
              onClick={() => { setSelected(e); setEditing(null); }}
              className={`w-full text-left px-3 py-2 border-b border-gray-800 transition-colors ${
                selected?.id === e.id ? 'bg-gray-800' : 'hover:bg-gray-800/50'
              }`}
            >
              <div className="flex items-center gap-2 text-xs mb-0.5">
                <span className={`px-1.5 py-0.5 rounded ${kindColor(e.kind)}`}>{e.kind}</span>
                <span className="text-gray-200 truncate flex-1">{e.name || '(unnamed)'}</span>
              </div>
              <div className="text-xs text-gray-500 truncate">{e.description || '—'}</div>
            </button>
          ))}
        </div>
        {err && (
          <div className="px-3 py-2 text-xs bg-red-900/40 text-red-300 border-t border-red-900">
            {err}
          </div>
        )}
      </aside>

      {/* Detail / editor */}
      <section className="flex-1 flex flex-col min-w-0">
        {editing ? (
          <Editor
            initial={editing}
            onCancel={() => setEditing(null)}
            onSave={save}
          />
        ) : selected ? (
          <Detail
            entry={selected}
            onEdit={() => setEditing(selected)}
            onDelete={() => remove(selected.id)}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-gray-500">
            Select an entry, or click + to create one.
          </div>
        )}
      </section>
    </div>
  );
}

function Detail({ entry, onEdit, onDelete }: {
  entry: MemoryEntry;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <>
      <header className="flex items-center gap-3 px-4 py-3 border-b border-gray-700">
        <span className={`text-xs px-2 py-0.5 rounded ${kindColor(entry.kind)}`}>{entry.kind}</span>
        <h2 className="text-base font-medium text-gray-100 truncate">{entry.name}</h2>
        <div className="ml-auto flex gap-2">
          <button onClick={onEdit} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100">Edit</button>
          <button onClick={onDelete} className="text-xs px-2 py-1 rounded bg-red-900/60 hover:bg-red-900 text-red-200">Delete</button>
        </div>
      </header>
      <div className="px-4 py-2 text-xs text-gray-400 border-b border-gray-800">
        {entry.description || <span className="italic text-gray-600">no description</span>}
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-4 text-sm text-gray-200 whitespace-pre-wrap font-mono">
        {entry.body || <span className="italic text-gray-600">empty body</span>}
      </div>
    </>
  );
}

function Editor({ initial, onCancel, onSave }: {
  initial: MemoryEntry;
  onCancel: () => void;
  onSave: (e: MemoryEntry) => void;
}) {
  const [e, setE] = useState<MemoryEntry>(initial);
  return (
    <>
      <header className="flex items-center gap-3 px-4 py-3 border-b border-gray-700">
        <select
          value={e.kind}
          onChange={(ev) => setE({ ...e, kind: ev.target.value as MemoryKind })}
          className="bg-gray-800 text-gray-100 text-sm rounded px-2 py-1 border border-gray-700"
        >
          <option value="user">user</option>
          <option value="feedback">feedback</option>
          <option value="project">project</option>
          <option value="reference">reference</option>
        </select>
        <input
          autoFocus
          placeholder="short-kebab-case-slug"
          value={e.name}
          onChange={(ev) => setE({ ...e, name: ev.target.value })}
          className="flex-1 bg-gray-800 text-gray-100 text-base rounded px-2 py-1 border border-gray-700"
        />
        <button onClick={onCancel} className="text-xs px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-gray-100">Cancel</button>
        <button
          onClick={() => onSave(e)}
          disabled={!e.name.trim()}
          className="text-xs px-2 py-1 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 disabled:text-gray-600 text-gray-100"
        >
          Save
        </button>
      </header>
      <input
        placeholder="one-line description (used to decide relevance)"
        value={e.description}
        onChange={(ev) => setE({ ...e, description: ev.target.value })}
        className="mx-4 mt-3 bg-gray-800 text-gray-100 text-sm rounded px-2 py-1 border border-gray-700"
      />
      <textarea
        placeholder="memory body — markdown"
        value={e.body}
        onChange={(ev) => setE({ ...e, body: ev.target.value })}
        className="mx-4 my-3 flex-1 bg-gray-800 text-gray-100 text-sm rounded px-3 py-2 border border-gray-700 font-mono resize-none"
      />
    </>
  );
}
