// Local chat panel — runs against the in-process LLM via Tauri.
// Includes a sidebar with conversation list + an "engine" widget for
// loading the checkpoint, and a transcript / input area.

import { useEffect, useRef, useState } from 'react';
import {
  llmStatus,
  llmLoad,
  llmUnload,
  type LlmStatus,
} from '../lib/tauri';
import { useLocalChat } from '../hooks/useLocalChat';
import AssistantBubble from './AssistantBubble';

export default function LocalChatPanel() {
  const chat = useLocalChat();
  const [status, setStatus] = useState<LlmStatus | null>(null);
  const [loadPath, setLoadPath] = useState('');
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { llmStatus().then(setStatus).catch(() => {}); }, []);

  async function pickAndLoad() {
    let path = loadPath.trim();
    if (!path) {
      try {
        const { open } = await import('@tauri-apps/plugin-dialog');
        const picked = await open({ directory: true, multiple: false, title: 'Select model directory' });
        if (typeof picked === 'string') path = picked;
        else return;
      } catch (e) { setLoadErr(String(e)); return; }
    }
    setLoadErr(null);
    setLoading(true);
    try {
      const s = await llmLoad(path);
      setStatus(s);
      setLoadPath('');
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e) ? (e as { message: string }).message : String(e);
      setLoadErr(msg);
    } finally { setLoading(false); }
  }

  async function unload() {
    await llmUnload();
    setStatus(await llmStatus());
  }

  return (
    <div className="flex flex-1 min-h-0">
      {/* Sidebar: engine + conversations */}
      <aside className="w-72 flex flex-col border-r border-gray-700 bg-gray-900">
        <EnginePanel
          status={status}
          loadPath={loadPath}
          setLoadPath={setLoadPath}
          onLoad={pickAndLoad}
          onUnload={unload}
          loading={loading}
          err={loadErr}
        />
        <div className="flex items-center justify-between px-3 py-2 border-t border-b border-gray-700 text-xs text-gray-400">
          <span>Conversations</span>
          <button
            onClick={() => chat.newConversation()}
            className="text-gray-300 hover:text-white px-1 rounded hover:bg-gray-800"
            title="New conversation"
          >
            +
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {chat.conversations.length === 0 && (
            <div className="px-3 py-4 text-xs text-gray-500">No conversations yet.</div>
          )}
          {chat.conversations.map((c) => (
            <button
              key={c.id}
              onClick={() => chat.selectConversation(c.id)}
              className={`w-full text-left px-3 py-2 border-b border-gray-800 transition-colors text-sm ${
                chat.conversation?.id === c.id ? 'bg-gray-800 text-gray-100' : 'text-gray-300 hover:bg-gray-800/50'
              }`}
            >
              <div className="truncate">{c.title || '(untitled)'}</div>
              <div className="text-[10px] text-gray-500">{new Date(c.updated_at * 1000).toLocaleString()}</div>
            </button>
          ))}
        </div>
      </aside>

      {/* Transcript */}
      <section className="flex-1 flex flex-col min-w-0">
        <Transcript chat={chat} engineReady={status?.loaded ?? false} />
        <Composer chat={chat} engineReady={status?.loaded ?? false} />
      </section>
    </div>
  );
}

function EnginePanel({ status, loadPath, setLoadPath, onLoad, onUnload, loading, err }: {
  status: LlmStatus | null;
  loadPath: string;
  setLoadPath: (s: string) => void;
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
        <span className="text-gray-200 font-medium">LLM engine</span>
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
          <span className="text-gray-500">hidden={status!.hidden_size}, layers={status!.num_layers}</span>
        </div>
      ) : (
        <>
          <input
            placeholder="model dir (or pick…)"
            value={loadPath}
            onChange={(e) => setLoadPath(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-100 mb-1"
          />
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

function Transcript({ chat, engineReady }: { chat: ReturnType<typeof useLocalChat>; engineReady: boolean; }) {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chat.messages.length, chat.streamingText]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {!engineReady && chat.messages.length === 0 && (
        <div className="text-sm text-gray-500 italic">
          Load a checkpoint in the sidebar to start chatting.
        </div>
      )}
      {chat.messages.map((m, i) => (
        <div key={m.id ?? i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
          {m.role === 'user' ? (
            <div className="max-w-[80%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap bg-blue-900/40 text-blue-100">
              {m.content}
            </div>
          ) : (
            <AssistantBubble text={m.content} messageId={m.id} />
          )}
        </div>
      ))}
      {(chat.streamingText || chat.generating) && (
        <div className="flex flex-col items-start gap-1">
          {chat.streamingText && <AssistantBubble text={chat.streamingText} streaming />}
          {chat.pendingTools.length > 0 && (
            <div className="flex flex-wrap gap-1 px-2">
              {chat.pendingTools.map((t, i) => (
                <span
                  key={`${t.iter}-${t.name}-${i}`}
                  className="text-[10px] px-2 py-0.5 rounded-full bg-amber-900/40 text-amber-200 animate-pulse"
                >
                  ⚙ running {t.name}…
                </span>
              ))}
            </div>
          )}
        </div>
      )}
      {chat.error && (
        <div className="text-xs text-red-300 bg-red-900/30 rounded px-2 py-1">{chat.error}</div>
      )}
    </div>
  );
}

function Composer({ chat, engineReady }: { chat: ReturnType<typeof useLocalChat>; engineReady: boolean; }) {
  const [text, setText] = useState('');
  async function submit() {
    if (!text.trim() || chat.generating) return;
    const t = text;
    setText('');
    await chat.send(t);
  }
  return (
    <div className="border-t border-gray-700 px-3 py-2 flex items-end gap-2">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
        }}
        disabled={!engineReady}
        placeholder={engineReady ? 'Message — Enter to send, Shift-Enter for newline' : 'Load a checkpoint first…'}
        className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm text-gray-100 disabled:opacity-60 resize-none min-h-[40px] max-h-40"
        rows={1}
      />
      {chat.generating ? (
        <button onClick={chat.abort} className="text-sm px-3 py-2 rounded bg-amber-700 hover:bg-amber-600 text-amber-50">
          Stop
        </button>
      ) : (
        <button onClick={submit} disabled={!engineReady || !text.trim()} className="text-sm px-3 py-2 rounded bg-blue-700 hover:bg-blue-600 disabled:bg-gray-800 disabled:text-gray-500 text-gray-100">
          Send
        </button>
      )}
    </div>
  );
}
