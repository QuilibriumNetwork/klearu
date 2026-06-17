// Hook driving the local in-process LLM chat. Mirrors the shape of
// usePrivateChat (messages, streamingText, generating, send) so the
// existing UI components plug in unchanged.

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  chatAbort,
  chatSend,
  conversationCreate,
  conversationMessages,
  conversationsList,
  type Conversation,
  type DbMessage,
  type StreamEvent,
} from '../lib/tauri';

export interface LocalChatMessage {
  id?: number;
  role: 'user' | 'assistant';
  content: string;
}

export interface PendingTool { name: string; iter: number; }

export interface UseLocalChat {
  conversation: Conversation | null;
  conversations: Conversation[];
  messages: LocalChatMessage[];
  streamingText: string;
  generating: boolean;
  pendingTools: PendingTool[];
  error: string | null;
  send: (content: string) => Promise<void>;
  abort: () => Promise<void>;
  newConversation: (title?: string) => Promise<void>;
  selectConversation: (id: number) => Promise<void>;
  refreshConversations: () => Promise<void>;
}

export function useLocalChat(): UseLocalChat {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<LocalChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [generating, setGenerating] = useState(false);
  const [pendingTools, setPendingTools] = useState<PendingTool[]>([]);
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef('');

  const refreshConversations = useCallback(async () => {
    try {
      const list = await conversationsList();
      setConversations(list);
      // Seed: if no conversation is open and at least one exists, pick the latest.
      setConversation((cur) => cur ?? list[0] ?? null);
    } catch (e) { setError(String(e)); }
  }, []);

  useEffect(() => { refreshConversations(); }, [refreshConversations]);

  // Load messages whenever the active conversation changes.
  useEffect(() => {
    if (!conversation) { setMessages([]); return; }
    (async () => {
      try {
        const list = await conversationMessages(conversation.id);
        setMessages(list.filter(m => m.role !== 'system').map(dbToLocal));
      } catch (e) { setError(String(e)); }
    })();
  }, [conversation?.id]);

  const newConversation = useCallback(async (title?: string) => {
    const c = await conversationCreate(title);
    setConversation(c);
    setMessages([]);
    setStreamingText('');
    streamRef.current = '';
    await refreshConversations();
  }, [refreshConversations]);

  const selectConversation = useCallback(async (id: number) => {
    const c = conversations.find((x) => x.id === id);
    if (c) setConversation(c);
  }, [conversations]);

  const send = useCallback(async (content: string) => {
    if (!content.trim() || generating) return;
    setError(null);
    let conv = conversation;
    if (!conv) {
      conv = await conversationCreate('New chat');
      setConversation(conv);
      await refreshConversations();
    }
    // Optimistic user message
    setMessages((m) => [...m, { role: 'user', content }]);
    setStreamingText('');
    streamRef.current = '';
    setGenerating(true);

    try {
      await chatSend(
        { conversation_id: conv.id, content },
        (e: StreamEvent) => {
          if (e.type === 'token') {
            streamRef.current += e.text;
            setStreamingText(streamRef.current);
          } else if (e.type === 'tool_call') {
            setPendingTools((p) => [...p, { name: e.name, iter: e.iter }]);
          } else if (e.type === 'tool_result') {
            setPendingTools((p) =>
              p.filter((x) => !(x.name === e.name && x.iter === e.iter)),
            );
            // Append the result block to the streaming buffer so subsequent
            // iteration tokens flow visually after the result (matches the
            // backend's persisted content exactly).
            const block = `\n<tool_result name="${e.name}">\n${JSON.stringify(e.result)}\n</tool_result>\n`;
            streamRef.current += block;
            setStreamingText(streamRef.current);
          } else if (e.type === 'done') {
            setMessages((m) => [...m, { id: e.assistant_message_id, role: 'assistant', content: e.total_text }]);
            setStreamingText('');
            streamRef.current = '';
            setPendingTools([]);
          } else if (e.type === 'aborted') {
            setMessages((m) => [...m, { id: e.assistant_message_id, role: 'assistant', content: e.partial }]);
            setStreamingText('');
            streamRef.current = '';
            setPendingTools([]);
          } else if (e.type === 'error') {
            setError(`${e.code}: ${e.message}`);
            setStreamingText('');
            streamRef.current = '';
            setPendingTools([]);
          }
          // iter_end is informational — no state change needed; the per-tool
          // tool_call/tool_result events drive the UI.
        },
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setGenerating(false);
    }
  }, [conversation, generating, refreshConversations]);

  const abort = useCallback(async () => {
    try { await chatAbort(); } catch (e) { setError(String(e)); }
  }, []);

  return {
    conversation, conversations, messages, streamingText, generating, pendingTools, error,
    send, abort, newConversation, selectConversation, refreshConversations,
  };
}

function dbToLocal(m: DbMessage): LocalChatMessage {
  return {
    id: m.id,
    role: m.role === 'user' ? 'user' : 'assistant',
    content: m.content,
  };
}
