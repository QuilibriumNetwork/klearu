import { useState, useCallback, useRef } from 'react';
import type { ChatMessage } from '../protocol/types';
import { ConnectionStatus } from '../protocol/types';
import { BpeTokenizer, loadTokenizer } from '../tokenizer';
import {
  reconstructLogitsQ16,
  sampleGreedy,
} from '../protocol/sharing';
import {
  AesPrg,
  dpfGen,
  dpfDepthForVocab,
  serializeDpfKey,
} from '../protocol/dpf';
import {
  type TemplateType,
  loadChatTemplate,
  applyTemplate,
} from '../protocol/chatTemplate';
import type { ProtocolEvent } from '../protocol/inspect';
import { MAX_CAPTURE_BYTES } from '../protocol/inspect';

// Protocol tags
const TAG_SESSION_INIT = 0x01;
const TAG_SESSION_READY = 0x02;
const TAG_JOIN_SESSION = 0x03;
const TAG_JOIN_ACK = 0x04;
const TAG_LOGIT_SHARE = 0x51;
const TAG_NEXT_TOKEN = 0x52;
const TAG_EOS_SIGNAL = 0x42;
const TAG_ERROR = 0xf0;

interface UsePrivateChatOptions {
  wsUrl: string;
  showThinking: boolean;
}

interface SessionConfig {
  vocab_size: number;
  hidden_size: number;
  eos_token_ids: number[];
  session_id: string;
  triple_seed: string;
  server_b_ws: string;
  prg_key: string;
}

// Cached resources (persist across hook instances)
let cachedTokenizer: BpeTokenizer | null = null;
let cachedTemplate: TemplateType | null = null;

export function usePrivateChat({ wsUrl, showThinking }: UsePrivateChatOptions) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [generating, setGenerating] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [tokensPerSec, setTokensPerSec] = useState<number | null>(null);
  const [connectionStatus, setConnectionStatus] =
    useState(ConnectionStatus.Disconnected);
  const [protocolEvents, setProtocolEvents] = useState<ProtocolEvent[]>([]);

  const historyRef = useRef<ChatMessage[]>([]);
  const wsARef = useRef<WebSocket | null>(null);
  const wsBRef = useRef<WebSocket | null>(null);
  const eventsRef = useRef<ProtocolEvent[]>([]);
  const eventIdRef = useRef(0);

  const sendMessage = useCallback(
    async (content: string) => {
      if (generating) return;

      const userMsg: ChatMessage = { role: 'user', content };
      historyRef.current = [...historyRef.current, userMsg];
      setMessages(historyRef.current);
      setGenerating(true);
      setStreamingText('');
      setTokensPerSec(null);

      // Reset protocol events for this session
      eventsRef.current = [];
      eventIdRef.current = 0;
      setProtocolEvents([]);

      // Event capture helpers
      function addEvent(
        direction: 'sent' | 'received',
        server: 'A' | 'B',
        data: Uint8Array
      ) {
        const event: ProtocolEvent = {
          id: eventIdRef.current++,
          timestamp: performance.now(),
          direction,
          server,
          tag: data[0],
          rawBytes: data.slice(
            0,
            Math.min(data.length, MAX_CAPTURE_BYTES)
          ),
          totalBytes: data.length,
        };
        eventsRef.current.push(event);
      }

      function flushEvents() {
        setProtocolEvents([...eventsRef.current]);
      }

      const startTime = performance.now();
      let tokenCount = 0;
      let accumulatedText = '';

      try {
        // 1. Load tokenizer + chat template (cached)
        if (!cachedTokenizer) {
          setConnectionStatus(ConnectionStatus.LoadingTokenizer);
          const baseUrl = wsUrl.replace(/^ws/, 'http').replace(/\/ws$/, '');
          const [tok, tmpl] = await Promise.all([
            loadTokenizer(`${baseUrl}/api/models/tokenizer.json`),
            loadChatTemplate(baseUrl),
          ]);
          cachedTokenizer = tok;
          cachedTemplate = tmpl;
        }
        const tokenizer = cachedTokenizer;
        const template = cachedTemplate!;

        // 2. Connect to Server A
        setConnectionStatus(ConnectionStatus.Connecting);
        const wsA = new WebSocket(wsUrl);
        wsA.binaryType = 'arraybuffer';
        wsARef.current = wsA;

        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            wsA.close();
            reject(new Error('Server A connection timeout'));
          }, 10000);
          wsA.onopen = () => {
            clearTimeout(timeout);
            resolve();
          };
          wsA.onerror = () => {
            clearTimeout(timeout);
            reject(new Error('Server A connection failed'));
          };
        });

        setConnectionStatus(ConnectionStatus.Connected);

        // 3. Format conversation with chat template and tokenize
        let formattedPrompt = applyTemplate(template, historyRef.current);
        // When thinking is disabled, prefill empty think block so the model skips reasoning
        if (!showThinking) {
          formattedPrompt += '<think>\n</think>\n';
        }
        const promptIds = tokenizer.encode(formattedPrompt);

        // 4. Send SessionInit
        const initBuf = new ArrayBuffer(22);
        const initView = new DataView(initBuf);
        initView.setUint8(0, TAG_SESSION_INIT);
        initView.setUint8(1, 1); // security = High
        initView.setUint32(2, 2048, true); // max_new_tokens
        initView.setFloat32(6, 0.0, true); // temperature (greedy)
        initView.setUint32(10, 0, true); // top_k
        initView.setFloat32(14, 1.0, true); // top_p
        initView.setUint32(18, promptIds.length, true); // num_prompt_tokens

        wsA.send(initBuf);
        addEvent('sent', 'A', new Uint8Array(initBuf));

        // 5. Wait for SessionReady
        const sessionConfig = await new Promise<SessionConfig>(
          (resolve, reject) => {
            const timeout = setTimeout(() => {
              reject(new Error('SessionReady timeout'));
            }, 30000);

            wsA.onmessage = (event) => {
              clearTimeout(timeout);
              const data = new Uint8Array(event.data as ArrayBuffer);
              addEvent('received', 'A', data);
              if (data[0] === TAG_SESSION_READY) {
                const jsonStr = new TextDecoder().decode(data.slice(1));
                resolve(JSON.parse(jsonStr) as SessionConfig);
              } else if (data[0] === TAG_ERROR) {
                const errMsg = new TextDecoder().decode(data.slice(1));
                reject(new Error(`Server A error: ${errMsg}`));
              } else {
                reject(
                  new Error(`Unexpected tag: 0x${data[0].toString(16)}`)
                );
              }
            };
            wsA.onerror = () => {
              clearTimeout(timeout);
              reject(new Error('Server A error'));
            };
          }
        );

        const {
          vocab_size: vocabSize,
          eos_token_ids: eosTokenIds,
          session_id: sessionId,
          server_b_ws: serverBWs,
          prg_key: prgKeyHex,
        } = sessionConfig;

        // 6. Initialize DPF PRG from prg_key
        setConnectionStatus(ConnectionStatus.InitDPF);
        const prgKeyBytes = new Uint8Array(16);
        for (let i = 0; i < 16; i++) {
          prgKeyBytes[i] = parseInt(prgKeyHex.slice(i * 2, i * 2 + 2), 16);
        }
        const prg = await AesPrg.create(prgKeyBytes);
        const depth = dpfDepthForVocab(vocabSize);

        // 7. Connect to Server B
        const wsB = new WebSocket(serverBWs);
        wsB.binaryType = 'arraybuffer';
        wsBRef.current = wsB;

        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            wsB.close();
            reject(new Error('Server B connection timeout'));
          }, 10000);
          wsB.onopen = () => {
            clearTimeout(timeout);
            resolve();
          };
          wsB.onerror = () => {
            clearTimeout(timeout);
            reject(new Error('Server B connection failed'));
          };
        });

        // 8. Send JoinSession to Server B
        const joinBuf = new ArrayBuffer(9);
        const joinView = new DataView(joinBuf);
        joinView.setUint8(0, TAG_JOIN_SESSION);
        // Parse session_id from hex string
        const sidHigh = parseInt(sessionId.slice(0, 8), 16);
        const sidLow = parseInt(sessionId.slice(8, 16), 16);
        joinView.setUint32(1, sidLow, true); // LE: low bytes first
        joinView.setUint32(5, sidHigh, true);
        wsB.send(joinBuf);
        addEvent('sent', 'B', new Uint8Array(joinBuf));

        // 9. Wait for JoinAck from Server B
        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => {
            reject(new Error('JoinAck timeout'));
          }, 10000);
          wsB.onmessage = (event) => {
            clearTimeout(timeout);
            const data = new Uint8Array(event.data as ArrayBuffer);
            addEvent('received', 'B', data);
            if (data[0] === TAG_JOIN_ACK) {
              resolve();
            } else if (data[0] === TAG_ERROR) {
              const errMsg = new TextDecoder().decode(data.slice(1));
              reject(new Error(`Server B error: ${errMsg}`));
            } else {
              reject(
                new Error(
                  `Expected JoinAck, got 0x${data[0].toString(16)}`
                )
              );
            }
          };
          wsB.onerror = () => {
            clearTimeout(timeout);
            reject(new Error('Server B error'));
          };
        });

        flushEvents();

        // Helper: wait for a logit share from a WebSocket
        function waitForLogitShare(
          ws: WebSocket,
          server: 'A' | 'B'
        ): Promise<Uint8Array> {
          return new Promise((resolve, reject) => {
            ws.onmessage = (event) => {
              const data = new Uint8Array(event.data as ArrayBuffer);
              addEvent('received', server, data);
              if (data[0] === TAG_LOGIT_SHARE) {
                resolve(data.slice(1));
              } else if (data[0] === TAG_ERROR) {
                const errMsg = new TextDecoder().decode(data.slice(1));
                reject(new Error(errMsg));
              } else {
                reject(
                  new Error(
                    `Expected LogitShare, got 0x${data[0].toString(16)}`
                  )
                );
              }
            };
            ws.onerror = () => reject(new Error('WS error'));
          });
        }

        // Helper: generate and send DPF keys for a token
        async function sendDpfKeys(tokenId: number) {
          const [key0, key1] = await dpfGen(prg, tokenId, 1, depth);
          const bytes0 = serializeDpfKey(key0);
          const bytes1 = serializeDpfKey(key1);

          // Send [TAG_NEXT_TOKEN, ...key0] to Server A
          const msgA = new Uint8Array(1 + bytes0.length);
          msgA[0] = TAG_NEXT_TOKEN;
          msgA.set(bytes0, 1);
          wsA.send(msgA.buffer);
          addEvent('sent', 'A', msgA);

          // Send [TAG_NEXT_TOKEN, ...key1] to Server B
          const msgB = new Uint8Array(1 + bytes1.length);
          msgB[0] = TAG_NEXT_TOKEN;
          msgB.set(bytes1, 1);
          wsB.send(msgB.buffer);
          addEvent('sent', 'B', msgB);
        }

        // 10. PREFILL — generate all DPF keys in parallel, then send in order
        setConnectionStatus(ConnectionStatus.Prefilling);
        let lastSampledToken = 0;

        // Generate all key pairs concurrently (each token is independent)
        const allKeyPairs = await Promise.all(
          promptIds.map((tokenId) => dpfGen(prg, tokenId, 1, depth))
        );

        // Send keys in order (WebSocket preserves message ordering)
        for (const [key0, key1] of allKeyPairs) {
          const bytes0 = serializeDpfKey(key0);
          const bytes1 = serializeDpfKey(key1);

          const msgA = new Uint8Array(1 + bytes0.length);
          msgA[0] = TAG_NEXT_TOKEN;
          msgA.set(bytes0, 1);
          wsA.send(msgA.buffer);
          addEvent('sent', 'A', msgA);

          const msgB = new Uint8Array(1 + bytes1.length);
          msgB[0] = TAG_NEXT_TOKEN;
          msgB.set(bytes1, 1);
          wsB.send(msgB.buffer);
          addEvent('sent', 'B', msgB);
        }

        // Wait for logit shares from both servers (only sent after last prefill token)
        if (promptIds.length > 0) {
          const [logitShareA, logitShareB] = await Promise.all([
            waitForLogitShare(wsA, 'A'),
            waitForLogitShare(wsB, 'B'),
          ]);
          const logits = reconstructLogitsQ16(
            logitShareA,
            logitShareB,
            vocabSize
          );
          lastSampledToken = sampleGreedy(logits);
        }

        flushEvents();

        // 11. DECODE LOOP
        setConnectionStatus(ConnectionStatus.Generating);
        const maxNewTokens = 2048;
        for (let step = 0; step < maxNewTokens; step++) {
          // Check for EOS
          if (eosTokenIds.includes(lastSampledToken)) {
            break;
          }

          // Decode the token
          const text = tokenizer.decode([lastSampledToken]);
          tokenCount++;
          accumulatedText += text;
          setStreamingText(accumulatedText);

          const elapsed = (performance.now() - startTime) / 1000;
          if (elapsed > 0.1) {
            setTokensPerSec(tokenCount / elapsed);
          }

          // Send DPF keys for sampled token
          await sendDpfKeys(lastSampledToken);

          const [logitShareA, logitShareB] = await Promise.all([
            waitForLogitShare(wsA, 'A'),
            waitForLogitShare(wsB, 'B'),
          ]);
          const logits = reconstructLogitsQ16(
            logitShareA,
            logitShareB,
            vocabSize
          );
          lastSampledToken = sampleGreedy(logits);

          flushEvents();
        }

        // Send EOS to both servers
        const eosA = new Uint8Array([TAG_EOS_SIGNAL]);
        wsA.send(eosA.buffer);
        addEvent('sent', 'A', eosA);

        const eosB = new Uint8Array([TAG_EOS_SIGNAL]);
        wsB.send(eosB.buffer);
        addEvent('sent', 'B', eosB);
      } catch (e) {
        console.error('Chat error:', e);
        if (!accumulatedText) {
          accumulatedText = `[Error: ${e instanceof Error ? e.message : 'Unknown error'}]`;
        }
      } finally {
        wsARef.current?.close();
        wsARef.current = null;
        wsBRef.current?.close();
        wsBRef.current = null;

        const assistantMsg: ChatMessage = {
          role: 'assistant',
          content: accumulatedText,
        };
        historyRef.current = [...historyRef.current, assistantMsg];
        setMessages(historyRef.current);

        const elapsed = (performance.now() - startTime) / 1000;
        if (tokenCount > 0) {
          setTokensPerSec(tokenCount / elapsed);
        }

        setStreamingText('');
        setGenerating(false);
        setConnectionStatus(ConnectionStatus.Disconnected);
        flushEvents();
      }
    },
    [generating, wsUrl, showThinking]
  );

  return {
    messages,
    streamingText,
    generating,
    modelConfig: null,
    tokensPerSec,
    connectionStatus,
    sendMessage,
    protocolEvents,
  };
}
