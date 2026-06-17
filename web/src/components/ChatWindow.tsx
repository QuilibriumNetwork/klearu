import { useEffect, useRef } from 'react';
import type { ChatMessage } from '../protocol/types';
import MessageBubble from './MessageBubble';
import { stripThinkBlocks } from '../protocol/thinkFilter';

interface Props {
  messages: ChatMessage[];
  streamingText: string;
  generating: boolean;
  showThinking: boolean;
}

export default function ChatWindow({ messages, streamingText, generating, showThinking }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingText]);

  const displayText = showThinking ? streamingText : stripThinkBlocks(streamingText);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-2">
      {messages.length === 0 && !generating && (
        <div className="flex items-center justify-center h-full">
          <div className="max-w-lg text-gray-400 space-y-5">
            <div className="text-center">
              <h2 className="text-2xl font-semibold text-gray-100 mb-1">Klearu Private Chat</h2>
              <p className="text-sm">
                Send a message to start a private 2PC inference session.
                Output will be slow due to the high security model and
                interactive inspector.
              </p>
            </div>

            <div className="bg-gray-700/40 rounded-xl p-4 space-y-3 text-sm leading-relaxed">
              <p className="text-gray-200 font-medium">
                Messages are not saved on your browser, and cannot be read by
                the servers processing your request.
              </p>
              <p>
                Every message is split into cryptographic secret shares using
                two-party computation (2PC). Each server only ever sees random
                noise &mdash; the actual content can only be reconstructed by
                combining both shares, which only your browser holds. No
                plaintext ever leaves your device.
              </p>
            </div>

            <div className="space-y-2 text-sm">
              <p>
                <span className="text-gray-200 font-medium">Inspector</span>{' '}
                &mdash; Switch to the <span className="text-gray-200">Inspector</span> tab
                in the header to see the exact bytes sent to each server. Every
                DPF key and logit share is marked <span className="text-red-400 font-semibold text-xs uppercase tracking-wide">opaque</span> because
                it is computationally indistinguishable from random data.
              </p>
              <p>
                <span className="text-gray-200 font-medium">Thinking</span>{' '}
                &mdash; Toggle the <span className="text-gray-200">Thinking</span> button
                to reveal the model&rsquo;s chain-of-thought reasoning before
                its final answer.
              </p>
            </div>

            <div className="bg-gray-700/30 rounded-xl p-4 text-sm leading-relaxed border border-gray-600/30">
              <p className="text-gray-200 font-medium mb-2">
                Don&rsquo;t take our word for it &mdash; verify.
              </p>
              <p>
                Other services like Venice and FAI market themselves
                as &ldquo;private&rdquo; AI. Open your browser&rsquo;s network
                inspector while using them and look at the traffic: you&rsquo;ll
                see your plaintext prompts sent directly to their servers. With
                Klearu, every payload is a cryptographic share that reveals
                nothing on its own. Check for yourself.
              </p>
            </div>
          </div>
        </div>
      )}

      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} showThinking={showThinking} />
      ))}

      {generating && displayText && (
        <div className="flex justify-start mb-4">
          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-gray-700 text-gray-100">
            <p
              className="whitespace-pre-wrap break-words text-sm leading-relaxed"
              data-streaming
            >
              {displayText}
              <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5" />
            </p>
          </div>
        </div>
      )}

      {generating && !displayText && (
        <div className="flex justify-start mb-4">
          <div className="max-w-[80%] rounded-2xl px-4 py-3 bg-gray-700 text-gray-100">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}
