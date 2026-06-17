import type { ChatMessage } from '../protocol/types';
import { stripThinkBlocks } from '../protocol/thinkFilter';

interface Props {
  message: ChatMessage;
  showThinking: boolean;
}

export default function MessageBubble({ message, showThinking }: Props) {
  const isUser = message.role === 'user';
  const content =
    isUser || showThinking
      ? message.content
      : stripThinkBlocks(message.content);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-gray-700 text-gray-100'
        }`}
      >
        <p className="whitespace-pre-wrap break-words text-sm leading-relaxed">
          {content}
        </p>
      </div>
    </div>
  );
}
