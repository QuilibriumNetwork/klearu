import { useState, useCallback, type KeyboardEvent } from 'react';

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
}

export default function MessageInput({ onSend, disabled }: Props) {
  const [text, setText] = useState('');

  const handleSend = useCallback(() => {
    const trimmed = text.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setText('');
    }
  }, [text, disabled, onSend]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="border-t border-gray-700 p-4">
      <div className="flex gap-3 max-w-3xl mx-auto">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none rounded-xl bg-gray-700 text-gray-100 px-4 py-3 text-sm
                     placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500
                     disabled:opacity-50"
        />
        <button
          onClick={handleSend}
          disabled={disabled || !text.trim()}
          className="rounded-xl bg-blue-600 px-5 py-3 text-sm font-medium text-white
                     hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  );
}
