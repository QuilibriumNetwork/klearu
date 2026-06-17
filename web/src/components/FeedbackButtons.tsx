// Tiny 👍 / 👎 chip used on assistant bubbles and image gallery thumbs.
// Optimistically toggles state, hits feedback_set, then settles to the
// returned row. Re-click of an active rating clears it.

import { useEffect, useState } from 'react';
import { feedbackGet, feedbackSet, type FeedbackKind } from '../lib/tauri';

export interface FeedbackButtonsProps {
  kind: FeedbackKind;
  targetId: number;
  /** Called when a rating change triggers a user-vector Welford update. */
  onUserVectorChanged?: () => void;
  /** Layout: 'row' (horizontal pair) or 'stack' (vertical). */
  layout?: 'row' | 'stack';
  /** Optional pixel size for the buttons. Default 14 fits chat bubbles. */
  size?: number;
}

export default function FeedbackButtons({
  kind, targetId, onUserVectorChanged, layout = 'row', size = 14,
}: FeedbackButtonsProps) {
  const [rating, setRating] = useState<number>(0);
  const [pending, setPending] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    feedbackGet(kind, targetId)
      .then((r) => setRating(r?.rating ?? 0))
      .catch(() => {});
  }, [kind, targetId]);

  async function apply(next: number) {
    if (pending) return;
    setErr(null);
    const newRating = next === rating ? 0 : next;
    const prev = rating;
    setRating(newRating);              // optimistic
    setPending(true);
    try {
      const res = await feedbackSet(kind, targetId, newRating);
      setRating(res.feedback.rating);
      if (res.user_vector) onUserVectorChanged?.();
    } catch (e) {
      setRating(prev);                 // rollback
      const msg = (e && typeof e === 'object' && 'message' in e)
        ? (e as { message: string }).message
        : String(e);
      setErr(msg);
    } finally {
      setPending(false);
    }
  }

  const cls = (active: boolean, color: 'green' | 'red') => {
    const base = 'flex items-center justify-center rounded transition-colors';
    if (active) {
      return color === 'green'
        ? `${base} bg-emerald-700/70 text-emerald-50`
        : `${base} bg-red-800/70 text-red-50`;
    }
    return `${base} bg-gray-800/50 text-gray-500 hover:bg-gray-700 hover:text-gray-200`;
  };

  const wrap = layout === 'row' ? 'flex gap-1' : 'flex flex-col gap-1';
  const box = { width: size + 8, height: size + 8, fontSize: size };
  return (
    <div className={wrap}>
      <button
        type="button"
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); apply(1); }}
        disabled={pending}
        className={cls(rating > 0, 'green')}
        style={box}
        title={err ?? 'Mark useful'}
      >
        ▲
      </button>
      <button
        type="button"
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); apply(-1); }}
        disabled={pending}
        className={cls(rating < 0, 'red')}
        style={box}
        title={err ?? 'Mark not useful'}
      >
        ▼
      </button>
    </div>
  );
}
