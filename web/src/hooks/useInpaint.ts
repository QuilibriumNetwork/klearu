// Drives a single in-flight inpaint job. Same channel-event plumbing as
// useImageGen but talks to `image_inpaint`, which writes the input + mask
// bytes to disk before invoking sd_generate.

import { useCallback, useState } from 'react';
import {
  imageAbort,
  imageInpaint,
  type GenEvent,
  type InpaintRequest,
} from '../lib/tauri';

export interface InpaintProgress {
  jobId: number;
  step: number;
  total: number;
  elapsedS: number;
  outPath?: string;
}

export interface UseInpaint {
  generating: boolean;
  progress: InpaintProgress | null;
  lastError: string | null;
  recentLogs: string[];
  resultPath: string | null;
  generate: (req: InpaintRequest) => Promise<void>;
  abort: () => Promise<void>;
}

const MAX_LOGS = 60;

export function useInpaint(): UseInpaint {
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<InpaintProgress | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);
  const [recentLogs, setRecentLogs] = useState<string[]>([]);
  const [resultPath, setResultPath] = useState<string | null>(null);

  const generate = useCallback(async (req: InpaintRequest) => {
    if (generating) return;
    setLastError(null);
    setProgress(null);
    setRecentLogs([]);
    setResultPath(null);
    setGenerating(true);

    let outPath: string | undefined;
    try {
      await imageInpaint(req, (e: GenEvent) => {
        if (e.type === 'started') {
          outPath = e.out_path;
          setProgress({ jobId: e.job_id, step: 0, total: 0, elapsedS: 0, outPath });
        } else if (e.type === 'step') {
          setProgress({
            jobId: e.job_id,
            step: e.step,
            total: e.total,
            elapsedS: e.elapsed_s,
            outPath,
          });
        } else if (e.type === 'log') {
          setRecentLogs((prev) => {
            const next = [...prev, e.line];
            return next.length > MAX_LOGS ? next.slice(-MAX_LOGS) : next;
          });
        } else if (e.type === 'done') {
          outPath = e.path;
          setResultPath(e.path);
          setProgress((p) => p ? { ...p, outPath: e.path } : p);
        } else if (e.type === 'aborted') {
          setLastError('aborted');
        } else if (e.type === 'error') {
          setLastError(`${e.code}: ${e.message}`);
        }
      });
    } catch (e) {
      const msg = (e && typeof e === 'object' && 'message' in e)
        ? (e as { message: string }).message
        : String(e);
      setLastError(msg);
    } finally {
      setGenerating(false);
    }
  }, [generating]);

  const abort = useCallback(async () => {
    try { await imageAbort(); } catch (e) { console.warn('abort failed', e); }
  }, []);

  return { generating, progress, lastError, recentLogs, resultPath, generate, abort };
}
