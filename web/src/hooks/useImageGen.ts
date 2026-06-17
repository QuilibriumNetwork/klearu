// Drives a single in-flight image generation. Holds form state, progress,
// and a recent-images gallery refreshed when a job completes.

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  imageAbort,
  imageGenerate,
  imageList,
  videoGenerate,
  type GenEvent,
  type GenerateRequest,
  type StoredImage,
  type VideoGenerateRequest,
} from '../lib/tauri';

export interface GenProgress {
  jobId: number;
  step: number;
  total: number;
  elapsedS: number;
  outPath?: string;
}

export interface UseImageGen {
  generating: boolean;
  progress: GenProgress | null;
  lastError: string | null;
  recentLogs: string[];
  gallery: StoredImage[];
  refreshGallery: () => Promise<void>;
  /** Resolves to `false` when the job was aborted (so a batch loop can break). */
  generate: (req: GenerateRequest) => Promise<boolean>;
  generateVideo: (req: VideoGenerateRequest) => Promise<void>;
  abort: () => Promise<void>;
  /** Clear the aborted flag — call once before starting a fresh batch. */
  resetAbort: () => void;
}

const MAX_LOGS = 60;

export function useImageGen(): UseImageGen {
  const [generating, setGenerating] = useState(false);
  const [progress, setProgress] = useState<GenProgress | null>(() => {
    // Restore last progress snapshot from sessionStorage (survives refresh).
    try {
      const s = sessionStorage.getItem('img_progress');
      return s ? JSON.parse(s) : null;
    } catch { return null; }
  });
  const [lastError, setLastError] = useState<string | null>(null);
  const [recentLogs, setRecentLogs] = useState<string[]>([]);
  const [gallery, setGallery] = useState<StoredImage[]>([]);
  // Set true by abort(), cleared at the start of each generate(). The batch
  // loop checks generate()'s return value to break out when the user aborts
  // mid-batch (the backend resets its abort flag per-job, so we track it here).
  const abortedRef = useRef(false);

  const refreshGallery = useCallback(async () => {
    try { setGallery(await imageList(60)); }
    catch (e) { console.warn('gallery refresh failed', e); }
  }, []);

  useEffect(() => { refreshGallery(); }, [refreshGallery]);

  const generate = useCallback(async (req: GenerateRequest): Promise<boolean> => {
    if (generating) return true;
    if (abortedRef.current) return false; // a prior batch image was aborted
    setLastError(null);
    setProgress(null);
    setRecentLogs([]);
    setGenerating(true);

    let outPath: string | undefined;
    try {
      await imageGenerate(req, (e: GenEvent) => {
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
          const updated = { jobId: e.job_id, step: 0, total: 0, elapsedS: 0, outPath: e.path };
          setProgress(updated);
          try { sessionStorage.setItem('img_progress', JSON.stringify(updated)); } catch {}
        } else if (e.type === 'aborted') {
          setLastError('aborted');
          abortedRef.current = true;
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
      await refreshGallery();
    }
    return !abortedRef.current;
  }, [generating, refreshGallery]);

  const generateVideo = useCallback(async (req: VideoGenerateRequest) => {
    if (generating) return;
    setLastError(null);
    setProgress(null);
    setRecentLogs([]);
    setGenerating(true);

    let outPath: string | undefined;
    try {
      await videoGenerate(req, (e: GenEvent) => {
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
          const updated = { jobId: e.job_id, step: 0, total: 0, elapsedS: 0, outPath: e.path };
          setProgress(updated);
          try { sessionStorage.setItem('img_progress', JSON.stringify(updated)); } catch {}
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
      await refreshGallery();
    }
  }, [generating, refreshGallery]);

  const abort = useCallback(async () => {
    abortedRef.current = true; // break any in-flight batch loop
    try { await imageAbort(); } catch (e) { console.warn('abort failed', e); }
  }, []);

  const resetAbort = useCallback(() => { abortedRef.current = false; }, []);

  return { generating, progress, lastError, recentLogs, gallery, refreshGallery, generate, generateVideo, abort, resetAbort };
}
