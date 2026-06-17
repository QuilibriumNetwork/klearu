// Tiny module-level store for handing a generated image off to the Inpaint
// panel as its source. The Image panel stashes PNG bytes here and switches
// to the Inpaint tab; InpaintPanel consumes (and clears) them on mount.

let pendingBytes: Uint8Array | null = null;
const listeners = new Set<() => void>();

/** Stash PNG bytes to be picked up by the Inpaint panel. */
export function setInpaintSource(bytes: Uint8Array): void {
  pendingBytes = bytes;
  for (const l of listeners) l();
}

/** Take the pending bytes (clearing them). Returns null if none pending. */
export function takeInpaintSource(): Uint8Array | null {
  const b = pendingBytes;
  pendingBytes = null;
  return b;
}

/** Subscribe to pending-source changes; returns an unsubscribe fn. */
export function onInpaintSource(fn: () => void): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Fetch the bytes behind an `imageSrc()` URL (works in tauri + remote). */
export async function fetchImageBytes(src: string): Promise<Uint8Array> {
  const resp = await fetch(src);
  const buf = await resp.arrayBuffer();
  return new Uint8Array(buf);
}
