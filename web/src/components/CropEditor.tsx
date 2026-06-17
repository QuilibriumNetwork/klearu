// Interactive square crop editor. The user drags a crop box over the
// source image; on confirm the cropped region is resized to the target
// size and returned as PNG bytes. Runs entirely in-memory via canvas.

import { useCallback, useEffect, useRef, useState } from 'react';

export interface CropResult {
  bytes: Uint8Array;
  objectUrl: string;
}

interface Props {
  /** Object URL or data URI of the source image. */
  imageSrc: string;
  /** Output square side length (e.g. 512 or 1024). */
  targetSize: number;
  onConfirm: (result: CropResult) => void;
  onCancel: () => void;
}

export default function CropEditor({ imageSrc, targetSize, onConfirm, onCancel }: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [naturalW, setNaturalW] = useState(0);
  const [naturalH, setNaturalH] = useState(0);
  // Crop box in natural-image coordinates (square).
  const [cropX, setCropX] = useState(0);
  const [cropY, setCropY] = useState(0);
  const [cropSide, setCropSide] = useState(0);
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef<{ mx: number; my: number; ox: number; oy: number } | null>(null);

  const onImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    setNaturalW(w);
    setNaturalH(h);
    // Default: largest centered square.
    const side = Math.min(w, h);
    setCropX(Math.floor((w - side) / 2));
    setCropY(Math.floor((h - side) / 2));
    setCropSide(side);
  }, []);

  // Convert a pointer event to natural-image coordinates.
  const toNatural = useCallback((clientX: number, clientY: number) => {
    const img = imgRef.current;
    if (!img) return { x: 0, y: 0 };
    const rect = img.getBoundingClientRect();
    const scaleX = naturalW / rect.width;
    const scaleY = naturalH / rect.height;
    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY,
    };
  }, [naturalW, naturalH]);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    const p = toNatural(e.clientX, e.clientY);
    // Only start drag if pointer is inside the crop box.
    if (p.x >= cropX && p.x <= cropX + cropSide && p.y >= cropY && p.y <= cropY + cropSide) {
      setDragging(true);
      dragStart.current = { mx: p.x, my: p.y, ox: cropX, oy: cropY };
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
    }
  }, [toNatural, cropX, cropY, cropSide]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging || !dragStart.current) return;
    const p = toNatural(e.clientX, e.clientY);
    const dx = p.x - dragStart.current.mx;
    const dy = p.y - dragStart.current.my;
    const newX = Math.max(0, Math.min(naturalW - cropSide, dragStart.current.ox + dx));
    const newY = Math.max(0, Math.min(naturalH - cropSide, dragStart.current.oy + dy));
    setCropX(Math.round(newX));
    setCropY(Math.round(newY));
  }, [dragging, toNatural, naturalW, naturalH, cropSide]);

  const onPointerUp = useCallback(() => {
    setDragging(false);
    dragStart.current = null;
  }, []);

  // Resize the crop box via a slider (percentage of the shorter edge).
  const onSizeChange = useCallback((pct: number) => {
    const maxSide = Math.min(naturalW, naturalH);
    const newSide = Math.max(64, Math.round(maxSide * pct));
    // Re-center if the box would go out of bounds.
    const cx = cropX + cropSide / 2;
    const cy = cropY + cropSide / 2;
    const nx = Math.max(0, Math.min(naturalW - newSide, Math.round(cx - newSide / 2)));
    const ny = Math.max(0, Math.min(naturalH - newSide, Math.round(cy - newSide / 2)));
    setCropSide(newSide);
    setCropX(nx);
    setCropY(ny);
  }, [naturalW, naturalH, cropX, cropY, cropSide]);

  const [confirmErr, setConfirmErr] = useState<string | null>(null);

  async function confirm() {
    setConfirmErr(null);
    const img = imgRef.current;
    if (!img) { setConfirmErr('image element not ready'); return; }
    if (!img.complete || img.naturalWidth === 0) {
      setConfirmErr('image not fully loaded yet — wait a moment and retry');
      return;
    }
    try {
      const canvas = document.createElement('canvas');
      canvas.width = targetSize;
      canvas.height = targetSize;
      const ctx = canvas.getContext('2d');
      if (!ctx) { setConfirmErr('canvas 2d context unavailable'); return; }
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, cropX, cropY, cropSide, cropSide, 0, 0, targetSize, targetSize);
      // Verify the canvas isn't blank (can happen with CORS-tainted images,
      // though object URLs are same-origin and shouldn't trigger this).
      const probe = ctx.getImageData(0, 0, 1, 1);
      if (!probe) { setConfirmErr('canvas is tainted — cannot export'); return; }

      const blob = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error('canvas.toBlob returned null'))),
          'image/png',
        ),
      );
      if (blob.size < 100) {
        setConfirmErr(`PNG too small (${blob.size} bytes) — the crop may be empty`);
        return;
      }
      const bytes = new Uint8Array(await blob.arrayBuffer());
      // Quick PNG header check.
      if (bytes[0] !== 0x89 || bytes[1] !== 0x50) {
        setConfirmErr(`output is not a valid PNG (header: ${bytes[0]},${bytes[1]})`);
        return;
      }
      const objectUrl = URL.createObjectURL(blob);
      onConfirm({ bytes, objectUrl });
    } catch (e) {
      setConfirmErr(`crop failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Compute the crop overlay position in CSS % relative to the displayed image.
  const pctL = naturalW > 0 ? (cropX / naturalW) * 100 : 0;
  const pctT = naturalH > 0 ? (cropY / naturalH) * 100 : 0;
  const pctW = naturalW > 0 ? (cropSide / naturalW) * 100 : 100;
  const pctH = naturalH > 0 ? (cropSide / naturalH) * 100 : 100;
  const sizePct = naturalW > 0 ? cropSide / Math.min(naturalW, naturalH) : 1;

  return (
    <div className="flex flex-col items-center gap-3 w-full">
      <div className="text-xs text-gray-400">
        Drag the box to position the crop · slider to resize · then Confirm
      </div>

      {/* Image + crop overlay */}
      <div
        ref={containerRef}
        className="relative inline-block max-w-full max-h-[60vh] select-none touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        <img
          ref={imgRef}
          src={imageSrc}
          onLoad={onImageLoad}
          alt="source"
          className="block max-w-full max-h-[60vh]"
          draggable={false}
        />
        {/* Darkened area outside the crop */}
        <div className="absolute inset-0 pointer-events-none">
          {/* Top */}
          <div className="absolute bg-black/60" style={{ top: 0, left: 0, right: 0, height: `${pctT}%` }} />
          {/* Bottom */}
          <div className="absolute bg-black/60" style={{ bottom: 0, left: 0, right: 0, height: `${100 - pctT - pctH}%` }} />
          {/* Left */}
          <div className="absolute bg-black/60" style={{ top: `${pctT}%`, left: 0, width: `${pctL}%`, height: `${pctH}%` }} />
          {/* Right */}
          <div className="absolute bg-black/60" style={{ top: `${pctT}%`, right: 0, width: `${100 - pctL - pctW}%`, height: `${pctH}%` }} />
          {/* Crop border */}
          <div
            className="absolute border-2 border-blue-400 rounded-sm"
            style={{ top: `${pctT}%`, left: `${pctL}%`, width: `${pctW}%`, height: `${pctH}%` }}
          />
        </div>
      </div>

      {/* Size slider */}
      <div className="flex items-center gap-2 w-full max-w-md px-4">
        <span className="text-[10px] text-gray-500">crop</span>
        <input
          type="range"
          min={0.1}
          max={1}
          step={0.01}
          value={sizePct}
          onChange={(e) => onSizeChange(Number(e.target.value))}
          className="flex-1"
        />
        <span className="text-[10px] text-gray-400 tabular-nums w-16 text-right">
          {cropSide}×{cropSide}
        </span>
      </div>

      {/* Status + confirm */}
      {naturalW > 0 && (
        <div className="text-[10px] text-gray-500">
          source: {naturalW}×{naturalH} · crop: {cropSide}×{cropSide} → output: {targetSize}×{targetSize}
        </div>
      )}
      {confirmErr && (
        <div className="text-xs text-red-300 bg-red-900/30 rounded px-2 py-1">{confirmErr}</div>
      )}
      <div className="flex gap-3">
        <button
          onClick={onCancel}
          className="text-xs px-3 py-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-100"
        >
          Cancel
        </button>
        <button
          onClick={confirm}
          className="text-xs px-4 py-1.5 rounded bg-blue-700 hover:bg-blue-600 text-gray-100"
        >
          Confirm crop → {targetSize}×{targetSize}
        </button>
      </div>
    </div>
  );
}
