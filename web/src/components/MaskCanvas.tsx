// Paint-on-image mask editor used by the inpaint panel.
//
// Internally tracks TWO canvases at the image's natural pixel dimensions:
//   * `displayRef` — semi-transparent white overlay, positioned over the
//     source image so the user sees what they're masking.
//   * `maskRef` — black background + opaque white strokes. This is the
//     PNG mask that ships to sd_generate (white = regenerate, black = keep).
//
// Drawing happens to both canvases on every stroke. CSS scales the canvases
// to fit the panel; pointer coordinates are unscaled back to native pixels
// so brush size and stroke geometry are consistent across zoom levels.

import { useCallback, useEffect, useImperativeHandle, useRef, useState, forwardRef } from 'react';

export type MaskTool = 'brush' | 'eraser';

export interface MaskCanvasHandle {
  /** Return the current mask as a PNG blob (grayscale). null if empty. */
  exportMask(): Promise<Blob | null>;
  clear(): void;
}

export interface MaskCanvasProps {
  /** URL or asset:// link to the source image. */
  imageSrc: string;
  brushSize: number;
  tool: MaskTool;
  /** Called whenever a stroke ends so the parent knows the mask changed. */
  onChange?: () => void;
}

const MaskCanvas = forwardRef<MaskCanvasHandle, MaskCanvasProps>(function MaskCanvas(
  { imageSrc, brushSize, tool, onChange },
  ref,
) {
  const wrapperRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const displayRef = useRef<HTMLCanvasElement>(null);
  const maskRef = useRef<HTMLCanvasElement>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const dirtyRef = useRef(false);

  // Set up canvases when the image loads or src changes.
  const onImageLoad = useCallback(() => {
    const img = imgRef.current;
    if (!img) return;
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    setNaturalSize({ w, h });

    const display = displayRef.current!;
    const mask = maskRef.current!;
    display.width = w;
    display.height = h;
    mask.width = w;
    mask.height = h;

    const dctx = display.getContext('2d')!;
    dctx.clearRect(0, 0, w, h);
    const mctx = mask.getContext('2d')!;
    mctx.fillStyle = 'black';
    mctx.fillRect(0, 0, w, h);
    dirtyRef.current = false;
  }, []);

  const toNative = (clientX: number, clientY: number) => {
    const c = displayRef.current!;
    const rect = c.getBoundingClientRect();
    const x = ((clientX - rect.left) / rect.width) * c.width;
    const y = ((clientY - rect.top) / rect.height) * c.height;
    return { x, y };
  };

  const drawSegment = (from: { x: number; y: number }, to: { x: number; y: number }) => {
    const dctx = displayRef.current!.getContext('2d')!;
    const mctx = maskRef.current!.getContext('2d')!;

    const isBrush = tool === 'brush';

    // Display canvas: semi-transparent white overlay (brush) or punch-out (eraser).
    dctx.lineCap = 'round';
    dctx.lineJoin = 'round';
    dctx.lineWidth = brushSize;
    if (isBrush) {
      dctx.globalCompositeOperation = 'source-over';
      dctx.strokeStyle = 'rgba(56, 189, 248, 0.55)'; // sky-400 so it pops against most photos
    } else {
      dctx.globalCompositeOperation = 'destination-out';
      dctx.strokeStyle = 'rgba(0, 0, 0, 1)';
    }
    dctx.beginPath();
    dctx.moveTo(from.x, from.y);
    dctx.lineTo(to.x, to.y);
    dctx.stroke();
    dctx.globalCompositeOperation = 'source-over';

    // Mask canvas: opaque white (brush) or opaque black (eraser).
    mctx.lineCap = 'round';
    mctx.lineJoin = 'round';
    mctx.lineWidth = brushSize;
    mctx.strokeStyle = isBrush ? 'white' : 'black';
    mctx.beginPath();
    mctx.moveTo(from.x, from.y);
    mctx.lineTo(to.x, to.y);
    mctx.stroke();
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!naturalSize) return;
    (e.currentTarget as HTMLCanvasElement).setPointerCapture(e.pointerId);
    drawingRef.current = true;
    const p = toNative(e.clientX, e.clientY);
    lastPointRef.current = p;
    drawSegment(p, p);
    dirtyRef.current = true;
  };
  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const p = toNative(e.clientX, e.clientY);
    if (lastPointRef.current) drawSegment(lastPointRef.current, p);
    lastPointRef.current = p;
  };
  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    (e.currentTarget as HTMLCanvasElement).releasePointerCapture(e.pointerId);
    drawingRef.current = false;
    lastPointRef.current = null;
    onChange?.();
  };

  useImperativeHandle(ref, () => ({
    async exportMask(): Promise<Blob | null> {
      if (!dirtyRef.current) return null;
      const c = maskRef.current!;
      return await new Promise<Blob | null>((resolve) => c.toBlob((b) => resolve(b), 'image/png'));
    },
    clear() {
      const display = displayRef.current!;
      const mask = maskRef.current!;
      const dctx = display.getContext('2d')!;
      dctx.clearRect(0, 0, display.width, display.height);
      const mctx = mask.getContext('2d')!;
      mctx.fillStyle = 'black';
      mctx.fillRect(0, 0, mask.width, mask.height);
      dirtyRef.current = false;
      onChange?.();
    },
  }), [onChange]);

  return (
    <div ref={wrapperRef} className="relative inline-block max-w-full max-h-[70vh]">
      <img
        ref={imgRef}
        src={imageSrc}
        onLoad={onImageLoad}
        alt="source"
        className="block max-w-full max-h-[70vh] select-none"
        draggable={false}
      />
      {/* Display overlay: visible to the user. */}
      <canvas
        ref={displayRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        className="absolute inset-0 w-full h-full cursor-crosshair touch-none"
        style={{ touchAction: 'none' }}
      />
      {/* Hidden mask canvas: ships to sd_generate on submit. */}
      <canvas ref={maskRef} className="hidden" />
    </div>
  );
});

export default MaskCanvas;
