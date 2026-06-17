// Client-side image resize + center-crop to a square target size.
// Runs entirely in-memory via an offscreen canvas — no disk, no fetch,
// incognito-safe.

/**
 * Load an image from bytes or a URL, center-crop to a square, resize to
 * `size × size`, and return the result as PNG `Uint8Array` + an object
 * URL for display.
 *
 * The crop is center-aligned: the shorter edge is preserved in full, and
 * equal amounts are trimmed from the longer edge's ends. This matches the
 * SD/SDXL convention of square inputs and avoids the distortion that a
 * naive stretch would introduce.
 */
export async function resizeCropSquare(
  source: Blob | string,
  size: number,
): Promise<{ bytes: Uint8Array; objectUrl: string }> {
  const img = await loadImage(source);
  const { naturalWidth: sw, naturalHeight: sh } = img;

  // Center-crop coordinates: take the largest centered square.
  const side = Math.min(sw, sh);
  const sx = Math.floor((sw - side) / 2);
  const sy = Math.floor((sh - side) / 2);

  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d')!;
  // Bicubic-ish: the browser's native drawImage uses bilinear by default;
  // `imageSmoothingQuality = 'high'` promotes to Lanczos in Chrome/Safari.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, sx, sy, side, side, 0, 0, size, size);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (b) => (b ? resolve(b) : reject(new Error('canvas.toBlob returned null'))),
      'image/png',
    );
  });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const objectUrl = URL.createObjectURL(blob);
  return { bytes, objectUrl };
}

function loadImage(source: Blob | string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    if (typeof source === 'string') {
      img.src = source;
    } else {
      img.src = URL.createObjectURL(source);
    }
  });
}
