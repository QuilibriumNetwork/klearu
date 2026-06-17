/**
 * Q32.32 secret sharing and logit reconstruction.
 *
 * All arithmetic is done with BigInt to match the Rust Q32.32 (u64) format.
 */

const SCALE_64 = 4294967296n; // 2^32
const MOD_64 = 18446744073709551616n; // 2^64

/** Convert f32 to Q32.32 fixed-point (u64). */
export function toFixed64(x: number): bigint {
  const scaled = BigInt(Math.round(x * Number(SCALE_64)));
  // Wrap to u64 range
  return ((scaled % MOD_64) + MOD_64) % MOD_64;
}

/** Convert Q32.32 fixed-point (u64) to f32. */
export function fromFixed64(x: bigint): number {
  // Interpret as signed i64, then divide by 2^32
  const signed = BigInt.asIntN(64, x);
  return Number(signed) / Number(SCALE_64);
}

/**
 * Create random Q32.32 additive shares from an f32 embedding row.
 *
 * Returns { shareA, shareB } where each is a Uint8Array of n*8 bytes (u64 LE).
 * shareA + shareB ≡ embedding (mod 2^64) in Q32.32.
 */
export function createEmbeddingShares(embedding: Float32Array): {
  shareA: Uint8Array;
  shareB: Uint8Array;
} {
  const n = embedding.length;
  // Generate random bytes for shareB
  const shareB = new Uint8Array(n * 8);
  crypto.getRandomValues(shareB);

  const shareA = new Uint8Array(n * 8);
  const bView = new DataView(shareB.buffer, shareB.byteOffset, shareB.byteLength);
  const aView = new DataView(shareA.buffer, shareA.byteOffset, shareA.byteLength);

  for (let i = 0; i < n; i++) {
    const q32 = toFixed64(embedding[i]);
    const rb = bView.getBigUint64(i * 8, true); // little-endian
    // shareA = q32 - shareB (wrapping subtraction mod 2^64)
    const sa = ((q32 - rb) % MOD_64 + MOD_64) % MOD_64;
    aView.setBigUint64(i * 8, sa, true);
  }

  return { shareA, shareB };
}

/**
 * Reconstruct f32 logits from two Q32.32 logit shares.
 *
 * Each share is a Uint8Array of vocabSize*8 bytes (u64 LE).
 */
export function reconstructLogits(
  shareA: Uint8Array,
  shareB: Uint8Array,
  vocabSize: number
): Float32Array {
  const logits = new Float32Array(vocabSize);
  const aView = new DataView(shareA.buffer, shareA.byteOffset, shareA.byteLength);
  const bView = new DataView(shareB.buffer, shareB.byteOffset, shareB.byteLength);

  for (let i = 0; i < vocabSize; i++) {
    const a = aView.getBigUint64(i * 8, true);
    const b = bView.getBigUint64(i * 8, true);
    const sum = (a + b) % MOD_64;
    logits[i] = fromFixed64(sum);
  }

  return logits;
}

/**
 * Reconstruct f32 logits from two Q16.16 logit shares.
 *
 * Each share is a Uint8Array of vocabSize*4 bytes (u32 LE).
 * Shares are added modulo 2^32 and interpreted as signed Q16.16.
 */
export function reconstructLogitsQ16(
  shareA: Uint8Array,
  shareB: Uint8Array,
  vocabSize: number
): Float32Array {
  const logits = new Float32Array(vocabSize);
  const aView = new DataView(shareA.buffer, shareA.byteOffset, shareA.byteLength);
  const bView = new DataView(shareB.buffer, shareB.byteOffset, shareB.byteLength);
  for (let i = 0; i < vocabSize; i++) {
    const a = aView.getUint32(i * 4, true);
    const b = bView.getUint32(i * 4, true);
    const sum = (a + b) | 0; // wrapping u32 add → interpret as i32
    logits[i] = sum / 65536.0; // Q16.16 → f32
  }
  return logits;
}

/** Greedy sampling (argmax). */
export function sampleGreedy(logits: Float32Array): number {
  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
