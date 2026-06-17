/**
 * Client-side DPF key generation using WebCrypto AES-128.
 *
 * Matches the Rust klearu-dpf crate exactly:
 * - AesPrg: MMO-based PRG (AES_k(seed) XOR seed)
 * - makeSeed: deterministic seed from (alpha, beta, party)
 * - dpfGen: BGI construction for point functions
 * - serializeDpfKey: wire format for server consumption
 */

// --- AES helpers ---

/** Import a raw 16-byte key for AES-CBC encryption. */
async function importAesKey(raw: Uint8Array): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    'raw',
    raw.buffer as ArrayBuffer,
    { name: 'AES-CBC' },
    false,
    ['encrypt']
  );
}

/** Single-block AES-128 encrypt (ECB via CBC with IV=0). Returns 16 bytes. */
async function aesEncryptBlock(
  key: CryptoKey,
  block: Uint8Array
): Promise<Uint8Array> {
  const iv = new Uint8Array(16);
  const ct = await crypto.subtle.encrypt(
    { name: 'AES-CBC', iv },
    key,
    block.buffer as ArrayBuffer
  );
  // CBC+PKCS7 on 16 bytes → 32 bytes output; first 16 is the ciphertext block
  return new Uint8Array(ct, 0, 16);
}

// --- AesPrg ---

export class AesPrg {
  private constructor(
    private cipherLeftKey: CryptoKey,
    private cipherRightKey: CryptoKey
  ) {}

  static async create(key: Uint8Array): Promise<AesPrg> {
    const keyLeft = new Uint8Array(16);
    const keyRight = new Uint8Array(16);
    for (let i = 0; i < 16; i++) {
      keyLeft[i] = key[i] ^ 0xaa;
      keyRight[i] = key[i] ^ 0x55;
    }
    const [ckLeft, ckRight] = await Promise.all([
      importAesKey(keyLeft),
      importAesKey(keyRight),
    ]);
    return new AesPrg(ckLeft, ckRight);
  }

  /** Expand seed → (leftSeed, leftControl, rightSeed, rightControl). */
  async expand(
    seed: Uint8Array
  ): Promise<[Uint8Array, boolean, Uint8Array, boolean]> {
    const [leftEnc, rightEnc] = await Promise.all([
      aesEncryptBlock(this.cipherLeftKey, seed),
      aesEncryptBlock(this.cipherRightKey, seed),
    ]);

    const leftControl = (leftEnc[0] & 1) !== 0;
    const leftSeed = new Uint8Array(16);
    for (let i = 0; i < 16; i++) leftSeed[i] = leftEnc[i] ^ seed[i];
    leftSeed[0] &= 0xfe;

    const rightControl = (rightEnc[0] & 1) !== 0;
    const rightSeed = new Uint8Array(16);
    for (let i = 0; i < 16; i++) rightSeed[i] = rightEnc[i] ^ seed[i];
    rightSeed[0] &= 0xfe;

    return [leftSeed, leftControl, rightSeed, rightControl];
  }

  /** Derive a u32 output from a seed (leaf evaluation). */
  async deriveOutput(seed: Uint8Array): Promise<number> {
    const enc = await aesEncryptBlock(this.cipherLeftKey, seed);
    // u32 from bytes [4..8] LE
    return (
      enc[4] | (enc[5] << 8) | (enc[6] << 16) | ((enc[7] << 24) >>> 0)
    ) >>> 0;
  }
}

// --- Seed generation matching Rust make_seed ---

// Cache seed CryptoKeys (party 0 = 0x10, party 1 = 0x11) — same for all tokens
let seedKeyCache: [CryptoKey | null, CryptoKey | null] = [null, null];

async function getSeedKey(party: number): Promise<CryptoKey> {
  if (!seedKeyCache[party]) {
    const keyByte = (0x10 + party) & 0xff;
    seedKeyCache[party] = await importAesKey(new Uint8Array(16).fill(keyByte));
  }
  return seedKeyCache[party]!;
}

async function makeSeed(
  alpha: number,
  beta: number,
  party: number
): Promise<Uint8Array> {
  const s = new Uint8Array(16);
  // alpha LE
  s[0] = alpha & 0xff;
  s[1] = (alpha >>> 8) & 0xff;
  s[2] = (alpha >>> 16) & 0xff;
  s[3] = (alpha >>> 24) & 0xff;
  // beta LE
  s[4] = beta & 0xff;
  s[5] = (beta >>> 8) & 0xff;
  s[6] = (beta >>> 16) & 0xff;
  s[7] = (beta >>> 24) & 0xff;
  s[8] = party & 0xff;
  s[9] = 0xab;
  // bytes 10..15 = 0

  const ck = await getSeedKey(party);
  return aesEncryptBlock(ck, s);
}

// --- DPF key type ---

interface CorrectionWord {
  seed: Uint8Array; // 16 bytes
  controlLeft: boolean;
  controlRight: boolean;
}

export interface DpfKey {
  party: number; // 0 or 1
  seed: Uint8Array; // 16 bytes
  correctionWords: CorrectionWord[];
  outputCorrection: number; // u32
}

// --- Bit helpers ---

function getBit(x: number, i: number, depth: number): boolean {
  return ((x >>> (depth - 1 - i)) & 1) !== 0;
}

function xorSeeds(a: Uint8Array, b: Uint8Array): Uint8Array {
  const r = new Uint8Array(16);
  for (let i = 0; i < 16; i++) r[i] = a[i] ^ b[i];
  return r;
}

// --- DPF key generation (BGI construction) ---

export async function dpfGen(
  prg: AesPrg,
  alpha: number,
  beta: number,
  depth: number
): Promise<[DpfKey, DpfKey]> {
  const seed0 = await makeSeed(alpha, beta, 0);
  const seed1 = await makeSeed(alpha, beta, 1);

  let s = [seed0, seed1] as [Uint8Array, Uint8Array];
  let t = [false, true] as [boolean, boolean];

  const correctionWords: CorrectionWord[] = [];

  for (let i = 0; i < depth; i++) {
    const alphaBit = getBit(alpha, i, depth);

    // Expand both parties' current seeds
    const [expanded0, expanded1] = await Promise.all([
      prg.expand(s[0]),
      prg.expand(s[1]),
    ]);

    // expanded = [leftSeed, leftControl, rightSeed, rightControl]
    const sChildren = [
      [expanded0[0], expanded0[2]], // party 0: [left, right]
      [expanded1[0], expanded1[2]], // party 1: [left, right]
    ] as [[Uint8Array, Uint8Array], [Uint8Array, Uint8Array]];

    const tChildren = [
      [expanded0[1], expanded0[3]], // party 0: [left_t, right_t]
      [expanded1[1], expanded1[3]], // party 1: [left_t, right_t]
    ] as [[boolean, boolean], [boolean, boolean]];

    const keep = alphaBit ? 1 : 0;
    const lose = 1 - keep;

    // CW seed: XOR of lose children
    const cwSeed = xorSeeds(
      sChildren[0][lose] as Uint8Array,
      sChildren[1][lose] as Uint8Array
    );

    // CW control bits: match Rust dpf_gen exactly
    const tcw = [false, false] as [boolean, boolean];
    // tcw[side]: if side == lose → t_children[0][side] ^ t_children[1][side]
    //            if side == keep → t_children[0][side] ^ t_children[1][side] ^ true
    for (let side = 0; side < 2; side++) {
      const base = tChildren[0][side] !== tChildren[1][side]; // XOR
      if (side === lose) {
        tcw[side] = base; // lose: want XOR = 0
      } else {
        tcw[side] = !base; // keep: want XOR = 1 → base ^ true = !base
      }
    }

    correctionWords.push({
      seed: cwSeed,
      controlLeft: tcw[0],
      controlRight: tcw[1],
    });

    // Advance both parties to the keep direction
    for (let b = 0; b < 2; b++) {
      let childSeed: Uint8Array = new Uint8Array(sChildren[b][keep] as Uint8Array);
      let childT = tChildren[b][keep] as boolean;

      if (t[b]) {
        const xored = xorSeeds(childSeed, cwSeed);
        childSeed = new Uint8Array(xored);
        childT = childT !== tcw[keep]; // XOR
      }

      s[b] = childSeed;
      t[b] = childT;
    }
  }

  // Output correction at the target leaf
  const [out0, out1] = await Promise.all([
    prg.deriveOutput(s[0]),
    prg.deriveOutput(s[1]),
  ]);

  let oc: number;
  if (t[0] && !t[1]) {
    // beta - out0 + out1 (wrapping u32)
    oc = ((beta - out0 + out1) >>> 0) & 0xffffffff;
  } else {
    // out0 - out1 - beta (wrapping u32)
    oc = ((out0 - out1 - beta) >>> 0) & 0xffffffff;
  }

  return [
    {
      party: 0,
      seed: seed0,
      correctionWords: correctionWords.map((cw) => ({
        seed: new Uint8Array(cw.seed),
        controlLeft: cw.controlLeft,
        controlRight: cw.controlRight,
      })),
      outputCorrection: oc,
    },
    {
      party: 1,
      seed: seed1,
      correctionWords,
      outputCorrection: oc,
    },
  ];
}

// --- Serialization ---

/**
 * Serialize a DPF key to bytes (wire format).
 * Format: party(1) + seed(16) + output_correction(4) + depth × (cw_seed(16) + cw_bits(1))
 */
export function serializeDpfKey(key: DpfKey): Uint8Array {
  const depth = key.correctionWords.length;
  const len = 1 + 16 + 4 + depth * 17;
  const buf = new Uint8Array(len);
  const view = new DataView(buf.buffer);
  let offset = 0;

  buf[offset++] = key.party;
  buf.set(key.seed, offset);
  offset += 16;
  view.setUint32(offset, key.outputCorrection, true);
  offset += 4;

  for (const cw of key.correctionWords) {
    buf.set(cw.seed, offset);
    offset += 16;
    const bits =
      (cw.controlLeft ? 1 : 0) | (cw.controlRight ? 2 : 0);
    buf[offset++] = bits;
  }

  return buf;
}

/** Compute the DPF depth for a given vocab size (smallest d where 2^d >= vocabSize). */
export function dpfDepthForVocab(vocabSize: number): number {
  let d = 1;
  while ((1 << d) < vocabSize) d++;
  return d;
}
