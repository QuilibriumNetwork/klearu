/**
 * Lightweight BPE tokenizer that loads from HuggingFace tokenizer.json.
 *
 * Supports byte-level BPE (GPT-2/Llama/Qwen-style).
 */

interface TokenizerJson {
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: string[];
  };
  added_tokens?: Array<{
    id: number;
    content: string;
    special: boolean;
  }>;
}

export class BpeTokenizer {
  private encoder: Map<string, number>;
  private decoder: Map<number, string>;
  private merges: Array<[string, string]>;
  private mergeRanks: Map<string, number>;
  private byteEncoder: Map<number, string>;
  private byteDecoder: Map<string, number>;
  private specialTokens: Map<string, number>;
  private specialPattern: RegExp | null;

  constructor(json: TokenizerJson) {
    // Build vocab maps
    this.encoder = new Map(Object.entries(json.model.vocab));
    this.decoder = new Map<number, string>();
    for (const [token, id] of this.encoder) {
      this.decoder.set(id, token);
    }

    // Parse merges
    this.merges = json.model.merges.map((m) => {
      const parts = m.split(' ');
      return [parts[0], parts[1]] as [string, string];
    });

    this.mergeRanks = new Map<string, number>();
    for (let i = 0; i < this.merges.length; i++) {
      this.mergeRanks.set(this.merges[i][0] + ' ' + this.merges[i][1], i);
    }

    // Build byte-level encoder (maps 0-255 to unicode chars)
    this.byteEncoder = new Map<number, string>();
    this.byteDecoder = new Map<string, number>();
    const bs: number[] = [];
    // Printable ASCII ranges
    for (let i = 33; i <= 126; i++) bs.push(i); // '!' to '~'
    for (let i = 161; i <= 172; i++) bs.push(i); // '¡' to '¬'
    for (let i = 174; i <= 255; i++) bs.push(i); // '®' to 'ÿ'

    const cs = [...bs];
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }

    for (let i = 0; i < bs.length; i++) {
      this.byteEncoder.set(bs[i], String.fromCodePoint(cs[i]));
      this.byteDecoder.set(String.fromCodePoint(cs[i]), bs[i]);
    }

    // Special / added tokens (matched literally during encoding)
    this.specialTokens = new Map<string, number>();
    if (json.added_tokens) {
      for (const t of json.added_tokens) {
        this.specialTokens.set(t.content, t.id);
      }
    }

    // Build regex for splitting on special tokens (longest first)
    this.specialPattern = null;
    if (this.specialTokens.size > 0) {
      const sorted = [...this.specialTokens.keys()].sort(
        (a, b) => b.length - a.length
      );
      const escaped = sorted.map((t) =>
        t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      );
      this.specialPattern = new RegExp(`(${escaped.join('|')})`);
    }
  }

  /** Encode text to token IDs, handling special tokens. */
  encode(text: string): number[] {
    if (!text) return [];

    // Split on special tokens, encode each segment
    const segments = this.splitOnSpecialTokens(text);
    const tokens: number[] = [];

    for (const seg of segments) {
      const specialId = this.specialTokens.get(seg);
      if (specialId !== undefined) {
        tokens.push(specialId);
      } else {
        tokens.push(...this.encodeBpe(seg));
      }
    }

    return tokens;
  }

  /** BPE-encode a text segment (no special tokens). */
  private encodeBpe(text: string): number[] {
    const tokens: number[] = [];
    const pat =
      /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    const matches = text.match(pat);
    if (!matches) return tokens;

    for (const word of matches) {
      const encoded = new TextEncoder().encode(word);
      const bpeChars = Array.from(encoded)
        .map((b) => this.byteEncoder.get(b)!)
        .join('');
      const bpeTokens = this.bpe(bpeChars);
      for (const t of bpeTokens) {
        const id = this.encoder.get(t);
        if (id !== undefined) {
          tokens.push(id);
        }
      }
    }
    return tokens;
  }

  /** Split text on special token boundaries, preserving them as separate segments. */
  private splitOnSpecialTokens(text: string): string[] {
    if (!this.specialPattern) return [text];
    return text.split(this.specialPattern).filter((s) => s.length > 0);
  }

  /** Decode token IDs to text. */
  decode(ids: number[]): string {
    const tokens = ids.map((id) => this.decoder.get(id) ?? '');
    const text = tokens.join('');
    // Convert byte-level characters back to bytes
    const bytes = new Uint8Array(
      Array.from(text).map((c) => this.byteDecoder.get(c) ?? 0)
    );
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  }

  get vocabSize(): number {
    return this.encoder.size;
  }

  private bpe(token: string): string[] {
    if (token.length <= 1) return [token];

    let word = Array.from(token);
    let pairs = getPairs(word);

    while (true) {
      // Find the highest-priority merge
      let minRank = Infinity;
      let bestPair: [string, string] | null = null;
      for (const [a, b] of pairs) {
        const rank = this.mergeRanks.get(a + ' ' + b);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          bestPair = [a, b];
        }
      }

      if (!bestPair) break;

      // Apply the merge
      const [first, second] = bestPair;
      const newWord: string[] = [];
      let i = 0;
      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        if (j < word.length - 1 && word[j] === first && word[j + 1] === second) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }
      word = newWord;

      if (word.length === 1) break;
      pairs = getPairs(word);
    }

    return word;
  }
}

function getPairs(word: string[]): Set<[string, string]> {
  const pairs = new Set<[string, string]>();
  for (let i = 0; i < word.length - 1; i++) {
    pairs.add([word[i], word[i + 1]]);
  }
  return pairs;
}

/** Load tokenizer from a URL pointing to tokenizer.json. */
export async function loadTokenizer(url: string): Promise<BpeTokenizer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load tokenizer: ${response.status}`);
  }
  const json: TokenizerJson = await response.json();
  return new BpeTokenizer(json);
}
