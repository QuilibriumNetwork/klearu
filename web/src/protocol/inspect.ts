/**
 * Protocol event capture and structure parsing for the inspector view.
 */

export const MAX_CAPTURE_BYTES = 1024;

export interface ProtocolEvent {
  id: number;
  timestamp: number; // performance.now() ms
  direction: 'sent' | 'received';
  server: 'A' | 'B';
  tag: number;
  rawBytes: Uint8Array; // first MAX_CAPTURE_BYTES
  totalBytes: number;
}

export interface StructureField {
  name: string;
  offset: number;
  length: number;
  value: string;
  opaque: boolean;
  description?: string;
}

const TAG_META: Record<number, { name: string; description: string }> = {
  0x01: {
    name: 'SessionInit',
    description: 'Client requests a new private inference session',
  },
  0x02: {
    name: 'SessionReady',
    description:
      'Server responds with session configuration and crypto parameters',
  },
  0x03: {
    name: 'JoinSession',
    description: 'Client joins the session on Server B using session ID',
  },
  0x04: {
    name: 'JoinAck',
    description: 'Server B acknowledges the session join',
  },
  0x51: {
    name: 'LogitShare',
    description:
      'Additive secret share of model output logits (Q16.16 fixed-point)',
  },
  0x52: {
    name: 'NextToken',
    description:
      'DPF key encoding a private token selection — hides the token ID from the server',
  },
  0x42: { name: 'EOS', description: 'End-of-sequence signal' },
  0xf0: { name: 'Error', description: 'Server error message' },
};

export function tagName(tag: number): string {
  return TAG_META[tag]?.name ?? `0x${tag.toString(16).padStart(2, '0')}`;
}

export function tagDescription(tag: number): string | undefined {
  return TAG_META[tag]?.description;
}

// --- byte readers ---

function readU8(buf: Uint8Array, off: number): number {
  return buf[off];
}

function readU32LE(buf: Uint8Array, off: number): number {
  return (
    (buf[off] |
      (buf[off + 1] << 8) |
      (buf[off + 2] << 16) |
      ((buf[off + 3] << 24) >>> 0)) >>>
    0
  );
}

function readF32LE(buf: Uint8Array, off: number): number {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  return view.getFloat32(off, true);
}

function hexStr(buf: Uint8Array, off: number, len: number): string {
  const end = Math.min(off + len, buf.length);
  const parts: string[] = [];
  for (let i = off; i < end; i++) {
    parts.push(buf[i].toString(16).padStart(2, '0'));
  }
  return parts.join(' ');
}

// --- structure parser ---

export function parseStructure(event: ProtocolEvent): StructureField[] {
  const { tag, rawBytes: buf, totalBytes } = event;
  const fields: StructureField[] = [];

  // Tag byte
  fields.push({
    name: 'Tag',
    offset: 0,
    length: 1,
    value: `0x${tag.toString(16).padStart(2, '0')} (${tagName(tag)})`,
    opaque: false,
    description: TAG_META[tag]?.description,
  });

  switch (tag) {
    case 0x01:
      parseSessionInit(buf, fields);
      break;
    case 0x02:
      parseSessionReady(buf, fields);
      break;
    case 0x03:
      parseJoinSession(buf, fields);
      break;
    case 0x04:
      break; // no payload
    case 0x52:
      parseDpfKey(buf, fields);
      break;
    case 0x51:
      parseLogitShare(buf, totalBytes, fields);
      break;
    case 0x42:
      break; // no payload
    case 0xf0:
      parseError(buf, fields);
      break;
  }

  return fields;
}

function parseSessionInit(buf: Uint8Array, fields: StructureField[]) {
  if (buf.length < 22) return;
  fields.push({
    name: 'Security Level',
    offset: 1,
    length: 1,
    value: ['Low', 'High'][readU8(buf, 1)] ?? `${readU8(buf, 1)}`,
    opaque: false,
  });
  fields.push({
    name: 'Max New Tokens',
    offset: 2,
    length: 4,
    value: `${readU32LE(buf, 2)}`,
    opaque: false,
  });
  fields.push({
    name: 'Temperature',
    offset: 6,
    length: 4,
    value: readF32LE(buf, 6).toFixed(4),
    opaque: false,
  });
  fields.push({
    name: 'Top-K',
    offset: 10,
    length: 4,
    value: `${readU32LE(buf, 10)}`,
    opaque: false,
  });
  fields.push({
    name: 'Top-P',
    offset: 14,
    length: 4,
    value: readF32LE(buf, 14).toFixed(4),
    opaque: false,
  });
  fields.push({
    name: 'Prompt Token Count',
    offset: 18,
    length: 4,
    value: `${readU32LE(buf, 18)}`,
    opaque: false,
  });
}

function parseSessionReady(buf: Uint8Array, fields: StructureField[]) {
  if (buf.length <= 1) return;
  const json = new TextDecoder().decode(buf.slice(1));
  let preview = json;
  if (preview.length > 300) preview = preview.slice(0, 300) + '\u2026';
  fields.push({
    name: 'Config JSON',
    offset: 1,
    length: buf.length - 1,
    value: preview,
    opaque: false,
    description:
      'Model configuration, session ID, and PRG key for DPF generation',
  });
}

function parseJoinSession(buf: Uint8Array, fields: StructureField[]) {
  if (buf.length < 9) return;
  const lo = readU32LE(buf, 1);
  const hi = readU32LE(buf, 5);
  const sid = ((BigInt(hi) << 32n) | BigInt(lo))
    .toString(16)
    .padStart(16, '0');
  fields.push({
    name: 'Session ID',
    offset: 1,
    length: 8,
    value: `0x${sid}`,
    opaque: false,
  });
}

function parseDpfKey(buf: Uint8Array, fields: StructureField[]) {
  if (buf.length < 22) return;
  fields.push({
    name: 'Party',
    offset: 1,
    length: 1,
    value: `${readU8(buf, 1)}`,
    opaque: true,
    description: 'DPF key party index (0 or 1)',
  });
  fields.push({
    name: 'Initial Seed',
    offset: 2,
    length: 16,
    value: hexStr(buf, 2, 16),
    opaque: true,
    description:
      'Pseudorandom 128-bit seed \u2014 computationally indistinguishable from random without the complementary key',
  });
  fields.push({
    name: 'Output Correction',
    offset: 18,
    length: 4,
    value: `0x${readU32LE(buf, 18).toString(16).padStart(8, '0')}`,
    opaque: true,
    description:
      'Leaf correction value \u2014 ensures the DPF evaluates to the target value at the secret index',
  });

  const cwStart = 22;
  const cwBytes = buf.length - cwStart;
  const depth = Math.floor(cwBytes / 17);
  if (depth > 0) {
    fields.push({
      name: 'Correction Words',
      offset: cwStart,
      length: depth * 17,
      value: `${depth} levels \u00d7 17 bytes = ${depth * 17} bytes`,
      opaque: true,
      description:
        'Tree-structured correction data: each level has a 16-byte pseudorandom seed and 1-byte control bits. ' +
        'The full key encodes a point function over the token vocabulary \u2014 the server evaluates it over all indices ' +
        'but learns nothing about which index the client selected.',
    });
  }
}

function parseLogitShare(
  buf: Uint8Array,
  totalBytes: number,
  fields: StructureField[]
) {
  const payloadBytes = totalBytes - 1;
  const numValues = Math.floor(payloadBytes / 4);

  fields.push({
    name: 'Logit Shares',
    offset: 1,
    length: payloadBytes,
    value: `${numValues} \u00d7 u32 values (${formatBytes(payloadBytes)})`,
    opaque: true,
    description:
      'Additive Q16.16 secret shares of the model logits. Each individual value is uniformly random \u2014 ' +
      'only the client, who holds shares from both servers, can reconstruct the actual logit distribution. ' +
      'Neither server can determine which token was selected or what the model predicted.',
  });

  // Show first few sample values
  const capturedValues = Math.min(
    8,
    Math.floor((buf.length - 1) / 4)
  );
  if (capturedValues > 0) {
    const samples: string[] = [];
    for (let i = 0; i < capturedValues; i++) {
      samples.push(
        `0x${readU32LE(buf, 1 + i * 4)
          .toString(16)
          .padStart(8, '0')}`
      );
    }
    fields.push({
      name: 'Sample Values',
      offset: 1,
      length: capturedValues * 4,
      value: samples.join(', ') + (numValues > capturedValues ? ', \u2026' : ''),
      opaque: true,
      description:
        'First few share values \u2014 note the uniform distribution, indistinguishable from random noise',
    });
  }
}

function parseError(buf: Uint8Array, fields: StructureField[]) {
  if (buf.length <= 1) return;
  fields.push({
    name: 'Error Message',
    offset: 1,
    length: buf.length - 1,
    value: new TextDecoder().decode(buf.slice(1)),
    opaque: false,
  });
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}
