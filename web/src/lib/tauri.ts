// Tauri runtime helpers. Detects whether the app is running inside the
// Tauri shell vs. a plain browser tab (`npm run dev` without `tauri dev`)
// so the same React code drives both targets.

import type { InvokeArgs } from '@tauri-apps/api/core';

let cachedCore: typeof import('@tauri-apps/api/core') | null = null;

export function isTauri(): boolean {
  return typeof window !== 'undefined'
    && ((window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== undefined
      || (window as unknown as { __TAURI__?: unknown }).__TAURI__ !== undefined);
}

async function core() {
  if (!cachedCore) cachedCore = await import('@tauri-apps/api/core');
  return cachedCore;
}

export async function invoke<T>(cmd: string, args?: InvokeArgs): Promise<T> {
  // Remote mode: route through the LAN server's HTTP API.
  if (isRemote()) {
    return remoteInvoke<T>(cmd, args as Record<string, unknown>);
  }
  if (!isTauri()) {
    throw new Error(`Tauri command '${cmd}' invoked outside the desktop shell`);
  }
  const { invoke } = await core();
  return invoke(cmd, args) as Promise<T>;
}

/** Wrapper around tauri's `Channel<T>`. Construct one, attach `onmessage`,
 *  pass it as a command argument. The command will push events through it.
 *
 *  In remote mode this is a no-op placeholder — streaming commands use
 *  the SSE-based wrappers (chatSend, imageGenerate) which have their own
 *  event handling and don't go through `newChannel`.
 */
export async function newChannel<T>(onMessage: (msg: T) => void) {
  if (isRemote()) {
    // Return a dummy object that the caller can pass as an arg; the
    // remote invoke path ignores it and uses SSE instead.
    return { __remote_channel: true, onMessage } as unknown as ReturnType<typeof Object.create>;
  }
  if (!isTauri()) throw new Error('Tauri Channel only available in desktop');
  const { Channel } = await core();
  const ch = new Channel<T>();
  ch.onmessage = onMessage;
  return ch;
}

// ---- app_info --------------------------------------------------------------

export interface AppFeatures {
  chat: boolean;
  image_generate: boolean;
  inpaint: boolean;
  classify: boolean;
  training: boolean;
  memory: boolean;
}

export interface AppInfo {
  name: string;
  version: string;
  data_dir: string;
  features: AppFeatures;
}

export async function appInfo(): Promise<AppInfo | null> {
  if (!isTauri()) return null;
  return invoke<AppInfo>('app_info');
}

// ---- llm -------------------------------------------------------------------

export interface LlmStatus {
  loaded: boolean;
  model_dir: string | null;
  hidden_size: number | null;
  num_layers: number | null;
}

export async function llmStatus(): Promise<LlmStatus> {
  return invoke<LlmStatus>('llm_status');
}

export async function llmLoad(modelDir: string): Promise<LlmStatus> {
  return invoke<LlmStatus>('llm_load', { modelDir });
}

export async function llmUnload(): Promise<void> {
  return invoke<void>('llm_unload');
}

// ---- memory ----------------------------------------------------------------

export type MemoryKind = 'user' | 'feedback' | 'project' | 'reference';

export interface MemoryEntry {
  id: number;
  kind: MemoryKind;
  name: string;
  description: string;
  body: string;
  created_at: number;
  updated_at: number;
}

export interface MemoryUpsert {
  id?: number | null;
  kind: MemoryKind;
  name: string;
  description: string;
  body: string;
}

export async function memoryList(kind?: MemoryKind): Promise<MemoryEntry[]> {
  return invoke<MemoryEntry[]>('memory_list', { kind: kind ?? null });
}

export async function memoryUpsert(entry: MemoryUpsert): Promise<MemoryEntry> {
  return invoke<MemoryEntry>('memory_upsert', { entry });
}

export async function memoryDelete(id: number): Promise<void> {
  return invoke<void>('memory_delete', { id });
}

// ---- conversations & chat --------------------------------------------------

export interface Conversation {
  id: number;
  title: string;
  created_at: number;
  updated_at: number;
}

export type Role = 'system' | 'user' | 'assistant';

export interface DbMessage {
  id: number;
  conversation_id: number;
  role: Role;
  content: string;
  created_at: number;
}

export interface SamplerInput {
  temperature?: number;
  top_k?: number;
  top_p?: number;
  repetition_penalty?: number;
  max_new_tokens?: number;
  seed?: number;
}

export type ToolResultPayload =
  | { ok: true; result: unknown }
  | { ok: false; code: string; message: string };

export type StreamEvent =
  | { type: 'token'; text: string }
  | { type: 'iter_end'; iter: number; count: number }
  | { type: 'tool_call'; iter: number; name: string; args: unknown; parse_error: string | null }
  | { type: 'tool_result'; iter: number; name: string; result: ToolResultPayload }
  | { type: 'done'; assistant_message_id: number; total_text: string }
  | { type: 'aborted'; assistant_message_id: number; partial: string }
  | { type: 'error'; code: string; message: string };

export async function conversationCreate(title?: string): Promise<Conversation> {
  return invoke<Conversation>('conversation_create', { title: title ?? null });
}

export async function conversationsList(): Promise<Conversation[]> {
  return invoke<Conversation[]>('conversations_list');
}

export async function conversationMessages(conversation_id: number): Promise<DbMessage[]> {
  return invoke<DbMessage[]>('conversation_messages', { conversationId: conversation_id });
}

export async function conversationDelete(id: number): Promise<void> {
  return invoke<void>('conversation_delete', { id });
}

export interface ChatSendArgs {
  conversation_id: number;
  content: string;
  system_prompt?: string;
  sampler?: SamplerInput;
}

export async function chatSend(args: ChatSendArgs, onEvent: (e: StreamEvent) => void): Promise<void> {
  if (isRemote()) {
    return remoteStream<StreamEvent>('chat_send', { args }, onEvent);
  }
  const channel = await newChannel<StreamEvent>(onEvent);
  return invoke<void>('chat_send', { args, onEvent: channel });
}

export async function chatAbort(): Promise<void> {
  return invoke<void>('chat_abort');
}

// ---- settings --------------------------------------------------------------

export async function settingsGet(key: string): Promise<string | null> {
  return invoke<string | null>('settings_get', { key });
}

export async function settingsSet(key: string, value: string): Promise<void> {
  return invoke<void>('settings_set', { key, value });
}

export interface SettingEntry { key: string; value: string; }
export async function settingsGetAll(): Promise<SettingEntry[]> {
  return invoke<SettingEntry[]>('settings_get_all');
}

// ---- image generation ------------------------------------------------------

export interface GenerateRequest {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  conversation_id?: number;
  // Inpaint / img2img — leave unset for plain txt2img.
  input_path?: string;
  mask_path?: string;
  strength?: number;
  /** Reference images (PNG bytes) for HiDream-O1 reference editing. When set,
   *  generation runs the vision-encoder + DeepStack path. Bytes (not paths) so
   *  it works incognito. */
  ref_image_bytes?: Uint8Array[];
}

export interface InpaintRequest {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  strength?: number;
  conversation_id?: number;
  /** PNG bytes of the source image (whatever the user dropped/picked). */
  input_bytes: Uint8Array;
  /** PNG bytes of the mask (white = regenerate, black = keep).
   *  Omit to regenerate the whole image (img2img mode). */
  mask_bytes?: Uint8Array;
}

export type GenEvent =
  | { type: 'started'; job_id: number; out_path: string; pid: number | null }
  | { type: 'step'; job_id: number; step: number; total: number; elapsed_s: number }
  | { type: 'log'; job_id: number; line: string }
  | { type: 'done'; job_id: number; path: string }
  | { type: 'aborted'; job_id: number }
  | { type: 'error'; job_id: number; code: string; message: string };

export interface StoredImage {
  id: number;
  path: string;
  prompt: string;
  negative_prompt: string | null;
  seed: number | null;
  steps: number | null;
  cfg_scale: number | null;
  width: number | null;
  height: number | null;
  checkpoint: string | null;
  conversation_id: number | null;
  created_at: number;
}

export async function imageGenerate(req: GenerateRequest, onEvent: (e: GenEvent) => void): Promise<number> {
  const refs = req.ref_image_bytes;
  if (isRemote()) {
    // Base64-encode reference images for JSON transport (same rationale as
    // imageInpaint — number arrays would blow the body size).
    const args: Record<string, unknown> = { ...req, ref_image_bytes: undefined };
    if (refs && refs.length) {
      args.ref_image_bytes_b64 = refs.map(uint8ToBase64);
    }
    await remoteStream<GenEvent>('image_generate', { args }, onEvent);
    return 0;
  }
  // Tauri IPC: number arrays over the native bridge.
  const args: Record<string, unknown> = { ...req };
  if (refs && refs.length) {
    args.ref_image_bytes = refs.map((b) => Array.from(b));
  }
  const channel = await newChannel<GenEvent>(onEvent);
  return invoke<number>('image_generate', { args, onEvent: channel });
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export async function imageInpaint(req: InpaintRequest, onEvent: (e: GenEvent) => void): Promise<number> {
  if (isRemote()) {
    // Base64-encode bytes for the JSON transport — a 512×512 PNG as a
    // number array would blow the body size limit; base64 is ~33% overhead
    // vs ~400% for JSON arrays.
    const args = {
      ...req,
      input_bytes: undefined,
      mask_bytes: undefined,
      input_bytes_b64: uint8ToBase64(req.input_bytes),
      mask_bytes_b64: req.mask_bytes ? uint8ToBase64(req.mask_bytes) : undefined,
    };
    await remoteStream<GenEvent>('image_inpaint', { args }, onEvent);
    return 0;
  }
  // Tauri IPC: number array works fine over the native bridge.
  const args = {
    ...req,
    input_bytes: Array.from(req.input_bytes),
    mask_bytes: req.mask_bytes ? Array.from(req.mask_bytes) : undefined,
  };
  const channel = await newChannel<GenEvent>(onEvent);
  return invoke<number>('image_inpaint', { args, onEvent: channel });
}

// ---- video generation (AnimateDiff) ----------------------------------------

export interface VideoGenerateRequest {
  prompt: string;
  negative_prompt?: string;
  steps?: number;
  cfg_scale?: number;
  seed?: number;
  num_frames?: number;
  fps?: number;
}

export async function videoGenerate(req: VideoGenerateRequest, onEvent: (e: GenEvent) => void): Promise<number> {
  if (isRemote()) {
    await remoteStream<GenEvent>('video_generate', { args: req }, onEvent);
    return 0;
  }
  const channel = await newChannel<GenEvent>(onEvent);
  return invoke<number>('video_generate', { args: req, onEvent: channel });
}

export async function imageAbort(): Promise<number | null> {
  return invoke<number | null>('image_abort');
}

export async function imageList(limit = 60): Promise<StoredImage[]> {
  return invoke<StoredImage[]>('image_list', { limit });
}

export async function imageDelete(id: number, deleteFile = false): Promise<void> {
  return invoke<void>('image_delete', { id, deleteFile });
}

/** Convert a local filesystem path into a Tauri asset:// URL usable directly
 *  as `<img src>`. Backed by the asset protocol whose scope is set in
 *  `tauri.conf.json`. Outside Tauri this falls back to the raw path. */
export async function imageSrc(path: string): Promise<string> {
  if (path.startsWith('data:')) return path;
  // Remote mode: serve images through the LAN server's asset endpoint.
  // The path from the DB is an absolute filesystem path on the host;
  // strip the data_dir prefix to get the relative segment the server
  // expects under `/api/asset/`.
  if (isRemote()) {
    // Quick heuristic: the path likely contains `.klearu/images/` — take
    // everything from `images/` onward. If not, pass the whole basename.
    const idx = path.indexOf('images/');
    const relative = idx >= 0 ? path.slice(idx) : path.split('/').pop() ?? path;
    return `/api/asset/${relative}${_remotePin ? '?pin=' + _remotePin : ''}`;
  }
  if (!isTauri()) return path;
  const { convertFileSrc } = await core();
  return convertFileSrc(path);
}

// ---- vision ----------------------------------------------------------------

export type VisionKind = 'davit' | 'vit';

export interface VisionStatus {
  loaded: boolean;
  model_dir: string | null;
  kind: VisionKind | null;
  num_classes: number | null;
  input_size: number | null;
}

export interface VisionPrediction {
  class_index: number;
  label: string;
  probability: number;
}

export async function visionStatus(): Promise<VisionStatus> {
  return invoke<VisionStatus>('vision_status');
}

export async function visionLoad(modelDir: string, forceKind?: VisionKind): Promise<VisionStatus> {
  return invoke<VisionStatus>('vision_load', {
    args: { model_dir: modelDir, force_kind: forceKind ?? null },
  });
}

export async function visionUnload(): Promise<void> {
  return invoke<void>('vision_unload');
}

export async function visionClassifyPath(path: string, topK = 5): Promise<VisionPrediction[]> {
  return invoke<VisionPrediction[]>('vision_classify', {
    args: { path, top_k: topK },
  });
}

export async function visionClassifyBytes(bytes: Uint8Array, topK = 5): Promise<VisionPrediction[]> {
  return invoke<VisionPrediction[]>('vision_classify', {
    args: { bytes: Array.from(bytes), top_k: topK },
  });
}

// ---- jobs ------------------------------------------------------------------

export type JobStatus = 'running' | 'completed' | 'failed' | 'aborted';

export interface JobProgress {
  step?: number;
  total?: number;
  loss?: number;
  elapsed_s?: number;
  extra?: Record<string, unknown>;
}

export interface JobRow {
  id: number;
  kind: string;
  label: string;
  command: string;
  status: JobStatus;
  /** Stringified `JobProgress` JSON. Parse with `JSON.parse` and cast. */
  progress: string;
  log_path: string;
  started_at: number;
  finished_at: number | null;
}

export interface JobTrainImageArgs {
  manifest: string;
  out: string;
  steps: number;
  lr?: number;
  log_every?: number;
  seed?: number;
  resume?: string;
  bpe?: string;
  binary_path?: string;
  label?: string;
}

export interface JobDistillArgs {
  prompts: string;
  sdxl: string;
  out_dir: string;
  planner_bin?: string;
  planner_args?: string;
  davit_model?: string;
  davit_conf?: number;
  n_images?: number;
  steps?: number;
  guidance?: number;
  seed?: number;
  max_prompts?: number;
  label?: string;
  python?: string;
  script?: string;
}

export async function jobList(limit = 50): Promise<JobRow[]> {
  return invoke<JobRow[]>('job_list', { limit });
}
export async function jobGet(id: number): Promise<JobRow | null> {
  return invoke<JobRow | null>('job_get', { id });
}
export async function jobAbort(id: number): Promise<boolean> {
  return invoke<boolean>('job_abort', { id });
}
export async function jobLogTail(id: number, maxLines = 200): Promise<string[]> {
  return invoke<string[]>('job_log_tail', { args: { id, max_lines: maxLines } });
}
export async function jobTrainImage(args: JobTrainImageArgs): Promise<number> {
  return invoke<number>('job_train_image', { args });
}
export async function jobDistill(args: JobDistillArgs): Promise<number> {
  return invoke<number>('job_distill', { args });
}

// ---- datasets --------------------------------------------------------------

export type DatasetKind = 'images' | 'manifest' | 'vae-tokens';

export interface DatasetRow {
  id: number;
  name: string;
  description: string;
  kind: string;
  path: string;
  item_count: number;
  created_at: number;
  updated_at: number;
}

export interface DatasetUpsertArgs {
  id?: number;
  name: string;
  description?: string;
  kind: DatasetKind;
  path: string;
}

export async function datasetList(): Promise<DatasetRow[]> {
  return invoke<DatasetRow[]>('dataset_list');
}
export async function datasetUpsert(args: DatasetUpsertArgs): Promise<DatasetRow> {
  return invoke<DatasetRow>('dataset_upsert', { args });
}
export async function datasetDelete(id: number): Promise<void> {
  return invoke<void>('dataset_delete', { id });
}

// ---- feedback + user steering vector ---------------------------------------

export type FeedbackKind = 'chat_message' | 'image';

export interface FeedbackRow {
  id: number;
  target_kind: string;
  target_id: number;
  /** +1 = 👍, -1 = 👎, 0 = neutral / cleared. */
  rating: number;
  note: string | null;
  created_at: number;
}

export interface UserVectorStatus {
  name: string;
  hidden_size: number;
  n_pos: number;
  n_neg: number;
  norm: number;
  alpha: number;
  /** First 8 floats of v; the rest stays on the backend. */
  head: number[];
}

export interface FeedbackSetResult {
  feedback: FeedbackRow;
  /** Present only when a Welford update fired (chat_message + non-zero rating
   *  + we had a captured hidden for that message). */
  user_vector: UserVectorStatus | null;
}

export async function feedbackSet(
  target_kind: FeedbackKind,
  target_id: number,
  rating: number,
  note?: string,
): Promise<FeedbackSetResult> {
  return invoke<FeedbackSetResult>('feedback_set', {
    args: { target_kind, target_id, rating, note: note ?? null },
  });
}

export async function feedbackGet(
  target_kind: FeedbackKind,
  target_id: number,
): Promise<FeedbackRow | null> {
  return invoke<FeedbackRow | null>('feedback_get', { targetKind: target_kind, targetId: target_id });
}

export async function feedbackList(target_kind?: FeedbackKind, limit = 200): Promise<FeedbackRow[]> {
  return invoke<FeedbackRow[]>('feedback_list', {
    args: { target_kind: target_kind ?? null, limit },
  });
}

export async function userVectorStatus(): Promise<UserVectorStatus> {
  return invoke<UserVectorStatus>('user_vector_status');
}

export async function userVectorSetAlpha(alpha: number): Promise<UserVectorStatus> {
  return invoke<UserVectorStatus>('user_vector_set_alpha', { args: { alpha } });
}

export async function userVectorReset(): Promise<void> {
  return invoke<void>('user_vector_reset');
}

export async function userVectorsList(): Promise<UserVectorStatus[]> {
  return invoke<UserVectorStatus[]>('user_vectors_list');
}

export async function userVectorSetAlphaNamed(name: string, alpha: number): Promise<UserVectorStatus> {
  return invoke<UserVectorStatus>('user_vector_set_alpha_named', { args: { name, alpha } });
}

export async function userVectorResetNamed(name: string): Promise<void> {
  return invoke<void>('user_vector_reset_named', { args: { name } });
}

// ---- incognito mode -------------------------------------------------------

export interface IncognitoStatus {
  active: boolean;
  started_at: number | null;
  /** Features that still work but don't persist anything to disk. */
  disabled_features: string[];
  /** Features that are refused outright because they require filesystem paths. */
  blocked_features: string[];
}

export async function incognitoStatus(): Promise<IncognitoStatus> {
  return invoke<IncognitoStatus>('incognito_status');
}

export async function incognitoStart(): Promise<IncognitoStatus> {
  return invoke<IncognitoStatus>('incognito_start');
}

export async function incognitoStop(): Promise<IncognitoStatus> {
  return invoke<IncognitoStatus>('incognito_stop');
}

// ---- LAN sharing -----------------------------------------------------------

export interface LanStatus {
  active: boolean;
  port: number | null;
  pin: string | null;
  urls: string[];
  cert_fingerprint: string | null;
}

export async function lanStatus(): Promise<LanStatus> {
  return invoke<LanStatus>('lan_status');
}
export async function lanStart(port?: number): Promise<LanStatus> {
  return invoke<LanStatus>('lan_start', { port: port ?? null });
}
export async function lanStop(): Promise<LanStatus> {
  return invoke<LanStatus>('lan_stop');
}

// ---- remote mode -----------------------------------------------------------

/** True when the page is served by the embedded LAN server (not Tauri
 *  IPC and not the Vite dev server). In this mode `invoke` routes through
 *  `fetch('/api/invoke')` with the PIN as a bearer token instead of the
 *  Tauri IPC bridge. */
export function isRemote(): boolean {
  if (isTauri()) return false;
  // The Vite dev server runs on localhost:1420; anything else is the LAN
  // server (which serves the built app on its own port).
  if (typeof window === 'undefined') return false;
  const { hostname, port } = window.location;
  if (hostname === 'localhost' && port === '1420') return false;
  return true;
}

let _remotePin: string | null = null;

export function setRemotePin(pin: string) { _remotePin = pin; }

/** Invoke a command via the LAN server's REST endpoint. Mirrors the Tauri
 *  IPC `invoke` contract so callers don't branch on transport. */
export async function remoteInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const resp = await fetch('/api/invoke', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ..._remotePin ? { 'Authorization': `Bearer ${_remotePin}` } : {},
    },
    body: JSON.stringify({ command: cmd, args: args ?? {} }),
  });
  if (resp.status === 401) throw new Error('Unauthorized — wrong PIN');
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(body.error || resp.statusText);
  }
  return resp.json() as Promise<T>;
}

/** Stream a command via the LAN server's SSE endpoint. Each `data:` line
 *  is parsed as JSON and dispatched to `onEvent`. Resolves when the
 *  EventSource closes (the server sends a final event then drops). */
export async function remoteStream<T>(
  cmd: string,
  args: Record<string, unknown>,
  onEvent: (e: T) => void,
): Promise<void> {
  const resp = await fetch('/api/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ..._remotePin ? { 'Authorization': `Bearer ${_remotePin}` } : {},
    },
    body: JSON.stringify({ command: cmd, args }),
  });
  if (resp.status === 401) throw new Error('Unauthorized — wrong PIN');
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(body || resp.statusText);
  }
  const reader = resp.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE lines are `data: {...}\n\n`. Parse them out.
    const lines = buf.split('\n');
    buf = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith('data:')) {
        const json = trimmed.slice(5).trim();
        if (json) {
          try { onEvent(JSON.parse(json) as T); }
          catch { /* skip unparseable lines */ }
        }
      }
    }
  }
}
