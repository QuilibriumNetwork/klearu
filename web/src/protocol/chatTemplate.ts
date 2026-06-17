/**
 * Chat template detection and formatting.
 *
 * Mirrors the Rust implementation in klearu-llm/src/generate/chat_template.rs.
 * Detects template format from tokenizer_config.json and formats conversations.
 */

export type TemplateType =
  | 'chatml'
  | 'zephyr'
  | 'llama2'
  | 'llama3'
  | 'mistral'
  | 'raw';

interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** Detect template format from the Jinja chat_template string in tokenizer_config.json. */
export function detectTemplate(
  chatTemplate: string | null | undefined
): TemplateType {
  if (!chatTemplate) return 'raw';
  if (chatTemplate.includes('<|im_start|>')) return 'chatml';
  if (chatTemplate.includes('<|start_header_id|>')) return 'llama3';
  if (
    chatTemplate.includes('<|system|>') ||
    chatTemplate.includes('<|user|>')
  )
    return 'zephyr';
  if (chatTemplate.includes('<<SYS>>')) return 'llama2';
  if (chatTemplate.includes('[INST]')) return 'mistral';
  return 'raw';
}

/** Format a conversation into a prompt string using the given template. */
export function applyTemplate(
  template: TemplateType,
  messages: Message[]
): string {
  switch (template) {
    case 'chatml':
      return formatChatML(messages);
    case 'zephyr':
      return formatZephyr(messages);
    case 'llama2':
      return formatLlama2(messages);
    case 'llama3':
      return formatLlama3(messages);
    case 'mistral':
      return formatMistral(messages);
    case 'raw':
      return messages.map((m) => m.content).join('\n');
  }
}

function formatChatML(messages: Message[]): string {
  let out = '';
  for (const msg of messages) {
    out += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
  }
  out += '<|im_start|>assistant\n';
  return out;
}

function formatZephyr(messages: Message[]): string {
  let out = '';
  for (const msg of messages) {
    out += `<|${msg.role}|>\n${msg.content}</s>\n`;
  }
  out += '<|assistant|>\n';
  return out;
}

function formatLlama2(messages: Message[]): string {
  let out = '';
  const systemMsg = messages.find((m) => m.role === 'system')?.content;
  const nonSystem = messages.filter((m) => m.role !== 'system');

  for (let i = 0; i < nonSystem.length; i++) {
    const msg = nonSystem[i];
    if (msg.role === 'user') {
      out += '[INST] ';
      if (i === 0 && systemMsg) {
        out += `<<SYS>>\n${systemMsg}\n<</SYS>>\n\n`;
      }
      out += `${msg.content} [/INST] `;
    } else if (msg.role === 'assistant') {
      out += `${msg.content} </s>`;
    }
  }
  return out;
}

function formatLlama3(messages: Message[]): string {
  let out = '<|begin_of_text|>';
  for (const msg of messages) {
    out += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id|>`;
  }
  out += '<|start_header_id|>assistant<|end_header_id|>\n\n';
  return out;
}

function formatMistral(messages: Message[]): string {
  let out = '';
  let pendingUser = '';

  for (const msg of messages) {
    if (msg.role === 'system') {
      pendingUser += msg.content + '\n\n';
    } else if (msg.role === 'user') {
      pendingUser += msg.content;
      out += `[INST] ${pendingUser} [/INST]`;
      pendingUser = '';
    } else if (msg.role === 'assistant') {
      out += `${msg.content}</s>`;
    }
  }

  if (pendingUser) {
    out += `[INST] ${pendingUser} [/INST]`;
  }

  return out;
}

/** Load tokenizer_config.json and extract the chat_template string. */
export async function loadChatTemplate(
  baseUrl: string
): Promise<TemplateType> {
  try {
    const resp = await fetch(`${baseUrl}/api/models/tokenizer_config.json`);
    if (!resp.ok) return 'raw';
    const config = await resp.json();
    return detectTemplate(config.chat_template ?? null);
  } catch {
    return 'raw';
  }
}
