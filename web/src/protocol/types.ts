export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export enum ConnectionStatus {
  Disconnected = 'disconnected',
  Connecting = 'connecting',
  Connected = 'connected',
  LoadingTokenizer = 'loading tokenizer',
  InitDPF = 'initializing DPF',
  Prefilling = 'prefilling',
  Generating = 'generating',
  Error = 'error',
}
