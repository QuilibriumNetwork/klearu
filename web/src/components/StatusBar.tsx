import { ConnectionStatus } from '../protocol/types';

interface Props {
  status: ConnectionStatus;
  tokensPerSec: number | null;
}

export default function StatusBar({ status, tokensPerSec }: Props) {
  const statusColor = {
    [ConnectionStatus.Disconnected]: 'text-gray-500',
    [ConnectionStatus.Connecting]: 'text-yellow-400',
    [ConnectionStatus.Connected]: 'text-green-400',
    [ConnectionStatus.LoadingTokenizer]: 'text-yellow-400',
    [ConnectionStatus.InitDPF]: 'text-yellow-400',
    [ConnectionStatus.Prefilling]: 'text-blue-400',
    [ConnectionStatus.Generating]: 'text-blue-400',
    [ConnectionStatus.Error]: 'text-red-400',
  }[status];

  const statusDot = {
    [ConnectionStatus.Disconnected]: 'bg-gray-500',
    [ConnectionStatus.Connecting]: 'bg-yellow-400 animate-pulse',
    [ConnectionStatus.Connected]: 'bg-green-400',
    [ConnectionStatus.LoadingTokenizer]: 'bg-yellow-400 animate-pulse',
    [ConnectionStatus.InitDPF]: 'bg-yellow-400 animate-pulse',
    [ConnectionStatus.Prefilling]: 'bg-blue-400 animate-pulse',
    [ConnectionStatus.Generating]: 'bg-blue-400 animate-pulse',
    [ConnectionStatus.Error]: 'bg-red-400',
  }[status];

  return (
    <div className="flex items-center justify-between px-4 py-1.5 bg-gray-900 text-xs text-gray-400 border-t border-gray-800">
      <div className="flex items-center gap-1.5">
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${statusDot}`} />
        <span className={statusColor}>{status}</span>
      </div>
      {tokensPerSec !== null && (
        <span>{tokensPerSec.toFixed(1)} tok/s</span>
      )}
    </div>
  );
}
