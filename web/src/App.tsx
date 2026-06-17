import { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import MessageInput from './components/MessageInput';
import StatusBar from './components/StatusBar';
import ProtocolInspector from './components/ProtocolInspector';
import DesktopBadge from './components/DesktopBadge';
import LocalChatPanel from './components/LocalChatPanel';
import MemoryPanel from './components/MemoryPanel';
import ImagePanel from './components/ImagePanel';
import InpaintPanel from './components/InpaintPanel';
import VisionPanel from './components/VisionPanel';
import TrainPanel from './components/TrainPanel';
import FeedbackPanel from './components/FeedbackPanel';
import IncognitoToggle from './components/IncognitoToggle';
import LanToggle from './components/LanToggle';
import { usePrivateChat } from './hooks/usePrivateChat';
import { isRemote, isTauri, setRemotePin, imageSrc, type StoredImage } from './lib/tauri';
import { setInpaintSource, fetchImageBytes } from './lib/inpaintHandoff';

type View = 'chat' | 'local' | 'image' | 'inpaint' | 'vision' | 'train' | 'feedback' | 'memory' | 'inspector';

export default function App() {
  const desktop = isTauri();
  const remote = isRemote();
  const capable = desktop || remote;

  // Auto-capture PIN from the URL query param so the user doesn't have
  // to enter it manually. The LAN toggle shows a URL like
  // `https://192.168.1.42:3100?pin=482937`.
  if (remote) {
    const params = new URLSearchParams(window.location.search);
    const pin = params.get('pin');
    if (pin) setRemotePin(pin);
  }
  const wsUrl = import.meta.env.VITE_SERVER_A_WS
    || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`;
  const [showThinking, setShowThinking] = useState(false);
  // Default the desktop/remote build to local chat; plain browser keeps 2PC.
  const [view, setView] = useState<View>(capable ? 'local' : 'chat');

  const {
    messages,
    streamingText,
    generating,
    tokensPerSec,
    connectionStatus,
    sendMessage,
    protocolEvents,
  } = usePrivateChat({ wsUrl, showThinking });

  const tabs: { id: View; label: string; show: boolean; badge?: number }[] = [
    { id: 'chat', label: '2PC Chat', show: true },
    { id: 'local', label: 'Local', show: capable },
    { id: 'image', label: 'Image', show: capable },
    { id: 'inpaint', label: 'Inpaint', show: capable },
    { id: 'vision', label: 'Classify', show: capable },
    { id: 'train', label: 'Train', show: capable },
    { id: 'feedback', label: 'Feedback', show: capable },
    { id: 'memory', label: 'Memory', show: capable },
    { id: 'inspector', label: 'Inspector', show: true, badge: protocolEvents.length },
  ];

  return (
    <div className="flex flex-col h-screen bg-gray-800 text-gray-100">
      <header className="flex flex-col sm:flex-row sm:items-center sm:justify-between px-4 py-2 sm:py-3 bg-gray-900 border-b border-gray-700 gap-2">
        <div className="flex items-center justify-between">
          <h1 className="text-lg font-semibold">Klearu</h1>
          {/* Mobile: controls go here on the right of the title row */}
          <div className="flex items-center gap-2 sm:hidden">
            <LanToggle />
            <IncognitoToggle />
            <DesktopBadge />
          </div>
        </div>
        <div className="flex items-center gap-3 min-w-0">
          {/* Tab bar: horizontally scrollable on mobile */}
          <div className="flex bg-gray-800 rounded-lg p-0.5 overflow-x-auto no-scrollbar min-w-0">
            {tabs.filter(t => t.show).map((t) => (
              <button
                key={t.id}
                onClick={() => setView(t.id)}
                className={`px-3 py-1 rounded-md text-xs font-medium transition-colors whitespace-nowrap flex-shrink-0 ${
                  view === t.id ? 'bg-gray-600 text-gray-100' : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {t.label}
                {t.badge && t.badge > 0 ? (
                  <span className="ml-1.5 px-1.5 py-0.5 bg-gray-700 rounded-full text-[10px] text-gray-300">
                    {t.badge}
                  </span>
                ) : null}
              </button>
            ))}
          </div>
          {view === 'chat' && (
            <button
              onClick={() => setShowThinking((v) => !v)}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                showThinking ? 'bg-purple-900/50 text-purple-300' : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
              }`}
            >
              <span className={`inline-block w-2 h-2 rounded-full transition-colors ${
                showThinking ? 'bg-purple-400' : 'bg-gray-500'
              }`} />
              Thinking
            </button>
          )}
          {view === 'chat' && (
            <span className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium bg-green-900/50 text-green-300">
              <span className="inline-block w-2 h-2 rounded-full bg-green-400" />
              Private 2PC
            </span>
          )}
          {/* Desktop: controls here after the tab bar */}
          <div className="hidden sm:flex items-center gap-2">
            <LanToggle />
            <IncognitoToggle />
            <DesktopBadge />
          </div>
        </div>
      </header>

      {view === 'chat' && (
        <>
          <ChatWindow
            messages={messages}
            streamingText={streamingText}
            generating={generating}
            showThinking={showThinking}
          />
          <MessageInput onSend={sendMessage} disabled={generating} />
          <StatusBar status={connectionStatus} tokensPerSec={tokensPerSec} />
        </>
      )}
      {view === 'local' && capable && <LocalChatPanel />}
      {view === 'image' && capable && (
        <ImagePanel
          onUseAsInput={async (img: StoredImage) => {
            try {
              const src = await imageSrc(img.path);
              const bytes = await fetchImageBytes(src);
              setInpaintSource(bytes);
              setView('inpaint');
            } catch (e) {
              console.warn('use-as-input failed', e);
            }
          }}
        />
      )}
      {view === 'inpaint' && capable && <InpaintPanel />}
      {view === 'vision' && capable && <VisionPanel />}
      {view === 'train' && capable && <TrainPanel />}
      {view === 'feedback' && capable && <FeedbackPanel />}
      {view === 'memory' && capable && <MemoryPanel />}
      {view === 'inspector' && <ProtocolInspector events={protocolEvents} />}
    </div>
  );
}
