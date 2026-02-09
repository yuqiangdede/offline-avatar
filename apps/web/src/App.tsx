// File: offline-avatar/apps/web/src/App.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import ChatPanel, { type ChatItem } from './components/ChatPanel';
import InputBar from './components/InputBar';
import VideoPanel from './components/VideoPanel';
import { blobToBase64 } from './lib/audio';
import { WebRTCClient } from './lib/webrtc';
import { WSClient, type WSMessage } from './lib/ws';

const STORAGE_KEY = 'offline-avatar-chat-history';
const DEFAULT_WS_URL = 'ws://localhost:8000/ws';

type Phase = 'idle' | 'recording' | 'thinking' | 'speaking';

function loadInitialMessages(): ChatItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as ChatItem[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toPhase(value: unknown): Phase {
  if (value === 'recording' || value === 'thinking' || value === 'speaking') {
    return value;
  }
  return 'idle';
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WSClient | null>(null);
  const rtcRef = useRef<WebRTCClient | null>(null);

  const [connected, setConnected] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [collapsed, setCollapsed] = useState(false);
  const [metrics, setMetrics] = useState<Record<string, number>>({});
  const [messages, setMessages] = useState<ChatItem[]>(() => loadInitialMessages());
  const [mediaBlocked, setMediaBlocked] = useState(false);

  const wsUrl = useMemo(() => {
    const custom = import.meta.env.VITE_WS_URL;
    return custom && typeof custom === 'string' ? custom : DEFAULT_WS_URL;
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-200)));
  }, [messages]);

  const ensurePlayback = useCallback(async () => {
    const el = videoRef.current;
    if (!el || !el.srcObject) {
      return;
    }
    try {
      await el.play();
      setMediaBlocked(false);
    } catch {
      setMediaBlocked(true);
    }
  }, []);

  useEffect(() => {
    const ws = new WSClient();
    wsRef.current = ws;

    const off = ws.onMessage(async (message: WSMessage) => {
      const type = String(message.type || '');

      if (type === 'webrtc.answer') {
        const sdp = String(message.sdp || '');
        const sdpType = (message.sdpType || 'answer') as RTCSdpType;
        await rtcRef.current?.handleAnswer(sdp, sdpType);
        return;
      }

      if (type === 'webrtc.ice') {
        const candidate = (message.candidate || null) as RTCIceCandidateInit | null;
        await rtcRef.current?.addIce(candidate);
        return;
      }

      if (type === 'state') {
        setPhase(toPhase(message.phase));
        if (message.phase === 'speaking') {
          void ensurePlayback();
        }
        return;
      }

      if (type === 'chat.append') {
        const role = message.role === 'assistant' ? 'assistant' : 'user';
        const lang = message.lang === 'en' ? 'en' : 'zh';
        const text = String(message.text || '');
        const ts = Number(message.ts || Date.now());

        setMessages((prev) => [
          ...prev,
          {
            id: `${ts}-${prev.length}-${Math.random().toString(16).slice(2, 8)}`,
            role,
            text,
            lang,
            ts,
          },
        ]);
        return;
      }

      if (type === 'metric') {
        const name = String(message.name || 'unknown');
        const value = Number(message.value || 0);
        setMetrics((prev) => ({ ...prev, [name]: value }));
      }
    });

    let canceled = false;

    ws.connect(wsUrl)
      .then(async () => {
        if (canceled) {
          return;
        }
        setConnected(true);

        const rtc = new WebRTCClient(
          (payload) => ws.send(payload),
          (stream) => {
            if (!videoRef.current) {
              return;
            }
            if (videoRef.current.srcObject !== stream) {
              videoRef.current.srcObject = stream;
            }
            void ensurePlayback();
          },
        );

        rtcRef.current = rtc;
        await rtc.start();
      })
      .catch(() => {
        if (!canceled) {
          setConnected(false);
        }
      });

    return () => {
      canceled = true;
      off();
      rtcRef.current?.close();
      ws.close();
      wsRef.current = null;
      rtcRef.current = null;
      setConnected(false);
    };
  }, [wsUrl, ensurePlayback]);

  useEffect(() => {
    const unlock = () => {
      void ensurePlayback();
    };
    window.addEventListener('pointerdown', unlock);
    window.addEventListener('keydown', unlock);
    return () => {
      window.removeEventListener('pointerdown', unlock);
      window.removeEventListener('keydown', unlock);
    };
  }, [ensurePlayback]);

  const sendText = (text: string) => {
    void ensurePlayback();
    wsRef.current?.send({ type: 'input.text', text });
  };

  const sendAudio = async (blob: Blob, format: 'webm_opus') => {
    await ensurePlayback();
    const data_base64 = await blobToBase64(blob);
    wsRef.current?.send({
      type: 'input.audio',
      format,
      data_base64,
    });
  };

  const clearChat = () => {
    wsRef.current?.send({ type: 'chat.clear' });
    setMessages([]);
  };

  return (
    <div className="app-shell">
      <div className="main-row">
        <VideoPanel
          videoRef={videoRef}
          phase={phase}
          connected={connected}
          mediaBlocked={mediaBlocked}
          onUnlock={() => void ensurePlayback()}
        />
        <ChatPanel
          messages={messages}
          collapsed={collapsed}
          metrics={metrics}
          onToggle={() => setCollapsed((prev) => !prev)}
          onClear={clearChat}
        />
      </div>

      <InputBar
        onSendText={sendText}
        onSendAudio={sendAudio}
        onLocalPhaseChange={(localPhase) => setPhase(localPhase)}
      />
    </div>
  );
}
