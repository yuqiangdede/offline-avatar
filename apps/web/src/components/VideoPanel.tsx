// File: offline-avatar/apps/web/src/components/VideoPanel.tsx
import type { RefObject } from 'react';

interface VideoPanelProps {
  videoRef: RefObject<HTMLVideoElement>;
  phase: 'idle' | 'recording' | 'thinking' | 'speaking';
  connected: boolean;
  mediaBlocked: boolean;
  onUnlock: () => void;
}

export default function VideoPanel({ videoRef, phase, connected, mediaBlocked, onUnlock }: VideoPanelProps) {
  return (
    <section className="video-panel">
      <video ref={videoRef} autoPlay playsInline controls={false} className="avatar-video" />
      <div className="video-overlay">
        <span className={`status-chip status-${phase}`}>{phase}</span>
        <span className={`status-chip ${connected ? 'status-ok' : 'status-bad'}`}>
          {connected ? 'ws:connected' : 'ws:disconnected'}
        </span>
      </div>
      {mediaBlocked ? (
        <button className="media-unlock-btn" onClick={onUnlock}>
          点击启用音视频
        </button>
      ) : null}
    </section>
  );
}
