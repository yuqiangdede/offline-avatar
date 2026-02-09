// File: offline-avatar/apps/web/src/components/InputBar.tsx
import { useRef, useState } from 'react';

import { AudioRecorder, type RecorderFormat } from '../lib/audio';

interface InputBarProps {
  onSendText: (text: string) => void;
  onSendAudio: (blob: Blob, format: RecorderFormat) => Promise<void>;
  onLocalPhaseChange?: (phase: 'idle' | 'recording' | 'thinking') => void;
}

export default function InputBar({ onSendText, onSendAudio, onLocalPhaseChange }: InputBarProps) {
  const [text, setText] = useState('');
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState(false);
  const recorderRef = useRef(new AudioRecorder());

  const doSendText = () => {
    const value = text.trim();
    if (!value) {
      return;
    }
    onSendText(value);
    setText('');
  };

  const toggleRecord = async () => {
    if (busy) {
      return;
    }

    if (!recording) {
      try {
        await recorderRef.current.start();
        setRecording(true);
        onLocalPhaseChange?.('recording');
      } catch {
        onLocalPhaseChange?.('idle');
      }
      return;
    }

    try {
      setBusy(true);
      const result = await recorderRef.current.stop();
      setRecording(false);
      onLocalPhaseChange?.('thinking');
      await onSendAudio(result.blob, result.format);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="input-bar">
      <button className={`record-btn ${recording ? 'on' : ''}`} onClick={toggleRecord} disabled={busy}>
        {recording ? '停止录音' : '开始录音'}
      </button>

      <input
        className="text-input"
        placeholder="输入文本，Enter 发送"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            doSendText();
          }
        }}
      />

      <button className="send-btn" onClick={doSendText}>
        发送
      </button>
    </div>
  );
}
