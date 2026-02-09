// File: offline-avatar/apps/web/src/lib/audio.ts
export type RecorderFormat = 'webm_opus';

export interface RecorderResult {
  blob: Blob;
  format: RecorderFormat;
}

export class AudioRecorder {
  private mediaStream: MediaStream | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  async start(): Promise<void> {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      return;
    }

    this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const preferredMime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';

    this.chunks = [];
    this.mediaRecorder = new MediaRecorder(this.mediaStream, {
      mimeType: preferredMime,
    });

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        this.chunks.push(event.data);
      }
    };

    this.mediaRecorder.start(100);
  }

  stop(): Promise<RecorderResult> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) {
        reject(new Error('Recorder is not initialized'));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
        const blob = new Blob(this.chunks, { type: mimeType });
        this.mediaStream?.getTracks().forEach((track) => track.stop());
        this.mediaRecorder = null;
        this.mediaStream = null;
        this.chunks = [];
        resolve({ blob, format: 'webm_opus' });
      };

      this.mediaRecorder.stop();
    });
  }
}

export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const raw = String(reader.result || '');
      const idx = raw.indexOf(',');
      resolve(idx >= 0 ? raw.slice(idx + 1) : raw);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}
