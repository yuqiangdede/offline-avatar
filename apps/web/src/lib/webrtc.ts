// File: offline-avatar/apps/web/src/lib/webrtc.ts
export type SignalSender = (payload: Record<string, unknown>) => void;
export type StreamHandler = (stream: MediaStream) => void;

export class WebRTCClient {
  private pc: RTCPeerConnection;
  private remoteStream: MediaStream;
  private sendSignal: SignalSender;
  private onStream: StreamHandler;
  private closed = false;
  private restarting = false;
  private lastRestartAt = 0;

  constructor(sendSignal: SignalSender, onStream: StreamHandler) {
    this.sendSignal = sendSignal;
    this.onStream = onStream;
    this.remoteStream = new MediaStream();
    this.pc = this.createPeerConnection();
    this.onStream(this.remoteStream);
  }

  async start(): Promise<void> {
    if (this.closed) {
      return;
    }
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    this.sendSignal({
      type: 'webrtc.offer',
      sdp: offer.sdp,
      sdpType: offer.type,
    });
  }

  async handleAnswer(sdp: string, sdpType: RTCSdpType): Promise<void> {
    if (this.closed || this.pc.signalingState === 'closed') {
      return;
    }
    try {
      await this.pc.setRemoteDescription({ sdp, type: sdpType });
    } catch (err) {
      console.warn('[webrtc] setRemoteDescription failed', err);
    }
  }

  async addIce(candidate: RTCIceCandidateInit | null): Promise<void> {
    if (this.closed || this.pc.signalingState === 'closed') {
      return;
    }
    try {
      await this.pc.addIceCandidate(candidate ?? null);
    } catch (err) {
      console.warn('[webrtc] addIceCandidate failed', err, candidate);
    }
  }

  close(): void {
    this.closed = true;
    this.pc.close();
  }

  private createPeerConnection(): RTCPeerConnection {
    const pc = new RTCPeerConnection({ iceServers: [] });
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });

    pc.onicecandidate = (event) => {
      this.sendSignal({
        type: 'webrtc.ice',
        candidate: event.candidate ? event.candidate.toJSON() : null,
      });
    };

    pc.ontrack = (event) => {
      const track = event.track;
      console.info('[webrtc] ontrack', track.kind, track.id, track.readyState);
      this.remoteStream.addTrack(track);
      this.onStream(this.remoteStream);
    };

    pc.oniceconnectionstatechange = () => {
      const state = pc.iceConnectionState;
      console.info('[webrtc] iceConnectionState', state);
      if (state === 'disconnected' || state === 'failed') {
        void this.restartIce(`ice=${state}`);
      } else if (state === 'closed' && !this.closed) {
        void this.restartPeerConnection('ice=closed');
      }
    };

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      console.info('[webrtc] connectionState', state);
      if (state === 'disconnected' || state === 'failed') {
        void this.restartIce(`conn=${state}`);
      } else if (state === 'closed' && !this.closed) {
        void this.restartPeerConnection('conn=closed');
      }
    };

    return pc;
  }

  private canRestartNow(): boolean {
    const now = Date.now();
    if (now - this.lastRestartAt < 3000) {
      return false;
    }
    this.lastRestartAt = now;
    return true;
  }

  private async restartIce(reason: string): Promise<void> {
    if (this.closed || this.restarting || this.pc.signalingState === 'closed') {
      return;
    }
    if (!this.canRestartNow()) {
      return;
    }

    let needHardRestart = false;
    this.restarting = true;
    try {
      console.warn('[webrtc] restarting ICE:', reason);
      this.pc.restartIce();
      const offer = await this.pc.createOffer({ iceRestart: true });
      await this.pc.setLocalDescription(offer);
      this.sendSignal({
        type: 'webrtc.offer',
        sdp: offer.sdp,
        sdpType: offer.type,
      });
    } catch (err) {
      console.warn('[webrtc] restartIce failed, fallback to hard restart', err);
      needHardRestart = true;
    } finally {
      this.restarting = false;
    }

    if (needHardRestart) {
      await this.restartPeerConnection(`${reason}:fallback`);
    }
  }

  private async restartPeerConnection(reason: string): Promise<void> {
    if (this.closed || this.restarting) {
      return;
    }
    if (!this.canRestartNow()) {
      return;
    }

    this.restarting = true;
    try {
      console.warn('[webrtc] hard restart peer connection:', reason);
      try {
        this.pc.close();
      } catch {
        // no-op
      }

      this.remoteStream = new MediaStream();
      this.onStream(this.remoteStream);
      this.pc = this.createPeerConnection();

      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);
      this.sendSignal({
        type: 'webrtc.offer',
        sdp: offer.sdp,
        sdpType: offer.type,
      });
    } finally {
      this.restarting = false;
    }
  }
}
