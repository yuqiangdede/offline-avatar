// File: offline-avatar/apps/web/src/lib/ws.ts
export type WSMessage = Record<string, unknown>;

type MessageHandler = (message: WSMessage) => void;

export class WSClient {
  private socket: WebSocket | null = null;
  private handlers: Set<MessageHandler> = new Set();

  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);
      this.socket = ws;

      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error('WebSocket connection failed'));
      ws.onmessage = (event) => {
        try {
          const parsed = JSON.parse(String(event.data)) as WSMessage;
          this.handlers.forEach((handler) => handler(parsed));
        } catch {
          // Ignore malformed payloads.
        }
      };
    });
  }

  send(message: WSMessage): void {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify(message));
  }

  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  close(): void {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) {
      this.socket.close();
    }
    this.socket = null;
  }
}
