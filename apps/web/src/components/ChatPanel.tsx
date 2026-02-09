// File: offline-avatar/apps/web/src/components/ChatPanel.tsx
export interface ChatItem {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  lang: 'zh' | 'en';
  ts: number;
}

interface ChatPanelProps {
  messages: ChatItem[];
  collapsed: boolean;
  metrics: Record<string, number>;
  onToggle: () => void;
  onClear: () => void;
}

function formatTime(ts: number): string {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(
    d.getSeconds(),
  ).padStart(2, '0')}`;
}

export default function ChatPanel({ messages, collapsed, metrics, onToggle, onClear }: ChatPanelProps) {
  return (
    <aside className={`chat-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="chat-header">
        <strong>Chat Log</strong>
        <button className="ghost-btn" onClick={onToggle}>
          {collapsed ? '展开' : '折叠'}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="metric-row">
            <span>e2e: {metrics.e2e_ms ?? '-'}ms</span>
            <span>asr: {metrics.asr_ms ?? '-'}ms</span>
            <span>llm: {metrics.llm_ms ?? '-'}ms</span>
            <span>tts: {metrics.tts_ms ?? '-'}ms</span>
            <span>avatar: {metrics.avatar_ms ?? '-'}ms</span>
          </div>

          <div className="chat-list">
            {messages.map((item) => (
              <div key={item.id} className={`bubble ${item.role}`}>
                <div className="bubble-meta">
                  <span>{item.role}</span>
                  <span>{item.lang}</span>
                  <span>{formatTime(item.ts)}</span>
                </div>
                <div className="bubble-text">{item.text}</div>
              </div>
            ))}
          </div>

          <div className="chat-footer">
            <button className="danger-btn" onClick={onClear}>
              Clear
            </button>
          </div>
        </>
      )}
    </aside>
  );
}
