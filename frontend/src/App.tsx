import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import './App.css';

type RetrievedChunk = {
  id: string;
  text: string;
};

type ChatResult = {
  question: string;
  answer: string;
  grade: string;
  retry_count: number;
  rewritten_question: string;
  retrieved_chunks: RetrievedChunk[];
  retrieved_count: number;
  outcome: 'grounded' | 'not_verified';
};

type ChatMessage =
  | { id: string; role: 'user'; content: string }
  | {
      id: string;
      role: 'assistant';
      content: string;
      result?: ChatResult;
      isError?: boolean;
    };

type HealthResponse = {
  status: string;
  knowledge_base_chunks: number;
};

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');

const suggestions = [
  'What is Machine Learning?',
  'Explain supervised and unsupervised learning.',
  'What is overfitting, and how can it be prevented?',
  'Who created Python and in which year?',
];

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [chunkCount, setChunkCount] = useState<number | null>(null);
  const [healthError, setHealthError] = useState('');
  const endOfMessagesRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  async function checkHealth() {
    try {
      const response = await fetch(`${API_BASE}/api/health`);
      if (!response.ok) {
        throw new Error(`Backend returned HTTP ${response.status}`);
      }
      const data = (await response.json()) as HealthResponse;
      setApiOnline(true);
      setChunkCount(data.knowledge_base_chunks);
      setHealthError('');
    } catch (error) {
      setApiOnline(false);
      setChunkCount(null);
      setHealthError(
        error instanceof Error ? error.message : 'Could not reach the backend.',
      );
    }
  }

  useEffect(() => {
    void checkHealth();
  }, []);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  async function sendQuestion(rawQuestion: string) {
    const cleanQuestion = rawQuestion.trim();
    if (!cleanQuestion || loading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: cleanQuestion,
    };
    setMessages((current) => [...current, userMessage]);
    setQuestion('');
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: cleanQuestion }),
      });

      const payload = await response.json();
      if (!response.ok) {
        const detail =
          typeof payload.detail === 'string'
            ? payload.detail
            : `Request failed with HTTP ${response.status}`;
        throw new Error(detail);
      }

      const result = payload as ChatResult;
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.answer,
          result,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            error instanceof Error
              ? error.message
              : 'The request could not be completed.',
          isError: true,
        },
      ]);
    } finally {
      setLoading(false);
      void checkHealth();
      textareaRef.current?.focus();
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendQuestion(question);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendQuestion(question);
    }
  }

  function startNewChat() {
    setMessages([]);
    setQuestion('');
    textareaRef.current?.focus();
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-lockup">
          <div className="brand-mark">R</div>
          <div>
            <div className="brand-name">RAG Studio</div>
            <div className="brand-subtitle">Self-Healing Demo</div>
          </div>
        </div>

        <button className="new-chat-button" onClick={startNewChat} type="button">
          <span className="plus-icon">＋</span>
          New conversation
        </button>

        <div className="sidebar-section-label">DEMO KNOWLEDGE BASE</div>
        <div className="knowledge-card">
          <div className="knowledge-icon">▤</div>
          <div className="knowledge-copy">
            <strong>Indexed documents</strong>
            <span>
              {chunkCount === null ? 'Checking collection…' : `${chunkCount} chunks available`}
            </span>
          </div>
          <span className={`tiny-status ${apiOnline ? 'is-online' : apiOnline === false ? 'is-offline' : ''}`} />
        </div>

        <div className="sidebar-section-label workflow-label">WORKFLOW</div>
        <div className="workflow-list">
          <div className="workflow-item"><span>01</span> Retrieve relevant chunks</div>
          <div className="workflow-item"><span>02</span> Generate an answer</div>
          <div className="workflow-item"><span>03</span> Grade grounding</div>
          <div className="workflow-item"><span>04</span> Rewrite and retry if needed</div>
        </div>

        <div className="sidebar-footer">
          <div className="connection-line">
            <span className={`connection-dot ${apiOnline ? 'is-online' : apiOnline === false ? 'is-offline' : ''}`} />
            <span>{apiOnline ? 'Backend connected' : apiOnline === false ? 'Backend unavailable' : 'Checking backend'}</span>
          </div>
          <span className="version-label">Local evaluation interface · v0.1</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>Knowledge assistant</strong></div>
          <div className="topbar-right">
            <span className="local-badge"><span /> Local demo</span>
            <button className="icon-button" type="button" onClick={() => void checkHealth()} aria-label="Refresh backend status" title="Refresh backend status">↻</button>
          </div>
        </header>

        <section className="content-area">
          <div className="page-heading">
            <div>
              <div className="eyebrow">DOCUMENT-GROUNDED AI</div>
              <h1>Knowledge assistant</h1>
              <p>Ask questions about your indexed documents and inspect how the RAG workflow responds.</p>
            </div>
            <div className={`backend-pill ${apiOnline ? 'online' : apiOnline === false ? 'offline' : ''}`}>
              <span className="connection-dot" />
              {apiOnline ? 'API online' : apiOnline === false ? 'API offline' : 'Connecting'}
            </div>
          </div>

          {apiOnline === false && (
            <div className="connection-alert" role="status">
              <strong>Cannot connect to the RAG backend.</strong>
              <span>{healthError || 'Start the FastAPI server, then refresh this page.'}</span>
              <code>uv run uvicorn api:api --reload --port 8000</code>
            </div>
          )}

          <div className="chat-card">
            <div className="chat-card-header">
              <div className="assistant-avatar">✳</div>
              <div className="chat-header-copy">
                <strong>RAG Assistant</strong>
                <span>Answers are generated from retrieved document context</span>
              </div>
              <button className="clear-button" onClick={startNewChat} type="button">Clear chat</button>
            </div>

            <div className="conversation" aria-live="polite">
              {messages.length === 0 ? (
                <div className="welcome-state">
                  <div className="welcome-symbol">✳</div>
                  <h2>What would you like to know?</h2>
                  <p>Ask a question about Machine Learning or Python using the documents in your knowledge base.</p>
                  <div className="suggestion-grid">
                    {suggestions.map((item) => (
                      <button
                        className="suggestion-card"
                        key={item}
                        type="button"
                        onClick={() => setQuestion(item)}
                        disabled={loading || apiOnline === false}
                      >
                        <span>{item}</span>
                        <span className="suggestion-arrow">↗</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="message-list">
                  {messages.map((message) => (
                    <article className={`message-row ${message.role}`} key={message.id}>
                      <div className={`message-avatar ${message.role}`}>
                        {message.role === 'user' ? 'Y' : '✳'}
                      </div>
                      <div className="message-body">
                        <div className="message-author">{message.role === 'user' ? 'You' : 'RAG Assistant'}</div>
                        <div className={`message-bubble ${ "isError" in message && message.isError ? 'error-bubble' : 'chat-message'}`}>
                          {message.content}
                        </div>
                        {message.role === 'assistant' && message.result && (
                          <div className="result-details">
                            <div className="result-metrics">
                              <span className={`grade-chip ${message.result.grade === 'PASS' ? 'pass' : 'fail'}`}>
                                {message.result.grade === 'PASS' ? '✓ Grounding check passed' : '! Not verified'}
                              </span>
                              <span className="metric-chip">{message.result.retry_count} retries</span>
                              <span className="metric-chip">{message.result.retrieved_count} chunks retrieved</span>
                            </div>
                            {message.result.rewritten_question && (
                              <details className="detail-block">
                                <summary>Rewritten retrieval query</summary>
                                <p>{message.result.rewritten_question}</p>
                              </details>
                            )}
                            <details className="detail-block">
                              <summary>Retrieved context ({message.result.retrieved_count})</summary>
                              {message.result.retrieved_chunks.length > 0 ? (
                                <div className="chunk-list">
                                  {message.result.retrieved_chunks.map((chunk) => (
                                    <div className="chunk-card" key={chunk.id}>
                                      <span>{chunk.id}</span>
                                      <p>{chunk.text}</p>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="empty-context">No chunks were returned by the retriever.</p>
                              )}
                            </details>
                            <p className="grader-note">The PASS/FAIL label is produced by the project's LLM grader; it is an evaluation signal, not a guarantee of factual correctness.</p>
                          </div>
                        )}
                      </div>
                    </article>
                  ))}
                  {loading && (
                    <div className="message-row assistant">
                      <div className="message-avatar assistant">✳</div>
                      <div className="message-body">
                        <div className="message-author">RAG Assistant</div>
                        <div className="typing-bubble"><span /><span /><span /> <em>Running retrieval and answer evaluation…</em></div>
                      </div>
                    </div>
                  )}
                  <div ref={endOfMessagesRef} />
                </div>
              )}
            </div>

            <form className="composer" onSubmit={handleSubmit}>
              <textarea
                ref={textareaRef}
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents…"
                rows={2}
                maxLength={2000}
                disabled={loading || apiOnline === false}
                aria-label="Ask a question"
              />
              <div className="composer-bottom">
                <span>Enter to send · Shift + Enter for a new line</span>
                <button className="send-button" type="submit" disabled={loading || !question.trim() || apiOnline === false}>
                  {loading ? 'Working…' : 'Send question'} <span>↑</span>
                </button>
              </div>
            </form>
          </div>

          <div className="disclaimer">Responses are limited to the documents indexed in this demo. Always check the retrieved context for important information.</div>
        </section>
      </main>
    </div>
  );
}

export default App;
