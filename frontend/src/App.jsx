import { useEffect, useRef, useState } from 'react'
import { askQuestion, checkHealth } from './api'
import './App.css'

const STARTER_QUESTIONS = [
  'What is the beneficial ownership threshold for High-risk customers?',
  'What does policy CDD-5.4 say?',
  'What weighted risk score makes a customer High risk, and how often must they be reviewed?',
  'A trust connected to a Medium-rated jurisdiction uses private banking and was onboarded face-to-face. What is its risk score and tier?',
  'What interest rate does Eastmere Bank pay on savings accounts?',
]

const DOCUMENT_NAMES = {
  kyc_faq: 'KYC and CDD Procedures FAQ',
  aml_risk_policy: 'AML Risk Assessment Policy',
}

// Policy codes such as CDD-4.2 or RA-3.5. The model sometimes writes a
// non-breaking hyphen (U+2011) instead of "-", so any dash variant is accepted.
const CODE_PATTERN = /\b(?:CDD|RA)[-\u2010-\u2015]\d+(?:\.\d+)?\b/g
const normalizeDashes = (text) => text.replace(/[\u2010-\u2015]/g, '-')

const sourceCites = (source, code) => Boolean(code) && normalizeDashes(source.text).includes(code)

/** Split one line of answer text into plain text and clickable policy-code chips. */
function LineWithCodes({ line, activeCode, onCodeClick }) {
  const parts = []
  let lastIndex = 0
  for (const match of line.matchAll(CODE_PATTERN)) {
    if (match.index > lastIndex) parts.push(line.slice(lastIndex, match.index))
    const code = normalizeDashes(match[0])
    parts.push(
      <button
        key={`${code}-${match.index}`}
        type="button"
        className={`code-chip${activeCode === code ? ' is-active' : ''}`}
        onClick={() => onCodeClick(code)}
        aria-pressed={activeCode === code}
        title={`Show the source for ${code}`}
      >
        {code}
      </button>,
    )
    lastIndex = match.index + match[0].length
  }
  if (lastIndex < line.length) parts.push(line.slice(lastIndex))
  return parts
}

/** Render the model's plain-text answer: paragraphs, "- " bullet lists, and code chips. */
function AnswerText({ text, activeCode, onCodeClick }) {
  const blocks = []
  let listItems = []
  const flushList = () => {
    if (listItems.length) {
      blocks.push(<ul key={`list-${blocks.length}`}>{listItems}</ul>)
      listItems = []
    }
  }

  text.split('\n').forEach((rawLine, index) => {
    const line = rawLine.trim()
    if (!line) {
      flushList()
      return
    }
    const bullet = line.match(/^[-•*]\s+(.*)$/)
    const content = (
      <LineWithCodes line={bullet ? bullet[1] : line} activeCode={activeCode} onCodeClick={onCodeClick} />
    )
    if (bullet) {
      listItems.push(<li key={index}>{content}</li>)
    } else {
      flushList()
      blocks.push(<p key={index}>{content}</p>)
    }
  })
  flushList()
  return <div className="answer-text">{blocks}</div>
}

function sourceLocation(source) {
  if (source.question_number) {
    const section = source.section?.replace(/^Section [A-Z]:\s*/, '')
    return section ? `Question ${source.question_number}, ${section}` : `Question ${source.question_number}`
  }
  const heading = source.text.match(/RA-\d+\.\d+ [^\n]+/)
  return heading ? heading[0] : 'Policy text'
}

function SourceCard({ source, rank, cited }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = source.text.length > 360
  const shown = expanded || !isLong ? source.text : `${source.text.slice(0, 360).trimEnd()}…`
  const [firstLine, ...rest] = shown.split('\n')
  // Policy chunks often start with their RA-x.x heading, already shown as the location line.
  const headingIsLocation = !source.question_number && firstLine === sourceLocation(source)

  return (
    <article className={`source-card${cited ? ' is-cited' : ''}`}>
      <header className="source-header">
        <span className="source-rank" aria-label={`Source ${rank}`}>
          {rank}
        </span>
        <div>
          <p className="source-doc">{DOCUMENT_NAMES[source.doc_type] ?? source.source_file}</p>
          <p className="source-location">
            {sourceLocation(source)}
            {source.page ? `, page ${source.page}` : ''}
          </p>
        </div>
      </header>
      <blockquote className="source-text">
        {source.question_number && <strong>{firstLine}</strong>}
        {!source.question_number && !headingIsLocation && `${firstLine} `}
        {rest.join(' ')}
      </blockquote>
      {isLong && (
        <button type="button" className="text-button" onClick={() => setExpanded((value) => !value)}>
          {expanded ? 'Show less' : 'Show full excerpt'}
        </button>
      )}
    </article>
  )
}

function SourceList({ sources, activeCode }) {
  return (
    <div className="source-list">
      {sources.map((source, index) => (
        <SourceCard
          key={`${source.source_file}-${source.question_number}-${index}`}
          source={source}
          rank={index + 1}
          cited={sourceCites(source, activeCode)}
        />
      ))}
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [isAsking, setIsAsking] = useState(false)
  const [apiReady, setApiReady] = useState(null) // null while the first check runs
  const [activeAnswerId, setActiveAnswerId] = useState(null)
  const [activeCode, setActiveCode] = useState(null)
  const conversationEnd = useRef(null)
  const sourcePanel = useRef(null)

  // Ping the API on load (this also wakes a scaled-to-zero server), and keep
  // checking every 5 seconds until it answers.
  useEffect(() => {
    let cancelled = false
    let timer
    const poll = async () => {
      const ready = await checkHealth()
      if (cancelled) return
      setApiReady(ready)
      if (!ready) timer = setTimeout(poll, 5000)
    }
    poll()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, isAsking])

  // When a code chip is selected, bring the first matching source into view:
  // in the side panel on wide screens, or under the answer on narrow ones.
  useEffect(() => {
    if (!activeCode) return
    const panelVisible = sourcePanel.current?.offsetParent !== null
    const scope = panelVisible ? sourcePanel.current : document.querySelector('.message-answer.is-active')
    scope?.querySelector('.source-card.is-cited')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [activeCode, activeAnswerId])

  const activeAnswer = messages.find((message) => message.id === activeAnswerId)

  async function submit(text) {
    const trimmed = text.trim()
    if (trimmed.length < 3 || isAsking) return

    setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'user', text: trimmed }])
    setQuestion('')
    setIsAsking(true)
    setActiveCode(null)

    try {
      const result = await askQuestion(trimmed)
      const answer = { id: crypto.randomUUID(), role: 'assistant', text: result.answer, sources: result.sources }
      setMessages((current) => [...current, answer])
      setActiveAnswerId(answer.id)
      setApiReady(true)
    } catch (error) {
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: 'error', text: error.message }])
    } finally {
      setIsAsking(false)
    }
  }

  function handleCodeClick(messageId, code) {
    const sameChip = activeCode === code && activeAnswerId === messageId
    setActiveAnswerId(messageId)
    setActiveCode(sameChip ? null : code)
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit(question)
    }
  }

  const statusKey = apiReady === null ? 'checking' : apiReady ? 'ready' : 'waiting'
  const statusLabel = { checking: 'Connecting', ready: 'Connected', waiting: 'Waiting for the API' }[statusKey]

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">E</span>
          <div>
            <p className="brand-name">Eastmere Bank</p>
            <p className="brand-sub">Financial crime policy assistant</p>
          </div>
        </div>
        <p className={`status status-${statusKey}`}>
          <span className="status-dot" aria-hidden="true" />
          {statusLabel}
        </p>
      </header>

      <main className="workspace">
        <section className="conversation" aria-label="Conversation">
          <div className="messages" aria-live="polite">
            {messages.length === 0 && (
              <div className="intro">
                <h1>Ask the bank&rsquo;s KYC and AML policies</h1>
                <p>
                  Answers come only from two documents: the KYC and CDD Procedures FAQ and the AML Risk Assessment
                  Policy. Each answer shows the policy text it used, and the policy codes in an answer take you to
                  their source.
                </p>
                <p className="intro-try">Try one of these:</p>
                <ul className="starters">
                  {STARTER_QUESTIONS.map((starter) => (
                    <li key={starter}>
                      <button type="button" onClick={() => submit(starter)} disabled={isAsking}>
                        {starter}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {messages.map((message) => {
              if (message.role === 'user') {
                return (
                  <div key={message.id} className="message message-user">
                    <p>{message.text}</p>
                  </div>
                )
              }
              if (message.role === 'error') {
                return (
                  <div key={message.id} className="message message-error" role="alert">
                    <p>{message.text}</p>
                  </div>
                )
              }
              const isActive = message.id === activeAnswerId
              return (
                <div key={message.id} className={`message message-answer${isActive ? ' is-active' : ''}`}>
                  <AnswerText
                    text={message.text}
                    activeCode={isActive ? activeCode : null}
                    onCodeClick={(code) => handleCodeClick(message.id, code)}
                  />
                  <div className="answer-footer">
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => {
                        setActiveAnswerId(message.id)
                        setActiveCode(null)
                      }}
                      aria-pressed={isActive}
                    >
                      {isActive
                        ? `Showing ${message.sources.length} sources`
                        : `View ${message.sources.length} sources`}
                    </button>
                  </div>
                  {/* On narrow screens the side panel is hidden, so sources sit under the answer. */}
                  {isActive && (
                    <details className="inline-sources" key={activeCode ?? 'closed'} open={Boolean(activeCode)}>
                      <summary>Sources used</summary>
                      <SourceList sources={message.sources} activeCode={activeCode} />
                    </details>
                  )}
                </div>
              )
            })}

            {isAsking && (
              <div className="message message-pending" role="status">
                <span className="pending-bar" aria-hidden="true" />
                Searching both policies for the relevant passages
              </div>
            )}
            <div ref={conversationEnd} />
          </div>

          <form
            className="composer"
            onSubmit={(event) => {
              event.preventDefault()
              submit(question)
            }}
          >
            <label htmlFor="question" className="visually-hidden">
              Your question
            </label>
            <textarea
              id="question"
              rows={2}
              maxLength={500}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about due diligence, risk scoring, PEPs, reviews…"
            />
            <div className="composer-actions">
              <span className="char-count">{question.length}/500</span>
              <button type="submit" className="ask-button" disabled={isAsking || question.trim().length < 3}>
                {isAsking ? 'Asking…' : 'Ask'}
              </button>
            </div>
          </form>
        </section>

        <aside className="source-panel" aria-label="Sources for the selected answer" ref={sourcePanel}>
          <h2>Sources</h2>
          {activeAnswer ? (
            <>
              <p className="panel-note">
                {activeCode
                  ? `Highlighted: passages that contain ${activeCode}.`
                  : 'Ranked by relevance. Select a policy code in the answer to find it here.'}
              </p>
              <SourceList sources={activeAnswer.sources} activeCode={activeCode} />
            </>
          ) : (
            <p className="panel-note">The policy passages behind each answer appear here.</p>
          )}
        </aside>
      </main>

      <footer className="disclaimer">
        Eastmere Bank is fictional. Its policies were written for this demo and are not legal or compliance advice.
      </footer>
    </div>
  )
}
