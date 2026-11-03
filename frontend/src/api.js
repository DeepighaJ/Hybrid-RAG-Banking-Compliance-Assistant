// Thin client for the FastAPI backend (api.py).
// The base URL comes from VITE_API_URL: .env.development locally, the Azure URL in production.

const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

/** True once the API is up and the pipeline is loaded. Also wakes a sleeping server. */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_URL}/health`)
    if (!response.ok) return false
    const data = await response.json()
    return Boolean(data.pipeline_loaded)
  } catch {
    return false
  }
}

/** Ask a question. Resolves to { answer, sources } or throws an ApiError with a user-facing message. */
export async function askQuestion(question) {
  let response
  try {
    response = await fetch(`${API_URL}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new ApiError(
      `Can't reach the assistant at ${API_URL}. Check that the API is running, then ask again.`,
      0,
    )
  }

  if (response.ok) return response.json()

  const messages = {
    422: 'Questions need to be between 3 and 500 characters.',
    429: 'The demo question limit has been reached. Wait a minute, then ask again.',
    503: 'The assistant is still starting. Ask again in a few seconds.',
  }
  throw new ApiError(
    messages[response.status] ?? "The assistant couldn't answer that question. Ask again, or rephrase it.",
    response.status,
  )
}
