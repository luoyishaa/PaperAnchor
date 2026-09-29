import type { ReactNode } from 'react'
import type { Answer, Evidence } from './api'

export default function AnswerText({ answer, onEvidence }: {
  answer: Answer, onEvidence: (evidence: Evidence) => void
}) {
  const evidence = new Map(answer.evidence.map(item => [item.evidence_id, item]))
  const parts: ReactNode[] = answer.text.split(/(\[E\d+\])/g).map((part, index) => {
    const item = evidence.get(part.slice(1, -1))
    return item && /^\[E\d+\]$/.test(part)
      ? <button className="citation" key={index} onClick={() => onEvidence(item)}>{part}</button>
      : <span key={index}>{part}</span>
  })
  return <article className="answer"><div className="answer-status">{answer.status.replaceAll('_', ' ')}</div>
    <div className="answer-basis">{answer.basis === 'papers' ? 'Based on your papers'
      : answer.basis === 'model' ? 'Model knowledge · not verified by your papers'
        : 'No supported answer from your papers'}</div>
    <div>{parts}</div>
    {answer.follow_up_question && <p className="follow-up">To narrow the search: {answer.follow_up_question}</p>}
    {answer.attempted_queries.length > 1 && <small>Searches tried: {answer.attempted_queries.join(' · ')}</small>}
  </article>
}
