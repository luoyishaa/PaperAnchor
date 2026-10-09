import type { Answer, Evidence } from './api'

export default function ReplySources({ answer, onEvidence }: {
  answer: Answer, onEvidence: (evidence: Evidence) => void
}) {
  const citedIds = new Set(answer.cited_evidence_ids)
  const cited = answer.evidence.filter(item => citedIds.has(item.evidence_id))
  const additional = answer.evidence.filter(item => !citedIds.has(item.evidence_id))
  const card = (item: Evidence) => <button key={item.evidence_id} type="button"
    className="evidence-card" onClick={() => onEvidence(item)}>
    <strong>[{item.evidence_id}] {item.paper_title}</strong>
    <small>Page {item.page_number} · View in PDF ↗</small>
  </button>
  return <>
    {cited.length > 0 && <div className="evidence-list">{cited.map(card)}</div>}
    {additional.length > 0 && <details className="evidence-disclosure">
      <summary>{additional.length} more retrieved {additional.length === 1 ? 'source' : 'sources'}</summary>
      <div className="evidence-list">{additional.map(card)}</div>
    </details>}
  </>
}
