import type { FormEvent } from 'react'
import type { Candidate } from './api'

export default function DiscoveryPanel({ query, source, candidates, searching, importing,
  onQuery, onSource, onSearch, onImport, onClose }: {
  query: string
  source: Candidate['source']
  candidates: Candidate[]
  searching: boolean
  importing: boolean
  onQuery: (query: string) => void
  onSource: (source: Candidate['source']) => void
  onSearch: (event: FormEvent) => void
  onImport: (candidate: Candidate) => Promise<void>
  onClose: () => void
}) {
  return <section id="discovery-content" className="discovery-panel">
    <div className="discovery-heading"><div><h2>Discover papers</h2>
      <p>Find a candidate, inspect its record, then import its PDF.</p></div>
      <button type="button" aria-label="Close discovery" onClick={onClose}>×</button></div>
    <form onSubmit={onSearch}>
      <div className="discovery-search-row">
        <input aria-label="Paper search" placeholder="Topic, title, or author…"
          value={query} onChange={event => onQuery(event.target.value)} />
        <button disabled={searching || !query.trim()}>
          {searching ? 'Searching…' : 'Search'}</button>
      </div>
      <fieldset className="discovery-sources"><legend>Search in</legend>
        <label><input type="radio" name="discovery-source" value="arxiv"
          checked={source === 'arxiv'}
          onChange={() => onSource('arxiv')} /><span>arXiv</span></label>
        <label><input type="radio" name="discovery-source" value="semantic_scholar"
          checked={source === 'semantic_scholar'}
          onChange={() => onSource('semantic_scholar')} /><span>Semantic Scholar</span></label>
      </fieldset>
    </form>
    <div className="candidate-list">{candidates.map(candidate => <article
      key={candidate.source + ':' + candidate.source_id} className="candidate">
      <strong>{candidate.title}</strong>
      <small>{candidate.year || 'Year unknown'} ·
        {' ' + (candidate.venue || candidate.source.replace('_', ' '))}
        {candidate.citation_count !== null && ' · ' + candidate.citation_count + ' citations'}</small>
      <p>{candidate.abstract || 'No abstract supplied.'}</p>
      <div className="candidate-actions">
        <a href={candidate.paper_url} target="_blank" rel="noreferrer">View record ↗</a>
        {candidate.importable && candidate.arxiv_id && <button type="button" disabled={importing}
          onClick={() => void onImport(candidate)}>Import PDF</button>}
        {!candidate.importable && <small>Upload its PDF manually.</small>}
      </div>
    </article>)}</div>
  </section>
}
