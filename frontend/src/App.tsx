import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api, type Answer, type Candidate, type Evidence, type Paper, type ResearchSpace, type SemanticStatus } from './api'
import Reader from './Reader'
import AnswerText from './AnswerText'
import './App.css'

function App() {
  const [spaces, setSpaces] = useState<ResearchSpace[]>([])
  const [spacesError, setSpacesError] = useState('')
  const [spaceId, setSpaceId] = useState(1)
  const [papers, setPapers] = useState<Paper[]>([])
  const [allPapers, setAllPapers] = useState<Paper[]>([])
  const [papersError, setPapersError] = useState('')
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const [page, setPage] = useState(1)
  const [focused, setFocused] = useState<Evidence | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [allowGeneralKnowledge, setAllowGeneralKnowledge] = useState(false)
  const [answerRevision, setAnswerRevision] = useState(0)
  const [busy, setBusy] = useState(false)
  const [semantic, setSemantic] = useState<SemanticStatus | null>(null)
  const [message, setMessage] = useState('')
  const [newSpace, setNewSpace] = useState('')
  const [existingPaperId, setExistingPaperId] = useState('')
  const [discoveryQuery, setDiscoveryQuery] = useState('')
  const [discoverySource, setDiscoverySource] = useState<'arxiv' | 'semantic_scholar'>('arxiv')
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [discoveryBusy, setDiscoveryBusy] = useState(false)
  const uploadInput = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let active = true
    api.spaces().then(value => { if (active) setSpaces(value) })
      .catch(error => { if (active) setSpacesError(String(error.message || error)) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    let active = true
    api.papers(spaceId).then(value => {
      if (active) { setPapers(value); setPapersError('') }
    }).catch(error => {
      if (active) setPapersError(String(error.message || error))
    })
    return () => { active = false }
  }, [spaceId])

  useEffect(() => {
    api.papers(1).then(setAllPapers).catch(() => {})
  }, [papers])

  useEffect(() => {
    let active = true
    const refresh = () => api.semanticStatus().then(value => {
      if (active) setSemantic(value)
    }).catch(() => {})
    void refresh()
    const timer = window.setInterval(refresh, 2500)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  function changeSpace(nextId: number) {
    setSpaceId(nextId)
    setPapers([])
    setAnswer(null)
    setFocused(null)
    setSelectedPaper(null)
  }

  function showEvidence(evidence: Evidence) {
    const paper = papers.find(item => item.id === evidence.paper_id)
    if (paper) {
      setSelectedPaper(paper)
      setPage(evidence.page_number)
      setFocused(evidence)
    }
  }

  async function createSpace(event: FormEvent) {
    event.preventDefault()
    if (!newSpace.trim()) return
    try {
      const space = await api.createSpace(newSpace.trim())
      setSpaces(await api.spaces())
      changeSpace(space.id)
      setNewSpace('')
      setMessage(`Created ${space.name}.`)
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
  }

  async function upload(file: File) {
    setBusy(true)
    setMessage(`Importing ${file.name}…`)
    try {
      const result = await api.upload(file, spaceId)
      setPapers(await api.papers(spaceId))
      setSpaces(await api.spaces())
      setSelectedPaper(result.paper)
      setPage(1)
      setFocused(null)
      setMessage(result.changed ? 'Paper added to this space.' : 'Paper already indexed; added to this space.')
      setSemantic(await api.semanticStatus())
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false); if (uploadInput.current) uploadInput.current.value = '' }
  }

  async function discoverPapers(event: FormEvent) {
    event.preventDefault()
    if (!discoveryQuery.trim()) return
    setDiscoveryBusy(true)
    setMessage(`Searching ${discoverySource === 'arxiv' ? 'arXiv' : 'Semantic Scholar'}…`)
    try {
      const found = await api.discover(discoveryQuery.trim(), discoverySource)
      setCandidates(found)
      setMessage(found.length ? `${found.length} candidate papers found. Review before importing.`
        : 'No papers found for this query.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setDiscoveryBusy(false) }
  }

  async function importCandidate(candidate: Candidate) {
    if (!candidate.arxiv_id) return
    setBusy(true)
    setMessage(`Importing ${candidate.title}…`)
    try {
      const result = await api.importCandidate(candidate.arxiv_id, spaceId)
      setPapers(await api.papers(spaceId))
      setSpaces(await api.spaces())
      setSelectedPaper(result.paper)
      setPage(1)
      setSemantic(await api.semanticStatus())
      setMessage('Paper imported. Semantic indexing continues in the background.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }

  async function attachPaper(paperId: number) {
    try {
      await api.addToSpace(spaceId, paperId)
      setPapers(await api.papers(spaceId))
      setSpaces(await api.spaces())
      setMessage('Paper added to this space.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
  }

  async function detachPaper(paperId: number) {
    try {
      await api.removeFromSpace(spaceId, paperId)
      setPapers(await api.papers(spaceId))
      setSpaces(await api.spaces())
      if (selectedPaper?.id === paperId) setSelectedPaper(null)
      setMessage('Paper removed from this space.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
  }

  async function buildSemantic() {
    try {
      await api.buildSemantic()
      setSemantic(await api.semanticStatus())
      setMessage('Semantic indexing started in the background.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
  }

  async function ask(event: FormEvent) {
    event.preventDefault()
    if (!question.trim()) return
    setBusy(true)
    setMessage('Searching this space and preparing an answer…')
    try {
      const result = await api.answer(question.trim(), spaceId, allowGeneralKnowledge)
      setAnswer(result)
      setAnswerRevision(value => value + 1)
      setMessage(result.status === 'answered'
        ? 'Select a citation to inspect its source.'
        : result.status === 'no_evidence'
          ? 'No matching paper evidence was found in this space.'
          : result.status === 'general_knowledge'
            ? 'This answer uses model knowledge; no paper source is attached.'
            : result.status === 'insufficient_evidence'
              ? 'The papers do not provide enough evidence yet.'
              : 'The answer needs citation review.')
    } catch (error) { setMessage(error instanceof Error ? error.message : String(error)) }
    finally { setBusy(false) }
  }

  return <>
    <header className="topbar">
      <div className="brand"><strong>PaperAnchor</strong><span>Read the evidence.</span></div>
      <label className="upload-button">Add PDF
        <input ref={uploadInput} type="file" accept="application/pdf,.pdf" hidden disabled={busy}
          onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file) }} />
      </label>
    </header>
    <main className="workspace">
      <aside className="library-panel">
        <h2>Research space</h2>
        <select aria-label="Research space" value={spaceId}
          onChange={event => changeSpace(Number(event.target.value))}>
          {spaces.map(space => <option key={space.id} value={space.id}>
            {space.name} ({space.paper_count})
          </option>)}
        </select>
        <form className="space-form" onSubmit={createSpace}>
          <input aria-label="New space name" placeholder="New space name" value={newSpace}
            onChange={event => setNewSpace(event.target.value)} maxLength={80} />
          <button disabled={!newSpace.trim()}>Create</button>
        </form>
        {spacesError && <p className="error">{spacesError}</p>}
        <div className="semantic-state">
          <strong>Semantic search</strong>
          <small>{semantic
            ? `${semantic.indexed_passages} / ${semantic.total_passages} passages indexed`
            : 'Checking index…'}</small>
          {semantic?.job?.state === 'failed' && <small className="error">
            {semantic.job.error || 'Indexing failed'}</small>}
          {semantic?.job?.state === 'queued' && <small>Indexing queued…</small>}
          {semantic?.job?.state === 'running' && <small>
            Processing {semantic.job.completed} / {semantic.job.total}</small>}
          {semantic && semantic.indexed_passages < semantic.total_passages &&
            !['queued', 'running'].includes(semantic.job?.state || '') &&
            <button onClick={() => void buildSemantic()}>Build semantic index</button>}
        </div>
        <div className="library-heading"><h2>Papers</h2><span>{papers.length}</span></div>
        <div className="timeline"><strong>Timeline</strong><span>{[...new Set(papers.map(paper => paper.year).filter(year => year !== null))]
          .sort((a, b) => a! - b!).map(year => <span key={year}>{year} · {papers.filter(paper => paper.year === year).length}</span>)}
          {papers.some(paper => paper.year === null) && <span>Undated · {papers.filter(paper => paper.year === null).length}</span>}</span></div>
        {spaceId !== 1 && <div className="attach-paper">
          <select aria-label="Add existing paper" value={existingPaperId}
            onChange={event => setExistingPaperId(event.target.value)}>
            <option value="">Add existing paper…</option>
            {allPapers.filter(paper => !papers.some(member => member.id === paper.id))
              .map(paper => <option key={paper.id} value={paper.id}>{paper.title}</option>)}
          </select>
          <button disabled={!existingPaperId} onClick={() => {
            void attachPaper(Number(existingPaperId)); setExistingPaperId('')
          }}>Add</button>
        </div>}
        {papersError && <p className="error">{papersError}</p>}
        {papers.length === 0 && <p className="muted">Upload a PDF to start researching this topic.</p>}
        <div className="paper-list">{papers.map(paper => <div key={paper.id}
          className={`paper-card ${selectedPaper?.id === paper.id ? 'active' : ''}`}>
          <button className="paper-open" onClick={() => { setSelectedPaper(paper); setPage(1); setFocused(null) }}>
            <strong>{paper.title}</strong>
            <small>{paper.year ? `${paper.year}${paper.year_basis === 'arxiv_filename' ? ' (arXiv ID)' : ''}` : 'Year unknown'} · {paper.page_count} pages · {paper.passage_count} passages
              {!paper.file_available && ' · PDF missing'}</small>
          </button>
          {spaceId !== 1 && <button className="remove-paper" title="Remove from this space"
            onClick={() => void detachPaper(paper.id)}>Remove</button>}
        </div>)}</div>
      </aside>
      <section className="research-panel">
        <details className="discovery-panel"><summary>Find papers online</summary>
          <form onSubmit={discoverPapers}>
            <input aria-label="Paper search" placeholder="Research topic or paper title" value={discoveryQuery}
              onChange={event => setDiscoveryQuery(event.target.value)} />
            <select aria-label="Search source" value={discoverySource}
              onChange={event => setDiscoverySource(event.target.value as 'arxiv' | 'semantic_scholar')}>
              <option value="arxiv">arXiv</option><option value="semantic_scholar">Semantic Scholar</option>
            </select>
            <button disabled={discoveryBusy || !discoveryQuery.trim()}>Search</button>
          </form>
          <div className="candidate-list">{candidates.map(candidate => <div
            key={`${candidate.source}:${candidate.source_id}`} className="candidate">
            <strong>{candidate.title}</strong>
            <small>{candidate.year || 'Year unknown'} · {candidate.venue || candidate.source.replace('_', ' ')}
              {candidate.citation_count !== null && ` · ${candidate.citation_count} citations`}</small>
            <p>{candidate.abstract || 'No abstract supplied.'}</p>
            <a href={candidate.paper_url} target="_blank" rel="noreferrer">View record ↗</a>
            {candidate.importable && candidate.arxiv_id && <button disabled={busy}
              onClick={() => void importCandidate(candidate)}>Import PDF</button>}
            {!candidate.importable && <small>Open the record and upload its PDF manually.</small>}
          </div>)}</div>
        </details>
        <div className="research-heading"><h1>Ask your papers</h1>
          <p>Answers connect back to the original PDF.</p></div>
        <form className="ask-form" onSubmit={ask}>
          <textarea aria-label="Question" placeholder="What do these papers say about…?"
            value={question} onChange={event => setQuestion(event.target.value)} required />
          <button disabled={busy || !question.trim()}>Ask</button>
          <label className="general-option"><input type="checkbox" checked={allowGeneralKnowledge}
            onChange={event => setAllowGeneralKnowledge(event.target.checked)} />
            Allow a clearly labeled model answer when paper evidence is insufficient
          </label>
        </form>
        <div className="status" role="status">{message}</div>
        {answer && <AnswerText answer={answer} onEvidence={showEvidence} />}
        <h2>Evidence</h2>
        <div className="evidence-list">{answer?.evidence.map(item => <button
          key={item.evidence_id} className="evidence-card" onClick={() => showEvidence(item)}>
          <strong>[{item.evidence_id}] {item.paper_title}</strong>
          <small>Page {item.page_number} · Open source location</small>
          <p>{item.text}</p>
        </button>)}</div>
      </section>
      <Reader paper={selectedPaper} page={page} onPage={setPage}
        focused={focused} answerRevision={answerRevision} />
    </main>
  </>
}

export default App
