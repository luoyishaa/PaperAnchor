import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { api, type Answer, type Candidate, type ConversationRecord, type ConversationTurn, type Evidence, type Paper, type PaperSearchPage, type ResearchSpace, type RetrievalProfile, type SemanticStatus, type VisualRegion } from './api'
import Reader from './Reader'
import AnswerText from './AnswerText'
import ReplySources from './ReplySources'
import DiscoveryPanel from './DiscoveryPanel'
import { loadRetrievalProfile, saveRetrievalProfile } from './retrievalPreferences'
import './App.css'

const retrievalProfileDescriptions: Partial<Record<RetrievalProfile, string>> = {
  evidence_minilm: 'Fastest response · MiniLM reranking',
  evidence_v2: 'Balanced speed and coverage · BGE reranking',
  evidence_optimized: 'Highest measured coverage · MiniLM + BGE fusion',
}

function App() {
  const [spaces, setSpaces] = useState<ResearchSpace[]>([])
  const [spacesError, setSpacesError] = useState('')
  const [spaceId, setSpaceId] = useState(1)
  const [papers, setPapers] = useState<Paper[]>([])
  const [questionPaperIds, setQuestionPaperIds] = useState<number[]>([])
  const [papersError, setPapersError] = useState('')
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null)
  const [page, setPage] = useState(1)
  const [focused, setFocused] = useState<Evidence | null>(null)
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<Answer | null>(null)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [conversations, setConversations] = useState<ConversationRecord[]>([])
  const [turns, setTurns] = useState<ConversationTurn[]>([])
  const [allowGeneralKnowledge, setAllowGeneralKnowledge] = useState(false)
  const [visualMode, setVisualMode] = useState<'auto' | 'off' | 'on'>('auto')
  const [retrievalProfile, setRetrievalProfile] = useState<RetrievalProfile>(loadRetrievalProfile)
  const [visualRegion, setVisualRegion] = useState<VisualRegion | null>(null)
  const [answerRevision, setAnswerRevision] = useState(0)
  const [busy, setBusy] = useState(false)
  const [semantic, setSemantic] = useState<SemanticStatus | null>(null)
  const optimizedAvailable = semantic?.retrieval_profiles?.includes('evidence_optimized') ?? false
  const effectiveRetrievalProfile = retrievalProfile === 'evidence_optimized' && !optimizedAvailable
    ? 'evidence_minilm' : retrievalProfile
  const [message, setMessage] = useState('')
  const [newSpace, setNewSpace] = useState('')
  const [spacePickerOpen, setSpacePickerOpen] = useState(false)
  const [spaceQuery, setSpaceQuery] = useState('')
  const [paperPickerMode, setPaperPickerMode] = useState<'sources' | 'add' | null>(null)
  const [pickerQuery, setPickerQuery] = useState('')
  const [pickerPage, setPickerPage] = useState(0)
  const [pickerSelected, setPickerSelected] = useState<number[]>([])
  const [pickerError, setPickerError] = useState('')
  const [pickerBusy, setPickerBusy] = useState(false)
  const [pickerLoading, setPickerLoading] = useState(false)
  const [availablePage, setAvailablePage] = useState<PaperSearchPage>({ items: [], total: 0 })
  const [discoveryQuery, setDiscoveryQuery] = useState('')
  const [discoverySource, setDiscoverySource] = useState<'arxiv' | 'semantic_scholar'>('arxiv')
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [discoveryBusy, setDiscoveryBusy] = useState(false)
  const [discoveryOpen, setDiscoveryOpen] = useState(false)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [libraryQuery, setLibraryQuery] = useState('')
  const [libraryPage, setLibraryPage] = useState(0)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [readerOpen, setReaderOpen] = useState(false)
  const [pendingQuestion, setPendingQuestion] = useState('')
  const uploadInput = useRef<HTMLInputElement>(null)
  const questionInput = useRef<HTMLTextAreaElement>(null)
  const chatScroll = useRef<HTMLDivElement>(null)
  const spaceDialog = useRef<HTMLDialogElement>(null)
  const paperDialog = useRef<HTMLDialogElement>(null)
  const answerRequest = useRef<AbortController | null>(null)
  const discoveryRequest = useRef<AbortController | null>(null)
  const navigationRevision = useRef(0)
  const libraryMatches = useMemo(() => papers.filter(paper =>
    paper.title.toLocaleLowerCase().includes(libraryQuery.trim().toLocaleLowerCase())),
  [papers, libraryQuery])
  const sourceMatches = useMemo(() => paperPickerMode === 'sources'
    ? papers.filter(paper => paper.title.toLocaleLowerCase().includes(pickerQuery.trim().toLocaleLowerCase()))
    : [], [papers, pickerQuery, paperPickerMode])

  useEffect(() => {
    const dialog = spaceDialog.current
    if (!dialog) return
    if (spacePickerOpen && !dialog.open) dialog.showModal()
    if (!spacePickerOpen && dialog.open) dialog.close()
  }, [spacePickerOpen])

  useEffect(() => {
    const dialog = paperDialog.current
    if (!dialog) return
    if (paperPickerMode && !dialog.open) dialog.showModal()
    if (!paperPickerMode && dialog.open) dialog.close()
  }, [paperPickerMode])

  useEffect(() => {
    if (paperPickerMode !== 'add') return
    let active = true
    const timer = window.setTimeout(() => {
      api.availablePapers(spaceId, pickerQuery, pickerPage * 50).then(value => {
        if (active) { setAvailablePage(value); setPickerError(''); setPickerLoading(false) }
      }).catch(error => {
        if (active) { setPickerError(error instanceof Error ? error.message : String(error)); setPickerLoading(false) }
      })
    }, 180)
    return () => { active = false; window.clearTimeout(timer) }
  }, [paperPickerMode, pickerQuery, pickerPage, spaceId])

  useEffect(() => {
    if (chatScroll.current) chatScroll.current.scrollTop = chatScroll.current.scrollHeight
  }, [turns, pendingQuestion, conversationId])

  useEffect(() => {
    if (discoveryOpen && chatScroll.current) chatScroll.current.scrollTop = 0
  }, [discoveryOpen])

  useEffect(() => {
    if (!sidebarOpen && !readerOpen) return
    const closeOverlay = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false)
        setReaderOpen(false)
      }
    }
    window.addEventListener('keydown', closeOverlay)
    return () => window.removeEventListener('keydown', closeOverlay)
  }, [sidebarOpen, readerOpen])

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
    let active = true
    const revision = navigationRevision.current
    api.conversations(spaceId).then(async chats => {
      if (!active || navigationRevision.current !== revision) return
      setConversations(chats)
      if (chats.length) {
        const history = await api.turns(chats[0].id, spaceId)
        if (active && navigationRevision.current === revision) {
          setConversationId(chats[0].id)
          setTurns(history)
          setAnswer(history.at(-1)?.answer || null)
        }
      }
    }).catch(() => {})
    return () => { active = false }
  }, [spaceId])

  useEffect(() => () => {
    answerRequest.current?.abort()
    discoveryRequest.current?.abort()
    navigationRevision.current += 1
  }, [])

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
    navigationRevision.current += 1
    cancelAnswer()
    cancelDiscovery()
    setCandidates([])
    setPaperPickerMode(null)
    setPickerError('')
    setSpaceId(nextId)
    setPapers([])
    setAnswer(null)
    setConversationId(null)
    setTurns([])
    setConversations([])
    setFocused(null)
    setSelectedPaper(null)
    setQuestionPaperIds([])
    setLibraryQuery('')
    setLibraryPage(0)
    setVisualRegion(null)
    setSidebarOpen(false)
  }

  function showEvidence(evidence: Evidence) {
    const paper = papers.find(item => item.id === evidence.paper_id)
    if (paper) {
      setSelectedPaper(paper)
      setPage(evidence.page_number)
      setFocused(evidence)
      setReaderOpen(true)
    }
  }

  async function createSpace(event: FormEvent) {
    event.preventDefault()
    if (!newSpace.trim()) return
    const revision = navigationRevision.current
    try {
      const space = await api.createSpace(newSpace.trim())
      const nextSpaces = await api.spaces()
      if (navigationRevision.current !== revision) return
      setSpaces(nextSpaces)
      changeSpace(space.id)
      setNewSpace('')
      setMessage(`Created ${space.name}.`)
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  async function upload(file: File) {
    if (busy) return
    const revision = navigationRevision.current
    setBusy(true)
    setMessage(`Importing ${file.name}…`)
    try {
      const result = await api.upload(file, spaceId)
      if (navigationRevision.current !== revision) return
      const [nextPapers, nextSpaces, status] = await Promise.all([
        api.papers(spaceId), api.spaces(), api.semanticStatus(),
      ])
      if (navigationRevision.current !== revision) return
      setPapers(nextPapers)
      setSpaces(nextSpaces)
      setSelectedPaper(result.paper)
      setReaderOpen(true)
      setPage(1)
      setFocused(null)
      setMessage(result.changed ? 'Paper added to this space.' : 'Paper already indexed; added to this space.')
      setSemantic(status)
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
    finally { setBusy(false); if (uploadInput.current) uploadInput.current.value = '' }
  }

  function cancelDiscovery() {
    discoveryRequest.current?.abort()
    discoveryRequest.current = null
    setDiscoveryBusy(false)
  }

  async function discoverPapers(event: FormEvent) {
    event.preventDefault()
    if (!discoveryQuery.trim()) return
    discoveryRequest.current?.abort()
    const controller = new AbortController()
    discoveryRequest.current = controller
    const revision = navigationRevision.current
    setDiscoveryBusy(true)
    setMessage(`Searching ${discoverySource === 'arxiv' ? 'arXiv' : 'Semantic Scholar'}…`)
    try {
      const found = await api.discover(discoveryQuery.trim(), discoverySource, controller.signal)
      if (controller.signal.aborted || discoveryRequest.current !== controller
          || navigationRevision.current !== revision) return
      setCandidates(found)
      setMessage(found.length ? `${found.length} candidate papers found. Review before importing.`
        : 'No papers found for this query.')
    } catch (error) {
      if (!controller.signal.aborted && discoveryRequest.current === controller
          && navigationRevision.current === revision) {
        setMessage(error instanceof Error ? error.message : String(error))
      }
    } finally {
      if (discoveryRequest.current === controller) {
        discoveryRequest.current = null
        setDiscoveryBusy(false)
      }
    }
  }

  async function importCandidate(candidate: Candidate) {
    if (busy || !candidate.arxiv_id) return
    const revision = navigationRevision.current
    setBusy(true)
    setMessage(`Importing ${candidate.title}…`)
    try {
      const result = await api.importCandidate(candidate.arxiv_id, spaceId)
      if (navigationRevision.current !== revision) return
      const [nextPapers, nextSpaces, status] = await Promise.all([
        api.papers(spaceId), api.spaces(), api.semanticStatus(),
      ])
      if (navigationRevision.current !== revision) return
      setPapers(nextPapers)
      setSpaces(nextSpaces)
      setSelectedPaper(result.paper)
      setReaderOpen(true)
      setPage(1)
      setSemantic(status)
      setMessage('Paper imported. Semantic indexing continues in the background.')
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
    finally { setBusy(false) }
  }

  function openPaperPicker(mode: 'sources' | 'add') {
    setPickerQuery('')
    setPickerPage(0)
    setPickerSelected(mode === 'sources' ? questionPaperIds : [])
    setPickerError('')
    setAvailablePage({ items: [], total: 0 })
    setPickerLoading(mode === 'add')
    setPaperPickerMode(mode)
    setSidebarOpen(false)
  }

  function togglePickerPaper(paperId: number) {
    setPickerSelected(ids => {
      if (ids.includes(paperId)) return ids.filter(id => id !== paperId)
      if (paperPickerMode === 'sources' && ids.length >= 12) {
        setPickerError('Choose up to 12 papers, or search the whole space.')
        return ids
      }
      setPickerError('')
      return [...ids, paperId]
    })
  }

  async function attachPapers(allMatching: boolean) {
    if (pickerBusy) return
    const revision = navigationRevision.current
    try {
      setPickerBusy(true)
      const result = await api.addPapersToSpace(spaceId,
        allMatching ? { query: pickerQuery } : { paper_ids: pickerSelected })
      if (navigationRevision.current !== revision) return
      const [nextPapers, nextSpaces] = await Promise.all([api.papers(spaceId), api.spaces()])
      if (navigationRevision.current !== revision) return
      setPapers(nextPapers)
      setSpaces(nextSpaces)
      setMessage(`${result.added} ${result.added === 1 ? 'paper' : 'papers'} added to this space.`)
      setPaperPickerMode(null)
    } catch (error) {
      if (navigationRevision.current === revision) setPickerError(error instanceof Error ? error.message : String(error))
    }
    finally { setPickerBusy(false) }
  }

  async function detachPaper(paperId: number) {
    const revision = navigationRevision.current
    try {
      await api.removeFromSpace(spaceId, paperId)
      if (navigationRevision.current !== revision) return
      const [nextPapers, nextSpaces] = await Promise.all([api.papers(spaceId), api.spaces()])
      if (navigationRevision.current !== revision) return
      setPapers(nextPapers)
      setSpaces(nextSpaces)
      if (selectedPaper?.id === paperId) setSelectedPaper(null)
      setQuestionPaperIds(ids => ids.filter(id => id !== paperId))
      setMessage('Paper removed from this space.')
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  async function openConversation(id: string) {
    const revision = ++navigationRevision.current
    cancelAnswer()
    try {
      const history = await api.turns(id, spaceId)
      if (navigationRevision.current !== revision) return
      setConversationId(id)
      setTurns(history)
      setAnswer(history.at(-1)?.answer || null)
      setQuestionPaperIds([])
      setVisualRegion(null)
      setSidebarOpen(false)
    } catch (error) {
      if (navigationRevision.current === revision) {
        setMessage(error instanceof Error ? error.message : String(error))
      }
    }
  }

  function newConversation() {
    navigationRevision.current += 1
    cancelAnswer()
    setConversationId(null)
    setTurns([])
    setAnswer(null)
    setQuestionPaperIds([])
    setVisualRegion(null)
    setQuestion('')
    setSidebarOpen(false)
    questionInput.current?.focus()
  }

  async function deletePaper(paper: Paper) {
    if (busy) return
    if (!window.confirm(`Delete "${paper.title}" from the entire local library? This also removes its index and annotations.`)) return
    const revision = navigationRevision.current
    setBusy(true)
    try {
      const result = await api.deletePaper(paper.id)
      if (navigationRevision.current !== revision) return
      const [nextPapers, nextSpaces, status] = await Promise.all([
        api.papers(spaceId), api.spaces(), api.semanticStatus(),
      ])
      if (navigationRevision.current !== revision) return
      setPapers(nextPapers)
      setSpaces(nextSpaces)
      setQuestionPaperIds(ids => ids.filter(id => id !== paper.id))
      if (selectedPaper?.id === paper.id) { setSelectedPaper(null); setFocused(null) }
      setAnswer(null)
      setSemantic(status)
      setMessage(result.file_was_managed && !result.file_removed
        ? 'Paper removed from the library. Its PDF copy could not be deleted because another program has it open.'
        : 'Paper deleted from the local library.')
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
    finally { setBusy(false) }
  }

  async function buildSemantic() {
    const revision = navigationRevision.current
    try {
      await api.buildSemantic()
      const status = await api.semanticStatus()
      if (navigationRevision.current !== revision) return
      setSemantic(status)
      setMessage('Semantic indexing started in the background.')
    } catch (error) {
      if (navigationRevision.current === revision) setMessage(error instanceof Error ? error.message : String(error))
    }
  }

  function cancelAnswer(restoreQuestion = false) {
    if (!answerRequest.current) return
    answerRequest.current.abort()
    answerRequest.current = null
    setBusy(false)
    setPendingQuestion('')
    if (restoreQuestion) setQuestion(current => current || pendingQuestion)
    setMessage('Stopped waiting. The server may still finish and save this answer.')
  }

  async function ask(event: FormEvent) {
    event.preventDefault()
    if (busy || !question.trim()) return
    navigationRevision.current += 1
    const controller = new AbortController()
    answerRequest.current = controller
    const asked = question.trim()
    setBusy(true)
    setAnswer(null)
    setFocused(null)
    setPendingQuestion(asked)
    setQuestion('')
    setMessage('Searching this space and preparing an answer…')
    try {
      const result = await api.answer(asked, spaceId, allowGeneralKnowledge,
        questionPaperIds, visualMode, visualRegion, conversationId, controller.signal,
        effectiveRetrievalProfile)
      if (controller.signal.aborted || answerRequest.current !== controller) return
      setAnswer(result)
      setVisualRegion(null)
      if (result.conversation_id) {
        setConversationId(result.conversation_id)
        const [history, chats] = await Promise.all([
          api.turns(result.conversation_id, spaceId), api.conversations(spaceId),
        ])
        if (controller.signal.aborted || answerRequest.current !== controller) return
        setTurns(history)
        setConversations(chats)
      }
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
    } catch (error) {
      if (controller.signal.aborted || answerRequest.current !== controller) return
      setQuestion(asked)
      setMessage(error instanceof Error ? error.message : String(error))
    } finally {
      if (answerRequest.current === controller) {
        answerRequest.current = null
        setBusy(false)
        setPendingQuestion('')
      }
    }
  }

  const activeConversationTitle = conversations.find(chat => chat.id === conversationId)?.title
  const pickerItems = paperPickerMode === 'sources'
    ? sourceMatches.slice(pickerPage * 50, (pickerPage + 1) * 50)
    : availablePage.items
  const pickerTotal = paperPickerMode === 'sources' ? sourceMatches.length : availablePage.total
  const pickerLastPage = Math.max(0, Math.ceil(pickerTotal / 50) - 1)
  const libraryItems = libraryMatches.slice(libraryPage * 50, (libraryPage + 1) * 50)

  return <div className="app-shell">
    <dialog ref={spaceDialog} className="picker-dialog space-dialog"
      aria-labelledby="space-dialog-title" onClose={() => setSpacePickerOpen(false)}>
      <div className="picker-heading"><div><h2 id="space-dialog-title">Research spaces</h2>
        <p>Switch the collection used for conversations and search.</p></div>
        <button type="button" aria-label="Close research spaces" onClick={() => setSpacePickerOpen(false)}>×</button></div>
      <input className="picker-search" aria-label="Search research spaces" type="search"
        placeholder="Find a research space…" value={spaceQuery}
        onChange={event => setSpaceQuery(event.target.value)} />
      <div className="space-options">
        {spaces.filter(space => space.name.toLocaleLowerCase().includes(spaceQuery.trim().toLocaleLowerCase()))
          .map(space => <button key={space.id} type="button"
            className={space.id === spaceId ? 'selected' : ''}
            onClick={() => { changeSpace(space.id); setSpacePickerOpen(false) }}>
            <strong>{space.name}</strong><span>{space.paper_count} papers</span>
          </button>)}
      </div>
      <button className="picker-secondary" type="button" onClick={() => {
        setSpacePickerOpen(false); setSettingsOpen(true); setSidebarOpen(true)
      }}>Create a research space</button>
    </dialog>

    <dialog ref={paperDialog} className="picker-dialog paper-dialog"
      aria-labelledby="paper-dialog-title" onClose={() => setPaperPickerMode(null)}>
      <div className="picker-heading"><div>
        <h2 id="paper-dialog-title">{paperPickerMode === 'add' ? 'Add existing papers' : 'Choose sources'}</h2>
        <p>{paperPickerMode === 'add'
          ? 'Search your library, select several papers, or add all matching results.'
          : 'Search this space and choose up to 12 papers. Leave it on all papers for broad questions.'}</p>
      </div><button type="button" aria-label="Close paper picker"
        onClick={() => setPaperPickerMode(null)}>×</button></div>
      <input className="picker-search" aria-label="Search papers" type="search"
        placeholder="Search paper titles…" value={pickerQuery}
        onChange={event => {
          setPickerQuery(event.target.value); setPickerPage(0)
          if (paperPickerMode === 'add') { setPickerLoading(true); setAvailablePage({ items: [], total: 0 }) }
        }} />
      <div className="picker-controls">
        <span>{pickerLoading ? 'Searching…' : pickerTotal + ' matching'} · {pickerSelected.length} selected</span>
        <button type="button" onClick={() => {
          setPickerSelected(ids => {
            const next = [...ids]
            for (const item of pickerItems) {
              if (next.includes(item.id)) continue
              if (paperPickerMode === 'sources' && next.length >= 12) break
              next.push(item.id)
            }
            return next
          })
        }} disabled={pickerLoading || pickerItems.length === 0}>Select visible</button>
        <button type="button" onClick={() => setPickerSelected([])}
          disabled={pickerSelected.length === 0}>Clear selection</button>
      </div>
      <div className="picker-list">
        {pickerItems.map(paper => <label className="picker-paper" key={paper.id}>
          <input type="checkbox" checked={pickerSelected.includes(paper.id)}
            onChange={() => togglePickerPaper(paper.id)} />
          <span><strong>{paper.title}</strong><small>{paper.page_count} pages</small></span>
        </label>)}
        {pickerItems.length === 0 && <div className="picker-empty">
          <p>{pickerLoading ? 'Searching library…'
            : paperPickerMode === 'add' ? 'No available papers match this search.'
              : papers.length === 0 ? 'This space has no papers yet.' : 'No papers match this search.'}</p>
          {!pickerLoading && paperPickerMode === 'sources' && papers.length === 0 &&
            (spaceId !== 1 ? <button className="picker-secondary" type="button"
              onClick={() => {
                setPickerQuery(''); setPickerSelected([]); setPickerLoading(true)
                setAvailablePage({ items: [], total: 0 }); setPaperPickerMode('add')
              }}>
              Add existing papers</button> : <label className="picker-upload">Add a PDF
              <input type="file" accept="application/pdf,.pdf" hidden
                onChange={event => {
                  const file = event.target.files?.[0]
                  if (file) { setPaperPickerMode(null); void upload(file) }
                }} /></label>)}
        </div>}
      </div>
      {pickerTotal > 50 && <div className="picker-pages">
        <button type="button" disabled={pickerPage === 0}
          onClick={() => { setPickerLoading(paperPickerMode === 'add'); setPickerPage(page => page - 1) }}>Previous</button>
        <span>Page {pickerPage + 1} of {pickerLastPage + 1}</span>
        <button type="button" disabled={pickerPage >= pickerLastPage}
          onClick={() => { setPickerLoading(paperPickerMode === 'add'); setPickerPage(page => page + 1) }}>Next</button>
      </div>}
      {pickerError && <p className="picker-error" role="alert">{pickerError}</p>}
      <div className="picker-actions">
        {paperPickerMode === 'sources' ? <>
          <button className="picker-secondary" type="button" onClick={() => {
            setQuestionPaperIds([]); setPaperPickerMode(null)
          }}>Use all {papers.length} papers</button>
          <button className="picker-primary" type="button" disabled={pickerSelected.length === 0}
            onClick={() => { setQuestionPaperIds(pickerSelected); setPaperPickerMode(null) }}>
            Ask from {pickerSelected.length} selected</button>
        </> : <>
          <button className="picker-secondary" type="button" disabled={pickerBusy || pickerLoading || pickerTotal === 0}
            onClick={() => void attachPapers(true)}>Add all {pickerTotal} matches</button>
          <button className="picker-primary" type="button" disabled={pickerBusy || pickerSelected.length === 0}
            onClick={() => void attachPapers(false)}>
            Add {pickerSelected.length} selected</button>
        </>}
      </div>
    </dialog>
    {sidebarOpen && <button className="mobile-scrim" type="button"
      aria-label="Close navigation" onClick={() => setSidebarOpen(false)} />}
    <aside className={'sidebar' + (sidebarOpen ? ' open' : '')} aria-label="Workspace navigation">
      <div className="sidebar-brand">
        <div className="brand-mark" aria-hidden="true">P</div>
        <div><strong>PaperAnchor</strong><small>Research, with a source.</small></div>
        <button className="sidebar-close" type="button" onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation">×</button>
      </div>

      <div className="sidebar-space">
        <span className="field-label">Research space</span>
        <button className="space-trigger" type="button" aria-haspopup="dialog"
          onClick={() => { setSpaceQuery(''); setSpacePickerOpen(true); setSidebarOpen(false) }}>
          <span>{spaces.find(space => space.id === spaceId)?.name || 'Choose a space'}
            <small>{spaces.find(space => space.id === spaceId)?.paper_count ?? 0} papers</small></span>
          <span aria-hidden="true">⌄</span>
        </button>
        {spacesError && <p className="error">{spacesError}</p>}
      </div>

      <button className="new-chat-button" type="button" onClick={newConversation}>
        <span aria-hidden="true">＋</span> New conversation
      </button>

      <section className="sidebar-section conversations-section">
        <div className="section-heading"><h2>Conversations</h2><span>{conversations.length}</span></div>
        <nav className="conversation-list" aria-label="Conversation history">
          {conversations.map(chat => <button key={chat.id} type="button"
            className={conversationId === chat.id ? 'active' : ''}
            aria-current={conversationId === chat.id ? 'page' : undefined}
            onClick={() => void openConversation(chat.id)}>
            <span className="conversation-icon" aria-hidden="true">□</span>
            <span className="conversation-copy"><strong>{chat.title}</strong>
              <small>{chat.turn_count} {chat.turn_count === 1 ? 'message' : 'messages'}</small></span>
          </button>)}
          {conversations.length === 0 && <p className="sidebar-empty">
            Your conversations will appear here.</p>}
        </nav>
      </section>

      <section className="sidebar-section library-section">
        <button className="sidebar-disclosure" type="button" aria-expanded={libraryOpen}
          aria-controls="library-content" onClick={() => setLibraryOpen(value => !value)}>
          <span>Paper library <em>{papers.length}</em></span><span aria-hidden="true">{libraryOpen ? '−' : '+'}</span>
        </button>
        {libraryOpen && <div id="library-content" className="library-content">
          <div className="paper-scope-summary">
            <span>{questionPaperIds.length
              ? questionPaperIds.length + ' selected for questions'
              : 'Searching this whole space'}</span>
            {questionPaperIds.length > 0 && <button type="button"
              onClick={() => setQuestionPaperIds([])}>Clear</button>}
          </div>
          {papersError && <p className="error">{papersError}</p>}
          {papers.length === 0 && <p className="sidebar-empty">
            Add a PDF to start asking grounded questions.</p>}
          {spaceId !== 1 && <button className="add-existing-button" type="button"
            onClick={() => openPaperPicker('add')}>＋ Add existing papers</button>}
          {papers.length > 10 && <input className="library-search" type="search"
            aria-label="Search papers in this space" placeholder="Find in this space…"
            value={libraryQuery} onChange={event => {
              setLibraryQuery(event.target.value); setLibraryPage(0)
            }} />}
          <div className="paper-list">{libraryItems.map(paper => <div key={paper.id}
            className={'paper-row' + (selectedPaper?.id === paper.id ? ' active' : '')}>
            <label className="paper-check" title="Include in question">
              <input type="checkbox" aria-label={'Include ' + paper.title + ' in question'}
                checked={questionPaperIds.includes(paper.id)}
                disabled={!questionPaperIds.includes(paper.id) && questionPaperIds.length >= 12}
                onChange={event => setQuestionPaperIds(ids => event.target.checked
                  ? [...ids, paper.id] : ids.filter(id => id !== paper.id))} />
            </label>
            <button className="paper-open" type="button" onClick={() => {
              setSelectedPaper(paper); setPage(1); setFocused(null)
              setReaderOpen(true); setSidebarOpen(false)
            }}>
              <strong>{paper.title}</strong>
              <small>{paper.year || 'Undated'} · {paper.page_count} pages
                {!paper.file_available && ' · PDF missing'}</small>
            </button>
            <details className="paper-menu">
              <summary aria-label={'Manage ' + paper.title}>···</summary>
              <div>
                {spaceId !== 1 && <button type="button"
                  onClick={() => void detachPaper(paper.id)}>Remove from space</button>}
                <button type="button" disabled={busy}
                  onClick={() => void deletePaper(paper)}>Delete everywhere</button>
              </div>
            </details>
          </div>)}</div>
          {libraryMatches.length > 50 && <div className="library-pages">
            <button type="button" disabled={libraryPage === 0}
              onClick={() => setLibraryPage(page => page - 1)}>Previous</button>
            <span>{libraryPage + 1} / {Math.ceil(libraryMatches.length / 50)}</span>
            <button type="button" disabled={(libraryPage + 1) * 50 >= libraryMatches.length}
              onClick={() => setLibraryPage(page => page + 1)}>Next</button>
          </div>}
        </div>}
      </section>

      <div className="sidebar-footer">
        <button type="button" aria-expanded={discoveryOpen} aria-controls="discovery-content"
          className={discoveryOpen ? 'active' : ''} onClick={() => {
          setDiscoveryOpen(true); setSidebarOpen(false)
        }}>Discover papers <span aria-hidden="true">↗</span></button>
        <button type="button" aria-expanded={settingsOpen} aria-controls="settings-content"
          onClick={() => setSettingsOpen(value => !value)}>
          Settings <span aria-hidden="true">{settingsOpen ? '−' : '+'}</span>
        </button>
      </div>
      {settingsOpen && <section id="settings-content" className="settings-content">
        <h2>Workspace settings</h2>
        <form className="space-form" onSubmit={createSpace}>
          <label htmlFor="new-space">New research space</label>
          <div><input id="new-space" placeholder="Space name" value={newSpace}
            onChange={event => setNewSpace(event.target.value)} maxLength={80} />
            <button disabled={!newSpace.trim()}>Create</button></div>
        </form>
        <label className="setting-option"><input type="checkbox" checked={allowGeneralKnowledge}
          onChange={event => setAllowGeneralKnowledge(event.target.checked)} />
          <span>Allow a labeled model answer when evidence is insufficient</span></label>
        <label className="setting-option"><input type="checkbox" checked={visualMode !== 'off'}
          onChange={event => setVisualMode(event.target.checked ? 'auto' : 'off')} />
          <span>Read figures and tables when asked</span></label>
        <label className="search-profile-option"><span>Text search</span>
          <select aria-label="Text search" value={effectiveRetrievalProfile} onChange={event => {
            const next = event.target.value as RetrievalProfile
            setRetrievalProfile(next)
            saveRetrievalProfile(next)
          }}>
            <option value="evidence_minilm">Fast · MiniLM</option>
            <option value="evidence_v2">Balanced · BGE</option>
            {optimizedAvailable && <option value="evidence_optimized">Quality · MiniLM + BGE</option>}
            <option value="legacy">Standard</option>
          </select>
        </label>
        {retrievalProfileDescriptions[effectiveRetrievalProfile] &&
          <small className="profile-description">
            {retrievalProfileDescriptions[effectiveRetrievalProfile]}
          </small>}
        {retrievalProfile === 'evidence_optimized' && semantic && !optimizedAvailable &&
          <small className="profile-unavailable">Optimized search is unavailable. Using Fast (MiniLM).
            {semantic.retrieval_profile_errors?.evidence_optimized &&
              <> {semantic.retrieval_profile_errors.evidence_optimized}</>}</small>}
        <div className="semantic-state">
          <strong>Local search index</strong>
          <small>{semantic
            ? semantic.indexed_passages + ' / ' + semantic.total_passages + ' passages indexed'
            : 'Checking index…'}</small>
          {semantic?.job?.state === 'failed' && <small className="error">
            {semantic.job.error || 'Indexing failed'}</small>}
          {semantic?.job?.state === 'queued' && <small>Indexing queued…</small>}
          {semantic?.job?.state === 'running' && <small>
            Processing {semantic.job.completed} / {semantic.job.total}</small>}
          {semantic && semantic.indexed_passages < semantic.total_passages &&
            !['queued', 'running'].includes(semantic.job?.state || '') &&
            <button type="button" onClick={() => void buildSemantic()}>Build index</button>}
        </div>
      </section>}
    </aside>

    <main className="chat-panel" inert={sidebarOpen}>
      <header className="chat-header">
        <button className="mobile-menu-button" type="button" aria-label="Open navigation"
          onClick={() => setSidebarOpen(true)}>☰</button>
        <div className="chat-title"><small>{spaces.find(space => space.id === spaceId)?.name || 'Research space'}</small>
          <strong>{activeConversationTitle || 'New conversation'}</strong></div>
        <div className="chat-header-actions">
          {(papers.length > 0 || turns.length > 0) && <button className="upload-button"
            type="button" disabled={busy} onClick={() => uploadInput.current?.click()}>
            Add PDF</button>}
          <input ref={uploadInput} type="file" accept="application/pdf,.pdf" hidden disabled={busy}
            onChange={event => { const file = event.target.files?.[0]; if (file) void upload(file) }} />
          <button className="reader-toggle" type="button" aria-expanded={readerOpen}
            onClick={() => setReaderOpen(value => !value)}>PDF preview</button>
        </div>
      </header>

      <div className="chat-scroll" ref={chatScroll}>
        {discoveryOpen && <DiscoveryPanel query={discoveryQuery} source={discoverySource}
          candidates={candidates} searching={discoveryBusy} importing={busy}
          onQuery={setDiscoveryQuery} onSource={setDiscoverySource}
          onSearch={discoverPapers} onImport={importCandidate}
          onClose={() => { cancelDiscovery(); setDiscoveryOpen(false) }} />}

        {!discoveryOpen && turns.length === 0 && !answer && !pendingQuestion && <div className="chat-empty">
          <div className="empty-mark" aria-hidden="true">P</div>
          <h1>{papers.length ? 'Read deeper. Ask better.' : 'Begin with a paper.'}</h1>
          <p>{papers.length
            ? 'Ask a question across your papers. Every supported answer leads back to its place in the PDF.'
            : 'Add a PDF or bring in papers from your library, then ask questions with page-level sources.'}</p>
          {papers.length === 0 && <div className="empty-actions">
            <button type="button" onClick={() => uploadInput.current?.click()}>Add a PDF</button>
            {spaceId !== 1 && <button type="button" onClick={() => openPaperPicker('add')}>
              Add existing papers</button>}
          </div>}
        </div>}

        <div className="message-list">
          {turns.map(turn => <article className="chat-turn" key={turn.id}>
            <div className="user-message"><span>You</span><p>{turn.question}</p></div>
            <div className="assistant-message">
              <div className="assistant-avatar" aria-hidden="true">P</div>
              <div className="assistant-content">
                <span className="message-author">PaperAnchor</span>
                <AnswerText answer={turn.answer} onEvidence={showEvidence} />
                <ReplySources answer={turn.answer} onEvidence={showEvidence} />
              </div>
            </div>
          </article>)}
          {turns.length === 0 && answer && <article className="chat-turn">
            <div className="assistant-message"><div className="assistant-avatar" aria-hidden="true">P</div>
              <div className="assistant-content"><span className="message-author">PaperAnchor</span>
                <AnswerText answer={answer} onEvidence={showEvidence} />
                <ReplySources answer={answer} onEvidence={showEvidence} /></div></div>
          </article>}
          {pendingQuestion && <article className="chat-turn pending-turn">
            <div className="user-message"><span>You</span><p>{pendingQuestion}</p></div>
            <div className="assistant-message"><div className="assistant-avatar" aria-hidden="true">P</div>
              <div className="assistant-content"><span className="message-author">PaperAnchor</span>
                <p className="thinking-text">Searching your papers…</p></div></div>
          </article>}
        </div>
      </div>

      <div className="composer-wrap">
        <div className="composer-context">
          <button type="button" onClick={() => openPaperPicker('sources')}>Sources: {questionPaperIds.length
            ? questionPaperIds.length + ' selected'
            : 'all ' + papers.length + ' papers'} <span aria-hidden="true">⌄</span></button>
          {visualRegion && <span className="region-chip">PDF region · page {visualRegion.page_number}
            <button type="button" onClick={() => setVisualRegion(null)}
              aria-label="Clear selected PDF region">×</button></span>}
        </div>
        <form className="ask-form" onSubmit={ask}>
          <textarea ref={questionInput} aria-label="Ask your papers" rows={2}
            placeholder="Ask a question about your papers…" value={question}
            onChange={event => setQuestion(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                event.preventDefault()
                event.currentTarget.form?.requestSubmit()
              }
            }} required />
          {pendingQuestion ? <button key="stop" type="button" aria-label="Cancel answer request"
            onClick={() => cancelAnswer(true)}>Stop</button>
            : <button key="send" type="submit" disabled={busy || !question.trim()} aria-label="Send question">
              <span aria-hidden="true">↑</span>
            </button>}
        </form>
        <div className="composer-meta"><span>Enter to send · Shift + Enter for a new line</span>
          <span role="status" aria-live="polite">{message}</span></div>
      </div>
    </main>

    <div className={'reader-dock' + (readerOpen ? ' open' : '')} inert={sidebarOpen}>
      <button className="reader-mobile-close" type="button"
        onClick={() => setReaderOpen(false)}>Close PDF preview</button>
      <Reader paper={selectedPaper} page={page} onPage={setPage}
        focused={focused} answerRevision={answerRevision}
        selectedRegion={visualRegion} onSelectRegion={setVisualRegion} />
    </div>
  </div>
}

export default App
