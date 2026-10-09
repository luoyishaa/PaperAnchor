import { useEffect, useRef, useState, type FormEvent, type PointerEvent } from 'react'
import { api, type Annotation, type Evidence, type PageGeometry, type Paper, type VisualRegion } from './api'

function Highlight({ box, geometry, kind }: {
  box: [number, number, number, number], geometry: PageGeometry,
  kind: 'evidence' | 'note' | 'selection'
}) {
  const [x0, y0, x1, y1] = box
  if (x1 <= x0 || y1 <= y0) return null
  return <div className={`highlight ${kind}`} style={{
    left: `${x0 / geometry.width * 100}%`,
    top: `${y0 / geometry.height * 100}%`,
    width: `${(x1 - x0) / geometry.width * 100}%`,
    height: `${(y1 - y0) / geometry.height * 100}%`,
  }} />
}

export default function Reader({ paper, page, onPage, focused, answerRevision,
  selectedRegion, onSelectRegion }: {
  paper: Paper | null, page: number, onPage: (page: number) => void,
  focused: Evidence | null, answerRevision: number,
  selectedRegion: VisualRegion | null,
  onSelectRegion: (region: VisualRegion | null) => void
}) {
  const location = paper ? `${paper.id}:${page}` : 'empty'
  const [geometry, setGeometry] = useState<{
    location: string, value: PageGeometry | null, error: string
  }>({ location: 'empty', value: null, error: '' })
  const [notes, setNotes] = useState<{
    location: string, value: Annotation[], error: string
  }>({ location: 'empty', value: [], error: '' })
  const [noteText, setNoteText] = useState('')
  const [noteError, setNoteError] = useState('')
  const [saving, setSaving] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [drag, setDrag] = useState<{start: [number, number], end: [number, number]} | null>(null)
  const [interactionLocation, setInteractionLocation] = useState(location)
  if (interactionLocation !== location) {
    setInteractionLocation(location)
    setNoteError('')
    setDrag(null)
    setSelecting(false)
  }
  const currentGeometry = geometry.location === location ? geometry.value : null
  const currentNotes = notes.location === location ? notes.value : []
  const noteLocation = useRef<string | null>(location)
  const noteRevision = useRef(0)

  useEffect(() => {
    noteLocation.current = location
    noteRevision.current += 1
    return () => { noteLocation.current = null }
  }, [location])

  function pointerPosition(event: PointerEvent<HTMLDivElement>): [number, number] {
    const rect = event.currentTarget.getBoundingClientRect()
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]
  }

  function finishSelection(event: PointerEvent<HTMLDivElement>) {
    if (!drag || !currentGeometry || !paper) return
    const end = pointerPosition(event)
    setDrag(null)
    const x0 = Math.min(drag.start[0], end[0])
    const y0 = Math.min(drag.start[1], end[1])
    const x1 = Math.max(drag.start[0], end[0])
    const y1 = Math.max(drag.start[1], end[1])
    if (x1 - x0 < 0.02 || y1 - y0 < 0.02) return
    onSelectRegion({ paper_id: paper.id, page_number: page,
      bbox: [x0 * currentGeometry.width, y0 * currentGeometry.height,
        x1 * currentGeometry.width, y1 * currentGeometry.height] })
    setSelecting(false)
  }

  useEffect(() => {
    if (!paper?.file_available) return
    let active = true
    api.pageGeometry(paper.id, page).then(value => {
      if (active) setGeometry({ location, value, error: '' })
    }).catch(error => {
      if (active) setGeometry({ location, value: null, error: String(error.message || error) })
    })
    return () => { active = false }
  }, [paper, page, location])

  useEffect(() => {
    if (!paper?.file_available) return
    let active = true
    api.annotations(paper.id, page).then(value => {
      if (active) setNotes({ location, value, error: '' })
    }).catch(error => {
      if (active) setNotes({ location, value: [], error: String(error.message || error) })
    })
    return () => { active = false }
  }, [paper, page, location, answerRevision])

  async function saveNote(event: FormEvent) {
    event.preventDefault()
    if (saving || !paper?.file_available || !noteText.trim()) return
    const draft = noteText
    const revision = noteRevision.current
    setSaving(true)
    try {
      const box = focused?.paper_id === paper.id && focused.page_number === page
        ? focused.bbox : [0, 0, 0, 0] as [number, number, number, number]
      await api.addNote(paper.id, page, draft.trim(), box)
      if (noteLocation.current !== location || noteRevision.current !== revision) return
      const value = await api.annotations(paper.id, page)
      if (noteLocation.current !== location || noteRevision.current !== revision) return
      setNotes({ location, value, error: '' })
      setNoteText(current => current === draft ? '' : current)
      setNoteError('')
    } catch (error) {
      if (noteLocation.current === location && noteRevision.current === revision) {
        setNoteError(error instanceof Error ? error.message : String(error))
      }
    } finally {
      setSaving(false)
    }
  }

  return <section className="reader-panel">
    <div className="reader-toolbar">
      <div className="reader-name">
        <strong>{paper?.title || 'Select a paper'}</strong>
        {paper && <span>{paper.page_count} pages</span>}
      </div>
      <div className="page-controls">
        <button aria-label="Previous page" disabled={!paper?.file_available || page <= 1}
          onClick={() => onPage(page - 1)}>‹</button>
        <span>{paper ? `${page} / ${paper.page_count}` : '–'}</span>
        <button aria-label="Next page" disabled={!paper?.file_available || page >= paper.page_count}
          onClick={() => onPage(page + 1)}>›</button>
        {paper?.file_available && <a href={`/api/papers/${paper.id}/file#page=${page}`}
          target="_blank" rel="noreferrer">PDF ↗</a>}
        {paper?.file_available && <button type="button" aria-label="Select PDF region"
          className={selecting ? 'active' : ''}
          onClick={() => { setSelecting(value => !value); setDrag(null) }}>
          {selecting ? 'Cancel selection' : 'Select region'}
        </button>}
      </div>
    </div>
    <div className="reader-scroll">
      {!paper && <p className="empty-reader">Choose a paper or select a citation to inspect its source.</p>}
      {paper && !paper.file_available && <p className="empty-reader">This PDF is no longer at its indexed path. Add the file again to read it.</p>}
      {paper?.file_available && !(geometry.location === location && geometry.error) && <div className="page-frame" key={location}>
        <img src={`/api/papers/${paper.id}/pages/${page}/image`}
          alt={`Page ${page} of ${paper.title}`} />
        {currentGeometry && <div className={`page-overlay ${selecting ? 'selecting' : ''}`}
          onPointerDown={event => {
            if (!selecting) return
            event.currentTarget.setPointerCapture(event.pointerId)
            const start = pointerPosition(event)
            setDrag({ start, end: start })
          }}
          onPointerMove={event => {
            if (drag) setDrag({ ...drag, end: pointerPosition(event) })
          }}
          onPointerUp={finishSelection}>
          {currentNotes.map(note => <Highlight key={note.id} box={note.bbox}
            geometry={currentGeometry} kind="note" />)}
          {focused?.paper_id === paper.id &&
            (focused.source_spans?.length ? focused.source_spans : [focused])
              .filter(span => span.page_number === page &&
                span.bbox[2] > span.bbox[0] && span.bbox[3] > span.bbox[1])
              .map((span, index) => <Highlight key={`${span.page_number}-${index}`}
                box={span.bbox} geometry={currentGeometry} kind="evidence" />)}
          {selectedRegion?.paper_id === paper.id && selectedRegion.page_number === page &&
            <Highlight box={selectedRegion.bbox} geometry={currentGeometry} kind="selection" />}
          {drag && <Highlight box={[
            Math.min(drag.start[0], drag.end[0]) * currentGeometry.width,
            Math.min(drag.start[1], drag.end[1]) * currentGeometry.height,
            Math.max(drag.start[0], drag.end[0]) * currentGeometry.width,
            Math.max(drag.start[1], drag.end[1]) * currentGeometry.height,
          ]} geometry={currentGeometry} kind="selection" />}
        </div>}
      </div>}
      {geometry.location === location && geometry.error && <p className="error">{geometry.error}</p>}
    </div>
    <div className="notes-panel">
      <h2>Notes <span>({currentNotes.length})</span></h2>
      {notes.location === location && notes.error && <p className="error">{notes.error}</p>}
      <div className="notes-list">{currentNotes.map(note => <div key={note.id}
        className={`note ${note.author}`}>
        <strong>{note.author === 'agent' ? 'Assistant' : 'You'}</strong>
        <span>{note.body}</span>
      </div>)}</div>
      <form onSubmit={saveNote}>
        <textarea value={noteText} onChange={event => setNoteText(event.target.value)}
          placeholder={focused ? 'Add a note to this evidence…' : 'Add a note to this page…'}
          aria-label="New note" disabled={!paper?.file_available} />
        <button disabled={!paper?.file_available || saving || !noteText.trim()}>Save note</button>
      </form>
      {noteError && <p className="error">{noteError}</p>}
    </div>
  </section>
}
