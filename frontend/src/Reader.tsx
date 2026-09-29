import { useEffect, useState, type FormEvent } from 'react'
import { api, type Annotation, type Evidence, type PageGeometry, type Paper } from './api'

function Highlight({ box, geometry, kind }: {
  box: [number, number, number, number], geometry: PageGeometry, kind: 'evidence' | 'note'
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

export default function Reader({ paper, page, onPage, focused, answerRevision }: {
  paper: Paper | null, page: number, onPage: (page: number) => void,
  focused: Evidence | null, answerRevision: number
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
  const currentGeometry = geometry.location === location ? geometry.value : null
  const currentNotes = notes.location === location ? notes.value : []

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
    if (!paper?.file_available || !noteText.trim()) return
    setSaving(true)
    try {
      const box = focused?.paper_id === paper.id && focused.page_number === page
        ? focused.bbox : [0, 0, 0, 0] as [number, number, number, number]
      await api.addNote(paper.id, page, noteText.trim(), box)
      setNotes({ location, value: await api.annotations(paper.id, page), error: '' })
      setNoteText('')
      setNoteError('')
    } catch (error) {
      setNoteError(error instanceof Error ? error.message : String(error))
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
      </div>
    </div>
    <div className="reader-scroll">
      {!paper && <p className="empty-reader">Choose a paper or select a citation to inspect its source.</p>}
      {paper && !paper.file_available && <p className="empty-reader">This PDF is no longer at its indexed path. Add the file again to read it.</p>}
      {paper?.file_available && !(geometry.location === location && geometry.error) && <div className="page-frame" key={location}>
        <img src={`/api/papers/${paper.id}/pages/${page}/image`}
          alt={`Page ${page} of ${paper.title}`} />
        {currentGeometry && <div className="page-overlay">
          {currentNotes.map(note => <Highlight key={note.id} box={note.bbox}
            geometry={currentGeometry} kind="note" />)}
          {focused?.paper_id === paper.id && focused.page_number === page &&
            <Highlight box={focused.bbox} geometry={currentGeometry} kind="evidence" />}
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
