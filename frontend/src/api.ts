export interface ResearchSpace {
  id: number
  name: string
  created_at: string
  paper_count: number
}

export interface Paper {
  id: number
  title: string
  page_count: number
  passage_count: number
  indexed_at: string
  file_available: boolean
  published: string | null
  year: number | null
  year_basis: 'source_metadata' | 'arxiv_filename' | null
  venue: string | null
  paper_url: string | null
}

export interface AvailablePaper {
  id: number
  title: string
  page_count: number
}

export interface PaperSearchPage {
  items: AvailablePaper[]
  total: number
}

export interface Candidate {
  source: 'arxiv' | 'semantic_scholar'
  source_id: string
  title: string
  authors: string[]
  abstract: string
  published: string | null
  year: number | null
  venue: string | null
  citation_count: number | null
  paper_url: string
  arxiv_id: string | null
  doi: string | null
  importable: boolean
}

export interface Evidence {
  evidence_id: string
  passage_id: number
  paper_id: number
  paper_title: string
  page_number: number
  text: string
  bbox: [number, number, number, number]
  source_spans?: {
    passage_id: number
    page_number: number
    bbox: [number, number, number, number]
  }[]
}

export interface VisualRegion {
  paper_id: number
  page_number: number
  bbox: [number, number, number, number]
}

export type RetrievalProfile = 'legacy' | 'evidence_v2' | 'evidence_minilm' | 'evidence_optimized'

export interface Answer {
  status: 'answered' | 'no_evidence' | 'uncited' | 'invalid_citation' |
    'insufficient_evidence' | 'general_knowledge'
  text: string
  evidence: Evidence[]
  cited_evidence_ids: string[]
  basis: 'papers' | 'model' | 'none'
  attempted_queries: string[]
  follow_up_question: string | null
  conversation_id?: string
  standalone_question?: string
}

export interface ConversationRecord {
  id: string
  space_id: number
  title: string
  updated_at: string
  turn_count: number
}

export interface ConversationTurn {
  id: number
  conversation_id: string
  question: string
  standalone_question: string
  answer: Answer
  created_at: string
}

export interface Annotation {
  id: number
  paper_id: number
  page_number: number
  bbox: [number, number, number, number]
  body: string
  author: 'user' | 'agent'
  created_at: string
}

export interface PageGeometry {
  width: number
  height: number
  page_number: number
}

export interface Job {
  id: number
  kind: string
  state: 'queued' | 'running' | 'succeeded' | 'failed'
  completed: number
  total: number
  error: string | null
}

export interface SemanticStatus {
  model: string
  indexed_passages: number
  total_passages: number
  job: Job | null
  retrieval_profiles?: RetrievalProfile[]
  retrieval_profile_errors?: Partial<Record<RetrievalProfile, string>>
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: unknown }
    const detail = error?.detail
    const messages = Array.isArray(detail) ? detail.flatMap(item => {
      if (!item || typeof item !== 'object' || typeof item.msg !== 'string') return []
      const location = Array.isArray(item.loc) ? item.loc.join('.') : ''
      return [location ? `${location}: ${item.msg}` : item.msg]
    }) : []
    const message = typeof detail === 'string' && detail.trim() ? detail : messages.join('; ')
    throw new Error(message || `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  spaces: () => request<ResearchSpace[]>('/api/spaces'),
  createSpace: (name: string) => request<ResearchSpace>('/api/spaces', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  }),
  papers: (spaceId: number) => request<Paper[]>(`/api/papers?space_id=${spaceId}`),
  discover: (query: string, source: 'arxiv' | 'semantic_scholar', signal?: AbortSignal) =>
    request<Candidate[]>(`/api/discover?q=${encodeURIComponent(query)}&source=${source}`, { signal }),
  importCandidate: (arxivId: string, spaceId: number) => request<{
    paper: Paper; changed: boolean; semantic_job_id: number | null
  }>('/api/discover/import', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arxiv_id: arxivId, space_id: spaceId }),
  }),
  addToSpace: (spaceId: number, paperId: number) =>
    request<{ added: boolean }>(`/api/spaces/${spaceId}/papers/${paperId}`, { method: 'POST' }),
  availablePapers: (spaceId: number, query: string, offset: number) =>
    request<PaperSearchPage>(`/api/spaces/${spaceId}/available-papers?q=${encodeURIComponent(query)}&offset=${offset}&limit=50`),
  addPapersToSpace: (spaceId: number, selection: { paper_ids: number[] } | { query: string }) =>
    request<{ added: number }>(`/api/spaces/${spaceId}/papers/batch`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(selection),
    }),
  removeFromSpace: (spaceId: number, paperId: number) =>
    request<{ removed: boolean }>(`/api/spaces/${spaceId}/papers/${paperId}`, { method: 'DELETE' }),
  deletePaper: (paperId: number) =>
    request<{ deleted: boolean; file_removed: boolean; file_was_managed: boolean }>(
      `/api/papers/${paperId}`, { method: 'DELETE' }),
  upload: (file: File, spaceId: number) => {
    const body = new FormData()
    body.append('file', file)
    body.append('space_id', String(spaceId))
    return request<{ paper: Paper; changed: boolean }>('/api/papers', { method: 'POST', body })
  },
  answer: (question: string, spaceId: number, allowGeneralKnowledge: boolean,
           paperIds?: number[], visualMode: 'auto' | 'off' | 'on' = 'auto',
            visualRegion?: VisualRegion | null, conversationId?: string | null,
            signal?: AbortSignal, retrievalProfile: RetrievalProfile = 'legacy',
            titleHint?: string) => request<Answer>('/api/ask', {
    signal,
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, space_id: spaceId,
      allow_general_knowledge: allowGeneralKnowledge,
      visual_mode: visualMode,
      ...(retrievalProfile !== 'legacy' ? { retrieval_profile: retrievalProfile } : {}),
      ...(titleHint?.trim() ? { title_hint: titleHint.trim() } : {}),
      ...(visualRegion ? { visual_region: visualRegion } : {}),
      ...(conversationId ? { conversation_id: conversationId } : {}),
      ...(paperIds?.length ? { paper_ids: paperIds } : {}) }),
  }),
  conversations: (spaceId: number) =>
    request<ConversationRecord[]>(`/api/conversations?space_id=${spaceId}`),
  turns: (conversationId: string, spaceId: number) =>
    request<ConversationTurn[]>(`/api/conversations/${conversationId}/turns?space_id=${spaceId}`),
  semanticStatus: () => request<SemanticStatus>('/api/semantic/status'),
  buildSemantic: () => request<Job>('/api/semantic/build', { method: 'POST' }),
  pageGeometry: (paperId: number, page: number) =>
    request<PageGeometry>(`/api/papers/${paperId}/pages/${page}`),
  annotations: (paperId: number, page: number) =>
    request<Annotation[]>(`/api/papers/${paperId}/annotations?page=${page}`),
  addNote: (paperId: number, page: number, body: string,
            bbox: [number, number, number, number]) =>
    request<Annotation>('/api/annotations', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paper_id: paperId, page_number: page, bbox, body }),
    }),
}
