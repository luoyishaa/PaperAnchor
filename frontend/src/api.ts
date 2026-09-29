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
}

export interface Answer {
  status: 'answered' | 'no_evidence' | 'uncited' | 'invalid_citation' |
    'insufficient_evidence' | 'general_knowledge'
  text: string
  evidence: Evidence[]
  cited_evidence_ids: string[]
  basis: 'papers' | 'model' | 'none'
  attempted_queries: string[]
  follow_up_question: string | null
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
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { detail?: string }
    throw new Error(error.detail || `Request failed (${response.status})`)
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
  discover: (query: string, source: 'arxiv' | 'semantic_scholar') =>
    request<Candidate[]>(`/api/discover?q=${encodeURIComponent(query)}&source=${source}`),
  importCandidate: (arxivId: string, spaceId: number) => request<{
    paper: Paper; changed: boolean; semantic_job_id: number | null
  }>('/api/discover/import', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arxiv_id: arxivId, space_id: spaceId }),
  }),
  addToSpace: (spaceId: number, paperId: number) =>
    request<{ added: boolean }>(`/api/spaces/${spaceId}/papers/${paperId}`, { method: 'POST' }),
  removeFromSpace: (spaceId: number, paperId: number) =>
    request<{ removed: boolean }>(`/api/spaces/${spaceId}/papers/${paperId}`, { method: 'DELETE' }),
  upload: (file: File, spaceId: number) => {
    const body = new FormData()
    body.append('file', file)
    body.append('space_id', String(spaceId))
    return request<{ paper: Paper; changed: boolean }>('/api/papers', { method: 'POST', body })
  },
  answer: (question: string, spaceId: number, allowGeneralKnowledge: boolean) => request<Answer>('/api/ask', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, space_id: spaceId,
      allow_general_knowledge: allowGeneralKnowledge }),
  }),
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
