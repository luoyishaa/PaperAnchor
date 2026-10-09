import type { RetrievalProfile } from './api'

export const retrievalProfileStorageKey = 'paperanchor.retrievalProfile'
const profiles: RetrievalProfile[] = ['legacy', 'evidence_v2', 'evidence_minilm', 'evidence_optimized']

export function loadRetrievalProfile(): RetrievalProfile {
  try {
    const stored = globalThis.localStorage?.getItem(retrievalProfileStorageKey)
    if (profiles.includes(stored as RetrievalProfile)) return stored as RetrievalProfile
  } catch {
    // Storage restrictions should not prevent using search in this tab.
  }
  return 'evidence_minilm'
}

export function saveRetrievalProfile(profile: RetrievalProfile): void {
  try {
    globalThis.localStorage?.setItem(retrievalProfileStorageKey, profile)
  } catch {
    // The selected profile remains active for this tab when persistence fails.
  }
}
