import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/retrievalPreferences.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
})
const { loadRetrievalProfile, saveRetrievalProfile, retrievalProfileStorageKey } =
  await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)

function storage(t, value) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value })
  t.after(() => {
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous)
    else delete globalThis.localStorage
  })
}

test('new and unknown preferences default to MiniLM', t => {
  let saved = null
  storage(t, { getItem: () => saved })
  assert.equal(loadRetrievalProfile(), 'evidence_minilm')
  saved = 'unknown-profile'
  assert.equal(loadRetrievalProfile(), 'evidence_minilm')
})

test('all supported selections persist including legacy and the BGE identifier', t => {
  const values = new Map()
  storage(t, { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value) })
  for (const profile of ['legacy', 'evidence_v2', 'evidence_minilm', 'evidence_optimized']) {
    saveRetrievalProfile(profile)
    assert.equal(values.get(retrievalProfileStorageKey), profile)
    assert.equal(loadRetrievalProfile(), profile)
  }
})

test('restricted storage does not prevent selecting search in the current tab', t => {
  storage(t, { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') } })
  assert.equal(loadRetrievalProfile(), 'evidence_minilm')
  assert.doesNotThrow(() => saveRetrievalProfile('evidence_v2'))
})
