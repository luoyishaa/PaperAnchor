import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { test } from 'node:test'
import ts from 'typescript'

const source = await readFile(new URL('../src/api.ts', import.meta.url), 'utf8')
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 },
})
const { request, api } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString('base64')}`)

test('request preserves successful response data and fetch options', async t => {
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    assert.equal(path, '/api/example')
    assert.equal(init.signal, controller.signal)
    return Response.json({ result: 42 })
  })
  assert.deepEqual(await request('/api/example', { signal: controller.signal }), { result: 42 })
})

test('request reports FastAPI validation locations and messages', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json({ detail: [
    { loc: ['body', 'question'], msg: 'Question cannot be blank' },
    { loc: ['body', 'limit'], msg: 'Input should be less than or equal to 20' },
  ] }, { status: 422 }))
  await assert.rejects(request('/api/ask'), {
    message: 'body.question: Question cannot be blank; body.limit: Input should be less than or equal to 20',
  })
})

test('request preserves provider error messages', async t => {
  t.mock.method(globalThis, 'fetch', async () => Response.json(
    { detail: 'Model unavailable' }, { status: 503 }))
  await assert.rejects(request('/api/ask'), { message: 'Model unavailable' })
})

test('request falls back safely for non-JSON and malformed error bodies', async t => {
  for (const body of ['<html>Bad gateway</html>', 'null', '{"detail":{}}']) {
    t.mock.method(globalThis, 'fetch', async () => new Response(body, { status: 502 }))
    await assert.rejects(request('/api/ask'), { message: 'Request failed (502)' })
    t.mock.restoreAll()
  }
})

test('request propagates cancellation without rewriting the error', async t => {
  const error = new DOMException('Canceled', 'AbortError')
  t.mock.method(globalThis, 'fetch', async () => { throw error })
  await assert.rejects(request('/api/ask'), actual => actual === error)
})

test('answer cancellation reaches fetch without changing the question scope', async t => {
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    assert.equal(path, '/api/ask')
    assert.equal(init.signal, controller.signal)
    assert.deepEqual(JSON.parse(init.body), {
      question: 'Compare methods', space_id: 7, allow_general_knowledge: false,
      visual_mode: 'off', conversation_id: 'conversation', paper_ids: [3, 5],
    })
    return new Promise((resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true })
    })
  })
  const pending = api.answer('Compare methods', 7, false, [3, 5], 'off', null,
    'conversation', controller.signal)
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
})

test('discovery forwards cancellation and encodes the query', async t => {
  const controller = new AbortController()
  t.mock.method(globalThis, 'fetch', async (path, init) => {
    const url = new URL(path, 'http://localhost')
    assert.equal(url.pathname, '/api/discover')
    assert.equal(url.searchParams.get('q'), 'graphs & evidence')
    assert.equal(url.searchParams.get('source'), 'semantic_scholar')
    assert.equal(init.signal, controller.signal)
    return new Promise((resolve, reject) => {
      init.signal.addEventListener('abort', () => reject(init.signal.reason), { once: true })
    })
  })
  const pending = api.discover('graphs & evidence', 'semantic_scholar', controller.signal)
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
})

test('expanded evidence sends its profile and optional public title separately', async t => {
  t.mock.method(globalThis, 'fetch', async (_path, init) => {
    assert.deepEqual(JSON.parse(init.body), {
      question: 'Which method?', space_id: 3, allow_general_knowledge: false,
      visual_mode: 'off', paper_ids: [5], retrieval_profile: 'evidence_v2',
      title_hint: 'A public paper title',
    })
    return Response.json({ status: 'answered' })
  })
  await api.answer('Which method?', 3, false, [5], 'off', null, null,
    undefined, 'evidence_v2', ' A public paper title ')
})
