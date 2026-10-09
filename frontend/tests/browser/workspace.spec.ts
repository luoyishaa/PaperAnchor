import { expect, test, type Page, type Route } from '@playwright/test'

const paper = {
  id: 10, title: 'Evidence paper', page_count: 2, passage_count: 3,
  indexed_at: '', file_available: true, published: null, year: null,
  year_basis: null, venue: null, paper_url: null,
}
const answer = {
  status: 'answered', text: 'Supported fact. [E1]', basis: 'papers',
  evidence: [{ evidence_id: 'E1', passage_id: 1, paper_id: 10,
    paper_title: paper.title, page_number: 2, text: 'Evidence', bbox: [10, 20, 100, 80] }],
  cited_evidence_ids: ['E1'], attempted_queries: [], follow_up_question: null,
}

async function mockWorkspace(page: Page) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url())
    let body: unknown = []
    if (url.pathname === '/api/spaces') body = [
      { id: 1, name: 'All papers', created_at: '', paper_count: 1 },
      { id: 2, name: 'Other space', created_at: '', paper_count: 0 },
    ]
    if (url.pathname === '/api/papers') body = url.searchParams.get('space_id') === '2' ? [] : [paper]
    if (url.pathname === '/api/semantic/status') body = {
      model: 'test', indexed_passages: 3, total_passages: 3, job: null,
    }
    if (/\/pages\/\d+$/.test(url.pathname)) body = { width: 600, height: 800, page_number: 2 }
    if (url.pathname.endsWith('/image')) {
      await route.fulfill({ contentType: 'image/png', body: Buffer.from(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aP1sAAAAASUVORK5CYII=', 'base64') })
      return
    }
    await route.fulfill({ json: body })
  })
}

async function ask(page: Page, question = 'Explain the evidence') {
  await page.getByRole('textbox', { name: 'Ask your papers' }).fill(question)
  await page.getByRole('button', { name: 'Send question' }).click()
}

test('MiniLM is the default and an explicit search preference survives reload', async ({ page }) => {
  await mockWorkspace(page)
  const requests: Record<string, unknown>[] = []
  await page.route('**/api/ask', route => {
    requests.push(route.request().postDataJSON())
    return route.fulfill({ json: answer })
  })
  await page.goto('/')
  await page.getByRole('button', { name: /^Settings/ }).click()
  const selector = page.getByRole('combobox', { name: 'Text search' })
  await expect(selector).toHaveValue('evidence_minilm')
  await ask(page)
  await expect.poll(() => requests.length).toBe(1)
  expect(requests[0].retrieval_profile).toBe('evidence_minilm')
  await selector.selectOption('evidence_v2')
  await page.reload()
  await page.getByRole('button', { name: /^Settings/ }).click()
  await expect(selector).toHaveValue('evidence_v2')
  await ask(page, 'What is the method?')
  await expect.poll(() => requests.length).toBe(2)
  expect(requests[1].retrieval_profile).toBe('evidence_v2')
  await selector.selectOption('legacy')
  await ask(page, 'Which data?')
  await expect.poll(() => requests.length).toBe(3)
  expect(requests[2].retrieval_profile).toBeUndefined()
})

test('Optimized is shown only when the installed profile is available', async ({ page }) => {
  await mockWorkspace(page)
  await page.route('**/api/semantic/status', route => route.fulfill({ json: {
    model: 'test', indexed_passages: 3, total_passages: 3, job: null,
    retrieval_profiles: ['legacy', 'evidence_v2', 'evidence_minilm', 'evidence_optimized'],
  } }))
  let selected: string | undefined
  await page.route('**/api/ask', route => {
    selected = route.request().postDataJSON().retrieval_profile
    return route.fulfill({ json: answer })
  })
  await page.goto('/')
  await page.getByRole('button', { name: /^Settings/ }).click()
  const selector = page.getByRole('combobox', { name: 'Text search' })
  await expect(selector.locator('option[value="evidence_optimized"]')).toHaveCount(1)
  await selector.selectOption('evidence_optimized')
  await ask(page)
  await expect.poll(() => selected).toBe('evidence_optimized')
})

test('an unavailable saved Optimized choice is retained while Fast is used visibly', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('paperanchor.retrievalProfile', 'evidence_optimized'))
  await mockWorkspace(page)
  let selected: string | undefined
  await page.route('**/api/ask', route => {
    selected = route.request().postDataJSON().retrieval_profile
    return route.fulfill({ json: answer })
  })
  await page.goto('/')
  await page.getByRole('button', { name: /^Settings/ }).click()
  const selector = page.getByRole('combobox', { name: 'Text search' })
  await expect(selector).toHaveValue('evidence_minilm')
  await expect(selector.locator('option[value="evidence_optimized"]')).toHaveCount(0)
  await expect(page.getByText('Optimized search is unavailable. Using Fast (MiniLM).')).toBeVisible()
  await ask(page)
  await expect.poll(() => selected).toBe('evidence_minilm')
  expect(await page.evaluate(() => localStorage.getItem('paperanchor.retrievalProfile'))).toBe('evidence_optimized')
})

test('answer citation opens its actual PDF page and highlight', async ({ page }) => {
  await mockWorkspace(page)
  await page.route('**/api/ask', route => route.fulfill({ json: answer }))
  await page.goto('/')
  await ask(page)
  await page.getByRole('button', { name: /E1.*Evidence paper/ }).last().click()
  await expect(page.getByRole('img', { name: 'Page 2 of Evidence paper' })).toBeVisible()
  await expect(page.locator('.highlight.evidence')).toHaveCount(1)
})

test('Stop restores the question and suppresses a delayed response', async ({ page }) => {
  await mockWorkspace(page)
  let held: Route | undefined
  let requests = 0
  await page.route('**/api/ask', route => { requests += 1; held = route })
  await page.goto('/')
  await ask(page)
  await expect.poll(() => Boolean(held)).toBe(true)
  await page.getByRole('button', { name: 'Cancel answer request' }).click()
  await expect(page.getByRole('textbox', { name: 'Ask your papers' })).toHaveValue('Explain the evidence')
  await held!.fulfill({ json: answer }).catch(() => {})
  await expect(page.getByText('Supported fact.', { exact: false })).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Send question' })).toBeEnabled()
  expect(requests).toBe(1)
})

test('switching research spaces suppresses an old answer', async ({ page }) => {
  await mockWorkspace(page)
  let held: Route | undefined
  await page.route('**/api/ask', route => { held = route })
  await page.goto('/')
  await ask(page)
  await expect.poll(() => Boolean(held)).toBe(true)
  await page.getByRole('button', { name: /All papers.*1 papers/ }).click()
  await page.getByRole('button', { name: 'Other space 0 papers' }).click()
  await held!.fulfill({ json: answer }).catch(() => {})
  await expect(page.getByRole('button', { name: 'Sources: all 0 papers' })).toBeVisible()
  await expect(page.getByText('Supported fact.', { exact: false })).toHaveCount(0)
})

test('model failure restores question for retry', async ({ page }) => {
  await mockWorkspace(page)
  await page.route('**/api/ask', route => route.fulfill({ status: 503, json: { detail: 'Model unavailable' } }))
  await page.goto('/')
  await ask(page)
  await expect(page.getByRole('status')).toContainText('Model unavailable')
  await expect(page.getByRole('textbox', { name: 'Ask your papers' })).toHaveValue('Explain the evidence')
  await page.route('**/api/ask', route => route.fulfill({ json: answer }))
  await page.getByRole('button', { name: 'Send question' }).click()
  await expect(page.getByText('Supported fact.', { exact: false })).toBeVisible()
})

test('mobile navigation can switch space and returns to composer', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await mockWorkspace(page)
  await page.goto('/')
  await page.getByRole('button', { name: 'Open navigation' }).click()
  await page.getByRole('button', { name: /All papers.*1 papers/ }).click()
  await page.getByRole('button', { name: 'Other space 0 papers' }).click()
  await expect(page.getByRole('textbox', { name: 'Ask your papers' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Sources: all 0 papers' })).toBeVisible()
})

test('late upload does not open a PDF from a previous space', async ({ page }) => {
  await mockWorkspace(page)
  let held: Route | undefined
  await page.route('**/api/papers', async route => {
    if (route.request().method() !== 'POST') return route.fallback()
    held = route
  })
  await page.goto('/')
  await page.locator('input[type=file]').first().setInputFiles({
    name: 'sample.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-test'),
  })
  await expect.poll(() => Boolean(held)).toBe(true)
  await page.getByRole('button', { name: /All papers.*1 papers/ }).click()
  await page.getByRole('button', { name: 'Other space 0 papers' }).click()
  await held!.fulfill({ json: { paper, changed: true } })
  await expect(page.getByRole('button', { name: 'Sources: all 0 papers' })).toBeVisible()
  await expect(page.getByRole('img', { name: /Evidence paper/ })).toHaveCount(0)
  await expect(page.getByRole('status')).not.toContainText('Paper added')
})

test('old canceled request cannot finish a newer pending question', async ({ page }) => {
  await mockWorkspace(page)
  const held: Route[] = []
  await page.route('**/api/ask', route => { held.push(route) })
  await page.goto('/')
  await ask(page, 'Old question')
  await expect.poll(() => held.length).toBe(1)
  await page.getByRole('button', { name: 'Cancel answer request' }).click()
  await ask(page, 'New question')
  await expect.poll(() => held.length).toBe(2)
  await held[0].fulfill({ json: answer }).catch(() => {})
  await expect(page.getByRole('button', { name: 'Cancel answer request' })).toBeVisible()
  await held[1].fulfill({ json: { ...answer, text: 'New supported fact. [E1]' } })
  await expect(page.getByText('New supported fact.', { exact: false })).toBeVisible()
})

test('late note save does not clear a draft on another PDF page', async ({ page }) => {
  await mockWorkspace(page)
  await page.route('**/api/ask', route => route.fulfill({ json: answer }))
  let held: Route | undefined
  await page.route('**/api/annotations', route => { held = route })
  await page.goto('/')
  await ask(page)
  await page.getByRole('button', { name: /E1.*Evidence paper/ }).last().click()
  await page.getByRole('textbox', { name: 'New note' }).fill('Old note')
  await page.getByRole('button', { name: 'Save note' }).click()
  await expect.poll(() => Boolean(held)).toBe(true)
  await page.getByRole('button', { name: 'Previous page' }).click()
  await expect(page.getByRole('img', { name: 'Page 1 of Evidence paper' })).toBeVisible()
  await page.getByRole('textbox', { name: 'New note' }).fill('Draft for page one')
  await held!.fulfill({ json: { id: 1, body: 'Old note' } })
  await expect(page.getByRole('button', { name: 'Save note' })).toBeEnabled()
  await expect(page.getByRole('textbox', { name: 'New note' })).toHaveValue('Draft for page one')
})
