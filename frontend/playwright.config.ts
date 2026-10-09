import { defineConfig } from '@playwright/test'

const port = Number(process.env.PAPERANCHOR_TEST_PORT || 5173)
const baseURL = `http://127.0.0.1:${port}`

export default defineConfig({
  testDir: './tests/browser',
  use: {
    baseURL,
    channel: process.env.PAPERANCHOR_TEST_BROWSER || undefined,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: `npm run dev -- --host 127.0.0.1 --port ${port} --strictPort`,
    url: baseURL,
    reuseExistingServer: false,
  },
})
