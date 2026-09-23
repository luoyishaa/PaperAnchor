const state = { papers: [], selected: null, page: 1, focusedEvidence: null, evidence: [] };
const byId = id => document.getElementById(id);

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function message(value) { byId("status").textContent = value; }

function paperCard(paper) {
  const card = document.createElement("div");
  card.className = "paper" + (state.selected?.id === paper.id ? " active" : "");
  const title = document.createElement("strong");
  title.textContent = paper.title;
  const meta = document.createElement("small");
  meta.textContent = `${paper.page_count} pages · ${paper.passage_count} passages` +
    (paper.file_available ? "" : " · PDF missing");
  card.append(title, meta);
  card.addEventListener("click", () => openPaper(paper.id, 1));
  return card;
}

async function refreshPapers() {
  state.papers = await request("/api/papers");
  byId("papers").replaceChildren(...state.papers.map(paperCard));
}

async function openPaper(id, page, evidence = null) {
  const paper = state.papers.find(item => item.id === id);
  if (!paper) return;
  state.selected = paper;
  state.page = Math.max(1, Math.min(page, paper.page_count));
  state.focusedEvidence = evidence;
  byId("reader-title").textContent = paper.title;
  byId("reader-count").textContent = `${paper.page_count} pages`;
  byId("page-label").textContent = `${state.page} / ${paper.page_count}`;
  byId("previous").disabled = state.page <= 1;
  byId("next").disabled = state.page >= paper.page_count;
  byId("open-pdf").hidden = false;
  byId("open-pdf").href = `/api/papers/${id}/file#page=${state.page}`;
  byId("empty-reader").hidden = true;
  byId("page-wrap").hidden = false;
  byId("page-image").src = `/api/papers/${id}/pages/${state.page}/image`;
  byId("papers").replaceChildren(...state.papers.map(paperCard));
  await Promise.all([renderHighlights(), refreshNotes()]);
}

function addHighlight(bbox, dimensions, className) {
  const [x0, y0, x1, y1] = bbox;
  const marker = document.createElement("div");
  marker.className = className;
  marker.style.left = `${x0 / dimensions.width * 100}%`;
  marker.style.top = `${y0 / dimensions.height * 100}%`;
  marker.style.width = `${(x1 - x0) / dimensions.width * 100}%`;
  marker.style.height = `${(y1 - y0) / dimensions.height * 100}%`;
  byId("overlay").append(marker);
}

async function renderHighlights() {
  const paper = state.selected, page = state.page;
  const dimensions = await request(`/api/papers/${paper.id}/pages/${page}`);
  if (state.selected?.id !== paper.id || state.page !== page) return;
  byId("overlay").replaceChildren();
  if (state.focusedEvidence?.paper_id === paper.id && state.focusedEvidence?.page_number === page) {
    addHighlight(state.focusedEvidence.bbox, dimensions, "highlight");
  }
  const notes = await request(`/api/papers/${paper.id}/annotations?page=${page}`);
  if (state.selected?.id !== paper.id || state.page !== page) return;
  for (const note of notes) {
    if (note.bbox[2] > note.bbox[0] && note.bbox[3] > note.bbox[1]) {
      addHighlight(note.bbox, dimensions, "highlight note-highlight");
    }
  }
}

async function refreshNotes() {
  if (!state.selected) return;
  const paper = state.selected, page = state.page;
  const notes = await request(`/api/papers/${paper.id}/annotations?page=${page}`);
  if (state.selected?.id !== paper.id || state.page !== page) return;
  byId("note-count").textContent = `(${notes.length})`;
  const cards = notes.map(note => {
    const card = document.createElement("div");
    card.className = `note ${note.author}`;
    const author = document.createElement("b");
    author.textContent = note.author;
    const body = document.createElement("span");
    body.textContent = note.body;
    card.append(author, body);
    return card;
  });
  byId("notes").replaceChildren(...cards);
}

function renderAnswer(answer) {
  const area = byId("answer");
  area.hidden = false;
  area.replaceChildren();
  const evidenceById = new Map(answer.evidence.map(item => [item.evidence_id, item]));
  for (const part of answer.text.split(/(\[E\d+\])/g)) {
    const evidence = evidenceById.get(part.slice(1, -1));
    if (evidence && /^\[E\d+\]$/.test(part)) {
      const button = document.createElement("button");
      button.className = "citation";
      button.textContent = part;
      button.addEventListener("click", () => openPaper(evidence.paper_id, evidence.page_number, evidence));
      area.append(button);
    } else {
      area.append(document.createTextNode(part));
    }
  }
  const cards = answer.evidence.map(item => {
    const card = document.createElement("div");
    card.className = "evidence-card";
    const heading = document.createElement("strong");
    heading.textContent = `[${item.evidence_id}] ${item.paper_title}`;
    const page = document.createElement("small");
    page.textContent = `Page ${item.page_number} · Open source location`;
    const text = document.createElement("p");
    text.textContent = item.text;
    card.append(heading, page, text);
    card.addEventListener("click", () => openPaper(item.paper_id, item.page_number, item));
    return card;
  });
  byId("evidence").replaceChildren(...cards);
}

byId("upload").addEventListener("change", async event => {
  const file = event.target.files[0];
  if (!file) return;
  message(`Importing ${file.name}…`);
  const form = new FormData();
  form.append("file", file);
  try {
    const result = await request("/api/papers", { method: "POST", body: form });
    await refreshPapers();
    await openPaper(result.paper.id, 1);
    message(result.changed ? "Paper added to your library." : "Paper is already in your library.");
  } catch (error) { message(error.message); }
  event.target.value = "";
});

byId("ask-form").addEventListener("submit", async event => {
  event.preventDefault();
  const question = byId("question").value.trim();
  if (!question) return;
  message("Searching your papers and preparing an answer…");
  byId("ask-form").querySelector("button").disabled = true;
  try {
    const answer = await request("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question })
    });
    renderAnswer(answer);
    message(answer.status === "answered" ? "Answer ready. Select a citation to inspect it." :
      `Answer status: ${answer.status.replaceAll("_", " ")}`);
  } catch (error) { message(error.message); }
  byId("ask-form").querySelector("button").disabled = false;
});

byId("previous").addEventListener("click", () => openPaper(state.selected.id, state.page - 1));
byId("next").addEventListener("click", () => openPaper(state.selected.id, state.page + 1));
byId("note-form").addEventListener("submit", async event => {
  event.preventDefault();
  if (!state.selected) { message("Select a paper before adding a note."); return; }
  const body = byId("note-body").value.trim();
  if (!body) return;
  const bbox = state.focusedEvidence?.paper_id === state.selected.id &&
    state.focusedEvidence?.page_number === state.page
    ? state.focusedEvidence.bbox : [0, 0, 0, 0];
  try {
    await request("/api/annotations", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paper_id: state.selected.id, page_number: state.page, bbox, body })
    });
    byId("note-body").value = "";
    await Promise.all([refreshNotes(), renderHighlights()]);
    message("Note saved.");
  } catch (error) { message(error.message); }
});

refreshPapers().catch(error => message(error.message));
