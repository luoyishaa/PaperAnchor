"""Local HTTP interface for the paper library and located evidence."""

from dataclasses import asdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4
from threading import Lock
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import pymupdf
import httpx
from pydantic import BaseModel, Field, field_validator

from .library import PaperLibrary
from .llm import ModelConfigError, ModelRequestError, generate_with_config
from .model_config import get_setting
from .agent import direct_question, research_question, standalone_question
from .semantic import (EmbeddingProvider, FastEmbedProvider, SemanticIndex,
                       build_retriever)
from .unit_index import UnitStore, UnitSemanticIndex
from .vision import (VisualRegion, answer_visual, choose_visual_pages,
                     is_explicit_visual_question)
from .conversations import ConversationStore
from .spaces import SpaceStore
from .jobs import JobStore, SemanticWorker, job_data
from .discovery import ARXIV_ID, Discovery, MetadataStore
from .evidence_profile import EvidenceProfile, EvidenceProfileUnavailable


MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class VisualRegionRequest(BaseModel):
    paper_id: int
    page_number: int
    bbox: tuple[float, float, float, float]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=8, ge=1, le=20)
    space_id: int = 1
    allow_general_knowledge: bool = False
    enable_decomposition: bool = False
    paper_ids: list[int] | None = Field(default=None, max_length=12)
    visual_mode: Literal["auto", "off", "on"] = "auto"
    visual_region: VisualRegionRequest | None = None
    conversation_id: str | None = Field(default=None, max_length=32)
    retrieval_profile: Literal["legacy", "evidence_v2", "evidence_minilm", "evidence_optimized"] = "legacy"
    title_hint: str = Field(default="", max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank")
        return value


class SpaceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class BatchPapersRequest(BaseModel):
    paper_ids: list[int] | None = Field(default=None, max_length=1000)
    query: str | None = Field(default=None, max_length=200)


class AnnotationRequest(BaseModel):
    paper_id: int
    page_number: int
    bbox: tuple[float, float, float, float]
    body: str = Field(min_length=1, max_length=4000)


class ImportRequest(BaseModel):
    arxiv_id: str = Field(min_length=1, max_length=100)
    space_id: int = 1


def _paper_data(paper, metadata: dict | None = None) -> dict:
    arxiv_name = re.fullmatch(r"(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?", paper.file_path.stem)
    inferred_year = None
    if arxiv_name:
        candidate_year = 2000 + int(arxiv_name.group(1))
        month = int(arxiv_name.group(2))
        if 2007 <= candidate_year <= datetime.now(timezone.utc).year and 1 <= month <= 12:
            inferred_year = candidate_year
    return {
        "id": paper.id,
        "title": paper.title,
        "page_count": paper.page_count,
        "passage_count": paper.passage_count,
        "indexed_at": paper.indexed_at,
        "file_available": paper.file_path.is_file(),
        "published": metadata.get("published") if metadata else None,
        "year": metadata.get("year") if metadata else inferred_year,
        "year_basis": "source_metadata" if metadata and metadata.get("year")
                      else "arxiv_filename" if inferred_year else None,
        "venue": metadata.get("venue") if metadata else None,
        "paper_url": metadata.get("paper_url") if metadata else None,
    }


def create_app(database_path: Path, *, generate=generate_with_config,
               embedding_provider: EmbeddingProvider | None = None,
               discovery: Discovery | None = None) -> FastAPI:
    library = PaperLibrary(database_path)
    spaces = SpaceStore(database_path)
    conversations = ConversationStore(database_path)
    representation = get_setting("PAPERANCHOR_REPRESENTATION").lower() or "units"
    if representation not in ("blocks", "units"):
        raise ValueError("PAPERANCHOR_REPRESENTATION must be 'blocks' or 'units'")
    unit_store = UnitStore(library) if representation == "units" else None
    semantic_index = (UnitSemanticIndex(unit_store) if unit_store is not None
                      else SemanticIndex(database_path))
    provider = embedding_provider or FastEmbedProvider(
        cache_dir=database_path.parent / "models"
    )
    retriever = build_retriever(
        unit_store or library, semantic_index, provider,
        rerank_model=get_setting("PAPERANCHOR_RERANK_MODEL"),
        cache_dir=database_path.parent / "models",
    )
    jobs = JobStore(database_path)
    metadata = MetadataStore(database_path)
    discover = discovery or Discovery()
    worker = SemanticWorker(jobs, semantic_index, lambda: provider)
    upload_dir = database_path.parent / "uploads"
    import_lock = Lock()
    evidence_profile = EvidenceProfile(database_path)
    static_dir = Path(__file__).parent / "static"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.start()
        if unit_store is not None:
            unit_store.refresh()
            if semantic_index.count(provider.name) < unit_store.count_units():
                worker.enqueue()
        try:
            yield
        finally:
            worker.stop()
            evidence_profile.close()
            if discovery is None:
                discover.close()

    app = FastAPI(title="PaperAnchor", docs_url="/api/docs", redoc_url=None,
                  lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def paper_or_404(paper_id: int):
        paper = library.get_paper(paper_id)
        if paper is None:
            raise HTTPException(404, "Paper not found")
        if not paper.file_path.is_file():
            raise HTTPException(410, "PDF file is no longer available")
        return paper

    @app.get("/")
    def home():
        return FileResponse(static_dir / "index.html")

    def space_or_404(space_id: int):
        space = spaces.get_space(space_id)
        if space is None:
            raise HTTPException(404, "Space not found")
        return space

    @app.get("/api/spaces")
    def list_spaces():
        return [asdict(space) for space in spaces.list_spaces()]

    @app.post("/api/spaces")
    def create_space(request: SpaceRequest):
        try:
            return asdict(spaces.create_space(request.name))
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.delete("/api/spaces/{space_id}")
    def delete_space(space_id: int):
        try:
            spaces.delete_space(space_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {"deleted": True}

    @app.get("/api/spaces/{space_id}/available-papers")
    def available_papers(space_id: int, q: str = "", limit: int = 50, offset: int = 0):
        if not 1 <= limit <= 100 or offset < 0 or len(q) > 200:
            raise HTTPException(422, "Invalid search page")
        try:
            items, total = spaces.available_papers(space_id, q, limit=limit, offset=offset)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return {"items": items, "total": total}

    @app.post("/api/spaces/{space_id}/papers/batch")
    def add_papers_to_space(space_id: int, request: BatchPapersRequest):
        try:
            added = spaces.add_papers(space_id, request.paper_ids, query=request.query)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {"added": added}

    @app.post("/api/spaces/{space_id}/papers/{paper_id}")
    def add_paper_to_space(space_id: int, paper_id: int):
        try:
            spaces.add_paper(space_id, paper_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        return {"added": True}

    @app.delete("/api/spaces/{space_id}/papers/{paper_id}")
    def remove_paper_from_space(space_id: int, paper_id: int):
        try:
            spaces.remove_paper(space_id, paper_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {"removed": True}

    @app.get("/api/papers")
    def papers(space_id: int = 1):
        space_or_404(space_id)
        return [_paper_data(paper, metadata.get(paper.id))
                for paper in library.list_papers(space_id=space_id)]

    @app.get("/api/conversations")
    def list_conversations(space_id: int = 1):
        space_or_404(space_id)
        return [asdict(item) for item in conversations.list(space_id)]

    @app.get("/api/conversations/{conversation_id}/turns")
    def conversation_turns(conversation_id: str, space_id: int = 1):
        space_or_404(space_id)
        if conversations.get(conversation_id, space_id=space_id) is None:
            raise HTTPException(404, "Conversation not found in this research space")
        return [asdict(item) for item in conversations.turns(conversation_id)]

    @app.delete("/api/papers/{paper_id}")
    def delete_paper(paper_id: int):
        stored_path = library.delete_paper(paper_id)
        if stored_path is None:
            raise HTTPException(404, "Paper not found")
        owned = stored_path.resolve().parent == upload_dir.resolve()
        file_removed = False
        if owned:
            try:
                stored_path.unlink(missing_ok=True)
                file_removed = True
            except OSError:
                # The database is already consistent; a locked local PDF can
                # be removed after the viewer releases it.
                pass
        return {"deleted": True, "file_removed": file_removed,
                "file_was_managed": owned}

    @app.get("/api/discover")
    def search_remote(q: str, source: str = "arxiv", limit: int = 10):
        try:
            return [asdict(candidate) for candidate in discover.search(q, source, limit)]
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 429:
                raise HTTPException(429, "Source rate limit reached. Try arXiv or add a Semantic Scholar API key.") from error
            raise HTTPException(502, f"Paper source unavailable: {error}") from error
        except httpx.HTTPError as error:
            raise HTTPException(502, f"Paper source unavailable: {error}") from error

    @app.post("/api/discover/import")
    def import_arxiv(request: ImportRequest):
        space_or_404(request.space_id)
        if not ARXIV_ID.fullmatch(request.arxiv_id):
            raise HTTPException(422, "Invalid arXiv ID")
        try:
            candidate = discover.arxiv_record(request.arxiv_id)
            content = discover.arxiv_pdf(request.arxiv_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except httpx.HTTPError as error:
            raise HTTPException(502, f"arXiv unavailable: {error}") from error
        upload_dir.mkdir(parents=True, exist_ok=True)
        destination = upload_dir / f"arxiv-{sha256(content).hexdigest()[:20]}.pdf"
        with import_lock:
            created = not destination.exists()
            if created:
                destination.write_bytes(content)
            try:
                result = library.index_pdf(destination, display_name=candidate.title)
            except (ValueError, pymupdf.FileDataError) as error:
                if created:
                    destination.unlink(missing_ok=True)
                raise HTTPException(422, f"Cannot index PDF: {error}") from error
        metadata.put(result.paper_id, candidate)
        if request.space_id != 1:
            spaces.add_paper(request.space_id, result.paper_id)
        job = worker.enqueue() if result.changed else None
        paper = library.get_paper(result.paper_id)
        return {"paper": _paper_data(paper, metadata.get(paper.id)),
                "changed": result.changed,
                "semantic_job_id": job.id if job else None}

    @app.post("/api/papers")
    def upload_paper(file: UploadFile = File(...), space_id: int = Form(1)):
        space_or_404(space_id)
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Choose a PDF file")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)[:100]
        temporary = upload_dir / f".upload-{uuid4().hex}.pdf"
        digest = sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                header = file.file.read(5)
                if header != b"%PDF-":
                    raise HTTPException(422, "File does not have a PDF header")
                digest.update(header)
                output.write(header)
                size += len(header)
                while chunk := file.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "PDF exceeds 40 MB")
                    digest.update(chunk)
                    output.write(chunk)
            if not size:
                raise HTTPException(400, "PDF is empty")
            destination = upload_dir / f"{digest.hexdigest()[:20]}-{safe_name}"
            # Publish and index together: one failed duplicate upload must not
            # remove the PDF another request has already indexed successfully.
            with import_lock:
                created = not destination.exists()
                if not created:
                    temporary.unlink()
                else:
                    temporary.replace(destination)
                try:
                    result = library.index_pdf(destination, display_name=Path(file.filename).stem)
                except (ValueError, pymupdf.FileDataError) as error:
                    if created:
                        destination.unlink(missing_ok=True)
                    raise HTTPException(422, f"Cannot index PDF: {error}") from error
            paper = library.get_paper(result.paper_id)
            if space_id != 1:
                spaces.add_paper(space_id, paper.id)
            job = worker.enqueue() if result.changed else None
            return {"paper": _paper_data(paper, metadata.get(paper.id)), "changed": result.changed,
                    "semantic_job_id": job.id if job else None}
        finally:
            temporary.unlink(missing_ok=True)
            file.file.close()

    @app.get("/api/papers/{paper_id}/file")
    def pdf_file(paper_id: int):
        paper = paper_or_404(paper_id)
        return FileResponse(paper.file_path, media_type="application/pdf",
                            filename=paper.file_path.name, content_disposition_type="inline")

    @app.get("/api/papers/{paper_id}/pages/{page_number}")
    def page_info(paper_id: int, page_number: int):
        paper = paper_or_404(paper_id)
        if not 1 <= page_number <= paper.page_count:
            raise HTTPException(404, "Page not found")
        with pymupdf.open(paper.file_path) as document:
            page = document[page_number - 1]
            page.set_rotation(0)
            return {"width": page.rect.width, "height": page.rect.height,
                    "page_number": page_number}

    @app.get("/api/papers/{paper_id}/pages/{page_number}/image")
    def page_image(paper_id: int, page_number: int):
        paper = paper_or_404(paper_id)
        if not 1 <= page_number <= paper.page_count:
            raise HTTPException(404, "Page not found")
        with pymupdf.open(paper.file_path) as document:
            page = document[page_number - 1]
            page.set_rotation(0)
            image = page.get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6), alpha=False)
            return Response(image.tobytes("png"), media_type="image/png")

    @app.get("/api/search")
    def search(q: str, limit: int = 5, space_id: int = 1):
        if not 1 <= limit <= 20:
            raise HTTPException(422, "limit must be between 1 and 20")
        space_or_404(space_id)
        return [asdict(hit) for hit in retriever.search(q, limit, space_id=space_id)]

    @app.get("/api/semantic/status")
    def semantic_status():
        latest = jobs.latest("semantic")
        if unit_store is not None:
            unit_store.refresh()
        profile_status = evidence_profile.availability()
        return {"model": provider.name, "representation": representation,
                "retrieval_profiles": profile_status["profiles"],
                "retrieval_profile_errors": profile_status["errors"],
                "indexed_passages": semantic_index.count(provider.name),
                "total_passages": (unit_store.count_units() if unit_store is not None
                                   else sum(paper.passage_count for paper in library.list_papers())),
                "job": job_data(latest) if latest else None}

    @app.post("/api/semantic/build")
    def build_semantic_index():
        return job_data(worker.enqueue())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: int):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return job_data(job)

    @app.post("/api/ask")
    def ask(request: QueryRequest):
        space_or_404(request.space_id)
        previous = None
        recent = ()
        if request.conversation_id is not None:
            previous = conversations.get(request.conversation_id,
                                         space_id=request.space_id)
            if previous is None:
                raise HTTPException(404, "Conversation not found in this research space")
            recent = tuple((turn.question, turn.answer.get("text", ""))
                           for turn in conversations.turns(previous.id, limit=3))
        try:
            standalone = standalone_question(request.question, recent, generate)
            selected = tuple(request.paper_ids) if request.paper_ids is not None else None
            region = (VisualRegion(**request.visual_region.model_dump())
                      if request.visual_region else None)
            visual = (region is not None or request.visual_mode == "on"
                      or (request.visual_mode == "auto"
                          and is_explicit_visual_question(standalone)
                          and (get_setting("PAPERANCHOR_PROVIDER").lower()
                               or "deepseek") in ("deepseek", "glm", "ollama")))
            if visual:
                pages = choose_visual_pages(library, retriever, standalone,
                                            space_id=request.space_id,
                                            paper_ids=selected, region=region)
                result = answer_visual(standalone, pages)
            else:
                answer_mode = get_setting("PAPERANCHOR_ANSWER_MODE").lower() or "direct"
                if answer_mode not in ("direct", "agent"):
                    raise ValueError("PAPERANCHOR_ANSWER_MODE must be direct or agent")
                if request.retrieval_profile != "legacy":
                    if request.enable_decomposition:
                        raise ValueError("Expanded evidence cannot be combined with question decomposition")
                    scoped_papers = library.list_papers(space_id=request.space_id)
                    title = request.title_hint.strip()
                    if not title and selected is not None and len(selected) == 1:
                        title = next((paper.title for paper in scoped_papers
                                      if paper.id == selected[0]), "")
                    result = direct_question(
                        retriever, standalone, generate, limit=request.limit,
                        space_id=request.space_id,
                        allow_general_knowledge=request.allow_general_knowledge,
                        collection=library, paper_ids=selected,
                        prepare_context=lambda question: evidence_profile.prepare(
                            question, title=title, paper_ids=selected,
                            scope_paper_ids=tuple(paper.id for paper in scoped_papers),
                            profile_name=request.retrieval_profile),
                    )
                elif answer_mode == "agent" or request.enable_decomposition:
                    result = research_question(
                        retriever, standalone, generate, limit=request.limit,
                        space_id=request.space_id,
                        allow_general_knowledge=request.allow_general_knowledge,
                        enable_decomposition=request.enable_decomposition,
                        collection=library, paper_ids=selected,
                    )
                else:
                    result = direct_question(
                        retriever, standalone, generate, limit=request.limit,
                        space_id=request.space_id,
                        allow_general_knowledge=request.allow_general_knowledge,
                        collection=library, paper_ids=selected,
                    )
        except ModelConfigError as error:
            raise HTTPException(400, str(error)) from error
        except ModelRequestError as error:
            raise HTTPException(503, str(error)) from error
        except EvidenceProfileUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except (ValueError, pymupdf.FileDataError) as error:
            raise HTTPException(422, str(error)) from error
        except FileNotFoundError as error:
            raise HTTPException(410, str(error)) from error
        answer = result.answer
        if answer.status == "answered":
            cited = set(answer.cited_evidence_ids)
            for item in answer.evidence:
                if item.evidence_id in cited and item.bbox != (0, 0, 0, 0):
                    body = f"Referenced in answer to: {request.question}"
                    existing = library.list_annotations(item.paper_id, item.page_number)
                    if not any(note.author == "agent" and note.bbox == item.bbox
                               and note.body == body for note in existing):
                        library.add_annotation(item.paper_id, item.page_number,
                                               item.bbox, body, author="agent")
        payload = {**asdict(answer), "basis": result.basis,
                "attempted_queries": result.attempted_queries,
                "subquestions": [asdict(task) for task in result.subquestions],
                "follow_up_question": result.follow_up_question}
        conversation = previous or conversations.create(request.space_id, request.question)
        encoded = jsonable_encoder(payload)
        conversations.append(conversation.id, request.question, standalone, encoded)
        return {**encoded, "conversation_id": conversation.id,
                "standalone_question": standalone}

    @app.get("/api/papers/{paper_id}/annotations")
    def annotations(paper_id: int, page: int):
        paper_or_404(paper_id)
        return [asdict(item) for item in library.list_annotations(paper_id, page)]

    @app.post("/api/annotations")
    def add_annotation(request: AnnotationRequest):
        paper_or_404(request.paper_id)
        try:
            return asdict(library.add_annotation(request.paper_id, request.page_number,
                                                 request.bbox, request.body, author="user"))
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    return app
