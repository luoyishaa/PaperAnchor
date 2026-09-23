"""Local HTTP interface for the paper library and located evidence."""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import re
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import pymupdf
from pydantic import BaseModel, Field

from .library import PaperLibrary
from .llm import ModelConfigError, generate_with_config
from .rag import answer_question


MAX_UPLOAD_BYTES = 40 * 1024 * 1024


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class AnnotationRequest(BaseModel):
    paper_id: int
    page_number: int
    bbox: tuple[float, float, float, float]
    body: str = Field(min_length=1, max_length=4000)


def _paper_data(paper) -> dict:
    return {
        "id": paper.id,
        "title": paper.title,
        "page_count": paper.page_count,
        "passage_count": paper.passage_count,
        "indexed_at": paper.indexed_at,
        "file_available": paper.file_path.is_file(),
    }


def create_app(database_path: Path, *, generate=generate_with_config) -> FastAPI:
    library = PaperLibrary(database_path)
    upload_dir = database_path.parent / "uploads"
    static_dir = Path(__file__).parent / "static"
    app = FastAPI(title="PaperAnchor", docs_url="/api/docs", redoc_url=None)
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

    @app.get("/api/papers")
    def papers():
        return [_paper_data(paper) for paper in library.list_papers()]

    @app.post("/api/papers")
    async def upload_paper(file: UploadFile = File(...)):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(400, "Choose a PDF file")
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename).name)[:100]
        temporary = upload_dir / f".upload-{uuid4().hex}.pdf"
        digest = sha256()
        size = 0
        try:
            with temporary.open("wb") as output:
                header = await file.read(5)
                if header != b"%PDF-":
                    raise HTTPException(422, "File does not have a PDF header")
                digest.update(header)
                output.write(header)
                size += len(header)
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "PDF exceeds 40 MB")
                    digest.update(chunk)
                    output.write(chunk)
            if not size:
                raise HTTPException(400, "PDF is empty")
            destination = upload_dir / f"{digest.hexdigest()[:20]}-{safe_name}"
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
            return {"paper": _paper_data(paper), "changed": result.changed}
        finally:
            temporary.unlink(missing_ok=True)
            await file.close()

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
            return {"width": page.rect.width, "height": page.rect.height,
                    "page_number": page_number}

    @app.get("/api/papers/{paper_id}/pages/{page_number}/image")
    def page_image(paper_id: int, page_number: int):
        paper = paper_or_404(paper_id)
        if not 1 <= page_number <= paper.page_count:
            raise HTTPException(404, "Page not found")
        with pymupdf.open(paper.file_path) as document:
            image = document[page_number - 1].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6),
                                                           alpha=False)
            return Response(image.tobytes("png"), media_type="image/png")

    @app.get("/api/search")
    def search(q: str, limit: int = 5):
        if not 1 <= limit <= 20:
            raise HTTPException(422, "limit must be between 1 and 20")
        return [asdict(hit) for hit in library.search(q, limit)]

    @app.post("/api/ask")
    def ask(request: QueryRequest):
        try:
            answer = answer_question(library, request.question, generate, limit=request.limit)
        except ModelConfigError as error:
            raise HTTPException(400, str(error)) from error
        if answer.status == "answered":
            cited = set(answer.cited_evidence_ids)
            for item in answer.evidence:
                if item.evidence_id in cited:
                    library.add_annotation(
                        item.paper_id, item.page_number, item.bbox,
                        f"Referenced in answer to: {request.question}", author="agent"
                    )
        return asdict(answer)

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
