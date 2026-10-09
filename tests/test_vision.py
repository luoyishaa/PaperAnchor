from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from paperanchor import vision
from paperanchor.llm import ModelRequestError
from paperanchor.library import PaperLibrary
from paperanchor.model_config import ModelConfig
from paperanchor.vision import (VisualPage, VisualRegion, answer_visual,
                                 choose_visual_pages, is_explicit_visual_question)


def test_empty_visual_provider_answer_uses_safe_request_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = ModelConfig("glm", None, "test-key", "glm-4.7-flash", "openai")

    class EmptyClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=""))])

    monkeypatch.setattr(vision, "load_model_config", lambda: config)
    monkeypatch.setattr(vision, "OpenAI", EmptyClient)
    with pytest.raises(ModelRequestError) as caught:
        vision._provider_completion("system", [])
    assert caught.value.provider_code == "empty_answer"


def make_pdf(path: Path) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Table 2: Method A scores 0.81")
        document.save(path)


def test_visual_page_answer_has_page_citation_without_invented_region(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    pages = (VisualPage(1, "Paper", pdf, 1),)
    seen = []

    def complete(system, content):
        seen.extend(content)
        return "Method A scores 0.81. [E1]"

    result = answer_visual("What does Table 2 show?", pages, complete=complete)
    assert result.answer.status == "answered"
    assert result.answer.evidence[0].page_number == 1
    assert result.answer.evidence[0].bbox == (0, 0, 0, 0)
    assert result.answer.evidence[0].source_spans == ()
    images = [part for part in seen if part["type"] == "image_url"]
    assert len(images) == 1
    assert images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_user_region_is_validated_and_retained_as_source(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf)
    library = PaperLibrary(tmp_path / "library.sqlite3")
    paper_id = library.index_pdf(pdf).paper_id
    region = VisualRegion(paper_id, 1, (50, 50, 200, 100))
    page = choose_visual_pages(library, library, "Read this", region=region)
    assert page[0].bbox == region.bbox
    result = answer_visual("Read this", page,
                           complete=lambda _system, _content: "0.81 [E1]")
    assert result.answer.evidence[0].source_spans[0].bbox == region.bbox
    with pytest.raises(ValueError, match="outside the page"):
        answer_visual("Read this", (VisualPage(paper_id, "Paper", pdf, 1,
                                            (0, 0, 10000, 10000)),),
                      complete=lambda _system, _content: "never called")


def test_explicit_visual_intent_is_narrow() -> None:
    assert is_explicit_visual_question("图 2 的最高柱是多少？")
    assert is_explicit_visual_question("What is the largest value in Table 3?")
    assert not is_explicit_visual_question("Compare the proposed methods")


def test_visual_comparison_selects_one_page_per_paper(tmp_path: Path) -> None:
    library = PaperLibrary(tmp_path / "library.sqlite3")
    identifiers = []
    for number in (1, 2):
        path = tmp_path / f"paper-{number}.pdf"
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), f"Table 2: Method {number} scores 0.{number}1")
            document.save(path)
        identifiers.append(library.index_pdf(path).paper_id)
    pages = choose_visual_pages(library, library, "Compare Table 2",
                                paper_ids=tuple(identifiers))
    assert [page.paper_id for page in pages] == identifiers


def test_glm_visual_request_uses_free_vision_model_not_text_model(
        monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config = ModelConfig("glm", "https://open.bigmodel.cn/api/paas/v4", "test-key",
                         "glm-4.7-flash", "openai", thinking_mode="disabled")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="0.81 [E1]"))])

    monkeypatch.setattr(vision, "load_model_config", lambda: config)
    monkeypatch.setattr(vision, "OpenAI", FakeClient)
    assert vision._provider_completion("system", [{"type": "text", "text": "Table 2"}]) == "0.81 [E1]"
    assert captured["model"] == "glm-4.6v-flash"
    assert "extra_body" not in captured
    assert captured["client"]["max_retries"] == 0
