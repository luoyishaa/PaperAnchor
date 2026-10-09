from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import sqlite3

import pymupdf
import pytest

from paperanchor.library import PaperLibrary
from paperanchor.spaces import SpaceStore
from paperanchor.vision import answer_visual, choose_visual_pages
from paperanchor.visual_index import detect_page, visual_label


def diagram_pdf(path: Path, *, number=3, prefix_page=False, vector=True):
    with pymupdf.open() as doc:
        if prefix_page:
            page = doc.new_page()
            page.insert_text((72, 72), f"Figure {number} is discussed here, but appears later.")
        page = doc.new_page()
        if vector:
            page.draw_rect((80, 100, 340, 300))
            page.draw_line((100, 270), (310, 130))
            page.insert_text((100, 290), "Axis and legend: blue baseline")
        else:
            # Two bitmap panels belong to one figure caption.
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
            pix.clear_with(180)
            page.insert_image((80, 100, 200, 280), stream=pix.tobytes("png"))
            page.insert_image((210, 100, 340, 280), stream=pix.tobytes("png"))
        page.insert_text((80, 330), f"Figure {number}: Comparison of the blue and red models.")
        doc.save(path)


def test_import_selects_actual_figure_over_prose_and_preserves_all_panels(tmp_path):
    path = tmp_path / "panels.pdf"
    diagram_pdf(path, prefix_page=True, vector=False)
    lib = PaperLibrary(tmp_path / "lib.db")
    paper_id = lib.index_pdf(path).paper_id
    regions = lib.visuals.regions((paper_id,))
    assert len(regions) == 1
    region = regions[0].region
    assert region.page_number == 2
    assert region.bbox[0] <= 80 and region.bbox[2] >= 340
    assert region.bbox[1] <= 100 and region.bbox[3] >= 330
    pages = choose_visual_pages(lib, lib, "图3里蓝色和红色有什么区别？", paper_ids=(paper_id,))
    assert pages[0].page_number == 2
    assert pages[0].region_id == regions[0].region_id
    assert pages[0].include_page_context


def test_vector_figure_has_region_citation_and_context_in_same_call(tmp_path):
    path = tmp_path / "vector.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    paper_id = lib.index_pdf(path).paper_id
    pages = choose_visual_pages(lib, lib, "What does Figure 3 show?")
    content = []
    def complete(_system, payload):
        content.extend(payload)
        return "Blue is the baseline. [E1]"
    result = answer_visual("Read Figure 3", pages, complete=complete)
    evidence = result.answer.evidence[0]
    assert evidence.paper_id == paper_id
    assert evidence.source_hash == sha256(path.read_bytes()).hexdigest()
    assert evidence.source_kind == "visual_region"
    assert evidence.source_label == "Figure 3"
    assert evidence.selection_method == "caption_graphics"
    assert evidence.bbox == evidence.source_spans[0].bbox
    assert len([part for part in content if part["type"] == "image_url"]) == 2


def test_caption_only_is_not_an_invented_region(tmp_path):
    path = tmp_path / "caption.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Table 2: scores are missing from this page")
        doc.save(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    paper_id = lib.index_pdf(path).paper_id
    assert lib.visuals.regions((paper_id,)) == ()
    page = choose_visual_pages(lib, lib, "Table 2 scores")[0]
    assert page.bbox is None
    assert page.region_id is None


def test_region_scope_and_one_source_per_selected_paper(tmp_path):
    lib = PaperLibrary(tmp_path / "lib.db")
    spaces = SpaceStore(lib.database_path)
    scope = spaces.create_space("Selected research")
    ids = []
    for i in range(3):
        path = tmp_path / f"paper-{i}.pdf"
        diagram_pdf(path)
        pid = lib.index_pdf(path).paper_id
        ids.append(pid)
        if i < 2:
            spaces.add_paper(scope.id, pid)
    selected = choose_visual_pages(lib, lib, "Compare Figure 3", space_id=scope.id,
                                   paper_ids=tuple(ids[:2]))
    assert [page.paper_id for page in selected] == ids[:2]
    assert all(page.region_id is not None for page in selected)
    with pytest.raises(ValueError, match="belong"):
        choose_visual_pages(lib, lib, "Figure 3", space_id=scope.id, paper_ids=(ids[2],))


def test_reimport_rebuilds_regions_and_delete_cascades(tmp_path):
    path = tmp_path / "figure.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    first = lib.visuals.regions((pid,))[0]
    assert not lib.index_pdf(path).changed
    assert lib.visuals.regions((pid,))[0].region_id == first.region_id
    path.unlink()
    diagram_pdf(path, number=4)
    with pytest.raises(ValueError, match="changed"):
        lib.visuals.regions((pid,))
    assert lib.index_pdf(path).changed
    second = lib.visuals.regions((pid,))[0]
    assert second.region.number == "4"
    assert second.region_id != first.region_id
    lib.delete_paper(pid)
    with sqlite3.connect(lib.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM visual_regions").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM visual_index_state").fetchone()[0] == 0


def test_existing_database_backfill_and_version_change(tmp_path, monkeypatch):
    path = tmp_path / "old.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    with lib._connect() as connection:
        connection.execute("DELETE FROM visual_regions")
        connection.execute("DELETE FROM visual_index_state")
    first = lib.visuals.regions((pid,))[0]
    monkeypatch.setattr("paperanchor.visual_index.DETECTOR_VERSION", 99)
    second = lib.visuals.regions((pid,))[0]
    assert second.region_id != first.region_id


def test_changed_pdf_after_selection_never_calls_model(tmp_path):
    path = tmp_path / "changed.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    pages = choose_visual_pages(lib, lib, "Figure 3", paper_ids=(pid,))
    path.write_bytes(path.read_bytes() + b"\n%changed")
    def unexpected(*_):
        raise AssertionError("stale region must not be sent")
    with pytest.raises(ValueError, match="changed"):
        answer_visual("Figure 3", pages, complete=unexpected)


def test_unreadable_or_invalid_citation_is_not_promoted(tmp_path):
    path = tmp_path / "figure.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    pages = choose_visual_pages(lib, lib, "Figure 3", paper_ids=(pid,))
    assert answer_visual("Figure 3", pages, complete=lambda *_: "Unclear.").answer.status == "uncited"
    assert answer_visual("Figure 3", pages, complete=lambda *_: "42 [E9]").answer.status == "invalid_citation"
    with pytest.raises(ValueError, match="outside the page"):
        answer_visual("Figure 3", (replace(pages[0], bbox=(0, 0, 10000, 10000)),),
                      complete=lambda *_: "unused")


def test_label_matching_distinguishes_table_figure_and_whole_number():
    assert visual_label("Fig. 20(a)") == ("figure", "20")
    assert visual_label("表 2") == ("table", "2")
    assert visual_label("Figure 2") != visual_label("Table 2")


def test_multiple_captions_in_one_extraction_block_stay_separate():
    class MergedCaptionPage:
        number = 0
        rect = pymupdf.Rect(0, 0, 600, 800)
        derotation_matrix = pymupdf.Matrix(1, 1)

        def get_text(self, mode):
            return {"blocks": [{"type": 0, "bbox": (80, 310, 540, 330),
                "lines": [
                    {"bbox": (80, 310, 270, 330),
                     "spans": [{"text": "Figure 1: Blue model."}]},
                    {"bbox": (350, 310, 540, 330),
                     "spans": [{"text": "Figure 2: Red model."}]},
                ]}]}

        def get_image_info(self):
            return []

        def cluster_drawings(self):
            return [pymupdf.Rect(80, 100, 270, 290),
                    pymupdf.Rect(350, 100, 540, 290)]

    regions = detect_page(MergedCaptionPage())
    assert [region.number for region in regions] == ["1", "2"]
    assert regions[0].caption == "Figure 1: Blue model."
    assert regions[1].caption == "Figure 2: Red model."
    assert regions[0].bbox[2] < regions[1].bbox[0]
    assert "Figure 2" not in regions[0].context
    assert "Figure 1" not in regions[1].context


def test_split_caption_label_keeps_its_continuation():
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_rect((80, 100, 340, 280))
        page.insert_text((80, 310), "Figure\n3: Comparison of models.\nBlue is the baseline.")
        regions = detect_page(page)
        assert len(regions) == 1
        assert regions[0].number == "3"
        assert "Blue is the baseline." in regions[0].caption


def test_unlabeled_graphics_do_not_receive_invented_figure_numbers():
    with pymupdf.open() as document:
        page = document.new_page()
        page.draw_rect((80, 100, 340, 280))
        page.insert_text((80, 310), "A diagram without an identifying caption.")
        assert detect_page(page) == ()


@pytest.mark.parametrize("caption_above", [True, False])
def test_borderless_table_keeps_header_rows_and_caption(tmp_path, caption_above):
    path = tmp_path / "table.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((80, 90 if caption_above else 190), "Table 2: Method results.")
        for y, text in [(120, "Method     Score"), (138, "Baseline     0.7"), (156, "Ours           0.9")]:
            page.insert_text((80, y), text)
        doc.save(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    regions = lib.visuals.regions((pid,))
    assert len(regions) == 1
    r = regions[0].region
    assert r.kind == "table"
    assert r.bbox[1] <= (78 if caption_above else 108)
    assert r.bbox[3] >= (156 if caption_above else 190)
    assert "Baseline" in r.context and "Score" in r.context


def test_raster_table_and_no_false_caption_from_prose(tmp_path):
    path = tmp_path / "raster.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((80, 90), "Table 2: Results in the image.")
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
        pix.clear_with(180)
        page.insert_image((80, 100, 300, 230), stream=pix.tobytes("png"))
        page.insert_text((80, 260), "Table 2 illustrates a general trend of improved results.")
        doc.save(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    assert len(lib.visuals.regions((pid,))) == 1


def test_tall_table_is_not_truncated_by_distance_budget(tmp_path):
    path = tmp_path / "tall.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((80, 90), "Table 1: Complete results.")
        for i in range(30):
            page.insert_text((80, 120 + i*16), f"Method {i}           Score {i}")
        doc.save(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    region = lib.visuals.regions((pid,))[0].region
    assert region.bbox[3] > 584
    assert "Method 29" in region.context


def test_import_failure_rolls_back_text_and_visual_regions(tmp_path, monkeypatch):
    path = tmp_path / "atomic.pdf"
    diagram_pdf(path)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    first = lib.visuals.regions((pid,))[0]
    path.unlink()
    diagram_pdf(path, number=4)
    def fail(*_):
        raise RuntimeError("simulated transaction failure")
    monkeypatch.setattr(lib.visuals, "replace_regions", fail)
    with pytest.raises(RuntimeError, match="transaction"):
        lib.index_pdf(path)
    with lib._connect() as connection:
        assert connection.execute("SELECT file_hash FROM papers WHERE id=?", (pid,)).fetchone()[0] == first.source_hash
        assert connection.execute("SELECT number FROM visual_regions WHERE paper_id=?", (pid,)).fetchone()[0] == "3"
        assert any("Figure 3" in row[0] for row in connection.execute("SELECT text FROM passages"))


def test_rotated_page_uses_unrotated_pdf_coordinates(tmp_path):
    path = tmp_path / "rotate.pdf"
    diagram_pdf(path)
    with pymupdf.open(path) as doc:
        doc[0].set_rotation(90)
        data = doc.tobytes()
    path.write_bytes(data)
    lib = PaperLibrary(tmp_path / "lib.db")
    pid = lib.index_pdf(path).paper_id
    pages = choose_visual_pages(lib, lib, "Figure 3", paper_ids=(pid,))
    answer = answer_visual("Figure 3", pages, complete=lambda *_: "Baseline [E1]").answer
    assert answer.evidence[0].bbox[1] <= 100
    assert answer.status == "answered"
    # A valid unrotated box can extend past the rotated page's height.
    selected = replace(pages[0], bbox=(80, 600, 330, 700), include_page_context=False)
    assert answer_visual("Read selected region", (selected,),
                         complete=lambda *_: "Unreadable [E1]").answer.status == "answered"
