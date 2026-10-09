"""Exercise an installed wheel's browser workflow without a provider API key."""

from pathlib import Path
import tempfile
import time

from fastapi.testclient import TestClient
import numpy as np
import paperanchor
import pymupdf

from paperanchor.web import create_app


class Embedding:
    name = "wheel-smoke"

    def embed_passages(self, texts):
        return [np.array([1.0, 0.0], dtype=np.float32) for _ in texts]

    def embed_query(self, _text):
        return np.array([1.0, 0.0], dtype=np.float32)


def main() -> None:
    package_path = Path(paperanchor.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"Test the installed wheel, not the source tree: {package_path}")
    with tempfile.TemporaryDirectory(prefix="paperanchor-wheel-") as directory:
        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 90), "Retrieval augmented generation uses retrieved evidence.")
            pdf = document.tobytes()
        app = create_app(Path(directory) / "library.sqlite3",
                         generate=lambda *_: "It uses retrieved evidence [E1].",
                         embedding_provider=Embedding())
        with TestClient(app) as client:
            home = client.get("/")
            assert home.status_code == 200 and "assets/index-" in home.text
            uploaded = client.post("/api/papers", files={
                "file": ("sample.pdf", pdf, "application/pdf")})
            assert uploaded.status_code == 200, uploaded.text
            paper = uploaded.json()["paper"]
            job_id = uploaded.json()["semantic_job_id"]
            for _ in range(100):
                job = client.get(f"/api/jobs/{job_id}").json()
                if job["state"] in ("succeeded", "failed"):
                    break
                time.sleep(0.02)
            assert job["state"] == "succeeded", job
            answer = client.post("/api/ask", json={"question": "What does RAG use?"})
            assert answer.status_code == 200, answer.text
            evidence = answer.json()["evidence"][0]
            assert evidence["paper_id"] == paper["id"]
            assert evidence["bbox"][2] > evidence["bbox"][0]
            image = client.get(f"/api/papers/{paper['id']}/pages/1/image")
            assert image.content.startswith(b"\x89PNG")
            assert client.get(f"/api/papers/{paper['id']}/file").content == pdf
    print(f"Installed wheel passed upload -> index -> answer -> PDF location: {package_path}")


if __name__ == "__main__":
    main()
