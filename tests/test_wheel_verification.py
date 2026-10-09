import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest


script = Path(__file__).resolve().parents[1] / "scripts" / "verify_wheel.py"
spec = importlib.util.spec_from_file_location("verify_wheel", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wheel_requires_matching_frontend_content(tmp_path):
    static = tmp_path / "src" / "paperanchor" / "static"
    static.mkdir(parents=True)
    (static / "index.html").write_text("new frontend")
    (static / "app.js").write_text("new script")
    wheel = tmp_path / "package.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("paperanchor/static/index.html", "old frontend")
        archive.writestr("paperanchor/static/app.js", "new script")
    with pytest.raises(ValueError, match="content differs"):
        module.verify_wheel(wheel, tmp_path)
    with ZipFile(wheel, "w") as archive:
        for file in static.iterdir():
            archive.write(file, f"paperanchor/static/{file.name}")
    module.verify_wheel(wheel, tmp_path)
