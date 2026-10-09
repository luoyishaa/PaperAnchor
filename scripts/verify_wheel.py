"""Reject a wheel with missing or stale frontend files."""

import argparse
from pathlib import Path
import sys
from zipfile import ZipFile


def verify_wheel(wheel: Path, source: Path) -> None:
    static = source / "src" / "paperanchor" / "static"
    expected = {f"paperanchor/static/{path.relative_to(static).as_posix()}"
                for path in static.rglob("*") if path.is_file()}
    if "paperanchor/static/index.html" not in expected or not any(
            path.endswith(".js") for path in expected):
        raise ValueError("Frontend must be built before packaging")
    with ZipFile(wheel) as archive:
        actual = {name for name in archive.namelist()
                   if name.startswith("paperanchor/static/") and not name.endswith("/")}
        if actual != expected:
            raise ValueError(f"Frontend files differ: missing={sorted(expected - actual)}, "
                             f"stale={sorted(actual - expected)}")
        for name in sorted(expected):
            path = source / "src" / name
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Frontend content differs: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        verify_wheel(args.wheel, args.source)
    except (OSError, ValueError) as error:
        print(f"Invalid wheel: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Wheel frontend matches source: {args.wheel}")
