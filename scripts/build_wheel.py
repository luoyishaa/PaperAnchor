"""Build and verify a wheel without retaining assets from an older frontend build."""

from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from verify_wheel import verify_wheel


def main() -> None:
    source = Path(__file__).resolve().parents[1]
    build = (source / "build").resolve()
    if build.parent != source or build.name != "build":
        raise RuntimeError("Refusing to remove an unexpected build directory")
    if build.exists():
        shutil.rmtree(build)
    output = source / "dist"
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(output)],
                   cwd=source, check=True)
    version = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    wheels = sorted(output.glob(f"paperanchor-{version}-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one PaperAnchor wheel in {output}; found {len(wheels)}")
    verify_wheel(wheels[0], source)
    print(f"Verified release wheel: {wheels[0]}")


if __name__ == "__main__":
    main()
