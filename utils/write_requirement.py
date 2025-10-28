#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def sh(cmd: List[str], cwd: Optional[Path] = None, capture: bool = False):
    if capture:
        return subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def has_pipreqs() -> bool:
    try:
        sh([sys.executable, "-m", "pipreqs", "--help"], capture=False)
        return True
    except Exception:
        return False


def run_pip_freeze(output: Path):
    print(f"[generate-requirements] Using pip freeze -> {output}")
    res = sh([sys.executable, "-m", "pip", "freeze"], capture=True)  # type: ignore[arg-type]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(res.stdout, encoding="utf-8")
    print(f"[generate-requirements] Wrote {output} ({len(res.stdout.splitlines())} lines)")


def run_pipreqs(project_dir: Path, output: Path, ignore: List[str]):
    print(f"[generate-requirements] Using pipreqs to infer imports in {project_dir}")
    args = [sys.executable, "-m", "pipreqs", str(project_dir), "--force", "--encoding", "utf-8"]
    if ignore:
        args += ["--ignore", ",".join(ignore)]
    # Try using --savepath if available in this pipreqs version
    try:
        test_args = args + ["--savepath", str(output)]
        sh(test_args)
        print(f"[generate-requirements] Wrote {output}")
        return
    except subprocess.CalledProcessError as e:
        if "unrecognized arguments: --savepath" not in (e.stderr or ""):
            raise
        # Fallback: run without --savepath, then move the generated file
        print("[generate-requirements] --savepath not supported, falling back to default generation path")
        sh(args)

    # pipreqs by default writes requirements.txt into the project dir
    default_out = project_dir / "requirements.txt"
    if not default_out.exists():
        raise FileNotFoundError("pipreqs did not produce requirements.txt as expected.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(default_out.read_text(encoding="utf-8"), encoding="utf-8")
    if output.resolve() != default_out.resolve():
        try:
            default_out.unlink()
        except Exception:
            pass
    print(f"[generate-requirements] Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate requirements.txt for a Python project.")
    parser.add_argument("--mode", choices=["auto", "infer", "freeze"], default="auto",
                        help="auto: prefer pipreqs, else freeze; infer: use pipreqs; freeze: use pip freeze")
    parser.add_argument("--project", default=".", help="Project directory to scan (for infer mode)")
    parser.add_argument("--output", default="requirements.txt", help="Path to write requirements.txt")
    parser.add_argument("--ignore", default="venv,.venv,build,dist,.git,__pycache__",
                        help="Comma-separated paths to ignore in infer mode")
    args = parser.parse_args()

    project_dir = Path(args.project).resolve()
    output = Path(args.output).resolve()
    ignore = [p.strip() for p in args.ignore.split(",") if p.strip()]

    if args.mode == "freeze":
        run_pip_freeze(output)
        return

    if args.mode == "infer":
        if not has_pipreqs():
            print("[generate-requirements] pipreqs not installed. Install with: pip install pipreqs", file=sys.stderr)
            sys.exit(2)
        run_pipreqs(project_dir, output, ignore)
        return

    # auto
    if has_pipreqs():
        try:
            run_pipreqs(project_dir, output, ignore)
            return
        except Exception as e:
            print(f"[generate-requirements] pipreqs failed ({e}); falling back to pip freeze", file=sys.stderr)
    run_pip_freeze(output)


if __name__ == "__main__":
    main()