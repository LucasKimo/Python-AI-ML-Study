"""Start Backend, ASR, and Gradio as child processes."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def spawn(args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def main() -> None:
    py = sys.executable
    procs = [
        spawn([py, "-m", "backend.app.main"]),
        spawn([py, "-m", "asr.app.main"]),
    ]
    time.sleep(1.5)
    procs.append(spawn([py, str(ROOT / "frontend" / "app.py")]))

    print("Started Backend(:8000), ASR(:8001), Gradio(:7860)")
    print("Press Ctrl+C to stop all.")
    try:
        while True:
            time.sleep(1)
            for p in procs:
                if p.poll() is not None:
                    raise SystemExit(f"Process exited early: {p.args} code={p.returncode}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    main()
