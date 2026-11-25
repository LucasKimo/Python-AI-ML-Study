"""Seed sample meeting and run LangGraph analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.graph.workflow import run_meeting_analysis
from shared import db


def main() -> None:
    db.init_db()
    sample_path = ROOT / "data" / "samples" / "sample_meeting.txt"
    lines = [
        ln.strip()
        for ln in sample_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    meeting = db.create_meeting(
        title="스프린트 계획 회의 (샘플)",
        participants=["김매니저", "이개발", "박기획"],
        agenda="스프린트 목표 및 일정",
    )

    utterances = []
    for i, line in enumerate(lines, start=1):
        if ":" in line:
            speaker, text = line.split(":", 1)
        else:
            speaker, text = "Speaker", line
        utterances.append(
            {
                "seq": i,
                "speaker": speaker.strip(),
                "text": text.strip(),
                "timestamp": None,
            }
        )

    result = run_meeting_analysis(
        meeting_id=meeting["id"],
        title=meeting["title"],
        transcript=utterances,
    )
    updated = db.update_meeting(
        meeting["id"],
        transcript=utterances,
        summary=result["summary"],
        decisions=result["decisions"],
        action_items=result["action_items"],
        status="analyzed",
    )
    print(
        f"Seeded meeting id={updated['id']} "
        f"engine={result['engine']} score={result['accuracy_score']}"
    )


if __name__ == "__main__":
    main()
