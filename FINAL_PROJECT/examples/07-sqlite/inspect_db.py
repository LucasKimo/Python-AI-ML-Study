"""demo.db 내용을 간단히 출력."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "demo.db"


def main() -> None:
    if not DB_PATH.exists():
        print("demo.db 가 없습니다. 먼저 python memo_db.py 를 실행하세요.")
        return

    conn = sqlite3.connect(DB_PATH)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    print("테이블:", [t[0] for t in tables])

    try:
        rows = conn.execute(
            "SELECT id, title, created_at FROM memos ORDER BY id"
        ).fetchall()
    except sqlite3.Error as exc:
        print("조회 실패:", exc)
        return

    print(f"메모 {len(rows)}건")
    for r in rows:
        print(f"  #{r[0]} {r[1]} @ {r[2]}")
    conn.close()


if __name__ == "__main__":
    main()
