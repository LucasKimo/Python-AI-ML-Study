"""본편과 비슷한 미니 LangGraph — 요약 → 결정 → 액션."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class MeetingState(TypedDict):
    transcript: str
    summary: str
    decisions: list[str]
    actions: list[str]


def summarize(state: MeetingState) -> dict[str, Any]:
    lines = [ln.strip() for ln in state["transcript"].splitlines() if ln.strip()]
    return {
        "summary": f"발화 {len(lines)}개. 핵심: {lines[0] if lines else '(없음)'}"
    }


def extract_decisions(state: MeetingState) -> dict[str, Any]:
    decisions = []
    for line in state["transcript"].splitlines():
        if any(k in line for k in ("결정", "합의", "확정")):
            decisions.append(line.strip())
    return {"decisions": decisions}


def extract_actions(state: MeetingState) -> dict[str, Any]:
    actions = []
    for line in state["transcript"].splitlines():
        if "액션" in line or "하기로" in line:
            actions.append(line.strip())
        elif re.search(r".+이 .+하기로", line):
            actions.append(line.strip())
    return {"actions": actions}


def build_app():
    g = StateGraph(MeetingState)
    g.add_node("summarize", summarize)
    g.add_node("decisions", extract_decisions)
    g.add_node("actions", extract_actions)
    g.add_edge(START, "summarize")
    g.add_edge("summarize", "decisions")
    g.add_edge("decisions", "actions")
    g.add_edge("actions", END)
    return g.compile()


SAMPLE = """김매니저: 오늘 목표를 공유합니다.
이개발: API는 수요일까지 완료하기로 했습니다.
박기획: UI 마감을 금요일로 결정합니다.
김매니저: 액션: 이개발이 테스트 케이스를 작성하기로 합니다.
"""


if __name__ == "__main__":
    app = build_app()
    result = app.invoke(
        {
            "transcript": SAMPLE,
            "summary": "",
            "decisions": [],
            "actions": [],
        }
    )
    print("=== LangGraph 회의 분석 미니 결과 ===")
    print("요약:", result["summary"])
    print("결정:")
    for d in result["decisions"]:
        print(" -", d)
    print("액션:")
    for a in result["actions"]:
        print(" -", a)
