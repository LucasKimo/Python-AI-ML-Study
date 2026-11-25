"""가장 작은 LangGraph — 줄 수 세기 → 요약."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class MiniState(TypedDict):
    text: str
    line_count: int
    summary: str


def count_lines(state: MiniState) -> dict:
    lines = [ln for ln in state["text"].splitlines() if ln.strip()]
    print(f"[node:count_lines] {len(lines)}줄")
    return {"line_count": len(lines)}


def make_summary(state: MiniState) -> dict:
    preview = state["text"].strip().replace("\n", " / ")[:80]
    summary = f"총 {state['line_count']}줄. 미리보기: {preview}"
    print(f"[node:make_summary] {summary}")
    return {"summary": summary}


def build_app():
    g = StateGraph(MiniState)
    g.add_node("count_lines", count_lines)
    g.add_node("make_summary", make_summary)
    g.add_edge(START, "count_lines")
    g.add_edge("count_lines", "make_summary")
    g.add_edge("make_summary", END)
    return g.compile()


if __name__ == "__main__":
    app = build_app()
    result = app.invoke(
        {
            "text": "김: 목표 공유\n이: API 수요일 완료\n박: UI 금요일 제출 결정",
            "line_count": 0,
            "summary": "",
        }
    )
    print("\n최종 상태:")
    for k, v in result.items():
        print(f"  {k}: {v}")
