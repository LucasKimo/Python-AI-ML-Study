"""회의록을 요약·결정·할 일로 바꾸는 LangGraph.

이론 — 왜 그래프인가
--------------------
한 함수 안에서 "요약하고, 결정 뽑고, 할 일 뽑고, 점수 매기기"를 순서대로 호출해도
결과는 같습니다. 그래프로 나누는 이유는 각 단계가
  - 이름을 갖고 (summarize, decisions, ...)
  - 자기 출력만 상태에 남기고
  - 순서를 데이터로 바꿀 수 있기 때문입니다.
지금은 한 줄로만 흐르지만, 나중에 "결정이 없으면 액션을 건너뛴다" 같은
조건 분기를 add_conditional_edges로 넣을 자리가 노드 경계입니다.

  START → summarize → decisions → actions → quality → END

START와 END는 개발자가 만든 함수가 아닙니다.
"여기서 들어온다 / 여기서 끝난다"를 표시하는 LangGraph의 가상 지점입니다.

add_node(이름, 함수)로 칸을 등록하고,
add_edge(이전, 다음)로 화살표를 긋습니다.
compile()이 그 설계도를 실행 가능한 앱으로 바꿉니다.
invoke(초기상태)는 입구부터 출구까지 한 번 돌리고, 합쳐진 최종 상태를 돌려줍니다.

노드는 backend/app/graph/nodes.py 에 있습니다.
각 노드는 OpenAI에 회의록을 보내고 JSON만 받아 상태 칸을 채웁니다.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.app.graph.nodes import (
    actions_node,
    decisions_node,
    quality_node,
    summarize_node,
)
from backend.app.graph.state import MeetingState


def build_graph():
    """상태 타입을 고정한 뒤 노드와 화살표를 등록하고 컴파일합니다."""
    graph = StateGraph(MeetingState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("decisions", decisions_node)
    graph.add_node("actions", actions_node)
    graph.add_node("quality", quality_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "decisions")
    graph.add_edge("decisions", "actions")
    graph.add_edge("actions", "quality")
    graph.add_edge("quality", END)

    return graph.compile()


_APP = None


def get_graph():
    """컴파일 결과를 프로세스 안에 하나만 둡니다. 요청마다 그래프를 다시 짜지 않습니다."""
    global _APP
    if _APP is None:
        _APP = build_graph()
    return _APP


def run_meeting_analysis(
    *,
    meeting_id: int,
    title: str,
    transcript: list[dict[str, Any]],
) -> dict[str, Any]:
    """Invoke LangGraph and return analysis fields. Analysis always uses OpenAI."""
    app = get_graph()
    initial: MeetingState = {
        "meeting_id": meeting_id,
        "title": title,
        "transcript": transcript,
        "summary": "",
        "decisions": [],
        "action_items": [],
        "accuracy_score": 0.0,
        "errors": [],
    }
    final = app.invoke(initial)
    return {
        "summary": final.get("summary", ""),
        "decisions": final.get("decisions", []),
        "action_items": final.get("action_items", []),
        "accuracy_score": final.get("accuracy_score", 0.0),
        "errors": final.get("errors", []),
        "engine": "langgraph",
    }
