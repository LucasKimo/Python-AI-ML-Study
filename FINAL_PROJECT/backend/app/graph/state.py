"""LangGraph가 노드 사이로 들고 다니는 회의 분석 상태.

이론 — 상태(State)
------------------
LangGraph의 그래프는 "함수 여러 개를 줄 세운 것"이고,
그 함수들이 같이 보는 공용 노트가 이 TypedDict입니다.

각 노드는 노트 전체를 다시 쓰지 않습니다.
자기가 채운 칸만 dict로 돌려주면, 프레임워크가 기존 상태에 합칩니다.
예: summarize_node가 {"summary": "..."}만 반환하면 title, transcript는 그대로 남습니다.

errors 칸만 합치는 규칙이 다릅니다.
Annotated[list[str], operator.add] 는 "새 리스트를 뒤에 이어 붙인다"는 뜻입니다.
노드가 {"errors": ["empty_transcript"]}를 주면 기존 오류 뒤에 추가되고,
{"errors": []}를 줘도 이전 오류가 지워지지는 않습니다. 빈 리스트를 더하면 그대로이기 때문입니다.
total=False 라서 처음 invoke할 때 일부 칸만 넣어도 됩니다.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import operator


class MeetingState(TypedDict, total=False):
    meeting_id: int
    title: str
    transcript: list[dict[str, Any]]
    summary: str
    decisions: list[str]
    action_items: list[dict[str, Any]]
    accuracy_score: float
    errors: Annotated[list[str], operator.add]
