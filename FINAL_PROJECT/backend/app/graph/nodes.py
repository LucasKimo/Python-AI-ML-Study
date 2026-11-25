"""LangGraph 노드 네 개: summarize → decisions → actions → quality.

노드는 "상태를 받아, 바꿀 칸만 dict로 돌려주는 함수"입니다.
전체 MeetingState를 다시 조립하지 않습니다. 합치기는 workflow.py의 그래프가 합니다.

요약·결정·액션은 OpenAI Chat Completions로 만듭니다.
모델에게 "JSON만 답하라"고 시키고, 그 JSON을 파싱해 상태 칸에 넣습니다.
마크다운 코드펜스(```json)로 감싸는 경우가 있어 _llm_json이 껍질을 벗깁니다.

quality는 LLM을 부르지 않습니다.
요약이 있으면 0.4, 결정이 있으면 0.3, 할 일이 있으면 0.3을 더해
"분석이 비어 있지 않은가"를 대략 점수화합니다. 사람 채점과 같은 정확도는 아닙니다.
"""

from __future__ import annotations

import json
import re
from typing import Any

from shared import config

from backend.app.graph.state import MeetingState


def _join_transcript(utterances: list[dict[str, Any]]) -> str:
    lines = []
    for u in utterances:
        lines.append(f"{u.get('speaker', 'Speaker')}: {u.get('text', '')}")
    return "\n".join(lines)


def _require_openai() -> None:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env에 키를 설정한 뒤 서버를 다시 시작하세요.")


def _llm_json(prompt: str) -> dict[str, Any]:
    from openai import OpenAI

    _require_openai()
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a meeting analyst. Reply with valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "{}").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return json.loads(content)


def summarize_node(state: MeetingState) -> dict[str, Any]:
    """회의록 전체를 짧은 요약 한 덩어리로 바꿉니다. 빈 회의록이면 LLM을 부르지 않습니다."""
    utterances = state.get("transcript") or []
    title = state.get("title") or "회의"
    text = _join_transcript(utterances)

    if not text.strip():
        return {
            "summary": "회의록이 비어 있어 요약할 내용이 없습니다.",
            "errors": ["empty_transcript"],
        }

    data = _llm_json(
        f"회의 제목: {title}\n회의록:\n{text}\n\n"
        'JSON으로만 답하세요: {"summary": "..."}'
    )
    summary = str(data.get("summary") or "").strip()
    if not summary:
        raise RuntimeError("OpenAI 응답에 summary가 없습니다.")
    return {"summary": summary, "errors": []}


def decisions_node(state: MeetingState) -> dict[str, Any]:
    """'그렇게 하자'로 모인 문장만 리스트로 뽑습니다. 요약 칸은 건드리지 않습니다."""
    utterances = state.get("transcript") or []
    text = _join_transcript(utterances)
    if not text.strip():
        return {"decisions": [], "errors": []}

    data = _llm_json(
        f"회의록:\n{text}\n\n"
        '결정사항만 JSON으로: {"decisions": ["..."]}'
    )
    decisions = [str(d) for d in (data.get("decisions") or [])]
    return {"decisions": decisions[:15], "errors": []}


def actions_node(state: MeetingState) -> dict[str, Any]:
    """누가 무엇을 할지. assignee / task / status 세 칸으로 모양을 고정합니다."""
    utterances = state.get("transcript") or []
    text = _join_transcript(utterances)
    if not text.strip():
        return {"action_items": [], "errors": []}

    data = _llm_json(
        f"회의록:\n{text}\n\n"
        '액션아이템 JSON: {"action_items": '
        '[{"assignee":"...","task":"...","status":"open"}]}'
    )
    items = data.get("action_items") or []
    normalized = []
    for item in items[:15]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "assignee": str(item.get("assignee") or "미정"),
                "task": str(item.get("task") or ""),
                "status": str(item.get("status") or "open"),
            }
        )
    return {
        "action_items": normalized,
        "errors": [],
    }


def quality_node(state: MeetingState) -> dict[str, Any]:
    """앞 노드가 채운 칸을 보고 0~1 점수를 만듭니다. 모델 호출은 없습니다."""
    score = 0.0
    if state.get("summary"):
        score += 0.4
    if state.get("decisions"):
        score += 0.3
    if state.get("action_items"):
        score += 0.3
    if state.get("errors"):
        score = max(0.0, score - 0.1 * min(3, len(state["errors"])))
    return {"accuracy_score": round(score, 2), "errors": []}
