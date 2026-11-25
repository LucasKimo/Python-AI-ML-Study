"""회의 메모 몇 줄만으로 동작하는 초소형 챗봇.

본편 backend/app/graph/chat.py 의 검색 답변을 의존성 없이 흉내 냅니다.
"""

NOTES = [
    "김매니저: 오늘 스프린트 목표를 공유합니다.",
    "이개발: 백엔드 API 연동은 수요일까지 완료하기로 했습니다.",
    "박기획: UI 초안은 금요일까지 제출하는 것으로 결정합니다.",
]


def answer(question: str) -> str:
    raw = [part for part in question.replace("?", " ").split() if len(part) >= 2]
    tokens = []
    for part in raw:
        for ending in ("은", "는", "이", "가", "을", "를", "에", "요"):
            if part.endswith(ending) and len(part) > len(ending) + 1:
                part = part[: -len(ending)]
                break
        tokens.append(part)
    hits = [line for line in NOTES if any(token in line for token in tokens)]
    if not hits:
        return "관련 발화를 찾지 못했습니다.\n" + "\n".join(f"- {line}" for line in NOTES)
    return "관련 발화:\n" + "\n".join(f"- {line}" for line in hits)


if __name__ == "__main__":
    print(answer("API 연동은 언제까지야?"))
    print()
    print(answer("UI는 언제 제출해?"))
