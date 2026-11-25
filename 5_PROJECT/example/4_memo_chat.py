"""
메모 챗봇 API 예제

- FastAPI로 HTTP API 서버를 만들고,
- LangGraph로 "메모 불러오기 -> LLM 답변" 2단계 흐름(그래프)을 구성합니다.
- LLM은 미리 저장해 둔 메모(MEMOS)에 적힌 내용만 근거로 답합니다.

실행: python 4_memo_chat.py  ->  http://127.0.0.1:8200/docs 에서 API 테스트
"""

import os
from pathlib import Path
from typing import TypedDict

import uvicorn  # FastAPI 앱을 실행하는 ASGI 웹 서버
from fastapi import FastAPI, HTTPException
from langgraph.graph import END, START, StateGraph  # 노드/엣지로 작업 흐름을 정의하는 도구
from pydantic import BaseModel, Field  # 요청/응답 데이터 검증용


# ---------------------------------------------------------------------------
# 1. 데이터: 챗봇이 답변 근거로 사용할 메모 목록
#    실제 서비스라면 DB(SQLite 등)에서 읽어 오겠지만, 예제에서는 코드에 직접 넣어 둡니다.
# ---------------------------------------------------------------------------
MEMOS = [
    {
        "id": 1,
        "title": "스프린트 일정",
        "content": "백엔드 API 연동은 수요일까지 완료한다. UI 초안은 금요일까지 제출한다.",
    },
    {
        "id": 2,
        "title": "역할 분담",
        "content": "테스트 케이스는 이개발이 작성한다. 중간 점검 회의는 다음 주 월요일이다.",
    },
    {
        "id": 3,
        "title": "저장 방식",
        "content": "회의 메모는 SQLite에 저장하기로 결정했다. 벡터 DB는 이번 스프린트에서 쓰지 않는다.",
    },
]


# ---------------------------------------------------------------------------
# 2. 타입 정의
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    """LangGraph의 각 노드가 함께 읽고 쓰는 상태(State).

    노드는 이 딕셔너리의 일부 키만 담은 dict를 반환하고,
    LangGraph가 그 값을 기존 상태에 덮어써서 다음 노드로 넘깁니다.
    """
    question: str  # 사용자의 현재 질문
    history: str   # 이전 대화 내용 (클라이언트가 문자열로 보내 줌)
    context: str   # 답변 근거가 되는 메모 텍스트 (load_memos 노드가 채움)
    answer: str    # LLM이 생성한 최종 답변 (answer 노드가 채움)


class ChatRequest(BaseModel):
    """POST /chat 요청 본문(JSON) 형식."""
    question: str = Field(..., min_length=1)  # 필수값, 빈 문자열이면 422 에러
    history: str = ""                         # 선택값, 없으면 빈 문자열


class ChatResponse(BaseModel):
    """POST /chat 응답 본문(JSON) 형식."""
    answer: str


# ---------------------------------------------------------------------------
# 3. 환경 변수 로드 및 FastAPI 앱 생성
# ---------------------------------------------------------------------------
def load_env() -> None:
    """프로젝트 루트 .env를 읽습니다. 이미 설정된 환경 변수는 그대로 둡니다."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # python-dotenv가 설치되어 있지 않으면 OS 환경 변수만 사용
        return
    # __file__ = .../5_PROJECT/example/4_memo_chat.py
    # parents[0] = example, parents[1] = 5_PROJECT (프로젝트 루트)
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")

# 모듈이 import될 때 바로 .env를 읽어 OPENAI_API_KEY 등을 준비
load_env()

# FastAPI 앱 객체. title/description/version은 /docs(Swagger UI)에 표시됩니다.
app = FastAPI(
    title="메모 챗봇",
    description="미리 저장한 메모만 근거로 답하는 LangGraph API",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# 4. LangGraph 노드 함수들
#    각 노드는 현재 상태(state)를 받아서, 갱신할 키만 담은 dict를 반환합니다.
# ---------------------------------------------------------------------------
def memo_context() -> str:
    """메모 목록을 LLM 프롬프트에 넣기 좋은 하나의 문자열로 합칩니다.

    예)
    #1 스프린트 일정
    백엔드 API 연동은 ...

    #2 역할 분담
    ...
    """
    blocks = []
    for memo in MEMOS:
        blocks.append(f"#{memo['id']} {memo['title']}\n{memo['content']}")
    return "\n\n".join(blocks)  # 메모 사이에 빈 줄을 넣어 구분

def load_memos(state: ChatState) -> dict:
    """미리 저장해 둔 메모 전체를 답변 근거로 넣습니다."""
    # 메모가 적으므로 검색 없이 전체를 그대로 사용합니다.
    # (메모가 많아지면 질문과 관련된 메모만 골라 오는 검색(RAG) 단계로 바꿀 수 있음)
    context = memo_context()
    print(f"[node:load_memos] 메모 {len(MEMOS)}개")  # 서버 로그로 노드 실행 확인
    return {"context": context}  # 상태의 context 키만 갱신


def answer(state: ChatState) -> dict:
    """메모 안에서만 LLM이 답하게 합니다."""
    # openai 패키지는 이 노드에서만 쓰므로 함수 안에서 import
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # .env에 없으면 기본 모델 사용
    history = state["history"].strip() or "(없음)"   # 이전 대화가 비어 있으면 "(없음)" 표시
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # 낮을수록 답변이 일정하고 덜 창의적 -> 사실 기반 답변에 적합
        messages=[
            {
                # system 메시지: 모델의 역할과 답변 규칙을 지정
                "role": "system",
                "content": (
                    "당신은 회의 메모 도우미입니다. "
                    "아래 메모에 적힌 내용만 근거로 한국어로 짧게 답하세요. "
                    "메모에 없는 내용은 추측하지 말고, 메모에 없다고 말하세요. "
                    "답변 앞에 '질문:'이나 '답변:' 같은 말은 붙이지 마세요."
                ),
            },
            {
                # user 메시지: 메모(근거) + 이전 대화 + 현재 질문을 한 번에 전달
                "role": "user",
                "content": (
                    f"메모:\n{state['context']}\n\n"
                    f"이전 대화:\n{history}\n\n"
                    f"질문: {state['question']}"
                ),
            },
        ],
    )
    # 응답 내용이 None일 수 있으므로 빈 문자열로 대체한 뒤 앞뒤 공백 제거
    text = (response.choices[0].message.content or "").strip()
    print(f"[node:answer] {model}")
    return {"answer": text}  # 상태의 answer 키만 갱신


# ---------------------------------------------------------------------------
# 5. 그래프 구성: START -> load_memos -> answer -> END
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ChatState)            # ChatState 형태의 상태를 공유하는 그래프
    graph.add_node("load_memos", load_memos)  # 노드 등록: (이름, 실행할 함수)
    graph.add_node("answer", answer)

    # 엣지: 노드 실행 순서를 정의
    graph.add_edge(START, "load_memos")
    graph.add_edge("load_memos", "answer")
    graph.add_edge("answer", END)
    return graph.compile()  # 실행 가능한 그래프 객체로 변환

# 요청마다 새로 만들지 않도록 서버 시작 시 한 번만 컴파일해 재사용
GRAPH = build_graph()


# ---------------------------------------------------------------------------
# 6. API 엔드포인트
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """서버 상태 확인용. 메모 개수와 API 키 설정 여부를 알려 줍니다."""
    return {
        "status": "ok",
        "memos": len(MEMOS),
        # 키 값 자체는 노출하지 않고, 설정되어 있는지만 True/False로 반환
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
    }


@app.get("/memos")
def list_memos():
    """저장된 메모 목록을 그대로 반환합니다."""
    return MEMOS


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    """질문을 받아 LangGraph를 실행하고 메모 기반 답변을 반환합니다."""
    question = payload.question.strip()
    # min_length=1은 "   "처럼 공백만 있는 입력은 막지 못하므로 한 번 더 검사
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력하세요.")
    # API 키가 없으면 LLM 호출 전에 미리 친절한 에러를 반환
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY가 없습니다. 프로젝트 루트 .env에 키를 넣어 주세요.",
        )

    # 초기 상태를 넣어 그래프 실행 -> 모든 노드를 거친 최종 상태가 반환됨
    result = GRAPH.invoke(
        {
            "question": question,
            "history": payload.history,
            "context": "",  # load_memos 노드가 채움
            "answer": "",   # answer 노드가 채움
        }
    )
    return {"answer": result["answer"]}

if __name__ == "__main__":
    # "모듈이름:앱변수" 형식으로 지정. 파일명이 4_memo_chat.py이므로 "4_memo_chat:app"
    # 문자열로 넘기면 uvicorn이 이 모듈을 다시 import해서 app을 찾습니다.
    uvicorn.run("4_memo_chat:app", host="127.0.0.1", port=8200, reload=False)
