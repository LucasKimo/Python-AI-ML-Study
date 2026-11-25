# 03. LangGraph — 노드를 이어서 실행하는 파이프라인

## 이 예제로 배우는 것

- LangGraph가 뭔지 (그냥 함수 호출과 뭐가 다른지)
- State(상태) → Node(단계) → Edge(연결) 개념
- 본편 `backend/app/graph/` 의 축소판

## LangGraph란?

LLM/분석 작업을 **그래프(순서도)** 로 묶는 라이브러리입니다.  
한 번에 긴 프롬프트를 던지기보다,

```
요약 노드 → 결정사항 노드 → 액션 노드 → 끝
```

처럼 **단계를 나눠** 상태를 업데이트합니다.

본편 류지 프로젝트 분석 흐름:

```
summarize → decisions → actions → quality → END
```

이 예제는 더 작게:

```
START → count_lines → make_summary → END
```

챗봇 예제는 본편 챗봇과 같은 두 단계입니다.

```
START → retrieve → answer → END
```

---

## 설치해야 하는 모듈

```powershell
pip install langgraph
```

| 모듈 | 용도 |
|------|------|
| `langgraph` | StateGraph, 노드/엣지 실행 |
| (선택) `langchain-openai`, `openai` | 실제 LLM 노드를 붙일 때 |

확인:

```powershell
python -c "import langgraph; print('langgraph OK')"
```

> 이 예제는 **LLM 키 없이** 동작합니다. (규칙 기반 노드만 사용)

---

## 파일 설명

| 파일 | 내용 |
|------|------|
| `mini_graph.py` | 회의록 텍스트 → 줄 수 세기 → 한 줄 요약 |
| `meeting_graph.py` | 본편과 비슷한 3노드(요약/결정/액션) 미니 버전 |
| `chat_graph.py` | 질문하면 회의 메모에서 답을 찾는 챗봇 (`retrieve → answer`) |

---

## 실행 방법

```powershell
cd examples\03-langgraph
python mini_graph.py
python meeting_graph.py
python chat_graph.py --demo
python chat_graph.py
```

`chat_graph.py`는 터미널에서 질문을 받습니다. 예시 질문:

- `API 연동은 언제까지야?`
- `UI는 언제 제출해?`
- `테스트 케이스는 누가 해?`

끝내려면 `quit`을 입력합니다. `--demo`는 샘플 질문 두 개만 출력하고 종료합니다.

예상 출력 예:

```
최종 상태: { 'line_count': 3, 'summary': '...', ... }
```

---

## 코드 핵심 아이디어

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str
    summary: str

def summarize(state: State) -> dict:
    return {"summary": state["text"][:50]}

g = StateGraph(State)
g.add_node("summarize", summarize)
g.add_edge(START, "summarize")
g.add_edge("summarize", END)
app = g.compile()
print(app.invoke({"text": "안녕하세요", "summary": ""}))
```

1. **State**: 파이프라인 전체에서 들고 다닐 데이터  
2. **Node**: State를 받아 일부 키만 업데이트해 반환  
3. **Edge**: 어떤 노드 다음에 어떤 노드인지  
4. **compile + invoke**: 실행  

---

## 연습 과제

1. `meeting_graph.py`에 `quality` 노드를 추가해 점수를 매겨 보세요.
2. `use_llm=True`일 때 OpenAI를 호출하도록 확장해 보세요. (본편 `nodes.py` 참고)
3. 에러가 나면 `errors` 리스트에 메시지를 쌓아 보세요.
4. `chat_graph.py`의 `NOTES`에 문장을 추가하고, 그 내용으로 질문해 보세요.

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| `No module named langgraph` | `pip install langgraph` |
| TypedDict 오류 | Python 3.10+ 권장 |
| invoke 결과가 비어 있음 | 노드가 `return {"키": 값}` 형태인지 확인 |
