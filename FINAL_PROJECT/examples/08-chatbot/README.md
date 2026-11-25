# 08. 챗봇 — 회의 메모에서 답 찾기

## 이 예제로 배우는 것

- 챗봇이 아무 말이나 만드는 것이 아니라 **가지고 있는 자료에서 답을 찾는** 흐름
- 본편 `backend/app/graph/chat.py`의 `retrieve → answer`를 아주 작게 줄인 버전

## 본편과의 차이

| | 이 예제 | 본편 |
|--|---------|------|
| 자료 | 코드 안의 문장 3개 | SQLite 회의록 |
| 그래프 | 함수 하나 | LangGraph retrieve → answer |
| LLM | 없음 | OpenAI 키가 있으면 사용 |
| 화면 | 터미널 | Gradio 4번 탭, MCP `ask_meeting` |

## 설치

추가 패키지가 필요 없습니다. 파이썬만 있으면 됩니다.

## 실행

```powershell
cd examples\08-chatbot
python mini_chat.py
```

예상 출력에는 `수요일`이 들어간 API 일정 문장과 `금요일` UI 문장이 나옵니다.

## 본편에서 써 보기

1. Backend와 Gradio를 실행합니다.
2. 회의를 만들고 회의록을 저장합니다.
3. Gradio **4. 회의 챗봇**에서 질문합니다.
4. API로도 확인할 수 있습니다.

```powershell
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"question\":\"API 연동 일정\",\"use_llm\":false}"
```

## 연습

1. `NOTES`에 문장을 하나 더 넣고 그 내용으로 질문해 보세요.
2. 토큰 길이가 1글자인 단어도 검색되게 바꿔 보세요.
3. 본편에서 `use_llm`을 켜고 같은 질문의 답 차이를 비교해 보세요.
