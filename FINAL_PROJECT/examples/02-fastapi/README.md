# 02. FastAPI — 작은 REST API 서버

## 이 예제로 배우는 것

- FastAPI가 뭔지
- `GET` / `POST` API를 만드는 방법
- Swagger 문서(`/docs`)로 API를 눌러 보는 방법
- 본편의 `backend/app/main.py`, `asr/app/main.py`의 기본 뼈대

## FastAPI란?

파이썬으로 **HTTP API 서버**를 빠르게 만드는 프레임워크입니다.  
프론트(Gradio)나 MCP가 `http://...`로 요청을 보내면, FastAPI가 JSON으로 답합니다.

---

## 설치해야 하는 모듈

```powershell
pip install fastapi "uvicorn[standard]"
```

| 모듈 | 용도 |
|------|------|
| `fastapi` | API 라우트, 요청/응답 모델 |
| `uvicorn` | ASGI 서버 (FastAPI를 실제로 실행) |

확인:

```powershell
python -c "import fastapi, uvicorn; print(fastapi.__version__, uvicorn.__version__)"
```

---

## 파일 설명

| 파일 | 내용 |
|------|------|
| `main.py` | 헬스체크 + 회의 메모 저장/조회 API |
| `client_demo.py` | API를 호출하는 간단 클라이언트 |

---

## 실행 방법

터미널 1 — 서버:

```powershell
cd examples\02-fastapi
python main.py
```

브라우저에서 열어보세요.

- 헬스: http://127.0.0.1:8100/health  
- **Swagger UI**: http://127.0.0.1:8100/docs  ← 여기가 중요!

터미널 2 — 클라이언트:

```powershell
python client_demo.py
```

---

## 코드 핵심 아이디어

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}
```

- `@app.get` / `@app.post` → URL과 함수를 연결
- 반환값이 자동으로 JSON이 됨
- `/docs`에서 버튼을 눌러 테스트 가능

본편 Backend는 여기에 회의 CRUD + LangGraph 분석을 붙인 것입니다.

---

## 연습 과제

1. `GET /memos/{memo_id}` 상세 조회를 추가해 보세요.
2. 메모에 `author` 필드를 추가해 보세요.
3. Gradio(01번)에서 이 API를 `httpx`로 호출해 보세요.

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| 포트 충돌 | `main.py`의 `port=8100`을 다른 번호로 |
| 연결 거부 | 서버를 먼저 실행했는지 확인 |
| CORS 오류 | 브라우저 다른 origin에서 호출 시 CORS 미들웨어 추가 (본편 참고) |
