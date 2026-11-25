"""
Backend
Server
"""

from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="FastAPI 미니 예제",
    description="회의 메모를 메모리에만 잠깐 저장하는 아주 작은 API",
    version="1.0.0",
)

_MEMOS: list[dict] = []  # 메모 저장 리스트 변수
_NEXT_ID = 1             # 메모 ID (1로 초기화)

# BaseModel: 데이터 구조와 타입을 정의하고 입력 데이터의 유효성을 자동으로 검증하는 기본 클래스
# 사용자가 보내는 데이터
class MemoCreate(BaseModel):  
    title: str = Field(..., min_length=1, max_length=100)
    content: str = ""

# 실제 저장될 때 포멧
class Memo(BaseModel):
    id: int
    title: str
    content: str
    created_at: str

# REST API
# 127.0.0.1:8100/health
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.post("/memos", response_model=Memo)
def create_memo(payload: MemoCreate):
    global _NEXT_ID
    memo = {
        "id": _NEXT_ID,
        "title": payload.title,
        "content": payload.content,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    _NEXT_ID += 1
    _MEMOS.append(memo)
    return memo

# 127.0.0.1:8100/memos
@app.get("/memos", response_model=list[Memo])
def list_memos():
    return list(reversed(_MEMOS))

# 127.0.0.1:8100/memos/1
@app.get("/memos/{memo_id}", response_model=Memo)
def get_memo(memo_id: int):
    for memo in _MEMOS:
        if memo["id"] == memo_id:
            return memo
    raise HTTPException(status_code=404, detail="메모를 찾을 수 없습니다.")


@app.delete("/memos/{memo_id}")
def delete_memo(memo_id: int):
    for i, memo in enumerate(_MEMOS):
        if memo["id"] == memo_id:
            _MEMOS.pop(i)
            return {"deleted": True, "id": memo_id}
    raise HTTPException(status_code=404, detail="메모를 찾을 수 없습니다.")

# uvicorn: Fast API 동작 시킬 수 있는 서버 이름
if __name__ == "__main__":
    uvicorn.run("3_main:app", host="127.0.0.1", port=8100, reload=False)