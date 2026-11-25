"""작은 FastAPI 서버 — 메모 저장/목록."""

from __future__ import annotations

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

# 실제 서비스에서는 DB를 씁니다. 여기는 이해를 위해 리스트만 사용합니다.
_MEMOS: list[dict] = []
_NEXT_ID = 1


class MemoCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100)
    content: str = ""


class Memo(BaseModel):
    id: int
    title: str
    content: str
    created_at: str


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


@app.get("/memos", response_model=list[Memo])
def list_memos():
    return list(reversed(_MEMOS))


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


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8100, reload=False)
