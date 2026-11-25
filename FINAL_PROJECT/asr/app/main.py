"""ASR FastAPI. 오디오 파일을 화자 구간 + 문장으로 바꿉니다.

이 프로세스는 Backend(:8000)와 따로 뜹니다. 기본 포트는 8001입니다.
모델을 기동 시점에 올리지 않는 이유와 pyannote/Qwen3의 역할 분담은
core/models.py, services/pipeline.py 맨 위 주석에 있습니다.

POST /transcribe
  파일 → 16kHz 모노 → pyannote 화자분리 → 구간마다 Qwen3 전사.
text_hint가 있으면 모델을 부르지 않고, 붙여 넣은 회의록 텍스트를 발화로 파싱합니다.
GPU나 HF 토큰이 없을 때의 실습 경로입니다.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr.app.core import models as model_hub  # noqa: E402
from asr.app.services import pipeline as asr_pipeline  # noqa: E402
from shared import config  # noqa: E402
from shared.schemas import HealthResponse, TranscribeResponse, Utterance  # noqa: E402

KST = timezone(timedelta(hours=9))

app = FastAPI(
    title="류지 프로젝트 ASR",
    description="Qwen3 ASR + pyannote speaker diarization",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    # 지연 로드: 첫 /transcribe 때 모델 로드 (기동 속도/메모리 절약)
    # 즉시 로드하려면: ASR_WARMUP=true
    import os

    if os.getenv("ASR_WARMUP", "false").lower() in {"1", "true", "yes"}:
        try:
            model_hub.warmup()
        except Exception as exc:  # noqa: BLE001
            print(f"[ASR] warmup failed (will retry on request): {exc}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status = model_hub.model_status()
    return HealthResponse(
        status="ok",
        service="asr",
        time=datetime.now(KST),
        details=status,
    )


@app.post("/warmup")
def warmup():
    try:
        return model_hub.warmup()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    file: Optional[UploadFile] = File(None),
    text_hint: str = Form(""),
):
    """
    오디오 파일 → pyannote 화자분리 → Qwen3 세그먼트 ASR.

    text_hint 가 있으면 모델 없이 수동 파싱(회의록 붙여넣기용 보조 경로).
    """
    if text_hint.strip():
        utterances = []
        for i, line in enumerate(text_hint.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                speaker, text = line.split(":", 1)
            else:
                speaker, text = f"SPEAKER_{(i - 1) % 3:02d}", line
            utterances.append(
                Utterance(seq=i, speaker=speaker.strip(), text=text.strip())
            )
        return TranscribeResponse(
            utterances=utterances,
            mode="text_hint",
            message="텍스트 힌트를 발화로 변환했습니다 (모델 미사용).",
            segment_count=len(utterances),
        )

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="audio file 이 필요합니다.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    dest = config.UPLOAD_DIR / file.filename
    dest.write_bytes(data)

    try:
        result = asr_pipeline.transcribe_file(str(dest))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return TranscribeResponse(
        utterances=[Utterance(**u) for u in result["utterances"]],
        mode=result["mode"],
        message=result["message"],
        segment_count=result["segment_count"],
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        "asr.app.main:app",
        host=config.ASR_HOST,
        port=config.ASR_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
