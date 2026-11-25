# 5. 이전 회의 도구에서 가져온 구성

## 기술 정렬

류지 프로젝트는 아래처럼 구성을 단순하게 맞췄습니다.

| 영역 | 이전 도구 | 류지 프로젝트 |
|------|------|-------------|
| UI | React | Gradio |
| 분석 | LangGraph (다도메인 Supervisor) | LangGraph (분석 선형 그래프) |
| 챗봇 | Knowledge RAG 챗봇 | LangGraph `retrieve → answer` (회의록 근거) |
| ASR | Qwen3 + Align + pyannote + WS | Qwen3 + pyannote (오프라인 REST) |
| DB | MySQL+Redis+Mongo | SQLite |
| MCP | 없음 | FastMCP |

## 의도적으로 단순화한 부분

1. WebSocket 실시간 스트림 → 파일 업로드 REST
2. ForcedAligner는 로드 가능하나 기본 오프라인 경로에서는 세그먼트 ASR 위주
3. Supervisor 라우팅 대신 고정 분석 파이프라인
4. OAuth/외부 SaaS 연동 제외

## 확장 과제

1. WebSocket 스트리밍 ASR 복원
2. ForcedAligner 단어 타임스탬프 정렬 노드 추가
3. 챗봇 retrieve를 Chroma RAG로 확장
4. SQLite → PostgreSQL
