# 1. 기획 자료 — 회의 분석 리포트 프로젝트

## 1.1 배경

류지 프로젝트는 목소리 등록, 실시간 화자 분리, 회의록 생성을 한 화면에서 하는 AI 회의 어시스턴트입니다. 처음 참고한 회의 도구는 저장소가 나뉘어 있었습니다.

- Frontend: React 협업 UI
- Backend: FastAPI + LangGraph 멀티 에이전트
- ASR: Qwen3 ASR + ForcedAligner + pyannote

핵심 기술은 **Qwen3+pyannote, LangGraph** 이고, UI는 Gradio, DB는 SQLite입니다.

## 1.2 목표

1. Gradio로 회의 생성~분석 UX를 실습한다.
2. **Qwen3 + pyannote** 오프라인 ASR/화자분리 파이프라인을 이해한다.
3. **LangGraph** 노드 그래프(`summarize → decisions → actions → quality`)를 이해한다.
4. **MCP**로 LLM 클라이언트가 회의 Tool을 호출하게 한다.
5. 저장된 회의록을 근거로 답하는 **챗봇**을 사용한다.

## 1.3 학습 성과

| ID | 내용 |
|----|------|
| L1 | Gradio ↔ FastAPI 연동 |
| L2 | pyannote 세그먼트 + Qwen3 전사 흐름 설명 |
| L3 | LangGraph State/Node/Edge 구성 |
| L4 | MCP Tool로 analyze/list/ask 호출 |
| L5 | 원본 3-repo와 수업판 매핑 |
| L6 | 회의 챗봇이 근거 회의를 찾아 답하는 흐름 설명 |

## 1.4 범위

### In
- 회의 CRUD, 회의록 저장
- Qwen3 + pyannote ASR (`/transcribe`)
- LangGraph 분석 API
- 회의 챗봇 (`retrieve → answer`, 세션 기록)
- Gradio UI, MCP server, SQLite

### Out (원본 대비 축소)
- React SPA, OAuth, Slack/Jira
- Redis 실시간 캐시 / MongoDB 최종 저장
- Supervisor 전체 도메인 라우팅, WBS/보고서 생성

## 1.5 시나리오

**A. 텍스트 → LangGraph**  
회의 생성 → 샘플 회의록 저장 → LangGraph 분석 → MCP `get_meeting`

**B. 오디오 → Qwen3+pyannote**  
오디오 업로드 → diarization+ASR → 회의록 저장 → 분석

**C. 챗봇**  
저장된 회의를 대상으로 「API 연동은 언제까지야?」라고 질문 → 관련 발화와 회의 제목을 답변으로 확인 → MCP `ask_meeting`으로 같은 질문

## 1.6 인프라 요구

| 항목 | 권장 |
|------|------|
| ASR 실사용 | Linux + NVIDIA GPU, CUDA, vLLM, HF Token |
| 분석 LLM | OpenAI API Key (없으면 규칙 기반 노드 폴백) |
| 최소 실습 | Backend + Gradio + 수동 회의록 + LangGraph(규칙) + 챗봇 |
| 챗봇 LLM | OpenAI API Key (없으면 키워드 검색 답변) |

## 1.7 90분 운영안

| 시간 | 내용 |
|------|------|
| 0–15 | 원본 vs 수업판 아키텍처 |
| 15–35 | 설치, Backend/ASR/Gradio 기동 |
| 35–50 | LangGraph 분석 실습 |
| 50–65 | 회의 챗봇 질문 실습 |
| 65–80 | ASR 파이프라인 / (가능 시) 오디오 전사 |
| 80–90 | MCP `ask_meeting` + 과제 안내 |

## 1.8 기술 미니 예제

본편 전에 하나씩 연습하려면 [`examples/`](../examples/) 폴더를 사용합니다.

- Gradio, FastAPI, LangGraph, Qwen3 ASR, pyannote, MCP, SQLite
- 각 폴더에 설치 모듈·설정·실행법·연습 과제가 있습니다.
