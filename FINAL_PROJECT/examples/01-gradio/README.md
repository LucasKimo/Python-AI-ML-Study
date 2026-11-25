# 01. Gradio — 5분 만에 웹 UI 만들기

## 이 예제로 배우는 것

- Gradio가 뭔지
- 입력창 + 버튼 + 결과 표시를 코드 몇 줄로 만드는 방법
- 류지 프로젝트의 `frontend/app.py`가 왜 Gradio인지

## Gradio란?

파이썬 함수를 **웹 화면**으로 바꿔 주는 라이브러리입니다.  
HTML/CSS/JavaScript를 몰라도, 함수만 만들면 브라우저에서 눌러볼 수 있습니다.

본 프로젝트에서는 React 대신 Gradio로 회의 생성·ASR 호출·분석 버튼을 만들었습니다.

---

## 설치해야 하는 모듈

```powershell
# 가상환경 활성화 후
pip install gradio
```

| 모듈 | 용도 | 최소 버전 예시 |
|------|------|----------------|
| `gradio` | 웹 UI | 5.0 이상 |

확인:

```powershell
python -c "import gradio; print(gradio.__version__)"
```

---

## 파일 설명

| 파일 | 내용 |
|------|------|
| `app.py` | 이름 입력 → 인사말 출력 (가장 작은 Gradio 앱) |
| `app_tabs.py` | 탭 2개 예제 (본편 UI와 비슷한 구조) |

---

## 실행 방법

```powershell
cd examples\01-gradio
python app.py
```

브라우저가 열리거나, 터미널에 나온 주소(보통 `http://127.0.0.1:7860`)로 접속합니다.

탭 예제:

```powershell
python app_tabs.py
```

종료: 터미널에서 `Ctrl + C`

---

## 코드 핵심 아이디어

```python
def greet(name: str) -> str:
    return f"안녕하세요, {name}님!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()
```

1. `greet` — 평범한 파이썬 함수  
2. `gr.Interface` — 함수를 UI에 연결  
3. `launch()` — 웹 서버 시작  

본편의 `frontend/app.py`는 `gr.Blocks` + `Tab`으로 더 복잡한 화면을 만든 버전입니다.

---

## 연습 과제

1. `greet`에 나이를 입력받아 `"OO님은 N살입니다"`를 출력해 보세요.
2. `app_tabs.py`에 세 번째 탭을 추가해 보세요.
3. 버튼 라벨을 한글로 바꿔 보세요.

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| 포트 사용 중 | `demo.launch(server_port=7861)` |
| 브라우저가 안 열림 | 터미널 URL을 직접 복사해 접속 |
| `No module named gradio` | `pip install gradio` 후 가상환경 확인 |
