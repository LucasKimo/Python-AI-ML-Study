# 07. SQLite — 파일 하나로 하는 작은 데이터베이스

## 이 예제로 배우는 것

- SQLite가 뭔지
- 테이블 만들고 / 넣고 / 조회하기
- 본편 `shared/db.py`의 축소판

## SQLite란?

**서버 설치가 필요 없는** 파일형 DB입니다.  
`ryuzy.db` 같은 파일 하나에 표(테이블)를 저장합니다.

이전 회의 도구는 MySQL + Redis + MongoDB를 쓰지만,  
수업판은 설치 부담을 줄이려고 SQLite 하나로 회의 데이터를 저장합니다.

---

## 설치해야 하는 모듈

**추가 pip 설치 없음!**  
파이썬 표준 라이브러리 `sqlite3`만 사용합니다.

확인:

```powershell
python -c "import sqlite3; print(sqlite3.sqlite_version)"
```

---

## 파일 설명

| 파일 | 내용 |
|------|------|
| `memo_db.py` | 메모 테이블 CRUD 예제 |
| `inspect_db.py` | 만들어진 DB 파일을 읽어 보기 |

실행하면 이 폴더에 `demo.db` 파일이 생성됩니다.

---

## 실행 방법

```powershell
cd examples\07-sqlite
python memo_db.py
python inspect_db.py
```

---

## 코드 핵심 아이디어

```python
import sqlite3

conn = sqlite3.connect("demo.db")
conn.execute("CREATE TABLE IF NOT EXISTS memos (id INTEGER PRIMARY KEY, title TEXT)")
conn.execute("INSERT INTO memos(title) VALUES (?)", ("주간회의",))
conn.commit()
rows = conn.execute("SELECT id, title FROM memos").fetchall()
```

1. `connect` — DB 파일 열기(없으면 생성)  
2. `execute` — SQL 실행 (`?` 자리표시자로 안전하게 값 전달)  
3. `commit` — 저장 확정  
4. `fetchall` — 조회 결과  

본편은 JSON 문자열로 회의록·액션을 컬럼에 넣습니다 (`shared/db.py`).

---

## 연습 과제

1. `memos`에 `created_at` 컬럼을 추가해 보세요.
2. 제목 키워드로 검색(`LIKE %키워드%`)을 구현해 보세요.
3. 본편 `data/ryuzy.db`를 `inspect_db.py` 방식으로 열어 보세요. (회의 테이블)

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| DB locked | 다른 프로그램이 파일을 열고 있는지 확인 |
| 한글 깨짐 | 콘솔 코드페이지 문제일 수 있음. 파일/내용 자체는 UTF-8로 저장 |
| 테이블이 안 보임 | 경로가 다른 `demo.db`를 보고 있는지 확인 |
