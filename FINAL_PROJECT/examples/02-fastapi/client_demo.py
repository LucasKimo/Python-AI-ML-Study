"""FastAPI 서버를 호출하는 간단 클라이언트."""

import httpx

BASE = "http://127.0.0.1:8100"


def main() -> None:
    with httpx.Client(timeout=10.0) as client:
        health = client.get(f"{BASE}/health")
        print("health:", health.json())

        created = client.post(
            f"{BASE}/memos",
            json={"title": "스프린트 메모", "content": "API 연동 일정 공유"},
        )
        print("created:", created.json())

        listed = client.get(f"{BASE}/memos")
        print("list:", listed.json())


if __name__ == "__main__":
    main()
