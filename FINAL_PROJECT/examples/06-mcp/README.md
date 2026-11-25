# 06. 슬랙 — 원하는 문장을 채널로 보내기

Incoming Webhook 주소로 문장을 보내면, 고른 채널에 그 글이 올라갑니다.  
주소는 프로젝트 루트 `.env` 의 `SLACK_WEBHOOK_URL` 에 넣습니다. 채팅이나 깃에는 붙이지 않습니다.

## 1. 슬랙 앱 만들기

1. https://api.slack.com/apps 를 열고, 메시지를 받을 워크스페이스로 로그인합니다.
2. **Create New App** → **From scratch** 를 누릅니다.
3. **App Name** 에 예: `류지 회의` 를 적고, 워크스페이스를 고른 뒤 **Create App** 을 누릅니다.

## 2. Webhook 주소 만들기

1. 왼쪽 메뉴 **Incoming Webhooks** 를 엽니다.
2. **Activate Incoming Webhooks** 를 **On** 으로 둡니다.
3. **Add New Webhook to Workspace** 를 누르고, 글을 올릴 채널을 고른 뒤 **허용** 합니다.
4. 표에 생긴 Webhook URL을 **Copy** 합니다. `https://` 부터 끝까지입니다.

   `https://hooks.slack.com/services/T.../B.../비밀문자열`

## 3. `.env` 에 넣기

프로젝트 루트 `.env` 에 아래 한 줄을 넣습니다. 파일이 없으면 `.env.example` 을 복사해 `.env` 로 저장합니다.

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/복사한_나머지
```

## 4. 실행

```powershell
cd C:\Ryuzy\ryuzy-project
.\.venv\Scripts\Activate.ps1
pip install httpx

cd examples\06-mcp
python send_slack.py 오늘 회의 끝났습니다
```

`httpx` 가 이미 있으면 `pip install` 은 생략합니다.  
인자 없이 `python send_slack.py` 를 실행하면 문장을 입력받습니다.

성공하면 터미널에 `슬랙으로 보냈습니다.` 가 나오고, 위에서 고른 채널에 문장이 보입니다.

## 문제 해결

| 증상 | 조치 |
|------|------|
| `SLACK_WEBHOOK_URL 이 없습니다` | 프로젝트 루트 `.env` 인지, 값이 `https://` 로 시작하는지 확인 |
| `404` | 웹훅이 삭제됨. **Add New Webhook to Workspace** 부터 다시 하고 주소를 바꿈 |
| `403` | 주소가 잘렸거나 앱이 제거됨. URL을 다시 복사 |
| 터미널은 성공인데 채널에 없음 | 웹훅을 만들 때 고른 채널이 맞는지 앱 설정 표에서 확인 |
| `No module named httpx` | 가상환경을 켠 뒤 `pip install httpx` |
