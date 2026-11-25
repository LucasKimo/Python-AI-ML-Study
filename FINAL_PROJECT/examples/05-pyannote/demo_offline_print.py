"""모델 없이 화자분리 개념만 출력."""

print(
    """
=== pyannote 화자분리 미니 개념 데모 ===

목표: 오디오에서 '누가 / 언제' 말했는지 구간을 얻는다.

예시 결과:
  SPEAKER_00   0.00 ~ 2.40
  SPEAKER_01   2.40 ~ 5.10
  SPEAKER_00   5.10 ~ 7.80

다음 단계(본편):
  각 구간 오디오를 잘라 Qwen3 ASR로 글자로 바꾼다.

이름을 붙이려면 참가자와 목소리를 먼저 등록합니다.
  python identify_speaker.py
등록된 목소리와 방금 말한 목소리를 비교해 누구인지 출력합니다.
이 흐름은 본편 backend/app/voice_engine.py 와 같습니다.

필요 준비:
  1) pip install pyannote.audio torch torchaudio soundfile
  2) Hugging Face 토큰 + 모델 라이선스 동의
  3) python diarize_file.py your.wav
"""
)
