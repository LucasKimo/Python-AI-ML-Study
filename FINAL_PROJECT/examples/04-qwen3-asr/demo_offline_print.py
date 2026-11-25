"""모델 없이 ASR 흐름만 설명하는 데모."""
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.xpu.is_available():
    DEVICE = torch.device('xpu')
else:
    DEVICE = torch.device('cpu')

print(DEVICE)

print(
    """
=== Qwen3 ASR 미니 개념 데모 (모델 미로드) ===

이 예제는 vLLM이 아니라 PyTorch CUDA로 전사합니다. (Windows + RTX 3060)

1) 오디오 파일을 16kHz mono float32 로 읽는다
2) Qwen3ASRModel.from_pretrained(..., device_map='cuda:0') 로 GPU에 올린다
3) asr.transcribe(audio=(wav, 16000), language='Korean')
4) results[0].text 에 문장이 나온다

실제 전사:
  python transcribe_file.py sample.wav
"""
)
