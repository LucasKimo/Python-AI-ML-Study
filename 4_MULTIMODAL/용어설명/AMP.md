# AMP (Automatic Mixed Precision, 자동 혼합 정밀도)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유

AMP는 연산에 따라 FP16 또는 BF16 같은 낮은 정밀도와 FP32(32비트 부동소수점)를 자동으로 섞어 사용하는 기법입니다. 지원하는 GPU에서는 큰 행렬 연산을 빠르게 수행하고 중간 결과의 메모리를 절약할 수 있습니다. 속도 향상과 절약량은 하드웨어와 모델에 따라 달라지며, 전체 메모리가 무조건 절반이 되는 것은 아닙니다.

모든 계산을 낮은 정밀도로 바꾸면 수치 오차나 값 표현 범위 때문에 학습이 불안정해질 수 있으므로, 연산별로 적절한 정밀도를 사용합니다.

## 두 도구의 역할

- `torch.amp.autocast`: 행렬 곱 등은 낮은 정밀도로, 수치적으로 민감한 일부 연산은 FP32로 실행하도록 연산별 자료형을 선택합니다. 모든 연산이나 모델 파라미터를 일괄적으로 FP16으로 바꾸는 것은 아닙니다.
- `torch.amp.GradScaler`: 주로 FP16 학습에서 작은 기울기가 표현 범위를 벗어나 0이 되는 underflow를 완화합니다. 손실에 배율을 곱해 역전파하고, 업데이트 전에 기울기를 원래 규모로 되돌립니다. inf/NaN이 검출되면 해당 업데이트를 건너뛰고 배율을 조정합니다. 모든 수치 문제를 해결하는 것은 아닙니다.

예를 들어 아주 작은 기울기가 FP16에서 0으로 표현되면 학습 신호가 사라질 수 있습니다. GradScaler는 역전파 전에 손실을 키워 이 위험을 줄입니다. 학습률을 크게 만드는 것과는 다릅니다.

## ViT 네 자리 분류 예시

[ViT 실습](../10_ViT.ipynb)의 모델, optimizer, criterion을 준비한 상태에서 사용하는 CUDA FP16 예시입니다. scaler는 반복문 밖에서 한 번 만듭니다.

```python
scaler = torch.amp.GradScaler("cuda")
model.train()

for images, targets in train_loader:
    images = images.to("cuda")
    targets = targets.to("cuda")  # 정답은 정수형 그대로 유지
    optimizer.zero_grad(set_to_none=True)

    # 순전파와 손실 계산에 자동 정밀도 선택 적용
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(images)
        loss = sum(
            criterion(outputs[i], targets[:, i])
            for i in range(4)
        ) / 4

    # 역전파는 autocast 블록 밖에서 실행
    scaler.scale(loss).backward()  # 확대된 손실로 기울기 계산
    scaler.step(optimizer)        # 배율 복원·검사 후 가중치 갱신
    scaler.update()               # 다음 단계에 사용할 배율 조절
```

이 흐름에서는 `optimizer.step()`을 별도로 호출하지 않습니다. 입력 이미지나 모델에 직접 `.half()`를 호출할 필요도 없습니다.

## 혼동하기 쉬운 개념

| 개념 | 차이 |
| --- | --- |
| FP32 학습 | 기본적으로 32비트 부동소수점을 사용합니다. AMP는 연산별로 정밀도를 혼합합니다. |
| BF16 | FP16과 같은 16비트지만 표현 가능한 값의 범위가 더 넓습니다. BF16 학습에서는 일반적으로 GradScaler가 필요하지 않습니다. |
| `torch.no_grad()` | 기울기 기록을 끄는 기능입니다. AMP는 정밀도를 조절하며 학습과 추론 모두에서 쓸 수 있습니다. |
| `model.eval()` | Dropout 등의 동작을 평가용으로 바꿉니다. AMP와 별개입니다. |
| 추론 | autocast는 사용할 수 있지만 역전파가 없으므로 GradScaler는 필요하지 않습니다. |

AMP는 GPU에서 자동으로 켜지는 것이 아닙니다. 코드로 활성화해야 하며 CPU 등에서도 지원되는 장치와 자료형 조합에 따라 사용할 수 있습니다.

## 참고

- [PyTorch AMP 공식 문서](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch AMP 사용 예시](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
