# Micro F1 (전체 판정의 TP·FP·FN을 합산한 F1)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
모든 문장과 라벨의 TP, FP, FN을 각각 합친 뒤 [F1](F1_Score.md)을 한 번 계산합니다. 전체 양성 예측과 누락을 종합해 평가할 때 사용합니다.

## 예시
합산 TP=80, FP=10, FN=30이면 Micro F1 = 160 / (160 + 10 + 30) = 0.8입니다.

## 코드
```python
f1_micro_average = f1_score(
    y_true, y_pred, average='micro', zero_division=0
)
```

## 혼동 주의
라벨별 F1의 평균이 아닙니다. 흔한 라벨의 TP/FP/FN이 더 많이 반영될 수 있어 드문 라벨의 부진이 덜 드러날 수 있습니다. 다중 라벨에서 Micro F1은 완전 일치 Accuracy와 같지 않습니다.

출처: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

