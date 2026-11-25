# Weighted F1 (라벨별 실제 양성 수로 가중한 F1)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
각 라벨의 [F1](F1_Score.md)을 구하고 실제 양성 수(support)로 가중평균합니다. 데이터에서 자주 등장하는 라벨에 더 큰 비중을 줍니다.

계산: Σ(support_k × F1_k) / Σsupport_k

## 예시
라벨 A의 F1=0.9, 실제 양성 수=90이고 B의 F1=0.3, 실제 양성 수=10이면 Weighted F1 = (90×0.9 + 10×0.3) / 100 = 0.84입니다. 같은 두 라벨의 Macro F1은 0.6입니다.

## 코드
```python
f1_weighted_average = f1_score(
    y_true, y_pred, average='weighted', zero_division=0
)
```

## 혼동 주의
가중치는 예측한 양성 수가 아니라 정답의 양성 수입니다. 다중 라벨에서는 한 문장이 여러 라벨에 집계되므로 support 합이 문장 수보다 클 수 있습니다. Micro와 계산법이 다르며 값도 일반적으로 다릅니다.

출처: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

