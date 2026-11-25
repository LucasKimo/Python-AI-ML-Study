# Macro F1 (라벨별 F1의 단순 평균)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
각 라벨의 [F1](F1_Score.md)을 따로 구한 뒤 동일한 비중으로 평균합니다. 드문 라벨도 흔한 라벨만큼 중요하게 평가할 때 유용합니다.

## 예시
세 라벨의 F1이 0.9, 0.6, 0.3이면 Macro F1 = (0.9 + 0.6 + 0.3) / 3 = 0.6입니다. 각 라벨의 양성 개수와 관계없이 같은 비중입니다.

## 코드
```python
f1_macro_average = f1_score(
    y_true, y_pred, average='macro', zero_division=0
)
```

## 혼동 주의
Precision 평균과 Recall 평균을 구한 뒤 F1을 계산하는 방식이 아닙니다. 라벨별 F1을 먼저 구합니다. Micro는 TP/FP/FN을 먼저 합산하고, Weighted는 실제 양성 개수로 가중합니다.

출처: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

