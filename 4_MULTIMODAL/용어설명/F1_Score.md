# F1 Score (F1 점수)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
양성이라고 예측한 결과의 정확함(Precision)과 실제 양성을 찾아낸 정도(Recall)를 조화평균한 지표입니다. 양성 라벨을 잘못 붙이는 문제와 놓치는 문제를 함께 평가합니다.

- TP: 실제 1을 1로 예측
- FP: 실제 0을 1로 예측
- FN: 실제 1을 0으로 예측
- Precision = TP / (TP + FP): 있다고 예측한 라벨 중 실제 있는 비율
- Recall = TP / (TP + FN): 실제 있는 라벨 중 찾아낸 비율
- F1 = 2 × Precision × Recall / (Precision + Recall)
- 같은 식: F1 = 2TP / (2TP + FP + FN)

## 예시
실제 욕설 문장 10개 중 6개를 찾고, 욕설이라고 예측한 문장은 총 8개라면 TP=6, FP=2, FN=4입니다. Precision=0.75, Recall=0.6, F1은 약 0.667입니다.

## 코드와 평균 방식
```python
f1_score(y_true, y_pred, average='macro', zero_division=0)
# y_pred는 임계값을 적용한 0/1 예측
# zero_division=0: 계산 분모가 0인 경우 해당 점수를 0으로 처리
```

다중 라벨에서는 [Macro F1](F1_Macro.md), [Micro F1](F1_Micro.md), [Weighted F1](F1_Weighted.md) 등으로 여러 라벨의 성능을 집계합니다. 여기서 1은 좋은 의미의 감정이 아니라 해당 라벨이 있다는 뜻입니다.

## 혼동 주의
F1은 실제 0을 0으로 맞힌 TN을 직접 사용하지 않습니다. 완전 일치 Accuracy와 다릅니다. 임계값이 바뀌면 예측과 F1도 달라질 수 있습니다.

출처: [scikit-learn f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

