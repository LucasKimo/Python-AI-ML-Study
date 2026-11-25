# ROC-AUC (ROC 곡선 아래 면적)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
여러 임계값에서 얻는 ROC 곡선 아래 면적입니다. 실제 양성에 실제 음성보다 더 높은 점수를 주는 순위 구분 능력을 평가합니다.

- 세로축 TPR = TP / (TP + FN): 실제 양성을 찾아낸 비율
- 가로축 FPR = FP / (FP + TN): 실제 음성을 양성으로 잘못 판단한 비율
- AUC 1.0: 완벽한 순위 구분, 0.5: 무작위 수준의 순위 구분, 0.5 미만: 반대 방향 경향
- 확률이 얼마나 정확하게 보정되어 있는지를 나타내는 지표는 아닙니다.

## 예시
정답이 [1, 0], 예측 점수가 [0.9, 0.8]이면 둘 다 0.5 이상이지만 양성의 점수가 더 높아 AUC는 1입니다. 0/1로 먼저 자르면 [1, 1]이 되어 이 순위 정보가 사라집니다.

## 코드: 제공된 함수에서 수정할 부분
```python
probs = torch.sigmoid(torch.as_tensor(predictions, dtype=torch.float32))
y_pred = (probs >= threshold).to(torch.int64).cpu().numpy()

# Accuracy와 F1에는 0/1인 y_pred를 전달
# ROC-AUC에는 임계값 적용 전의 연속 점수를 전달
roc_auc = roc_auc_score(
    labels, probs.detach().cpu().numpy(), average='micro'
)
```

## Micro 평균과 혼동 주의
다중 라벨의 micro AUC는 (문장 수, 라벨 수) 배열의 모든 칸을 펼쳐 하나의 이진 분류처럼 계산합니다. 모든 라벨에서 나온 점수를 함께 비교하므로 라벨별 AUC의 평균과 다릅니다.

원본의 `roc_auc_score(y_true, y_pred, average='micro')`도 숫자가 계산될 수 있지만 임계값을 적용하며 점수 정보를 잃으므로 의도한 순위 성능 평가에는 연속 점수를 사용해야 합니다. 평가 대상 정답에 양성과 음성이 모두 없으면 AUC가 정의되지 않습니다. micro에서는 펼친 정답 전체를 기준으로 확인합니다.

출처: [scikit-learn roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

