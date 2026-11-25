# Accuracy (정확도): 다중 라벨의 완전 일치 정확도

[용어 목록으로 돌아가기](../용어설명.md)

## 정의와 필요한 이유
다중 라벨의 `accuracy_score(y_true, y_pred)`는 한 문장의 모든 라벨이 정확히 일치한 비율(Subset Accuracy)입니다. 문장 전체의 정답 집합을 맞혔는지 확인합니다. 값은 0~1이며 높을수록 좋습니다.

## 예시
클래스 순서가 [욕설, 혐오, 정상]일 때 정답 [1, 1, 0], 예측 [1, 0, 0]은 한 문장 전체로는 오답입니다. 10문장 중 6문장의 전체 라벨이 일치하면 0.6입니다.

## 코드
```python
accuracy = accuracy_score(y_true, y_pred)  # 둘 다 (문장 수, 클래스 수)
```

## 혼동 주의
각 칸을 몇 개 맞혔는지 세는 원소별 정확도와 다릅니다. 예시에서 두 칸을 맞혔어도 문장 단위 점수는 0입니다. F1은 부분적으로 맞힌 양성 라벨도 TP로 계산합니다.

출처: [scikit-learn accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

