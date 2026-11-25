# IoU (Intersection over Union, 교집합 / 합집합)

[용어 목록으로 돌아가기](../용어설명.md)

## 정의

IoU는 **두 영역이 겹치는 면적을, 두 영역이 차지하는 전체 면적으로 나눈 값**입니다. 객체 탐지에서는 주로 두 bounding box가 얼마나 겹치는지 나타냅니다.

```text
Intersection(교집합): 두 bounding box가 공통으로 덮는 부분
Union(합집합): 두 bounding box 중 하나라도 덮는 부분 전체

IoU = 교집합 면적 / 합집합 면적
합집합 면적 = A의 면적 + B의 면적 - 교집합 면적
```

면적을 더할 때 겹친 부분을 두 번 세므로, 합집합을 구할 때는 교집합을 한 번 빼 줍니다.

면적이 양수인 정상적인 bounding box에서 값은 `0`부터 `1` 사이입니다.

| 값 | 의미 |
| --- | --- |
| 0 | 겹치는 면적이 없음 |
| 1에 가까움 | 두 영역이 대부분 일치함 |
| 1 | 두 영역이 완전히 일치함 |

IoU `0.7`은 모델이 70% 확신한다는 뜻이 아닙니다. 합집합 면적 중 교집합 면적의 비율이 70%라는 뜻입니다.

## 필요한 이유

객체 탐지에서는 다음 두 상황에 사용할 수 있습니다.

- **위치 평가:** 예측 bounding box와 정답 bounding box가 얼마나 잘 겹치는지 확인합니다. 클래스와 평가 규칙 등도 함께 고려해야 하므로 IoU 하나가 탐지 정확도 전체를 뜻하지는 않습니다.
- **중복 정리:** 예측 bounding box끼리 비교해 많이 겹치는지 확인합니다. [NMS](./NMS.md)는 이 값을 이용해 중복 후보를 제거합니다.

## 구체적인 예시

두 bounding box의 좌표를 다음과 같이 가정합니다. [NMS 예시](./NMS.md)의 A와 B와 같은 bounding box입니다.

```text
A: 왼쪽 위 (0, 0), 오른쪽 아래 (10, 10)
B: 왼쪽 위 (1, 1), 오른쪽 아래 (11, 11)

A의 면적 = 10 × 10 = 100
B의 면적 = 10 × 10 = 100

겹치는 영역: 왼쪽 위 (1, 1), 오른쪽 아래 (10, 10)
교집합 면적 = 9 × 9 = 81
합집합 면적 = 100 + 100 - 81 = 119

IoU = 81 / 119 ≈ 0.681
```

두 bounding box가 차지하는 전체 영역 중 약 68.1%가 겹칩니다. 이는 A 면적만을 기준으로 계산한 `81/100`과 다릅니다.

여기서는 연속적인 좌표를 가정해 너비를 `x2 - x1`, 높이를 `y2 - y1`로 계산합니다.

## 관련 코드와 주석

아래는 PyTorch와 torchvision이 설치된 환경을 가정한 독립 예시입니다. 코드를 실행한 결과가 아니라 예상 값을 설명합니다.

```python
import torch
from torchvision.ops import box_iou

# 비교 기준 bounding box A. 한 행에 [x1, y1, x2, y2]를 저장
boxes_a = torch.tensor([
    [0.0, 0.0, 10.0, 10.0],
])  # shape: (1, 4)

# A와 비교할 bounding box B, C
boxes_bc = torch.tensor([
    [1.0, 1.0, 11.0, 11.0],    # B: A와 많이 겹침
    [20.0, 20.0, 30.0, 30.0],  # C: A와 겹치지 않음
])  # shape: (2, 4)

# 첫 번째 묶음의 모든 bounding box와 두 번째 묶음의 모든 bounding box를 비교
ious = box_iou(boxes_a, boxes_bc)  # shape: (1, 2)

# 예상 값: [[약 0.6807, 0.0]]
# ious[0, 0]: A와 B의 IoU
# ious[0, 1]: A와 C의 IoU
```

IoU는 bounding box의 좌표로 계산하며, bounding box의 클래스 이름이나 탐지 점수를 입력으로 사용하지 않습니다. 다른 클래스의 bounding box도 같은 위치라면 IoU가 높을 수 있습니다.

## 혼동하기 쉬운 개념과의 차이

| 개념 | IoU와의 차이 |
| --- | --- |
| [NMS](./NMS.md) | IoU를 이용해 중복 후보를 제거하는 절차입니다. IoU 자체는 bounding box를 제거하지 않습니다. |
| 탐지 점수 | 모델이 후보에 부여한 점수입니다. IoU는 두 영역의 위치·크기에서 계산합니다. |
| [Cosine 유사도](./Cosine_Similarity.md) | 벡터의 방향을 비교합니다. IoU는 영역이 공간적으로 겹치는 정도를 비교합니다. |

### NMS 기준과 평가 기준은 목적이 다릅니다

- **NMS:** 예측 bounding box끼리의 IoU가 기준보다 크면 낮은 점수의 중복 후보를 제거합니다.
- **위치 평가:** 예측 bounding box와 정답 bounding box의 IoU가 기준 이상인지 확인합니다. 이때 기준을 높이면 위치를 더 정확하게 맞혀야 통과합니다.

같은 IoU를 사용해도, 무엇과 무엇을 비교하고 어떤 결정을 내리는지가 다릅니다.

함수의 좌표 형식과 반환 행렬은 [torchvision box_iou 공식 문서](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.box_iou.html)를 참고하세요.
