# FFN (Feed-Forward Network, 순전파 신경망)

[용어 목록으로 돌아가기](../용어설명.md)

## 1. 정의

FFN은 입력에서 출력으로 정보가 흐르며 순환 연결이 없는 신경망을 가리키는 넓은 개념입니다. 여러 완전연결층과 활성화 함수로 구성한 [MLP](MLP.md)는 FFN의 한 종류입니다.

다만 Transformer 문맥의 FFN은 보통 각 토큰에 적용하는 특정한 MLP 부분을 뜻합니다. 따라서 Transformer와 ViT 설명에서는 FFN과 MLP가 같은 구성 요소를 가리키기도 합니다.

## 2. 왜 필요한가요?

Self-Attention은 토큰 사이 정보를 주고받게 합니다. FFN은 그 결과를 받은 각 토큰 벡터의 특징을 비선형으로 가공합니다. 두 연산을 함께 사용하여 토큰 간 관계와 토큰 내부의 특징을 처리합니다.

## 3. 구체적인 예시

[Transformer 실습](../8_Transformer.ipynb)에서는 다음 구조를 사용합니다.

```text
토큰 벡터 8차원 → Linear → 32차원 → ReLU → Linear → 8차원
```

모든 토큰에 같은 가중치를 가진 FFN을 적용하지만 각 토큰은 독립적으로 계산합니다. FFN 자체가 서로 다른 토큰의 벡터를 섞지는 않습니다.

```text
커피 벡터 → 같은 FFN → 가공된 커피 벡터
한잔 벡터 → 같은 FFN → 가공된 한잔 벡터
어때 벡터 → 같은 FFN → 가공된 어때 벡터
```

각 입력 벡터에는 이미 Attention을 거치며 다른 토큰의 정보가 반영되어 있을 수 있습니다.

## 4. 코드와 주석

```python
import torch
import torch.nn as nn

ffn = nn.Sequential(
    nn.Linear(8, 32),  # 각 토큰의 특징 차원을 확장
    nn.ReLU(),        # 비선형 변환
    nn.Linear(32, 8)  # 원래 특징 차원으로 복원
)

x = torch.randn(2, 3, 8)  # 문장 2개, 토큰 3개, 특징 8개
y = ffn(x)               # (2, 3, 8): 마지막 차원에 적용
```

입출력 차원을 같게 하면 바깥에서 Residual 덧셈을 적용하기 편리합니다. 하지만 일반적인 FFN의 입력·출력 차원이 반드시 같아야 하는 것은 아닙니다. 확장 비율 4배도 흔한 설계 예시이며 필수 조건은 아닙니다.

## 5. 혼동하기 쉬운 개념

| 개념 | 차이 |
| --- | --- |
| MLP | FFN을 완전연결층과 활성화 함수로 구성한 형태 |
| Self-Attention | 토큰 사이 정보를 섞음. Transformer FFN은 토큰별 특징을 가공 |
| RNN | 이전 시점의 상태를 다음 시점으로 전달하는 순환 구조를 사용 |
| 순전파(forward pass) | 출력을 계산하는 실행 과정. FFN은 구조의 이름이며 RNN도 순전파를 수행 |

Feed-forward라고 해서 학습할 때 역전파를 하지 않는다는 뜻은 아닙니다. FFN도 loss의 역전파로 가중치를 학습합니다.

관련 문서: [MLP](MLP.md), [Transformer](Transformer.md), [ViT](ViT.md)
