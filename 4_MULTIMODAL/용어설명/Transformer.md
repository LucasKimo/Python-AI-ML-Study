# Transformer와 8_Transformer.ipynb 전체 요약

Transformer는 Attention을 중심으로 토큰 사이의 관계를 계산하고, 각 토큰을 문맥이 반영된 벡터로 바꾸는 신경망 구조입니다.

학습 자료: [8_Transformer.ipynb](../8_Transformer.ipynb) · [용어 목록으로 돌아가기](../용어설명.md)

## 1. 이 노트북에서 하는 일

RNN의 한계 → Seq2Seq → Attention → Multi-Head Attention → Transformer Encoder 블록 → Decoder의 Cross-Attention 순서로 학습합니다. 작은 임의 초기화 텐서로 계산과 shape를 확인하는 실습입니다. 데이터셋, 손실 함수, 역전파, Optimizer를 이용한 실제 학습이나 완성된 번역·생성 모델 구현은 포함하지 않습니다. 출력된 Attention 가중치는 학습된 언어 이해의 증거가 아닙니다.

## 2. 왜 Transformer가 필요한가?

RNN은 이전 시점의 hidden state를 받아 다음 시점을 계산합니다. 따라서 시간축 계산을 병렬화하기 어렵고, 멀리 떨어진 입력 사이의 관계를 학습하기도 어렵습니다.

Seq2Seq는 입력 시퀀스를 출력 시퀀스로 바꾸는 구조입니다. 예를 들어 Encoder가 영어 문장을 읽고 Decoder가 한국어 문장을 만듭니다. 고전적인 Attention 없는 RNN Seq2Seq에서는 입력을 마지막 hidden state 같은 고정 크기 벡터에 압축하므로 긴 문장의 정보가 손실되는 병목이 생길 수 있습니다.

Attention은 출력 위치마다 필요한 입력 위치를 직접 참고하도록 합니다. Transformer는 순환 계산 대신 Attention을 중심으로 구성하여 토큰 사이의 관계를 계산합니다. 다만 다음 토큰을 생성하는 추론에서는 보통 이전에 생성한 토큰을 이용해 순차적으로 생성합니다.

## 3. Self-Attention 계산 과정

초반 실습은 `lets / get / coffee` 세 토큰을 4차원으로 표현합니다.

1. 토큰에 정수 ID를 붙입니다. ID는 식별 번호이며 의미의 크기가 아닙니다.
2. `nn.Embedding(3, 4)`에서 벡터를 조회하여 `X`를 만듭니다. Shape는 `(3, 4)`입니다.
3. 서로 다른 학습 가능한 선형 변환으로 `Q`, `K`, `V`를 만듭니다.
4. `Q @ K.T`로 모든 토큰 쌍의 점수를 계산합니다. Shape는 `(3, 3)`입니다.
5. Key 차원의 제곱근으로 점수를 나눕니다.
6. 각 행에 Softmax를 적용하여 합이 1인 가중치를 만듭니다.
7. 가중치와 `V`를 곱하여 `(3, 4)`의 문맥 벡터를 얻습니다.

```python
# X의 같은 입력에서 서로 다른 역할의 표현을 학습
Q = W_Q(X)  # Query: 어떤 정보를 찾는가
K = W_K(X)  # Key: 비교 대상의 특징
V = W_V(X)  # Value: 실제로 가져올 정보

scores = Q @ K.transpose(-2, -1)
weights = torch.softmax(scores / math.sqrt(K.size(-1)), dim=-1)
output = weights @ V
```

수식: `Attention(Q, K, V) = softmax(QKᵀ / √d_k)V`

가중치 행렬의 행은 질문하는 Query 위치, 열은 참고하는 Key 위치입니다. 한 행이 `[0.2, 0.5, 0.3]`이면 새 벡터는 `0.2×V₀ + 0.5×V₁ + 0.3×V₂`입니다. 가중치는 정답 확률이 아니라 정보를 섞는 비율입니다. Scaled Dot-Product의 scaling은 차원이 커질 때 내적 점수가 커져 Softmax가 과도하게 포화되는 것을 완화합니다.

## 4. Multi-Head Attention

여러 Head가 서로 다른 투영 공간에서 Attention을 계산하고, 결과를 이어 붙여 출력 선형 변환을 적용합니다. 단순히 원본 임베딩을 자르는 것만으로 설명하면 학습 가능한 Q/K/V 변환이 빠집니다.

노트북의 예시는 `d_model=8`, `num_heads=2`이므로 Head당 4차원을 사용합니다. Head마다 다른 관계를 학습할 수 있지만, 특정 Head에 문법이나 의미 역할이 미리 지정되지는 않습니다.

| 항목 | Shape |
| --- | --- |
| 입력과 출력 | `(1, 3, 8)` |
| 기본 반환 가중치: Head 평균 | `(1, 3, 3)` |
| `average_attn_weights=False`: Head별 가중치 | `(1, 2, 3, 3)` |

## 5. Encoder는 어떻게 작동하는가?

후반 실습의 입력은 `커피 / 한잔 / 어때`입니다. 사전은 `<PAD>:0, 커피:1, 한잔:2, 어때:3`이며, 정수 ID `(1, 3)`을 8차원 임베딩 `(1, 3, 8)`으로 바꿉니다.

```text
문장 → 토큰 → ID → Embedding
                      ↓
             Positional Encoding 더하기
                      ↓
            Multi-Head Self-Attention
                      ↓
       원래 입력 더하기(Residual) → LayerNorm
                      ↓
            FFN: Linear → ReLU → Linear
                      ↓
        FFN 입력 더하기(Residual) → LayerNorm
                      ↓
             문맥이 반영된 토큰 벡터
```

- **Positional Encoding:** 순환 구조 없이도 순서를 활용하도록 위치 정보를 더합니다. 이 실습에서는 위치와 차원마다 다른 Sin/Cos 패턴을 만들며 `register_buffer`로 등록합니다. 위치 벡터는 학습 파라미터가 아닙니다. 더하기 전후 shape는 같습니다.
- **Residual Connection:** `x + F(x)`로 기존 표현에 새 정보를 보탭니다. 정보와 기울기가 전달될 직접적인 경로를 제공하여 깊은 모델의 학습을 돕습니다.
- **Layer Normalization:** `nn.LayerNorm(8)`은 각 토큰의 마지막 8개 특징을 기준으로 정규화합니다. 배치 전체나 문장 전체를 한꺼번에 정규화하는 것이 아닙니다. 초기 설정에서는 평균 약 0, 분산 약 1이지만 학습 가능한 스케일·편향이 바뀌면 최종 출력이 항상 이 값인 것은 아닙니다.
- **FFN:** 각 토큰에 동일한 작은 신경망을 독립적으로 적용합니다. 여기서는 `8 → 32 → 8`이며 ReLU로 비선형성을 추가합니다. Attention이 토큰 간 정보를 섞는다면 FFN은 각 토큰 내부의 특징을 가공합니다.
- **Dropout:** 블록 구현에서 학습 중 일부 값을 무작위로 제거하여 과적합을 줄이는 데 사용합니다.

직접 만든 `SimpleTransformerEncoderBlock`은 Attention과 FFN 뒤에 각각 Residual과 LayerNorm을 두는 Post-LN 구조입니다. 입력과 출력은 `(batch, seq_len, d_model)`을 유지하므로 같은 구조를 여러 층 쌓을 수 있습니다. 임베딩과 위치 정보 추가는 이 클래스 밖에서 수행합니다.

## 6. 두 가지 Mask

| 종류 | 필요한 이유 | 예시 |
| --- | --- | --- |
| Padding Mask | 길이를 맞추려고 붙인 PAD를 Key로 참고하지 않도록 함 | `커피 어때 <PAD>`의 마지막 열 가중치를 0으로 만듦 |
| Causal Mask | 다음 토큰 예측에서 미래 정답을 미리 보지 못하게 함 | 현재 위치 오른쪽의 토큰을 차단 |

수동 Attention 함수는 `mask == 0`을 차단합니다. 반면 노트북의 `nn.MultiheadAttention` Boolean 마스크는 `True`가 차단이라는 뜻입니다. 두 규칙을 혼동하면 안 됩니다.

`padding_idx=0`인 임베딩만으로는 Attention에서 PAD를 무시하지 않습니다. `key_padding_mask`가 필요합니다. 이 마스크는 PAD를 참고하는 열을 막으며, PAD 위치의 출력 벡터까지 자동으로 0으로 만들지는 않습니다.

Causal Mask는 상삼각 부분을 차단합니다. 학습에서는 여러 위치의 다음 토큰 예측을 병렬 계산하되 미래를 볼 수 없게 합니다. 노트북에서는 마스크를 생성해 확인하며, 직접 만든 Encoder 블록에는 이를 전달하는 기능을 구현하지 않았습니다.

## 7. Encoder와 Decoder, Cross-Attention

Encoder는 입력 토큰 사이 관계를 처리합니다. Encoder–Decoder 구조의 Decoder는 이전 출력에 대한 Masked Self-Attention과 입력에 대한 Cross-Attention 등을 이용해 출력을 생성합니다.

- Self-Attention: Q/K/V의 원천이 같은 시퀀스입니다.
- Cross-Attention: Query는 Decoder 표현에서, Key와 Value는 Encoder 출력에서 옵니다. 각 원천에 대한 투영은 Attention 모듈 내부에서 수행됩니다.

마지막 실습은 Encoder 메모리 `(1, 5, 8)`과 Decoder 상태 `(1, 3, 8)`를 임의로 만듭니다. 결과 `(1, 3, 8)`은 Query 위치마다 하나씩 나오며 가중치는 `(1, 3, 5)`입니다. 즉 출력의 세 위치가 입력의 다섯 위치를 얼마나 참고하는지 확인합니다. 완성된 Decoder나 토큰 생성 루프는 구현하지 않습니다.

## 8. 헷갈리기 쉬운 점과 노트북의 작은 오류

- Seq2Seq는 입력 시퀀스를 출력 시퀀스로 바꾸는 구조이고, Transformer는 이를 구현하는 데 쓸 수 있는 아키텍처입니다. 모든 Transformer가 Encoder와 Decoder를 함께 갖는 것은 아닙니다.
- 임베딩은 토큰의 기본 표현이고, Attention 출력은 주변 토큰 정보를 섞은 표현입니다. 두 벡터는 같은 차원을 가질 수 있지만 담는 정보가 다릅니다.
- Q와 K를 서로 다른 가중치로 만들므로 Attention 점수는 일반적으로 대칭 행렬이 아닙니다.
- 설명 속 한국어 세 토큰과 초반 코드의 영어 세 토큰은 서로 다른 예시입니다. 실제 코드를 읽을 때는 `token` 목록을 기준으로 행과 열을 해석합니다.
- Value 설명의 가중치가 `0.2, 0.5, 0.3`인데 가중합에는 마지막 계수가 `0.4`로 적혀 있습니다. `0.3`이 맞습니다.
- Head별 가중치를 확인하는 셀은 `head_weights`를 받지만 shape 출력에는 `attn_weights.shape`를 사용합니다. 이 줄은 `head_weights.shape`로 읽어야 의도한 `(1, 2, 3, 3)`과 일치합니다.
- `scaled_socres`, `Casual Mask` 등은 각각 `scaled_scores`, `Causal Mask`의 오타입니다. 전자는 코드에서 같은 이름을 일관되게 사용하여 그 자체로 실행 오류를 만들지는 않습니다.

## 9. 학습 후 설명할 수 있어야 하는 것

1. 고전적인 RNN Seq2Seq에서 왜 정보 병목이 생기는가?
2. Q/K/V의 역할과 Attention 수식의 각 연산은 무엇인가?
3. Attention 가중치와 최종 출력 벡터는 어떻게 다른가?
4. 여러 Head와 위치 정보는 왜 필요한가?
5. Encoder에서 Attention, Residual, LayerNorm, FFN이 하는 일은 무엇인가?
6. Padding Mask와 Causal Mask는 각각 무엇을 가리는가?
7. Cross-Attention 가중치가 `(출력 길이, 입력 길이)`인 이유는 무엇인가?

관련 용어: [Embedding](Embedding.md), [Hidden State](Hidden_State.md)
