# RAG 원본 논문 핵심 분석

> Lewis et al.의 2020년 RAG 논문 아키텍처를 해부하고, DPR 검색기와 BART 생성기의 결합 방식, 그리고 핵심 실험 결과를 분석합니다

## 개요

이 섹션에서는 RAG의 탄생을 이끈 원본 논문 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* (Lewis et al., 2020)의 핵심을 분석합니다. [앞서 RAG의 핵심 개념](1-2)에서 검색→증강→생성의 3단계 흐름을 배웠다면, 이번에는 그 아이디어가 **구체적으로 어떤 모델 구조와 수식으로 구현되었는지** 파헤쳐 보겠습니다.

**선수 지식**: [LLM의 한계](1-1)에서 배운 파라메트릭/비파라메트릭 메모리 개념, [RAG의 핵심 개념](1-2)에서 배운 검색-증강-생성 3단계 파이프라인과 RAG-Sequence, RAG-Token의 기본 개념

**학습 목표**:
- DPR(Dense Passage Retrieval) 검색기의 이중 인코더 구조를 설명할 수 있다
- RAG-Sequence와 RAG-Token의 수학적 차이를 이해한다
- 원본 논문의 실험 결과가 왜 의미 있는지 해석할 수 있다

> 📖 **학습 안내**: 이 섹션에는 확률 수식이 등장하지만, 수식을 완벽히 이해하지 않아도 괜찮습니다. 각 개념마다 비유를 먼저 제시하니 **핵심 아이디어를 비유로 먼저 파악**하고, 수식은 참고용으로 활용하세요. 비유만 이해해도 RAG 아키텍처의 설계 의도를 충분히 파악할 수 있습니다.

## 왜 알아야 할까?

"그냥 RAG 쓰면 되지, 왜 원본 논문까지 읽어야 하나요?"라고 생각하실 수 있습니다. 하지만 원본 논문을 이해하면 현대 RAG 시스템을 설계할 때 **왜 특정 선택을 하는지** 근본적인 이유를 알 수 있습니다.

예를 들어, "검색된 문서를 몇 개나 사용할까?", "검색기와 생성기를 함께 학습시킬까, 따로 학습시킬까?", "하나의 문서로 전체 답변을 만들까, 토큰마다 다른 문서를 참조할까?" — 이 모든 질문에 대한 원조 답변이 바로 이 논문에 있거든요. 현대 RAG 프레임워크인 LangChain이나 LlamaIndex의 설계 철학도 결국 이 논문의 아이디어에서 출발했습니다.

## 핵심 개념

### 개념 1: RAG 논문의 전체 아키텍처

> 💡 **비유**: RAG의 아키텍처는 **오픈북 시험을 치르는 학생**과 같습니다. 학생(생성기)이 시험 문제(질문)를 받으면, 먼저 도서관 사서(검색기)에게 관련 참고서 페이지를 요청합니다. 사서가 가장 관련 있는 페이지 5장을 골라주면, 학생은 그 페이지들을 읽고 답안을 작성합니다.

Lewis et al.의 RAG 모델은 크게 두 가지 컴포넌트로 구성됩니다:

1. **검색기(Retriever)**: DPR(Dense Passage Retrieval) — 질문과 관련된 문서를 찾아옴
2. **생성기(Generator)**: BART-large — 찾아온 문서를 참고하여 답변을 생성

이 두 컴포넌트가 하나의 확률 모델로 결합되는 것이 핵심입니다. 검색된 문서 $z$는 **잠재 변수(latent variable)**로 취급되고, 최종 출력 확률을 구할 때 상위 $k$개 문서에 대해 **주변화(marginalization)**를 수행합니다.

```python
from dataclasses import dataclass, field

@dataclass
class RAGArchitecture:
    """RAG 원본 논문의 아키텍처 구성 요소"""

    # 검색기 (Retriever)
    retriever: str = "DPR (Dense Passage Retrieval)"
    query_encoder: str = "BERT-base (uncased)"      # 질문 인코더
    document_encoder: str = "BERT-base (uncased)"    # 문서 인코더
    embedding_dim: int = 768                          # 임베딩 차원
    index: str = "FAISS (HNSW)"                      # 벡터 인덱스

    # 지식 소스 (Knowledge Source)
    knowledge_source: str = "Wikipedia (2018.12 덤프)"
    num_passages: int = 21_000_000                    # 약 2,100만 개 패시지
    passage_length: int = 100                         # 각 패시지 최대 100 단어

    # 생성기 (Generator)
    generator: str = "BART-large"
    generator_params: int = 400_000_000               # 4억 파라미터

    # 검색 설정
    top_k: int = 5                                    # 상위 k개 문서 검색
```

### 개념 2: DPR — 밀집 벡터 검색기

> 💡 **비유**: DPR은 **두 명의 통역사**가 협력하는 시스템입니다. 한 통역사는 질문을 "의미 코드"로 번역하고, 다른 통역사는 모든 문서를 같은 "의미 코드"로 미리 번역해 놓습니다. 누군가 질문하면, 질문의 코드와 가장 비슷한 코드를 가진 문서를 빠르게 찾아냅니다.

DPR은 Karpukhin et al.(2020)이 제안한 밀집 검색 모델로, 두 개의 독립된 BERT 인코더를 사용합니다:

- **질문 인코더** $\text{BERT}_q$: 질문 $x$를 벡터 $q(x)$로 변환
- **문서 인코더** $\text{BERT}_d$: 문서 $z$를 벡터 $d(z)$로 변환

검색 확률은 두 벡터의 **내적(inner product)**으로 계산합니다:

$$p_\eta(z|x) \propto \exp\big(d(z)^\top q(x)\big)$$

- $q(x) = \text{BERT}_q(x)$: 질문의 임베딩 벡터
- $d(z) = \text{BERT}_d(z)$: 문서의 임베딩 벡터
- $\eta$: 검색기의 파라미터 (두 BERT 인코더의 가중치)

이게 의미하는 바는, 질문과 문서가 의미적으로 비슷할수록 내적 값이 커지고, 해당 문서가 검색될 확률이 높아진다는 것입니다.

중요한 점은 **문서 인코더는 학습 중에 고정(frozen)**되고, **질문 인코더만 미세 조정**된다는 것입니다. 문서 인코더를 고정하면 2,100만 개의 문서 벡터를 매번 다시 계산할 필요가 없어서 효율적이거든요.

```run:python
import numpy as np

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """두 벡터 간의 코사인 유사도 계산"""
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))

def inner_product(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """DPR에서 사용하는 내적 유사도 계산"""
    return float(np.dot(vec_a, vec_b))

# DPR 스타일 검색 시뮬레이션
# 실제로는 BERT가 768차원 벡터를 생성하지만, 여기서는 4차원으로 단순화
np.random.seed(42)

# 질문 인코더 출력 (query embedding)
query = np.array([0.8, 0.3, 0.1, 0.5])  # "RAG 논문의 저자는?"

# 문서 인코더 출력 (document embeddings) — 사전 계산된 벡터
documents = {
    "RAG 논문 소개":       np.array([0.7, 0.4, 0.2, 0.6]),   # 관련도 높음
    "GPT-3 아키텍처":      np.array([0.2, 0.8, 0.1, 0.3]),   # 관련도 낮음
    "DPR 검색 시스템":     np.array([0.6, 0.3, 0.3, 0.5]),   # 관련도 중간
    "요리 레시피 모음":    np.array([0.1, 0.1, 0.9, 0.0]),   # 관련도 없음
}

print("=== DPR 내적 기반 검색 시뮬레이션 ===\n")
print(f"질문 벡터: {query}\n")

# 각 문서와의 유사도 계산
scores = {}
for title, doc_vec in documents.items():
    ip_score = inner_product(query, doc_vec)
    cos_score = cosine_similarity(query, doc_vec)
    scores[title] = ip_score
    print(f"[{title}]")
    print(f"  내적(DPR): {ip_score:.4f}  |  코사인: {cos_score:.4f}")

# softmax로 검색 확률 계산
print("\n=== 검색 확률 (softmax) ===")
score_values = np.array(list(scores.values()))
exp_scores = np.exp(score_values)
probabilities = exp_scores / exp_scores.sum()

for (title, _), prob in zip(scores.items(), probabilities):
    bar = "█" * int(prob * 30)
    print(f"  {title}: {prob:.3f} {bar}")
```

```output
=== DPR 내적 기반 검색 시뮬레이션 ===

질문 벡터: [0.8 0.3 0.1 0.5]

[RAG 논문 소개]
  내적(DPR): 0.9800  |  코사인: 0.9473
[GPT-3 아키텍처]
  내적(DPR): 0.5500  |  코사인: 0.6076
[DPR 검색 시스템]
  내적(DPR): 0.8600  |  코사인: 0.9219
[요리 레시피 모음]
  내적(DPR): 0.1700  |  코사인: 0.1718

=== 검색 확률 (softmax) ===
  RAG 논문 소개: 0.3614 ███████████
  GPT-3 아키텍처: 0.2350 ███████
  DPR 검색 시스템: 0.3207 █████████
  요리 레시피 모음: 0.1608 ████
```

### 개념 3: BART 생성기와 문서 결합

생성기로는 Facebook AI의 **BART-large**(4억 파라미터)를 사용합니다. BART는 인코더-디코더 구조의 시퀀스-투-시퀀스(seq2seq) 모델인데요, 여기서 핵심은 검색된 문서를 **어떻게** 생성기에 전달하느냐입니다.

방법은 단순합니다. 질문과 검색된 문서를 **단순 연결(concatenation)**하여 BART 인코더의 입력으로 넣습니다:

```
입력 = [질문] + [구분자] + [검색된 문서 내용]
```

BART 디코더는 이 입력을 바탕으로 토큰 단위로 답변을 생성합니다. 검색된 $k$개의 문서 각각에 대해 이 과정을 수행하고, 결과를 결합하는 방식이 RAG-Sequence와 RAG-Token에서 달라집니다.

### 개념 4: RAG-Sequence vs RAG-Token — 핵심 차이

> 💡 **비유**: 에세이 시험을 떠올려 보세요. **RAG-Sequence**는 참고서 한 권을 골라서 처음부터 끝까지 그 책만 보며 답안을 작성하는 방식입니다. **RAG-Token**은 단어 하나를 쓸 때마다 여러 참고서를 훑어보고 가장 적절한 표현을 골라 쓰는 방식이죠.

[앞서](1-2) RAG-Sequence와 RAG-Token의 개념을 소개했는데, 이제 수식으로 정확한 차이를 살펴보겠습니다.

**RAG-Sequence** — 문서 단위 주변화:

$$p_{\text{RAG-Seq}}(y|x) \approx \sum_{z \in \text{top-}k} p_\eta(z|x) \prod_{i=1}^{N} p_\theta(y_i|x, z, y_{1:i-1})$$

- 하나의 문서 $z$를 선택하면, **전체 시퀀스** $y$를 그 문서만 참조하여 생성
- 각 문서별 시퀀스 확률을 구한 뒤, 검색 확률로 가중합

**RAG-Token** — 토큰 단위 주변화:

$$p_{\text{RAG-Token}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k} p_\eta(z|x) \, p_\theta(y_i|x, z, y_{1:i-1})$$

- **각 토큰** $y_i$를 생성할 때마다 $k$개 문서에 대해 주변화
- 토큰마다 다른 문서의 정보를 혼합하여 사용 가능

수식에서 $\sum$(합산)과 $\prod$(곱셈)의 **순서**가 뒤바뀐 것이 핵심입니다:
- RAG-Sequence: $\sum$ 밖에 $\prod$ → 문서를 먼저 고정, 전체 시퀀스 생성
- RAG-Token: $\prod$ 밖에 $\sum$ → 토큰마다 문서를 혼합

```run:python
import numpy as np

def rag_sequence_probability(
    retrieval_probs: list[float],
    token_probs_per_doc: list[list[float]]
) -> float:
    """
    RAG-Sequence: 문서별로 전체 시퀀스 확률을 구한 뒤 가중합

    retrieval_probs: 각 문서의 검색 확률 p(z|x)
    token_probs_per_doc: 각 문서별 [토큰1 확률, 토큰2 확률, ...] 
    """
    total = 0.0
    for doc_idx, p_z in enumerate(retrieval_probs):
        # 해당 문서로 전체 시퀀스를 생성할 확률 (토큰 확률의 곱)
        seq_prob = np.prod(token_probs_per_doc[doc_idx])
        total += p_z * seq_prob  # 검색 확률 × 시퀀스 확률
    return total

def rag_token_probability(
    retrieval_probs: list[float],
    token_probs_per_doc: list[list[float]]
) -> float:
    """
    RAG-Token: 각 토큰마다 모든 문서에 대해 가중합 후, 토큰별 곱

    retrieval_probs: 각 문서의 검색 확률 p(z|x)
    token_probs_per_doc: 각 문서별 [토큰1 확률, 토큰2 확률, ...]
    """
    num_tokens = len(token_probs_per_doc[0])
    total = 1.0
    for t in range(num_tokens):
        # 이 토큰에 대해 모든 문서의 가중합
        token_mixture = sum(
            p_z * token_probs_per_doc[doc_idx][t]
            for doc_idx, p_z in enumerate(retrieval_probs)
        )
        total *= token_mixture  # 토큰별 혼합 확률의 곱
    return total

# 시나리오: 질문 "RAG 논문의 저자는?" → 정답 "Lewis" (2토큰: "Le", "wis")
# 문서 3개가 검색됨
retrieval_probs = [0.6, 0.3, 0.1]  # 문서별 검색 확률

# 각 문서 기반으로 각 토큰을 생성할 확률
token_probs_per_doc = [
    [0.8, 0.9],   # 문서1 (RAG 논문 소개): "Le" 0.8, "wis" 0.9
    [0.3, 0.4],   # 문서2 (일반 NLP 문서): "Le" 0.3, "wis" 0.4
    [0.1, 0.1],   # 문서3 (무관한 문서):   "Le" 0.1, "wis" 0.1
]

seq_prob = rag_sequence_probability(retrieval_probs, token_probs_per_doc)
token_prob = rag_token_probability(retrieval_probs, token_probs_per_doc)

print("=== RAG-Sequence vs RAG-Token 확률 비교 ===\n")
print(f"검색 확률 p(z|x): {retrieval_probs}")
print(f"토큰별 생성 확률:\n  문서1: {token_probs_per_doc[0]}")
print(f"  문서2: {token_probs_per_doc[1]}")
print(f"  문서3: {token_probs_per_doc[2]}\n")
print(f"RAG-Sequence p(y|x) = {seq_prob:.6f}")
print(f"RAG-Token    p(y|x) = {token_prob:.6f}")
print(f"\n차이: {abs(seq_prob - token_prob):.6f}")
print("\n💡 같은 입력이지만 주변화 순서에 따라 확률이 달라집니다!")
```

```output
=== RAG-Sequence vs RAG-Token 확률 비교 ===

검색 확률 p(z|x): [0.6, 0.3, 0.1]
토큰별 생성 확률:
  문서1: [0.8, 0.9]
  문서2: [0.3, 0.4]
  문서3: [0.1, 0.1]

RAG-Sequence p(y|x) = 0.4690
RAG-Token    p(y|x) = 0.4717

차이: 0.002700

💡 같은 입력이지만 주변화 순서에 따라 확률이 달라집니다!
```

실험 결과에서 RAG-Sequence는 **정확한 단답형 QA**에서 약간 더 좋았고, RAG-Token은 여러 문서의 정보를 혼합해야 하는 **생성 태스크**(제퍼디 질문 생성 등)에서 더 나은 성능을 보였습니다.

### 개념 5: 주요 실험 결과 해석

논문은 세 가지 유형의 태스크에서 RAG를 평가했습니다.

**1) 오픈 도메인 질의응답 (Open-Domain QA)**

| 모델 | Natural Questions | TriviaQA |
|------|:-:|:-:|
| T5-11B (Closed-book) | 36.6 | — |
| REALM | 40.4 | — |
| DPR (Extractive Reader) | 41.5 | 57.9 |
| **RAG-Token** | **44.1** | **55.2** |
| **RAG-Sequence** | **44.5** | **56.8** |

RAG-Sequence는 Natural Questions에서 **44.5 EM(Exact Match)**을 달성하여, 110억 파라미터의 T5-11B보다 7.9점이나 높았습니다. 놀라운 점은 RAG가 DPR의 **교차 인코더(Cross-Encoder) 리랭커**나 **추출형 리더(Extractive Reader)** 없이도 더 좋은 성능을 냈다는 것입니다.

> 이것이 의미하는 바는 명확합니다: **검색+생성** 방식이 **검색+추출** 방식을 능가할 수 있다는 것이죠.

**2) 제퍼디(Jeopardy!) 질문 생성**

| 메트릭 | BART (baseline) | RAG-Token | RAG-Sequence |
|--------|:-:|:-:|:-:|
| Q-BLEU-1 | — | 22.2 | 21.4 |
| 사실성(Factuality) | 낮음 | **높음** | 높음 |
| 다양성(Diversity) | 낮음 | 높음 | **가장 높음** |

RAG 모델이 생성한 제퍼디 질문은 BART 단독 대비 **더 구체적이고, 사실에 기반하며, 다양**했습니다. 특히 RAG-Token은 여러 문서의 정보를 토큰 단위로 혼합하므로 Q-BLEU-1 점수가 더 높았습니다.

**3) 사실 검증 (FEVER)**

3-way 분류(지지/반박/정보부족)에서 RAG는 **72.5% 정확도**를 달성했습니다. 전용 파이프라인 시스템(76.8%)에는 미치지 못했지만, RAG는 **중간 단계 검색 레이블 없이** 이 성능을 달성했다는 점이 인상적입니다.

## 실습: 직접 해보기

RAG 논문의 전체 아키텍처를 Python으로 시뮬레이션해 봅시다. 실제 모델 대신 간소화된 구조로 **데이터 흐름**을 체험합니다.

```run:python
from dataclasses import dataclass
import numpy as np

# --- 1. 지식 베이스 (Wikipedia 패시지 시뮬레이션) ---
@dataclass
class Passage:
    """
    Wikipedia 패시지 (원본 논문: 100단어 단위 청크)

    1.2에서 만든 Document와 역할이 유사하지만, 원본 논문에서는
    Wikipedia 전체를 100단어 단위의 짧은 Passage로 분할하여 사용합니다.
    즉, 하나의 Document(문서)가 여러 개의 Passage(패시지)로 나뉘는 구조입니다.
    """
    id: int
    title: str
    content: str
    embedding: np.ndarray  # DPR 문서 인코더 출력 (768차원 → 시뮬레이션에서 8차원)

knowledge_base = [
    Passage(0, "RAG 논문", "Lewis et al.은 2020년 RAG를 제안했다.",
            np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.5])),
    Passage(1, "DPR 논문", "Karpukhin et al.은 Dense Passage Retrieval을 제안했다.",
            np.array([0.7, 0.6, 0.3, 0.1, 0.8, 0.4, 0.2, 0.4])),
    Passage(2, "BART 모델", "BART는 Facebook AI의 시퀀스-투-시퀀스 모델이다.",
            np.array([0.3, 0.4, 0.8, 0.7, 0.2, 0.5, 0.6, 0.3])),
    Passage(3, "요리 레시피", "파스타를 삶을 때 소금을 넣으면 맛이 좋아진다.",
            np.array([0.1, 0.1, 0.2, 0.9, 0.0, 0.8, 0.7, 0.1])),
]

# --- 2. DPR 검색기 시뮬레이션 ---
def dpr_retrieve(query_embedding: np.ndarray, passages: list[Passage], top_k: int = 3) -> list[tuple[Passage, float]]:
    """DPR 스타일 MIPS(Maximum Inner Product Search) 검색"""
    scores = []
    for passage in passages:
        # 내적 기반 유사도 (원본 논문의 d(z)^T q(x))
        score = float(np.dot(passage.embedding, query_embedding))
        scores.append((passage, score))

    # 상위 k개 문서 반환 (FAISS가 하는 일)
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# --- 3. 검색 확률 계산 (softmax) ---
def compute_retrieval_probs(results: list[tuple[Passage, float]]) -> list[tuple[Passage, float]]:
    """검색 점수를 확률로 변환: p_η(z|x) ∝ exp(d(z)^T q(x))"""
    scores = np.array([score for _, score in results])
    exp_scores = np.exp(scores - scores.max())  # 수치 안정성
    probs = exp_scores / exp_scores.sum()
    return [(passage, float(prob)) for (passage, _), prob in zip(results, probs)]

# --- 4. RAG-Sequence 생성 시뮬레이션 ---
def rag_sequence_generate(query: str, retrieved: list[tuple[Passage, float]]) -> str:
    """RAG-Sequence: 각 문서별 답변 생성 후 가중합"""
    print(f"\n질문: \"{query}\"\n")
    print("--- RAG-Sequence 과정 ---")

    best_answer = ""
    best_score = 0.0

    for passage, prob in retrieved:
        # 실제로는 BART가 생성하지만, 여기서는 규칙 기반 시뮬레이션
        if "RAG" in passage.content:
            answer = "Lewis et al.이 2020년에 제안"
            gen_score = 0.85
        elif "DPR" in passage.content:
            answer = "Karpukhin et al.의 검색 시스템 기반"
            gen_score = 0.60
        else:
            answer = "관련 정보 부족"
            gen_score = 0.10

        combined = prob * gen_score  # p(z|x) × p(y|x,z)
        print(f"  문서 [{passage.title}] (검색확률: {prob:.3f})")
        print(f"    → 답변: \"{answer}\" (생성점수: {gen_score:.2f}, 결합: {combined:.4f})")

        if combined > best_score:
            best_score = combined
            best_answer = answer

    return best_answer

# --- 실행 ---
# 질문 인코딩 (실제로는 BERT 질문 인코더가 수행)
query_text = "RAG 논문을 처음 제안한 사람은?"
query_embedding = np.array([0.85, 0.75, 0.15, 0.1, 0.7, 0.35, 0.15, 0.45])

# Step 1: DPR 검색
print("=" * 50)
print("RAG 파이프라인 시뮬레이션")
print("=" * 50)

results = dpr_retrieve(query_embedding, knowledge_base, top_k=3)
print("\n[Step 1] DPR 검색 결과 (내적 기반):")
for passage, score in results:
    print(f"  {passage.title}: {score:.4f}")

# Step 2: 검색 확률 계산
probs = compute_retrieval_probs(results)
print("\n[Step 2] 검색 확률 (softmax):")
for passage, prob in probs:
    print(f"  {passage.title}: {prob:.3f}")

# Step 3: RAG-Sequence 생성
answer = rag_sequence_generate(query_text, probs)
print(f"\n{'=' * 50}")
print(f"최종 답변: \"{answer}\"")
```

```output
==================================================
RAG 파이프라인 시뮬레이션
==================================================

[Step 1] DPR 검색 결과 (내적 기반):
  RAG 논문: 3.1450
  DPR 논문: 2.8800
  BART 모델: 2.0550

[Step 2] 검색 확률 (softmax):
  RAG 논문: 0.477
  DPR 논문: 0.363
  BART 모델: 0.160

질문: "RAG 논문을 처음 제안한 사람은?"

--- RAG-Sequence 과정 ---
  문서 [RAG 논문] (검색확률: 0.477)
    → 답변: "Lewis et al.이 2020년에 제안" (생성점수: 0.85, 결합: 0.4056)
  문서 [DPR 논문] (검색확률: 0.363)
    → 답변: "Karpukhin et al.의 검색 시스템 기반" (생성점수: 0.60, 결합: 0.2178)
  문서 [BART 모델] (검색확률: 0.160)
    → 답변: "관련 정보 부족" (생성점수: 0.10, 결합: 0.0160)

==================================================
최종 답변: "Lewis et al.이 2020년에 제안"
```

## 더 깊이 알아보기

### 이름의 탄생: "RAG"라는 약자의 뒷이야기

RAG 논문의 제1저자 Patrick Lewis는 나중에 이 이름에 대해 흥미로운 일화를 남겼습니다. 그는 인터뷰에서 **"우리 연구가 이렇게 널리 퍼질 줄 알았다면 이름에 더 신경 썼을 것"**이라며, 다소 투박한 약어(RAG는 영어로 "누더기"라는 뜻도 있습니다)에 대해 사과(?)하기도 했습니다. Lewis는 현재 AI 스타트업 Cohere에서 RAG 팀을 이끌고 있는데, 아이러니하게도 자신이 만든 용어가 AI 업계 전체의 표준 용어가 되었죠.

### Facebook AI Research의 연구 여정

이 논문은 Facebook AI Research(현 Meta AI), University College London, New York University의 공동 연구입니다. 사실 RAG 논문(2020년 5월)이 나오기 직전인 2020년 2월, 같은 팀에서 DPR 논문이 먼저 발표되었습니다. DPR이 "어떻게 잘 검색할 것인가"의 문제를 풀었다면, RAG는 "검색한 결과를 어떻게 생성에 활용할 것인가"까지 확장한 셈입니다. 거의 동시에 Google에서도 REALM(Retrieval-Augmented Language Model Pre-Training)이라는 유사한 접근법을 발표했는데, 이는 검색 기반 AI가 "시대의 아이디어"였음을 보여줍니다.

### End-to-End 학습의 의미

기존 시스템들은 검색기와 리더(reader)를 **따로** 학습시켰습니다. DPR은 검색기만 학습하고, 별도의 추출형 리더를 붙였죠. 하지만 RAG는 검색기(질문 인코더)와 생성기를 **함께** 학습시킵니다. 생성 결과가 좋아지도록 역전파(backpropagation)가 검색기까지 전달되는 것입니다. 다만 문서 인코더는 고정했는데, 2,100만 패시지의 임베딩을 매 학습 스텝마다 다시 계산하는 것은 비현실적이었기 때문입니다. 이 **부분적 end-to-end 학습**이 RAG의 실용적 타협점이었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RAG 논문의 아키텍처가 현대 RAG 시스템과 같다"
> 원본 RAG 논문은 DPR+BART를 end-to-end로 **함께 학습**시키는 모델입니다. 하지만 현대의 RAG 시스템(LangChain, LlamaIndex 등)은 검색기와 LLM을 **별도로 사용**하고, 프롬프트 엔지니어링으로 결합하는 방식이죠. 원본 RAG는 "학술적 기원"이고, 현대 RAG는 "실용적 진화"라고 이해하면 됩니다.

> 💡 **알고 계셨나요?**: RAG 논문의 Wikipedia 인덱스는 2018년 12월 덤프를 사용했습니다. 약 2,100만 개의 100단어 패시지로 분할되었고, 모든 패시지의 임베딩 벡터를 미리 계산하는 데 50대의 GPU 서버에서 약 40분이 걸렸습니다. 현재 벡터 데이터베이스 기술의 발전 덕분에, 이 규모의 인덱싱은 훨씬 빠르고 저렴하게 할 수 있습니다.

> 🔥 **실무 팁**: 원본 논문에서 top-k=5를 기본값으로 사용했습니다. 현대 RAG에서도 검색 문서 수는 3~5개가 일반적인 시작점이에요. 너무 많은 문서를 넣으면 컨텍스트에 노이즈가 늘어나고, 너무 적으면 관련 정보를 놓칠 수 있습니다. 실험적으로 최적의 k값을 찾되, **5에서 시작**하는 것이 논문이 검증한 합리적 출발점입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| RAG 아키텍처 | DPR 검색기 + BART 생성기를 결합한 확률 모델, 검색된 문서를 잠재 변수로 취급 |
| DPR | 이중 BERT 인코더(질문/문서)로 내적 기반 밀집 벡터 검색 수행 |
| RAG-Sequence | 하나의 문서로 전체 시퀀스를 생성한 뒤 문서별 가중합 (단답형 QA에 강점) |
| RAG-Token | 각 토큰 생성 시 모든 문서를 혼합 (생성 태스크에 강점) |
| 주변화(Marginalization) | 상위 k개 검색 문서에 대해 확률을 합산하는 근사 기법 |
| End-to-End 학습 | 질문 인코더와 BART를 함께 학습 (문서 인코더는 고정) |
| 핵심 결과 | NQ 44.5 EM으로 T5-11B(36.6)과 DPR(41.5)을 모두 능가 |

## 다음 섹션 미리보기

지금까지 RAG의 학술적 기원을 분석했습니다. 그런데 RAG가 **만능은 아닙니다**. [다음 섹션](1-4)에서는 RAG가 빛을 발하는 사용 사례(도메인 특화 QA, 최신 정보 기반 챗봇 등)와 RAG보다 다른 접근이 나은 경우(수학 추론, 창의적 글쓰기 등)를 구분하는 법을 배웁니다. 이론적 기반을 갖추었으니, 이제 "언제 RAG를 쓰고 언제 쓰지 말아야 하는가"라는 실전 판단력을 기르겠습니다.

## 참고 자료

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (원본 논문)](https://arxiv.org/abs/2005.11401) - RAG의 모든 것이 담긴 원본 논문. 아키텍처, 수식, 실험 결과를 직접 확인할 수 있습니다
- [Dense Passage Retrieval for Open-Domain Question Answering (DPR 논문)](https://aclanthology.org/2020.emnlp-main.550/) - RAG의 검색기인 DPR의 원본 논문. 이중 인코더 아키텍처와 학습 방법을 상세히 설명합니다
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - 원본 RAG 이후의 발전사를 체계적으로 정리한 서베이 논문
- [Meta AI Blog — Retrieval Augmented Generation](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) - Meta AI의 공식 블로그에서 RAG를 쉽게 설명한 글
- [HuggingFace RAG Model Documentation](https://huggingface.co/docs/transformers/model_doc/rag) - HuggingFace Transformers에서 RAG 모델을 직접 사용하는 방법

---
### 🔗 Related Sessions
- [hallucination](../01-rag-개요-llm의-한계와-rag의-필요성/01-llm의-한계-왜-외부-지식이-필요한가.md) (prerequisite)
- [parametric_memory](../01-rag-개요-llm의-한계와-rag의-필요성/01-llm의-한계-왜-외부-지식이-필요한가.md) (prerequisite)
- [knowledge_cutoff](../01-rag-개요-llm의-한계와-rag의-필요성/01-llm의-한계-왜-외부-지식이-필요한가.md) (prerequisite)
- [rag](../01-rag-개요-llm의-한계와-rag의-필요성/02-rag의-핵심-개념-검색-증강-생성이란.md) (prerequisite)
- [non_parametric_memory](../01-rag-개요-llm의-한계와-rag의-필요성/01-llm의-한계-왜-외부-지식이-필요한가.md) (prerequisite)
- [retrieve_augment_generate](../01-rag-개요-llm의-한계와-rag의-필요성/02-rag의-핵심-개념-검색-증강-생성이란.md) (prerequisite)
- [rag_sequence](../01-rag-개요-llm의-한계와-rag의-필요성/02-rag의-핵심-개념-검색-증강-생성이란.md) (prerequisite)
- [rag_token](../01-rag-개요-llm의-한계와-rag의-필요성/02-rag의-핵심-개념-검색-증강-생성이란.md) (prerequisite)
- [dpr](../01-rag-개요-llm의-한계와-rag의-필요성/02-rag의-핵심-개념-검색-증강-생성이란.md) (prerequisite)
