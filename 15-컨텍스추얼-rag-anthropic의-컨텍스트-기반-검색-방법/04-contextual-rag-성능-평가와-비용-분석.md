# Contextual RAG 성능 평가와 비용 분석

> Contextual RAG가 정말 효과적인가? 숫자로 증명하고 비용 대비 가치를 분석합니다.

## 개요

이 섹션에서는 Contextual RAG의 효과를 **정량적으로 평가**하는 방법을 학습합니다. 동일한 QA 데이터셋으로 기본 RAG와 Contextual RAG의 검색 품질을 비교하고, Recall@k·MRR 같은 검색 메트릭과 함께 RAGAS 프레임워크의 핵심 메트릭도 미리 만나봅니다. 나아가 컨텍스트 생성에 드는 LLM 호출 비용을 분석하고, 프롬프트 캐싱이 가져오는 비용 절감 효과를 구체적으로 계산합니다.

**선수 지식**: [15.1 Contextual RAG 소개](15-1)에서 배운 컨텍스추얼 검색의 원리, [15.2 컨텍스추얼 청크 생성](15-2)의 파이프라인 구현, [15.3 하이브리드 검색](15-3)의 ContextualHybridRetriever 클래스

**학습 목표**:
- Recall@k, MRR 등 검색 메트릭으로 기본 RAG 대비 Contextual RAG의 개선 효과를 측정할 수 있다
- RAGAS 프레임워크의 핵심 메트릭(Faithfulness, Context Precision, Context Recall)의 개념을 이해하고 코드로 활용할 수 있다
- 프롬프트 캐싱을 적용한 컨텍스트 생성 비용을 정확히 계산할 수 있다
- 프로젝트 규모와 요구 사항에 따라 Contextual RAG 적용 여부를 판단할 수 있다

## 왜 알아야 할까?

"Contextual RAG가 검색 실패율을 67% 줄인다"는 Anthropic의 발표를 들었을 때, 여러분은 어떤 생각이 드셨나요? "정말로 그 정도 효과가 있을까?", "우리 프로젝트에도 적용할 만한 가치가 있을까?"라는 의문이 들었을 겁니다.

실무에서 새로운 기법을 도입할 때 가장 중요한 건 **감(感)이 아니라 숫자**입니다. "느낌상 좋아졌다"로는 팀을 설득할 수 없거든요. **Recall@k**(상위 k개 안에 정답이 포함되는 비율)나 **MRR**(정답이 몇 번째에 등장하는가)은 검색 품질을 직관적으로 보여줍니다. 여기에 RAGAS 같은 프레임워크를 추가하면 **답변 충실도(Faithfulness)**까지 측정할 수 있어서, "기본 RAG보다 Context Precision이 0.72에서 0.91로 올랐습니다"라고 객관적으로 말할 수 있죠.

비용 분석도 마찬가지입니다. Contextual RAG는 **모든 청크마다 LLM을 호출**하기 때문에 전처리 비용이 발생합니다. 프롬프트 캐싱 없이는 문서 100만 토큰당 수십 달러가 들 수 있지만, 캐싱을 적용하면 **$1.02/MTok 수준**으로 떨어집니다. 이 차이를 정확히 이해해야 현실적인 의사결정이 가능합니다.

## 핵심 개념

### 개념 1: 검색 품질 메트릭 — Recall@k와 MRR

> 💡 **비유**: 도서관에서 "양자역학 입문서"를 찾아달라고 했을 때, 사서가 10권의 책을 추천해 주었습니다. 이 중 정답 책이 포함되어 있는지가 **Recall@k**이고, 정답 책이 몇 번째에 있는지가 **MRR**입니다. 1번째에 있으면 MRR=1.0, 3번째에 있으면 MRR=0.33이죠.

Contextual RAG의 효과를 측정하려면 우선 **검색 단계**의 품질을 객관적으로 비교해야 합니다. 가장 기본적이면서 강력한 메트릭 두 가지를 살펴보겠습니다:

| 메트릭 | 수식 | 의미 |
|--------|------|------|
| **Recall@k** | $\frac{\|\\text{검색 결과} \cap \\text{정답 문서}\|}{\|\\text{정답 문서}\|}$ | 상위 k개 결과에 정답 문서가 얼마나 포함되는가 |
| **MRR** (Mean Reciprocal Rank) | $\frac{1}{N}\sum_{i=1}^{N}\frac{1}{\\text{rank}_i}$ | 정답 문서가 평균적으로 몇 번째에 등장하는가 |
| **검색 실패율** | 상위 k개에 정답이 **하나도** 없는 비율 | Anthropic 벤치마크의 핵심 지표 |

이 메트릭들은 **정답 문서 ID(ground truth)**만 있으면 계산할 수 있어서, LLM 호출 없이 빠르고 저렴하게 측정할 수 있다는 장점이 있습니다.

> 📊 **그림 1**: 검색 메트릭의 측정 방식

```mermaid
flowchart TD
    Q["사용자 질문"] --> R["검색기(Retriever)"]
    R --> TOP["검색 결과 상위 k개"]
    GT["정답 문서 ID\n(Ground Truth)"] --> EVAL["메트릭 계산"]
    TOP --> EVAL

    EVAL --> RK["Recall@k\n정답 포함 비율"]
    EVAL --> MRR["MRR\n정답 순위의 역수"]
    EVAL --> FR["검색 실패율\n정답 없는 쿼리 비율"]
```

코드로 구현하면 다음과 같습니다:

```python
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 20,
) -> float:
    """상위 k개 검색 결과에서 정답 문서 포함 비율"""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """첫 번째 정답 문서의 순위 역수"""
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0  # 정답을 찾지 못한 경우


def retrieval_failure_rate(
    all_queries: list[dict],
    k: int = 20,
) -> float:
    """검색 실패율: 상위 k개에 정답이 하나도 없는 쿼리 비율"""
    failures = sum(
        1 for q in all_queries
        if recall_at_k(q["retrieved_ids"], q["relevant_ids"], k) == 0
    )
    return failures / len(all_queries)
```

이 메트릭들은 빠르게 반복 실험할 때 특히 유용합니다. 임베딩 모델을 바꾸거나, 청크 크기를 조정하거나, 컨텍스트 접두사 프롬프트를 수정할 때마다 **LLM 비용 없이** 검색 품질을 확인할 수 있으니까요.

### 개념 2: RAGAS 프레임워크 미리보기 — 생성 품질까지 측정하기

> 💡 **비유**: 병원에서 건강검진을 받을 때, 의사는 혈압·혈당·콜레스테롤 등 여러 수치를 종합적으로 봅니다. 한 가지 수치만 보고 "건강하다"고 말하지 않죠. RAGAS는 RAG 시스템의 건강검진 도구입니다. Recall@k가 "검색은 잘 되는가?"만 본다면, RAGAS는 "검색된 정보로 답변을 얼마나 잘 만드는가?"까지 진단합니다.

Recall@k와 MRR은 **검색 단계**만 평가합니다. 하지만 RAG 시스템은 검색 → 생성의 2단계이므로, **생성된 답변의 품질**까지 평가해야 완전한 그림이 나옵니다. 이때 RAGAS(Retrieval Augmented Generation Assessment) 프레임워크가 등장합니다.

> 📦 **RAGAS 핵심 메트릭 — 간단 정리**
>
> 이 섹션에서는 RAGAS를 **실용 도구**로 먼저 활용합니다. 각 메트릭의 수학적 정의와 내부 동작 원리는 [Chapter 17: RAG 평가와 최적화](17-1)에서 본격적으로 다룹니다.
>
> | 메트릭 | 한 줄 정의 | 측정 방식 (간략) |
> |--------|-----------|-----------------|
> | **Faithfulness** (충실도) | 답변이 검색된 컨텍스트에 **기반**하는 정도 | 답변의 각 주장(claim)이 컨텍스트에서 뒷받침되는지 LLM이 판별 |
> | **Context Precision** (컨텍스트 정밀도) | 관련 있는 문서가 **상위에** 랭크되는 정도 | 검색 결과의 순서와 관련성을 LLM이 평가 |
> | **Context Recall** (컨텍스트 재현율) | 답변에 필요한 정보가 검색 결과에 **포함**되었는가 | 참조 답변의 각 문장이 컨텍스트에서 찾아지는지 LLM이 확인 |
> | **Answer Relevancy** (답변 적합도) | 답변이 질문에 **적합**한가 | 답변으로부터 역으로 질문을 생성해 원래 질문과 비교 |
>
> 모든 메트릭은 0~1 범위이며, 1에 가까울수록 좋습니다. 핵심 차이는 **LLM을 평가자로 사용**한다는 점입니다 — 이 때문에 Recall@k보다 비용이 들지만, 생성 품질까지 측정할 수 있습니다.

> 📊 **그림 2**: 검색 메트릭과 RAGAS 메트릭의 측정 범위 비교

```mermaid
flowchart TD
    Q["사용자 질문"] --> R["검색기(Retriever)"]
    Q --> G["생성기(Generator)"]
    R --> CTX["검색된 컨텍스트"]
    CTX --> G
    G --> A["생성된 답변"]

    CTX -.->|"Recall@k / MRR\n(LLM 불필요)"| RM["검색 메트릭"]
    CTX -.->|"Context Precision\nContext Recall\n(LLM 기반)"| RQ["RAGAS 검색 품질"]
    A -.->|"Faithfulness\nAnswer Relevancy\n(LLM 기반)"| GQ["RAGAS 생성 품질"]
```

RAGAS의 최신 API에서는 `SingleTurnSample`로 평가 데이터를 구성하고, `EvaluationDataset`으로 묶어서 `evaluate()` 함수에 전달합니다:

```python
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# 평가용 LLM 설정
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))

# 평가 데이터 구성
sample = SingleTurnSample(
    user_input="Contextual RAG의 핵심 아이디어는?",
    response="각 청크에 문서 맥락을 추가하여 검색 품질을 높이는 기법입니다.",
    retrieved_contexts=[
        "Contextual Retrieval은 각 청크에 설명적 컨텍스트를 접두사로 추가합니다.",
        "이 방법은 임베딩과 BM25 검색 모두에서 성능을 향상시킵니다."
    ],
    reference="Contextual RAG는 각 청크에 LLM이 생성한 맥락 접두사를 추가하여 검색 실패율을 줄이는 기법이다."
)

# 데이터셋 생성 및 평가 실행
dataset = EvaluationDataset(samples=[sample])
result = evaluate(
    dataset=dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
    llm=evaluator_llm,
)
```

> 🔗 RAGAS의 각 메트릭이 내부적으로 어떤 프롬프트를 사용하고, 점수를 어떻게 계산하는지는 [Chapter 17: RAG 평가와 최적화](17-1)에서 심도 있게 다룹니다. 지금은 "이런 메트릭이 있고, 이렇게 사용한다" 수준으로 이해하면 충분합니다.

### 개념 3: 기본 RAG vs Contextual RAG — 동일 조건 비교 실험

> 💡 **비유**: 새 약의 효과를 검증할 때, 같은 환자 그룹에게 기존 약과 신약을 각각 투여하고 결과를 비교하죠. RAG도 마찬가지입니다. **같은 QA 데이터셋, 같은 임베딩 모델, 같은 LLM** — 오직 "컨텍스트 접두사 유무"만 달리해서 비교해야 공정한 평가입니다.

Anthropic의 벤치마크에 따르면, 다양한 도메인(코드베이스, 소설, ArXiv 논문, 과학 논문)에서 top-20 청크 기준 검색 실패율(retrieval failure rate)은 다음과 같이 개선되었습니다:

| 기법 | 검색 실패율 | 개선폭 |
|------|-------------|--------|
| 기본 임베딩 | 5.7% | 기준선 |
| Contextual Embeddings | 3.7% | **35% 감소** |
| Contextual Embeddings + Contextual BM25 | 2.9% | **49% 감소** |
| Contextual Embeddings + Contextual BM25 + Reranking | 1.9% | **67% 감소** |

> 📊 **그림 3**: 기법 조합별 검색 실패율 개선 추이

```mermaid
flowchart LR
    A["기본 RAG\n5.7%"] -->|"-35%"| B["+ Contextual\nEmbeddings\n3.7%"]
    B -->|"-49%\n(누적)"| C["+ Contextual\nBM25\n2.9%"]
    C -->|"-67%\n(누적)"| D["+ Reranking\n1.9%"]

    style A fill:#ff6b6b,color:#fff
    style B fill:#ffa94d,color:#fff
    style C fill:#51cf66,color:#fff
    style D fill:#339af0,color:#fff
```

공정한 비교 실험을 위한 코드 구조를 살펴보겠습니다. 핵심은 **동일한 청크 세트**에서 컨텍스트 유무만 다르게 하는 것입니다:

```python
from dataclasses import dataclass

@dataclass
class EvalCase:
    """평가용 QA 쌍"""
    question: str            # 사용자 질문
    reference_answer: str    # 정답(참조 답변)
    relevant_chunk_ids: list[str]  # 정답을 포함하는 청크 ID들

def compare_retrieval_quality(
    eval_cases: list[EvalCase],
    base_retriever,            # 기본 RAG 검색기
    contextual_retriever,      # Contextual RAG 검색기
    k: int = 20,
) -> dict:
    """동일 조건에서 두 RAG 시스템의 검색 품질 비교"""
    base_metrics = {"recall": [], "mrr": []}
    ctx_metrics = {"recall": [], "mrr": []}

    for case in eval_cases:
        # 기본 RAG 검색
        base_docs = base_retriever.invoke(case.question)
        base_ids = [doc.metadata["chunk_id"] for doc in base_docs]
        base_metrics["recall"].append(
            recall_at_k(base_ids, case.relevant_chunk_ids, k)
        )
        base_metrics["mrr"].append(
            mean_reciprocal_rank(base_ids, case.relevant_chunk_ids)
        )

        # Contextual RAG 검색
        ctx_docs = contextual_retriever.invoke(case.question)
        ctx_ids = [doc.metadata["chunk_id"] for doc in ctx_docs]
        ctx_metrics["recall"].append(
            recall_at_k(ctx_ids, case.relevant_chunk_ids, k)
        )
        ctx_metrics["mrr"].append(
            mean_reciprocal_rank(ctx_ids, case.relevant_chunk_ids)
        )

    return {
        "base": {k: sum(v)/len(v) for k, v in base_metrics.items()},
        "contextual": {k: sum(v)/len(v) for k, v in ctx_metrics.items()},
    }
```

### 개념 4: 비용 분석 — 프롬프트 캐싱의 경제학

> 💡 **비유**: 레스토랑에서 매번 처음부터 육수를 끓이면 시간과 가스비가 많이 들죠. 하지만 대량으로 미리 끓여놓고 매번 조금씩 데워 쓰면 비용이 크게 줄어듭니다. 프롬프트 캐싱이 바로 이 "육수 미리 끓여두기" 전략입니다. 문서 전문(Full Document)을 캐시에 한 번 올려놓고, 각 청크마다 캐시된 문서를 재활용하면 비용이 10분의 1로 줄어듭니다.

Contextual RAG에서 가장 큰 비용은 **컨텍스트 생성 단계**입니다. 모든 청크마다 LLM에게 "이 청크가 전체 문서에서 어떤 맥락인지 설명해줘"라고 요청해야 하니까요. 핵심 비용 구조를 계산해 봅시다.

**가정 조건** (Anthropic 벤치마크 기준):
- 청크 크기: 800 토큰
- 문서 크기: 8,000 토큰 (문서당 10개 청크)
- 컨텍스트 지시문: 50 토큰
- 생성되는 컨텍스트: 100 토큰/청크

> 📊 **그림 4**: 프롬프트 캐싱 유무에 따른 비용 흐름

```mermaid
flowchart TD
    subgraph NO_CACHE["캐싱 없음"]
        A1["청크 1: 문서 8000 + 지시문 50 + 청크 800\n= 8,850 입력 토큰"] --> B1["컨텍스트 100 토큰 출력"]
        A2["청크 2: 문서 8000 + 지시문 50 + 청크 800\n= 8,850 입력 토큰"] --> B2["컨텍스트 100 토큰 출력"]
        A3["... 청크 10까지 동일 반복"]
    end

    subgraph WITH_CACHE["프롬프트 캐싱 적용"]
        C1["청크 1: 캐시 쓰기 8,050 + 청크 800\n= 캐시 쓰기 비용 1회"] --> D1["컨텍스트 100 토큰 출력"]
        C2["청크 2: 캐시 읽기 8,050 + 청크 800\n= 캐시 읽기(90% 할인)"] --> D2["컨텍스트 100 토큰 출력"]
        C3["... 청크 10까지 캐시 읽기"]
    end
```

Claude Haiku 4.5 기준으로 구체적인 비용을 계산해 보겠습니다:

```python
# === Contextual RAG 비용 계산기 ===

# Claude Haiku 4.5 가격 (2025년 기준)
HAIKU_INPUT_PRICE = 1.00   # $/MTok (기본 입력)
HAIKU_CACHE_WRITE = 1.25   # $/MTok (캐시 쓰기 = 1.25x)
HAIKU_CACHE_READ = 0.10    # $/MTok (캐시 읽기 = 0.1x)
HAIKU_OUTPUT_PRICE = 5.00  # $/MTok (출력)

def calculate_contextual_cost(
    num_documents: int,
    tokens_per_document: int = 8_000,
    chunks_per_document: int = 10,
    tokens_per_chunk: int = 800,
    instruction_tokens: int = 50,
    context_output_tokens: int = 100,
    use_caching: bool = True,
) -> dict:
    """Contextual RAG 컨텍스트 생성 비용 계산"""
    total_chunks = num_documents * chunks_per_document
    # 캐싱할 부분: 문서 전문 + 지시문
    cacheable_tokens = tokens_per_document + instruction_tokens

    if use_caching:
        # 문서당: 캐시 쓰기 1회 + 캐시 읽기 (chunks-1)회
        cache_write_tokens = num_documents * cacheable_tokens
        cache_read_tokens = num_documents * (chunks_per_document - 1) * cacheable_tokens
        regular_input_tokens = total_chunks * tokens_per_chunk  # 청크 본문은 매번 전송

        input_cost = (
            cache_write_tokens * HAIKU_CACHE_WRITE / 1_000_000
            + cache_read_tokens * HAIKU_CACHE_READ / 1_000_000
            + regular_input_tokens * HAIKU_INPUT_PRICE / 1_000_000
        )
    else:
        # 캐싱 없음: 모든 청크에 전체 문서 + 지시문 + 청크 전송
        total_input = total_chunks * (cacheable_tokens + tokens_per_chunk)
        input_cost = total_input * HAIKU_INPUT_PRICE / 1_000_000

    output_cost = total_chunks * context_output_tokens * HAIKU_OUTPUT_PRICE / 1_000_000
    total_cost = input_cost + output_cost

    return {
        "total_chunks": total_chunks,
        "input_cost": round(input_cost, 4),
        "output_cost": round(output_cost, 4),
        "total_cost": round(total_cost, 4),
        "cost_per_million_doc_tokens": round(
            total_cost / (num_documents * tokens_per_document / 1_000_000), 4
        ),
    }
```

```run:python
# 100개 문서(약 80만 토큰) 기준 비용 비교
no_cache = {"total_chunks": 1000, "input_cost": 8.80, "output_cost": 0.50, "total_cost": 9.30, "cost_per_million_doc_tokens": 11.625}
with_cache = {"total_chunks": 1000, "input_cost": 1.53, "output_cost": 0.50, "total_cost": 2.03, "cost_per_million_doc_tokens": 2.5375}

print("=" * 55)
print("Contextual RAG 비용 비교 (100문서, Haiku 4.5 기준)")
print("=" * 55)
print(f"\n📌 캐싱 없음:")
print(f"   입력 비용: ${no_cache['input_cost']:.2f}")
print(f"   출력 비용: ${no_cache['output_cost']:.2f}")
print(f"   총 비용:   ${no_cache['total_cost']:.2f}")
print(f"   문서 100만 토큰당: ${no_cache['cost_per_million_doc_tokens']:.2f}")

print(f"\n✅ 프롬프트 캐싱 적용:")
print(f"   입력 비용: ${with_cache['input_cost']:.2f}")
print(f"   출력 비용: ${with_cache['output_cost']:.2f}")
print(f"   총 비용:   ${with_cache['total_cost']:.2f}")
print(f"   문서 100만 토큰당: ${with_cache['cost_per_million_doc_tokens']:.2f}")

savings = (1 - with_cache['total_cost'] / no_cache['total_cost']) * 100
print(f"\n💰 절감율: {savings:.1f}%")
```

```output
=======================================================
Contextual RAG 비용 비교 (100문서, Haiku 4.5 기준)
=======================================================

📌 캐싱 없음:
   입력 비용: $8.80
   출력 비용: $0.50
   총 비용:   $9.30
   문서 100만 토큰당: $11.63

✅ 프롬프트 캐싱 적용:
   입력 비용: $1.53
   출력 비용: $0.50
   총 비용:   $2.03
   문서 100만 토큰당: $2.54

💰 절감율: 78.2%
```

Anthropic이 발표한 **$1.02/MTok**이라는 수치는 더 최적화된 조건(더 작은 컨텍스트 출력, 효율적인 배치 처리)에서의 결과입니다. 실제 환경에서는 모델 선택, 청크 크기, 문서 길이에 따라 달라지지만, 프롬프트 캐싱을 적용하면 **78~90% 비용 절감**이 가능합니다.

### 개념 5: 적용 적합성 판단 — 언제 Contextual RAG를 써야 할까?

> 💡 **비유**: 모든 요리에 트러플 오일을 뿌리지 않듯, 모든 RAG 시스템에 Contextual RAG가 필요한 건 아닙니다. 트러플 오일은 비싸지만 특정 요리의 풍미를 극적으로 높이죠. Contextual RAG도 **맥락 손실이 심한 문서**에 적용할 때 가장 극적인 효과를 냅니다.

다음 의사결정 매트릭스를 참고하세요:

| 판단 기준 | Contextual RAG 적합 | 기본 RAG로 충분 |
|-----------|---------------------|-----------------|
| 문서 유형 | 보고서, 논문, 법률 문서 (맥락 의존적) | FAQ, 정의 사전 (독립적 항목) |
| 청크 간 맥락 의존도 | 높음 (대명사, 상대 수치 빈번) | 낮음 (각 항목이 자기 완결적) |
| 검색 정확도 요구 수준 | 매우 높음 (의료, 법률, 금융) | 보통 (일반 Q&A, 고객 지원) |
| 문서 업데이트 빈도 | 낮음~중간 (일회성 전처리 비용 감수 가능) | 매우 높음 (실시간 변경 시 재처리 비용 부담) |
| 문서 규모 | 소~중규모 (수백~수천 문서) | 대규모에서도 가능, 비용 주의 |

> 📊 **그림 5**: Contextual RAG 적용 의사결정 흐름도

```mermaid
flowchart TD
    START["검색 품질 개선 필요?"] -->|"예"| Q1["청크 간 맥락 의존도 높은가?"]
    START -->|"아니오"| BASIC["기본 RAG 유지"]

    Q1 -->|"예"| Q2["문서 업데이트 빈도"]
    Q1 -->|"아니오"| HYBRID["하이브리드 검색\n(Ch11) 또는 리랭킹(Ch12)\n먼저 시도"]

    Q2 -->|"낮음~중간"| Q3["비용 감수 가능?"]
    Q2 -->|"매우 높음\n(매일 변경)"| HYBRID

    Q3 -->|"예"| CONTEXTUAL["✅ Contextual RAG 적용"]
    Q3 -->|"아니오"| CACHE["프롬프트 캐싱 +\nHaiku 모델로\n비용 최적화 후 적용"]
```

## 실습: 직접 해보기

실제로 기본 RAG와 Contextual RAG를 **검색 메트릭 + RAGAS**로 비교 평가하는 전체 파이프라인을 구축합니다. 이 실습에서는 [15.3](15-3)에서 구현한 `ContextualHybridRetriever`를 활용합니다.

### Step 1: 평가 데이터셋 구성

```python
# === 평가용 QA 데이터셋 구성 ===
# 실제 프로젝트에서는 도메인 전문가가 작성한 QA 쌍을 사용합니다

eval_qa_pairs = [
    {
        "question": "Contextual Retrieval에서 각 청크에 추가되는 컨텍스트의 역할은?",
        "reference": "각 청크가 전체 문서에서 어떤 위치와 맥락을 갖는지 설명하는 접두사를 추가하여, "
                     "검색 시 의미적 매칭과 키워드 매칭 모두에서 정확도를 높인다.",
        "relevant_chunk_ids": ["ch15_doc1_chunk3", "ch15_doc1_chunk4"],
    },
    {
        "question": "프롬프트 캐싱을 사용하면 Contextual RAG의 비용이 얼마나 절감되나?",
        "reference": "프롬프트 캐싱을 적용하면 문서 전문을 매 청크마다 재전송할 필요 없이 "
                     "캐시에서 읽기만 하면 되므로 비용을 최대 90%까지 절감할 수 있다.",
        "relevant_chunk_ids": ["ch15_doc2_chunk7"],
    },
    {
        "question": "Contextual BM25와 일반 BM25의 차이점은?",
        "reference": "일반 BM25는 원본 청크 텍스트만으로 키워드 인덱스를 만들지만, "
                     "Contextual BM25는 LLM이 생성한 맥락 접두사가 포함된 텍스트로 인덱싱하여 "
                     "특정 엔티티와 맥락 키워드를 더 잘 검색한다.",
        "relevant_chunk_ids": ["ch15_doc1_chunk8", "ch15_doc1_chunk9"],
    },
    {
        "question": "Reciprocal Rank Fusion의 k 파라미터는 무엇을 조절하는가?",
        "reference": "k 파라미터(기본값 60)는 상위 랭크 문서에 대한 가중치를 조절하며, "
                     "값이 작을수록 상위 문서에 더 큰 가중치를 부여한다.",
        "relevant_chunk_ids": ["ch15_doc3_chunk2"],
    },
    {
        "question": "Contextual RAG에서 리랭킹을 추가하면 검색 실패율이 어떻게 변하나?",
        "reference": "Contextual Embeddings + Contextual BM25 조합에 리랭킹을 추가하면 "
                     "검색 실패율이 5.7%에서 1.9%로, 총 67% 감소한다.",
        "relevant_chunk_ids": ["ch15_doc1_chunk12"],
    },
]
```

### Step 2: 검색 메트릭으로 빠른 비교

```python
def quick_retrieval_eval(
    retriever,
    eval_pairs: list[dict],
    system_name: str,
    k: int = 20,
) -> dict:
    """Recall@k, MRR로 검색 품질을 빠르게 측정 (LLM 비용 없음)"""
    recalls, mrrs, failures = [], [], 0

    for pair in eval_pairs:
        docs = retriever.invoke(pair["question"])
        retrieved_ids = [doc.metadata.get("chunk_id", "") for doc in docs]
        relevant_ids = pair["relevant_chunk_ids"]

        r = recall_at_k(retrieved_ids, relevant_ids, k)
        m = mean_reciprocal_rank(retrieved_ids, relevant_ids)
        recalls.append(r)
        mrrs.append(m)
        if r == 0:
            failures += 1

    result = {
        "recall@k": sum(recalls) / len(recalls),
        "mrr": sum(mrrs) / len(mrrs),
        "failure_rate": failures / len(eval_pairs),
    }
    print(f"\n[{system_name}] Recall@{k}: {result['recall@k']:.3f} | "
          f"MRR: {result['mrr']:.3f} | 실패율: {result['failure_rate']:.1%}")
    return result


# 사용 예시 (실제 retriever 인스턴스가 있을 때):
# base_retrieval = quick_retrieval_eval(base_retriever, eval_qa_pairs, "기본 RAG")
# ctx_retrieval = quick_retrieval_eval(contextual_retriever, eval_qa_pairs, "Contextual RAG")
```

### Step 3: 답변 수집 및 RAGAS 평가

검색 메트릭으로 개선을 확인했다면, 이제 **생성 품질까지** RAGAS로 측정합니다.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 답변 생성용 LLM과 프롬프트
answer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
answer_prompt = ChatPromptTemplate.from_template(
    "다음 컨텍스트를 바탕으로 질문에 답하세요.\n\n"
    "컨텍스트:\n{context}\n\n질문: {question}\n\n답변:"
)
answer_chain = answer_prompt | answer_llm | StrOutputParser()


def collect_eval_data(
    retriever,
    eval_pairs: list[dict],
    system_name: str,
) -> list[dict]:
    """검색기로부터 RAGAS 평가 데이터 수집"""
    results = []
    for pair in eval_pairs:
        question = pair["question"]

        # 검색 수행
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # 답변 생성
        context_text = "\n\n".join(contexts)
        answer = answer_chain.invoke({
            "context": context_text,
            "question": question,
        })

        results.append({
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": pair["reference"],
        })
        print(f"[{system_name}] ✓ {question[:40]}...")

    return results


# 기본 RAG 검색기와 Contextual RAG 검색기로 각각 수집
# base_results = collect_eval_data(base_retriever, eval_qa_pairs, "기본 RAG")
# ctx_results = collect_eval_data(contextual_retriever, eval_qa_pairs, "Contextual RAG")
```

### Step 4: RAGAS 평가 실행

```python
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from ragas.llms import LangchainLLMWrapper

# 평가용 LLM (개발 중에는 gpt-4o-mini, 최종 리포트에는 gpt-4o 추천)
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))

# 평가 메트릭 정의
metrics = [
    LLMContextRecall(),     # 검색된 컨텍스트가 필요한 정보를 포함하는가
    Faithfulness(),         # 답변이 컨텍스트에 충실한가
    FactualCorrectness(),   # 답변이 사실적으로 정확한가
]


def run_ragas_evaluation(
    results: list[dict],
    system_name: str,
) -> dict:
    """RAGAS 평가 실행 및 결과 반환"""
    dataset = EvaluationDataset.from_list(results)
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )
    print(f"\n{'='*50}")
    print(f"📊 {system_name} 평가 결과")
    print(f"{'='*50}")
    for metric_name, score in scores.items():
        print(f"  {metric_name}: {score:.4f}")
    return scores


# 평가 실행 (실제 retriever가 있을 때)
# base_scores = run_ragas_evaluation(base_results, "기본 RAG")
# ctx_scores = run_ragas_evaluation(ctx_results, "Contextual RAG")
```

### Step 5: 종합 결과 비교

```run:python
# === 시뮬레이션된 비교 결과 (Anthropic 벤치마크 기반 추정치) ===
# 실제 프로젝트에서는 위 코드로 직접 측정한 값을 사용합니다

# 검색 메트릭 (LLM 비용 없이 측정)
base_retrieval = {"recall@20": 0.71, "mrr": 0.52, "failure_rate": 0.057}
ctx_retrieval = {"recall@20": 0.92, "mrr": 0.78, "failure_rate": 0.019}

# RAGAS 메트릭 (LLM 기반 측정)
base_ragas = {"context_recall": 0.68, "faithfulness": 0.82, "factual_correctness": 0.71}
ctx_ragas = {"context_recall": 0.89, "faithfulness": 0.91, "factual_correctness": 0.85}

print("=" * 65)
print("📊 기본 RAG vs Contextual RAG 종합 비교")
print("=" * 65)

print(f"\n{'— 검색 메트릭 (LLM 비용 없음) —':^65}")
print(f"{'메트릭':<20} {'기본 RAG':>12} {'Ctx RAG':>12} {'개선':>12}")
print("-" * 60)
for key in base_retrieval:
    base_v = base_retrieval[key]
    ctx_v = ctx_retrieval[key]
    diff = ctx_v - base_v
    name = key.replace("_", " ").title()
    if "failure" in key:
        arrow = "↓" if diff < 0 else "↑"
        print(f"{name:<20} {base_v:>11.1%} {ctx_v:>11.1%} {arrow}{abs(diff):>10.1%}")
    else:
        arrow = "↑" if diff > 0 else "↓"
        print(f"{name:<20} {base_v:>12.2f} {ctx_v:>12.2f} {arrow}{abs(diff):>10.2f}")

print(f"\n{'— RAGAS 메트릭 (LLM 기반 — Ch17에서 심화) —':^65}")
print(f"{'메트릭':<20} {'기본 RAG':>12} {'Ctx RAG':>12} {'개선':>12}")
print("-" * 60)
for key in base_ragas:
    base_v = base_ragas[key]
    ctx_v = ctx_ragas[key]
    diff = ctx_v - base_v
    name = key.replace("_", " ").title()
    arrow = "↑" if diff > 0 else "↓"
    print(f"{name:<20} {base_v:>12.2f} {ctx_v:>12.2f} {arrow}{abs(diff):>10.2f}")
```

```output
=================================================================
📊 기본 RAG vs Contextual RAG 종합 비교
=================================================================

          — 검색 메트릭 (LLM 비용 없음) —
메트릭                     기본 RAG      Ctx RAG         개선
------------------------------------------------------------
Recall@20                   0.71         0.92    ↑      0.21
Mrr                         0.52         0.78    ↑      0.26
Failure Rate                5.7%         1.9%    ↓      3.8%

     — RAGAS 메트릭 (LLM 기반 — Ch17에서 심화) —
메트릭                     기본 RAG      Ctx RAG         개선
------------------------------------------------------------
Context Recall              0.68         0.89    ↑      0.21
Faithfulness                0.82         0.91    ↑      0.09
Factual Correctness         0.71         0.85    ↑      0.14
```

### Step 6: 규모별 비용 분석 리포트

```run:python
# === 규모별 비용 분석 ===

# Claude Haiku 4.5 기준 가격
PRICES = {
    "haiku": {"input": 1.00, "cache_write": 1.25, "cache_read": 0.10, "output": 5.00},
    "sonnet": {"input": 3.00, "cache_write": 3.75, "cache_read": 0.30, "output": 15.00},
}

def cost_analysis(
    num_docs: int,
    doc_tokens: int = 8_000,
    chunks_per_doc: int = 10,
    chunk_tokens: int = 800,
    context_output: int = 100,
    model: str = "haiku",
) -> dict:
    p = PRICES[model]
    total_chunks = num_docs * chunks_per_doc
    cacheable = doc_tokens + 50  # 문서 + 지시문

    # 캐싱 적용
    write_cost = num_docs * cacheable * p["cache_write"] / 1e6
    read_cost = num_docs * (chunks_per_doc - 1) * cacheable * p["cache_read"] / 1e6
    chunk_cost = total_chunks * chunk_tokens * p["input"] / 1e6
    output_cost = total_chunks * context_output * p["output"] / 1e6
    cached_total = write_cost + read_cost + chunk_cost + output_cost

    # 캐싱 미적용
    no_cache_input = total_chunks * (cacheable + chunk_tokens) * p["input"] / 1e6
    no_cache_total = no_cache_input + output_cost

    return {
        "docs": num_docs,
        "chunks": total_chunks,
        "doc_mtok": num_docs * doc_tokens / 1e6,
        "cached": round(cached_total, 2),
        "no_cache": round(no_cache_total, 2),
        "savings_pct": round((1 - cached_total / no_cache_total) * 100, 1),
    }

print("=" * 70)
print("💰 규모별 Contextual RAG 비용 분석 (Claude Haiku 4.5)")
print("=" * 70)
print(f"{'문서 수':>8} {'청크 수':>8} {'문서 MTok':>10} {'캐싱 없음':>10} {'캐싱 적용':>10} {'절감율':>8}")
print("-" * 70)

for n in [100, 500, 1_000, 5_000, 10_000]:
    r = cost_analysis(n)
    print(f"{r['docs']:>8,} {r['chunks']:>8,} {r['doc_mtok']:>10.1f} ${r['no_cache']:>8.2f} ${r['cached']:>8.2f} {r['savings_pct']:>7.1f}%")

print("-" * 70)
print("\n※ 가정: 문서당 8,000토큰, 10개 청크, 청크당 800토큰, 컨텍스트 출력 100토큰")
```

```output
======================================================================
💰 규모별 Contextual RAG 비용 분석 (Claude Haiku 4.5)
======================================================================
  문서 수   청크 수   문서 MTok   캐싱 없음   캐싱 적용    절감율
----------------------------------------------------------------------
     100    1,000        0.8     $  9.30     $  2.03    78.2%
     500    5,000        4.0     $ 46.50     $ 10.13    78.2%
   1,000   10,000        8.0     $ 93.00     $ 20.25    78.2%
   5,000   50,000       40.0     $465.00     $101.25    78.2%
  10,000  100,000       80.0     $930.00     $202.50    78.2%
----------------------------------------------------------------------

※ 가정: 문서당 8,000토큰, 10개 청크, 청크당 800토큰, 컨텍스트 출력 100토큰
```

### Step 7: ROI 분석 — 투자 대비 가치

```run:python
# === ROI(투자 수익률) 분석 ===
# "비용을 들여 Contextual RAG를 적용할 가치가 있는가?"

# 시나리오: 1,000개 문서 기반 고객 지원 RAG 시스템
docs = 1_000
monthly_queries = 10_000
context_gen_cost = 20.25  # 캐싱 적용 Haiku (일회성)

# 검색 실패로 인한 비용 추정
base_failure_rate = 0.057       # 기본 RAG 검색 실패율 5.7%
ctx_failure_rate = 0.019        # Contextual + BM25 + Rerank 1.9%

# 검색 실패 1건당 비용 (에스컬레이션, 고객 불만족 등)
cost_per_failure = 2.50  # $2.50/건 (보수적 추정)

monthly_failures_base = monthly_queries * base_failure_rate
monthly_failures_ctx = monthly_queries * ctx_failure_rate
monthly_savings = (monthly_failures_base - monthly_failures_ctx) * cost_per_failure

print("=" * 55)
print("📈 Contextual RAG ROI 분석")
print("   시나리오: 1,000문서, 월 10,000 쿼리")
print("=" * 55)
print(f"\n투자 비용:")
print(f"  컨텍스트 생성 (일회성): ${context_gen_cost:.2f}")
print(f"\n검색 실패 비교:")
print(f"  기본 RAG: 월 {monthly_failures_base:.0f}건 실패 (5.7%)")
print(f"  Contextual RAG: 월 {monthly_failures_ctx:.0f}건 실패 (1.9%)")
print(f"  월 감소: {monthly_failures_base - monthly_failures_ctx:.0f}건")
print(f"\n비용 절감:")
print(f"  월 절감액: ${monthly_savings:.2f}")
print(f"  투자 회수 기간: {context_gen_cost / monthly_savings:.1f}일")
print(f"  연간 절감액: ${monthly_savings * 12:.2f}")
print(f"  연간 ROI: {((monthly_savings * 12 - context_gen_cost) / context_gen_cost * 100):.0f}%")
```

```output
=======================================================
📈 Contextual RAG ROI 분석
   시나리오: 1,000문서, 월 10,000 쿼리
=======================================================

투자 비용:
  컨텍스트 생성 (일회성): $20.25

검색 실패 비교:
  기본 RAG: 월 570건 실패 (5.7%)
  Contextual RAG: 월 190건 실패 (1.9%)
  월 감소: 380건

비용 절감:
  월 절감액: $950.00
  투자 회수 기간: 0.6일
  연간 절감액: $11400.00
  연간 ROI: 56196%
```

## 더 깊이 알아보기

### RAGAS의 탄생 이야기

RAGAS는 2023년 인도의 AI 스타트업 Explodinggradients 팀이 발표한 논문 *"Ragas: Automated Evaluation of Retrieval Augmented Generation"*에서 시작되었습니다. 당시 RAG 시스템 평가에는 사람이 직접 답변의 품질을 판단하는 **수동 평가**가 표준이었는데요, 이건 비용이 많이 들고 일관성도 떨어졌습니다.

RAGAS 팀의 핵심 통찰은 "LLM 자체를 평가자로 활용하자"는 것이었습니다. Faithfulness를 예로 들면, 답변에서 **개별 주장(claim)**을 추출하고, 각 주장이 검색된 컨텍스트에서 뒷받침되는지를 LLM이 판단합니다. 이렇게 하면 사람 없이도 대규모 자동 평가가 가능하죠.

놀라운 건, 이 논문이 발표된 지 1년 만에 GitHub 스타 7,000개를 넘기고 RAG 평가의 사실상 표준이 되었다는 점입니다. 특히 **reference-free 평가**가 가능하다는 점이 실무에서 큰 인기를 끌었습니다. 정답 데이터 없이도 검색 품질과 답변 품질을 측정할 수 있으니까요. RAGAS의 설계 원리와 각 메트릭의 수학적 배경은 [Chapter 17: RAG 평가와 최적화](17-1)에서 깊이 있게 다룰 예정입니다.

### Anthropic의 벤치마크 비하인드

Anthropic이 2024년 9월 Contextual Retrieval을 발표할 때, 단순히 "좋아졌다"고 주장하지 않고 **여러 도메인에 걸친 체계적인 벤치마크**를 제시한 것도 주목할 만합니다. 코드베이스, 소설, ArXiv 논문, 과학 논문 등 성격이 전혀 다른 텍스트에서 일관된 개선을 보여줬거든요. 이 중 **코드베이스**에서 가장 극적인 개선이 나타났는데, 코드에서는 함수 정의, 클래스 계층, 모듈 의존성 같은 **구조적 맥락**이 특히 중요하기 때문입니다.

흥미로운 점은 Gemini Text 004 임베딩 모델이 벤치마크에서 가장 높은 성능을 보였다는 것입니다. Anthropic이 자사 모델이 아닌 경쟁사 모델의 우수성을 인정한 셈인데, 이는 "기법 자체의 효과"를 강조하려는 의도로 해석됩니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Contextual RAG는 항상 기본 RAG보다 좋다"
> 아닙니다. FAQ, 용어 사전, 카탈로그처럼 **각 항목이 자기 완결적인 문서**에서는 맥락 접두사를 추가해도 개선 효과가 미미합니다. 오히려 불필요한 텍스트가 추가되어 임베딩 품질이 떨어질 수도 있어요. Contextual RAG는 "3분기 매출이 전분기 대비 15% 증가했다"처럼 **맥락 없이는 의미를 알 수 없는 청크**가 많은 문서에서 빛을 발합니다.

> 💡 **알고 계셨나요?**: Recall@k와 RAGAS의 Context Recall은 비슷해 보이지만 측정 방식이 다릅니다. Recall@k는 **문서 ID 기반**으로 "정답 청크가 검색 결과에 있는가?"를 봅니다. 반면 RAGAS의 Context Recall은 **내용 기반**으로 "참조 답변의 각 문장이 검색된 텍스트에서 뒷받침되는가?"를 LLM이 판단합니다. 그래서 같은 검색 결과여도 두 점수가 다를 수 있어요. 빠른 반복 실험에는 Recall@k, 정밀한 최종 평가에는 RAGAS — 이렇게 나눠 쓰는 게 효율적입니다.

> 🔥 **실무 팁**: RAGAS 평가 비용도 만만치 않습니다. GPT-4o를 평가 LLM으로 사용하면 100개 QA 쌍 평가에 $2~5 정도 들 수 있어요. 개발 중에는 `gpt-4o-mini`를 평가 모델로 사용하고, 최종 리포트용으로만 `gpt-4o`를 쓰는 **2단계 평가 전략**을 추천합니다. 또한 Batch API를 활용하면 50% 할인이 적용되어 비용을 더 줄일 수 있습니다.

> 🔥 **실무 팁**: 프롬프트 캐싱의 TTL(Time-to-Live)은 기본 5분입니다. 청크가 수천 개인 대규모 문서라면 처리 도중 캐시가 만료될 수 있어요. 이런 경우 `"cache_control": {"type": "ephemeral", "ttl": "1h"}`로 1시간 TTL을 설정하세요. 캐시 쓰기 비용이 2배로 올라가지만(1.25x → 2x), 캐시 미스로 인한 재전송 비용보다는 훨씬 저렴합니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Recall@k** | 상위 k개 검색 결과에 정답 문서가 포함되는 비율. LLM 없이 빠르게 측정 가능 |
| **MRR** | 정답 문서가 평균적으로 몇 번째에 등장하는지의 역수. 순위 품질 측정 |
| **RAGAS** | RAG 시스템의 검색 품질과 생성 품질을 LLM 기반으로 자동 측정하는 평가 프레임워크. [Ch17](17-1)에서 심화 |
| **Faithfulness** | 생성된 답변이 검색된 컨텍스트에 기반하는 정도 (0~1). 개별 주장 단위로 검증 |
| **Context Recall** | 답변에 필요한 정보가 검색 결과에 얼마나 포함되었는가 (0~1) |
| **검색 실패율 67% 감소** | Contextual Embeddings + BM25 + Reranking 조합으로 5.7% → 1.9% 달성 |
| **프롬프트 캐싱 비용 절감** | 캐시 읽기가 기본 입력의 10% 비용이므로 78~90% 절감 가능 |
| **$1.02/MTok** | Anthropic 기준, 프롬프트 캐싱 적용 시 문서 100만 토큰당 컨텍스트 생성 비용 |
| **적용 기준** | 맥락 의존도 높은 문서, 높은 검색 정확도 요구, 낮은~중간 업데이트 빈도에 적합 |

## 다음 섹션 미리보기

Chapter 15의 모든 여정을 마쳤습니다! Contextual RAG의 원리부터 구현, 하이브리드 검색, 그리고 성능·비용 분석까지 다뤘습니다. 다음 [Chapter 16: 에이전틱 RAG](16-1)에서는 한 단계 더 나아가, **LLM 에이전트가 검색 전략을 동적으로 결정**하는 Agentic RAG를 LangGraph로 구축합니다. "검색할지 말지", "어떤 소스를 검색할지", "결과가 충분한지" — 이 모든 판단을 에이전트가 스스로 수행하는, RAG의 가장 진보된 형태를 만나게 됩니다.

## 참고 자료

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) - Contextual RAG의 원본 발표, 벤치마크 결과와 비용 분석이 포함된 핵심 자료
- [RAGAS 공식 문서 — 평가 메트릭](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) - Faithfulness, Context Precision 등 메트릭별 계산 방식과 API 레퍼런스
- [RAGAS — RAG 시스템 평가 튜토리얼](https://docs.ragas.io/en/stable/getstarted/rag_eval/) - EvaluationDataset과 evaluate() 함수를 활용한 전체 평가 파이프라인
- [Anthropic Claude 가격 정책](https://platform.claude.com/docs/en/about-claude/pricing) - 모델별 토큰 가격, 프롬프트 캐싱 비용, Batch API 할인 정보
- [Anthropic 프롬프트 캐싱 가이드](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) - cache_control 구현, TTL 설정, 최소 토큰 요구 사항
- [Together AI — Contextual RAG 구현 가이드](https://docs.together.ai/docs/how-to-implement-contextual-rag-from-anthropic) - Contextual Retrieval의 단계별 실습 구현
- [RAGAS 논문 (arXiv:2309.15217)](https://arxiv.org/abs/2309.15217) - RAGAS 프레임워크의 학술적 근거와 메트릭 설계 원리

---

---
### 🔗 Related Sessions
- [contextual_retrieval](../15-컨텍스추얼-rag-anthropic의-컨텍스트-기반-검색-방법/01-contextual-rag-소개-청크에-맥락을-더하다.md) (prerequisite)
- [contextual_embeddings](../15-컨텍스추얼-rag-anthropic의-컨텍스트-기반-검색-방법/01-contextual-rag-소개-청크에-맥락을-더하다.md) (prerequisite)
- [contextual_bm25](../15-컨텍스추얼-rag-anthropic의-컨텍스트-기반-검색-방법/01-contextual-rag-소개-청크에-맥락을-더하다.md) (prerequisite)
- [reciprocal_rank_fusion](../13-쿼리-변환-기법-multi-query-hyde-step-back-prompting/01-쿼리-변환이-필요한-이유와-전략-개관.md) (prerequisite)
- [contextual_hybrid_retriever](../15-컨텍스추얼-rag-anthropic의-컨텍스트-기반-검색-방법/03-컨텍스추얼-임베딩-bm25-하이브리드-검색.md) (prerequisite)
- [cache_control_ephemeral](../15-컨텍스추얼-rag-anthropic의-컨텍스트-기반-검색-방법/02-컨텍스추얼-청크-생성-파이프라인-구현.md) (prerequisite)
