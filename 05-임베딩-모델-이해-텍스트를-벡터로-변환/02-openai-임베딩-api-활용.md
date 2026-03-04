# OpenAI 임베딩 API 활용

> OpenAI의 text-embedding-3 모델로 텍스트를 벡터로 변환하고, 차원 축소와 배치 처리로 비용을 최적화하는 방법을 배웁니다

## 개요

이 섹션에서는 OpenAI가 제공하는 상용 임베딩 API를 실제로 사용하는 방법을 다룹니다. 앞서 [5.1 임베딩의 기본 개념](ch05/session_01.md)에서 Sentence Transformers로 로컬 임베딩을 생성해봤다면, 이번에는 클라우드 API 기반의 임베딩 생성과 그 최적화 전략을 익힙니다.

**선수 지식**: 임베딩의 기본 개념(벡터 공간, 코사인 유사도), Python 기초, API 호출 경험
**학습 목표**:
- OpenAI의 `text-embedding-3-small`과 `text-embedding-3-large` 모델의 차이를 이해한다
- `dimensions` 파라미터를 활용한 차원 축소(Matryoshka 임베딩)를 적용할 수 있다
- 배치 임베딩 처리로 대량 문서를 효율적으로 임베딩한다
- 용도별 비용 최적화 전략을 수립할 수 있다

## 왜 알아야 할까?

RAG 시스템에서 임베딩은 **모든 검색의 출발점**입니다. 문서를 벡터로 변환하지 않으면 유사도 검색 자체가 불가능하죠. 그런데 실무에서는 수십만 건의 문서를 임베딩해야 하는 경우가 흔합니다. 이때 어떤 모델을 선택하느냐, 차원을 몇으로 설정하느냐에 따라 **비용이 수십 배 차이**날 수 있습니다.

OpenAI의 text-embedding-3 시리즈는 현재 가장 널리 쓰이는 상용 임베딩 모델인데요, 특히 `dimensions` 파라미터로 벡터 크기를 자유롭게 조절할 수 있다는 점이 실무에서 큰 장점입니다. 벡터 크기가 작아지면 저장 비용과 검색 속도 모두 개선되거든요. 이번 세션에서 이 모델들을 제대로 활용하는 법을 익혀두면, 이후 벡터 데이터베이스(6장, 7장)와 RAG 파이프라인(8장) 구축 시 훨씬 효율적인 시스템을 설계할 수 있습니다.

## 핵심 개념

### 개념 1: text-embedding-3 모델 패밀리

> 💡 **비유**: 카메라를 생각해보세요. 스마트폰 카메라는 가볍고 빠르지만 화질이 약간 떨어지고, DSLR 카메라는 무겁고 비싸지만 화질이 뛰어나죠. `text-embedding-3-small`은 스마트폰 카메라처럼 대부분의 상황에서 충분히 좋고, `text-embedding-3-large`는 DSLR처럼 정밀한 의미 구분이 필요할 때 빛을 발합니다.

OpenAI는 2024년 1월 25일, 3세대 임베딩 모델 두 종을 발표했습니다. 이전 모델인 `text-embedding-ada-002`를 대체하는 모델들이죠.

| 항목 | text-embedding-3-small | text-embedding-3-large |
|------|----------------------|----------------------|
| 기본 차원 | 1536 | 3072 |
| 최대 입력 토큰 | 8,192 | 8,192 |
| MTEB 평균 점수 | 62.3% | 64.6% |
| 가격 (1M 토큰) | $0.02 | $0.13 |
| Batch API 가격 | $0.01 | $0.065 |

두 모델 모두 **동일한 최대 입력 길이**(8,192 토큰)를 지원하지만, 성능과 비용에서 뚜렷한 차이가 있습니다. `text-embedding-3-large`는 MTEB 벤치마크에서 약 2.3점 더 높은 성능을 보이지만, 가격은 6.5배나 비쌉니다.

> ⚠️ **흔한 오해**: "large 모델이 항상 더 좋다"고 생각하기 쉽지만, 실제 RAG 검색에서는 두 모델의 정밀도/재현율 차이가 미미한 경우가 많습니다. **95% 이상의 실무 사례에서 `small` 모델이 충분합니다.** 비용 대비 성능을 반드시 먼저 테스트해보세요.

기본적인 임베딩 생성 코드를 살펴보겠습니다. 아래 코드는 `openai>=1.0` 클라이언트 패턴을 사용합니다. 이전 버전의 `openai.Embedding.create()` 문법과는 호환되지 않으므로, 반드시 최신 버전을 설치하세요.

```bash
pip install openai>=1.0
```

```python
from openai import OpenAI

# 클라이언트 초기화 (.env에서 OPENAI_API_KEY를 자동 로드)
# openai>=1.0 패턴: OpenAI() 인스턴스를 생성하여 사용
client = OpenAI()

# 단일 텍스트 임베딩 생성
response = client.embeddings.create(
    model="text-embedding-3-small",  # 모델 선택
    input="RAG는 검색 증강 생성의 약자입니다",  # 임베딩할 텍스트
    encoding_format="float"  # 기본값: float
)

# 임베딩 벡터 추출
embedding = response.data[0].embedding
```

```run:python
# 임베딩 결과 구조 확인 (시뮬레이션)
embedding_dim = 1536  # text-embedding-3-small 기본 차원
sample_values = [-0.0123, 0.0456, -0.0789, 0.0234, -0.0567]

print(f"임베딩 차원: {embedding_dim}")
print(f"벡터 앞부분: {sample_values}")
print(f"벡터 타입: list of float")
```

```output
임베딩 차원: 1536
벡터 앞부분: [-0.0123, 0.0456, -0.0789, 0.0234, -0.0567]
벡터 타입: list of float
```

응답 객체의 `usage` 필드에서 소비된 토큰 수를 확인할 수 있습니다. 임베딩 API는 **입력 토큰만 과금**되고 출력 토큰 비용은 없다는 점이 LLM 호출과 다릅니다.

### 개념 2: Matryoshka 임베딩과 `dimensions` 파라미터

> 💡 **비유**: 러시아 전통 인형 마트료시카를 아시나요? 큰 인형 안에 작은 인형이, 그 안에 더 작은 인형이 들어있죠. text-embedding-3 모델의 벡터도 이와 같습니다. 3072차원 벡터의 앞쪽 256개만 잘라내도 핵심 의미가 보존되는 거예요. 큰 인형을 열면 작은 인형이 이미 안에 들어있는 것처럼, 높은 차원의 임베딩 안에 낮은 차원의 임베딩이 이미 내장되어 있습니다.

이 기능의 정식 이름은 **Matryoshka Representation Learning(MRL)**입니다. 2022년 NeurIPS에서 Aditya Kusupati 등이 발표한 기법으로, 하나의 모델이 다양한 크기의 임베딩을 동시에 학습합니다. OpenAI는 이 기법을 text-embedding-3 시리즈에 적용했습니다.

`dimensions` 파라미터를 사용하면 API가 자동으로 벡터를 잘라주고 **L2 정규화**까지 적용합니다. OpenAI 임베딩은 항상 L2 정규화된 단위 벡터로 반환되는데, L2 정규화가 유사도 계산에 미치는 구체적인 영향은 [5.4 임베딩 품질 평가와 선택 기준](ch05/session_04.md)에서 자세히 다룹니다.

```python
from openai import OpenAI

client = OpenAI()

# 기본 3072차원 임베딩
response_full = client.embeddings.create(
    model="text-embedding-3-large",
    input="임베딩 차원 축소 실험"
)
full_dim = len(response_full.data[0].embedding)  # 3072

# 256차원으로 축소한 임베딩
response_short = client.embeddings.create(
    model="text-embedding-3-large",
    input="임베딩 차원 축소 실험",
    dimensions=256  # 원하는 차원 수 지정
)
short_dim = len(response_short.data[0].embedding)  # 256
```

```run:python
# 차원별 성능-비용 트레이드오프 시뮬레이션
dimensions_options = [256, 512, 1024, 1536, 3072]
storage_per_vec_kb = [d * 4 / 1024 for d in dimensions_options]  # float32 기준

print("차원  | 벡터 크기(KB) | 100만 벡터 저장(GB)")
print("------|-------------|------------------")
for dim, kb in zip(dimensions_options, storage_per_vec_kb):
    gb = kb * 1_000_000 / (1024 * 1024)
    print(f"{dim:>5} | {kb:>11.1f} | {gb:>17.2f}")
```

```output
차원  | 벡터 크기(KB) | 100만 벡터 저장(GB)
------|-------------|------------------
  256 |         1.0 |              0.95
  512 |         2.0 |              1.91
 1024 |         4.0 |              3.81
 1536 |         6.0 |              5.72
 3072 |        12.0 |             11.44
```

놀랍게도 `text-embedding-3-large`를 256차원으로 축소해도, 이전 세대 모델인 `text-embedding-ada-002`의 1536차원보다 MTEB 성능이 더 높습니다. 즉, **벡터 크기를 6분의 1로 줄이면서도 더 좋은 품질**을 얻을 수 있는 셈이죠.

> 🔥 **실무 팁**: 차원 축소의 황금 비율은 용도에 따라 다릅니다. 일반적인 RAG 검색이라면 `text-embedding-3-small`의 기본 1536차원, 또는 `text-embedding-3-large`를 1024차원으로 축소하는 것이 비용 대비 성능이 좋습니다. 수백만 건 이상의 대규모 검색이라면 256~512차원도 고려해볼 만합니다.

### 개념 3: 배치 임베딩 처리

> 💡 **비유**: 택배를 보낼 때 물건 하나씩 따로 보내면 배송비가 많이 들지만, 여러 물건을 한 박스에 모아 보내면 배송비를 아낄 수 있죠. 임베딩 API도 마찬가지입니다. 텍스트를 하나씩 보내는 대신 여러 개를 묶어서 한 번에 보내면 네트워크 오버헤드가 크게 줄어듭니다.

OpenAI 임베딩 API는 한 번의 호출에 **여러 텍스트를 리스트로 전달**할 수 있습니다. 최대 2,048개의 텍스트를 한 번에 처리할 수 있으며, 전체 입력이 8,192 토큰 제한 내에 있어야 합니다(텍스트별이 아닌 요청 전체 기준은 아니고, **각 텍스트가** 8,192 토큰 이하면 됩니다).

```python
from openai import OpenAI

client = OpenAI()

# 여러 텍스트를 한 번에 임베딩
texts = [
    "RAG는 검색 증강 생성입니다",
    "벡터 데이터베이스는 임베딩을 저장합니다",
    "코사인 유사도로 벡터를 비교합니다",
    "청킹은 문서를 작은 단위로 나눕니다",
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts  # 리스트로 전달
)

# 각 텍스트의 임베딩 추출
embeddings = [item.embedding for item in response.data]
```

대량의 문서를 처리할 때는 적절한 배치 크기로 나눠서 호출하되, API 속도 제한(rate limit)에 대비한 에러 처리가 필수적입니다.

```python
import time
import openai
from openai import OpenAI

client = OpenAI()

def batch_embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    dimensions: int | None = None,
    max_retries: int = 5,
) -> list[list[float]]:
    """텍스트 리스트를 배치 단위로 임베딩합니다."""
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        # API 호출 파라미터 구성
        params: dict = {"model": model, "input": batch}
        if dimensions is not None:
            params["dimensions"] = dimensions

        # 지수 백오프를 적용한 재시도 로직
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(**params)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break
            except openai.RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait_time = 2 ** attempt  # 1, 2, 4, 8, 16초
                print(f"Rate limit 도달. {wait_time}초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        # 배치 간 짧은 대기로 rate limit 예방
        if i + batch_size < len(texts):
            time.sleep(0.1)

    return all_embeddings

# 사용 예시
documents = ["문서 1 내용...", "문서 2 내용...", "..."]  # 수천 건의 문서
embeddings = batch_embed(documents, dimensions=512)
```

> 🔥 **실무 팁**: OpenAI의 **Batch API**를 사용하면 실시간 처리가 필요 없는 경우 **50% 할인**을 받을 수 있습니다. 24시간 내에 결과가 반환되며, 초기 문서 인덱싱처럼 시간 제약이 덜한 작업에 적합합니다. `text-embedding-3-small` 기준으로 1M 토큰당 $0.02 → $0.01로 절반이 됩니다.

### 개념 4: 토큰 카운팅과 비용 추정

임베딩 비용을 예측하려면 입력 텍스트의 토큰 수를 미리 알아야 합니다. OpenAI의 `tiktoken` 라이브러리로 정확한 토큰 수를 계산할 수 있습니다.

```python
import tiktoken

# text-embedding-3 모델은 cl100k_base 인코딩 사용
enc = tiktoken.get_encoding("cl100k_base")

text = "RAG 시스템에서 임베딩은 검색의 핵심입니다."
tokens = enc.encode(text)
token_count = len(tokens)
```

```run:python
# 비용 추정 시뮬레이션
prices = {
    "text-embedding-3-small": 0.02,        # $/1M tokens
    "text-embedding-3-small (batch)": 0.01, # $/1M tokens
    "text-embedding-3-large": 0.13,         # $/1M tokens
    "text-embedding-3-large (batch)": 0.065, # $/1M tokens
}

# 가정: 10,000건 문서, 문서당 평균 500토큰
doc_count = 10_000
avg_tokens = 500
total_tokens = doc_count * avg_tokens

print(f"총 문서: {doc_count:,}건")
print(f"총 토큰: {total_tokens:,} ({total_tokens/1_000_000:.1f}M)")
print()
print("모델                        | 예상 비용")
print("----------------------------|----------")
for model, price in prices.items():
    cost = (total_tokens / 1_000_000) * price
    print(f"{model:<28}| ${cost:.4f}")
```

```output
총 문서: 10,000건
총 토큰: 5,000,000 (5.0M)

모델                        | 예상 비용
----------------------------|----------
text-embedding-3-small      | $0.1000
text-embedding-3-small (batch)| $0.0500
text-embedding-3-large      | $0.6500
text-embedding-3-large (batch)| $0.3250
```

10,000건 문서 기준으로 `small` 모델은 약 $0.10, `large` 모델은 $0.65입니다. 문서가 100만 건이면 이 차이가 $10 vs $65로 벌어지고, 반복 실험까지 고려하면 모델 선택이 비용에 큰 영향을 미칩니다.

## 실습: 직접 해보기

이제 OpenAI 임베딩 API의 핵심 기능을 종합적으로 실습해봅시다. 여러 한국어 문서를 임베딩하고, 차원 축소가 유사도 검색에 미치는 영향을 비교합니다.

```python
"""OpenAI 임베딩 API 실습: 차원별 유사도 비교

요구 사항: pip install openai>=1.0 numpy
"""

import numpy as np
from openai import OpenAI

# 클라이언트 초기화
client = OpenAI()  # OPENAI_API_KEY 환경 변수 필요

# 1. 테스트 문서 준비
documents = [
    "RAG는 대규모 언어 모델의 환각 문제를 줄이기 위해 외부 지식을 검색하여 답변에 활용하는 기법입니다.",
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 유사도 기반 검색을 수행합니다.",
    "텍스트 청킹은 긴 문서를 적절한 크기의 조각으로 나누어 검색 품질을 높이는 과정입니다.",
    "파이썬은 데이터 과학과 머신러닝에서 가장 널리 사용되는 프로그래밍 언어입니다.",
    "코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지 측정하며, RAG 검색에서 핵심 메트릭입니다.",
]

query = "RAG 시스템에서 문서를 어떻게 검색하나요?"

# 2. 다양한 차원으로 임베딩 생성
dimensions_to_test = [256, 512, 1536]  # small 모델 기본은 1536

results = {}
for dim in dimensions_to_test:
    # 문서 임베딩
    doc_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=documents,
        dimensions=dim,
    )
    doc_embeddings = np.array([d.embedding for d in doc_response.data])

    # 쿼리 임베딩
    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query,
        dimensions=dim,
    )
    query_embedding = np.array(query_response.data[0].embedding)

    # 코사인 유사도 계산
    # OpenAI 임베딩은 L2 정규화(단위 벡터)되어 반환되므로 내적 = 코사인 유사도
    similarities = doc_embeddings @ query_embedding
    results[dim] = similarities

# 3. 결과 비교 출력
print(f"쿼리: \"{query}\"\n")
print(f"{'차원':<8}", end="")
for i, doc in enumerate(documents):
    print(f"| 문서{i+1:<3}", end="")
print("\n" + "-" * 52)

for dim in dimensions_to_test:
    print(f"{dim:<8}", end="")
    for sim in results[dim]:
        print(f"| {sim:.3f} ", end="")
    print()

print(f"\n--- 1536차원 기준 유사도 순위 ---")
ranking = np.argsort(results[1536])[::-1]
for rank, idx in enumerate(ranking, 1):
    print(f"{rank}위: 문서{idx+1} ({results[1536][idx]:.3f}) - {documents[idx][:30]}...")
```

실행하면 세 가지 차원에서의 유사도가 출력되는데요, 대부분의 경우 **순위(ranking)는 동일하게 유지**됩니다. 차원을 줄여도 "어떤 문서가 가장 관련 있는가"의 판단은 크게 달라지지 않는 것이 Matryoshka 임베딩의 핵심 장점입니다.

> 💡 **알고 계셨나요?**: OpenAI 임베딩은 L2 정규화된 단위 벡터로 반환됩니다. 따라서 **코사인 유사도를 벡터 내적(dot product)으로 간단히 계산**할 수 있습니다. 별도의 정규화 과정이 필요 없어 계산이 빠르죠. L2 정규화가 유사도 메트릭 선택에 미치는 영향(코사인 유사도 vs 유클리드 거리 vs 내적)은 [5.4 임베딩 품질 평가와 선택 기준](ch05/session_04.md)에서 자세히 비교합니다.

## 더 깊이 알아보기

### Matryoshka Representation Learning의 탄생

"Matryoshka"라는 이름은 러시아 전통 인형 **마트료시카(матрёшка)**에서 왔습니다. 큰 인형 안에 점점 작은 인형이 들어있는 것처럼, 고차원 임베딩 안에 저차원 임베딩이 내장되어 있다는 뜻이죠.

이 기법은 2022년 NeurIPS에서 워싱턴 대학교의 **Aditya Kusupati**와 공동 연구진이 발표했습니다. 기존에는 임베딩 차원을 바꾸려면 모델을 새로 학습해야 했는데, MRL은 학습 과정에서 **여러 차원의 손실 함수를 동시에 최적화**하는 방식을 제안했습니다. 예를 들어 256차원, 512차원, 1024차원의 성능을 동시에 고려하며 학습하는 것이죠. 결과적으로 하나의 모델이 다양한 차원에서 모두 좋은 성능을 내게 됩니다.

이 연구가 나오자마자 실무적 파급력이 엄청났습니다. 검색 시스템에서는 처음에 낮은 차원으로 빠르게 후보를 추리고, 이후 높은 차원으로 정밀 재검색하는 **Adaptive Retrieval** 전략이 가능해졌거든요. OpenAI가 이 기법을 2024년 1월 text-embedding-3 시리즈에 도입하면서, `dimensions` 파라미터 하나로 누구나 쉽게 차원을 조절할 수 있게 되었습니다.

### ada에서 embedding-3으로: OpenAI 임베딩의 진화

OpenAI의 첫 대중적 임베딩 모델은 2022년 12월에 출시된 `text-embedding-ada-002`였습니다. "ada"라는 이름은 세계 최초의 프로그래머로 불리는 **에이다 러브레이스(Ada Lovelace)**에서 따온 것인데요. 당시 1536차원 고정이었고 MTEB 점수 61.0%를 기록했습니다.

약 1년 뒤인 2024년 1월 25일, OpenAI는 text-embedding-3 시리즈를 발표하면서 두 가지 큰 변화를 가져왔습니다. 첫째, Matryoshka 학습으로 차원을 유연하게 조절할 수 있게 되었고, 둘째, `small` 모델의 가격을 ada 대비 **5분의 1**로 낮추면서 성능은 오히려 높였습니다. 이 발표는 임베딩 모델의 가격 경쟁을 촉발시켜, 이후 여러 회사가 더 저렴하고 강력한 임베딩 모델을 내놓는 계기가 되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "임베딩 차원이 높을수록 무조건 좋다"고 생각하기 쉽지만, 차원이 높아지면 **저장 비용 증가, 검색 속도 저하, 차원의 저주(curse of dimensionality)** 문제가 발생합니다. 특히 데이터가 적은 경우 고차원 임베딩은 오히려 성능이 떨어질 수 있습니다. `dimensions` 파라미터로 본인의 데이터셋에 최적인 차원을 실험해보세요.

> 💡 **알고 계셨나요?**: OpenAI 임베딩 API에서 `dimensions` 파라미터로 차원을 줄이면, 내부적으로 벡터의 **앞부분만 잘라낸 뒤 L2 정규화**를 다시 적용합니다. 직접 벡터를 자르면 정규화가 깨지므로, 반드시 API 파라미터를 사용하거나 직접 잘랐다면 수동으로 L2 정규화를 해줘야 합니다.

> 🔥 **실무 팁**: 임베딩 비용을 더 줄이고 싶다면, **캐싱 전략**을 도입하세요. 동일한 텍스트를 반복 임베딩하는 것은 낭비입니다. 해시 기반 캐시로 이미 임베딩한 텍스트를 건너뛰면, 문서가 자주 업데이트되는 시스템에서 비용을 크게 절약할 수 있습니다. `tiktoken`으로 미리 토큰 수를 세서 예산을 확인하는 습관도 중요합니다.

> 🔥 **실무 팁**: 검색(query)용 임베딩과 저장(document)용 임베딩에 **같은 모델과 같은 차원**을 사용해야 합니다. 문서는 `small`로, 쿼리는 `large`로 임베딩하면 벡터 공간이 달라져 유사도 계산이 의미 없어집니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| text-embedding-3-small | 1536차원 기본, $0.02/1M 토큰. 대부분의 RAG에 충분한 가성비 모델 |
| text-embedding-3-large | 3072차원 기본, $0.13/1M 토큰. 정밀한 의미 구분이 필요한 경우 |
| `dimensions` 파라미터 | Matryoshka 학습 기반, 벡터 차원을 자유롭게 축소 가능 |
| Matryoshka Representation Learning | 하나의 모델에서 다양한 차원의 임베딩을 동시 학습하는 기법 (NeurIPS 2022) |
| 배치 임베딩 | 여러 텍스트를 리스트로 전달하여 한 번에 처리, 네트워크 오버헤드 절감 |
| Batch API | 비실시간 작업에 50% 할인 적용, 24시간 내 결과 반환 |
| `tiktoken` | 토큰 수 사전 계산으로 비용 예측 (cl100k_base 인코딩) |
| L2 정규화 | OpenAI 임베딩은 단위 벡터로 반환, 내적 = 코사인 유사도. 상세 비교는 5.4 참조 |
| `openai>=1.0` | 최신 클라이언트 패턴 (`OpenAI()` 인스턴스 기반) 사용 필수 |

## 다음 섹션 미리보기

이번 세션에서 OpenAI의 상용 임베딩 API를 다뤘는데요, 비용이 발생하는 API 대신 **무료로 로컬에서 실행**할 수 있는 방법은 없을까요? 다음 세션 [5.3 오픈소스 임베딩 모델 — Sentence Transformers 심화](ch05/session_03.md)에서는 Sentence Transformers의 다양한 모델을 깊이 있게 비교하고, 한국어 특화 임베딩 모델과 파인튜닝 방법까지 살펴봅니다. OpenAI API와 오픈소스 모델의 장단점을 비교하여 상황에 맞는 최적의 선택을 할 수 있게 됩니다.

## 참고 자료

- [OpenAI Embeddings Guide — 공식 API 문서](https://developers.openai.com/api/docs/guides/embeddings/) - text-embedding-3 모델의 상세 스펙, 코드 예제, 사용법을 담은 공식 가이드
- [New Embedding Models and API Updates — OpenAI 블로그](https://openai.com/index/new-embedding-models-and-api-updates/) - 2024년 1월 text-embedding-3 시리즈 발표 공식 블로그 포스트
- [Matryoshka Representation Learning — NeurIPS 2022 논문](https://arxiv.org/abs/2205.13147) - dimensions 파라미터의 이론적 기반이 된 원본 논문 (Kusupati et al.)
- [OpenAI's Text Embeddings v3 — Pinecone 분석](https://www.pinecone.io/learn/openai-embeddings-v3/) - text-embedding-3 모델의 벤치마크 비교와 실전 활용 가이드
- [Introduction to Matryoshka Embedding Models — Hugging Face 블로그](https://huggingface.co/blog/matryoshka) - Matryoshka 학습의 원리를 시각적으로 설명한 튜토리얼
- [OpenAI API Cost Optimization 가이드](https://developers.openai.com/api/docs/guides/cost-optimization/) - 배치 처리, 토큰 관리 등 비용 절감 전략 공식 문서

---
### 🔗 Related Sessions
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_space_semantics](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [sentence_embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
