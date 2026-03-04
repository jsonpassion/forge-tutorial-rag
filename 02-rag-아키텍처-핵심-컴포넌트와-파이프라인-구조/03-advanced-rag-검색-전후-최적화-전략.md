# Advanced RAG — 검색 전/후 최적화 전략

> Naive RAG의 한계를 극복하는 Pre-Retrieval과 Post-Retrieval 최적화 기법을 체계적으로 학습합니다.

## 개요

이 섹션에서는 Advanced RAG 패러다임의 핵심인 **검색 전(Pre-Retrieval) 최적화**와 **검색 후(Post-Retrieval) 최적화** 전략을 깊이 있게 다룹니다. [앞서 Session 2.2](02-naive-rag-기본-패턴과-한계.md)에서 살펴본 Naive RAG의 다섯 가지 한계—낮은 정밀도, 낮은 재현율, Lost in the Middle, 컨텍스트 통합 부족, 쿼리-문서 불일치—각각에 대응하는 구체적인 해결책을 배웁니다.

**선수 지식**: Naive RAG의 인덱싱-검색-생성 구조(Session 2.1), Naive RAG의 한계점(Session 2.2)에서 다룬 `stuff_method`, `low_precision`, `low_recall`, `query_document_mismatch` 개념
**학습 목표**:
- Pre-Retrieval 최적화(쿼리 재작성, 쿼리 확장, HyDE)의 원리와 적용 시점을 이해한다
- Post-Retrieval 최적화(리랭킹, 컨텍스트 압축)의 동작 방식을 설명할 수 있다
- LangChain의 MultiQueryRetriever, ContextualCompressionRetriever를 코드로 구현한다
- Advanced RAG가 Naive RAG의 각 한계를 어떻게 개선하는지 매핑할 수 있다

## 왜 알아야 할까?

Naive RAG는 "질문 → 검색 → 답변"의 단순 직선 구조입니다. 하지만 현실에서는 사용자의 질문이 모호하고, 검색 결과에 노이즈가 많으며, 정작 중요한 문서가 누락되는 일이 빈번하죠. 실제 프로덕션 RAG 시스템에서 가장 흔한 불만은 **"검색은 되는데 답변이 엉뚱하다"**입니다.

Advanced RAG는 이 문제를 검색 **앞뒤에 최적화 레이어**를 추가하여 해결합니다. 마치 도서관에서 책을 찾을 때, 단순히 제목만 보고 고르는 게 아니라(Naive RAG), 사서에게 먼저 "이런 주제의 책을 찾고 있어요"라고 상담하고(Pre-Retrieval), 가져온 책들 중에서 가장 관련 있는 부분만 골라 읽는 것(Post-Retrieval)과 같습니다.

구글, 마이크로소프트, Anthropic 등 주요 기업의 프로덕션 RAG 시스템은 거의 모두 Advanced RAG 패턴을 채택하고 있습니다. 이 패러다임을 이해하는 것이 실무에서 RAG 품질을 높이는 첫 번째 열쇠입니다.

## 핵심 개념

### 개념 1: Advanced RAG의 전체 구조

> 💡 **비유**: Naive RAG가 "마트에서 장 보기"라면, Advanced RAG는 **"전문 셰프의 재료 구매"**입니다. 셰프는 장을 보기 전에 레시피를 확인하고 필요한 재료 목록을 정리하며(Pre-Retrieval), 마트에서 가져온 재료 중 신선도와 품질을 꼼꼼히 검수합니다(Post-Retrieval). 단순히 "야채 코너 가서 아무거나 집어오기"와는 차원이 다르죠.

Advanced RAG는 Naive RAG의 단순 파이프라인에 두 가지 최적화 단계를 추가합니다:

```
[사용자 쿼리]
    ↓
┌─────────────────────────┐
│  Pre-Retrieval 최적화    │  ← 쿼리 재작성, 쿼리 확장, HyDE
│  (검색 전 쿼리 개선)      │
└─────────────────────────┘
    ↓ (개선된 쿼리)
┌─────────────────────────┐
│  Retrieval (검색)        │  ← 벡터 유사도 검색 (Naive RAG와 동일)
└─────────────────────────┘
    ↓ (검색된 문서들)
┌─────────────────────────┐
│  Post-Retrieval 최적화   │  ← 리랭킹, 컨텍스트 압축, 필터링
│  (검색 후 결과 정제)      │
└─────────────────────────┘
    ↓ (정제된 컨텍스트)
[LLM 응답 생성]
```

Gao et al.의 서베이 논문 *"Retrieval-Augmented Generation for Large Language Models: A Survey"*에서는 RAG의 발전을 Naive RAG → Advanced RAG → Modular RAG의 세 가지 패러다임으로 정리했는데요, Advanced RAG의 핵심 차별점이 바로 이 **Pre-Retrieval과 Post-Retrieval 최적화**입니다.

| Naive RAG 한계 | Advanced RAG 해결책 | 최적화 단계 |
|---|---|---|
| 낮은 재현율(`low_recall`) | 쿼리 확장(Multi-Query) | Pre-Retrieval |
| 쿼리-문서 불일치(`query_document_mismatch`) | 쿼리 재작성, HyDE | Pre-Retrieval |
| 낮은 정밀도(`low_precision`) | 리랭킹(Reranking) | Post-Retrieval |
| Lost in the Middle | 컨텍스트 압축 | Post-Retrieval |
| 컨텍스트 통합 부족 | 리랭킹 + 압축 조합 | Post-Retrieval |

---

### 개념 2: Pre-Retrieval 최적화 — 검색 전에 쿼리를 다듬다

> 💡 **비유**: 외국 도서관에서 책을 찾는다고 상상해보세요. "한국 요리"라고 검색하면 결과가 적을 수 있지만, 사서가 "Korean cuisine", "Korean food recipes", "Asian cooking" 등으로 **검색어를 다양하게 바꿔서** 찾아주면 훨씬 많은 관련 도서를 찾을 수 있겠죠? Pre-Retrieval 최적화가 바로 이 "똑똑한 사서" 역할입니다.

Pre-Retrieval 최적화는 **사용자의 원본 쿼리를 검색에 더 적합한 형태로 변환**하는 기법들입니다. 세 가지 주요 기법을 살펴보겠습니다.

#### 2-1. 쿼리 재작성(Query Rewriting)

사용자의 모호하거나 구어적인 질문을 검색에 최적화된 형태로 **다시 작성**합니다.

```
원본 쿼리: "RAG 느린 거 어떻게 해?"
↓ (쿼리 재작성)
개선된 쿼리: "RAG 시스템의 검색 속도를 최적화하는 방법과 지연 시간 감소 전략"
```

LLM이 쿼리를 재작성하면, 검색 시 **임베딩 유사도가 높아지면서** 더 관련성 높은 문서를 찾아올 수 있습니다.

#### 2-2. 쿼리 확장(Query Expansion) — Multi-Query

하나의 질문을 **여러 관점의 질문으로 확장**하고, 각각 검색한 뒤 결과를 합칩니다. LangChain의 `MultiQueryRetriever`가 대표적 구현입니다.

```
원본 쿼리: "벡터 데이터베이스의 장점은?"
↓ (Multi-Query 확장)
쿼리 1: "벡터 데이터베이스가 전통적 데이터베이스보다 우수한 점"
쿼리 2: "벡터 DB를 사용해야 하는 이유와 주요 활용 사례"
쿼리 3: "벡터 검색 시스템의 성능상 이점과 비용 효율성"
```

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# LLM이 쿼리를 여러 관점으로 확장
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# base_retriever는 기존 벡터스토어의 retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# 내부적으로 3개의 쿼리를 생성하고, 각각 검색 후 결과를 합침
docs = retriever.invoke("벡터 데이터베이스의 장점은?")
```

Multi-Query의 핵심은 **재현율(Recall) 향상**입니다. 단일 쿼리로는 놓칠 수 있는 관련 문서를 여러 관점의 쿼리가 보완적으로 찾아줍니다.

#### 2-3. HyDE(Hypothetical Document Embeddings)

HyDE는 매우 독창적인 접근법인데요, LLM에게 **"가설 답변 문서"를 먼저 생성**시킨 뒤, 그 가설 문서의 임베딩으로 실제 문서를 검색합니다.

```
사용자 쿼리: "트랜스포머의 어텐션 메커니즘이란?"
↓ (LLM이 가설 답변 생성)
가설 문서: "트랜스포머의 어텐션 메커니즘은 입력 시퀀스의 
각 위치가 다른 모든 위치에 대해 가중치를 계산하여..."
↓ (가설 문서를 임베딩)
가설 임베딩으로 벡터 DB 검색
↓
실제 관련 문서 반환
```

왜 이게 효과적일까요? **질문과 답변은 문체가 다릅니다.** "트랜스포머의 어텐션이란?"이라는 질문 형태의 임베딩보다, "어텐션 메커니즘은 입력 시퀀스의..."라는 답변 형태의 임베딩이 실제 문서와 **의미적으로 더 가깝기** 때문입니다. 이것이 바로 Naive RAG에서 다뤘던 `query_document_mismatch` 문제의 해결책이죠.

> ⚠️ **흔한 오해**: "HyDE는 LLM이 만든 가짜 답변으로 검색하니까 부정확하지 않나요?" — HyDE의 가설 문서는 **검색용 임베딩을 생성하기 위한 것**이지, 최종 답변이 아닙니다. 가설 문서가 100% 정확하지 않아도, "답변 스타일"의 임베딩이 실제 문서와 더 유사한 벡터 공간에 위치하기 때문에 검색 성능이 향상됩니다.

> 📌 **HyDE 구현 안내**: 이 세션에서는 HyDE의 개념과 원리에 집중합니다. LangChain을 활용한 HyDE의 전체 구현 — 가설 문서 생성 프롬프트 설계, 임베딩 체인 구성, 일반 검색 대비 성능 비교 — 은 [Ch13 쿼리 변환 기법](../13-쿼리-변환-기법-multi-query-hyde-step-back-prompting/03-hyde-가설-문서-임베딩.md)에서 코드와 함께 상세히 다룹니다.

---

### 개념 3: Post-Retrieval 최적화 — 검색 후에 결과를 정제하다

> 💡 **비유**: 서점에서 "파이썬 프로그래밍" 관련 책 10권을 가져왔다고 합시다. 이 중에는 초급 입문서도 있고, 데이터 사이언스 책도 있고, 웹 개발 책도 있겠죠. Post-Retrieval 최적화는 **북 큐레이터**처럼, 당신의 실제 목적(예: "FastAPI로 웹 서버 만들기")에 가장 맞는 책을 골라주고(리랭킹), 그 책에서도 해당 챕터만 발췌해주는(컨텍스트 압축) 역할을 합니다.

검색된 문서들을 LLM에 전달하기 전에 **품질을 높이는** 단계입니다.

#### 3-1. 리랭킹(Reranking)

초기 검색에서 가져온 문서들의 관련성 점수를 **Cross-Encoder 모델로 다시 계산**하여 순위를 재정렬합니다.

임베딩 기반 검색(Bi-Encoder)은 빠르지만 정확도에 한계가 있습니다. 쿼리와 문서를 각각 독립적으로 임베딩하기 때문이죠. 반면 Cross-Encoder는 쿼리와 문서를 **함께 입력받아** 관련성을 직접 판단합니다.

```
Bi-Encoder (초기 검색):        Cross-Encoder (리랭킹):
┌──────┐  ┌──────┐            ┌─────────────────┐
│쿼리   │  │문서   │            │ 쿼리 + 문서 함께  │
│임베딩  │  │임베딩  │            │ 입력하여 분석     │
└──┬───┘  └──┬───┘            └────────┬────────┘
   │코사인     │                        │관련성 점수
   │유사도     │                        │(0~1)
   └────┬─────┘                        │
       점수                           점수
```

Bi-Encoder는 수백만 문서를 빠르게 훑을 수 있지만, Cross-Encoder는 문서 하나당 계산이 무거워서 소수의 후보에만 적용합니다. 그래서 **Bi-Encoder로 넓게 후보를 검색 → Cross-Encoder로 정밀하게 재정렬**하는 2단계 전략이 표준 패턴입니다.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# Cohere Rerank 모델로 리랭커 생성
compressor = CohereRerank(
    model="rerank-v3.5",  # 최신 리랭킹 모델
    top_n=3,              # 상위 3개만 선택
)

# 기존 retriever를 리랭킹으로 감싸기
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)

# k=10으로 넓게 검색 → Rerank로 top_n=3 정밀 선택
docs = reranking_retriever.invoke("RAG 시스템의 평가 방법")
```

위 코드에서 주목할 점은 `k=10`과 `top_n=3`의 조합입니다. 넓게 10개를 가져온 뒤 리랭킹으로 3개만 추리는 거죠. Naive RAG에서 `k=3`으로 검색하면 애초에 관련 문서가 누락될 수 있지만, 이 전략은 **재현율과 정밀도를 동시에** 개선합니다.

#### 3-2. 컨텍스트 압축(Contextual Compression)

검색된 문서에서 **쿼리와 관련된 부분만 추출**하거나, 관련 없는 문서를 **통째로 제거**합니다.

LangChain은 두 가지 압축 방식을 제공합니다:

**LLMChainExtractor** — LLM이 문서에서 관련 내용만 추출:

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

# LLM이 각 문서에서 쿼리 관련 부분만 추출
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(),
)
```

**EmbeddingsFilter** — 임베딩 유사도 기반으로 저관련 문서 필터링 (LLM 호출 없이 빠르게):

```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_openai import OpenAIEmbeddings

# 유사도 임계값 이하 문서 자동 제거
embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76,  # 0.76 미만 유사도 문서는 제거
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
)
```

컨텍스트 압축은 특히 **Lost in the Middle 문제**를 직접적으로 해결합니다. 불필요한 문서를 제거하면, LLM이 처리해야 할 컨텍스트가 줄어들어 중간에 묻히는 정보 없이 핵심 내용에 집중할 수 있거든요.

> 🔥 **실무 팁**: `LLMChainExtractor`는 문서마다 LLM을 호출하므로 비용과 지연이 발생합니다. 프로덕션 환경에서는 `EmbeddingsFilter`로 1차 필터링 후 리랭킹을 적용하는 **파이프라인 조합**이 비용 대비 효과가 좋습니다.

---

### 개념 4: 최적화 기법 조합하기 — DocumentCompressorPipeline

> 💡 **비유**: 정수기를 떠올려보세요. 물이 한 번의 필터만 거치는 게 아니라, **침전 → 활성탄 → 역삼투 → UV 살균** 여러 단계를 순차적으로 통과하면서 점점 깨끗해지죠. Advanced RAG도 마찬가지로 여러 최적화 기법을 **파이프라인으로 조합**하면 시너지가 납니다.

LangChain의 `DocumentCompressorPipeline`은 여러 압축/필터링 기법을 순차적으로 적용할 수 있게 해줍니다:

```python
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.text_splitter import CharacterTextSplitter

# 파이프라인 구성: 분할 → 중복 제거 → 유사도 필터
pipeline = DocumentCompressorPipeline(
    transformers=[
        CharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            separator=". ",
        ),
        EmbeddingsRedundantFilter(embeddings=embeddings),  # 중복 문서 제거
        EmbeddingsFilter(
            embeddings=embeddings,
            similarity_threshold=0.76,
        ),
    ]
)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

이 파이프라인은 다음 순서로 동작합니다:
1. **CharacterTextSplitter**: 검색된 문서를 더 작은 단위로 재분할
2. **EmbeddingsRedundantFilter**: 의미적으로 중복되는 청크 제거
3. **EmbeddingsFilter**: 쿼리와 유사도가 낮은 청크 제거

결과적으로 LLM에 전달되는 컨텍스트는 **관련성 높고, 중복 없는, 핵심 정보만** 담게 됩니다.

## 실습: 직접 해보기

이제 Naive RAG와 Advanced RAG를 나란히 구축하고, 검색 품질 차이를 눈으로 확인해보겠습니다. 전체 코드를 순서대로 실행하면 됩니다.

### 환경 설정

```python
# 필수 패키지 설치
# pip install langchain langchain-openai langchain-community langchain-cohere chromadb

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 API 키 로드
# 필요한 환경 변수: OPENAI_API_KEY, COHERE_API_KEY
```

### 1단계: 샘플 문서와 벡터스토어 준비

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# RAG 관련 샘플 문서 생성
documents = [
    Document(
        page_content="RAG 시스템의 검색 품질을 높이려면 쿼리 재작성 기법을 활용할 수 있다. "
        "사용자의 원본 질문을 LLM으로 변환하여 검색에 최적화된 형태로 만든다.",
        metadata={"source": "rag_optimization.pdf", "page": 1},
    ),
    Document(
        page_content="리랭킹(Reranking)은 초기 검색 결과를 Cross-Encoder 모델로 재평가하는 기법이다. "
        "Bi-Encoder보다 정확하지만 계산 비용이 높아 상위 N개 문서에만 적용한다.",
        metadata={"source": "rag_optimization.pdf", "page": 5},
    ),
    Document(
        page_content="벡터 데이터베이스는 임베딩을 저장하고 유사도 검색을 수행하는 특수 DB이다. "
        "ChromaDB, FAISS, Pinecone 등이 대표적이다.",
        metadata={"source": "vector_db_guide.pdf", "page": 1},
    ),
    Document(
        page_content="HyDE는 Hypothetical Document Embeddings의 약자로, LLM이 가설 답변을 "
        "생성하고 해당 임베딩으로 검색하는 기법이다. 쿼리-문서 간 의미 격차를 줄인다.",
        metadata={"source": "rag_optimization.pdf", "page": 8},
    ),
    Document(
        page_content="컨텍스트 압축은 검색된 문서에서 쿼리와 관련 있는 부분만 추출하는 기법이다. "
        "LLM 기반 추출과 임베딩 기반 필터링 두 가지 방식이 있다.",
        metadata={"source": "rag_optimization.pdf", "page": 12},
    ),
    Document(
        page_content="프롬프트 엔지니어링은 LLM에 입력하는 프롬프트를 최적화하는 기법이다. "
        "Few-shot, Chain-of-Thought 등 다양한 전략이 있다.",
        metadata={"source": "llm_basics.pdf", "page": 3},
    ),
    Document(
        page_content="RAG 파이프라인의 성능 평가에는 Faithfulness, Answer Relevancy, "
        "Context Precision, Context Recall 등의 메트릭을 사용한다.",
        metadata={"source": "rag_evaluation.pdf", "page": 1},
    ),
    Document(
        page_content="Advanced RAG는 Naive RAG에 Pre-Retrieval과 Post-Retrieval 최적화를 "
        "추가한 패러다임이다. 쿼리 확장, 리랭킹, 컨텍스트 압축 등을 포함한다.",
        metadata={"source": "rag_survey.pdf", "page": 15},
    ),
]

# 벡터스토어 생성
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents, embeddings)

print(f"벡터스토어 생성 완료: {vectorstore._collection.count()}개 문서")
```

### 2단계: Naive RAG vs Advanced RAG 비교

```run:python
# === Naive RAG: 단순 유사도 검색 ===
naive_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
naive_results = naive_retriever.invoke("RAG 검색 성능을 개선하는 방법들은?")

print("=" * 60)
print("🔍 Naive RAG 검색 결과 (상위 3개)")
print("=" * 60)
for i, doc in enumerate(naive_results, 1):
    print(f"\n[{i}] ({doc.metadata['source']}, p.{doc.metadata['page']})")
    print(f"    {doc.page_content[:80]}...")
```

```output
============================================================
🔍 Naive RAG 검색 결과 (상위 3개)
============================================================

[1] (rag_optimization.pdf, p.1)
    RAG 시스템의 검색 품질을 높이려면 쿼리 재작성 기법을 활용할 수 있다. 사용자의 원본 질문을 LLM으로 변환하여 검...

[2] (rag_survey.pdf, p.15)
    Advanced RAG는 Naive RAG에 Pre-Retrieval과 Post-Retrieval 최적화를 추가한 패러다임이다. 쿼리 확장, 리랭킹...

[3] (rag_evaluation.pdf, p.1)
    RAG 파이프라인의 성능 평가에는 Faithfulness, Answer Relevancy, Context Precision, Context...
```

```run:python
# === Advanced RAG: Multi-Query + Reranking ===
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# Step 1: Multi-Query로 쿼리 확장 (Pre-Retrieval)
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# Step 2: Reranking으로 결과 정제 (Post-Retrieval)
compressor = CohereRerank(model="rerank-v3.5", top_n=3)
advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever,
)

advanced_results = advanced_retriever.invoke("RAG 검색 성능을 개선하는 방법들은?")

print("=" * 60)
print("🚀 Advanced RAG 검색 결과 (Multi-Query + Reranking)")
print("=" * 60)
for i, doc in enumerate(advanced_results, 1):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"\n[{i}] 관련성 점수: {score:.4f}")
    print(f"    ({doc.metadata['source']}, p.{doc.metadata['page']})")
    print(f"    {doc.page_content[:80]}...")
```

```output
============================================================
🚀 Advanced RAG 검색 결과 (Multi-Query + Reranking)
============================================================

[1] 관련성 점수: 0.9823
    Advanced RAG는 Naive RAG에 Pre-Retrieval과 Post-Retrieval 최적화를 추가한 패러다임이다. 쿼리 확장, 리랭킹...

[2] 관련성 점수: 0.9541
    RAG 시스템의 검색 품질을 높이려면 쿼리 재작성 기법을 활용할 수 있다. 사용자의 원본 질문을 LLM으로 변환하여 검...

[3] 관련성 점수: 0.8876
    리랭킹(Reranking)은 초기 검색 결과를 Cross-Encoder 모델로 재평가하는 기법이다. Bi-Encoder보다 정확하지만 계...
```

Advanced RAG의 결과를 보면, Naive RAG에서는 3위였던 평가 메트릭 문서(질문과 직접 관련이 적은) 대신, **리랭킹에 대한 문서가 3위로 올라왔습니다.** Reranking이 쿼리의 의도("검색 성능 개선")에 더 부합하는 문서를 정확하게 선별한 것이죠.

### 3단계: 컨텍스트 압축 적용

```run:python
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 임베딩 유사도 기반 필터링 (LLM 호출 없이 빠르게)
embeddings_filter = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.80,  # 80% 이상 유사도만 통과
)

filtered_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
)

filtered_results = filtered_retriever.invoke("RAG 검색 성능을 개선하는 방법들은?")

print("=" * 60)
print("🎯 컨텍스트 압축 결과 (EmbeddingsFilter)")
print(f"   8개 검색 → {len(filtered_results)}개 통과 (유사도 ≥ 0.80)")
print("=" * 60)
for i, doc in enumerate(filtered_results, 1):
    print(f"\n[{i}] ({doc.metadata['source']}, p.{doc.metadata['page']})")
    print(f"    {doc.page_content[:80]}...")
```

```output
============================================================
🎯 컨텍스트 압축 결과 (EmbeddingsFilter)
   8개 검색 → 4개 통과 (유사도 ≥ 0.80)
============================================================

[1] (rag_optimization.pdf, p.1)
    RAG 시스템의 검색 품질을 높이려면 쿼리 재작성 기법을 활용할 수 있다. 사용자의 원본 질문을 LLM으로 변환하여 검...

[2] (rag_survey.pdf, p.15)
    Advanced RAG는 Naive RAG에 Pre-Retrieval과 Post-Retrieval 최적화를 추가한 패러다임이다. 쿼리 확장, 리랭킹...

[3] (rag_optimization.pdf, p.5)
    리랭킹(Reranking)은 초기 검색 결과를 Cross-Encoder 모델로 재평가하는 기법이다. Bi-Encoder보다 정확하지만 계...

[4] (rag_optimization.pdf, p.12)
    컨텍스트 압축은 검색된 문서에서 쿼리와 관련 있는 부분만 추출하는 기법이다. LLM 기반 추출과 임베딩 기반 필터링...
```

8개 문서 중 유사도 80% 미만인 4개(벡터 DB, 프롬프트 엔지니어링, HyDE, 평가 메트릭)가 걸러지고, **실제로 "검색 성능 개선"과 직결되는 4개 문서만** 남았습니다.

## 더 깊이 알아보기

### HyDE의 탄생 — "정답을 모르면 만들어서 찾자"

HyDE는 2022년 Carnegie Mellon University의 Luyu Gao 등이 발표한 논문 *"Precise Zero-Shot Dense Retrieval without Relevance Labels"*에서 처음 제안되었습니다. 당시 Dense Retrieval 분야에서는 검색 성능을 높이려면 대규모 관련성 레이블 데이터가 필요하다는 것이 상식이었는데요, Gao는 "LLM이 만든 **가짜 문서**의 임베딩이 진짜 문서를 찾는 데 오히려 효과적"이라는 반직관적 발견을 했습니다.

놀랍게도, HyDE는 학습 데이터 없이(Zero-Shot)도 지도 학습 기반 모델인 Contriever-ft보다 높은 검색 성능을 보였습니다. 이 발견은 "검색을 잘하려면 질문을 잘 해야 한다"가 아니라 **"검색을 잘하려면 답변의 형태를 예측해야 한다"**는 새로운 패러다임을 열었습니다.

### 리랭킹의 기원 — 정보 검색 분야의 오래된 지혜

리랭킹 개념 자체는 RAG보다 훨씬 오래되었습니다. 1990년대 정보 검색(Information Retrieval) 분야에서 이미 **2-stage retrieval** 패턴이 연구되었거든요. 1단계에서 BM25 같은 가벼운 알고리즘으로 후보를 추리고, 2단계에서 더 정교한 모델로 재순위를 매기는 방식이죠. 

이 전통이 딥러닝 시대를 거쳐 **Bi-Encoder + Cross-Encoder** 조합으로 진화했고, 2023~2024년 Cohere, Jina AI 등이 리랭킹 전용 API를 상용화하면서 RAG 생태계의 핵심 컴포넌트로 자리잡았습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Advanced RAG는 항상 Naive RAG보다 좋다" — 그렇지 않습니다. 문서 수가 적고 쿼리가 명확한 간단한 사용 사례에서는 Naive RAG가 충분히 좋은 결과를 내며, Advanced RAG의 추가 레이어가 오히려 **지연 시간과 비용만 증가**시킬 수 있습니다. 최적화가 필요한 병목이 어디인지 먼저 진단하세요.

> 💡 **알고 계셨나요?**: Multi-Query에서 LLM이 생성하는 관점별 쿼리 수는 기본 3개인데요, 이 숫자를 늘린다고 항상 좋아지진 않습니다. 쿼리 5개 이상부터는 **노이즈가 오히려 증가**하여 검색 품질이 떨어질 수 있다는 연구 결과가 있습니다. 3~4개가 최적의 균형점입니다.

> 🔥 **실무 팁**: 리랭킹 모델 선택이 어렵다면 **Cohere Rerank v3.5**부터 시작하세요. API 기반이라 인프라 구축 없이 바로 테스트할 수 있고, 다국어(한국어 포함)를 잘 지원합니다. 비용이 부담되면 오픈소스 `BAAI/bge-reranker-v2-m3`를 로컬에서 실행하는 것도 좋은 대안입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **Pre-Retrieval 최적화** | 검색 전에 쿼리를 개선하여 검색 품질을 높이는 기법 (쿼리 재작성, 쿼리 확장, HyDE) |
| **Post-Retrieval 최적화** | 검색 후 결과를 정제하여 LLM에 전달하는 컨텍스트 품질을 높이는 기법 (리랭킹, 압축) |
| **Multi-Query** | 하나의 질문을 여러 관점으로 확장하여 재현율(Recall)을 향상시키는 쿼리 확장 기법 |
| **HyDE** | LLM이 가설 답변을 생성하고 그 임베딩으로 검색하여 쿼리-문서 의미 격차를 해소하는 기법 |
| **리랭킹(Reranking)** | Cross-Encoder로 검색 결과의 관련성을 재계산하여 정밀도(Precision)를 높이는 기법 |
| **컨텍스트 압축** | 검색된 문서에서 관련 부분만 추출하거나, 저관련 문서를 제거하여 노이즈를 줄이는 기법 |
| **DocumentCompressorPipeline** | 여러 압축/필터링 기법을 순차 적용하는 LangChain 파이프라인 |
| **Bi-Encoder vs Cross-Encoder** | Bi-Encoder는 빠르고 대량 검색에 적합, Cross-Encoder는 느리지만 정확하여 리랭킹에 적합 |

## 다음 섹션 미리보기

지금까지 Advanced RAG가 **기존 파이프라인에 최적화 레이어를 추가**하는 방식이었다면, 다음 세션에서 다룰 **Modular RAG**는 파이프라인 자체를 **레고 블록처럼 자유롭게 재구성**하는 패러다임입니다. 검색-생성의 순서를 바꾸거나, 특정 모듈을 교체하거나, 피드백 루프를 추가하는 등 훨씬 유연한 아키텍처 설계가 가능해집니다. Advanced RAG의 각 기법이 어떻게 독립 모듈로 분리되어 조합되는지 살펴보겠습니다.

## 참고 자료

- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - Gao et al.의 RAG 서베이 논문. Naive/Advanced/Modular RAG 패러다임을 체계적으로 분류한 핵심 레퍼런스
- [LangChain: How to do retrieval with contextual compression](https://python.langchain.com/v0.2/docs/how_to/contextual_compression/) - ContextualCompressionRetriever, EmbeddingsFilter, LLMChainExtractor의 공식 구현 가이드
- [LangChain: MultiQueryRetriever](https://python.langchain.com/docs/how_to/MultiQueryRetriever/) - Multi-Query 검색의 공식 문서. 쿼리 확장 구현 패턴과 커스텀 프롬프트 설정 방법
- [Cohere Rerank on LangChain](https://docs.cohere.com/docs/rerank-on-langchain) - Cohere 리랭킹 API의 LangChain 통합 가이드. CohereRerank 설정 및 사용 예제
- [RAG Techniques Repository (GitHub)](https://github.com/NirDiamant/RAG_Techniques) - Nir Diamant의 Advanced RAG 기법 총정리. 각 기법별 구현 코드와 설명을 제공하는 실용적 레퍼런스
- [Cloudflare RAG Reference Architecture](https://developers.cloudflare.com/reference-architecture/diagrams/ai/ai-rag/) - RAG 아키텍처의 전체 흐름을 다이어그램으로 이해할 수 있는 레퍼런스

---
### 🔗 Related Sessions
- [ingestion_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [inference_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_database](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [naive_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [stuff_method](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [low_precision](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [low_recall](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [lost_in_the_middle](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [query_document_mismatch](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
