# Modular RAG — 유연한 모듈 조합 아키텍처

> RAG 시스템을 레고 블록처럼 자유롭게 조합하는 차세대 아키텍처 패러다임

## 개요

이 섹션에서는 Naive RAG와 Advanced RAG를 넘어, RAG 시스템의 각 구성 요소를 독립 모듈로 분리하고 자유롭게 조합하는 **Modular RAG** 아키텍처를 학습합니다. 고정된 파이프라인이 아닌, 쿼리의 특성에 따라 경로를 동적으로 결정하는 오케스트레이션(Orchestration) 개념을 이해하고, LangChain의 LCEL(LangChain Expression Language)로 이를 직접 구현해봅니다.

**선수 지식**: [세션 2.3: Advanced RAG — 검색 전/후 최적화 전략](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md)에서 배운 Pre-Retrieval, Post-Retrieval 최적화 개념과 MultiQueryRetriever, ContextualCompressionRetriever 사용법

**학습 목표**:
- Modular RAG의 3계층 아키텍처(모듈 → 서브모듈 → 오퍼레이터)를 설명할 수 있다
- 6개 핵심 모듈(Indexing, Pre-retrieval, Retrieval, Post-retrieval, Generation, Orchestration)의 역할을 구분할 수 있다
- 4가지 오케스트레이션 패턴(Linear, Conditional, Branching, Loop)의 차이를 이해한다
- LangChain LCEL의 `RunnableBranch`, `RunnableParallel`, `RunnableLambda`로 모듈형 RAG 파이프라인을 구축할 수 있다

## 왜 알아야 할까?

[세션 2.2](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md)에서 Naive RAG의 한계를, [세션 2.3](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md)에서 Advanced RAG의 개선 전략을 살펴보았습니다. 하지만 Advanced RAG에도 근본적인 제약이 남아 있습니다 — **모든 쿼리가 동일한 파이프라인을 거친다**는 점이죠.

실무에서는 이런 상황이 빈번합니다:
- "오늘 날씨 어때?"는 검색이 필요 없지만, "2024년 매출 데이터 분석해줘"는 정밀한 검색이 필수입니다
- 단순 사실 확인 질문과 복잡한 분석 질문에 같은 검색 전략을 쓰는 건 낭비이거나 부족합니다
- PDF, 웹페이지, 데이터베이스 등 소스가 다양해지면 하나의 고정 파이프라인으로는 한계가 있습니다

Modular RAG는 이런 문제를 해결합니다. **쿼리의 특성에 따라 어떤 모듈을 쓸지, 어떤 순서로 실행할지를 동적으로 결정**할 수 있거든요. 2024년 Gao et al.의 연구에 따르면, 현재 대부분의 프로덕션 RAG 시스템은 이미 Modular RAG 패러다임으로 진화하고 있습니다.

## 핵심 개념

### 개념 1: Modular RAG의 3계층 아키텍처 — 레고 블록 비유

> 💡 **비유**: 레고 세트를 떠올려보세요. 레고에는 **테마**(우주선, 성 등 = 모듈)가 있고, 각 테마 안에 **조립 파트**(엔진부, 날개부 = 서브모듈)가 있으며, 그 파트를 구성하는 **개별 블록**(2×4 브릭, 바퀴 = 오퍼레이터)이 있죠. Modular RAG도 정확히 이 구조입니다. 블록을 교체하면 같은 테마라도 완전히 다른 모양이 되듯, 오퍼레이터를 교체하면 같은 RAG 시스템이 전혀 다른 방식으로 동작합니다.

Modular RAG는 **3계층(Three-tier)** 구조로 설계됩니다:

| 계층 | 역할 | 예시 |
|------|------|------|
| **모듈(Module)** | RAG 프로세스의 핵심 단계 | Retrieval, Generation |
| **서브모듈(Sub-module)** | 모듈 내의 기능 단위 | Query Rewriting, Reranking |
| **오퍼레이터(Operator)** | 실제 실행되는 기본 연산 단위 | BM25 검색 함수, Cosine Similarity 계산 |

이 구조의 핵심은 **각 계층이 독립적으로 교체 가능**하다는 것입니다. 예를 들어 Retrieval 모듈의 Dense Retriever 오퍼레이터를 BM25로 바꾸더라도 나머지 모듈은 전혀 영향을 받지 않습니다.

```run:python
# Modular RAG의 3계층 구조를 딕셔너리로 표현
modular_rag_architecture = {
    "modules": {
        "indexing": {
            "sub_modules": ["chunk_optimization", "structural_organization"],
            "operators": ["RecursiveCharacterTextSplitter", "SemanticChunker", "HierarchicalIndexer"]
        },
        "pre_retrieval": {
            "sub_modules": ["query_transformation", "query_expansion"],
            "operators": ["MultiQueryRewriter", "HyDE_Generator", "StepBackPrompter"]
        },
        "retrieval": {
            "sub_modules": ["sparse_retrieval", "dense_retrieval", "hybrid_retrieval"],
            "operators": ["BM25_Retriever", "FAISS_Retriever", "EnsembleRetriever"]
        },
        "post_retrieval": {
            "sub_modules": ["reranking", "compression"],
            "operators": ["CohereRerank", "LongContextReorder", "EmbeddingsFilter"]
        },
        "generation": {
            "sub_modules": ["response_synthesis", "verification"],
            "operators": ["ChatOpenAI", "ChatAnthropic", "HallucinationChecker"]
        },
        "orchestration": {
            "sub_modules": ["routing", "scheduling", "fusion"],
            "operators": ["SemanticRouter", "ConfidenceScheduler", "RRF_Fusion"]
        }
    }
}

# 모듈 수와 전체 오퍼레이터 수 출력
total_operators = sum(
    len(m["operators"]) for m in modular_rag_architecture["modules"].values()
)
print(f"총 {len(modular_rag_architecture['modules'])}개 모듈, {total_operators}개 오퍼레이터")
print(f"\n모듈 목록: {', '.join(modular_rag_architecture['modules'].keys())}")
```

```output
총 6개 모듈, 18개 오퍼레이터

모듈 목록: indexing, pre_retrieval, retrieval, post_retrieval, generation, orchestration
```

### 개념 2: 6개 핵심 모듈 상세

> 💡 **비유**: Modular RAG의 6개 모듈은 **음식점의 주방 라인**과 비슷합니다. 식재료 준비(Indexing), 주문 확인(Pre-retrieval), 재료 꺼내기(Retrieval), 재료 손질(Post-retrieval), 요리(Generation), 그리고 주방장의 지시(Orchestration). 간단한 메뉴면 재료를 바로 꺼내 요리하지만, 복잡한 코스 요리라면 주방장이 순서를 정하고 여러 팀이 동시에 움직이죠.

앞서 [세션 2.1](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 배운 인제스천(Ingestion)과 인퍼런스(Inference) 파이프라인을 기억하시나요? Modular RAG는 이 흐름을 6개의 독립 모듈로 세분화합니다:

**1. Indexing 모듈** — 문서를 검색 가능한 형태로 변환
- 청크 최적화: 크기, 오버랩, 구조 결정
- 구조적 조직화: 계층적 인덱스, 지식 그래프 구성
- [세션 2.1](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md)의 `document_loading` → `text_splitting` → `embedding_generation` → `vector_storing` 과정에 해당

**2. Pre-retrieval 모듈** — 검색 전에 쿼리를 최적화
- [세션 2.3](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md)에서 배운 `query_rewriting`, `multi_query`, `hyde`가 모두 이 모듈의 오퍼레이터

**3. Retrieval 모듈** — 실제 검색 수행
- Sparse(BM25), Dense(벡터 검색), Hybrid 방식 중 선택
- 검색기(Retriever) 자체를 파인튜닝할 수도 있음

**4. Post-retrieval 모듈** — 검색 결과 정제
- [세션 2.3](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md)의 `reranking`, `contextual_compression`이 이 모듈에 해당

**5. Generation 모듈** — LLM으로 최종 답변 생성
- 답변 합성과 검증(Hallucination 체크)을 포함

**6. Orchestration 모듈** — 전체 흐름 제어 (Modular RAG의 핵심 차별점!)
- **라우팅(Routing)**: 쿼리 특성에 따라 다른 파이프라인으로 분기
- **스케줄링(Scheduling)**: 반복 검색의 종료 조건 결정
- **퓨전(Fusion)**: 여러 분기의 결과를 병합

바로 이 **Orchestration 모듈이 Modular RAG를 Naive/Advanced RAG와 구분 짓는 핵심**입니다. Naive RAG와 Advanced RAG에는 이 모듈이 없거든요.

### 개념 3: 4가지 오케스트레이션 패턴

오케스트레이션 모듈은 모듈들의 실행 흐름을 4가지 패턴으로 제어합니다:

**Linear(순차)** — 가장 단순한 패턴
```
Pre-retrieval → Retrieval → Post-retrieval → Generation
```
Naive RAG, Advanced RAG가 사용하는 방식입니다. 모든 쿼리가 동일한 경로를 거칩니다.

**Conditional(조건부)** — 쿼리에 따라 다른 경로 선택
```
Query → Router → [경로 A: 벡터 검색 → 생성]
                  [경로 B: SQL 검색 → 생성]
                  [경로 C: 검색 없이 → 직접 생성]
```
라우터가 쿼리의 의도를 분류하고 적절한 파이프라인으로 보냅니다.

**Branching(분기)** — 여러 경로를 동시 실행 후 병합
```
Query → [Branch 1: Dense 검색] → Fusion → Generation
        [Branch 2: Sparse 검색] ↗
        [Branch 3: KG 검색]    ↗
```
다양한 검색 전략을 병렬 실행하고 Reciprocal Rank Fusion 등으로 결과를 합칩니다.

**Loop(반복)** — 결과가 충분할 때까지 반복
```
Query → Retrieval → Generation → Judge → [충분?] → 최종 답변
                                    ↓ [불충분]
                                  Query 수정 → Retrieval (반복)
```
LLM이 검색 결과의 충분성을 판단하고, 부족하면 쿼리를 수정해 재검색합니다. [세션 2.5](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/05-rag-아키텍처-설계-실습-요구사항에서-설계까지.md)에서 다룰 RAG 시스템 설계 실습과 직접 연결되는 패턴입니다.

```run:python
# 4가지 오케스트레이션 패턴의 특성 비교
patterns = {
    "Linear": {
        "complexity": "★☆☆☆☆",
        "flexibility": "낮음",
        "use_case": "단순 Q&A, 프로토타입",
        "example": "Naive RAG, Advanced RAG"
    },
    "Conditional": {
        "complexity": "★★★☆☆",
        "flexibility": "중간",
        "use_case": "멀티 소스 RAG, 의도 분류 필요 시",
        "example": "쿼리 라우팅, 도메인별 분기"
    },
    "Branching": {
        "complexity": "★★★★☆",
        "flexibility": "높음",
        "use_case": "정밀도가 중요한 검색, 앙상블",
        "example": "하이브리드 검색, Multi-Query"
    },
    "Loop": {
        "complexity": "★★★★★",
        "flexibility": "매우 높음",
        "use_case": "복잡한 질문, 에이전틱 RAG",
        "example": "Self-RAG, CRAG, Agentic RAG"
    }
}

for name, info in patterns.items():
    print(f"[{name}] 복잡도: {info['complexity']}")
    print(f"  유연성: {info['flexibility']} | 용도: {info['use_case']}")
    print(f"  대표 예시: {info['example']}\n")
```

```output
[Linear] 복잡도: ★☆☆☆☆
  유연성: 낮음 | 용도: 단순 Q&A, 프로토타입
  대표 예시: Naive RAG, Advanced RAG

[Conditional] 복잡도: ★★★☆☆
  유연성: 중간 | 용도: 멀티 소스 RAG, 의도 분류 필요 시
  대표 예시: 쿼리 라우팅, 도메인별 분기

[Branching] 복잡도: ★★★★☆
  유연성: 높음 | 용도: 정밀도가 중요한 검색, 앙상블
  대표 예시: 하이브리드 검색, Multi-Query

[Loop] 복잡도: ★★★★★
  유연성: 매우 높음 | 용도: 복잡한 질문, 에이전틱 RAG
  대표 예시: Self-RAG, CRAG, Agentic RAG
```

### 개념 4: LangChain LCEL로 모듈형 RAG 구현하기

Modular RAG의 개념을 코드로 구현하려면 **각 모듈을 독립적인 컴포넌트로 만들고, 이들을 파이프 연산자(`|`)로 조합**할 수 있어야 합니다. LangChain의 LCEL이 바로 이 역할을 합니다.

LCEL의 핵심 빌딩 블록 4가지:

| 컴포넌트 | 역할 | Modular RAG 대응 |
|----------|------|-------------------|
| `RunnablePassthrough` | 입력을 그대로 전달 | 모듈 간 데이터 전달 |
| `RunnableLambda` | 임의의 Python 함수를 모듈화 | 커스텀 오퍼레이터 생성 |
| `RunnableParallel` | 여러 모듈을 동시 실행 | Branching 패턴 구현 |
| `RunnableBranch` | 조건에 따라 다른 경로 선택 | Conditional 패턴 구현 |

```python
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# 각 모듈을 독립적인 Runnable로 정의
def query_classifier(query: str) -> dict:
    """쿼리를 분류하여 적절한 경로를 결정하는 라우터"""
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["코드", "구현", "함수", "클래스"]):
        return {"type": "code", "query": query}
    elif any(kw in query_lower for kw in ["비교", "차이", "versus", "vs"]):
        return {"type": "comparison", "query": query}
    else:
        return {"type": "general", "query": query}

# Conditional 패턴: RunnableBranch로 쿼리 유형별 분기
routing_chain = RunnableBranch(
    # (조건 함수, 실행할 Runnable) 쌍
    (lambda x: x["type"] == "code", code_retrieval_chain),       # 코드 전용 검색
    (lambda x: x["type"] == "comparison", comparison_chain),     # 비교 분석 체인
    general_retrieval_chain,  # 기본값: 일반 검색
)

# 전체 파이프라인 조합
modular_pipeline = (
    RunnableLambda(query_classifier)  # Pre-retrieval: 쿼리 분류
    | routing_chain                    # Orchestration: 조건부 라우팅
)
```

`RunnableParallel`을 활용한 **Branching 패턴**도 살펴보겠습니다:

```python
from langchain_core.runnables import RunnableParallel

# Branching 패턴: 여러 검색 전략을 동시 실행
parallel_retrieval = RunnableParallel(
    dense_results=dense_retriever,     # Dense 벡터 검색
    sparse_results=sparse_retriever,   # BM25 키워드 검색
    kg_results=kg_retriever,           # 지식 그래프 검색
)

# 결과 병합 함수 (Fusion)
def fuse_results(results: dict) -> list:
    """Reciprocal Rank Fusion으로 여러 검색 결과를 병합"""
    all_docs = []
    for source, docs in results.items():
        for rank, doc in enumerate(docs):
            doc.metadata["rrf_score"] = 1 / (rank + 60)  # RRF 공식
            doc.metadata["source_retriever"] = source
            all_docs.append(doc)
    # RRF 점수 기준으로 정렬
    return sorted(all_docs, key=lambda d: d.metadata["rrf_score"], reverse=True)

# Branching → Fusion → Generation
branching_pipeline = (
    parallel_retrieval
    | RunnableLambda(fuse_results)
    | generation_chain
)
```

### 개념 5: 세 패러다임 비교 — Naive vs Advanced vs Modular

지금까지 세 가지 RAG 패러다임을 모두 살펴보았으니, 전체적으로 비교해봅시다:

| 특성 | Naive RAG | Advanced RAG | Modular RAG |
|------|-----------|--------------|-------------|
| **구조** | 고정 직선형 | 고정 직선형 + 최적화 | 동적 그래프형 |
| **흐름 제어** | 없음 | 없음 | Orchestration 모듈 |
| **검색 전략** | 단일 (Dense만) | 단일 (개선됨) | 복수 (상황별 선택) |
| **모듈 교체** | 불가 | 부분적 | 완전히 독립적 |
| **쿼리 적응** | 동일 처리 | 동일 처리 (개선됨) | 쿼리별 다른 경로 |
| **대표 패턴** | Stuff Chain | Multi-Query + Rerank | 라우팅 + 앙상블 |
| **적합한 상황** | 프로토타입 | 단일 도메인 | 프로덕션 멀티 도메인 |

> ⚠️ **흔한 오해**: "Modular RAG가 항상 Advanced RAG보다 좋다"고 생각하기 쉽지만, 그렇지 않습니다. 단일 도메인의 간단한 Q&A라면 Advanced RAG의 고정 파이프라인이 오히려 구현과 유지보수가 쉽습니다. Modular RAG는 **복잡성이 정당화될 때** 빛을 발합니다.

## 실습: 직접 해보기

이제 LangChain LCEL을 사용하여 **쿼리 유형에 따라 다른 검색 전략을 선택하는 Modular RAG 파이프라인**을 구축해봅시다. 이 실습에서는 Conditional 패턴과 Branching 패턴을 모두 구현합니다.

```python
"""
Modular RAG 파이프라인 실습
- Conditional 패턴: 쿼리 의도별 라우팅
- Branching 패턴: 하이브리드 검색 앙상블
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.documents import Document

load_dotenv()

# ──────────────────────────────────────────────
# 1단계: 모듈 준비 (Indexing 모듈의 결과물)
# ──────────────────────────────────────────────

# 샘플 문서 — 실무에서는 벡터 DB에서 로드
sample_docs = [
    Document(page_content="RAG는 Retrieval-Augmented Generation의 약자로, 검색 증강 생성 기법입니다.", 
             metadata={"source": "basics", "type": "definition"}),
    Document(page_content="LangChain의 LCEL은 파이프 연산자(|)로 컴포넌트를 조합하는 선언적 문법입니다.", 
             metadata={"source": "code", "type": "tutorial"}),
    Document(page_content="Naive RAG는 단순 검색-생성 구조이고, Advanced RAG는 검색 전/후 최적화를 추가합니다.", 
             metadata={"source": "basics", "type": "comparison"}),
    Document(page_content="ChromaDB는 오픈소스 벡터 데이터베이스로, pip install chromadb로 설치합니다.", 
             metadata={"source": "code", "type": "tutorial"}),
    Document(page_content="Modular RAG는 6개 모듈로 구성되며, Orchestration 모듈이 흐름을 제어합니다.", 
             metadata={"source": "basics", "type": "definition"}),
    Document(page_content="BM25는 키워드 빈도 기반 검색이고, Dense Retrieval은 의미 기반 벡터 검색입니다.", 
             metadata={"source": "basics", "type": "comparison"}),
]

# 임베딩 모델과 벡터 스토어 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(sample_docs, embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ──────────────────────────────────────────────
# 2단계: Pre-retrieval 모듈 — 쿼리 분류기(Router)
# ──────────────────────────────────────────────

def classify_query(query: str) -> dict:
    """쿼리 의도를 분류하는 라우터 함수"""
    query_lower = query.lower()
    
    # 메타데이터 기반 라우팅 (실무에서는 LLM 기반 분류도 가능)
    if any(kw in query_lower for kw in ["비교", "차이", "vs", "versus"]):
        intent = "comparison"
    elif any(kw in query_lower for kw in ["코드", "구현", "설치", "import"]):
        intent = "code"
    elif any(kw in query_lower for kw in ["뭐", "무엇", "정의", "약자"]):
        intent = "definition"
    else:
        intent = "general"
    
    return {"intent": intent, "query": query}

# ──────────────────────────────────────────────
# 3단계: 의도별 프롬프트 모듈 (Generation 모듈)
# ──────────────────────────────────────────────

# 비교 질문용 프롬프트
comparison_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 바탕으로 비교 분석을 해주세요.
표 형태로 차이점을 정리하고, 각각의 장단점을 설명해주세요.

컨텍스트: {context}
질문: {query}
"""
)

# 코드/구현 질문용 프롬프트
code_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 바탕으로 구현 방법을 설명해주세요.
가능하면 코드 예제를 포함해주세요.

컨텍스트: {context}
질문: {query}
"""
)

# 일반 질문용 프롬프트
general_prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 바탕으로 질문에 답변해주세요.
명확하고 간결하게 설명해주세요.

컨텍스트: {context}
질문: {query}
"""
)

# ──────────────────────────────────────────────
# 4단계: Orchestration 모듈 — Conditional 라우팅
# ──────────────────────────────────────────────

def format_docs(docs: list) -> str:
    """검색된 문서들을 하나의 문자열로 합치는 유틸리티"""
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain_for_intent(prompt_template):
    """의도별 체인을 생성하는 팩토리 함수"""
    return (
        RunnableParallel(
            context=lambda x: format_docs(dense_retriever.invoke(x["query"])),
            query=lambda x: x["query"],
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

# 의도별 체인 생성
comparison_chain = build_chain_for_intent(comparison_prompt)
code_chain = build_chain_for_intent(code_prompt)
general_chain = build_chain_for_intent(general_prompt)

# Conditional 라우팅: RunnableBranch로 의도별 분기
routing_chain = RunnableBranch(
    (lambda x: x["intent"] == "comparison", comparison_chain),
    (lambda x: x["intent"] == "code", code_chain),
    general_chain,  # 기본 분기 (definition, general 등)
)

# ──────────────────────────────────────────────
# 5단계: 전체 Modular RAG 파이프라인 조합
# ──────────────────────────────────────────────

modular_rag_pipeline = (
    RunnableLambda(classify_query)   # Pre-retrieval: 쿼리 분류
    | routing_chain                   # Orchestration: 조건부 라우팅
)

# ──────────────────────────────────────────────
# 6단계: 다양한 쿼리로 테스트
# ──────────────────────────────────────────────

test_queries = [
    "Naive RAG와 Advanced RAG의 차이가 뭐야?",    # → comparison
    "ChromaDB 설치하는 코드 알려줘",                 # → code
    "RAG가 뭐야?",                                   # → definition (→ general)
]

for query in test_queries:
    classified = classify_query(query)
    print(f"[쿼리] {query}")
    print(f"[분류] {classified['intent']}")
    
    # 실제 LLM 호출 (API 키 필요)
    # result = modular_rag_pipeline.invoke(query)
    # print(f"[답변] {result}")
    print("---")
```

> 🔥 **실무 팁**: 위 코드에서는 키워드 기반으로 쿼리를 분류했지만, 프로덕션에서는 LLM 자체를 라우터로 쓰는 **시맨틱 라우팅(Semantic Routing)**이 더 정확합니다. LLM에게 "이 쿼리는 비교/코드/정의/일반 중 어떤 유형인지 분류해줘"라고 요청하면 됩니다. 다만 라우팅 자체에 LLM 호출이 추가되므로 지연 시간과 비용의 트레이드오프를 고려해야 합니다.

## 더 깊이 알아보기

### Modular RAG의 탄생 배경

Modular RAG라는 개념이 공식적으로 정립된 것은 2024년입니다. Yunfan Gao 등의 연구진이 2023년 말에 발표한 서베이 논문 "Retrieval-Augmented Generation for Large Language Models: A Survey"에서 RAG의 발전을 Naive → Advanced → Modular 세 패러다임으로 분류했죠. 이 분류는 RAG 커뮤니티에서 빠르게 표준으로 자리잡았습니다.

이어서 2024년 7월, 같은 연구진이 "Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks"라는 논문을 발표하며 Modular RAG의 구체적인 아키텍처를 정의했습니다. 이 논문에서 제안한 **3계층 구조(모듈-서브모듈-오퍼레이터)**와 **6개 핵심 모듈**, **4가지 오케스트레이션 패턴**이 현재 Modular RAG의 표준 정의가 되었습니다.

흥미로운 점은, "Modular RAG"라는 이름이 붙기 전에도 이미 많은 프로덕션 시스템이 사실상 Modular RAG를 구현하고 있었다는 것입니다. LangChain의 LCEL, LlamaIndex의 Query Engine, Haystack의 Pipeline 등 주요 프레임워크들은 이미 모듈형 설계를 지원하고 있었거든요. 연구진은 이런 실무 패턴을 관찰하고 체계화한 것입니다.

### "레고" 비유의 유래

논문 제목에 "LEGO-like"이라는 표현이 들어간 데는 이유가 있습니다. 소프트웨어 공학에서 "레고 블록 아키텍처"는 마이크로서비스, 플러그인 시스템 등 모듈형 설계의 대명사인데요, RAG 분야에서도 동일한 설계 철학이 필요하다는 메시지를 담고 있습니다. 실제로 FlashRAG, UltraRAG 같은 오픈소스 도구킷들은 이 논문의 구조를 직접 참조하여 설계되었습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Modular RAG를 구현하려면 처음부터 6개 모듈을 모두 만들어야 한다"고 생각하기 쉽습니다. 하지만 실제로는 **점진적 모듈화**가 핵심입니다. Naive RAG로 시작해서, 성능 병목이 발견되는 지점부터 하나씩 모듈을 교체하거나 추가하면 됩니다. 예를 들어 검색 정밀도가 문제라면 Retrieval 모듈만 개선하고, 다양한 질문 유형 대응이 필요하면 Orchestration 모듈을 추가하는 식이죠.

> 💡 **알고 계셨나요?**: 2024년 발표된 MBA-RAG(Multi-Branch Adaptive RAG) 연구에 따르면, 모듈형 적응 라우팅을 도입하면 기존 분류기 기반 방식 대비 평균 검색 단계가 2.17에서 1.80으로 줄어들면서도 QA 정확도(EM/F1)는 오히려 향상됩니다. 불필요한 검색을 줄이면서 정확도를 높이는 셈이죠.

> 🔥 **실무 팁**: Modular RAG를 설계할 때 가장 먼저 고민해야 할 것은 **오케스트레이션 패턴 선택**입니다. 대부분의 프로덕션 시스템에서는 Conditional 패턴으로 시작하는 것이 적절합니다. "검색이 필요한 쿼리인가?"를 판단하는 단순한 라우터 하나만 추가해도 불필요한 검색 비용을 크게 줄일 수 있습니다. Loop 패턴은 복잡도가 높으므로 Conditional → Branching → Loop 순서로 점진적으로 도입하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Modular RAG | RAG 시스템을 독립 모듈로 분리하고 동적으로 조합하는 아키텍처 패러다임 |
| 3계층 구조 | 모듈(Module) → 서브모듈(Sub-module) → 오퍼레이터(Operator)의 계층적 설계 |
| 6개 핵심 모듈 | Indexing, Pre-retrieval, Retrieval, Post-retrieval, Generation, Orchestration |
| Orchestration | 라우팅, 스케줄링, 퓨전을 통해 모듈 실행 흐름을 제어하는 핵심 모듈 |
| Linear 패턴 | 순차 실행 — Naive/Advanced RAG 방식 |
| Conditional 패턴 | 쿼리 의도에 따라 다른 파이프라인으로 분기 (`RunnableBranch`) |
| Branching 패턴 | 여러 검색 전략을 동시 실행 후 병합 (`RunnableParallel`) |
| Loop 패턴 | 충분한 결과를 얻을 때까지 반복 검색 — Agentic RAG로 연결 |
| LCEL | LangChain Expression Language — 파이프 연산자로 모듈을 조합하는 선언적 문법 |

## 다음 섹션 미리보기

이번 세션에서 Modular RAG의 이론적 프레임워크를 이해했다면, 다음 [세션 2.5: RAG 시스템 설계 실습](02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/05-rag-아키텍처-설계-실습-요구사항에서-설계까지.md)에서는 이 모듈형 아키텍처를 **실제 프로젝트에 적용하는 설계 프로세스**를 실습합니다. 요구사항 분석 프레임워크로 시스템의 목표와 제약 조건을 정의하고, 컴포넌트 선택 매트릭스를 활용하여 각 모듈에 적합한 기술 스택을 비교·선정합니다. 또한 ADR(Architecture Decision Record) 형식으로 설계 결정을 문서화하는 방법과, 선택한 아키텍처의 비용을 추정하는 과정까지 다루며, Modular RAG의 개념을 구체적인 설계 산출물로 연결합니다.

## 참고 자료

- [Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks (Gao et al., 2024)](https://arxiv.org/abs/2407.21059) - Modular RAG의 3계층 아키텍처와 4가지 오케스트레이션 패턴을 정의한 핵심 논문
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997) - Naive → Advanced → Modular RAG 패러다임 분류의 원본 서베이 논문
- [Azure RAG Solution Design and Evaluation Guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide) - 프로덕션 RAG 아키텍처 설계의 실무 가이드
- [LangChain LCEL — Runnable 컴포넌트 공식 문서](https://docs.langchain.com/oss/python/langchain/rag) - RunnableBranch, RunnableParallel 등 LCEL 빌딩 블록 API 레퍼런스
- [FlashRAG — Python Toolkit for Reproducible RAG Research](https://github.com/RUC-NLPIR/FlashRAG) - Modular RAG 구조를 참조하여 설계된 오픈소스 RAG 연구 도구킷

---
### 🔗 Related Sessions
- [ingestion_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [inference_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [naive_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [advanced_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [query_rewriting](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [multi_query](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [hyde](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [reranking](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [contextual_compression](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
