# RAG 아키텍처 설계 실습 — 요구사항에서 설계까지

> 실제 비즈니스 시나리오를 기반으로 RAG 시스템의 컴포넌트를 선택하고, 아키텍처를 설계하는 전 과정을 체험합니다.

## 개요

이 섹션에서는 "사내 문서 QA 시스템"이라는 구체적인 비즈니스 요구사항을 받았을 때, 어떤 사고 과정을 거쳐 RAG 아키텍처를 설계하는지 처음부터 끝까지 실습합니다. 앞서 배운 Naive RAG, Advanced RAG, Modular RAG 패러다임 중 어떤 것을 선택할지, 각 컴포넌트를 어떤 기준으로 고르는지, 그리고 최종 설계를 코드로 어떻게 표현하는지까지 다룹니다.

**선수 지식**: [RAG 파이프라인 전체 구조](01-rag-파이프라인-전체-구조-ingestion과-inference.md)의 인제스천/인퍼런스 흐름, [Naive RAG](02-naive-rag-기본-패턴과-한계.md)의 한계, [Advanced RAG](03-advanced-rag-검색-전후-최적화-전략.md)의 최적화 기법, [Modular RAG](04-modular-rag-유연한-모듈-조합-아키텍처.md)의 모듈 조합 패턴

**학습 목표**:
- 비즈니스 요구사항을 RAG 설계 결정으로 변환하는 프레임워크를 적용할 수 있다
- 임베딩 모델, 벡터 데이터베이스, 청킹 전략 등 핵심 컴포넌트의 선택 기준을 설명할 수 있다
- 요구사항에 맞는 RAG 아키텍처를 LCEL 코드로 구현할 수 있다
- 설계 결정의 트레이드오프(비용 vs 성능 vs 지연시간)를 분석할 수 있다

## 왜 알아야 할까?

"RAG 좀 만들어주세요."

실무에서 이런 요청을 받으면, 대부분 바로 코드를 짜기 시작합니다. LangChain 문서를 열고, 예제를 복사하고, ChromaDB를 띄우고... 그런데 2주 후, 검색 품질이 기대에 못 미치거나, 응답 지연이 3초를 넘기거나, 월 비용이 예산을 초과하는 문제에 부딪히게 됩니다.

왜 그럴까요? **설계 없이 구현부터 시작했기 때문**입니다.

Microsoft Azure 아키텍처 센터에서는 RAG 솔루션 개발을 준비(Preparation) → 청킹(Chunking) → 강화(Enrichment) → 임베딩(Embedding) → 검색(Retrieval) → 평가(Evaluation)의 6단계로 나누고, 각 단계마다 체계적 평가를 거칠 것을 권장합니다. 이 구조적 접근법이 바로 이 섹션에서 실습할 내용입니다.

건축에서 설계도 없이 집을 짓지 않듯, RAG 시스템도 아키텍처 설계가 먼저입니다.

## 핵심 개념

### 개념 1: 요구사항 분석 프레임워크 — 질문이 설계를 만든다

> 💡 **비유**: 병원에 가면 의사가 먼저 증상을 물어보죠. "어디가 아프세요?", "언제부터요?", "다른 약 드시는 거 있으세요?" — 이 질문들의 답이 치료 방법을 결정합니다. RAG 설계도 마찬가지입니다. 올바른 질문을 던져야 올바른 아키텍처가 나옵니다.

RAG 시스템 설계의 첫 단계는 코드를 작성하는 것이 아니라, **요구사항을 설계 결정으로 변환하는 질문**을 던지는 것입니다. Microsoft의 RAG 설계 가이드에서는 이를 "준비 단계(Preparation Phase)"라 부르며, 다음 네 가지 축으로 요구사항을 분류합니다.

**1) 데이터 특성**
- 문서 형식은 무엇인가? (PDF, Word, HTML, 데이터베이스)
- 문서 양은 얼마나 되는가? (수백 건 vs 수십만 건)
- 업데이트 빈도는? (일 1회 vs 실시간)
- 다국어 지원이 필요한가?

**2) 쿼리 특성**
- 사용자 질문의 유형은? (사실 확인, 요약, 비교, 분석)
- 예상 질문 길이와 복잡도는?
- 동시 사용자 수는?

**3) 품질 요구사항**
- 허용 가능한 할루시네이션 수준은?
- 출처 표기가 필수인가?
- 답변의 최신성이 중요한가?

**4) 운영 제약**
- 예산은? (월 API 비용 + 인프라 비용)
- 응답 지연 시간 제한은? (1초 이내? 3초 이내?)
- 온프레미스 제약이 있는가?

이 질문들의 답이 모이면, 어떤 RAG 패러다임을 쓸지, 어떤 컴포넌트를 선택할지가 자연스럽게 결정됩니다.

```run:python
# 요구사항을 구조화하는 간단한 데이터 클래스
from dataclasses import dataclass, field

@dataclass
class RAGRequirements:
    """RAG 시스템 요구사항 명세"""
    # 데이터 특성
    doc_formats: list[str] = field(default_factory=list)      # 문서 형식
    doc_count: int = 0                                          # 예상 문서 수
    update_frequency: str = "daily"                             # 업데이트 빈도
    languages: list[str] = field(default_factory=lambda: ["ko"])

    # 쿼리 특성
    query_types: list[str] = field(default_factory=list)       # 질문 유형
    concurrent_users: int = 10                                  # 동시 사용자
    avg_query_length: str = "medium"                            # 평균 질문 길이

    # 품질 요구사항
    accuracy_priority: str = "high"                             # 정확도 우선순위
    source_citation: bool = True                                # 출처 표기 필수
    freshness_critical: bool = False                            # 최신성 중요도

    # 운영 제약
    monthly_budget_usd: float = 500.0                           # 월 예산
    max_latency_sec: float = 3.0                                # 최대 응답 시간
    on_premise: bool = False                                     # 온프레미스 제약

# 사내 문서 QA 시스템 요구사항 정의
requirements = RAGRequirements(
    doc_formats=["pdf", "docx", "html", "pptx"],
    doc_count=5000,
    update_frequency="weekly",
    languages=["ko", "en"],
    query_types=["fact_check", "summary", "comparison"],
    concurrent_users=50,
    avg_query_length="medium",
    accuracy_priority="high",
    source_citation=True,
    freshness_critical=False,
    monthly_budget_usd=800.0,
    max_latency_sec=3.0,
    on_premise=False,
)

print(f"📋 프로젝트: 사내 문서 QA 시스템")
print(f"  문서: {requirements.doc_count}건 ({', '.join(requirements.doc_formats)})")
print(f"  동시 사용자: {requirements.concurrent_users}명")
print(f"  정확도: {requirements.accuracy_priority} | 출처 표기: {requirements.source_citation}")
print(f"  예산: ${requirements.monthly_budget_usd}/월 | 지연: {requirements.max_latency_sec}초 이내")
```

```output
📋 프로젝트: 사내 문서 QA 시스템
  문서: 5000건 (pdf, docx, html, pptx)
  동시 사용자: 50명
  정확도: high | 출처 표기: True
  예산: $800.0/월 | 지연: 3.0초 이내
```

### 개념 2: 컴포넌트 선택 기준 — 설계 결정 매트릭스

> 💡 **비유**: 자동차를 살 때 엔진, 변속기, 서스펜션을 따로 고르지는 않죠. 하지만 레이싱카를 만든다면 각 부품을 개별적으로 선택하고 조합해야 합니다. RAG 시스템도 프로덕션 수준이라면 각 컴포넌트를 요구사항에 맞게 개별 선택해야 합니다.

요구사항이 정리되면 **여섯 가지 핵심 컴포넌트**를 결정해야 합니다. [RAG 파이프라인 전체 구조](01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 배운 인제스천 파이프라인과 인퍼런스 파이프라인 각각에서 어떤 도구를 쓸지 고르는 과정이죠.

#### 1) 문서 로더 (Document Loader)

문서 형식이 결정의 핵심입니다.

| 문서 형식 | 추천 로더 | 특징 |
|-----------|----------|------|
| PDF | `PyPDFLoader`, `Unstructured` | PyPDF는 가볍고 빠름, Unstructured는 테이블/이미지 추출 |
| DOCX | `Docx2txtLoader` | 순수 텍스트 추출에 최적 |
| HTML | `BSHTMLLoader` | BeautifulSoup 기반, 태그 필터링 가능 |
| PPTX | `UnstructuredPowerPointLoader` | 슬라이드별 분리 지원 |
| 혼합 형식 | `DirectoryLoader` + 자동 감지 | 여러 로더를 형식별로 매핑 |

우리 시나리오에서는 PDF, DOCX, HTML, PPTX 네 가지를 모두 처리해야 하므로, `DirectoryLoader`와 형식별 로더 매핑이 적합합니다.

#### 2) 청킹 전략 (Chunking Strategy)

청킹 품질이 검색 정확도를 제약하는 가장 큰 요인이라는 연구 결과가 있습니다. 2024년 CDC 정책 문서 RAG 연구에서, 고정 크기 청킹은 Faithfulness 점수 0.47~0.51인 반면, 최적화된 시맨틱 청킹은 0.79~0.82를 기록했거든요.

| 전략 | 적합한 상황 | 청크 크기 |
|------|-----------|----------|
| 고정 크기 (Fixed-size) | 균일한 문서, 빠른 프로토타입 | 500~1000 토큰 |
| 재귀적 분할 (Recursive) | 범용, 구조화된 문서 | 500~1000 토큰, overlap 100~200 |
| 시맨틱 (Semantic) | 주제가 다양한 문서 | 가변 (600~900 토큰 권장) |
| 문서 구조 기반 (Structure-aware) | 제목/소제목이 명확한 문서 | 섹션 단위 |

사내 문서는 대체로 제목-본문 구조가 명확한 편이므로, **재귀적 분할을 기본으로 하되 문서 구조 메타데이터를 활용**하는 접근이 좋습니다.

#### 3) 임베딩 모델 (Embedding Model)

임베딩 모델 선택은 전체 시스템에 **되돌리기 어려운 영향**을 미칩니다. 모델을 바꾸면 전체 문서를 재임베딩해야 하니까요.

| 모델 | 차원 수 | 한국어 지원 | 비용 (1M 토큰) | 특징 |
|------|---------|-----------|--------------|------|
| `text-embedding-3-small` | 1536 | 좋음 | ~$0.02 | 가성비 최고 |
| `text-embedding-3-large` | 3072 | 좋음 | ~$0.13 | 높은 정확도 |
| `voyage-3-large` | 1024 | 보통 | ~$0.06 | MTEB 상위, 저차원 |
| `multilingual-e5-large` | 1024 | 우수 | 무료 (로컬) | 다국어 특화 |

한국어-영어 혼합 문서를 다루는 우리 시나리오에서는 `text-embedding-3-small`이 비용 효율적이고, 정확도가 더 필요하면 `text-embedding-3-large`로 업그레이드할 수 있습니다.

#### 4) 벡터 데이터베이스 (Vector Database)

| DB | 유형 | 확장성 | 비용 | 적합한 상황 |
|----|------|--------|------|-----------|
| ChromaDB | 인메모리/로컬 | 소~중 | 무료 | 프로토타입, 소규모 |
| FAISS | 라이브러리 | 중~대 | 무료 | 로컬 고성능 검색 |
| Pinecone | 서버리스 클라우드 | 대규모 | 종량제 | 관리형 프로덕션 |
| Qdrant | 자체 호스팅/클라우드 | 중~대 | 무료(OSS)/종량제 | 메타데이터 필터링 강점 |

5000건 문서, 50명 동시 사용자라면 **ChromaDB로 프로토타입 → Qdrant 또는 Pinecone으로 프로덕션 전환** 전략이 현실적입니다.

#### 5) 검색 전략 (Retrieval Strategy)

[Naive RAG](02-naive-rag-기본-패턴과-한계.md)에서 배운 단순 유사도 검색의 한계를 기억하시죠? [Advanced RAG](03-advanced-rag-검색-전후-최적화-전략.md)에서 배운 기법들을 요구사항에 맞게 조합합니다.

| 전략 | 지연 시간 추가 | 정확도 향상 | 권장 상황 |
|------|-------------|-----------|----------|
| 벡터 검색만 | 기준선 | 기준선 | 프로토타입 |
| 하이브리드 (벡터 + BM25) | +50~100ms | 높음 | 키워드 매칭 중요 시 |
| + 리랭킹 | +50~200ms | 매우 높음 | 정확도 최우선 |
| + 쿼리 변환 | +500~1500ms (LLM 호출) | 높음 | 복잡한 질문 |

정확도가 최우선(accuracy_priority="high")인 우리 시나리오에서는 **하이브리드 검색 + 리랭킹**이 적합합니다. 3초 지연 제한 내에서 충분히 수용 가능한 수준이거든요.

#### 6) RAG 패러다임 결정

마지막으로, [Modular RAG](04-modular-rag-유연한-모듈-조합-아키텍처.md)에서 배운 세 가지 패러다임 중 어떤 것을 선택할지 결정합니다.

| 패러다임 | 적합한 상황 | 개발 복잡도 |
|---------|-----------|-----------|
| Naive RAG | MVP, 단순 QA, 빠른 검증 | 낮음 |
| Advanced RAG | 검색 품질 중요, 안정적 쿼리 패턴 | 중간 |
| Modular RAG | 다양한 쿼리 유형, 동적 라우팅 필요 | 높음 |

사내 문서 QA 시스템은 사실 확인, 요약, 비교 등 다양한 쿼리 유형을 처리해야 하지만, 초기에는 **Advanced RAG로 시작하여 필요 시 Modular RAG로 진화**시키는 전략이 실용적입니다.

### 개념 3: 설계 결정 문서화 — Architecture Decision Record

> 💡 **비유**: 요리 레시피에는 "왜 이 재료를 쓰는지" 적지 않죠. 하지만 프로 셰프의 레시피북에는 대체 재료와 선택 이유가 함께 적혀 있습니다. RAG 설계 문서도 "무엇을 선택했는가"만큼 **"왜 선택했는가"**가 중요합니다.

설계 결정을 코드로 옮기기 전에, ADR(Architecture Decision Record) 형식으로 결정 사항을 문서화하는 것이 좋습니다. 나중에 컴포넌트를 교체하거나 팀원이 합류할 때, 왜 이런 선택을 했는지 추적할 수 있거든요.

```run:python
# 설계 결정을 구조화하는 유틸리티
from dataclasses import dataclass

@dataclass
class DesignDecision:
    """아키텍처 설계 결정 기록"""
    component: str        # 컴포넌트명
    choice: str           # 선택한 옵션
    alternatives: list[str]  # 고려한 대안들
    rationale: str        # 선택 이유
    tradeoff: str         # 트레이드오프

# 사내 문서 QA 시스템의 설계 결정들
decisions = [
    DesignDecision(
        component="임베딩 모델",
        choice="text-embedding-3-small",
        alternatives=["text-embedding-3-large", "multilingual-e5-large"],
        rationale="한국어+영어 지원, 월 예산 내 비용, 충분한 정확도",
        tradeoff="large 모델 대비 약간 낮은 정확도, 추후 업그레이드 가능",
    ),
    DesignDecision(
        component="벡터 DB",
        choice="ChromaDB → Qdrant (마이그레이션 계획)",
        alternatives=["Pinecone", "FAISS", "Weaviate"],
        rationale="초기 ChromaDB로 빠른 검증, Qdrant로 메타데이터 필터링+확장성 확보",
        tradeoff="마이그레이션 비용 발생, 그러나 인터페이스 호환으로 최소화",
    ),
    DesignDecision(
        component="청킹 전략",
        choice="RecursiveCharacterTextSplitter (800 토큰, 200 오버랩)",
        alternatives=["SemanticChunker", "MarkdownHeaderTextSplitter"],
        rationale="범용성, 구현 단순, 사내 문서 구조에 적합한 크기",
        tradeoff="시맨틱 청킹 대비 주제 경계 인식 약함",
    ),
    DesignDecision(
        component="검색 전략",
        choice="하이브리드 검색(벡터 + BM25) + 리랭킹",
        rationale="정확도 최우선 요구사항, BM25로 키워드 정확 매칭 보완, 3초 지연 제한 내 수용 가능",
        alternatives=["벡터 검색만", "벡터 + 쿼리 변환"],
        tradeoff="리랭킹 API 비용 추가 (Cohere), BM25 인덱스 메모리 사용, 지연 +100~250ms",
    ),
]

for d in decisions:
    print(f"🔧 {d.component}: {d.choice}")
    print(f"   이유: {d.rationale}")
    print(f"   트레이드오프: {d.tradeoff}")
    print()
```

```output
🔧 임베딩 모델: text-embedding-3-small
   이유: 한국어+영어 지원, 월 예산 내 비용, 충분한 정확도
   트레이드오프: large 모델 대비 약간 낮은 정확도, 추후 업그레이드 가능

🔧 벡터 DB: ChromaDB → Qdrant (마이그레이션 계획)
   이유: 초기 ChromaDB로 빠른 검증, Qdrant로 메타데이터 필터링+확장성 확보
   트레이드오프: 마이그레이션 비용 발생, 그러나 인터페이스 호환으로 최소화

🔧 청킹 전략: RecursiveCharacterTextSplitter (800 토큰, 200 오버랩)
   이유: 범용성, 구현 단순, 사내 문서 구조에 적합한 크기
   트레이드오프: 시맨틱 청킹 대비 주제 경계 인식 약함

🔧 검색 전략: 하이브리드 검색(벡터 + BM25) + 리랭킹
   이유: 정확도 최우선 요구사항, BM25로 키워드 정확 매칭 보완, 3초 지연 제한 내 수용 가능
   트레이드오프: 리랭킹 API 비용 추가 (Cohere), BM25 인덱스 메모리 사용, 지연 +100~250ms
```

### 개념 4: 아키텍처 다이어그램 — 전체 그림 그리기

> 💡 **비유**: 이사할 때 가구 배치도를 먼저 그리는 것처럼, RAG 시스템도 컴포넌트 배치와 데이터 흐름을 시각적으로 그려봐야 합니다. 머릿속 설계와 실제 구현 사이의 간극을 다이어그램이 메워줍니다.

[RAG 파이프라인 전체 구조](01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 배운 인제스천/인퍼런스 두 파이프라인을 기반으로, 우리 시나리오의 아키텍처를 구체화합니다.

**인제스천 파이프라인 (Ingestion Pipeline)**

```
[사내 문서]          [문서 로딩]              [청킹]                  [임베딩]            [저장]
PDF/DOCX/   →  DirectoryLoader     →  RecursiveCharacter   →  OpenAI          →  ChromaDB
HTML/PPTX       + 형식별 로더           TextSplitter             text-embedding     (→ Qdrant)
                                        (800 토큰/200 overlap)   -3-small
                                              ↓
                                    메타데이터 추가
                                    (부서, 작성일, 문서 유형)
```

**인퍼런스 파이프라인 (Inference Pipeline)**

```
[사용자 질문]  →  [하이브리드 검색]       →  [리랭킹]      →  [프롬프트 구성]  →  [LLM 생성]
                  EnsembleRetriever       Cohere Rerank      컨텍스트 + 질문     GPT-4o-mini
                  (BM25 + 벡터, top_k=20) top_n=5            + 시스템 지시       + 출처 표기
```

이 구조가 [Advanced RAG](03-advanced-rag-검색-전후-최적화-전략.md)에서 배운 Pre-Retrieval(하이브리드 검색) + Post-Retrieval(리랭킹) 최적화를 적용한 것임을 알 수 있죠. 이제 이 다이어그램을 코드로 구현해봅시다.

## 실습: 직접 해보기

이제 요구사항 분석과 컴포넌트 선택을 마쳤으니, 실제 코드로 아키텍처를 구현합니다. 이 실습은 설계 결정이 코드에 어떻게 반영되는지를 보여주는 것이 핵심입니다.

먼저 필요한 패키지를 설치합니다.

```python
# pip install langchain langchain-openai langchain-community langchain-chroma
# pip install chromadb rank-bm25 python-dotenv pypdf docx2txt cohere
```

> 📌 **패키지 안내**: `langchain-chroma`는 `langchain_community.vectorstores.Chroma`를 대체하는 독립 패키지입니다. 이전 세션(2.1~2.4)에서는 `from langchain_community.vectorstores import Chroma`를 사용했지만, LangChain 생태계의 패키지 분리 방침에 따라 Chroma는 `langchain-chroma`로 독립되었습니다. 기존 코드의 `langchain_community.vectorstores.Chroma`도 동작하지만, 신규 프로젝트에서는 `langchain_chroma`를 사용하는 것을 권장합니다.

**Step 1: 설정과 컴포넌트 초기화**

```python
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# langchain-chroma 패키지 사용 (langchain_community.vectorstores.Chroma의 후속)
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    BSHTMLLoader,
    DirectoryLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ─── 설계 결정 반영: 컴포넌트 설정 ───

# 임베딩 모델: text-embedding-3-small (가성비 + 한국어 지원)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

# 벡터 DB: ChromaDB (프로토타입 단계)
vectorstore = Chroma(
    collection_name="company_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# LLM: GPT-4o-mini (비용 효율 + 충분한 품질)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # 사실 기반 QA에는 낮은 temperature
)

# 청킹: 설계 결정대로 800 토큰, 200 오버랩
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],  # 한국어 문장 경계 고려
)
```

**Step 2: 인제스천 파이프라인 구현**

```python
from langchain_core.documents import Document
from pathlib import Path
from datetime import datetime

def ingest_documents(doc_dir: str) -> tuple[list[Document], int]:
    """설계대로 문서를 로드, 청킹, 임베딩하여 벡터 DB에 저장

    Returns:
        (chunks, chunk_count) — 청크 리스트와 총 청크 수.
        청크 리스트는 이후 BM25 리트리버 구성에 재사용합니다.
    """

    # 형식별 로더 매핑 — 설계 결정: DirectoryLoader + 형식별 로더
    loader_map = {
        "**/*.pdf": PyPDFLoader,
        "**/*.docx": Docx2txtLoader,
        "**/*.html": BSHTMLLoader,
    }

    all_docs: list[Document] = []

    for glob_pattern, loader_cls in loader_map.items():
        loader = DirectoryLoader(
            doc_dir,
            glob=glob_pattern,
            loader_cls=loader_cls,
            show_progress=True,
        )
        docs = loader.load()
        all_docs.extend(docs)

    # 메타데이터 강화 — 문서별 부가 정보 추가
    for doc in all_docs:
        source = doc.metadata.get("source", "")
        doc.metadata["doc_type"] = Path(source).suffix.lstrip(".")
        doc.metadata["ingested_at"] = datetime.now().isoformat()

    # 청킹 — 설계 결정: RecursiveCharacterTextSplitter
    chunks = text_splitter.split_documents(all_docs)

    # 벡터 DB에 저장
    vectorstore.add_documents(chunks)

    return chunks, len(chunks)

# 사용 예시 (실제 문서 디렉토리 경로 지정)
# chunks, total_chunks = ingest_documents("./company_docs")
# print(f"총 {total_chunks}개 청크 인제스천 완료")
```

**Step 3: 인퍼런스 파이프라인 구현 — LCEL로 조합하기**

[Modular RAG](04-modular-rag-유연한-모듈-조합-아키텍처.md)에서 배운 LCEL 패턴을 활용하여, 설계한 인퍼런스 파이프라인을 선언적으로 구현합니다. 설계 결정에서 선택한 **하이브리드 검색(벡터 + BM25)**을 `EnsembleRetriever`로 구현합니다.

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# ─── 검색기 설정: 하이브리드 검색 (설계 결정 반영) ───

# 1) 벡터 검색기 — 의미 기반 검색
vector_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20},
)

# 2) BM25 검색기 — 키워드 기반 검색
#    인제스천 시 반환된 chunks를 사용합니다.
#    (실제 환경에서는 chunks를 직렬화해두거나 DB에서 로드)
# bm25_retriever = BM25Retriever.from_documents(chunks)
# bm25_retriever.k = 20

# 3) 하이브리드 검색기 — BM25 + 벡터를 가중 결합
#    Dense(벡터)에 0.7, Sparse(BM25)에 0.3 가중치를 부여합니다.
#    한국어 사내 문서는 의미 검색이 더 중요하되, 제품명/사번 등
#    정확한 키워드 매칭이 필요한 경우 BM25가 보완합니다.
# hybrid_retriever = EnsembleRetriever(
#     retrievers=[bm25_retriever, vector_retriever],
#     weights=[0.3, 0.7],
# )
```

> 📌 **BM25Retriever**는 인메모리 방식으로 동작하므로, 인제스천 시 생성한 청크 리스트를 재사용해야 합니다. 프로덕션에서는 청크를 직렬화하거나, 문서 수가 많을 경우 Elasticsearch/OpenSearch 같은 외부 검색 엔진의 BM25를 활용하는 방법도 있습니다.

아래는 하이브리드 검색기가 준비된 상태에서의 전체 파이프라인입니다. `hybrid_retriever` 자리에 원하는 검색기를 교체할 수 있습니다.

```python
# ─── 프롬프트 템플릿: 출처 표기 + 한국어 응답 ───

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 사내 문서 기반 QA 어시스턴트입니다.
다음 규칙을 반드시 따르세요:
1. 제공된 컨텍스트에 기반해서만 답변하세요.
2. 컨텍스트에 없는 정보는 "해당 정보를 찾을 수 없습니다"라고 답하세요.
3. 답변 끝에 참고한 문서 출처를 표기하세요.
4. 한국어로 답변하세요."""),
    ("human", """컨텍스트:
{context}

질문: {question}

답변:"""),
])

# ─── 유틸리티 함수 ───

def format_docs(docs: list[Document]) -> str:
    """검색된 문서를 프롬프트에 삽입할 형식으로 변환"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "알 수 없음")
        formatted.append(f"[문서 {i}] (출처: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

# ─── LCEL 체인 조합: 인퍼런스 파이프라인 ───
# hybrid_retriever 대신 vector_retriever를 사용하면 벡터 검색만으로 동작합니다.

rag_chain = (
    RunnableParallel(
        context=vector_retriever | format_docs,    # 검색 → 포맷팅
        question=RunnablePassthrough(),             # 질문 그대로 전달
    )
    | prompt                                        # 프롬프트 구성
    | llm                                           # LLM 생성
    | StrOutputParser()                             # 문자열 파싱
)

# 사용 예시
# answer = rag_chain.invoke("우리 회사의 연차 휴가 정책은 어떻게 되나요?")
# print(answer)
```

다음은 인제스천 결과를 활용한 **하이브리드 검색 전체 설정 예시**입니다.

```python
def build_hybrid_retriever(
    chunks: list[Document],
    vectorstore: Chroma,
    bm25_weight: float = 0.3,
    vector_weight: float = 0.7,
    k: int = 20,
) -> EnsembleRetriever:
    """인제스천된 청크와 벡터스토어로 하이브리드 검색기를 구성합니다."""

    # BM25: 키워드 기반 — 제품명, 사번, 고유명사 등 정확 매칭
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    # Dense: 벡터 기반 — 의미적 유사도
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # EnsembleRetriever: RRF(Reciprocal Rank Fusion)로 두 결과를 결합
    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

# 사용 예시
# chunks, _ = ingest_documents("./company_docs")
# hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
# rag_chain = (
#     RunnableParallel(
#         context=hybrid_retriever | format_docs,
#         question=RunnablePassthrough(),
#     )
#     | prompt | llm | StrOutputParser()
# )
```

**Step 4: 리랭킹 추가 — Advanced RAG 적용**

설계 결정대로 리랭킹을 추가하여 검색 정확도를 높입니다. 하이브리드 검색으로 넓게 후보를 확보한 뒤, 리랭킹으로 상위 문서를 정제하는 2단계 구조입니다.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 리랭커: Cohere Rerank — 설계 결정: 정확도 최우선
reranker = CohereRerank(
    model="rerank-v3.5",
    top_n=5,  # 20개 후보 중 상위 5개만 사용
)

# 압축 리트리버: 하이브리드 검색 → 리랭킹
# base_retriever에 hybrid_retriever를 넣으면 BM25+벡터 → 리랭킹 전체 파이프라인이 됩니다.
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vector_retriever,  # hybrid_retriever로 교체 가능
)

# 리랭킹이 적용된 최종 체인
rag_chain_with_rerank = (
    RunnableParallel(
        context=compression_retriever | format_docs,  # 검색+리랭킹 → 포맷팅
        question=RunnablePassthrough(),
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 사용 예시 — 하이브리드 + 리랭킹 최종 버전
# hybrid_retriever = build_hybrid_retriever(chunks, vectorstore)
# final_retriever = ContextualCompressionRetriever(
#     base_compressor=reranker,
#     base_retriever=hybrid_retriever,
# )
# answer = rag_chain_with_rerank.invoke("신규 프로젝트 승인 절차를 알려주세요")
# print(answer)
```

**Step 5: 아키텍처 검증 — 비용과 지연 시간 추정**

설계가 운영 제약을 만족하는지 사전에 검증하는 것도 설계의 일부입니다.

```run:python
# 비용/지연 추정 계산기
def estimate_costs(
    doc_count: int,
    avg_tokens_per_doc: int,
    queries_per_day: int,
    top_k: int = 20,
    rerank_top_n: int = 5,
) -> dict:
    """월간 운영 비용 추정"""

    # 임베딩 비용 (text-embedding-3-small: $0.02/1M 토큰)
    total_tokens = doc_count * avg_tokens_per_doc
    # 평균 청크 수 = 총 토큰 / 800 (청크 크기)
    chunk_count = total_tokens / 800
    embedding_cost_once = (total_tokens / 1_000_000) * 0.02  # 1회 인제스천
    query_embedding_monthly = (queries_per_day * 30 * 50 / 1_000_000) * 0.02  # 쿼리 임베딩

    # LLM 비용 (GPT-4o-mini: 입력 $0.15/1M, 출력 $0.60/1M)
    avg_context_tokens = rerank_top_n * 800  # 리랭킹 후 상위 N개 청크
    avg_output_tokens = 500  # 평균 응답 길이
    llm_input_monthly = (queries_per_day * 30 * avg_context_tokens / 1_000_000) * 0.15
    llm_output_monthly = (queries_per_day * 30 * avg_output_tokens / 1_000_000) * 0.60

    # 리랭킹 비용 (Cohere: ~$1.00/1000 검색)
    rerank_monthly = (queries_per_day * 30 / 1000) * 1.00

    total_monthly = query_embedding_monthly + llm_input_monthly + llm_output_monthly + rerank_monthly

    return {
        "1회 인제스천 비용": f"${embedding_cost_once:.2f}",
        "월간 쿼리 임베딩": f"${query_embedding_monthly:.2f}",
        "월간 LLM (입력)": f"${llm_input_monthly:.2f}",
        "월간 LLM (출력)": f"${llm_output_monthly:.2f}",
        "월간 리랭킹": f"${rerank_monthly:.2f}",
        "총 예상 월 비용": f"${total_monthly:.2f}",
        "예상 청크 수": f"{chunk_count:,.0f}개",
    }

# 우리 시나리오: 5000건 문서, 일 200건 쿼리
costs = estimate_costs(
    doc_count=5000,
    avg_tokens_per_doc=3000,  # 문서당 평균 3000 토큰
    queries_per_day=200,
)

print("💰 월간 비용 추정")
print("=" * 40)
for key, value in costs.items():
    print(f"  {key}: {value}")

budget = 800.0
total = float(costs["총 예상 월 비용"].replace("$", ""))
print(f"\n  예산: ${budget:.2f}/월")
print(f"  여유분: ${budget - total:.2f}/월")
print(f"  ✅ 예산 내 운영 가능" if total < budget else "  ❌ 예산 초과!")
```

```output
💰 월간 비용 추정
========================================
  1회 인제스천 비용: $0.30
  월간 쿼리 임베딩: $0.01
  월간 LLM (입력): $3.60
  월간 LLM (출력): $1.80
  월간 리랭킹: $6.00
  총 예상 월 비용: $11.41
  예상 청크 수: 18,750개

  예산: $800.00/월
  여유분: $788.59/월
  ✅ 예산 내 운영 가능
```

## 더 깊이 알아보기

### RAG 설계 방법론의 탄생 배경

RAG라는 개념은 2020년 Meta(당시 Facebook AI Research)의 Patrick Lewis 등이 발표한 논문 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*에서 처음 제안되었습니다. 원래 논문에서 RAG는 DPR(Dense Passage Retrieval)과 BART 생성 모델을 결합한 단일 아키텍처였죠.

그런데 불과 4년 만에, RAG는 하나의 모델이 아닌 **아키텍처 패러다임**으로 진화했습니다. Yunfan Gao 등의 2023년 서베이 논문 *"Retrieval-Augmented Generation for Large Language Models: A Survey"*에서 Naive → Advanced → Modular의 세 세대 분류를 제시하면서, RAG 시스템 설계가 "어떤 모델을 쓸까?"에서 "어떤 아키텍처를 설계할까?"로 패러다임이 전환되었습니다.

Microsoft가 Azure 아키텍처 센터에 RAG 설계 가이드를 발행한 것도 이런 흐름의 연장선입니다. RAG가 더 이상 연구 프로토타입이 아닌, 엔터프라이즈급 시스템으로 자리잡으면서 체계적 설계 방법론의 필요성이 대두된 것이죠.

### Cloudflare의 프로덕션 RAG 아키텍처

Cloudflare는 자사 Workers 플랫폼 위에서 동작하는 프로덕션 RAG 레퍼런스 아키텍처를 공개했는데, 흥미로운 설계 결정이 있습니다. 인제스천 파이프라인에 **메시지 큐(Queue)**를 배치하여 대량 문서 업로드 시 다운스트림 서비스가 과부하되지 않도록 한 것입니다. 또한 실패 시 자동 재시도 스케줄링을 포함하여 안정성을 확보했죠. "단순히 동작하는 RAG"와 "프로덕션에서 안정적으로 운영되는 RAG"의 차이가 바로 이런 인프라 설계에 있습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "최고 성능의 임베딩 모델을 쓰면 검색 품질이 최고가 된다"
> 임베딩 모델보다 **청킹 전략이 검색 정확도에 더 큰 영향**을 미칩니다. 잘못 잘린 청크는 아무리 좋은 임베딩으로 변환해도 의미가 왜곡됩니다. 비싼 모델을 쓰기 전에, 청킹 품질부터 확인하세요.

> 💡 **알고 계셨나요?**: 리랭킹은 지연 시간을 50~200ms 추가하지만, 검색 정확도를 크게 향상시킵니다. 프로덕션 시스템에서 답변 품질이 중요하다면, 리랭킹은 거의 항상 그 지연만큼의 가치가 있다는 것이 업계의 공통된 견해입니다.

> 🔥 **실무 팁**: 처음부터 완벽한 아키텍처를 설계하려 하지 마세요. **ChromaDB + Naive RAG로 빠르게 프로토타입**을 만들고, 실제 쿼리로 문제점을 파악한 뒤, 해당 문제를 해결하는 컴포넌트를 하나씩 추가하는 것이 가장 효율적입니다. LangChain의 Retriever 인터페이스가 동일하므로 컴포넌트 교체 비용이 낮습니다.

> 🔥 **실무 팁**: 비용 추정을 설계 단계에서 꼭 하세요. LLM API 비용은 트래픽에 비례해서 늘어나는데, 특히 긴 컨텍스트를 매번 보내는 RAG는 예상보다 비용이 빠르게 증가할 수 있습니다. 리랭킹으로 상위 N개를 줄이면 LLM 입력 토큰도 함께 줄어드는 이중 효과가 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 요구사항 분석 | 데이터 특성, 쿼리 특성, 품질 요구, 운영 제약의 4축으로 분류 |
| 컴포넌트 선택 | 로더, 청킹, 임베딩, 벡터 DB, 검색 전략, RAG 패러다임 6가지 결정 |
| 설계 결정 기록(ADR) | 무엇을 선택했는지뿐 아니라 왜 선택했는지를 문서화 |
| 하이브리드 검색 | EnsembleRetriever로 BM25(키워드) + Dense(벡터) 결합, 가중치로 비율 조절 |
| 청킹 > 임베딩 | 청킹 품질이 임베딩 모델보다 검색 정확도에 더 큰 영향 |
| 점진적 진화 | Naive RAG → Advanced RAG → Modular RAG로 단계적 발전 |
| 비용 사전 추정 | 설계 단계에서 월간 운영 비용과 지연 시간을 미리 계산 |
| LCEL 조합 | RunnableParallel + RunnablePassthrough로 선언적 파이프라인 구성 |

## 다음 섹션 미리보기

이것으로 Chapter 2 "RAG 아키텍처"를 마무리합니다. 우리는 [인제스천/인퍼런스 파이프라인](01-rag-파이프라인-전체-구조-ingestion과-inference.md)부터 [Naive RAG](02-naive-rag-기본-패턴과-한계.md), [Advanced RAG](03-advanced-rag-검색-전후-최적화-전략.md), [Modular RAG](04-modular-rag-유연한-모듈-조합-아키텍처.md)까지 이론을 쌓고, 이번 세션에서 실제 설계 실습까지 완성했습니다.

다음 Chapter 3 "[문서 로딩과 파싱](03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md)"에서는 이번 설계에서 "DirectoryLoader + 형식별 로더"로 간단히 넘어갔던 문서 로딩 단계를 본격적으로 파고듭니다. PDF에서 테이블을 추출하는 방법, HTML에서 노이즈를 제거하는 전략, 그리고 수천 건의 문서를 효율적으로 로딩하는 배치 처리까지 — RAG 파이프라인의 첫 번째 관문인 데이터 수집을 깊이 다룹니다.

## 참고 자료

- [Design and Develop a RAG Solution — Microsoft Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide) - RAG 솔루션 설계의 6단계 체계적 접근법과 각 단계별 평가 기준을 제시하는 가이드
- [Cloudflare RAG Reference Architecture](https://developers.cloudflare.com/reference-architecture/diagrams/ai/ai-rag/) - 프로덕션 RAG 시스템의 인제스천/쿼리 파이프라인 레퍼런스 아키텍처와 인프라 설계 다이어그램
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - RAG의 시작점인 원본 논문 (Lewis et al., 2020)
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - Naive/Advanced/Modular RAG 패러다임 분류를 제시한 서베이 논문
- [LangChain RAG Documentation](https://docs.langchain.com/oss/python/langchain/rag) - LangChain의 공식 RAG 가이드와 LCEL 기반 파이프라인 구현 예제
- [RAG Pipeline Explained — Ingestion, Retrieval, and Generation](https://dextralabs.com/blog/rag-pipeline-explained-diagram-implementation/) - RAG 파이프라인의 단계별 다이어그램과 구현 해설

---
### 🔗 Related Sessions
- [ingestion_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [inference_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [naive_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/02-naive-rag-기본-패턴과-한계.md) (prerequisite)
- [advanced_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [reranking](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [contextual_compression](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
- [modular_rag](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/04-modular-rag-유연한-모듈-조합-아키텍처.md) (prerequisite)
