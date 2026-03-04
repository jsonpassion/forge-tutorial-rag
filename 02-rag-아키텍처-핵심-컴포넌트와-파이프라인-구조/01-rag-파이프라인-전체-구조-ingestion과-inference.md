# RAG 파이프라인 전체 구조 — Ingestion과 Inference

> 문서를 집어넣는 길과 답을 꺼내는 길, RAG의 두 가지 고속도로를 한눈에 파악합니다.

## 개요

이 세션에서는 RAG 시스템의 전체 데이터 흐름을 두 개의 파이프라인으로 나누어 살펴봅니다. 문서를 벡터로 변환해 저장하는 **인제스천(Ingestion) 파이프라인**과, 사용자 질문에 맞는 정보를 찾아 답변을 생성하는 **인퍼런스(Inference) 파이프라인**의 각 단계별 입출력 형태를 구체적으로 이해합니다.

**선수 지식**: [1장: RAG 개요](../ch01/)에서 배운 LLM의 한계(할루시네이션, 지식 단절)와 RAG가 이를 해결하는 기본 원리
**학습 목표**:
- 인제스천 파이프라인의 4단계(로드 → 분할 → 임베딩 → 저장)를 설명할 수 있다
- 인퍼런스 파이프라인의 4단계(쿼리 → 검색 → 증강 → 생성)를 설명할 수 있다
- 각 단계에서 데이터가 어떤 형태로 변환되는지 파악한다
- 두 파이프라인이 벡터 데이터베이스를 중심으로 어떻게 연결되는지 이해한다

## 왜 알아야 할까?

앞서 1장에서 RAG가 "LLM에게 오픈북 시험을 치르게 하는 것"이라고 배웠죠? 그런데 오픈북 시험을 준비하려면 두 가지가 필요합니다. 먼저 **교재를 정리해서 찾기 쉽게 만드는 작업**(인제스천)이 있고, 시험 당일에 **문제를 보고 교재에서 답을 찾아 쓰는 과정**(인퍼런스)이 있거든요.

실무에서 RAG 시스템을 구축할 때, 이 두 파이프라인의 구조를 모르면 어디서 문제가 생겼는지 진단할 수가 없습니다. "답변이 이상해요"라는 피드백을 받았을 때, 문서 분할이 잘못된 건지, 임베딩 모델이 부적절한 건지, 검색 결과가 부족한 건지, 프롬프트가 문제인지 — 파이프라인 전체를 이해해야 정확히 짚어낼 수 있죠. 이 세션은 RAG 시스템의 **지도**를 그리는 과정입니다.

## 핵심 개념

### 개념 1: 두 개의 파이프라인 — 도서관 비유

> 💡 **비유**: RAG 시스템은 **도서관**과 같습니다. 도서관에는 두 가지 핵심 업무가 있어요. 첫째, 새 책이 들어오면 분류하고 번호를 매겨서 서가에 꽂는 **수서·정리 업무**(인제스천). 둘째, 이용자가 "조선시대 과거제도에 대해 알려주세요"라고 물으면 관련 책을 찾아서 답변을 정리해주는 **참고봉사 업무**(인퍼런스). 이 두 업무는 서로 독립적이지만, **서가(벡터 데이터베이스)**를 공유합니다.

RAG 파이프라인은 크게 **오프라인**으로 실행되는 인제스천과 **온라인**으로 실행되는 인퍼런스로 나뉩니다. 인제스천은 미리 준비해두는 작업이고, 인퍼런스는 사용자 요청이 들어올 때마다 실시간으로 수행됩니다.

```
┌─────────────────────────────────────────────────────┐
│              인제스천 파이프라인 (오프라인)               │
│                                                     │
│  문서 소스 → 로더 → 텍스트 분할 → 임베딩 → 벡터 DB     │
│  (PDF,Web)  (Load)  (Split)    (Embed)  (Store)     │
└──────────────────────────┬──────────────────────────┘
                           │ 벡터 DB (공유)
┌──────────────────────────┴──────────────────────────┐
│              인퍼런스 파이프라인 (온라인)                │
│                                                     │
│  사용자 쿼리 → 임베딩 → 유사도 검색 → 프롬프트 증강 → LLM │
│  (Query)     (Embed)  (Retrieve)   (Augment)  (Generate) │
└─────────────────────────────────────────────────────┘
```

핵심은 두 파이프라인 모두 **같은 임베딩 모델**을 사용한다는 점입니다. 인제스천에서 문서를 벡터로 변환할 때 쓴 모델과, 인퍼런스에서 쿼리를 벡터로 변환할 때 쓴 모델이 동일해야 같은 벡터 공간에서 유사도를 비교할 수 있거든요.

### 개념 2: 인제스천 파이프라인 — 4단계 상세 흐름

> 💡 **비유**: 인제스천은 **요리 재료 손질**과 같습니다. 시장에서 재료를 사오고(로드), 먹기 좋은 크기로 썰고(분할), 양념해서(임베딩), 냉장고에 넣어두는(저장) 거예요. 나중에 요리할 때(인퍼런스) 바로 꺼내 쓸 수 있도록요.

**1단계: 문서 로딩(Loading)**

다양한 형태의 원본 데이터를 읽어들입니다. PDF, 웹페이지, CSV, Notion, Confluence 등 실무에서 만나는 거의 모든 포맷을 다룹니다.

- **입력**: 원본 파일(PDF, HTML, DOCX, CSV 등)
- **출력**: `Document` 객체 리스트 (텍스트 + 메타데이터)

```python
from langchain_community.document_loaders import PyPDFLoader

# PDF 파일을 Document 객체 리스트로 변환
loader = PyPDFLoader("company_report.pdf")
documents = loader.load()

# Document 객체 구조: page_content(텍스트) + metadata(출처 정보)
print(type(documents[0]))  # <class 'langchain_core.documents.Document'>
```

**2단계: 텍스트 분할(Splitting/Chunking)**

로딩된 문서를 검색에 적합한 작은 단위(청크)로 나눕니다. 임베딩 모델의 토큰 제한(예: `text-embedding-3-small`은 8,191토큰)도 있지만, 더 중요한 이유는 **의미 단위로 잘게 나눠야 정확한 검색이 가능**하기 때문입니다.

- **입력**: `Document` 객체 리스트
- **출력**: 더 작은 `Document` 객체 리스트 (각 청크가 하나의 Document)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 500자 단위로 분할, 50자 겹침(overlap)으로 문맥 유지
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # 각 청크의 최대 문자 수
    chunk_overlap=50,     # 인접 청크 간 겹치는 문자 수
    separators=["\n\n", "\n", ".", " "]  # 분할 우선순위
)
chunks = splitter.split_documents(documents)
```

**3단계: 임베딩(Embedding)**

텍스트 청크를 고차원 숫자 벡터로 변환합니다. 이 벡터는 텍스트의 **의미를 수치화**한 것으로, 비슷한 의미의 텍스트는 벡터 공간에서 가까이 위치하게 됩니다.

- **입력**: 텍스트 문자열
- **출력**: 부동소수점 숫자 배열 (예: 1,536차원 벡터)

```run:python
# 임베딩 벡터의 구조를 이해하기 위한 예시
# 실제로는 OpenAI, HuggingFace 등의 임베딩 모델을 사용합니다

import random
random.seed(42)

# 실제 임베딩 벡터는 이런 형태 (여기서는 축약)
sample_vector = [random.uniform(-1, 1) for _ in range(5)]
print(f"임베딩 벡터 예시 (5차원 축약): {[round(v, 4) for v in sample_vector]}")
print(f"실제 text-embedding-3-small 차원 수: 1,536")
print(f"실제 text-embedding-3-large 차원 수: 3,072")
```

```output
임베딩 벡터 예시 (5차원 축약): [0.2576, -0.2214, 0.208, -0.1653, 0.5765]
실제 text-embedding-3-small 차원 수: 1,536
실제 text-embedding-3-large 차원 수: 3,072
```

**4단계: 벡터 저장(Storing)**

생성된 벡터를 원본 텍스트, 메타데이터와 함께 벡터 데이터베이스에 저장합니다. 이 단계를 거치면 유사도 검색이 가능한 상태가 됩니다.

- **입력**: 벡터 + 원본 텍스트 + 메타데이터
- **출력**: 검색 가능한 벡터 인덱스

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 임베딩 모델 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 청크를 벡터로 변환하고 ChromaDB에 저장 (한 번에 처리)
# 인메모리 모드: 실습 및 프로토타이핑에 적합
# 영구 저장이 필요하면 chromadb.PersistentClient를 사용하세요 (ChromaDB v0.4+)
vectorstore = Chroma.from_documents(
    documents=chunks,        # 분할된 청크들
    embedding=embeddings,    # 임베딩 모델
)
```

### 개념 3: 인퍼런스 파이프라인 — 4단계 상세 흐름

> 💡 **비유**: 인퍼런스는 **오픈북 시험**과 같습니다. 문제를 읽고(쿼리), 교재에서 관련 내용을 찾고(검색), 찾은 내용을 답안지에 정리하고(증강), 최종 답변을 작성하는(생성) 과정이죠.

**1단계: 쿼리 임베딩(Query Embedding)**

사용자의 질문을 인제스천 때와 **동일한 임베딩 모델**로 벡터로 변환합니다.

- **입력**: 사용자 질문 문자열
- **출력**: 쿼리 벡터 (인제스천과 같은 차원)

**2단계: 유사도 검색(Retrieval)**

쿼리 벡터와 저장된 문서 벡터 간의 유사도를 계산하여, 가장 관련성 높은 청크를 찾아냅니다. 코사인 유사도(Cosine Similarity)가 가장 널리 쓰이는 방식입니다.

- **입력**: 쿼리 벡터
- **출력**: 상위 k개의 관련 `Document` 객체 리스트

**3단계: 프롬프트 증강(Augmentation)**

검색된 청크를 사용자 질문과 함께 LLM 프롬프트에 삽입합니다. 이것이 바로 RAG의 "Augmented" — 증강 — 에 해당하는 핵심 단계입니다.

- **입력**: 사용자 질문 + 검색된 청크들
- **출력**: 컨텍스트가 포함된 완성 프롬프트

```run:python
# 프롬프트 증강의 원리를 보여주는 예시
query = "RAG 시스템에서 청킹이 중요한 이유는?"

# 검색된 청크 (실제로는 벡터 DB에서 검색)
retrieved_chunks = [
    "청킹은 긴 문서를 작은 단위로 분할하는 과정입니다. 적절한 크기의 청크는 검색 정확도를 높이고...",
    "임베딩 모델은 토큰 제한이 있어 긴 문서를 한 번에 처리할 수 없습니다. 또한 짧은 텍스트가..."
]

# 증강된 프롬프트 구성
augmented_prompt = f"""다음 컨텍스트를 참고하여 질문에 답변하세요.

컨텍스트:
{chr(10).join(f'[{i+1}] {chunk}' for i, chunk in enumerate(retrieved_chunks))}

질문: {query}
답변:"""

print(augmented_prompt)
```

```output
다음 컨텍스트를 참고하여 질문에 답변하세요.

컨텍스트:
[1] 청킹은 긴 문서를 작은 단위로 분할하는 과정입니다. 적절한 크기의 청크는 검색 정확도를 높이고...
[2] 임베딩 모델은 토큰 제한이 있어 긴 문서를 한 번에 처리할 수 없습니다. 또한 짧은 텍스트가...

질문: RAG 시스템에서 청킹이 중요한 이유는?
답변:
```

**4단계: 응답 생성(Generation)**

LLM이 증강된 프롬프트를 입력받아 최종 답변을 생성합니다. 검색된 컨텍스트에 기반하므로 할루시네이션이 크게 줄어들죠.

- **입력**: 증강된 프롬프트
- **출력**: LLM 생성 답변 문자열

### 개념 4: 데이터 형태 변환 추적

각 단계에서 데이터가 어떤 형태로 변환되는지 한눈에 정리해보겠습니다.

| 단계 | 입력 형태 | 출력 형태 | 예시 |
|------|----------|----------|------|
| 로딩 | 파일 (PDF, HTML 등) | `Document(page_content, metadata)` | 10페이지 PDF → Document 10개 |
| 분할 | Document 리스트 | 더 작은 Document 리스트 | Document 10개 → 청크 85개 |
| 임베딩 | 텍스트 문자열 | float 배열 `[0.012, -0.034, ...]` | "안녕하세요" → 1,536차원 벡터 |
| 저장 | 벡터 + 텍스트 + 메타데이터 | 검색 가능한 인덱스 | ChromaDB 컬렉션 |
| 쿼리 임베딩 | 질문 문자열 | float 배열 | "RAG란?" → 1,536차원 벡터 |
| 검색 | 쿼리 벡터 | Document 리스트 (상위 k개) | 쿼리 벡터 → 유사 청크 4개 |
| 증강 | 질문 + 청크들 | 완성된 프롬프트 문자열 | 시스템 프롬프트 + 컨텍스트 + 질문 |
| 생성 | 프롬프트 | 답변 문자열 | LLM 응답 텍스트 |

## 실습: 직접 해보기

아래 코드는 인제스천과 인퍼런스 파이프라인의 전체 흐름을 하나의 스크립트로 구현한 것입니다. 각 단계의 데이터 변환을 직접 확인할 수 있습니다.

```python
"""
RAG 파이프라인 전체 구조 실습
- 인제스천: 텍스트 → 분할 → 임베딩 → 저장
- 인퍼런스: 쿼리 → 검색 → 증강 → 생성
"""
import os
from dotenv import load_dotenv

# 환경 변수에서 API 키 로드 (.env 파일에 OPENAI_API_KEY 설정 필요)
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# 1. 인제스천 파이프라인
# ============================================================

# 1-1. 문서 로딩 (여기서는 예시 텍스트를 직접 생성)
raw_documents = [
    Document(
        page_content="""RAG(Retrieval-Augmented Generation)는 2020년 Meta AI 연구팀의 
        Patrick Lewis가 제안한 기법입니다. LLM이 학습 데이터에 없는 최신 정보나 
        도메인 특화 지식에 접근할 수 있게 해줍니다. 외부 문서를 검색하여 
        LLM의 컨텍스트에 추가함으로써 할루시네이션을 줄이고 답변의 정확도를 높입니다.""",
        metadata={"source": "rag_overview.md", "chapter": 1}
    ),
    Document(
        page_content="""벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 
        유사도 검색을 수행하는 특수 데이터베이스입니다. ChromaDB, FAISS, Pinecone, 
        Qdrant 등이 대표적이며, ANN(Approximate Nearest Neighbor) 알고리즘을 
        사용하여 대규모 데이터에서도 빠른 검색이 가능합니다.""",
        metadata={"source": "vectordb_intro.md", "chapter": 6}
    ),
    Document(
        page_content="""임베딩(Embedding)은 텍스트를 고차원 벡터 공간의 숫자 배열로 
        변환한 것입니다. 의미적으로 유사한 텍스트는 벡터 공간에서 가까이 위치하게 됩니다.
        OpenAI의 text-embedding-3-small 모델은 1,536차원의 벡터를 생성하며, 
        코사인 유사도를 통해 두 텍스트 간의 의미적 거리를 측정할 수 있습니다.""",
        metadata={"source": "embedding_basics.md", "chapter": 5}
    ),
]

print(f"[로딩 완료] 문서 {len(raw_documents)}개 로드")

# 1-2. 텍스트 분할
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 실습을 위해 작은 청크 사이즈 사용
    chunk_overlap=30,     # 문맥 유지를 위한 겹침
)
chunks = splitter.split_documents(raw_documents)
print(f"[분할 완료] {len(raw_documents)}개 문서 → {len(chunks)}개 청크")

# 1-3 & 1-4. 임베딩 + 벡터 저장 (Chroma가 한 번에 처리)
# 인메모리 모드: 실습에서는 별도 저장 경로 없이 메모리에서 동작
# 프로덕션에서 영구 저장이 필요하면 chromadb.PersistentClient를 사용하세요 (ChromaDB v0.4+)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="rag_pipeline_demo",
)
print(f"[저장 완료] {len(chunks)}개 청크가 벡터 DB에 저장됨")

# ============================================================
# 2. 인퍼런스 파이프라인
# ============================================================

# 2-1. 리트리버 생성 (상위 2개 검색)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2-2 & 2-3. 프롬프트 템플릿 (증강 역할)
prompt = ChatPromptTemplate.from_template("""
다음 컨텍스트를 참고하여 질문에 답변하세요.
컨텍스트에 없는 내용은 "제공된 정보에서 확인할 수 없습니다"라고 답하세요.

컨텍스트:
{context}

질문: {question}
""")

# 2-4. LLM 초기화
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 검색된 Document 리스트를 텍스트로 변환하는 헬퍼
def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# LCEL로 전체 인퍼런스 체인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 질문 실행
question = "RAG는 누가 처음 제안했나요?"
answer = rag_chain.invoke(question)
print(f"\n질문: {question}")
print(f"답변: {answer}")

# 정리: 인메모리 벡터스토어 삭제
vectorstore.delete_collection()
```

> 🔥 **실무 팁**: 실습 코드를 실행하려면 `pip install langchain langchain-openai langchain-community chromadb python-dotenv`를 설치하고, `.env` 파일에 `OPENAI_API_KEY=sk-...`를 설정하세요.

## 더 깊이 알아보기

### RAG의 탄생 — Patrick Lewis와 오픈북 시험의 아이디어

RAG라는 이름은 2020년 Meta AI(당시 Facebook AI Research)의 **Patrick Lewis**와 동료들이 발표한 논문 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*에서 처음 등장했습니다. Lewis는 University College London과 New York University 소속으로, "LLM의 파라미터에 모든 지식을 저장하는 것은 비효율적이다"라는 문제의식에서 출발했어요.

당시 이들의 핵심 아이디어는 의외로 단순했습니다. 사전 학습된 seq2seq 모델(BART)을 **파라메트릭(parametric) 메모리**로, 위키피디아 전체를 DPR(Dense Passage Retrieval)로 인덱싱한 것을 **논파라메트릭(non-parametric) 메모리**로 결합한 것이죠. 쉽게 말해, 모델의 내부 기억과 외부 참고서를 동시에 활용하게 한 겁니다.

놀라운 점은 이 간단한 조합이 당시 3개의 오픈 도메인 QA 벤치마크에서 최고 성능을 기록했다는 것입니다. 그리고 이 아키텍처의 뼈대 — "검색 → 증강 → 생성"이라는 3단계 흐름 — 는 2024~2026년 현재의 RAG 시스템에서도 핵심 구조로 그대로 이어지고 있습니다.

이 원조 RAG 아키텍처는 이후 연구자들에 의해 **"Naive RAG"**라는 이름으로 분류되었습니다. 다음 세션에서는 Gao et al.의 서베이 논문을 통해 이 "Naive"라는 이름이 어떤 맥락에서 붙었는지, 그리고 어떤 한계가 Advanced RAG와 Modular RAG로의 발전을 이끌었는지 살펴보겠습니다.

### 인제스천과 인퍼런스의 분리 — 왜 두 개로 나눌까?

오늘날의 RAG 파이프라인이 인제스천과 인퍼런스를 명확히 분리하는 이유는 실용적입니다. 인제스천은 대량의 문서를 처리하는 **배치(batch) 작업**이라 시간이 오래 걸리고 비용이 많이 들지만, 한 번만 하면 됩니다. 반면 인퍼런스는 사용자 요청에 **실시간**으로 응답해야 하므로 속도가 생명이죠. 이 두 가지를 같은 프로세스에서 처리하면 자원 낭비가 심해집니다.

NVIDIA의 RAG 레퍼런스 아키텍처도 이 분리를 강조합니다. 인제스천은 GPU를 활용한 대규모 임베딩 연산에 최적화하고, 인퍼런스는 낮은 레이턴시에 초점을 맞춰 설계하는 것이 프로덕션 RAG의 기본 원칙입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "인제스천과 인퍼런스에서 서로 다른 임베딩 모델을 써도 된다"고 생각하는 분들이 있는데, 이렇게 하면 RAG가 전혀 작동하지 않습니다. 문서 벡터와 쿼리 벡터가 서로 다른 벡터 공간에 놓이기 때문에 유사도 비교 자체가 의미 없어지거든요. **반드시 같은 임베딩 모델을 사용해야 합니다.** 모델을 변경하려면 인제스천부터 다시 수행해야 합니다.

> 💡 **알고 계셨나요?**: RAG의 원래 논문에서는 위키피디아 전체(약 2,100만 문서)를 논파라메트릭 메모리로 사용했습니다. 이 거대한 인덱스를 FAISS로 구축했는데, 검색 시간은 불과 수십 밀리초밖에 걸리지 않았습니다. ANN 알고리즘의 위력을 보여주는 사례죠.

> 🔥 **실무 팁**: 인제스천 파이프라인을 구축할 때 청크 사이즈와 오버랩은 가장 많은 튜닝이 필요한 하이퍼파라미터입니다. 경험적으로 **chunk_size=500~1000**, **chunk_overlap=50~100**에서 시작해보세요. 너무 작으면 문맥이 잘리고, 너무 크면 검색 정밀도가 떨어집니다. [4장: 텍스트 청킹 전략](../ch04/)에서 이 주제를 깊이 다룹니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 인제스천 파이프라인 | 문서를 벡터로 변환해 저장하는 오프라인 전처리 과정 (로드→분할→임베딩→저장) |
| 인퍼런스 파이프라인 | 사용자 질문에 실시간으로 답하는 온라인 처리 과정 (쿼리→검색→증강→생성) |
| 벡터 데이터베이스 | 두 파이프라인의 접점. 인제스천의 결과물을 저장하고 인퍼런스에서 검색하는 공유 저장소 |
| 임베딩 모델 일관성 | 인제스천과 인퍼런스에서 반드시 동일한 임베딩 모델을 사용해야 함 |
| 프롬프트 증강 | 검색된 컨텍스트를 LLM 프롬프트에 삽입하는 단계. RAG의 "A"에 해당 |
| Document 객체 | LangChain의 기본 데이터 단위. page_content(텍스트)와 metadata(출처 정보)로 구성 |

## 다음 세션 미리보기

이번 세션에서는 RAG 파이프라인의 전체 지도를 그려봤습니다. 다음 세션 **"Naive RAG의 구조와 한계"**에서는 지금 배운 기본 파이프라인이 실제로 어떤 한계에 부딪히는지 살펴보고, 이를 개선하기 위해 Advanced RAG와 Modular RAG 패러다임이 어떻게 발전해왔는지 비교합니다.

## 참고 자료

- [RAG 101: Demystifying Retrieval-Augmented Generation Pipelines (NVIDIA)](https://developer.nvidia.com/blog/rag-101-demystifying-retrieval-augmented-generation-pipelines/) - RAG 파이프라인의 각 컴포넌트를 다이어그램과 함께 체계적으로 설명
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (원본 논문)](https://arxiv.org/abs/2005.11401) - Patrick Lewis 등이 발표한 RAG의 원조 논문. 개념의 근원을 이해하려면 필독
- [Cloudflare RAG Reference Architecture](https://developers.cloudflare.com/reference-architecture/diagrams/ai/ai-rag/) - 프로덕션 RAG의 인제스천/쿼리 아키텍처 다이어그램을 명확하게 정리
- [LangChain RAG Documentation](https://docs.langchain.com/oss/python/langchain/rag) - LangChain 기반 RAG 구현의 공식 가이드. 코드 패턴의 최신 레퍼런스
- [RAG Pipeline Explained: Diagram & Implementation (Dextralabs)](https://dextralabs.com/blog/rag-pipeline-explained-diagram-implementation/) - 인제스천, 검색, 생성 3단계를 다이어그램으로 시각화한 실용적 가이드

---
### 🔗 Related Sessions
- [hallucination](../01-rag-개요-llm의-한계와-rag의-필요성/01-llm의-한계-왜-외부-지식이-필요한가.md) (prerequisite)
