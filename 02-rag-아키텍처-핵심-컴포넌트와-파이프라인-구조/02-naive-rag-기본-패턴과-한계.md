# Naive RAG — 기본 패턴과 한계

> 가장 단순한 RAG 구현 패턴인 Naive RAG의 구조를 이해하고, 왜 이것만으로는 부족한지 그 한계를 정확히 파악합니다.

## 개요

이 섹션에서는 RAG의 가장 기본적인 구현 방식인 Naive RAG를 깊이 있게 살펴봅니다. [이전 섹션 2.1: RAG 파이프라인 전체 구조](01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 배운 인제스천(Ingestion)과 인퍼런스(Inference) 파이프라인이 실제로 어떤 형태로 구현되는지 확인하고, 이 단순한 접근 방식이 실전에서 부딪히는 벽이 무엇인지 알아봅니다.

**선수 지식**: 인제스천 파이프라인과 인퍼런스 파이프라인의 개념, Document 객체, 임베딩과 벡터 데이터베이스의 기본 역할 ([세션 2.1](01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 학습)

**학습 목표**:
- Naive RAG의 세 단계(인덱싱, 검색, 생성)를 구체적으로 설명할 수 있다
- Naive RAG의 주요 한계점 5가지를 실제 예시와 함께 이해한다
- LangChain으로 Naive RAG 파이프라인을 직접 구현할 수 있다
- 왜 Advanced RAG가 필요한지 논리적으로 설명할 수 있다

## 왜 알아야 할까?

"RAG를 구현했는데 답변 품질이 기대에 못 미쳐요." — 이런 불만을 가진 개발자가 의외로 많습니다. 대부분의 경우, 그들이 만든 것은 **Naive RAG**입니다. 문서를 넣고, 검색하고, LLM에 던져주는 가장 단순한 형태죠.

Naive RAG를 이해하는 것은 두 가지 이유에서 중요합니다. 첫째, RAG의 **기본 동작 원리**를 가장 명확하게 보여주기 때문입니다. 둘째, 그 **한계를 정확히 알아야** Advanced RAG, Modular RAG 같은 개선 기법이 왜 필요한지 이해할 수 있기 때문입니다. 의사가 정상 상태를 알아야 질병을 진단할 수 있듯이, Naive RAG의 구조와 약점을 파악해야 더 나은 RAG 시스템을 설계할 수 있습니다.

실제로 RAG 관련 서베이 논문(Gao et al., 2024)에서도 RAG의 발전을 **Naive → Advanced → Modular** 세 단계로 구분하며, Naive RAG의 한계를 극복하는 과정이 곧 RAG 기술의 발전사임을 강조하고 있습니다.

## 핵심 개념

### 개념 1: Naive RAG의 세 단계 — 인덱싱, 검색, 생성

> 💡 **비유**: Naive RAG는 **도서관의 신입 사서**와 같습니다. 책을 일정 페이지 수로 잘라서 선반에 꽂고(인덱싱), 질문이 오면 가장 비슷해 보이는 페이지 몇 장을 뽑아오고(검색), 그 페이지만 보고 답변하는(생성) 거죠. 열심히 하지만, 책 전체 맥락을 모르고 기계적으로 일합니다.

Naive RAG는 [세션 2.1](01-rag-파이프라인-전체-구조-ingestion과-inference.md)에서 배운 인제스천/인퍼런스 파이프라인을 **가장 직선적으로(straight-through)** 구현한 것입니다. 중간에 어떤 최적화나 피드백 루프도 없이, 세 단계가 한 방향으로만 흐릅니다.

**1단계: 인덱싱(Indexing)**
- 문서를 로드한다
- 고정 크기로 텍스트를 분할(청킹)한다
- 각 청크를 임베딩 벡터로 변환한다
- 벡터 데이터베이스에 저장한다

**2단계: 검색(Retrieval)**
- 사용자 쿼리를 동일한 임베딩 모델로 벡터화한다
- 벡터 데이터베이스에서 코사인 유사도(Cosine Similarity) 기반으로 상위 k개 청크를 검색한다

**3단계: 생성(Generation)**
- 검색된 청크들을 프롬프트에 그대로 삽입한다
- LLM이 이 컨텍스트와 질문을 기반으로 답변을 생성한다

핵심은 **"그대로"**라는 단어입니다. 청크를 그대로 저장하고, 그대로 검색하고, 그대로 전달합니다. 이 단순함이 장점이자 한계입니다.

```python
# Naive RAG의 전형적인 구조 (의사 코드)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ⚠️ 참고: RetrievalQA는 LangChain 레거시 API입니다.
# 프로덕션에서는 세션 2.1에서 배운 LCEL 패턴을 사용하세요.

# 1단계: 인덱싱
loader = PyPDFLoader("document.pdf")          # 문서 로드
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(     # 고정 크기 분할
    chunk_size=1000, chunk_overlap=200
)
chunks = splitter.split_documents(docs)
vectorstore = Chroma.from_documents(           # 벡터 저장
    chunks, OpenAIEmbeddings()
)

# 2단계 + 3단계: 검색 → 생성 (한 줄로 연결)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)

# 질문 → 답변
result = qa_chain.invoke({"query": "RAG의 핵심 원리는 무엇인가요?"})
```

이 코드가 바로 Naive RAG의 전형적인 모습입니다. 놀라울 정도로 간단하죠? 하지만 이 단순함 속에 여러 문제가 숨어 있습니다.

### 개념 2: Naive RAG의 5가지 한계

> 💡 **비유**: Naive RAG의 한계는 **오픈북 시험에서 발생하는 문제**와 비슷합니다. 교재를 펼쳐볼 수 있지만, 관련 없는 페이지를 펼치거나(낮은 검색 정밀도), 정답이 있는 페이지를 못 찾거나(낮은 재현율), 여러 페이지의 정보를 종합하지 못하거나(컨텍스트 통합 부족), 중간에 있는 중요한 내용을 건너뛰는(Lost in the Middle) 문제가 생기는 것과 같습니다.

#### 한계 1: 낮은 검색 정밀도(Low Precision)

검색된 청크 중 실제로 질문과 관련 있는 것의 비율이 낮습니다. 코사인 유사도만으로는 **의미적 관련성**을 정확히 판단하기 어렵거든요.

```run:python
# 검색 정밀도 문제 시뮬레이션
query = "Python에서 리스트 정렬 방법"
retrieved_chunks = [
    "Python의 sort() 메서드는 리스트를 제자리에서 정렬합니다.",        # 관련 ✅
    "Python은 1991년 귀도 반 로섬이 만든 프로그래밍 언어입니다.",      # 무관 ❌
    "sorted() 함수는 새로운 정렬된 리스트를 반환합니다.",              # 관련 ✅
    "Python의 리스트는 대괄호[]로 생성합니다.",                       # 약간 관련 ⚠️
]

relevant_count = 2  # 실제 관련 있는 청크 수
precision = relevant_count / len(retrieved_chunks)
print(f"검색 정밀도: {precision:.0%}")
print(f"4개 중 {relevant_count}개만 실제로 관련 있음")
print("→ 나머지는 LLM의 컨텍스트 윈도우만 낭비합니다")
```

```output
검색 정밀도: 50%
4개 중 2개만 실제로 관련 있음
→ 나머지는 LLM의 컨텍스트 윈도우만 낭비합니다
```

#### 한계 2: 낮은 검색 재현율(Low Recall)

답변에 필요한 정보를 담은 청크를 놓치는 문제입니다. 고정 크기 청킹으로 인해 핵심 정보가 두 청크에 걸쳐 분리되거나, 쿼리와 표현 방식이 달라 유사도가 낮게 나올 수 있습니다.

```run:python
# 청킹으로 인한 정보 분리 예시
original_text = """
트랜스포머(Transformer)는 2017년 구글이 발표한 아키텍처입니다.
셀프 어텐션(Self-Attention) 메커니즘을 핵심으로 사용하며,
이전의 RNN/LSTM 기반 모델보다 병렬 처리가 뛰어납니다.
트랜스포머의 핵심 구성 요소는 다음과 같습니다:
1. 멀티헤드 어텐션 (Multi-Head Attention)
2. 피드포워드 네트워크 (Feed-Forward Network)
3. 레이어 정규화 (Layer Normalization)
"""

# 고정 크기 청킹 (100자 단위)으로 분할하면...
chunk_size = 100
chunks = [original_text[i:i+chunk_size] for i in range(0, len(original_text), chunk_size)]
for i, chunk in enumerate(chunks):
    print(f"청크 {i+1}: '{chunk.strip()[:50]}...'")
print(f"\n→ '트랜스포머의 핵심 구성 요소'가 여러 청크에 흩어졌습니다!")
print("→ 쿼리: '트랜스포머의 구성 요소'로 검색하면 일부만 검색될 수 있습니다.")
```

```output
청크 1: '트랜스포머(Transformer)는 2017년 구글이 발표한 아키텍처입니다.
셀프 어텐...'
청크 2: '이전의 RNN/LSTM 기반 모델보다 병렬 처리가 뛰어납니다.
트랜스포머의 핵심 구...'
청크 3: '1. 멀티헤드 어텐션 (Multi-Head Attention)
2. 피드포워드 네트워크 (Fee...'
청크 4: '3. 레이어 정규화 (Layer Normalization)...'

→ '트랜스포머의 핵심 구성 요소'가 여러 청크에 흩어졌습니다!
→ 쿼리: '트랜스포머의 구성 요소'로 검색하면 일부만 검색될 수 있습니다.
```

#### 한계 3: Lost in the Middle 문제

2023년 스탠포드의 Nelson Liu 등이 발표한 논문 *"Lost in the Middle"*에서 밝혀진 현상입니다. LLM은 컨텍스트의 **처음과 끝에 있는 정보**는 잘 활용하지만, **중간에 있는 정보는 무시하는 경향**이 있습니다. Naive RAG는 검색된 청크를 단순히 유사도 순으로 나열하기 때문에, 중요한 정보가 중간에 위치할 경우 답변 품질이 떨어집니다.

```python
# Lost in the Middle 문제 시각화
positions = ["1번(처음)", "2번", "3번(중간)", "4번", "5번(끝)"]
attention = ["🟢 높음", "🟡 보통", "🔴 낮음", "🟡 보통", "🟢 높음"]

print("📊 LLM의 컨텍스트 위치별 정보 활용도:")
print("-" * 40)
for pos, att in zip(positions, attention):
    print(f"  청크 {pos}: {att}")
print("-" * 40)
print("→ 3번 청크에 핵심 정보가 있으면 놓칠 수 있습니다!")
```

#### 한계 4: 컨텍스트 통합 부족

Naive RAG는 검색된 여러 청크를 **단순 연결(concatenation)**하여 프롬프트에 넣습니다. 청크 간의 관계나 모순을 고려하지 않죠. 서로 다른 문서에서 온 청크가 상충하는 정보를 담고 있어도, LLM은 이를 구분하기 어렵습니다.

```python
# 상충하는 정보가 들어있는 컨텍스트 예시
retrieved_context = """
[문서 A - 2022년 보고서]
회사의 연간 매출은 500억 원입니다.

[문서 B - 2024년 보고서]  
회사의 연간 매출은 800억 원입니다.
"""

question = "회사의 매출은 얼마인가요?"
# → Naive RAG는 두 문서의 시점 차이를 고려하지 않고 모두 전달
# → LLM이 500억인지 800억인지 혼동할 수 있음
```

#### 한계 5: 쿼리-문서 간 의미 불일치

사용자의 질문과 문서의 표현 방식이 다를 때 검색이 실패합니다. 예를 들어, 사용자가 "메모리 부족 에러"라고 질문했는데 문서에는 "OutOfMemoryError"로만 기록되어 있다면, 임베딩 유사도만으로는 연결하기 어려울 수 있습니다.

```run:python
# 쿼리와 문서 표현 불일치 예시
query_terms = "메모리 부족 에러 해결 방법"
document_terms = "java.lang.OutOfMemoryError: Java heap space"

print(f"사용자 쿼리: '{query_terms}'")
print(f"문서 내용 : '{document_terms}'")
print()
print("→ 같은 문제를 설명하지만 표현이 완전히 다릅니다!")
print("→ 임베딩 유사도가 낮게 나와 검색에서 놓칠 수 있습니다.")
print("→ Advanced RAG에서는 쿼리 변환(Query Transformation)으로 이를 해결합니다.")
```

```output
사용자 쿼리: '메모리 부족 에러 해결 방법'
문서 내용 : 'java.lang.OutOfMemoryError: Java heap space'

→ 같은 문제를 설명하지만 표현이 완전히 다릅니다!
→ 임베딩 유사도가 낮게 나와 검색에서 놓칠 수 있습니다.
→ Advanced RAG에서는 쿼리 변환(Query Transformation)으로 이를 해결합니다.
```

### 개념 3: Naive RAG의 프롬프트 구조

Naive RAG에서 LLM에 전달되는 프롬프트는 매우 단순합니다. 검색된 청크를 그대로 붙여넣고, 사용자 질문을 추가하는 것이 전부입니다. 이른바 **"stuff" 방식**이라고 부릅니다 — 말 그대로 컨텍스트를 "꾸겨 넣는" 거죠.

```python
# Naive RAG의 전형적인 프롬프트 템플릿
from langchain_core.prompts import PromptTemplate

naive_rag_prompt = PromptTemplate.from_template("""
다음 컨텍스트를 사용하여 질문에 답변하세요.
컨텍스트에서 답을 찾을 수 없으면 모른다고 말하세요.

컨텍스트:
{context}

질문: {question}

답변:""")

# 이것이 stuff 방식 — 모든 청크를 하나의 프롬프트에 그대로 삽입
# chain_type="stuff"가 Naive RAG의 기본 설정
```

이 방식은 청크 수가 적고 내용이 명확할 때는 잘 작동합니다. 하지만 청크가 많아지면 컨텍스트 윈도우 제한에 부딪히고, 노이즈가 늘어나면 할루시네이션(Hallucination)의 위험도 커집니다.

## 실습: 직접 해보기

Naive RAG 파이프라인을 처음부터 끝까지 직접 구현해봅시다. 여기서는 간단한 텍스트 데이터를 사용하여 Naive RAG의 동작을 확인하고, 그 한계를 눈으로 확인합니다.

```python
# === Naive RAG 파이프라인 전체 구현 ===
# 필요 패키지: pip install langchain langchain-openai langchain-community chromadb langchain-text-splitters

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

# ⚠️ 참고: RetrievalQA는 LangChain 레거시 API입니다.
# 프로덕션에서는 세션 2.1에서 배운 LCEL 패턴을 사용하세요.

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드

# ============================
# 1단계: 인덱싱 (Indexing)
# ============================

# 샘플 문서 준비 (실제로는 PDF, 웹 등에서 로드)
documents = [
    Document(
        page_content="""
        RAG(Retrieval-Augmented Generation)는 2020년 Facebook AI Research(현 Meta AI)의 
        Patrick Lewis 등이 발표한 기법입니다. LLM의 파라미터에 저장된 지식만으로는 
        최신 정보에 대응하기 어렵다는 문제를 해결하기 위해 외부 지식 소스를 활용합니다.
        RAG는 검색기(Retriever)와 생성기(Generator)를 결합하여 동작합니다.
        """,
        metadata={"source": "rag_overview.txt", "page": 1}
    ),
    Document(
        page_content="""
        벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색하는 특수 데이터베이스입니다.
        대표적으로 ChromaDB, FAISS, Pinecone, Qdrant 등이 있습니다.
        ANN(Approximate Nearest Neighbor) 알고리즘을 사용하여 
        정확도를 약간 희생하는 대신 검색 속도를 크게 향상시킵니다.
        """,
        metadata={"source": "vector_db.txt", "page": 1}
    ),
    Document(
        page_content="""
        임베딩(Embedding)은 텍스트를 숫자 벡터로 변환하는 과정입니다.
        OpenAI의 text-embedding-3-small 모델은 1536차원의 벡터를 생성하며,
        의미적으로 유사한 텍스트는 벡터 공간에서 가까운 위치에 매핑됩니다.
        이를 통해 키워드 매칭이 아닌 의미 기반 검색이 가능해집니다.
        """,
        metadata={"source": "embedding.txt", "page": 1}
    ),
    Document(
        page_content="""
        LLM의 할루시네이션(Hallucination)은 모델이 사실이 아닌 정보를 
        그럴듯하게 생성하는 현상입니다. RAG는 외부 지식을 제공함으로써 
        할루시네이션을 줄이는 효과가 있지만, 검색된 문서의 품질이 낮으면 
        오히려 잘못된 정보를 기반으로 답변할 위험도 있습니다.
        """,
        metadata={"source": "hallucination.txt", "page": 1}
    ),
]

# 텍스트 분할 — Naive RAG는 고정 크기 청킹 사용
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 청크 크기 (문자 수)
    chunk_overlap=50,     # 청크 간 겹침
    separators=["\n\n", "\n", ". ", " ", ""]  # 분할 우선순위
)
chunks = text_splitter.split_documents(documents)
print(f"원본 문서 수: {len(documents)}")
print(f"분할된 청크 수: {len(chunks)}")

# 임베딩 생성 + 벡터 저장소 구축
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="naive_rag_demo"
)
print(f"벡터 저장소 구축 완료! 총 {vectorstore._collection.count()}개 벡터")

# ============================
# 2단계 + 3단계: 검색 + 생성
# ============================

# Naive RAG 체인 구성
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",          # 단순 삽입 방식 (Naive RAG 기본)
    retriever=vectorstore.as_retriever(
        search_type="similarity",  # 코사인 유사도 검색
        search_kwargs={"k": 3}     # 상위 3개 청크 검색
    ),
    return_source_documents=True   # 출처 문서도 반환
)

# ============================
# 질의 테스트
# ============================
queries = [
    "RAG란 무엇인가요?",
    "벡터 데이터베이스의 종류는?",
    "RAG가 할루시네이션을 완전히 해결하나요?",  # 한계를 드러내는 질문
]

for query in queries:
    print(f"\n{'='*50}")
    print(f"질문: {query}")
    print(f"{'='*50}")
    result = qa_chain.invoke({"query": query})
    print(f"답변: {result['result']}")
    print(f"\n[검색된 소스 {len(result['source_documents'])}개]")
    for i, doc in enumerate(result['source_documents']):
        print(f"  {i+1}. {doc.metadata['source']} — '{doc.page_content[:60].strip()}...'")
```

> 🔥 **실무 팁**: 위 코드에서 `chain_type="stuff"`를 주목하세요. 이것이 Naive RAG의 핵심입니다. 검색된 모든 청크를 하나의 프롬프트에 "그대로 밀어넣는" 방식이죠. 청크가 많아지면 컨텍스트 윈도우를 초과할 수 있어서, 실무에서는 `k` 값을 신중하게 설정해야 합니다.

## 더 깊이 알아보기

### RAG의 탄생 — 검색과 생성의 만남

RAG라는 개념은 하루아침에 등장한 것이 아닙니다. 2020년, Facebook AI Research(현 Meta AI)의 **Patrick Lewis**와 동료들은 한 가지 근본적인 질문을 던졌습니다. "LLM의 파라미터에 모든 지식을 담을 수 있을까?"

당시 GPT-3가 1750억 개의 파라미터로 세상을 놀라게 했지만, 여전히 사실관계 오류가 빈번했습니다. Lewis 등은 정보 검색(Information Retrieval) 분야의 오래된 아이디어 — **필요한 정보를 외부에서 가져온다** — 를 신경망 생성 모델과 결합하는 방법을 제안했습니다. 이것이 바로 원조 RAG 논문 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*(NeurIPS 2020)입니다.

흥미로운 점은, 이 논문에서 사용한 검색기가 **DPR(Dense Passage Retrieval)**이고 생성기가 **BART**였다는 것입니다. 지금의 RAG와 비교하면 상당히 단순한 구조인데, 그럼에도 당시 오픈 도메인 QA에서 최고 성능을 달성했습니다. 이 최초의 RAG가 바로 오늘날 우리가 "Naive RAG"라고 부르는 것의 원형입니다.

### "Naive"라는 이름은 어디서 왔을까?

"Naive RAG"라는 용어는 원래 Lewis 논문에 있던 것이 아닙니다. 2023년 Gao 등의 서베이 논문 *"Retrieval-Augmented Generation for Large Language Models: A Survey"*에서 RAG의 발전 단계를 **Naive → Advanced → Modular**로 분류하면서 붙여진 이름입니다. "Naive"는 "순진한, 단순한"이라는 뜻으로, 최적화 없이 가장 기본적인 형태로 구현한 RAG를 의미합니다. 통계학에서 "Naive Bayes"가 독립 가정이라는 단순한 전제에서 출발하듯, Naive RAG도 "검색 결과를 그대로 쓰면 된다"는 단순한 전제에서 출발합니다.

### Lost in the Middle — 스탠포드의 발견

2023년 스탠포드 대학의 **Nelson Liu** 등은 흥미로운 실험을 했습니다. LLM에게 여러 문서를 제공하고, 정답이 포함된 문서의 **위치**를 바꿔가며 정확도를 측정한 것이죠. 결과는 놀라웠습니다. 정답이 컨텍스트의 처음이나 끝에 있을 때는 정확도가 높았지만, **중간에 있으면 성능이 크게 떨어졌습니다**. 이 "Lost in the Middle" 현상은 Naive RAG의 단순한 검색 결과 나열이 왜 문제가 되는지를 잘 보여줍니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "RAG를 쓰면 할루시네이션이 사라진다"고 생각하는 분이 많습니다. 하지만 Naive RAG에서는 오히려 **검색된 잘못된 정보**를 기반으로 할루시네이션이 발생할 수 있습니다. 관련 없는 청크가 검색되면 LLM이 그 내용을 사실로 받아들여 엉뚱한 답변을 생성하거든요. RAG는 할루시네이션을 줄이는 도구이지, 제거하는 마법이 아닙니다.

> 💡 **알고 계셨나요?**: Naive RAG의 검색 정확도는 일반적으로 **25~50%** 수준이라고 알려져 있습니다. 즉, 검색된 4개의 청크 중 실제로 답변에 유용한 것은 1~2개인 셈이죠. Advanced RAG 기법을 적용하면 이 정확도를 **80~90%**까지 끌어올릴 수 있습니다. 이 차이가 바로 다음 세션들에서 배울 내용의 가치입니다.

> 🔥 **실무 팁**: Naive RAG로 프로토타입을 만들 때 `k` 값(검색할 청크 수)을 무작정 높이지 마세요. k를 늘리면 재현율은 올라가지만, 정밀도는 떨어지고 노이즈가 증가합니다. 또한 LLM의 컨텍스트 윈도우 비용도 늘어납니다. 실무에서는 보통 **k=3~5**에서 시작하여 답변 품질을 보면서 조정하는 것이 좋습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| Naive RAG | 인덱싱 → 검색 → 생성의 단순 직선 구조, 중간 최적화 없음 |
| 인덱싱 | 문서 로드 → 고정 크기 청킹 → 임베딩 → 벡터 저장 |
| 검색 | 코사인 유사도 기반 상위 k개 청크 추출 |
| 생성 | stuff 방식으로 청크를 프롬프트에 삽입 후 LLM 답변 생성 |
| 낮은 정밀도 | 검색된 청크 중 무관한 내용이 포함되는 문제 |
| 낮은 재현율 | 필요한 정보가 검색에서 누락되는 문제 |
| Lost in the Middle | LLM이 컨텍스트 중간 위치의 정보를 무시하는 현상 |
| 컨텍스트 통합 부족 | 여러 청크 간 관계/모순을 고려하지 못하는 문제 |
| 의미 불일치 | 쿼리와 문서의 표현 차이로 검색이 실패하는 문제 |

## 다음 섹션 미리보기

Naive RAG의 한계를 확인했으니, 이제 이를 극복하기 위한 방법을 알아볼 차례입니다. 다음 [세션 2.3: Advanced RAG](03-advanced-rag-검색-전후-최적화-전략.md)에서는 **사전 검색(Pre-Retrieval)** 최적화와 **사후 검색(Post-Retrieval)** 최적화를 통해 검색 품질과 생성 품질을 동시에 높이는 기법들을 살펴봅니다. 쿼리 재작성, 리랭킹, 컨텍스트 압축 등 Naive RAG의 각 한계점에 대응하는 구체적인 해결책을 배우게 됩니다.

## 참고 자료

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (원조 RAG 논문, Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) - RAG 개념의 시작점. 검색과 생성을 결합한 최초의 프레임워크를 제안한 논문
- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2024)](https://arxiv.org/abs/2312.10997) - Naive/Advanced/Modular RAG 패러다임을 체계적으로 정리한 서베이 논문
- [Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)](https://arxiv.org/abs/2307.03172) - LLM의 컨텍스트 위치 편향을 실험적으로 밝힌 논문. Naive RAG의 한계를 이해하는 핵심 자료
- [LangChain RAG Documentation](https://docs.langchain.com/oss/python/langchain/rag) - LangChain을 사용한 RAG 파이프라인 구축 공식 문서
- [Prompt Engineering Guide — RAG for LLMs](https://www.promptingguide.ai/research/rag) - Naive RAG부터 Advanced RAG까지의 기법을 정리한 실용적 가이드

---
### 🔗 Related Sessions
- [ingestion_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [inference_pipeline](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_database](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [text_splitting](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/01-rag-파이프라인-전체-구조-ingestion과-inference.md) (prerequisite)
