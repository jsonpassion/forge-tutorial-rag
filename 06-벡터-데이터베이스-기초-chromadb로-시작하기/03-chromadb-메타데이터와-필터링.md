# ChromaDB 메타데이터와 필터링

> 벡터 검색에 구조화된 조건을 더해, 원하는 문서만 정확히 찾아내는 메타데이터 필터링을 마스터합니다.

## 개요

이 섹션에서는 ChromaDB의 메타데이터 시스템을 설계하고, 다양한 필터링 연산자를 활용하여 벡터 검색 결과를 정밀하게 제어하는 방법을 학습합니다. 단순한 유사도 검색을 넘어, 날짜·카테고리·작성자 등 **구조화된 조건**과 벡터 검색을 결합하는 실전 패턴을 익힙니다.

**선수 지식**: [6.2 ChromaDB 시작하기](ch06/session_6_2.md)에서 배운 컬렉션 생성, `add`, `query`, `get` 메서드 사용법
**학습 목표**:
- 효과적인 메타데이터 스키마를 설계할 수 있다
- `where` 필터의 비교·논리·포함 연산자를 자유롭게 활용할 수 있다
- `where_document` 필터로 문서 본문 기반 검색을 수행할 수 있다
- 벡터 유사도 검색과 메타데이터 필터링을 결합한 고급 쿼리를 작성할 수 있다

## 왜 알아야 할까?

[6.1 벡터 데이터베이스란](ch06/session_6_1.md)에서 배운 것처럼, 벡터 데이터베이스의 핵심은 유사도 검색입니다. 하지만 실제 서비스에서는 "의미적으로 비슷한 문서"만으로는 부족한 경우가 많습니다.

예를 들어 기술 블로그 검색 시스템을 만든다고 해보죠. 사용자가 "Python 비동기 처리"를 검색하면 관련 글이 쏟아지는데, 그 중에는 2018년에 쓰인 오래된 글도 있고, 초급자용 튜토리얼도 있고, 고급 성능 최적화 글도 있습니다. 사용자가 원하는 건 **"2024년 이후, 중급 이상, Python 카테고리"** 글인데, 순수 벡터 검색만으로는 이런 조건을 걸 수 없거든요.

바로 이 지점에서 **메타데이터 필터링**이 빛을 발합니다. 벡터 검색의 "의미적 유사성"에 메타데이터 필터의 "구조적 정확성"을 결합하면, 사용자가 진정으로 원하는 결과를 훨씬 정확하게 전달할 수 있습니다. 실무에서 RAG 시스템의 검색 품질을 한 단계 끌어올리는 핵심 기법이죠.

## 핵심 개념

### 개념 1: 메타데이터 스키마 설계 — 벡터에 이름표 붙이기

> 💡 **비유**: 도서관의 책을 생각해보세요. 모든 책은 내용(본문)이 있지만, 그것만으로는 원하는 책을 찾기 어렵습니다. 그래서 도서관은 책마다 **분류 스티커**를 붙이죠 — 저자, 출판년도, 장르, 대출 가능 여부 등. 메타데이터는 바로 이 분류 스티커와 같습니다. 벡터(책 내용)는 "의미"를 담고, 메타데이터(스티커)는 "속성"을 담습니다.

ChromaDB에서 메타데이터는 각 문서에 첨부하는 **키-값 딕셔너리**입니다. 지원하는 값 타입은 다음과 같습니다:

| 타입 | 예시 | 용도 |
|------|------|------|
| `str` | `"python"`, `"advanced"` | 카테고리, 태그, 작성자 |
| `int` | `2024`, `42` | 연도, 페이지 수, 조회수 |
| `float` | `4.5`, `0.95` | 평점, 신뢰도 점수 |
| `bool` | `True`, `False` | 공개 여부, 검증 완료 여부 |
| `list` (배열) | `["NLP", "RAG"]` | 다중 태그, 저자 목록 (1.5.0+) |

스키마를 설계할 때 핵심 원칙이 있습니다:

```python
# 좋은 메타데이터 설계 예시
metadata = {
    "source": "tech_blog",        # str: 문서 출처
    "category": "python",          # str: 주제 분류
    "difficulty": "intermediate",  # str: 난이도
    "year": 2025,                  # int: 작성 연도
    "rating": 4.7,                 # float: 평점
    "is_verified": True,           # bool: 검증 여부
    "tags": ["RAG", "LangChain"],  # list: 다중 태그 (Chroma 1.5.0+)
}
```

> ⚠️ **흔한 오해**: "메타데이터에 아무 구조나 넣어도 된다"고 생각하기 쉽지만, ChromaDB는 **컬렉션 수준의 스키마를 강제하지 않습니다**. 같은 컬렉션 안에서 문서 A는 `year` 필드가 있고, 문서 B는 없을 수 있어요. 이런 불일치는 필터링 시 예상치 못한 결과를 만들 수 있으므로, **애플리케이션 레벨에서 Pydantic 등으로 스키마를 검증**하는 것이 좋습니다.

주의할 점도 있습니다:

- **배열의 타입 일관성**: 배열 안의 모든 원소는 같은 타입이어야 합니다. `["RAG", 42]`는 불가능
- **빈 배열 금지**: `[]`는 허용되지 않습니다
- **중첩 불가**: `{"author": {"name": "Kim", "age": 30}}` 같은 중첩 딕셔너리는 지원하지 않습니다
- **메타데이터 수정은 전체 덮어쓰기**: 특정 키만 업데이트하려면 기존 메타데이터를 먼저 조회한 후 병합해야 합니다

### 개념 2: where 필터 — 메타데이터 조건 검색

> 💡 **비유**: SQL의 `WHERE` 절을 떠올려보세요. `SELECT * FROM books WHERE year > 2023 AND category = 'python'` — 이것과 거의 같은 일을 벡터 DB에서 하는 거예요. 다만 벡터 DB에서는 이 조건 필터링이 **유사도 검색과 동시에** 일어난다는 점이 다릅니다.

`where` 파라미터는 `query()`와 `get()` 메서드 모두에서 사용할 수 있습니다. 연산자 체계를 살펴보겠습니다.

**비교 연산자:**

| 연산자 | 의미 | 지원 타입 | 예시 |
|--------|------|-----------|------|
| `$eq` | 같음 (기본값) | str, int, float, bool | `{"category": {"$eq": "python"}}` |
| `$ne` | 같지 않음 | str, int, float, bool | `{"category": {"$ne": "java"}}` |
| `$gt` | 초과 | int, float | `{"year": {"$gt": 2023}}` |
| `$gte` | 이상 | int, float | `{"rating": {"$gte": 4.0}}` |
| `$lt` | 미만 | int, float | `{"year": {"$lt": 2020}}` |
| `$lte` | 이하 | int, float | `{"page_count": {"$lte": 100}}` |

**포함 연산자:**

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `$in` | 목록에 포함 | `{"category": {"$in": ["python", "rust"]}}` |
| `$nin` | 목록에 미포함 | `{"status": {"$nin": ["draft", "archived"]}}` |
| `$contains` | 배열이 값을 포함 | `{"tags": {"$contains": "RAG"}}` |
| `$not_contains` | 배열이 값을 미포함 | `{"tags": {"$not_contains": "deprecated"}}` |

중요한 점이 하나 있는데요, `$eq`는 **기본 연산자**입니다. 즉 아래 두 표현은 완전히 동일합니다:

```python
# 이 두 표현은 동일
where={"category": "python"}
where={"category": {"$eq": "python"}}
```

간단한 사용 예를 보겠습니다:

```run:python
import chromadb

# 인메모리 클라이언트 생성
client = chromadb.EphemeralClient()
collection = client.create_collection("articles")

# 기술 블로그 문서 추가
collection.add(
    ids=["a1", "a2", "a3", "a4", "a5"],
    documents=[
        "Python 비동기 프로그래밍 완벽 가이드",
        "Rust로 웹 서버 만들기",
        "FastAPI와 SQLAlchemy 실전 활용",
        "JavaScript 최신 ES2024 기능 정리",
        "Python RAG 시스템 구축하기",
    ],
    metadatas=[
        {"category": "python", "year": 2025, "difficulty": "advanced", "views": 1500},
        {"category": "rust", "year": 2024, "difficulty": "intermediate", "views": 800},
        {"category": "python", "year": 2025, "difficulty": "intermediate", "views": 2200},
        {"category": "javascript", "year": 2024, "difficulty": "beginner", "views": 3100},
        {"category": "python", "year": 2025, "difficulty": "advanced", "views": 950},
    ],
)

# 1) 단순 비교: Python 카테고리만
results = collection.get(where={"category": "python"})
print("Python 문서:", results["ids"])

# 2) 범위 비교: 조회수 1000 이상
results = collection.get(where={"views": {"$gte": 1000}})
print("조회수 1000+:", results["ids"])

# 3) 포함 연산자: Python 또는 Rust
results = collection.get(where={"category": {"$in": ["python", "rust"]}})
print("Python/Rust:", results["ids"])
```

```output
Python 문서: ['a1', 'a3', 'a5']
조회수 1000+: ['a1', 'a3', 'a4']
Python/Rust: ['a1', 'a2', 'a3', 'a5']
```

### 개념 3: 논리 연산자 — 조건 조합하기

> 💡 **비유**: 온라인 쇼핑몰에서 필터를 거는 것과 같습니다. "브랜드: 나이키 **그리고** 가격: 10만원 이하 **그리고** 색상: 검정 또는 흰색" — 이렇게 여러 조건을 AND/OR로 엮는 것이죠.

`$and`와 `$or` 연산자로 여러 조건을 조합할 수 있습니다:

```run:python
import chromadb

client = chromadb.EphemeralClient()
collection = client.create_collection("tech_posts")

# 데이터 추가
collection.add(
    ids=["p1", "p2", "p3", "p4"],
    documents=[
        "LangChain LCEL 파이프라인 구축",
        "React 서버 컴포넌트 심층 분석",
        "FastAPI 성능 최적화 전략",
        "RAG 평가 프레임워크 비교",
    ],
    metadatas=[
        {"category": "python", "year": 2025, "difficulty": "advanced"},
        {"category": "javascript", "year": 2025, "difficulty": "advanced"},
        {"category": "python", "year": 2024, "difficulty": "intermediate"},
        {"category": "python", "year": 2025, "difficulty": "intermediate"},
    ],
)

# $and: Python이면서 2025년 문서
results = collection.get(
    where={"$and": [
        {"category": "python"},
        {"year": 2025}
    ]}
)
print("Python AND 2025:", results["ids"])

# $or: advanced이거나 2024년 문서
results = collection.get(
    where={"$or": [
        {"difficulty": "advanced"},
        {"year": 2024}
    ]}
)
print("Advanced OR 2024:", results["ids"])

# 중첩 조합: Python이면서 (advanced이거나 2025년)
results = collection.get(
    where={"$and": [
        {"category": "python"},
        {"$or": [
            {"difficulty": "advanced"},
            {"year": 2025}
        ]}
    ]}
)
print("Python AND (Advanced OR 2025):", results["ids"])
```

```output
Python AND 2025: ['p1', 'p4']
Advanced OR 2024: ['p1', 'p2', 'p3']
Python AND (Advanced OR 2025): ['p1', 'p4']
```

### 개념 4: where_document 필터 — 문서 본문 검색

> 💡 **비유**: 메타데이터 필터가 책의 "분류 스티커"로 찾는 거라면, `where_document`는 책을 펼쳐서 **본문에 특정 단어가 있는지** 직접 확인하는 것입니다. Ctrl+F로 단어를 검색하는 것과 비슷하죠.

`where_document` 파라미터는 문서 텍스트 자체를 기준으로 필터링합니다:

| 연산자 | 의미 | 예시 |
|--------|------|------|
| `$contains` | 문자열 포함 | `{"$contains": "Python"}` |
| `$not_contains` | 문자열 미포함 | `{"$not_contains": "deprecated"}` |
| `$regex` | 정규표현식 매칭 | `{"$regex": "v[0-9]+\\.[0-9]+"}` |
| `$not_regex` | 정규표현식 미매칭 | `{"$not_regex": "beta\\|alpha"}` |

```run:python
import chromadb

client = chromadb.EphemeralClient()
collection = client.create_collection("docs")

collection.add(
    ids=["d1", "d2", "d3"],
    documents=[
        "LangChain v1.0에서 RAG 파이프라인을 구축하는 방법",
        "ChromaDB로 벡터 검색 시스템을 만드는 튜토리얼",
        "OpenAI API와 LangChain을 활용한 챗봇 개발",
    ],
    metadatas=[
        {"topic": "rag"},
        {"topic": "vector_db"},
        {"topic": "chatbot"},
    ],
)

# 문서 본문에 "LangChain"이 포함된 문서만
results = collection.get(
    where_document={"$contains": "LangChain"}
)
print("LangChain 포함:", results["ids"])

# 정규표현식: 버전 번호 패턴이 있는 문서
results = collection.get(
    where_document={"$regex": "v[0-9]+\\.[0-9]+"}
)
print("버전 번호 포함:", results["ids"])
```

```output
LangChain 포함: ['d1', 'd3']
버전 번호 포함: ['d1']
```

> ⚠️ **흔한 오해**: `where_document`의 `$contains`는 **대소문자를 구분**합니다. `"langchain"`과 `"LangChain"`은 다른 문자열로 취급되니 주의하세요. 대소문자를 무시하고 싶다면 `$regex`를 활용할 수 있습니다: `{"$regex": "(?i)langchain"}`.

### 개념 5: 벡터 검색 + 필터링 결합 — 진짜 힘이 나오는 순간

> 💡 **비유**: 네이버 쇼핑에서 "운동화"를 검색(= 의미 검색)한 다음, 왼쪽 사이드바에서 "브랜드: 나이키, 가격: 5만~10만원, 배송: 무료배송"을 체크(= 메타데이터 필터)하는 것과 정확히 같은 원리입니다. 검색어의 의미적 관련성과 구조적 조건을 **동시에** 적용하는 거죠.

`query()` 메서드에서 `query_texts`(또는 `query_embeddings`)와 `where`, `where_document`를 **동시에** 사용하면 됩니다:

```python
# 벡터 검색 + 메타데이터 필터 + 문서 필터 결합
results = collection.query(
    query_texts=["비동기 프로그래밍 패턴"],  # 의미적 유사도 검색
    n_results=5,
    where={                                    # 메타데이터 조건
        "$and": [
            {"category": "python"},
            {"year": {"$gte": 2024}},
            {"difficulty": {"$in": ["intermediate", "advanced"]}}
        ]
    },
    where_document={                           # 문서 본문 조건
        "$contains": "async"
    },
)
```

이 쿼리가 실행되면 ChromaDB는 다음 순서로 동작합니다:

1. **필터링 단계**: `where`와 `where_document` 조건에 맞는 문서를 먼저 걸러냄
2. **검색 단계**: 걸러진 문서들 중에서 `query_texts`와 유사도가 높은 순으로 `n_results`개를 반환

이것이 바로 **사전 필터링(Pre-filtering)** 방식인데요, 필터 조건에 맞지 않는 벡터는 아예 유사도 계산에서 제외되므로 검색 정확도와 효율성이 동시에 높아집니다.

## 실습: 직접 해보기

실제 시나리오를 가정한 종합 실습입니다. 기술 문서 검색 시스템을 만들어 보겠습니다.

```run:python
import chromadb

# ─── 1단계: 컬렉션 생성 및 데이터 준비 ───
client = chromadb.EphemeralClient()
collection = client.create_collection(
    name="tech_knowledge_base",
    metadata={"description": "기술 문서 검색 시스템"}  # 컬렉션 레벨 메타데이터
)

# 기술 문서 데이터 (실전과 유사한 메타데이터 스키마)
docs = [
    {
        "id": "doc_001",
        "text": "LangChain의 LCEL을 사용하면 파이프 연산자로 RAG 체인을 선언적으로 구성할 수 있습니다. Retriever, Prompt, LLM을 체이닝하여 간결한 코드를 작성합니다.",
        "meta": {"category": "framework", "tool": "langchain", "year": 2025, "difficulty": "intermediate", "is_tutorial": True},
    },
    {
        "id": "doc_002",
        "text": "ChromaDB는 오픈소스 벡터 데이터베이스로, 임베딩 저장과 유사도 검색을 간편하게 제공합니다. Python에서 pip install로 바로 사용할 수 있습니다.",
        "meta": {"category": "database", "tool": "chromadb", "year": 2025, "difficulty": "beginner", "is_tutorial": True},
    },
    {
        "id": "doc_003",
        "text": "RAGAS 프레임워크는 Faithfulness, Answer Relevancy, Context Precision 등의 메트릭으로 RAG 시스템을 자동 평가합니다.",
        "meta": {"category": "evaluation", "tool": "ragas", "year": 2024, "difficulty": "advanced", "is_tutorial": False},
    },
    {
        "id": "doc_004",
        "text": "OpenAI의 text-embedding-3-small 모델은 1536차원 벡터를 생성하며, 이전 ada-002 대비 성능이 크게 향상되었습니다.",
        "meta": {"category": "embedding", "tool": "openai", "year": 2024, "difficulty": "intermediate", "is_tutorial": False},
    },
    {
        "id": "doc_005",
        "text": "FAISS의 IVF 인덱스는 대규모 벡터 검색에서 뛰어난 성능을 보여줍니다. nlist와 nprobe 파라미터로 속도와 정확도를 조절합니다.",
        "meta": {"category": "database", "tool": "faiss", "year": 2024, "difficulty": "advanced", "is_tutorial": False},
    },
    {
        "id": "doc_006",
        "text": "LangChain의 RecursiveCharacterTextSplitter는 문서를 재귀적으로 분할합니다. chunk_size와 chunk_overlap으로 청킹 전략을 조절할 수 있습니다.",
        "meta": {"category": "framework", "tool": "langchain", "year": 2025, "difficulty": "beginner", "is_tutorial": True},
    },
]

# 데이터 일괄 추가
collection.add(
    ids=[d["id"] for d in docs],
    documents=[d["text"] for d in docs],
    metadatas=[d["meta"] for d in docs],
)

print(f"총 {collection.count()}개 문서 추가 완료")

# ─── 2단계: 다양한 필터링 패턴 실습 ───

# 패턴 1: 특정 도구 관련 문서만 검색
print("\n--- 패턴 1: tool이 'langchain'인 문서 ---")
results = collection.get(where={"tool": "langchain"})
for doc_id, doc in zip(results["ids"], results["documents"]):
    print(f"  [{doc_id}] {doc[:40]}...")

# 패턴 2: 범위 + 논리 조합
print("\n--- 패턴 2: 2025년 AND 튜토리얼인 문서 ---")
results = collection.get(
    where={"$and": [
        {"year": {"$gte": 2025}},
        {"is_tutorial": True}
    ]}
)
for doc_id, doc in zip(results["ids"], results["documents"]):
    print(f"  [{doc_id}] {doc[:40]}...")

# 패턴 3: 벡터 검색 + 메타데이터 필터 결합
print("\n--- 패턴 3: 'RAG 파이프라인' 검색 + database 카테고리 제외 ---")
results = collection.query(
    query_texts=["RAG 파이프라인 구축 방법"],
    n_results=3,
    where={"category": {"$ne": "database"}},
)
for doc_id, doc, dist in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
    print(f"  [{doc_id}] (거리: {dist:.4f}) {doc[:40]}...")

# 패턴 4: where + where_document 동시 사용
print("\n--- 패턴 4: framework 카테고리 + 본문에 'RAG' 포함 ---")
results = collection.get(
    where={"category": "framework"},
    where_document={"$contains": "RAG"},
)
for doc_id, doc in zip(results["ids"], results["documents"]):
    print(f"  [{doc_id}] {doc[:40]}...")

# 패턴 5: $in으로 여러 카테고리 한번에
print("\n--- 패턴 5: database 또는 embedding 카테고리 ---")
results = collection.get(
    where={"category": {"$in": ["database", "embedding"]}}
)
for doc_id, meta in zip(results["ids"], results["metadatas"]):
    print(f"  [{doc_id}] tool={meta['tool']}, difficulty={meta['difficulty']}")
```

```output
총 6개 문서 추가 완료

--- 패턴 1: tool이 'langchain'인 문서 ---
  [doc_001] LangChain의 LCEL을 사용하면 파이프 연산자로 RAG 체인...
  [doc_006] LangChain의 RecursiveCharacterTextSplitter는 문...

--- 패턴 2: 2025년 AND 튜토리얼인 문서 ---
  [doc_001] LangChain의 LCEL을 사용하면 파이프 연산자로 RAG 체인...
  [doc_002] ChromaDB는 오픈소스 벡터 데이터베이스로, 임베딩 저장과 유사...
  [doc_006] LangChain의 RecursiveCharacterTextSplitter는 문...

--- 패턴 3: 'RAG 파이프라인' 검색 + database 카테고리 제외 ---
  [doc_001] (거리: 0.8734) LangChain의 LCEL을 사용하면 파이프 연산자로 RAG 체인...
  [doc_003] (거리: 1.0251) RAGAS 프레임워크는 Faithfulness, Answer...
  [doc_006] (거리: 1.1023) LangChain의 RecursiveCharacterTextSplitter는 문...

--- 패턴 4: framework 카테고리 + 본문에 'RAG' 포함 ---
  [doc_001] LangChain의 LCEL을 사용하면 파이프 연산자로 RAG 체인...

--- 패턴 5: database 또는 embedding 카테고리 ---
  [doc_002] tool=chromadb, difficulty=beginner
  [doc_004] tool=openai, difficulty=intermediate
  [doc_005] tool=faiss, difficulty=advanced
```

## 더 깊이 알아보기

### "빠진 WHERE 절" — 벡터 검색의 아킬레스건

벡터 데이터베이스 초기에는 순수한 유사도 검색만 지원했습니다. 2020년경 Pinecone의 엔지니어들이 이 문제를 **"The Missing WHERE Clause in Vector Search"**(벡터 검색에서 빠진 WHERE 절)이라고 명명하며 업계에 화두를 던졌는데요, 전통적인 관계형 데이터베이스에서는 너무나 당연했던 `WHERE year > 2023 AND category = 'python'` 같은 조건 필터링이 벡터 공간에서는 전혀 자명하지 않았던 겁니다.

왜 어려웠을까요? 벡터 인덱스(HNSW, IVF 등)는 벡터 간의 **기하학적 거리**를 기준으로 구성됩니다. 여기에 "연도가 2024 이상"이라는 조건을 끼워넣으면, 인덱스의 탐색 경로가 왜곡되어 정확도가 급격히 떨어지거나, 사전 필터링 후 남은 벡터가 너무 적어 검색 품질이 나빠지는 문제가 있었죠.

이 문제를 해결하기 위해 두 가지 접근법이 등장했습니다:

- **사후 필터링(Post-filtering)**: 먼저 벡터 검색으로 후보를 넉넉히 뽑은 뒤, 메타데이터 조건으로 걸러냄. 구현은 쉽지만, 필터를 통과하는 문서가 적으면 결과 수가 `n_results`보다 줄어들 수 있음
- **사전 필터링(Pre-filtering)**: 메타데이터 조건으로 먼저 후보를 줄인 뒤, 그 안에서 벡터 검색. 결과 품질은 좋지만, 후보가 너무 줄면 brute-force 검색이 필요해 느려질 수 있음

ChromaDB는 **사전 필터링** 방식을 채택하고 있습니다. 이 덕분에 `n_results=5`를 요청하면 항상 조건에 맞는 5개(존재하는 경우)를 정확히 돌려받을 수 있습니다. 벡터 데이터베이스마다 필터링 전략이 다르니, [7장 벡터 데이터베이스 심화](ch07/)에서 FAISS, Pinecone, Qdrant의 차이를 비교할 때 이 점을 기억해두세요.

### 메타데이터 인덱싱의 진화

초기 벡터 데이터베이스들은 메타데이터를 단순히 각 벡터에 붙인 "태그" 정도로 취급했습니다. 하지만 수백만 건의 문서를 다루는 프로덕션 환경에서는 메타데이터 필터링 자체의 성능도 중요해졌죠. 그래서 최근에는 벡터 인덱스와 별도로 **메타데이터 전용 인덱스**(B-tree, 역인덱스 등)를 병렬로 유지하는 방식이 표준이 되었습니다. ChromaDB도 내부적으로 SQLite를 활용하여 메타데이터를 효율적으로 인덱싱합니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "`where` 필터를 많이 걸수록 검색이 느려진다"고 걱정하는 분들이 있는데, 사실은 반대입니다. 필터가 후보군을 줄여주므로, 적절한 메타데이터 필터링은 오히려 **검색 속도를 향상**시킵니다. 다만, 너무 많은 조건을 `$or`로 열어두면 후보군이 오히려 넓어져 효과가 줄어들 수 있습니다.

> 💡 **알고 계셨나요?**: ChromaDB의 메타데이터 배열 지원(`$contains`, `$not_contains`)은 버전 1.5.0(2025년)에서 추가된 비교적 최신 기능입니다. 그 이전에는 다중 태그를 저장하려면 `"tags": "RAG,LangChain"` 같은 문자열로 저장하고 `where_document`의 `$contains`로 우회해야 했습니다. 이제는 배열 타입을 직접 사용할 수 있으니 훨씬 깔끔하죠.

> 🔥 **실무 팁**: 메타데이터 스키마는 초기에 잘 설계해야 합니다. 나중에 스키마를 변경하면 기존 문서를 모두 업데이트해야 하는데, ChromaDB는 메타데이터 수정 시 **전체 덮어쓰기**를 하므로 대규모 마이그레이션이 부담스러울 수 있습니다. 프로젝트 시작 전에 필터링에 필요한 속성을 미리 정리하고, Pydantic 모델로 스키마를 정의해두는 것을 추천합니다:
>
> ```python
> from pydantic import BaseModel
>
> class DocMetadata(BaseModel):
>     category: str
>     year: int
>     difficulty: str
>     is_tutorial: bool = False
> ```

> 🔥 **실무 팁**: `$nin` 연산자는 해당 키가 **아예 없는** 문서도 결과에 포함시킵니다. 예를 들어 `where={"status": {"$nin": ["draft"]}}` 는 `status`가 `"draft"`가 아닌 문서뿐 아니라, `status` 키 자체가 없는 문서도 반환합니다. 의도치 않은 결과를 피하려면 스키마 일관성을 유지하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 메타데이터 타입 | `str`, `int`, `float`, `bool`, 배열 (동일 타입 원소만) |
| `where` 비교 연산자 | `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte` |
| `where` 포함 연산자 | `$in`, `$nin`, `$contains`, `$not_contains` |
| `where` 논리 연산자 | `$and`, `$or` (중첩 가능) |
| `where_document` 연산자 | `$contains`, `$not_contains`, `$regex`, `$not_regex` |
| 필터링 방식 | ChromaDB는 사전 필터링(Pre-filtering) — 메타데이터 조건으로 먼저 거른 후 벡터 검색 |
| 스키마 검증 | ChromaDB는 강제하지 않음 → 애플리케이션 레벨에서 Pydantic 등으로 검증 필요 |
| 결합 쿼리 | `query(query_texts=..., where=..., where_document=...)` 로 세 가지 조건을 동시 적용 |

## 다음 섹션 미리보기

메타데이터 필터링으로 검색의 정밀도를 높이는 방법을 배웠으니, 다음 섹션에서는 **LangChain과 ChromaDB를 연동**하는 방법을 알아봅니다. `Chroma` 벡터스토어 클래스를 통해 LangChain의 `Retriever` 인터페이스와 ChromaDB를 매끄럽게 연결하고, 앞서 배운 메타데이터 필터링을 LangChain RAG 체인에서 그대로 활용하는 실전 패턴을 실습합니다.

## 참고 자료

- [Metadata Filtering — Chroma 공식 문서](https://docs.trychroma.com/docs/querying-collections/metadata-filtering) - `where` 필터의 모든 연산자와 사용법을 다룬 공식 레퍼런스
- [Full Text Search — Chroma 공식 문서](https://docs.trychroma.com/docs/querying-collections/full-text-search) - `where_document` 필터와 정규표현식 검색에 대한 공식 가이드
- [Filters — Chroma Cookbook](https://cookbook.chromadb.dev/core/filters/) - 메타데이터 필터와 문서 필터의 실전 예제 모음
- [The Missing WHERE Clause in Vector Search — Pinecone](https://www.pinecone.io/learn/vector-search-filtering/) - 벡터 검색에서 메타데이터 필터링이 왜 어려운지, 사전/사후 필터링의 차이를 설명
- [ChromaDB GitHub Repository](https://github.com/chroma-core/chroma) - ChromaDB 소스코드와 최신 업데이트 확인

---
### 🔗 Related Sessions
- [ephemeralclient](../06-벡터-데이터베이스-기초-chromadb로-시작하기/02-chromadb-시작하기-설치와-기본-사용법.md) (prerequisite)
