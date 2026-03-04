# ChromaDB 시작하기 — 설치와 기본 사용법

> ChromaDB를 설치하고, 세 가지 클라이언트 모드를 이해하며, 컬렉션 생성부터 문서 추가·검색까지 핵심 CRUD를 실습합니다.

## 개요

이 섹션에서는 ChromaDB를 실제로 설치하고 사용하는 방법을 처음부터 끝까지 다룹니다. [6.1 벡터 데이터베이스란 — 왜 필요한가](06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md)에서 벡터 데이터베이스의 개념과 ANN 알고리즘을 배웠다면, 이번에는 직접 손으로 코드를 작성하며 ChromaDB의 동작을 체득할 차례입니다.

**선수 지식**: 벡터 데이터베이스의 필요성(정확 매칭 vs 유사도 검색), ANN의 기본 개념, `chromadb.Client()` 기본 사용법(6.1에서 학습)

**학습 목표**:
- ChromaDB의 세 가지 클라이언트 모드(EphemeralClient, PersistentClient, HttpClient)를 구분하고 적절히 선택할 수 있다
- 컬렉션의 생성, 조회, 수정, 삭제 등 CRUD 작업을 수행할 수 있다
- `add()`, `query()`, `get()`, `update()`, `delete()` 메서드로 문서를 관리할 수 있다

## 왜 알아야 할까?

벡터 데이터베이스의 개념을 이해하는 것과 실제로 사용하는 것은 완전히 다른 이야기입니다. RAG 파이프라인을 구축하려면 결국 "어딘가에" 임베딩 벡터를 저장하고 검색해야 하는데, ChromaDB는 그 "어딘가"를 가장 쉽게 만들어주는 도구거든요.

실무에서는 크게 세 가지 시나리오가 반복됩니다:
1. **프로토타이핑** — "이 아이디어가 되는지 빠르게 확인해보자" → 인메모리 모드
2. **로컬 개발** — "데이터를 저장해두고 반복 실험하자" → 영구 저장 모드
3. **팀 협업/배포** — "여러 서비스에서 접근해야 한다" → 클라이언트-서버 모드

이 세 가지를 어떻게 전환하는지, 그리고 데이터를 어떻게 넣고 빼는지를 모르면 RAG의 나머지 부분을 아무리 잘 알아도 결국 시스템을 완성할 수 없습니다.

## 핵심 개념

### 개념 1: 설치 — 한 줄이면 충분합니다

ChromaDB 설치는 놀랍도록 간단합니다. pip 한 줄이면 끝이에요.

```bash
pip install chromadb
```

이 명령어 하나로 로컬 임베디드 모드에 필요한 모든 것이 설치됩니다. 기본 임베딩 함수(`all-MiniLM-L6-v2` 모델)까지 포함되어 있어서, 별도의 API 키 없이도 바로 벡터 검색을 시작할 수 있죠.

> 💡 **알고 계셨나요?**: ChromaDB의 기본 임베딩 모델은 ONNX Runtime 위에서 로컬로 실행됩니다. 즉, OpenAI API 키가 없어도, 인터넷이 끊겨 있어도 임베딩을 생성할 수 있습니다. 384차원의 벡터를 생성하는 `all-MiniLM-L6-v2` 모델이 패키지에 포함되어 있거든요.

설치 확인도 간단합니다:

```run:python
import chromadb

# 버전 확인
print(f"ChromaDB 버전: {chromadb.__version__}")
print("설치 성공!")
```

```output
ChromaDB 버전: 1.5.2
설치 성공!
```

### 개념 2: 세 가지 클라이언트 모드 — 노트, 서랍, 도서관

> 💡 **비유**: ChromaDB의 세 가지 클라이언트 모드를 문서 보관 방식에 비유해볼까요?
> - **EphemeralClient**: 포스트잇에 메모하는 것. 빠르고 간편하지만, 떼어내면 사라집니다.
> - **PersistentClient**: 서랍에 파일을 정리해 넣는 것. 컴퓨터를 꺼도 다시 꺼내 볼 수 있습니다.
> - **HttpClient**: 회사 도서관에 문서를 보관하는 것. 팀원 누구나 접근할 수 있습니다.

#### EphemeralClient — 메모리에서만 동작

데이터를 디스크에 저장하지 않는 순수 인메모리 클라이언트입니다. 프로그램이 종료되면 모든 데이터가 사라지죠. 빠른 프로토타이핑이나 테스트에 최적입니다.

```python
import chromadb

# 인메모리 클라이언트 생성
client = chromadb.EphemeralClient()
```

#### PersistentClient — 디스크에 영구 저장

지정한 경로에 데이터를 저장하는 클라이언트입니다. 프로그램을 다시 실행해도 이전 데이터가 그대로 남아 있습니다. 로컬 개발이나 소규모 애플리케이션에 적합합니다.

```python
import chromadb

# 영구 저장 클라이언트 — path에 데이터가 저장됨
client = chromadb.PersistentClient(path="./chroma_data")
```

`path`에 지정한 디렉토리가 없으면 자동으로 생성됩니다. 다음에 같은 경로로 `PersistentClient`를 만들면 이전 데이터를 그대로 이어서 사용할 수 있어요.

#### HttpClient — 원격 서버 연결

별도로 실행 중인 Chroma 서버에 네트워크로 연결하는 클라이언트입니다. 여러 애플리케이션이 동시에 같은 벡터 데이터베이스에 접근해야 할 때 사용합니다.

```python
import chromadb

# 원격 서버 연결 (기본 포트: 8000)
client = chromadb.HttpClient(host="localhost", port=8000)
```

서버는 별도 터미널에서 `chroma run --path ./chroma_server_data` 명령으로 실행합니다.

어떤 클라이언트를 선택하든 **이후의 API 사용법은 완전히 동일**합니다. 이것이 ChromaDB 설계의 핵심이에요 — 프로토타이핑 단계에서 `EphemeralClient`로 시작하고, 이후 `PersistentClient`나 `HttpClient`로 한 줄만 바꾸면 됩니다.

### 개념 3: 컬렉션(Collection) — 벡터의 서랍장

> 💡 **비유**: 컬렉션은 파일 캐비닛의 서랍 하나와 같습니다. "이력서" 서랍, "계약서" 서랍처럼 관련 문서를 분류해서 넣는 거죠. 각 서랍(컬렉션)은 독립적인 검색 공간을 형성합니다.

#### 컬렉션 생성

```python
import chromadb

client = chromadb.EphemeralClient()

# 새 컬렉션 생성
collection = client.create_collection("my_documents")
```

이미 같은 이름의 컬렉션이 존재하면 에러가 발생합니다. "있으면 가져오고, 없으면 만드는" 안전한 방법도 있어요:

```python
# 있으면 가져오고, 없으면 생성
collection = client.get_or_create_collection("my_documents")
```

#### 컬렉션에 HNSW 설정 커스터마이징

[6.1](06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md)에서 배운 HNSW 파라미터를 컬렉션 생성 시 직접 지정할 수도 있습니다:

```python
# HNSW 인덱스 설정을 커스터마이징한 컬렉션
collection = client.create_collection(
    "optimized_collection",
    configuration={
        "hnsw": {
            "space": "cosine",           # 거리 함수: cosine, l2, ip
            "ef_construction": 200,      # 인덱스 구축 시 탐색 범위
            "max_neighbors": 32,         # 각 노드의 최대 이웃 수
        }
    },
)
```

#### 컬렉션 조회와 관리

```run:python
import chromadb

client = chromadb.EphemeralClient()

# 컬렉션 여러 개 생성
client.create_collection("articles")
client.create_collection("qa_pairs")
client.create_collection("user_feedback")

# 전체 컬렉션 목록 조회
collections = client.list_collections()
print(f"컬렉션 수: {client.count_collections()}")
for col in collections:
    print(f"  - {col.name}")

# 특정 컬렉션 가져오기
articles = client.get_collection("articles")
print(f"\n가져온 컬렉션: {articles.name}")

# 컬렉션 삭제
client.delete_collection("user_feedback")
print(f"삭제 후 컬렉션 수: {client.count_collections()}")
```

```output
컬렉션 수: 3
  - articles
  - qa_pairs
  - user_feedback

가져온 컬렉션: articles
삭제 후 컬렉션 수: 2
```

### 개념 4: 문서 추가 — add()

> 💡 **비유**: `add()`는 서랍에 문서를 넣는 행위입니다. 다만 일반 서랍과 달리, 문서를 넣는 순간 자동으로 색인 카드(임베딩 벡터)가 생성되어 빠른 검색이 가능해집니다.

`collection.add()`의 핵심 파라미터 세 가지를 알아보겠습니다:

```python
collection.add(
    ids=["doc1", "doc2", "doc3"],        # 고유 식별자 (필수)
    documents=["첫 번째 문서", "두 번째 문서", "세 번째 문서"],  # 원본 텍스트
    metadatas=[                           # 부가 정보 (선택)
        {"source": "blog", "year": 2024},
        {"source": "paper", "year": 2025},
        {"source": "blog", "year": 2025},
    ],
)
```

| 파라미터 | 필수 여부 | 설명 |
|----------|----------|------|
| `ids` | 필수 | 각 문서의 고유 식별자. 문자열 리스트 |
| `documents` | 선택* | 원본 텍스트. 지정하면 자동으로 임베딩 생성 |
| `embeddings` | 선택* | 미리 생성한 임베딩 벡터를 직접 전달 |
| `metadatas` | 선택 | 필터링에 사용할 부가 정보 딕셔너리 리스트 |

\* `documents`와 `embeddings` 중 최소 하나는 제공해야 합니다.

> ⚠️ **흔한 오해**: `documents`를 전달하면 ChromaDB가 **자동으로** 임베딩을 생성합니다. "임베딩을 직접 만들어서 넣어야 하나?"라고 걱정할 필요가 없어요. 물론 `embeddings` 파라미터로 직접 넣을 수도 있지만, 시작 단계에서는 기본 임베딩 함수에 맡기는 것이 훨씬 편합니다.

### 개념 5: 검색 — query()

> 💡 **비유**: `query()`는 도서관 사서에게 "이런 내용의 책 찾아주세요"라고 요청하는 것입니다. 사서(ChromaDB)는 당신의 요청을 이해하고(임베딩 생성), 가장 비슷한 문서를 유사도 순으로 정렬해서 돌려줍니다.

```python
results = collection.query(
    query_texts=["검색하고 싶은 내용"],  # 검색 쿼리
    n_results=3,                         # 반환할 결과 수
    include=["documents", "distances", "metadatas"],  # 포함할 필드
)
```

`include` 파라미터로 반환받을 정보를 지정할 수 있습니다:

| include 값 | 설명 |
|------------|------|
| `"documents"` | 원본 텍스트 |
| `"distances"` | 쿼리와의 거리(작을수록 유사) |
| `"metadatas"` | 부가 정보 |
| `"embeddings"` | 임베딩 벡터 (대용량 주의) |

반환되는 `results`는 딕셔너리 형태이며, 각 값은 이중 리스트(쿼리별 결과 리스트)입니다:

```python
# results 구조 예시
{
    "ids": [["doc2", "doc3", "doc1"]],
    "documents": [["두 번째 문서", "세 번째 문서", "첫 번째 문서"]],
    "distances": [[0.12, 0.35, 0.78]],
    "metadatas": [[{"source": "paper"}, {"source": "blog"}, {"source": "blog"}]],
}
```

### 개념 6: 문서 수정과 삭제 — update(), upsert(), delete()

데이터를 넣었으면 수정하고 삭제할 줄도 알아야겠죠?

```python
# 기존 문서 수정 (id가 존재해야 함)
collection.update(
    ids=["doc1"],
    documents=["수정된 첫 번째 문서"],
    metadatas=[{"source": "blog", "year": 2026, "revised": True}],
)

# upsert: 있으면 수정, 없으면 추가
collection.upsert(
    ids=["doc1", "doc4"],
    documents=["다시 수정된 문서", "새로 추가된 문서"],
)

# 특정 문서 삭제
collection.delete(ids=["doc3"])
```

`update()`와 `upsert()`의 차이가 헷갈릴 수 있는데요:
- `update()`: 해당 ID가 **반드시 존재**해야 합니다. 없으면 에러가 발생합니다.
- `upsert()`: 있으면 수정하고, 없으면 새로 추가합니다. **"update + insert"** 의 합성어예요.

> 🔥 **실무 팁**: 문서가 주기적으로 갱신되는 시스템(뉴스, 위키 등)에서는 `upsert()`를 사용하세요. "이미 있나 확인하고 → 있으면 수정, 없으면 추가"라는 두 단계를 한 번에 처리할 수 있습니다.

### 개념 7: 문서 조회 — get(), count(), peek()

검색(`query()`)은 유사도 기반이지만, 때로는 ID나 조건으로 정확히 가져와야 할 때가 있습니다.

```python
# ID로 특정 문서 가져오기
docs = collection.get(ids=["doc1", "doc2"])

# 메타데이터 조건으로 필터링
docs = collection.get(
    where={"source": "blog"},
    include=["documents", "metadatas"],
)

# 문서 내용으로 필터링
docs = collection.get(
    where_document={"$contains": "수정"},
    include=["documents"],
)

# 컬렉션 내 전체 문서 수 확인
total = collection.count()

# 처음 몇 개 문서 미리보기
preview = collection.peek(limit=5)
```

`get()`은 유사도 검색이 아니라 **정확한 조건 매칭**입니다. "이 ID의 문서를 보여줘" 또는 "source가 blog인 문서를 모두 보여줘" 같은 용도죠.

## 실습: 직접 해보기

지금까지 배운 내용을 하나의 흐름으로 연결해보겠습니다. 간단한 기술 블로그 검색 시스템을 만들어봅시다.

```run:python
import chromadb

# 1. 클라이언트 생성 (인메모리)
client = chromadb.EphemeralClient()

# 2. 컬렉션 생성
collection = client.get_or_create_collection("tech_blog")

# 3. 기술 블로그 글 추가
collection.add(
    ids=["post1", "post2", "post3", "post4", "post5"],
    documents=[
        "벡터 데이터베이스는 고차원 벡터를 저장하고 유사도 기반 검색을 수행합니다.",
        "LangChain은 LLM 기반 애플리케이션을 쉽게 만들 수 있는 프레임워크입니다.",
        "RAG는 검색 증강 생성으로, 외부 지식으로 LLM 응답을 보강합니다.",
        "임베딩 모델은 텍스트를 고차원 벡터로 변환하여 의미를 수치로 표현합니다.",
        "FastAPI로 REST API를 만들면 RAG 시스템을 웹 서비스로 배포할 수 있습니다.",
    ],
    metadatas=[
        {"category": "database", "level": "beginner"},
        {"category": "framework", "level": "beginner"},
        {"category": "technique", "level": "intermediate"},
        {"category": "ml", "level": "intermediate"},
        {"category": "deployment", "level": "advanced"},
    ],
)

print(f"총 {collection.count()}개의 문서가 추가되었습니다.\n")

# 4. 유사도 검색
print("=" * 50)
print("검색 쿼리: '텍스트를 숫자로 바꾸는 방법'")
print("=" * 50)

results = collection.query(
    query_texts=["텍스트를 숫자로 바꾸는 방법"],
    n_results=3,
    include=["documents", "distances", "metadatas"],
)

# 결과 출력
for i in range(len(results["ids"][0])):
    doc_id = results["ids"][0][i]
    document = results["documents"][0][i]
    distance = results["distances"][0][i]
    metadata = results["metadatas"][0][i]
    print(f"\n[{i+1}] ID: {doc_id} (거리: {distance:.4f})")
    print(f"    카테고리: {metadata['category']}, 난이도: {metadata['level']}")
    print(f"    내용: {document[:50]}...")

# 5. 문서 업데이트
collection.update(
    ids=["post5"],
    documents=["Docker와 Kubernetes로 RAG 시스템을 프로덕션에 배포하는 방법을 다룹니다."],
    metadatas=[{"category": "deployment", "level": "advanced", "updated": True}],
)

# 6. upsert로 새 문서 추가
collection.upsert(
    ids=["post6"],
    documents=["프롬프트 엔지니어링은 LLM에게 정확한 응답을 유도하는 기술입니다."],
    metadatas=[{"category": "technique", "level": "beginner"}],
)

print(f"\n\n업데이트 후 총 문서 수: {collection.count()}")

# 7. 메타데이터 필터링으로 조회
beginner_docs = collection.get(
    where={"level": "beginner"},
    include=["documents", "metadatas"],
)
print(f"\n초급(beginner) 문서 {len(beginner_docs['ids'])}개:")
for doc_id, doc in zip(beginner_docs["ids"], beginner_docs["documents"]):
    print(f"  - [{doc_id}] {doc[:40]}...")
```

```output
총 5개의 문서가 추가되었습니다.

==================================================
검색 쿼리: '텍스트를 숫자로 바꾸는 방법'
==================================================

[1] ID: post4 (거리: 0.8372)
    카테고리: ml, 난이도: intermediate
    내용: 임베딩 모델은 텍스트를 고차원 벡터로 변환하여 의미를 수치로 표현합니다....

[2] ID: post1 (거리: 1.2541)
    카테고리: database, 난이도: beginner
    내용: 벡터 데이터베이스는 고차원 벡터를 저장하고 유사도 기반 검색을 수행합니다....

[3] ID: post3 (거리: 1.3298)
    카테고리: technique, 난이도: intermediate
    내용: RAG는 검색 증강 생성으로, 외부 지식으로 LLM 응답을 보강합니다....


업데이트 후 총 문서 수: 6

초급(beginner) 문서 3개:
  - [post1] 벡터 데이터베이스는 고차원 벡터를 저장하고 유사도 기반 검색을 수행합니...
  - [post2] LangChain은 LLM 기반 애플리케이션을 쉽게 만들 수 있는 프레임워...
  - [post6] 프롬프트 엔지니어링은 LLM에게 정확한 응답을 유도하는 기술입니다....
```

"텍스트를 숫자로 바꾸는 방법"이라는 쿼리에 "임베딩 모델" 관련 문서가 가장 높은 유사도로 반환된 것을 확인할 수 있습니다. 정확히 "숫자"라는 단어가 들어있지 않아도 **의미적으로 가장 가까운** 문서를 찾아낸 거죠 — 이것이 벡터 검색의 힘입니다.

## 더 깊이 알아보기

### ChromaDB의 탄생 — 발렌타인 데이에 시작된 오픈소스

ChromaDB를 만든 사람은 Jeff Huber와 Anton Troynikov입니다. 두 사람은 이전에 Standard Cyborg라는 3D 스캐닝 스타트업을 공동 창업했었는데(Y Combinator 2015년 겨울 배치), 거기서 뼈저리게 느낀 것이 있었습니다 — "머신러닝으로 뭔가를 만드는 건 가능한데, 그걸 **데모에서 프로덕션으로** 가져가는 건 정말 어렵다."

2022년 12월, Huber는 트위터에서 LangChain과 임베딩에 관해 이야기하는 개발자들에게 연락해 피드백을 모았습니다. 개발자들에게 쉽게 쓸 수 있는 임베딩 데이터베이스가 부족하다는 것을 확인한 그는 곧바로 개발에 착수했고, **2023년 발렌타인 데이(2월 14일)**에 오픈소스로 공개했습니다. 정확히는 버그 수정 때문에 하루 늦은 2월 15일이었다고 하네요.

성장은 폭발적이었습니다. 별도의 마케팅 없이 순수 입소문만으로 2023년 9월까지 **150만 건의 누적 다운로드**를 기록했고, 당시 월간 다운로드만 60만 건을 넘었습니다. 2023년 4월에는 1,800만 달러의 투자를 유치하기도 했습니다. "개발자에게 가장 쉬운 벡터 데이터베이스"라는 포지셔닝이 정확히 맞아떨어진 셈이죠.

### 기본 임베딩 함수의 비밀

ChromaDB가 "설치만 하면 바로 쓸 수 있는" 비결은 기본 임베딩 함수에 있습니다. `all-MiniLM-L6-v2`라는 Sentence Transformers 모델을 ONNX 형식으로 패키징해서 함께 배포하거든요. 덕분에 PyTorch나 TensorFlow 같은 무거운 딥러닝 프레임워크 없이도, 그리고 외부 API 호출 없이도 로컬에서 임베딩을 생성할 수 있습니다.

물론 프로덕션에서는 OpenAI의 `text-embedding-3-small`이나 Cohere의 임베딩 모델처럼 더 강력한 모델을 사용하는 것이 일반적입니다. 이 부분은 [6.3](06-벡터-데이터베이스-기초-chromadb로-시작하기/03-chromadb-메타데이터와-필터링.md)에서 자세히 다룹니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "EphemeralClient는 성능이 PersistentClient보다 빠르다?" — 엄밀히 말하면 **검색 성능은 거의 동일**합니다. 차이가 나는 것은 데이터 로딩 시점뿐이에요. PersistentClient는 시작할 때 디스크에서 데이터를 메모리로 읽어오는 시간이 필요하지만, 일단 로딩되면 검색 속도는 같습니다. EphemeralClient가 "빠르다"는 것은 초기 설정이 간단하다는 의미에 가깝습니다.

> 💡 **알고 계셨나요?**: ChromaDB의 `ids`는 반드시 **문자열**이어야 합니다. 정수 `1`이 아니라 문자열 `"1"`을 넣어야 해요. 이걸 모르고 `ids=[1, 2, 3]`처럼 넣으면 타입 에러가 발생합니다. 또한, 같은 ID로 `add()`를 두 번 호출하면 에러가 나니, 중복이 걱정되면 `upsert()`를 사용하세요.

> 🔥 **실무 팁**: `query()` 결과의 `distances`는 **작을수록 유사**합니다. 기본 거리 함수(`cosine`)에서 0에 가까울수록 완전히 같은 문서이고, 2에 가까울수록 완전히 반대 의미입니다. 결과를 사용자에게 보여줄 때 "유사도 95%"처럼 표현하고 싶다면, `similarity = 1 - (distance / 2)` 공식으로 변환할 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `EphemeralClient` | 인메모리 전용 클라이언트. 프로그램 종료 시 데이터 소멸. 프로토타이핑용 |
| `PersistentClient` | 디스크에 영구 저장. `path` 파라미터로 저장 경로 지정. 로컬 개발용 |
| `HttpClient` | 원격 Chroma 서버에 연결. 팀 협업과 프로덕션 배포용 |
| `create_collection()` | 새 컬렉션 생성. 동일 이름 존재 시 에러 |
| `get_or_create_collection()` | 있으면 가져오고, 없으면 생성하는 안전한 방법 |
| `add()` | 문서 추가. `ids` 필수, `documents` 또는 `embeddings` 중 하나 필요 |
| `query()` | 유사도 기반 검색. `query_texts`와 `n_results`로 제어 |
| `get()` | ID 또는 메타데이터 조건으로 정확 조회 |
| `update()` / `upsert()` | 문서 수정. `upsert()`는 없으면 자동 추가 |
| `delete()` | ID 기반 문서 삭제 |
| 기본 임베딩 | `all-MiniLM-L6-v2` (384차원, ONNX Runtime, 로컬 실행) |

## 다음 섹션 미리보기

지금까지 ChromaDB의 기본 CRUD를 마스터했습니다. 그런데 실전에서는 한 가지 중요한 질문이 남아 있죠 — "기본 임베딩 함수 말고 **더 좋은 임베딩**을 쓰려면 어떻게 해야 하나?" 다음 섹션에서는 ChromaDB에 **커스텀 임베딩 함수**를 연동하는 방법과, OpenAI·HuggingFace 등 외부 임베딩 모델을 활용하는 방법을 다룹니다. 또한 **메타데이터 필터링**의 고급 기능(`$gt`, `$lt`, `$in` 등 연산자)을 배워 검색 정밀도를 한 단계 끌어올리겠습니다.

## 참고 자료

- [ChromaDB 공식 GitHub](https://github.com/chroma-core/chroma) - 소스코드, 이슈 트래커, 릴리즈 정보를 확인할 수 있는 공식 저장소
- [Chroma 공식 사용 가이드](https://docs.trychroma.com/guides) - 클라이언트 생성, 컬렉션 관리, 문서 CRUD의 공식 가이드
- [Chroma Cookbook — Clients](https://cookbook.chromadb.dev/core/clients/) - EphemeralClient, PersistentClient, HttpClient의 상세 파라미터와 예제
- [Chroma Cookbook — Collections](https://cookbook.chromadb.dev/core/collections/) - 컬렉션 CRUD 작업의 심화 레시피와 HNSW 설정 가이드
- [DataCamp ChromaDB 튜토리얼](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide) - 초보자를 위한 단계별 ChromaDB 사용법
- [Jeff Huber 인터뷰 — Chroma의 탄생과 비전](https://www.madrona.com/chromas-jeff-huber-on-vector-databases-and-getting-ai-into-production/) - ChromaDB 창업자가 직접 말하는 탄생 배경과 벡터 데이터베이스의 미래

---
### 🔗 Related Sessions
- [vector_database](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [ann](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [hnsw](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
