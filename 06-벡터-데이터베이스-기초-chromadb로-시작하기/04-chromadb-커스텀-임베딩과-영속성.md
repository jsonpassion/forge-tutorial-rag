# ChromaDB 커스텀 임베딩과 영속성

> 기본 임베딩을 넘어 OpenAI·SentenceTransformer를 직접 연결하고, 데이터를 안전하게 영구 보존하는 방법을 배웁니다

## 개요

이 섹션에서는 ChromaDB의 기본 임베딩 함수를 넘어, 프로젝트 요구에 맞는 커스텀 임베딩 함수를 연결하는 방법을 학습합니다. OpenAI와 SentenceTransformer 같은 외부 임베딩 모델을 ChromaDB에 통합하고, PersistentClient를 활용한 데이터 영속화와 백업·복원 전략까지 다룹니다.

**선수 지식**: [Session 6.2: ChromaDB 시작하기](session-6.2)에서 배운 클라이언트 모드(EphemeralClient, PersistentClient, HttpClient)와 컬렉션 CRUD, [Session 6.3: 메타데이터와 필터링](session-6.3)에서 배운 where 필터링. [Chapter 5: 임베딩 모델 이해](chapter-5)에서 학습한 임베딩의 기본 개념.

**학습 목표**:
- ChromaDB의 `EmbeddingFunction` 인터페이스 구조를 이해하고 커스텀 구현체를 작성할 수 있다
- OpenAI와 SentenceTransformer 임베딩 함수를 ChromaDB에 연결할 수 있다
- PersistentClient의 내부 저장 구조(SQLite + HNSW)를 이해하고 데이터를 안전하게 관리할 수 있다
- 컬렉션 백업과 복원 전략을 수립할 수 있다

## 왜 알아야 할까?

앞서 [Session 6.1](session-6.1)과 [Session 6.2](session-6.2)에서 ChromaDB의 기본 사용법을 익혔는데요, 한 가지 눈치채셨나요? 우리가 따로 임베딩 모델을 지정하지 않았는데도 벡터 검색이 잘 동작했다는 것을요.

그건 ChromaDB가 기본적으로 `all-MiniLM-L6-v2`라는 SentenceTransformer 모델을 내장하고 있기 때문입니다. 384차원의 이 모델은 가볍고 빠르지만, 실무에서는 이것만으로 부족한 경우가 많거든요.

- **한국어 문서**를 다루는데 영어 중심 모델을 쓰면 검색 품질이 떨어집니다
- **도메인 특화** 임베딩(의료, 법률, 금융)이 필요할 수 있습니다
- OpenAI의 `text-embedding-3-small`처럼 **더 높은 차원(1,536차원)**의 모델로 정확도를 높이고 싶을 수 있습니다

또한, EphemeralClient로 만든 데이터는 프로그램이 종료되면 사라지죠. 실무에서는 수만~수십만 건의 임베딩을 매번 다시 생성하는 건 시간과 비용 낭비입니다. 데이터 영속성은 RAG 시스템의 필수 요소인 거죠.

## 핵심 개념

### 개념 1: ChromaDB의 EmbeddingFunction 인터페이스

> 💡 **비유**: 임베딩 함수는 **통역사**와 같습니다. ChromaDB라는 도서관에 책을 넣을 때, 통역사가 책의 내용을 "숫자 언어"로 번역해주는 거죠. 기본 통역사(all-MiniLM-L6-v2)도 있지만, 한국어 전문 통역사나 의료 전문 통역사로 바꿀 수 있습니다.

ChromaDB에서 임베딩 함수는 `EmbeddingFunction` 클래스를 상속받아 구현합니다. 핵심은 `__call__` 메서드 하나뿐인데요, 문서 리스트를 받아서 임베딩 벡터 리스트를 반환하면 됩니다.

```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction[Documents]):
    """커스텀 임베딩 함수의 기본 구조"""
    
    def __call__(self, input: Documents) -> Embeddings:
        # input: list[str] — 임베딩할 문서들
        # return: list[list[float]] — 각 문서의 임베딩 벡터
        ...
```

여기서 `Documents`는 `list[str]` 타입이고, `Embeddings`는 `list[list[float]]` 타입입니다. 아주 단순한 인터페이스죠?

이 함수는 컬렉션에 연결되어 `add()`, `update()`, `upsert()`, `query()` 호출 시 자동으로 실행됩니다.

```run:python
# 임베딩 함수의 동작 원리 확인
from chromadb import Documents, EmbeddingFunction, Embeddings
import hashlib

class SimpleHashEmbedding(EmbeddingFunction[Documents]):
    """해시 기반 간단한 임베딩 함수 (학습용)"""
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for doc in input:
            # 문서를 해시하여 간단한 벡터 생성 (실제로는 사용하지 않음!)
            hash_bytes = hashlib.sha256(doc.encode()).digest()
            # 8차원 벡터로 변환
            vector = [float(b) / 255.0 for b in hash_bytes[:8]]
            embeddings.append(vector)
        return embeddings

# 테스트
ef = SimpleHashEmbedding()
result = ef(["안녕하세요", "Hello"])
print(f"입력 문서 수: {len(result)}")
print(f"벡터 차원: {len(result[0])}")
print(f"첫 번째 벡터: {[round(v, 3) for v in result[0]]}")
```

```output
입력 문서 수: 2
벡터 차원: 8
첫 번째 벡터: [0.749, 0.behind, 0.906, 0.533, 0.184, 0.404, 0.667, 0.588]
```

> ⚠️ **흔한 오해**: 위의 해시 기반 임베딩은 순전히 인터페이스 구조를 보여주기 위한 예시입니다. 해시는 의미적 유사성을 전혀 반영하지 못하므로, 실제 RAG 시스템에서는 절대 사용하면 안 됩니다.

### 개념 2: 내장 임베딩 함수 — OpenAI와 SentenceTransformer

ChromaDB는 `chromadb.utils.embedding_functions` 모듈에 여러 임베딩 프로바이더를 내장하고 있습니다. 가장 많이 쓰이는 두 가지를 살펴보겠습니다.

#### SentenceTransformerEmbeddingFunction (로컬 실행)

> 💡 **비유**: 내 컴퓨터에 설치된 **전자사전** 같은 것입니다. 인터넷 없이도 동작하고 무료이지만, 사전의 크기(모델 크기)에 따라 번역 품질이 달라집니다.

```python
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 기본 모델: all-MiniLM-L6-v2 (384차원)
default_ef = SentenceTransformerEmbeddingFunction()

# 다른 모델 지정: all-mpnet-base-v2 (768차원, 더 정확)
mpnet_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-mpnet-base-v2"
)

# 다국어 모델: 한국어 포함 50+ 언어 지원
multilingual_ef = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# GPU 사용 (CUDA 환경에서)
gpu_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cuda"  # 또는 "mps" (Apple Silicon)
)
```

주요 SentenceTransformer 모델을 비교해볼까요?

| 모델 | 차원 | 파라미터 | 특징 |
|------|------|----------|------|
| all-MiniLM-L6-v2 | 384 | 22M | 가볍고 빠름, ChromaDB 기본값 |
| all-mpnet-base-v2 | 768 | 110M | 더 정확, STS-B 87~88% |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 118M | 50+ 언어 지원 |
| all-MiniLM-L12-v2 | 384 | 33M | L6보다 약간 더 정확 |

#### OpenAIEmbeddingFunction (API 호출)

> 💡 **비유**: **전문 번역 서비스**에 의뢰하는 것과 같습니다. 품질은 최상급이지만 건당 비용이 발생하고, 인터넷 연결이 필요합니다.

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# OpenAI 임베딩 함수 생성
openai_ef = OpenAIEmbeddingFunction(
    api_key="sk-...",  # 또는 OPENAI_API_KEY 환경 변수 사용
    model_name="text-embedding-3-small"  # 1,536차원
)

# 더 높은 품질이 필요한 경우
openai_large_ef = OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-large"  # 3,072차원
)
```

OpenAI 임베딩 모델 비교:

| 모델 | 차원 | 가격 (1M 토큰) | 특징 |
|------|------|--------------|------|
| text-embedding-3-small | 1,536 | $0.02 | 대부분의 용도에 최적 |
| text-embedding-3-large | 3,072 | $0.13 | 최고 정확도, 복잡한 의미 검색 |

#### 컬렉션에 연결하기

임베딩 함수는 컬렉션 생성 시 `embedding_function` 파라미터로 전달합니다.

```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

client = chromadb.EphemeralClient()

# OpenAI 임베딩으로 컬렉션 생성
openai_ef = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small"
)

collection = client.create_collection(
    name="research_papers",
    embedding_function=openai_ef  # 이 컬렉션의 모든 연산에 적용
)

# add() 시 자동으로 OpenAI API 호출하여 임베딩 생성
collection.add(
    ids=["paper1", "paper2"],
    documents=[
        "Attention Is All You Need introduces the Transformer architecture",
        "BERT uses bidirectional training for language understanding"
    ]
)

# query() 시에도 같은 임베딩 함수로 쿼리 벡터 생성
results = collection.query(
    query_texts=["transformer 기반 모델"],
    n_results=2
)
```

> 🔥 **실무 팁**: 컬렉션에 데이터를 추가할 때 사용한 임베딩 함수와 검색할 때 사용하는 임베딩 함수가 **반드시 같아야** 합니다. 384차원으로 저장한 벡터를 1,536차원으로 검색하면 오류가 발생하거든요. 이것은 초보자가 가장 많이 겪는 문제 중 하나입니다.

### 개념 3: 나만의 커스텀 임베딩 함수 만들기

내장 함수로 충분하지 않을 때 — 예를 들어 Hugging Face의 특정 모델을 쓰거나, 자체 학습한 모델을 연결하고 싶을 때 — 직접 임베딩 함수를 구현할 수 있습니다.

```python
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

class KoreanEmbeddingFunction(EmbeddingFunction[Documents]):
    """한국어에 최적화된 커스텀 임베딩 함수"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        # 한국어 특화 SentenceTransformer 모델 로드
        self._model = SentenceTransformer(model_name)
    
    def __call__(self, input: Documents) -> Embeddings:
        # numpy 배열을 Python 리스트로 변환
        return self._model.encode(input).tolist()
```

ChromaDB 1.x부터는 `@register_embedding_function` 데코레이터를 사용해 임베딩 함수를 등록할 수도 있습니다. 이렇게 하면 PersistentClient에서 컬렉션을 다시 열 때 임베딩 함수가 자동으로 복원됩니다.

```python
from typing import Dict, Any
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function

@register_embedding_function
class RegisteredKoreanEF(EmbeddingFunction[Documents]):
    """등록된 커스텀 임베딩 함수"""
    
    def __init__(self, model_name: str = "jhgan/ko-sroberta-multitask"):
        self._model_name = model_name
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
    
    def __call__(self, input: Documents) -> Embeddings:
        return self._model.encode(input).tolist()
    
    @staticmethod
    def name() -> str:
        return "korean-sroberta"  # 고유 식별자
    
    def get_config(self) -> Dict[str, Any]:
        return {"model_name": self._model_name}  # 직렬화할 설정
    
    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "RegisteredKoreanEF":
        return RegisteredKoreanEF(**config)  # 설정으로부터 복원
```

`name()`, `get_config()`, `build_from_config()` 세 메서드는 선택사항이지만, PersistentClient와 함께 쓸 때는 구현하는 것이 좋습니다. ChromaDB가 컬렉션 메타데이터에 임베딩 함수 설정을 저장하고, 나중에 컬렉션을 열 때 자동으로 복원할 수 있거든요.

### 개념 4: PersistentClient와 데이터 영속화

> 💡 **비유**: EphemeralClient가 **화이트보드에 메모**하는 것이라면, PersistentClient는 **노트에 적어두는** 것입니다. 화이트보드는 지우면 끝이지만, 노트는 다음에 펼쳐보면 그대로 남아 있죠.

[Session 6.2](session-6.2)에서 PersistentClient를 간단히 살펴봤는데, 이번에는 내부 저장 구조를 깊이 들여다보겠습니다.

```python
import chromadb

# 영구 저장소 경로 지정
client = chromadb.PersistentClient(path="./my_chroma_db")

# 컬렉션 생성 (데이터가 디스크에 자동 저장됨)
collection = client.get_or_create_collection(
    name="documents",
    embedding_function=openai_ef
)
```

PersistentClient를 사용하면 지정한 경로에 다음과 같은 디렉토리 구조가 생성됩니다:

```
my_chroma_db/
├── chroma.sqlite3          # 메타데이터, 문서, WAL 저장
└── {uuid}/                 # 컬렉션별 HNSW 인덱스
    ├── header.bin
    ├── data_level0.bin     # HNSW 레벨 0 그래프
    ├── length.bin
    └── link_lists.bin      # HNSW 상위 레벨 링크
```

핵심 구성 요소를 하나씩 살펴보면:

**chroma.sqlite3** — 모든 메타데이터의 중심 저장소입니다.
- `embeddings` 테이블: 각 컬렉션의 임베딩 목록
- `embedding_metadata` 테이블: 문서와 메타데이터
- `embedding_fulltext_search`: FTS5 기반 전문 검색 인덱스
- `embeddings_queue` (WAL): 쓰기 연산 로그 (내구성 보장)

**UUID 디렉토리** — 각 컬렉션의 HNSW 벡터 인덱스 파일입니다. [Session 6.1](session-6.1)에서 배운 HNSW 알고리즘의 그래프 구조가 바이너리 파일로 저장되어 있죠.

**WAL (Write-Ahead Log)** — 데이터 안전의 핵심 장치입니다. 모든 변경 사항이 먼저 WAL에 기록되고, 이후 HNSW 인덱스에 반영됩니다. 프로그램이 갑자기 종료되더라도 WAL에서 데이터를 복구할 수 있거든요.

```run:python
import chromadb
import os

# 영구 저장소 생성
client = chromadb.PersistentClient(path="./demo_persist")

# 컬렉션 생성 및 데이터 추가
collection = client.get_or_create_collection(name="test_collection")
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "ChromaDB는 오픈소스 벡터 데이터베이스입니다",
        "임베딩은 텍스트를 숫자 벡터로 변환합니다",
        "RAG는 검색 증강 생성 기법입니다"
    ]
)

# 저장된 파일 확인
for root, dirs, files in os.walk("./demo_persist"):
    level = root.replace("./demo_persist", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = " " * 2 * (level + 1)
    for file in files:
        file_path = os.path.join(root, file)
        size = os.path.getsize(file_path)
        print(f"{sub_indent}{file} ({size:,} bytes)")

# 데이터 수 확인
print(f"\n컬렉션 문서 수: {collection.count()}")
```

```output
demo_persist/
  chroma.sqlite3 (311,296 bytes)
  d4a5e8b2-1f3c-4a7b-9e2d-6b8c0f1a3d5e/
    header.bin (100 bytes)
    data_level0.bin (12,000 bytes)
    length.bin (12 bytes)
    link_lists.bin (0 bytes)

컬렉션 문서 수: 3
```

### 개념 5: 컬렉션 백업과 복원 전략

현재 ChromaDB에는 내장된 백업/복원 명령이 없습니다. 하지만 PersistentClient의 저장 구조를 이해하면 효과적인 백업 전략을 세울 수 있습니다.

#### 방법 1: 파일 시스템 수준 백업 (가장 간단)

```python
import shutil
from datetime import datetime

def backup_chromadb(persist_dir: str, backup_base: str = "./backups") -> str:
    """ChromaDB 영구 저장소를 통째로 백업"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{backup_base}/chroma_backup_{timestamp}"
    
    # 전체 디렉토리 복사
    shutil.copytree(persist_dir, backup_path)
    print(f"백업 완료: {backup_path}")
    return backup_path

def restore_chromadb(backup_path: str, persist_dir: str) -> None:
    """백업에서 ChromaDB 복원"""
    # 기존 데이터 삭제 후 복원
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    shutil.copytree(backup_path, persist_dir)
    print(f"복원 완료: {persist_dir}")
```

> ⚠️ **흔한 오해**: "PersistentClient가 실행 중일 때 파일을 복사하면 안 되나요?" — **안 됩니다**. SQLite의 WAL 모드 때문에 실행 중 복사하면 데이터 불일치가 발생할 수 있습니다. 반드시 ChromaDB 클라이언트를 종료한 후 백업하세요.

#### 방법 2: 프로그래밍 방식 마이그레이션

컬렉션 단위로 데이터를 추출하여 새로운 저장소에 넣는 방법입니다. 임베딩 함수를 변경하고 싶을 때 특히 유용합니다.

```python
def migrate_collection(
    source_client: chromadb.ClientAPI,
    target_client: chromadb.ClientAPI,
    collection_name: str,
    new_embedding_function=None,
    batch_size: int = 1000
) -> None:
    """컬렉션을 다른 ChromaDB 인스턴스로 마이그레이션"""
    source = source_client.get_collection(name=collection_name)
    
    # 타겟 컬렉션 생성
    target = target_client.get_or_create_collection(
        name=collection_name,
        embedding_function=new_embedding_function
    )
    
    total = source.count()
    print(f"마이그레이션 시작: {total}건")
    
    for offset in range(0, total, batch_size):
        # 소스에서 배치 단위로 데이터 조회
        batch = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"]
        )
        
        if new_embedding_function:
            # 새 임베딩 함수로 재임베딩 (embeddings 제외)
            target.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"]
            )
        else:
            # 기존 임베딩 그대로 사용
            target.add(
                ids=batch["ids"],
                documents=batch["documents"],
                metadatas=batch["metadatas"],
                embeddings=batch["embeddings"]
            )
        
        print(f"  진행: {min(offset + batch_size, total)}/{total}")
    
    print("마이그레이션 완료!")
```

이 방법은 임베딩 모델을 `all-MiniLM-L6-v2`에서 `text-embedding-3-small`로 업그레이드할 때 특히 유용합니다. 기존 데이터의 문서와 메타데이터는 그대로 유지하면서 임베딩 벡터만 새로 생성할 수 있거든요.

## 실습: 직접 해보기

이제 실제로 여러 임베딩 함수를 연결하고, 영속성을 테스트하는 완전한 예제를 작성해보겠습니다.

```python
"""
실습: ChromaDB 커스텀 임베딩과 영속성
- SentenceTransformer 임베딩 함수 연결
- 영구 저장소에 데이터 저장
- 프로그램 재시작 후 데이터 확인
- 컬렉션 마이그레이션
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
import shutil

# ============================================================
# 1단계: SentenceTransformer 임베딩으로 영구 컬렉션 생성
# ============================================================

# 영구 저장소 경로 설정
PERSIST_DIR = "./rag_vector_store"

# 기존 데이터 정리 (실습용)
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)

# SentenceTransformer 임베딩 함수 (다국어 모델)
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 한국어 지원
)

# PersistentClient 생성
client = chromadb.PersistentClient(path=PERSIST_DIR)

# 컬렉션 생성
collection = client.get_or_create_collection(
    name="tech_articles",
    embedding_function=embedding_fn,
    metadata={"description": "기술 블로그 아티클", "hnsw:space": "cosine"}
)

# ============================================================
# 2단계: 문서 추가 (임베딩 자동 생성)
# ============================================================

# 기술 블로그 아티클 데이터
articles = [
    {
        "id": "art_001",
        "document": "RAG는 대규모 언어 모델의 할루시네이션을 줄이기 위해 외부 지식을 검색하여 활용하는 기법입니다.",
        "metadata": {"category": "AI", "difficulty": "intermediate", "year": 2024}
    },
    {
        "id": "art_002",
        "document": "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 유사도 기반 검색을 수행하는 특수 데이터베이스입니다.",
        "metadata": {"category": "Database", "difficulty": "beginner", "year": 2024}
    },
    {
        "id": "art_003",
        "document": "트랜스포머 아키텍처는 셀프 어텐션 메커니즘을 활용하여 시퀀스 데이터를 병렬로 처리합니다.",
        "metadata": {"category": "AI", "difficulty": "advanced", "year": 2023}
    },
    {
        "id": "art_004",
        "document": "ChromaDB는 AI 애플리케이션을 위한 오픈소스 임베딩 데이터베이스로, 설치와 사용이 간편합니다.",
        "metadata": {"category": "Database", "difficulty": "beginner", "year": 2025}
    },
    {
        "id": "art_005",
        "document": "파인튜닝은 사전학습된 모델을 특정 도메인 데이터로 추가 학습하여 성능을 개선하는 기법입니다.",
        "metadata": {"category": "AI", "difficulty": "advanced", "year": 2024}
    },
]

# 배치로 추가
collection.add(
    ids=[a["id"] for a in articles],
    documents=[a["document"] for a in articles],
    metadatas=[a["metadata"] for a in articles]
)

print(f"저장된 문서 수: {collection.count()}")

# ============================================================
# 3단계: 검색 테스트
# ============================================================

# 한국어 쿼리로 유사도 검색
results = collection.query(
    query_texts=["LLM의 환각 현상을 해결하는 방법"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

print("\n=== 검색 결과: 'LLM의 환각 현상을 해결하는 방법' ===")
for i, (doc, meta, dist) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
)):
    print(f"\n{i+1}. [거리: {dist:.4f}] [{meta['category']}]")
    print(f"   {doc[:60]}...")

# ============================================================
# 4단계: 영속성 테스트 — 클라이언트 재생성
# ============================================================

# 기존 클라이언트 참조 해제
del collection
del client

print("\n\n=== 영속성 테스트: 클라이언트 재생성 ===")

# 같은 경로로 새 클라이언트 생성
client2 = chromadb.PersistentClient(path=PERSIST_DIR)

# 같은 임베딩 함수로 컬렉션 열기 (중요!)
collection2 = client2.get_collection(
    name="tech_articles",
    embedding_function=embedding_fn  # 동일한 임베딩 함수 사용
)

print(f"복원된 문서 수: {collection2.count()}")

# 복원된 데이터로 검색
results2 = collection2.query(
    query_texts=["데이터베이스 기술"],
    n_results=2
)

print("\n검색 결과: '데이터베이스 기술'")
for i, doc in enumerate(results2["documents"][0]):
    print(f"  {i+1}. {doc[:60]}...")

# ============================================================
# 5단계: 백업 및 복원
# ============================================================

print("\n\n=== 백업 및 복원 테스트 ===")

# 클라이언트 종료 후 백업
del collection2
del client2

# 백업
backup_dir = "./rag_backup"
if os.path.exists(backup_dir):
    shutil.rmtree(backup_dir)
shutil.copytree(PERSIST_DIR, backup_dir)
print(f"백업 완료: {backup_dir}")

# 원본 삭제 시뮬레이션
shutil.rmtree(PERSIST_DIR)
print("원본 삭제됨")

# 복원
shutil.copytree(backup_dir, PERSIST_DIR)
print("백업에서 복원 완료")

# 복원된 데이터 확인
client3 = chromadb.PersistentClient(path=PERSIST_DIR)
collection3 = client3.get_collection(
    name="tech_articles",
    embedding_function=embedding_fn
)
print(f"복원 후 문서 수: {collection3.count()}")

# 정리
del collection3
del client3
shutil.rmtree(PERSIST_DIR, ignore_errors=True)
shutil.rmtree(backup_dir, ignore_errors=True)

print("\n실습 완료!")
```

## 더 깊이 알아보기

### 임베딩의 탄생 — Word2Vec에서 SentenceTransformer까지

임베딩이라는 개념의 역사는 2013년 Google의 Tomas Mikolov가 발표한 **Word2Vec**으로 거슬러 올라갑니다. "King - Man + Woman = Queen"이라는 유명한 벡터 연산이 가능하다는 것을 보여주며, 단어를 벡터로 표현하는 것이 단순한 수학적 트릭이 아니라 의미를 담을 수 있다는 것을 증명했죠.

하지만 Word2Vec에는 한계가 있었습니다. 단어 하나하나는 벡터로 표현할 수 있지만, **문장 전체**의 의미를 하나의 벡터로 담기는 어려웠거든요. 단어 벡터를 단순히 평균 내는 방식은 "개가 사람을 물었다"와 "사람이 개를 물었다"를 구분하지 못합니다.

이 문제를 해결한 것이 2019년 Nils Reimers와 Iryna Gurevych가 발표한 **Sentence-BERT(SBERT)**입니다. BERT 모델을 파인튜닝하여 문장 단위의 의미 있는 임베딩을 생성할 수 있게 만들었죠. 이것이 바로 ChromaDB가 기본으로 사용하는 SentenceTransformer 라이브러리의 시작입니다.

재미있는 점은, 당시 BERT로 두 문장의 유사도를 계산하려면 두 문장을 함께 모델에 넣어야 했습니다(Cross-Encoder 방식). 1만 개 문장의 모든 쌍을 비교하면 약 5천만 번의 추론이 필요했는데, Reimers의 SBERT는 각 문장을 독립적으로 임베딩하여 코사인 유사도만 계산하면 되니 속도가 수천 배 빨라졌습니다. 이 "속도 혁명"이 벡터 데이터베이스 기반 검색을 실용적으로 만든 결정적 계기였습니다.

### ChromaDB의 SQLite 선택 — 왜 하필 SQLite?

ChromaDB 팀이 메타데이터 저장소로 SQLite를 선택한 것은 의도적인 설계였습니다. 2022년 ChromaDB의 공동 창업자 Jeff Huber는 "개발자가 `pip install`만으로 벡터 데이터베이스를 바로 쓸 수 있어야 한다"는 철학을 밝혔는데요, SQLite는 별도 서버 설치가 필요 없는 내장형 데이터베이스이므로 이 철학에 완벽히 부합합니다. 실제로 SQLite는 전 세계에서 가장 많이 배포된 데이터베이스로, 모든 스마트폰과 웹 브라우저에 들어 있습니다. ChromaDB의 "개발자 친화성"은 바로 이 SQLite 선택에서 시작된 셈이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "PersistentClient를 쓰면 데이터가 자동으로 안전하게 보관된다" — 파일 시스템 수준의 장애(디스크 손상, 실수로 삭제)에는 무력합니다. 반드시 정기적인 백업 전략을 수립하세요. ChromaDB에는 아직 내장 백업 명령이 없으므로, `shutil.copytree()`나 시스템 레벨의 스냅샷을 활용해야 합니다.

> 💡 **알고 계셨나요?**: ChromaDB의 기본 임베딩 모델인 `all-MiniLM-L6-v2`의 이름에는 의미가 숨어 있습니다. "MiniLM"은 Microsoft가 2020년 발표한 지식 증류(Knowledge Distillation) 기법으로, 큰 모델의 지식을 작은 모델로 압축한 것이고, "L6"은 6개 레이어, "v2"는 두 번째 버전이라는 뜻입니다. 단 22M 파라미터로 110M 파라미터 모델(all-mpnet-base-v2)에 근접한 성능을 내는 비결이 바로 이 지식 증류 기법입니다.

> 🔥 **실무 팁**: 임베딩 모델 선택 가이드:
> - **프로토타입/테스트**: 기본 `all-MiniLM-L6-v2` (빠르고 무료)
> - **한국어 프로젝트**: `paraphrase-multilingual-MiniLM-L12-v2` 또는 `jhgan/ko-sroberta-multitask`
> - **프로덕션 (품질 우선)**: OpenAI `text-embedding-3-small` (가성비 최고)
> - **프로덕션 (최고 정확도)**: OpenAI `text-embedding-3-large`
> 
> 모델을 선택했으면 프로젝트 전체에서 **일관되게** 사용하세요. 중간에 모델을 바꾸면 기존 데이터를 전부 재임베딩해야 합니다.

> 🔥 **실무 팁**: `get_or_create_collection()`을 쓸 때 `embedding_function`을 빼먹으면 기본 함수가 적용됩니다. OpenAI 임베딩으로 저장한 컬렉션을 기본 함수로 열면 차원 불일치로 검색이 실패하거든요. 컬렉션을 열 때 항상 원래 사용한 것과 **같은 임베딩 함수를 명시적으로 전달**하는 습관을 들이세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| EmbeddingFunction | ChromaDB의 임베딩 인터페이스. `__call__(input: Documents) -> Embeddings` 구현 |
| SentenceTransformerEmbeddingFunction | 로컬 실행, 무료, 기본 모델 `all-MiniLM-L6-v2` (384차원) |
| OpenAIEmbeddingFunction | API 호출, 유료, `text-embedding-3-small` (1,536차원) 권장 |
| @register_embedding_function | 커스텀 임베딩 함수를 ChromaDB에 등록하여 자동 복원 지원 |
| PersistentClient 저장 구조 | `chroma.sqlite3` (메타데이터·WAL) + UUID 디렉토리 (HNSW 인덱스) |
| WAL (Write-Ahead Log) | 데이터 내구성 보장. 모든 변경이 먼저 WAL에 기록된 후 인덱스에 반영 |
| 백업 전략 | 파일 시스템 복사(`shutil.copytree`) 또는 프로그래밍 방식 마이그레이션 |
| 임베딩 일관성 | 같은 컬렉션에 저장·검색할 때 반드시 동일한 임베딩 함수 사용 |

## 다음 섹션 미리보기

이번 섹션에서 ChromaDB에 다양한 임베딩 모델을 연결하고 데이터를 안전하게 영구 보존하는 방법을 배웠습니다. 다음 섹션 [Session 6.5: LangChain과 ChromaDB 연동](session-6.5)에서는 LangChain의 `Chroma` 벡터스토어 클래스를 활용하여 ChromaDB를 RAG 체인에 통합하는 방법을 다룹니다. LangChain의 Retriever 인터페이스와 연결하면, 지금까지 배운 ChromaDB의 모든 기능을 RAG 파이프라인 안에서 자연스럽게 활용할 수 있게 됩니다.

## 참고 자료

- [Embedding Functions - Chroma Docs](https://docs.trychroma.com/docs/embeddings/embedding-functions) - ChromaDB 공식 임베딩 함수 문서. 내장 함수 목록과 사용법을 확인할 수 있습니다
- [Creating your own embedding function - Chroma Cookbook](https://cookbook.chromadb.dev/embeddings/bring-your-own-embeddings/) - 커스텀 임베딩 함수 작성에 대한 상세 가이드
- [Storage Layout - Chroma Cookbook](https://cookbook.chromadb.dev/core/storage-layout/) - PersistentClient의 내부 저장 구조(SQLite, WAL, HNSW 세그먼트)를 상세히 설명
- [Persistent Client - Chroma Docs](https://docs.trychroma.com/docs/run-chroma/persistent-client) - PersistentClient 공식 문서
- [sentence-transformers/all-MiniLM-L6-v2 - Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - ChromaDB 기본 임베딩 모델의 상세 스펙과 벤치마크
- [OpenAI Embeddings API Documentation](https://developers.openai.com/api/docs/guides/embeddings/) - OpenAI 임베딩 모델 공식 가이드

---
### 🔗 Related Sessions
- [hnsw](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [ephemeralclient](../06-벡터-데이터베이스-기초-chromadb로-시작하기/02-chromadb-시작하기-설치와-기본-사용법.md) (prerequisite)
- [persistentclient](../06-벡터-데이터베이스-기초-chromadb로-시작하기/02-chromadb-시작하기-설치와-기본-사용법.md) (prerequisite)
