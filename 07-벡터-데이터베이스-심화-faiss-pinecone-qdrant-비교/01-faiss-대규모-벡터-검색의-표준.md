# FAISS — 대규모 벡터 검색의 표준

> Meta AI가 만든 오픈소스 벡터 검색 라이브러리, 수십억 벡터도 거뜬히 처리하는 업계 표준 도구

## 개요

이 섹션에서는 FAISS(Facebook AI Similarity Search)의 핵심 인덱스 타입을 이해하고, Python에서 직접 벡터를 색인·검색하는 방법을 배웁니다. 정확도와 속도 사이의 트레이드오프를 체감하고, 백만 규모 데이터에서도 밀리초 단위로 유사 벡터를 찾아내는 실전 기술을 익힙니다.

**선수 지식**: [Ch6: 벡터 데이터베이스 기초](ch06)에서 배운 벡터 저장·검색 개념, 코사인 유사도(Cosine Similarity)와 L2 거리, ANN(Approximate Nearest Neighbor) 알고리즘의 기본 원리

**학습 목표**:
- FAISS의 세 가지 핵심 인덱스(IndexFlatL2, IndexIVFFlat, IndexHNSWFlat)의 동작 원리와 차이를 설명할 수 있다
- 인덱스 학습(train)과 검색(search) 과정을 Python 코드로 구현할 수 있다
- 데이터 규모와 요구사항에 따라 적절한 인덱스를 선택할 수 있다
- GPU 가속의 원리와 적용 방법을 이해한다

## 왜 알아야 할까?

앞서 Ch6에서 ChromaDB로 벡터 데이터베이스의 기본기를 다졌는데요, 실무에서 RAG 시스템을 운영하다 보면 한 가지 질문에 부딪히게 됩니다. "벡터가 100만 개, 1000만 개로 늘어나면 어떻게 하지?"

ChromaDB는 프로토타이핑과 소규모 프로젝트에 훌륭하지만, **대규모 벡터 검색**에서는 보다 정교한 인덱싱 전략이 필요합니다. FAISS는 바로 이 문제를 해결하기 위해 Meta AI(구 Facebook AI Research)가 만든 라이브러리로, **10억 개 벡터**에서도 밀리초 단위로 유사한 벡터를 찾아냅니다.

실제로 FAISS는 Meta의 내부 검색 시스템에서 매일 수십억 건의 쿼리를 처리하고 있고, LangChain·LlamaIndex 등 주요 RAG 프레임워크가 모두 FAISS 통합을 기본 제공할 만큼 업계 표준으로 자리 잡았습니다. PyPI 기준 누적 다운로드 수가 600만 건을 넘었다는 사실이 그 인기를 증명하죠.

## 핵심 개념

### 개념 1: FAISS의 기본 구조 — 도서관의 책 정리 시스템

> 💡 **비유**: FAISS를 **거대한 도서관**이라고 생각해보세요. 책(벡터)이 10권이면 하나씩 넘겨보면 되지만, 100만 권이라면? 주제별로 서가를 나누고(클러스터링), 서가 안에서도 색인표를 만들어 빠르게 찾는 시스템이 필요합니다. FAISS의 인덱스가 바로 이 "정리 시스템"입니다.

FAISS는 C++로 작성되어 있고, Python/NumPy 래퍼를 완벽하게 제공합니다. 핵심 워크플로우는 단 세 단계입니다:

1. **인덱스 생성**: 어떤 방식으로 벡터를 정리할지 결정
2. **학습/추가**: 벡터를 인덱스에 넣기 (일부 인덱스는 사전 학습 필요)
3. **검색**: 쿼리 벡터와 가장 가까운 k개의 벡터 찾기

먼저 설치부터 해볼까요?

```bash
# CPU 버전 설치 (대부분의 경우 이것으로 충분)
pip install faiss-cpu

# GPU 버전 설치 (CUDA 12.x 환경)
pip install faiss-gpu-cu12
```

가장 기본적인 사용법을 살펴봅시다:

```run:python
import numpy as np
import faiss

# 벡터 차원과 개수 설정
d = 128          # 벡터 차원 (임베딩 모델 출력 크기에 맞춤)
nb = 10000       # 데이터베이스 벡터 수
nq = 3           # 쿼리 벡터 수

# 재현 가능한 랜덤 벡터 생성
np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')  # 데이터베이스 벡터
xq = np.random.random((nq, d)).astype('float32')  # 쿼리 벡터

# 가장 기본적인 인덱스 생성 (정확 검색)
index = faiss.IndexFlatL2(d)
print(f"인덱스 학습 필요 여부: {not index.is_trained}")
print(f"인덱스에 저장된 벡터 수: {index.ntotal}")

# 벡터 추가
index.add(xb)
print(f"벡터 추가 후: {index.ntotal}개")

# 검색: 각 쿼리에 대해 가장 가까운 5개 벡터 찾기
k = 5
distances, indices = index.search(xq, k)
print(f"\n첫 번째 쿼리의 최근접 5개 인덱스: {indices[0]}")
print(f"첫 번째 쿼리의 L2 거리: {np.round(distances[0], 2)}")
```

```output
인덱스 학습 필요 여부: False
인덱스에 저장된 벡터 수: 0
벡터 추가 후: 10000개

첫 번째 쿼리의 최근접 5개 인덱스: [6815 1445 3858 9624 7375]
첫 번째 쿼리의 L2 거리: [15.52 15.6  15.7  15.72 15.74]
```

`search()` 메서드는 두 개의 배열을 반환합니다. `distances`는 쿼리와 결과 벡터 사이의 L2 거리, `indices`는 원본 데이터베이스에서의 위치(인덱스 번호)입니다.

### 개념 2: IndexFlatL2 — 완벽하지만 느린 전수 조사

> 💡 **비유**: 시험에서 **모든 답안지를 하나씩 채점**하는 것과 같습니다. 100% 정확하지만, 답안지가 100만 장이면 시간이 오래 걸리겠죠?

`IndexFlatL2`는 FAISS에서 가장 단순한 인덱스입니다. 모든 벡터를 원본 그대로 저장하고, 검색할 때 **모든 벡터와 하나씩 거리를 계산**합니다(brute-force). 그래서:

- **장점**: 100% 정확한 결과를 보장하는 유일한 인덱스
- **단점**: 벡터 수에 비례해서 검색 시간이 선형으로 증가
- **용도**: 소규모 데이터셋(수만 건 이하), 정확도 기준선(baseline) 측정

```python
# IndexFlatL2 — 사전 학습(train) 불필요
index_flat = faiss.IndexFlatL2(d)     # L2(유클리드) 거리 기준
# index_flat_ip = faiss.IndexFlatIP(d)  # 내적(Inner Product) 기준 — 코사인 유사도에 활용

index_flat.add(xb)  # 바로 벡터 추가 가능 (train 불필요)
```

> ⚠️ **흔한 오해**: "IndexFlatL2는 코사인 유사도를 지원하지 않는다"고 생각하기 쉬운데요, 벡터를 **L2 정규화**(단위 벡터로 변환)한 뒤 `IndexFlatIP`(내적 기반)를 쓰면 코사인 유사도와 동일한 결과를 얻습니다. FAISS는 `faiss.normalize_L2()` 유틸리티를 제공합니다.

### 개념 3: IndexIVFFlat — 클러스터 기반의 빠른 검색

> 💡 **비유**: 도서관에서 책을 찾을 때, **모든 서가를 뒤지는 대신** 우선 "이 책은 과학 코너일 것"이라고 범위를 좁힌 뒤 그 코너만 뒤지는 방식입니다. 약간의 정확도를 희생하지만 검색 속도가 비약적으로 빨라집니다.

IVF(Inverted File Index)는 벡터 공간을 **여러 클러스터(셀)**로 나누고, 쿼리와 가장 가까운 클러스터만 탐색합니다:

1. **학습(train)**: k-means 클러스터링으로 `nlist`개의 중심점(centroid) 생성
2. **추가(add)**: 각 벡터를 가장 가까운 클러스터에 배정
3. **검색(search)**: 쿼리에 가장 가까운 `nprobe`개 클러스터만 탐색

핵심 파라미터 두 가지를 기억하세요:

| 파라미터 | 의미 | 가이드라인 |
|---------|------|----------|
| `nlist` | 전체 클러스터 수 | 일반적으로 `sqrt(n)` ~ `4 * sqrt(n)` |
| `nprobe` | 검색 시 탐색할 클러스터 수 | 클수록 정확하지만 느림. 기본값 1 |

```run:python
import numpy as np
import faiss

d = 128
nb = 100000  # 10만 개 벡터
np.random.seed(42)
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((1, d)).astype('float32')

nlist = 100  # 클러스터 수 (sqrt(100000) ≈ 316이지만, 예시용으로 100)

# IVF 인덱스는 양자화기(quantizer)가 필요
quantizer = faiss.IndexFlatL2(d)  # 클러스터 중심점 비교용
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# 반드시 학습(train)을 먼저 해야 함!
print(f"학습 전 is_trained: {index_ivf.is_trained}")
index_ivf.train(xb)  # k-means 클러스터링 수행
print(f"학습 후 is_trained: {index_ivf.is_trained}")

# 학습 후 벡터 추가
index_ivf.add(xb)
print(f"저장된 벡터 수: {index_ivf.ntotal}")

# nprobe=1 (기본값) vs nprobe=10 비교
index_ivf.nprobe = 1
_, indices_1 = index_ivf.search(xq, 5)
print(f"\nnprobe=1  결과: {indices_1[0]}")

index_ivf.nprobe = 10
_, indices_10 = index_ivf.search(xq, 5)
print(f"nprobe=10 결과: {indices_10[0]}")

# 정확 검색(Flat)과 비교
index_exact = faiss.IndexFlatL2(d)
index_exact.add(xb)
_, indices_exact = index_exact.search(xq, 5)
print(f"정확 검색  결과: {indices_exact[0]}")
```

```output
학습 전 is_trained: False
학습 후 is_trained: True
저장된 벡터 수: 100000

nprobe=1  결과: [26896 92054 40113 38920 95498]
nprobe=10 결과: [71900 26896 92054 40113 38920]
정확 검색  결과: [71900 26896 92054 40113 38920]
```

`nprobe=10`이면 정확 검색과 동일한 결과를 보여주지만, `nprobe=1`에서는 일부 결과가 빠져있죠? 이것이 바로 **속도와 정확도의 트레이드오프**입니다. `nprobe`를 높일수록 정확도는 올라가지만 검색 시간도 증가합니다.

### 개념 4: IndexHNSWFlat — 그래프로 빠르게 탐색

> 💡 **비유**: HNSW는 **소셜 네트워크에서 사람 찾기**와 비슷합니다. "김 교수님을 아는 사람은?" → "그의 동료 박 교수님이요" → "박 교수님과 같은 연구실의 이 연구원이요"처럼, 아는 사람의 아는 사람을 따라가면 몇 단계 안에 목표에 도달합니다. 이것이 "6단계 분리(Six Degrees of Separation)" 원리이고, HNSW는 이를 벡터 검색에 적용한 것입니다.

HNSW(Hierarchical Navigable Small World)는 벡터들을 **다층 그래프**로 연결합니다. 상위 계층에서 대략적인 위치를 잡고, 하위 계층으로 내려가면서 점점 정밀하게 탐색하는 방식이죠.

핵심 파라미터:

| 파라미터 | 의미 | 가이드라인 |
|---------|------|----------|
| `M` | 각 노드의 이웃 연결 수 | 클수록 정확하지만 메모리 증가. 기본값 32 |
| `efConstruction` | 그래프 구축 시 탐색 범위 | 클수록 정확한 그래프. 기본값 40 |
| `efSearch` | 검색 시 탐색 범위 | 클수록 정확하지만 느림. 기본값 16 |

```python
import faiss

d = 128
M = 32  # 이웃 연결 수

# HNSW 인덱스 생성 — FAISS 클래스명은 IndexHNSWFlat
index_hnsw = faiss.IndexHNSWFlat(d, M)

# 그래프 구축 품질 설정 (add 전에 설정)
index_hnsw.hnsw.efConstruction = 64

# HNSW는 train 불필요 — 벡터 추가 시 자동으로 그래프 구축
index_hnsw.add(xb)

# 검색 품질 설정 (search 전에 설정)
index_hnsw.hnsw.efSearch = 32
distances, indices = index_hnsw.search(xq, 5)
```

**IndexIVFFlat vs IndexHNSWFlat 비교**:

| 특성 | IndexIVFFlat | IndexHNSWFlat |
|------|-------------|---------------|
| 사전 학습 | 필요 (k-means) | 불필요 |
| 검색 속도 | 빠름 | 매우 빠름 |
| 메모리 사용 | 낮음 | 높음 (그래프 구조 저장) |
| 벡터 삭제 | 가능 | 불가능 (재구축 필요) |
| 적합한 규모 | 100만~10억 | 수천~수백만 |
| 정확도 조절 | `nprobe`로 간편 | `efSearch`로 조절 |

> 🔥 **실무 팁**: 메모리 여유가 있고 데이터가 수백만 건 이하라면 HNSW가 속도·정확도 모두 최고입니다. 그러나 10억 건 이상이라면 IVF 계열 인덱스에 **Product Quantization(PQ)**을 결합한 `IndexIVFPQ`가 메모리·속도·정확도의 최적 균형을 제공합니다.

### 개념 5: GPU 가속 — 검색 속도를 10배 높이기

FAISS는 NVIDIA GPU를 활용한 가속을 지원합니다. CPU 대비 **5~10배**(최신 GPU에서는 20배 이상)의 속도 향상을 얻을 수 있죠.

```python
import faiss

# GPU 리소스 초기화
res = faiss.StandardGpuResources()

# 방법 1: CPU 인덱스를 GPU로 변환
index_cpu = faiss.IndexFlatL2(d)
index_cpu.add(xb)
index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)  # 0번 GPU 사용

# 방법 2: 모든 GPU 활용 (멀티 GPU)
index_multi_gpu = faiss.index_cpu_to_all_gpus(index_cpu)

# 검색은 CPU 인덱스와 동일한 인터페이스
distances, indices = index_gpu.search(xq, k)
```

> ⚠️ **흔한 오해**: "GPU가 있으면 무조건 GPU 버전을 써야 한다"고 생각할 수 있는데, 벡터 수가 10만 건 미만이면 CPU와 GPU의 차이가 미미합니다. GPU 메모리로 벡터를 옮기는 오버헤드 때문이죠. GPU 가속은 **대규모 데이터셋(100만 건 이상)**에서 진가를 발휘합니다.

## 실습: 직접 해보기

세 가지 인덱스의 성능을 직접 비교해봅시다. 100만 개 벡터에서 검색 속도와 정확도(Recall)를 측정합니다.

```python
import numpy as np
import faiss
import time

# ========================================
# 데이터 준비: 100만 개 벡터
# ========================================
d = 128           # 벡터 차원
nb = 1_000_000    # 데이터 벡터 수 (100만)
nq = 100          # 쿼리 벡터 수
k = 10            # 검색할 이웃 수

np.random.seed(42)
print("100만 개 벡터 생성 중...")
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')
print(f"데이터 크기: {xb.nbytes / 1024 / 1024:.0f} MB")

# ========================================
# 1. IndexFlatL2 — 정확 검색 (기준선)
# ========================================
print("\n[IndexFlatL2] 정확 검색...")
index_flat = faiss.IndexFlatL2(d)
index_flat.add(xb)

start = time.time()
distances_gt, indices_gt = index_flat.search(xq, k)  # ground truth
flat_time = time.time() - start
print(f"  검색 시간: {flat_time:.3f}초")
print(f"  Recall@{k}: 100.0% (기준선)")

# ========================================
# 2. IndexIVFFlat — 클러스터 기반 검색
# ========================================
print("\n[IndexIVFFlat] 클러스터 기반 검색...")
nlist = 1024  # sqrt(1M) ≈ 1000
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)

# 학습 (전체 데이터 또는 샘플로 가능)
train_start = time.time()
index_ivf.train(xb)
print(f"  학습 시간: {time.time() - train_start:.2f}초")
index_ivf.add(xb)

# nprobe 값별 성능 비교
for nprobe in [1, 8, 32, 64]:
    index_ivf.nprobe = nprobe
    start = time.time()
    _, indices_ivf = index_ivf.search(xq, k)
    search_time = time.time() - start

    # Recall 계산: ground truth와 비교
    recall = np.mean([
        len(set(indices_ivf[i]) & set(indices_gt[i])) / k
        for i in range(nq)
    ])
    print(f"  nprobe={nprobe:>2}: 검색 {search_time:.4f}초, Recall@{k}: {recall*100:.1f}%")

# ========================================
# 3. IndexHNSWFlat — 그래프 기반 검색
# ========================================
print("\n[IndexHNSWFlat] 그래프 기반 검색...")
M = 32
index_hnsw = faiss.IndexHNSWFlat(d, M)
index_hnsw.hnsw.efConstruction = 64

add_start = time.time()
index_hnsw.add(xb)
print(f"  그래프 구축 시간: {time.time() - add_start:.2f}초")

for efSearch in [16, 32, 64, 128]:
    index_hnsw.hnsw.efSearch = efSearch
    start = time.time()
    _, indices_hnsw = index_hnsw.search(xq, k)
    search_time = time.time() - start

    recall = np.mean([
        len(set(indices_hnsw[i]) & set(indices_gt[i])) / k
        for i in range(nq)
    ])
    print(f"  efSearch={efSearch:>3}: 검색 {search_time:.4f}초, Recall@{k}: {recall*100:.1f}%")

# ========================================
# 4. 결과 요약
# ========================================
print("\n" + "="*50)
print("인덱스 선택 가이드:")
print(f"  벡터 수: {nb:,}개, 차원: {d}, 검색 top-{k}")
print(f"  FlatL2     → 정확도 100%, 속도 기준선")
print(f"  IVFFlat    → nprobe 조절로 속도/정확도 균형")
print(f"  HNSWFlat   → 메모리 더 쓰지만 속도·정확도 최고")
```

> 🔥 **실무 팁**: 실제 프로젝트에서는 먼저 `IndexFlatL2`로 **정확도 기준선**을 측정하고, 그 결과를 ground truth로 삼아 다른 인덱스의 Recall을 비교하세요. Recall@10이 95% 이상이면 대부분의 RAG 용도에 충분합니다.

### LangChain과 FAISS 연동

Ch8에서 본격적으로 다루겠지만, LangChain에서 FAISS를 사용하는 기본 패턴을 미리 살펴봅시다:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 문서 준비
docs = [
    Document(page_content="RAG는 검색 증강 생성 기법입니다.", metadata={"source": "ch1"}),
    Document(page_content="FAISS는 Meta가 만든 벡터 검색 라이브러리입니다.", metadata={"source": "ch7"}),
    Document(page_content="임베딩은 텍스트를 벡터로 변환합니다.", metadata={"source": "ch5"}),
]

# FAISS 벡터 스토어 생성 (내부적으로 IndexFlatL2 사용)
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# 유사도 검색
results = vectorstore.similarity_search("벡터 검색이란?", k=2)
for doc in results:
    print(f"[{doc.metadata['source']}] {doc.page_content}")

# 로컬 파일로 저장/로드 (인덱스 재구축 없이 재사용)
vectorstore.save_local("faiss_index")
loaded_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

## 더 깊이 알아보기

### FAISS의 탄생 — "10억 벡터를 1초 안에"

2015년, Meta(당시 Facebook)의 AI Research 팀은 거대한 도전 과제를 안고 있었습니다. 소셜 미디어 플랫폼에 매일 올라오는 수십억 장의 이미지와 동영상에서 "유사한 콘텐츠"를 실시간으로 찾아야 했거든요. 기존 검색 알고리즘으로는 이 규모를 감당할 수 없었습니다.

2년간의 연구와 엔지니어링 끝에, 2017년 3월 FAISS가 공개됩니다. 발표 논문에서 팀은 놀라운 성과를 보여줬는데요 — **10억 개의 고차원 벡터에서 최근접 이웃 그래프(k-NN graph)를 구축하는 데 성공**한 것입니다. 이는 당시 최고 기록 대비 **8.5배 빠른** 속도였습니다. GPU 상에서의 k-selection 알고리즘도 학계에 알려진 것 중 가장 빠르다고 보고되었죠.

흥미로운 점은 FAISS가 완전히 새로운 발명이 아니라는 것입니다. 2003년의 "Video Google" 논문에서 나온 역파일 인덱스(Inverted File Index)와 2011년의 곱 양자화(Product Quantization) 연구를 핵심 기반으로 삼았거든요. 수십 년간 축적된 검색 연구의 결정체를 하나의 실용적 라이브러리로 엮어낸 것이 FAISS의 진짜 가치입니다.

### 이름의 유래

**FAISS**는 **F**acebook **AI** **S**imilarity **S**earch의 약자입니다. 2023년 Meta로 사명이 바뀐 후에도 라이브러리 이름은 그대로 유지되고 있죠. 현재 공식 문서 도메인은 [faiss.ai](https://faiss.ai)이며, GitHub 저장소는 여전히 `facebookresearch/faiss`입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "FAISS는 벡터 데이터베이스다"라고 생각하기 쉽지만, 정확히는 **벡터 검색 라이브러리**입니다. Pinecone이나 Qdrant처럼 데이터 영속성, 필터링, 메타데이터 관리, 분산 처리를 기본 제공하지 않습니다. FAISS 위에 이런 기능을 추가한 것이 벡터 데이터베이스죠. 실제로 많은 벡터 DB가 내부적으로 FAISS를 검색 엔진으로 사용합니다.

> 💡 **알고 계셨나요?**: FAISS의 `IndexIVFFlat`에서 `nlist`(클러스터 수)를 정하는 경험 법칙이 있습니다. 데이터 포인트 수를 `n`이라 할 때, `nlist = C × sqrt(n)`으로 설정합니다(C는 보통 1~4). 100만 개 벡터라면 `nlist`는 1,000~4,000 정도가 적당하죠. 너무 작으면 각 클러스터에 벡터가 많아져 검색이 느리고, 너무 크면 학습 시간이 길어지고 빈 클러스터가 생깁니다.

> 🔥 **실무 팁**: FAISS 인덱스를 디스크에 저장하고 다시 불러올 수 있습니다. 매번 인덱스를 새로 구축하지 않아도 되니 시간을 크게 절약할 수 있죠:
> ```python
> # 저장
> faiss.write_index(index, "my_index.faiss")
> # 불러오기
> index = faiss.read_index("my_index.faiss")
> ```
> 특히 `IndexIVFFlat`처럼 학습(train)에 시간이 걸리는 인덱스는 반드시 저장해두세요.

> 🔥 **실무 팁**: `IndexIVFFlat`에 벡터를 추가한 뒤 `train()`을 호출하면 에러가 나는 게 아니라, **이미 추가된 벡터의 클러스터 배정이 틀어집니다**. 반드시 `train()` → `add()` 순서를 지키세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| FAISS | Meta AI가 개발한 대규모 벡터 유사도 검색 라이브러리. C++ 기반, Python 래퍼 제공 |
| IndexFlatL2 | 전수 조사(brute-force) 인덱스. 100% 정확하지만 대규모에서 느림 |
| IndexIVFFlat | 클러스터 기반 ANN 인덱스. `nlist`로 클러스터 수, `nprobe`로 정확도-속도 트레이드오프 조절 |
| IndexHNSWFlat | 그래프 기반 ANN 인덱스. 메모리 더 사용하지만 속도·정확도 최고 |
| train() → add() → search() | IVF 계열 인덱스의 필수 워크플로우. Flat/HNSW는 train 불필요 |
| nprobe / efSearch | 검색 시 정확도-속도 트레이드오프를 제어하는 런타임 파라미터 |
| GPU 가속 | `index_cpu_to_gpu()`로 CPU 인덱스를 GPU로 변환. 대규모에서 5~20배 속도 향상 |
| 인덱스 저장/로드 | `faiss.write_index()` / `faiss.read_index()`로 디스크 영속화 |

## 다음 섹션 미리보기

FAISS의 강력한 검색 능력을 확인했지만, FAISS는 관리형 서비스가 아닌 라이브러리이기 때문에 인프라 운영은 직접 해야 합니다. 다음 섹션 **[7.2: Pinecone — 완전 관리형 벡터 데이터베이스]**에서는 클라우드에서 벡터 검색을 서비스로 제공하는 Pinecone을 살펴보고, "직접 운영 vs 관리형 서비스"의 트레이드오프를 비교합니다.

## 참고 자료

- [FAISS GitHub 리포지토리](https://github.com/facebookresearch/faiss) - 소스 코드, 튜토리얼, 위키 문서 모두 포함된 공식 저장소
- [FAISS 공식 문서](https://faiss.ai/index.html) - API 레퍼런스와 인덱스별 상세 가이드
- [Guidelines to choose an index (FAISS Wiki)](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index) - 데이터 규모·요구사항별 인덱스 선택 가이드. 실무에서 가장 자주 참고하는 페이지
- [Faiss: A library for efficient similarity search (Meta Engineering Blog, 2017)](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) - FAISS 최초 공개 발표 글. 탄생 배경과 벤치마크 수록
- [Introduction to FAISS (Pinecone Learning)](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/) - FAISS의 각 인덱스 타입을 시각적으로 잘 설명한 튜토리얼 시리즈
- [LangChain FAISS Integration](https://docs.langchain.com/oss/python/integrations/vectorstores/faiss) - LangChain에서 FAISS를 벡터 스토어로 사용하는 공식 가이드

---
### 🔗 Related Sessions
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_database](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [ann](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [hnsw](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
