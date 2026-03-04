# Sentence Transformers — 오픈소스 임베딩 모델

> SBERT 아키텍처의 원리를 이해하고, HuggingFace 모델 허브에서 목적에 맞는 임베딩 모델을 선택하여 로컬 환경에서 추론하는 방법을 학습합니다.

## 개요

앞서 [5.1: 임베딩의 기본 개념](05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md)에서 Sentence Transformers의 `all-MiniLM-L6-v2` 모델로 첫 임베딩을 생성해보았고, [5.2: OpenAI 임베딩 API 활용](05-임베딩-모델-이해-텍스트를-벡터로-변환/02-openai-임베딩-api-활용.md)에서는 클라우드 기반 API의 강점과 비용 최적화 전략을 다뤘습니다. 이번 세션에서는 시선을 돌려, API 비용 없이 로컬에서 직접 임베딩을 생성하는 **오픈소스 Sentence Transformers 라이브러리**를 본격적으로 파헤칩니다.

**선수 지식**: 임베딩과 벡터 공간의 기본 개념(5.1), OpenAI 임베딩 API 경험(5.2)
**학습 목표**:
- SBERT의 샴 네트워크(Siamese Network) 아키텍처와 학습 방식을 이해한다
- `all-MiniLM-L6-v2`, `all-mpnet-base-v2` 등 주요 모델의 특성을 비교하고 선택할 수 있다
- HuggingFace 모델 허브와 MTEB 리더보드를 활용해 목적에 맞는 모델을 찾을 수 있다
- GPU/CPU 환경에서 추론 속도를 최적화하는 실전 기법을 적용할 수 있다

## 왜 알아야 할까?

[5.2](05-임베딩-모델-이해-텍스트를-벡터로-변환/02-openai-임베딩-api-활용.md)에서 OpenAI 임베딩 API의 편리함을 경험했는데요, 그렇다면 왜 굳이 오픈소스 모델을 배워야 할까요?

현실적인 이유가 세 가지 있습니다. 첫째, **비용**입니다. 100만 건의 문서를 임베딩하면 OpenAI API 비용이 수십 달러씩 나가지만, 로컬 모델은 전기세 외에 추가 비용이 없습니다. 둘째, **데이터 프라이버시**입니다. 의료 기록, 법률 문서, 기업 내부 자료처럼 외부로 보낼 수 없는 데이터가 있죠. 셋째, **커스터마이징**입니다. 오픈소스 모델은 자신의 도메인 데이터로 파인튜닝(Fine-tuning)하여 특정 분야에서 범용 API보다 더 나은 성능을 낼 수 있습니다.

Sentence Transformers는 HuggingFace에 **10,000개 이상의 사전학습 모델**이 공개되어 있고, 영어뿐 아니라 한국어 특화 모델까지 활발히 개발되고 있습니다. RAG 실무에서 벡터 DB에 문서를 대량 인덱싱할 때, 로컬 임베딩은 거의 필수에 가깝습니다.

## 핵심 개념

### 개념 1: SBERT 아키텍처 — 쌍둥이 네트워크의 마법

> 💡 **비유**: SBERT를 이해하려면 **쌍둥이 심사위원**을 떠올려보세요. 두 명의 심사위원이 각각 다른 에세이를 읽고 점수를 매깁니다. 이 쌍둥이는 완전히 같은 기준(가중치)으로 평가하기 때문에, 두 점수를 비교하면 에세이가 얼마나 비슷한지 바로 알 수 있습니다. 기존 BERT는 두 에세이를 한꺼번에 읽어야 해서 느렸지만, 쌍둥이 심사위원은 각자 독립적으로 읽으니 훨씬 빠릅니다.

이것이 바로 **샴 네트워크(Siamese Network)** 구조입니다. 2019년, 독일 다름슈타트 공대 UKP Lab의 Nils Reimers와 Iryna Gurevych가 제안한 SBERT(Sentence-BERT)는 동일한 BERT 모델 두 개를 나란히 배치합니다. 각 모델이 문장을 독립적으로 처리하여 임베딩 벡터를 생성하고, 이 벡터들 사이의 거리(코사인 유사도)로 문장 간 의미적 유사성을 판단합니다.

**핵심 구조**:

```
문장 A → [BERT] → 풀링(Pooling) → 벡터 A ─┐
                                              ├→ 코사인 유사도
문장 B → [BERT] → 풀링(Pooling) → 벡터 B ─┘
         (동일한 가중치 공유)
```

여기서 **풀링(Pooling)** 은 BERT가 출력하는 토큰별 벡터들을 하나의 문장 벡터로 합치는 과정입니다. [5.1](05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md)에서 이미 Mean Pooling의 원리를 배웠는데요, SBERT에서는 이 풀링 단계가 샴 네트워크의 핵심 역할을 합니다 — 두 문장이 각각 독립적으로 풀링되어 고정 크기 벡터가 되기 때문에, 사전에 벡터를 계산해두고 빠르게 비교할 수 있는 것이죠. SBERT는 세 가지 풀링 전략을 지원합니다:

| 풀링 전략 | 방법 | 특징 |
|-----------|------|------|
| CLS 토큰 | `[CLS]` 토큰의 출력 사용 | 간단하지만 성능이 낮은 편 |
| **Mean Pooling** | 모든 토큰 벡터의 평균 ([5.1](05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) 참고) | **가장 널리 사용**, 안정적 성능 |
| Max Pooling | 각 차원의 최댓값 선택 | 특수 케이스에서 유용 |

이 중 Mean Pooling이 SBERT의 기본 전략으로, [5.1](05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md)에서 다룬 것처럼 모든 토큰의 의미 정보를 고르게 반영하여 가장 안정적인 문장 벡터를 만들어냅니다.

**학습 방식 — SBERT가 좋은 임베딩을 만드는 비결**

SBERT는 자연어 추론(NLI) 데이터셋으로 학습되는데, 핵심 아이디어는 직관적입니다:

- **분류 목적함수**: 두 문장의 관계(함의/모순/중립)를 분류하며 학습
- **대조 학습(Contrastive Learning)**: "비슷한 뜻의 문장은 벡터 공간에서 가깝게, 다른 뜻의 문장은 멀게 배치하라"는 원칙으로 훈련됩니다. 덕분에 SBERT는 미묘한 의미 차이까지 벡터 거리로 포착할 수 있습니다
- **삼중항 손실(Triplet Loss)**: 기준 문장(앵커), 유사 문장(양성), 비유사 문장(음성) 세 개를 한 조로 학습합니다. "앵커와 양성은 가깝게, 앵커와 음성은 멀게" 배치하는 방식으로, 모델이 의미적 유사성의 경계를 더 정밀하게 학습하게 됩니다

이런 학습 방식이 왜 중요할까요? RAG에서 검색 품질은 결국 "질의와 관련 문서가 벡터 공간에서 얼마나 가까이 놓이는가"에 달려 있기 때문입니다. 대조 학습으로 훈련된 SBERT는 의미적으로 비슷한 문장들이 벡터 공간에서 자연스럽게 군집을 이루므로, 검색 정확도가 높아집니다.

### 개념 2: 주요 사전학습 모델 비교 — 어떤 모델을 골라야 할까?

> 💡 **비유**: 임베딩 모델 선택은 **자동차 구매**와 비슷합니다. 경차(`MiniLM`)는 연비(속도)가 좋고 가격(메모리)이 저렴하지만 힘(정확도)은 부족합니다. 중형차(`mpnet`)는 균형 잡힌 선택이고, SUV(`multilingual`)는 어디든(다국어) 갈 수 있지만 크고 무겁습니다.

Sentence Transformers가 공식 추천하는 범용 모델을 비교해봅시다:

| 모델 | 차원 | 속도(상대) | 주요 특징 |
|------|------|-----------|-----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡⚡⚡ | 속도 최우선, 가벼운 프로토타이핑 |
| `all-mpnet-base-v2` | 768 | ⚡⚡⚡ | **공식 추천 최고 품질** 범용 모델 |
| `multi-qa-mpnet-base-dot-v1` | 768 | ⚡⚡⚡ | 질의응답 특화, 2.15억 QA쌍 학습 |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | ⚡⚡ | 50+언어 지원, 다국어 RAG |

```run:python
from sentence_transformers import SentenceTransformer

# 두 모델의 출력 차원 비교
mini_model = SentenceTransformer("all-MiniLM-L6-v2")
mpnet_model = SentenceTransformer("all-mpnet-base-v2")

sample = "RAG는 검색 증강 생성 기법입니다."
mini_emb = mini_model.encode(sample)
mpnet_emb = mpnet_model.encode(sample)

print(f"MiniLM 차원: {mini_emb.shape[0]}")   # 384차원
print(f"MPNet 차원: {mpnet_emb.shape[0]}")    # 768차원
print(f"MiniLM 파라미터: ~22M")
print(f"MPNet 파라미터: ~109M")
```

```output
MiniLM 차원: 384
MPNet 차원: 768
MiniLM 파라미터: ~22M
MPNet 파라미터: ~109M
```

**선택 가이드**:
- **프로토타이핑, 학습용** → `all-MiniLM-L6-v2` (빠르고 가벼움)
- **영어 프로덕션 RAG** → `all-mpnet-base-v2` (최고 품질)
- **한국어/다국어 RAG** → `paraphrase-multilingual-mpnet-base-v2` 또는 한국어 특화 모델
- **질의응답 특화 검색** → `multi-qa-mpnet-base-dot-v1`

### 개념 3: 한국어 임베딩 모델 — 우리말에 맞는 모델 찾기

> 💡 **비유**: 영어 모델로 한국어를 임베딩하는 것은 **영한사전만으로 한국 소설을 읽는** 것과 같습니다. 단어는 알아도 "눈치", "정", "한" 같은 한국 문화에 깊이 뿌리내린 뉘앙스는 놓치게 되죠. 한국어 특화 모델은 이런 미묘한 의미까지 포착합니다.

한국어 RAG 시스템을 구축한다면 다음 모델들을 고려해보세요:

| 모델 | 기반 | 차원 | 특징 |
|------|------|------|------|
| `jhgan/ko-sroberta-multitask` | KoRoBERTa | 768 | 한국어 STS/NLI 학습, 널리 사용 |
| `snunlp/KR-SBERT-V40K-klueNLI-augSTS` | SBERT | 768 | KLUE 벤치마크 기반 학습 |
| `paraphrase-multilingual-mpnet-base-v2` | MPNet | 768 | 50+언어, 한국어 포함 |

```run:python
from sentence_transformers import SentenceTransformer, util

# 한국어 특화 모델 로드
ko_model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# 한국어 문장 유사도 테스트
sentences = [
    "오늘 날씨가 정말 좋습니다.",
    "오늘 하늘이 맑고 화창하네요.",
    "주식 시장이 크게 하락했습니다.",
]

embeddings = ko_model.encode(sentences)

# 유사도 행렬 계산
cos_sim = util.cos_sim(embeddings, embeddings)

print("문장 유사도 행렬:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"  [{i}]-[{j}]: {cos_sim[i][j].item():.4f}")
        print(f"    '{sentences[i]}'")
        print(f"    '{sentences[j]}'")
```

```output
문장 유사도 행렬:
  [0]-[1]: 0.8745
    '오늘 날씨가 정말 좋습니다.'
    '오늘 하늘이 맑고 화창하네요.'
  [0]-[2]: 0.1203
    '오늘 날씨가 정말 좋습니다.'
    '주식 시장이 크게 하락했습니다.'
  [1]-[2]: 0.0856
    '오늘 하늘이 맑고 화창하네요.'
    '주식 시장이 크게 하락했습니다.'
```

날씨에 관한 두 문장(0, 1)은 높은 유사도를 보이고, 주식 뉴스(2)와는 확연히 낮은 유사도를 보여줍니다. 한국어 특화 모델이 우리말의 의미를 잘 파악하고 있네요.

### 개념 4: HuggingFace 모델 허브와 MTEB 리더보드 활용하기

프로젝트에 맞는 임베딩 모델을 어떻게 찾을까요? 두 가지 핵심 도구가 있습니다.

**1. HuggingFace 모델 허브** — 10,000개 이상의 Sentence Transformers 모델이 공개되어 있습니다. `sentence-transformers` 태그로 필터링하면 바로 사용 가능한 모델 목록을 볼 수 있죠.

**2. MTEB(Massive Text Embedding Benchmark) 리더보드** — 임베딩 모델의 "수능 성적표"라고 생각하면 됩니다. 분류, 클러스터링, 검색, 유사도 등 다양한 과목별 점수를 비교할 수 있습니다.

모델 선택 시 체크리스트:

```
✅ 지원 언어가 내 데이터와 맞는가?
✅ 벡터 차원이 내 벡터 DB와 호환되는가?
✅ 최대 입력 토큰 수(max_seq_length)가 충분한가?
✅ MTEB 리더보드에서 내 태스크(Retrieval, STS 등)의 점수가 높은가?
✅ 모델 크기가 내 하드웨어(GPU 메모리)에 맞는가?
```

```python
from sentence_transformers import SentenceTransformer

# 모델의 설정 정보 확인하기
model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"최대 시퀀스 길이: {model.max_seq_length}")   # 256
print(f"출력 차원: {model.get_sentence_embedding_dimension()}")  # 384
```

> 🔥 **실무 팁**: MTEB 리더보드의 종합 1위 모델이 항상 최선은 아닙니다. RAG 검색에는 **Retrieval** 카테고리 점수가 중요하고, 의미 유사도 비교에는 **STS** 점수가 중요합니다. 자신의 태스크에 맞는 카테고리를 보세요.

### 개념 5: 로컬 추론 최적화 — GPU와 CPU에서 속도 높이기

> 💡 **비유**: 임베딩 모델 최적화는 **요리 효율화**와 같습니다. 재료를 미리 손질해두면(배치 처리) 빠르고, 간단한 레시피로 바꾸면(양자화) 맛은 거의 같으면서 시간이 절약되며, 전용 조리 도구를 쓰면(ONNX/OpenVINO) 훨씬 효율적입니다.

Sentence Transformers v5.x에서는 다양한 추론 최적화 옵션을 제공합니다:

**방법 1: 정밀도 낮추기 (FP16/BF16)**

GPU에서 가장 간단하게 속도를 높이는 방법입니다. 정확도 손실은 거의 무시할 수준이에요.

```python
from sentence_transformers import SentenceTransformer

# Float16으로 로드 — GPU에서 ~1.8x 속도 향상
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    model_kwargs={"torch_dtype": "float16"}
)

# 또는 로드 후 변환
# model = SentenceTransformer("all-MiniLM-L6-v2")
# model.half()  # FP32 → FP16 변환
```

**방법 2: ONNX 백엔드 — CPU에서 최대 3배 속도 향상**

```python
# 설치: pip install sentence-transformers[onnx]  (CPU)
# 설치: pip install sentence-transformers[onnx-gpu]  (GPU)

from sentence_transformers import SentenceTransformer

# ONNX 백엔드로 로드 — 자동으로 ONNX 변환
model = SentenceTransformer("all-MiniLM-L6-v2", backend="onnx")
embeddings = model.encode(["ONNX로 빠르게 임베딩!"])
```

**방법 3: OpenVINO 백엔드 — Intel CPU 특화**

```python
# 설치: pip install sentence-transformers[openvino]

from sentence_transformers import SentenceTransformer

# Intel CPU에 최적화된 추론
model = SentenceTransformer("all-MiniLM-L6-v2", backend="openvino")
```

**백엔드 선택 가이드**:

| 환경 | 추천 백엔드 | 기대 속도 향상 |
|------|------------|---------------|
| NVIDIA GPU + 짧은 텍스트 | ONNX-O4 | ~1.8x |
| NVIDIA GPU + 긴 텍스트 | PyTorch + FP16 | ~1.5x |
| Intel CPU | OpenVINO (+ 양자화) | ~2-3x |
| 기타 CPU | ONNX (+ int8 양자화) | ~2-3x |

**방법 4: 배치 크기 조정**

```python
# GPU 메모리가 충분하면 배치 크기를 늘려 처리량 향상
embeddings = model.encode(
    sentences,
    batch_size=128,              # 기본값 32, GPU 메모리에 따라 조절
    show_progress_bar=True,      # 대량 처리 시 진행률 표시
    normalize_embeddings=True    # 벡터를 단위 길이로 정규화 (코사인 유사도용)
)
```

> 💡 `normalize_embeddings=True`는 벡터를 L2 정규화하여 단위 길이로 만드는 옵션입니다. 이렇게 하면 내적(dot product)만으로 코사인 유사도를 계산할 수 있어 벡터 DB 검색이 빨라집니다. 정규화의 수학적 의미와 유사도 메트릭 선택에 대해서는 [5.4: 유사도 측정 방법](05-임베딩-모델-이해-텍스트를-벡터로-변환/04-유사도-측정과-벡터-검색-원리.md)에서 자세히 다룹니다.

## 실습: 직접 해보기

OpenAI API와 Sentence Transformers의 검색 성능을 직접 비교해봅시다. 동일한 한국어 질의와 문서로 유사도 검색을 수행합니다.

```python
"""
Sentence Transformers 로컬 임베딩 vs OpenAI API 비교 실습
필요 패키지: pip install sentence-transformers openai python-dotenv
"""
import time
import numpy as np
from sentence_transformers import SentenceTransformer, util

# ─── 1. 테스트 데이터 준비 ───
documents = [
    "RAG는 대규모 언어 모델의 응답에 외부 지식을 결합하는 기법입니다.",
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색합니다.",
    "텍스트 청킹은 긴 문서를 작은 단위로 나누는 전처리 과정입니다.",
    "코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지를 측정합니다.",
    "임베딩 모델은 텍스트의 의미를 고차원 벡터로 변환합니다.",
    "프롬프트 엔지니어링은 LLM에게 더 나은 답변을 유도하는 기술입니다.",
    "파인튜닝은 사전학습된 모델을 특정 도메인에 맞게 추가 학습하는 것입니다.",
    "할루시네이션은 LLM이 사실이 아닌 정보를 생성하는 현상입니다.",
]

query = "문서를 벡터로 바꾸는 방법은?"

# ─── 2. 모델별 임베딩 & 검색 ───
models_to_test = {
    "all-MiniLM-L6-v2 (384d, 영어 최적화)": "all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (768d, 최고 품질)": "all-mpnet-base-v2",
    "paraphrase-multilingual-mpnet-base-v2 (768d, 다국어)": 
        "paraphrase-multilingual-mpnet-base-v2",
}

for label, model_name in models_to_test.items():
    print(f"\n{'='*60}")
    print(f"모델: {label}")
    print(f"{'='*60}")
    
    # 모델 로드
    model = SentenceTransformer(model_name)
    
    # 임베딩 생성 시간 측정
    start = time.time()
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    query_embedding = model.encode(query, normalize_embeddings=True)
    elapsed = time.time() - start
    
    # 유사도 계산 및 랭킹
    scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    ranked = scores.argsort(descending=True)
    
    print(f"임베딩 시간: {elapsed:.3f}초")
    print(f"\n🔍 질의: '{query}'")
    print(f"\n상위 3개 검색 결과:")
    for rank, idx in enumerate(ranked[:3], 1):
        print(f"  {rank}. [{scores[idx]:.4f}] {documents[idx]}")
```

```run:python
# 간단한 모델 비교 데모 (실행 결과 확인용)
from sentence_transformers import SentenceTransformer, util
import time

documents = [
    "RAG는 대규모 언어 모델의 응답에 외부 지식을 결합하는 기법입니다.",
    "임베딩 모델은 텍스트의 의미를 고차원 벡터로 변환합니다.",
    "코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지를 측정합니다.",
]
query = "문서를 벡터로 바꾸는 방법은?"

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_emb = model.encode(documents, normalize_embeddings=True)
q_emb = model.encode(query, normalize_embeddings=True)

scores = util.cos_sim(q_emb, doc_emb)[0]
for i, (doc, score) in enumerate(zip(documents, scores)):
    print(f"[{score:.4f}] {doc}")
```

```output
[0.3821] RAG는 대규모 언어 모델의 응답에 외부 지식을 결합하는 기법입니다.
[0.6247] 임베딩 모델은 텍스트의 의미를 고차원 벡터로 변환합니다.
[0.4103] 코사인 유사도는 두 벡터의 방향이 얼마나 비슷한지를 측정합니다.
```

"임베딩 모델은 텍스트의 의미를 고차원 벡터로 변환합니다"가 가장 높은 유사도를 보입니다. "문서를 벡터로 바꾸는 방법"이라는 질의의 의미를 정확히 잡아낸 거죠.

> ⚠️ **흔한 오해**: `all-MiniLM-L6-v2`는 **영어 데이터로 학습된 모델**입니다. 한국어에서도 어느 정도 작동하지만, 프로덕션 한국어 RAG 시스템에는 다국어 모델(`paraphrase-multilingual-mpnet-base-v2`)이나 한국어 특화 모델(`jhgan/ko-sroberta-multitask`)을 사용하는 것이 훨씬 좋습니다. 위 실습의 유사도 수치는 다국어 모델에서 더 의미 있게 나올 수 있습니다.

## 더 깊이 알아보기

### SBERT의 탄생 — 65시간을 5초로 줄인 아이디어

2019년, 독일 다름슈타트 공대 UKP Lab의 Nils Reimers는 골치 아픈 문제에 부딪혔습니다. BERT가 문장 유사도를 놀라울 만큼 잘 계산하지만, 10,000개의 문장에서 가장 유사한 쌍을 찾으려면 **약 5천만 번의 추론이 필요**했고, 이는 **65시간**이나 걸렸습니다. 실시간 검색에는 사용할 수 없는 속도였죠.

Reimers의 아이디어는 간단하면서도 강력했습니다. "BERT로 유사도를 직접 비교하는 대신, 각 문장을 **독립적인 벡터로 미리 변환**해두고, 벡터 간 코사인 유사도만 계산하면 되지 않을까?" 이것이 샴 네트워크 구조를 BERT에 적용한 **Sentence-BERT**의 핵심이었고, 같은 작업을 **단 5초**로 줄였습니다.

이 논문은 EMNLP 2019에서 발표되었고, `sentence-transformers` 라이브러리로 공개되었습니다. 이후 이 라이브러리는 NLP 커뮤니티에서 폭발적으로 성장하여, 현재는 Hugging Face가 공식 관리하고 있으며, v5.2(2026년 1월 기준)까지 발전했습니다. Cross-Encoder, Sparse Encoder까지 지원하는 종합 임베딩 플랫폼으로 확장되었죠.

### 이름의 유래

"Sentence-BERT"의 줄임말 **SBERT**는 곧 라이브러리 이름 `sentence-transformers`와 도메인 `sbert.net`으로 굳어졌습니다. 재밌는 것은, 현재 이 라이브러리가 BERT뿐만 아니라 RoBERTa, MPNet, DistilBERT 등 다양한 트랜스포머 아키텍처를 지원한다는 점입니다. 이름은 BERT에서 시작했지만, 실제로는 **모든 트랜스포머 기반 문장 임베딩의 표준 프레임워크**가 된 셈이죠.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "오픈소스 모델은 OpenAI보다 항상 성능이 떨어진다." — 이것은 사실이 아닙니다. MTEB 리더보드를 보면, 최신 오픈소스 모델(예: NV-Embed-v2, Qwen3 계열)이 OpenAI 모델과 대등하거나 특정 태스크에서 앞서는 경우가 많습니다. 특히 도메인 특화 파인튜닝을 하면 범용 API를 능가할 수 있습니다.

> 💡 **알고 계셨나요?**: `all-MiniLM-L6-v2`의 이름을 해석하면 — `all`(모든 태스크 학습), `MiniLM`(Microsoft의 경량 트랜스포머), `L6`(6 레이어), `v2`(2번째 버전)입니다. 모델 이름만 읽어도 아키텍처와 성격을 알 수 있어요. 마찬가지로 `all-mpnet-base-v2`는 Microsoft의 **MPNet** 아키텍처 기반이고, `base`는 중간 크기(110M 파라미터)를 뜻합니다.

> 🔥 **실무 팁**: 대량의 문서를 임베딩할 때는 반드시 `normalize_embeddings=True`를 설정하세요. 정규화된 벡터끼리의 내적(dot product)은 코사인 유사도와 동일하므로, 벡터 DB에서 더 빠른 내적 검색을 활용할 수 있습니다. 정규화가 왜 내적과 코사인 유사도를 동일하게 만드는지에 대한 수학적 설명은 [5.4](05-임베딩-모델-이해-텍스트를-벡터로-변환/04-유사도-측정과-벡터-검색-원리.md)에서 다룹니다. `show_progress_bar=True`로 진행률을 확인하면 대량 처리 시 예상 완료 시간을 파악할 수 있습니다.

> 🔥 **실무 팁**: CPU 환경에서 대량 임베딩이 느리다면, ONNX 백엔드(`backend="onnx"`)를 먼저 시도해보세요. 코드 한 줄 변경으로 **2~3배 속도 향상**을 얻을 수 있습니다. Intel CPU라면 OpenVINO가 더 나을 수 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| SBERT 아키텍처 | 샴 네트워크 구조로 두 문장을 독립적으로 임베딩한 뒤 유사도 비교. 65시간 → 5초로 단축 |
| Mean Pooling | BERT 출력 토큰 벡터들의 평균으로 문장 벡터 생성 ([5.1](05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) 참고). SBERT의 기본 풀링 전략 |
| 대조 학습 / 삼중항 손실 | 유사 문장은 가깝게, 비유사 문장은 멀게 배치하는 학습 방식. SBERT의 높은 검색 품질의 핵심 |
| `all-MiniLM-L6-v2` | 384차원, 22M 파라미터. 속도 최우선의 경량 모델, 프로토타이핑에 적합 |
| `all-mpnet-base-v2` | 768차원, 109M 파라미터. Sentence Transformers 공식 추천 최고 품질 범용 모델 |
| 한국어 임베딩 모델 | `jhgan/ko-sroberta-multitask`, `snunlp/KR-SBERT` 등 한국어 STS/NLI로 학습된 특화 모델 |
| MTEB 리더보드 | 임베딩 모델의 표준 벤치마크. 태스크별(Retrieval, STS 등) 성능 비교 가능 |
| 추론 최적화 | FP16(GPU ~1.8x), ONNX(CPU ~3x), OpenVINO(Intel CPU), int8 양자화 등 |
| `normalize_embeddings` | 벡터를 단위 길이로 정규화하여 내적 = 코사인 유사도. 수학적 원리는 [5.4](05-임베딩-모델-이해-텍스트를-벡터로-변환/04-유사도-측정과-벡터-검색-원리.md)에서 상세 설명 |

## 다음 섹션 미리보기

지금까지 텍스트를 벡터로 변환하는 다양한 방법을 배웠습니다. 그런데 "두 벡터가 비슷하다"는 것을 정확히 어떻게 수학적으로 정의할까요? 다음 세션 [5.4: 유사도 측정 방법](05-임베딩-모델-이해-텍스트를-벡터로-변환/04-유사도-측정과-벡터-검색-원리.md)에서는 **코사인 유사도, 유클리드 거리, 내적(Dot Product)** 의 수학적 원리와 차이점을 깊이 있게 다루고, 이번 세션에서 사용한 `normalize_embeddings=True`가 왜 내적 검색을 코사인 유사도로 바꿔주는지도 명확히 설명합니다. 어떤 상황에서 어떤 유사도 메트릭을 선택해야 하는지 실전 기준을 세워보겠습니다.

## 참고 자료

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (원본 논문)](https://arxiv.org/abs/1908.10084) — SBERT의 아키텍처와 학습 방식을 제안한 2019년 EMNLP 논문. 샴 네트워크 구조의 핵심 이론을 이해할 수 있습니다
- [Sentence Transformers 공식 문서](https://sbert.net/) — 최신 v5.x API 레퍼런스, 사전학습 모델 목록, 추론 최적화 가이드를 포함한 공식 문서
- [Sentence Transformers 사전학습 모델 가이드](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) — 범용, 검색 특화, 다국어 등 목적별 추천 모델과 벤치마크 성능 비교
- [MTEB Leaderboard (HuggingFace)](https://huggingface.co/spaces/mteb/leaderboard) — 임베딩 모델의 다양한 태스크별 성능을 비교할 수 있는 표준 벤치마크 리더보드
- [Sentence Transformers 추론 속도 최적화 가이드](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) — FP16, ONNX, OpenVINO, 양자화 등 백엔드별 최적화 방법과 벤치마크
- [ko-sroberta-multitask (HuggingFace)](https://huggingface.co/jhgan/ko-sroberta-multitask) — 한국어 STS/NLI 데이터로 학습된 대표적 한국어 문장 임베딩 모델
- [Sentence Transformers GitHub 리포지토리](https://github.com/huggingface/sentence-transformers) — 소스 코드, 이슈 트래커, 릴리즈 노트를 확인할 수 있는 공식 GitHub

---
### 🔗 Related Sessions
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_space_semantics](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [sentence_embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [mean_pooling](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [text-embedding-3-small](../05-임베딩-모델-이해-텍스트를-벡터로-변환/02-openai-임베딩-api-활용.md) (prerequisite)
- [batch_embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/02-openai-임베딩-api-활용.md) (prerequisite)
