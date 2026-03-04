# 에이전틱 RAG란 — 왜 에이전트가 필요한가

> LLM이 스스로 검색 여부를 판단하고, 결과를 평가하며, 필요하면 다시 검색하는 "똑똑한 RAG"의 세계로 들어갑니다.

## 개요

이 섹션에서는 기존 RAG 파이프라인의 구조적 한계를 분석하고, 이를 해결하기 위해 등장한 에이전틱 RAG(Agentic RAG)의 핵심 아이디어를 살펴봅니다. Corrective RAG, Self-RAG, Adaptive RAG라는 세 가지 대표적인 접근법을 이해하고, 왜 "에이전트"가 RAG 시스템에 필요한지 그 근본적인 이유를 파악합니다.

**선수 지식**: 앞서 [챕터 8](08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md)~[챕터 10](10-검색-품질-향상-유사도-검색과-메타데이터-필터링/01-유사도-검색-심화-top-k와-임계값-최적화.md)에서 배운 기본 RAG 파이프라인(검색 → 생성) 구조, LangChain의 기본 체인 개념, 그리고 [챕터 12](12-리랭킹으로-검색-정확도-높이기-cohere-rerank-활용/01-리랭킹의-원리-왜-초기-검색으로는-부족한가.md)에서 다룬 리랭킹의 개념을 알고 있으면 좋습니다.

**학습 목표**:
- 기존(정적) RAG의 세 가지 근본적 한계를 설명할 수 있다
- 에이전틱 RAG의 핵심 아이디어 세 가지(검색 판단, 결과 평가, 반복 검색)를 이해한다
- Corrective RAG와 Self-RAG의 차이점과 각각의 작동 원리를 설명할 수 있다
- 에이전틱 RAG가 실무에서 어떤 문제를 해결하는지 구체적으로 이해한다

## 왜 알아야 할까?

여러분이 지금까지 구축해온 RAG 시스템을 떠올려 보세요. 사용자가 질문하면 벡터 DB에서 문서를 검색하고, 그 결과를 LLM에 넣어 답변을 생성하는 구조였죠. 이 파이프라인은 깔끔하고 직관적이지만, 실제 프로덕션 환경에서 운영하다 보면 예상치 못한 문제들이 속출합니다.

"오늘 날씨 어때?"라는 질문에도 벡터 DB를 뒤지고 있고, 검색된 문서가 질문과 전혀 관련 없는데도 그걸 기반으로 답변을 만들어내고, 한 번 검색으로 충분한 정보를 얻지 못했는데도 그냥 답변을 생성해버리는 거죠. 실무 현장의 다수 보고에 따르면, 프로덕션 RAG 시스템의 상당수가 "검색 품질 불일치"로 인한 할루시네이션 문제를 겪고 있습니다.

에이전틱 RAG는 이런 문제를 근본적으로 해결합니다. LLM에게 "생각하는 능력"을 부여해서, 검색이 필요한지 스스로 판단하고, 검색 결과가 충분한지 평가하며, 부족하면 다른 전략으로 재검색하는 시스템을 만드는 거죠. 이제 RAG는 단순한 파이프라인이 아니라 **의사결정 시스템**이 됩니다.

## 핵심 개념

### 개념 1: 정적 RAG의 세 가지 한계

> 💡 **비유**: 정적 RAG는 마치 도서관에서 일하는 신입 사서와 같습니다. 누가 무엇을 물어보든 무조건 서가로 달려가서 책을 가져오고, 그 책이 질문과 관련 있든 없든 그냥 읽어주는 사서예요. "화장실이 어디예요?"라고 물어도 서가로 뛰어가는 거죠.

기존 RAG 파이프라인에는 세 가지 구조적 한계가 있습니다.

**한계 1: 항상 검색한다 (Always Retrieve)**

모든 쿼리에 대해 무조건 벡터 검색을 수행합니다. "안녕하세요"같은 인사나, "Python에서 리스트를 정렬하는 방법"처럼 LLM이 이미 잘 알고 있는 일반 지식 질문에도 불필요하게 검색을 수행하죠. 이는 불필요한 지연 시간(latency)과 API 비용을 발생시킵니다.

**한계 2: 단일 검색이다 (Single Retrieval)**

한 번 검색하고 끝입니다. 복잡한 질문이나, 첫 번째 검색 결과가 불충분한 경우에도 추가 검색 없이 바로 답변을 생성합니다. 예를 들어 "LangChain과 LlamaIndex의 RAG 구현 방식 차이점"이라는 질문에는 양쪽 모두에 대한 문서가 필요한데, 단일 검색으로는 한쪽만 가져올 수 있습니다.

**한계 3: 검색 품질을 평가하지 않는다 (No Quality Assessment)**

검색된 문서가 실제로 질문과 관련이 있는지, 답변에 충분한 정보를 담고 있는지 평가하지 않습니다. 관련성이 낮은 문서를 그대로 컨텍스트로 넣으면, LLM이 할루시네이션을 생성할 확률이 크게 높아집니다.

```run:python
# 정적 RAG의 한계를 코드로 표현해보면
def naive_rag(query: str, retriever, llm) -> str:
    """기존 정적 RAG — 항상 검색, 한 번만, 평가 없이"""
    # 한계 1: 어떤 질문이든 무조건 검색
    docs = retriever.invoke(query)  # 항상 실행됨
    
    # 한계 2: 단 한 번만 검색
    # (결과가 부족해도 추가 검색 없음)
    
    # 한계 3: 검색 결과 품질 평가 없음
    # (관련 없는 문서도 그대로 사용)
    context = "\n".join([doc.page_content for doc in docs])
    
    response = llm.invoke(
        f"다음 컨텍스트를 기반으로 답변하세요:\n{context}\n\n질문: {query}"
    )
    return response

# 문제 상황 예시
queries = [
    "안녕하세요!",           # ← 검색 불필요한 인사
    "RAG와 파인튜닝 비교",   # ← 복잡한 질문, 다중 검색 필요
    "2025년 최신 LLM 동향",  # ← 벡터 DB에 없을 수 있는 정보
]

for q in queries:
    print(f"질문: {q}")
    print(f"  → 정적 RAG: 무조건 검색 → 한 번만 → 평가 없이 생성")
    print()
```

```output
질문: 안녕하세요!
  → 정적 RAG: 무조건 검색 → 한 번만 → 평가 없이 생성

질문: RAG와 파인튜닝 비교
  → 정적 RAG: 무조건 검색 → 한 번만 → 평가 없이 생성

질문: 2025년 최신 LLM 동향
  → 정적 RAG: 무조건 검색 → 한 번만 → 평가 없이 생성
```

세 가지 전혀 다른 성격의 질문에 정적 RAG는 모두 동일한 전략을 적용합니다. 이것이 바로 에이전틱 RAG가 해결하려는 핵심 문제입니다.

### 개념 2: 에이전틱 RAG의 핵심 아이디어

> 💡 **비유**: 에이전틱 RAG는 베테랑 사서와 같습니다. 질문을 듣고 먼저 "이건 서가에서 찾아야 할 내용인가?"를 판단하고, 책을 가져온 뒤에는 "이 책이 정말 도움이 되나?"를 평가합니다. 부족하면 다른 서가를 뒤지거나, 아예 인터넷 검색을 하기도 하죠. 답변할 준비가 됐을 때만 고객에게 돌아갑니다.

에이전틱 RAG는 정적 RAG의 세 가지 한계를 각각 정확히 해결합니다.

| 정적 RAG의 한계 | 에이전틱 RAG의 해결책 |
|---|---|
| 항상 검색 | **검색 필요성 판단** — 쿼리를 분석해 검색이 필요한지 먼저 결정 |
| 단일 검색 | **다중/반복 검색** — 부족하면 쿼리를 재작성하여 재검색 |
| 품질 무평가 | **결과 평가** — 검색된 문서의 관련성을 채점하고 필터링 |

에이전틱 RAG의 핵심은 RAG를 **파이프라인(pipeline)**이 아닌 **루프(loop)**로 바꾸는 것입니다. 직선으로 흐르는 처리 과정 대신, 조건에 따라 분기하고 반복하는 그래프 구조를 사용합니다.

```run:python
# 에이전틱 RAG의 의사결정 흐름을 의사 코드로 표현
def agentic_rag(query: str, retriever, llm, web_search) -> str:
    """에이전틱 RAG — 판단, 평가, 반복"""
    
    # 1단계: 검색 필요성 판단
    needs_retrieval = llm.classify(query)  # "검색 필요?" 판단
    
    if not needs_retrieval:
        return llm.invoke(query)  # 일반 지식으로 직접 답변
    
    # 2단계: 검색 수행
    docs = retriever.invoke(query)
    
    # 3단계: 검색 결과 품질 평가
    relevant_docs = []
    for doc in docs:
        grade = llm.grade_relevance(query, doc)  # 관련성 채점
        if grade == "relevant":
            relevant_docs.append(doc)
    
    # 4단계: 품질에 따른 분기
    if not relevant_docs:
        # 관련 문서 없음 → 쿼리 재작성 후 웹 검색
        new_query = llm.rewrite_query(query)
        docs = web_search.invoke(new_query)
        relevant_docs = docs
    
    # 5단계: 답변 생성
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return llm.invoke(f"컨텍스트:\n{context}\n\n질문: {query}")

# 같은 세 질문에 에이전틱 RAG가 내리는 결정
decisions = {
    "안녕하세요!": "검색 불필요 → LLM 직접 응답",
    "RAG와 파인튜닝 비교": "검색 필요 → 문서 평가 → 부족하면 재검색",
    "2025년 최신 LLM 동향": "검색 필요 → 벡터 DB 부족 → 웹 검색으로 전환",
}

for q, decision in decisions.items():
    print(f"질문: {q}")
    print(f"  → 에이전틱 RAG: {decision}")
    print()
```

```output
질문: 안녕하세요!
  → 에이전틱 RAG: 검색 불필요 → LLM 직접 응답

질문: RAG와 파인튜닝 비교
  → 에이전틱 RAG: 검색 필요 → 문서 평가 → 부족하면 재검색

질문: 2025년 최신 LLM 동향
  → 에이전틱 RAG: 검색 필요 → 벡터 DB 부족 → 웹 검색으로 전환
```

각 질문의 성격에 맞게 서로 다른 전략을 선택하는 것이 보이시나요? 이것이 바로 "에이전틱"의 핵심입니다.

### 개념 3: Corrective RAG (CRAG) — 검색 결과를 교정하는 RAG

> 💡 **비유**: CRAG는 시험지 채점하는 선생님과 같습니다. 학생(검색기)이 제출한 답안(문서)을 확인해서, 맞으면 활용하고, 틀리면 다른 출처(웹 검색)에서 정답을 찾아옵니다. 심지어 애매한 답안은 핵심만 골라내는 꼼꼼함도 갖추고 있죠.

Corrective RAG(CRAG)는 2024년 1월 Shi-Qi Yan 등이 발표한 논문에서 제안된 방법입니다. 핵심 아이디어는 **검색된 문서의 품질을 평가하고, 그 결과에 따라 교정 조치(corrective action)를 취하는 것**입니다.

CRAG의 작동 과정은 세 단계로 나뉩니다:

**1단계: 검색 평가 (Retrieval Evaluator)**

경량화된 평가 모델(논문에서는 T5-large를 파인튜닝)이 검색된 문서가 쿼리와 얼마나 관련 있는지 **신뢰도 점수(confidence score)**를 매깁니다. 이 점수에 따라 세 가지 판정이 내려집니다:

- **Correct** (정확함): 신뢰도가 높은 경우 → 문서를 정제(knowledge refinement)하여 활용
- **Incorrect** (부정확함): 신뢰도가 낮은 경우 → 검색 결과를 버리고 웹 검색으로 전환
- **Ambiguous** (애매함): 중간 신뢰도 → 검색 결과와 웹 검색 결과를 모두 결합

**2단계: 지식 정제 (Knowledge Refinement)**

"Correct"로 판정된 문서도 전체를 그대로 쓰지 않습니다. **분해-재조합(decompose-then-recompose)** 알고리즘을 적용하여 문서를 세밀한 단위로 쪼갠 뒤, 관련 있는 부분만 선택적으로 재조합합니다. 이렇게 하면 노이즈를 제거하고 핵심 정보만 LLM에 전달할 수 있습니다.

**3단계: 웹 검색 보강 (Web Search Augmentation)**

정적 벡터 DB만으로는 한계가 있기에, 필요한 경우 웹 검색을 통해 최신 정보나 누락된 정보를 보충합니다.

```python
# CRAG의 핵심 로직을 개념적으로 표현
from enum import Enum

class RetrievalConfidence(Enum):
    CORRECT = "correct"       # 신뢰도 높음
    INCORRECT = "incorrect"   # 신뢰도 낮음
    AMBIGUOUS = "ambiguous"   # 애매함

def corrective_rag(query: str, docs: list, evaluator, web_search, llm) -> str:
    """CRAG: 검색 결과를 평가하고 교정하는 RAG"""
    
    # 1단계: 검색 결과 평가
    confidence = evaluator.assess(query, docs)
    
    if confidence == RetrievalConfidence.CORRECT:
        # 관련 문서 → 지식 정제 후 사용
        refined_docs = knowledge_refinement(docs)
        context = refined_docs
        
    elif confidence == RetrievalConfidence.INCORRECT:
        # 관련 없음 → 웹 검색으로 전환
        web_results = web_search.invoke(query)
        context = web_results
        
    else:  # AMBIGUOUS
        # 애매함 → 정제된 문서 + 웹 검색 결합
        refined_docs = knowledge_refinement(docs)
        web_results = web_search.invoke(query)
        context = refined_docs + web_results
    
    # 2단계: 답변 생성
    return llm.generate(query, context)

def knowledge_refinement(docs: list) -> list:
    """문서를 분해하고 관련 부분만 재조합"""
    refined = []
    for doc in docs:
        # 문서를 세밀한 단위로 분해
        units = decompose(doc)
        # 관련성 높은 단위만 필터링
        relevant_units = [u for u in units if is_relevant(u)]
        # 재조합
        refined.append(recompose(relevant_units))
    return refined
```

> ⚠️ **흔한 오해**: "CRAG는 그냥 리랭킹(Reranking)과 같은 거 아닌가요?" — 아닙니다! 리랭킹([챕터 12](12-리랭킹으로-검색-정확도-높이기-cohere-rerank-활용/01-리랭킹의-원리-왜-초기-검색으로는-부족한가.md)에서 배운)은 검색 결과의 **순서만 재조정**하는 반면, CRAG는 검색 결과의 **품질 자체를 판정**하고, 부족하면 **완전히 다른 소스(웹 검색)**로 전환합니다. 리랭킹은 "좋은 순서로 정렬"이고, CRAG는 "이게 쓸 만한가?"를 판단하는 거죠.

### 개념 4: Self-RAG — 스스로 반성하는 RAG

> 💡 **비유**: Self-RAG는 자기 답안지를 스스로 채점하는 학생과 같습니다. 문제를 풀고 나서 "내가 참고한 자료가 맞았나?", "내 답변이 자료에 근거하는가?", "답변이 질문에 적합한가?"를 스스로 점검하고, 부족하면 처음부터 다시 풀죠.

Self-RAG는 2023년 10월 Akari Asai 등이 발표한 논문에서 제안되었습니다. CRAG가 "검색 결과"를 교정한다면, Self-RAG는 한 발 더 나아가 **생성 결과까지 자기 평가**합니다.

Self-RAG의 가장 독창적인 아이디어는 **반성 토큰(Reflection Token)**입니다. 일반 텍스트를 생성하는 것처럼 특수한 토큰을 생성하여, 검색과 생성 과정의 품질을 스스로 판단합니다.

Self-RAG에서 사용하는 반성 토큰은 네 종류입니다:

| 토큰 | 역할 | 판정 기준 |
|------|------|-----------|
| **Retrieve** | 검색이 필요한가? | `yes` / `no` / `continue` |
| **IsRel** | 검색된 문서가 관련 있는가? | `relevant` / `irrelevant` |
| **IsSup** | 생성된 답변이 문서에 근거하는가? | `fully supported` / `partially` / `no support` |
| **IsUse** | 생성된 답변이 질문에 유용한가? | 1~5 등급 |

```run:python
# Self-RAG의 반성 토큰 시스템을 시뮬레이션
class ReflectionTokens:
    """Self-RAG의 4가지 반성 토큰"""
    
    @staticmethod
    def retrieve(query: str) -> str:
        """검색이 필요한가?"""
        # 실제로는 학습된 모델이 판단
        if any(kw in query for kw in ["최신", "문서", "데이터"]):
            return "yes"
        return "no"
    
    @staticmethod
    def is_relevant(query: str, doc: str) -> str:
        """검색된 문서가 관련 있는가?"""
        return "relevant"  # 실제로는 모델이 판단
    
    @staticmethod
    def is_supported(answer: str, doc: str) -> str:
        """답변이 문서에 근거하는가?"""
        return "fully_supported"  # 실제로는 모델이 판단
    
    @staticmethod
    def is_useful(query: str, answer: str) -> int:
        """답변이 질문에 유용한가? (1~5)"""
        return 5  # 실제로는 모델이 판단

# Self-RAG 워크플로우 시뮬레이션
tokens = ReflectionTokens()

query = "RAG 시스템의 최신 평가 방법론은?"
print(f"질문: {query}")
print(f"[Retrieve 토큰] 검색 필요?: {tokens.retrieve(query)}")
print(f"[IsRel 토큰]   문서 관련성: {tokens.is_relevant(query, 'RAGAS 프레임워크...')}")
print(f"[IsSup 토큰]   근거 충분성: {tokens.is_supported('RAGAS는...', 'RAGAS 프레임워크...')}")
print(f"[IsUse 토큰]   답변 유용성: {tokens.is_useful(query, 'RAGAS는...')}/5")
```

```output
질문: RAG 시스템의 최신 평가 방법론은?
[Retrieve 토큰] 검색 필요?: yes
[IsRel 토큰]   문서 관련성: relevant
[IsSup 토큰]   근거 충분성: fully_supported
[IsUse 토큰]   답변 유용성: 5/5
```

Self-RAG의 진정한 힘은 이 토큰들이 **생성 과정에 자연스럽게 통합**된다는 점입니다. 별도의 평가 모델이 아니라, LLM 자체가 텍스트를 생성하면서 동시에 반성 토큰도 출력하도록 학습됩니다.

### 개념 5: Adaptive RAG — 쿼리에 맞게 전략을 선택하는 RAG

CRAG가 "검색 후 교정", Self-RAG가 "생성 후 반성"이라면, **Adaptive RAG**는 **"검색 전 라우팅"**에 해당합니다. 쿼리의 복잡도를 분석해서 아예 다른 RAG 전략을 선택하는 거죠.

Adaptive RAG는 쿼리를 세 가지로 분류합니다:

- **단순 질문** → 검색 없이 LLM 직접 응답 (예: "Python의 리스트 정렬 방법")
- **도메인 질문** → 벡터 DB에서 검색 후 생성 (예: "우리 회사의 휴가 정책은?")
- **최신/복합 질문** → 웹 검색 또는 다중 검색 (예: "2025년 AI 규제 동향")

```python
from pydantic import BaseModel, Field
from typing import Literal

class RouteQuery(BaseModel):
    """쿼리를 적절한 데이터 소스로 라우팅"""
    datasource: Literal["vectorstore", "web_search", "llm_direct"] = Field(
        description="쿼리 특성에 따라 가장 적합한 데이터 소스 선택"
    )

# LLM이 구조화된 출력으로 라우팅 결정
# 실제 구현에서는 LLM.with_structured_output(RouteQuery)를 사용
routing_examples = {
    "안녕하세요": "llm_direct",
    "우리 회사 보안 정책": "vectorstore",
    "2025년 최신 LLM 벤치마크": "web_search",
}
```

> 🔥 **실무 팁**: 실무에서는 CRAG, Self-RAG, Adaptive RAG를 개별적으로 사용하기보다, **세 가지를 결합**하는 경우가 많습니다. LangGraph를 사용하면 쿼리 라우팅(Adaptive) → 검색 결과 평가(CRAG) → 생성 결과 반성(Self-RAG)을 하나의 그래프로 엮을 수 있습니다. 다음 세션에서 LangGraph의 구조를 배우면 이 조합이 어떻게 가능한지 직접 확인하게 됩니다.

## 실습: 직접 해보기

실제 LLM 호출 없이, 에이전틱 RAG의 의사결정 로직을 시뮬레이션하는 코드를 작성해 봅시다. 이 코드는 "에이전틱 RAG가 어떤 판단을 내리는지"를 이해하는 데 초점을 맞춥니다.

```run:python
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# ─── 데이터 모델 정의 ───
class Action(Enum):
    """에이전트가 취할 수 있는 행동"""
    DIRECT_ANSWER = "직접 답변"        # 검색 불필요
    VECTOR_SEARCH = "벡터 검색"        # 벡터 DB에서 검색
    WEB_SEARCH = "웹 검색"            # 웹에서 검색
    REWRITE_QUERY = "쿼리 재작성"      # 쿼리를 바꿔서 재검색

class Relevance(Enum):
    """검색 결과의 관련성 등급"""
    HIGH = "높음"
    LOW = "낮음"
    NONE = "없음"

@dataclass
class Document:
    """검색된 문서"""
    content: str
    relevance_score: float  # 0.0 ~ 1.0

@dataclass
class AgentState:
    """에이전트의 현재 상태"""
    query: str
    action_history: list    # 지금까지 취한 행동들
    documents: list         # 검색된 문서들
    answer: Optional[str] = None
    iteration: int = 0
    max_iterations: int = 3  # 무한 루프 방지

# ─── 에이전틱 RAG 시뮬레이터 ───
class AgenticRAGSimulator:
    """에이전틱 RAG의 의사결정 과정을 시뮬레이션"""
    
    def __init__(self):
        # 시뮬레이션용 벡터 DB (실제로는 ChromaDB 등)
        self.vector_db = {
            "RAG": Document("RAG는 검색 증강 생성으로...", 0.92),
            "임베딩": Document("임베딩은 텍스트를 벡터로...", 0.88),
            "LangChain": Document("LangChain은 LLM 프레임워크...", 0.85),
        }
    
    def route_query(self, query: str) -> Action:
        """1단계: 쿼리 분석 및 라우팅 (Adaptive RAG)"""
        # 간단한 규칙 기반 라우팅 (실제로는 LLM이 판단)
        greetings = ["안녕", "hello", "hi"]
        if any(g in query.lower() for g in greetings):
            return Action.DIRECT_ANSWER
        
        if any(kw in query for kw in ["최신", "2025", "2026", "뉴스"]):
            return Action.WEB_SEARCH
        
        return Action.VECTOR_SEARCH
    
    def search(self, query: str, source: str = "vector") -> list[Document]:
        """2단계: 검색 수행"""
        if source == "vector":
            # 키워드 매칭으로 시뮬레이션
            results = []
            for key, doc in self.vector_db.items():
                if key in query:
                    results.append(doc)
            return results
        else:  # web
            return [Document("웹 검색 결과: 최신 정보...", 0.75)]
    
    def grade_documents(self, query: str, docs: list[Document]) -> Relevance:
        """3단계: 검색 결과 품질 평가 (CRAG)"""
        if not docs:
            return Relevance.NONE
        
        avg_score = sum(d.relevance_score for d in docs) / len(docs)
        if avg_score >= 0.8:
            return Relevance.HIGH
        elif avg_score >= 0.5:
            return Relevance.LOW
        else:
            return Relevance.NONE
    
    def rewrite_query(self, query: str) -> str:
        """4단계: 쿼리 재작성"""
        # 실제로는 LLM이 더 구체적인 쿼리로 변환
        return f"{query} (재작성: 더 구체적인 검색어)"
    
    def run(self, query: str) -> None:
        """에이전틱 RAG 전체 워크플로우 실행"""
        state = AgentState(
            query=query,
            action_history=[],
            documents=[],
        )
        
        print(f"{'='*55}")
        print(f"질문: {query}")
        print(f"{'='*55}")
        
        while state.iteration < state.max_iterations:
            state.iteration += 1
            print(f"\n--- 반복 {state.iteration} ---")
            
            # 1단계: 라우팅
            action = self.route_query(state.query)
            state.action_history.append(action)
            print(f"[라우팅] 결정: {action.value}")
            
            if action == Action.DIRECT_ANSWER:
                state.answer = f"(LLM 직접 응답) {state.query}에 대한 답변"
                print(f"[생성] 검색 없이 직접 답변 생성")
                break
            
            # 2단계: 검색
            source = "web" if action == Action.WEB_SEARCH else "vector"
            docs = self.search(state.query, source)
            state.documents.extend(docs)
            print(f"[검색] {source}에서 {len(docs)}건 검색됨")
            
            # 3단계: 평가 (CRAG)
            relevance = self.grade_documents(state.query, docs)
            print(f"[평가] 관련성: {relevance.value}")
            
            if relevance == Relevance.HIGH:
                state.answer = f"(검색 기반 답변) {len(docs)}건의 문서 활용"
                print(f"[생성] 고품질 문서로 답변 생성")
                break
            elif relevance == Relevance.NONE:
                # 쿼리 재작성 후 웹 검색으로 전환
                state.query = self.rewrite_query(state.query)
                print(f"[재작성] 새 쿼리: {state.query[:40]}...")
                # 다음 반복에서 웹 검색 시도
                continue
            else:
                # 애매함 → 웹 검색으로 보강
                web_docs = self.search(state.query, "web")
                state.documents.extend(web_docs)
                state.answer = f"(혼합 답변) 벡터 + 웹 결합"
                print(f"[보강] 웹 검색으로 추가 {len(web_docs)}건 확보")
                break
        
        print(f"\n📋 최종 결과:")
        print(f"  답변: {state.answer}")
        print(f"  총 행동: {[a.value for a in state.action_history]}")
        print(f"  사용 문서: {len(state.documents)}건")
        print(f"  반복 횟수: {state.iteration}회")

# ─── 다양한 쿼리로 테스트 ───
simulator = AgenticRAGSimulator()

# 테스트 1: 인사 (검색 불필요)
simulator.run("안녕하세요!")

print()

# 테스트 2: 도메인 질문 (벡터 검색)
simulator.run("RAG의 기본 개념이 뭔가요?")

print()

# 테스트 3: 최신 정보 (웹 검색)
simulator.run("2025년 최신 LLM 모델 비교")

print()

# 테스트 4: 벡터 DB에 없는 질문 (재검색)
simulator.run("HNSW 인덱스의 성능 최적화 방법")
```

```output
=======================================================
질문: 안녕하세요!
=======================================================

--- 반복 1 ---
[라우팅] 결정: 직접 답변
[생성] 검색 없이 직접 답변 생성

📋 최종 결과:
  답변: (LLM 직접 응답) 안녕하세요!에 대한 답변
  총 행동: ['직접 답변']
  사용 문서: 0건
  반복 횟수: 1회

=======================================================
질문: RAG의 기본 개념이 뭔가요?
=======================================================

--- 반복 1 ---
[라우팅] 결정: 벡터 검색
[검색] vector에서 1건 검색됨
[평가] 관련성: 높음
[생성] 고품질 문서로 답변 생성

📋 최종 결과:
  답변: (검색 기반 답변) 1건의 문서 활용
  총 행동: ['벡터 검색']
  사용 문서: 1건
  반복 횟수: 1회

=======================================================
질문: 2025년 최신 LLM 모델 비교
=======================================================

--- 반복 1 ---
[라우팅] 결정: 웹 검색
[검색] web에서 1건 검색됨
[평가] 관련성: 낮음
[보강] 웹 검색으로 추가 1건 확보

📋 최종 결과:
  답변: (혼합 답변) 벡터 + 웹 결합
  총 행동: ['웹 검색']
  사용 문서: 2건
  반복 횟수: 1회

=======================================================
질문: HNSW 인덱스의 성능 최적화 방법
=======================================================

--- 반복 1 ---
[라우팅] 결정: 벡터 검색
[검색] vector에서 0건 검색됨
[평가] 관련성: 없음
[재작성] 새 쿼리: HNSW 인덱스의 성능 최적화 방법 (재작성: 더 구체적인...

--- 반복 2 ---
[라우팅] 결정: 벡터 검색
[검색] vector에서 0건 검색됨
[평가] 관련성: 없음
[재작성] 새 쿼리: HNSW 인덱스의 성능 최적화 방법 (재작성: 더 구체적인...

--- 반복 3 ---
[라우팅] 결정: 벡터 검색
[검색] vector에서 0건 검색됨
[평가] 관련성: 없음
[재작성] 새 쿼리: HNSW 인덱스의 성능 최적화 방법 (재작성: 더 구체적인...

📋 최종 결과:
  답변: None
  총 행동: ['벡터 검색', '벡터 검색', '벡터 검색']
  사용 문서: 0건
  반복 횟수: 3회
```

마지막 테스트 케이스가 흥미롭죠? 벡터 DB에 관련 문서가 없는 경우, 재작성만 반복하다가 최대 반복 횟수에 도달합니다. 실제 에이전틱 RAG에서는 이런 경우 CRAG처럼 **웹 검색으로 자동 전환**하는 폴백(fallback) 전략이 필수입니다. 이런 세밀한 분기 로직을 체계적으로 관리하기 위해 다음 세션에서 배울 **LangGraph**가 필요한 것입니다.

## 더 깊이 알아보기

### RAG의 진화: Naive에서 Agentic까지

에이전틱 RAG를 이해하려면, RAG가 어떻게 진화해 왔는지를 살펴보는 것이 도움이 됩니다.

RAG의 시작은 2020년 Facebook AI Research(현 Meta AI)의 Patrick Lewis 등이 발표한 논문 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"입니다. 이 논문은 사전 학습된 언어 모델과 비파라메트릭 메모리(검색 가능한 문서 인덱스)를 결합하는 아이디어를 제안했습니다. 당시 사용된 검색기는 DPR(Dense Passage Retrieval)이었고, 구조는 지극히 단순했습니다. 검색하고, 생성하고, 끝.

2023년 Yunfan Gao 등의 서베이 논문 "Retrieval-Augmented Generation for Large Language Models: A Survey"에서는 RAG의 발전을 세 세대로 정리했습니다:

1. **Naive RAG** (2020~2022): 단순한 검색-생성 파이프라인. 인덱싱 → 검색 → 생성의 직선 구조
2. **Advanced RAG** (2022~2023): 검색 전/후 처리를 추가. 쿼리 변환, 리랭킹, 청킹 최적화 등
3. **Modular RAG** (2023~현재): 모듈화된 구성 요소를 자유롭게 조합. 여기에서 에이전틱 RAG가 등장

에이전틱 RAG는 Modular RAG의 가장 진보된 형태로, 2023년 하반기에 Self-RAG가 발표되고, 2024년 초에 CRAG가 뒤따르면서 본격적으로 연구 커뮤니티의 주목을 받았습니다. 특히 LangChain 팀이 LangGraph를 통해 이런 패턴들을 쉽게 구현할 수 있는 프레임워크를 제공하면서, 학술 연구에서 실무 도구로의 전환이 빠르게 이루어졌습니다.

### Self-RAG 탄생 비화

Self-RAG의 제1저자 Akari Asai는 워싱턴 대학교 박사과정 학생이었습니다. 그녀의 연구 동기는 단순했습니다. "RAG 시스템이 검색한 문서를 무비판적으로 사용하는 것이 문제"라는 것이었죠. 기존 RAG가 검색 결과를 "있는 그대로" 받아들이는 것에 반해, 인간은 참고 자료를 읽을 때 자연스럽게 "이게 맞나?", "이게 내 질문에 도움이 되나?"를 판단합니다. Self-RAG는 이 인간의 자기 반성(self-reflection) 과정을 LLM에 녹여낸 것입니다.

흥미로운 점은, Self-RAG가 별도의 "평가 모델"을 두지 않고 **하나의 LLM이 텍스트 생성과 자기 평가를 동시에** 하도록 설계했다는 것입니다. 반성 토큰(reflection token)이라는 특수 토큰을 어휘(vocabulary)에 추가하고, 이 토큰의 생성 확률을 통해 품질을 판단합니다. 논문 발표 후 ICLR 2024에서 채택되었으며, 현재 에이전틱 RAG 분야의 가장 영향력 있는 논문 중 하나로 꼽힙니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "에이전틱 RAG는 무조건 정적 RAG보다 좋다" — 그렇지 않습니다! 에이전틱 RAG는 추가적인 LLM 호출(라우팅 판단, 문서 평가, 쿼리 재작성 등)이 필요하므로 **지연 시간(latency)과 비용이 증가**합니다. 단순한 FAQ 챗봇처럼 질문과 문서의 패턴이 예측 가능한 경우, 잘 튜닝된 정적 RAG가 더 효율적일 수 있습니다. 에이전틱 RAG는 "질문의 다양성이 크고, 검색 품질이 불균일한" 시나리오에서 가장 빛을 발합니다.

> 💡 **알고 계셨나요?**: Corrective RAG 논문의 실험 결과에 따르면, CRAG를 기존 RAG에 적용했을 때 PopQA 데이터셋에서 최대 **20% 이상**의 성능 향상을 보였습니다. 특히 기존 RAG가 부정확한 문서를 검색한 경우("Incorrect" 케이스)에서 가장 극적인 개선이 나타났습니다. 이는 "검색 품질 평가" 한 단계만 추가해도 엄청난 차이를 만든다는 것을 보여줍니다.

> 🔥 **실무 팁**: 에이전틱 RAG를 처음 도입할 때는 한꺼번에 모든 기능을 구현하지 마세요. 단계적으로 접근하는 것이 효과적입니다:
> 1. **1단계**: 기존 RAG에 문서 관련성 평가(grading)만 추가 → 이것만으로도 큰 품질 개선
> 2. **2단계**: 관련성 낮을 때 쿼리 재작성(rewrite) 추가
> 3. **3단계**: 웹 검색 폴백과 쿼리 라우팅 추가
> 4. **4단계**: 생성 결과 반성(self-reflection) 추가
>
> 각 단계마다 RAGAS(챕터 17에서 다룰 예정) 등의 평가 프레임워크로 성능을 측정하고, ROI를 확인하세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 정적 RAG의 한계 | 항상 검색, 단일 검색, 품질 무평가 — 세 가지 구조적 문제 |
| 에이전틱 RAG | LLM이 검색 여부를 판단하고, 결과를 평가하며, 필요 시 재검색하는 동적 RAG |
| Corrective RAG (CRAG) | 검색 결과의 신뢰도를 평가하고 Correct/Incorrect/Ambiguous로 분류하여 교정 |
| Self-RAG | 반성 토큰(Retrieve, IsRel, IsSup, IsUse)으로 검색과 생성을 자기 평가 |
| Adaptive RAG | 쿼리 복잡도를 분석하여 직접 응답/벡터 검색/웹 검색 중 전략을 선택 |
| 파이프라인 → 루프 | 에이전틱 RAG의 핵심 전환: 직선형 처리에서 조건부 분기·반복 구조로 |
| 반성 토큰 | Self-RAG에서 생성 품질을 자동 판단하기 위해 어휘에 추가된 특수 토큰 |
| 지식 정제 | CRAG의 decompose-then-recompose로 문서에서 핵심 정보만 추출 |

## 다음 섹션 미리보기

에이전틱 RAG의 "왜"를 이해했으니, 다음 세션에서는 "어떻게" 구현하는지를 다룹니다. **LangGraph의 핵심 개념 — State, Node, Edge, Conditional Edge** — 을 배우면서, 오늘 배운 라우팅, 평가, 재검색 같은 의사결정 로직을 그래프 구조로 표현하는 방법을 익힙니다. LangGraph가 왜 에이전틱 RAG 구현의 사실상 표준이 되었는지 직접 확인하게 될 것입니다.

## 참고 자료

- [Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al., 2023)](https://arxiv.org/abs/2312.10997) - RAG의 Naive/Advanced/Modular 세 세대 분류를 제안한 종합 서베이. 에이전틱 RAG의 이론적 배경을 이해하는 데 필수
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)](https://arxiv.org/abs/2310.11511) - 반성 토큰을 제안한 Self-RAG 원본 논문. ICLR 2024 채택
- [Corrective Retrieval Augmented Generation (Yan et al., 2024)](https://arxiv.org/abs/2401.15884) - 검색 결과 평가와 교정 메커니즘을 제안한 CRAG 논문
- [Build a custom RAG agent with LangGraph — LangChain 공식 문서](https://docs.langchain.com/oss/python/langgraph/agentic-rag) - LangGraph로 에이전틱 RAG를 구현하는 공식 튜토리얼. 다음 세션의 선행 자료
- [Adaptive RAG Tutorial — LangGraph](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/) - 쿼리 라우팅과 적응형 RAG를 LangGraph로 구현하는 단계별 가이드

---
### 🔗 Related Sessions
- [embedding](../05-임베딩-모델-이해-텍스트를-벡터로-변환/01-임베딩의-기본-개념-단어에서-문장까지.md) (prerequisite)
- [vector_database](../06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md) (prerequisite)
- [reranking](../02-rag-아키텍처-핵심-컴포넌트와-파이프라인-구조/03-advanced-rag-검색-전후-최적화-전략.md) (prerequisite)
