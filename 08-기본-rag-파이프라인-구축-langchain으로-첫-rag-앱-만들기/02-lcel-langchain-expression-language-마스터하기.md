# LCEL — LangChain Expression Language 마스터하기

> LCEL의 파이프 연산자와 Runnable 인터페이스로 RAG 체인을 자유자재로 조합하는 방법을 배웁니다.

## 개요

이 섹션에서는 LangChain의 핵심 조합 문법인 LCEL(LangChain Expression Language)을 깊이 있게 다룹니다. 앞서 [세션 8.1: LangChain v1 핵심 개념과 설정](ch08/session_8_1.md)에서 `prompt | model | parser` 형태의 간단한 체인을 만들어 봤는데요, 이번에는 그 `|` 연산자가 어떻게 동작하는지, 그리고 RAG 파이프라인에서 필수적인 `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel` 등의 도구를 마스터합니다.

**선수 지식**: 세션 8.1에서 다룬 ChatModel, LCEL 파이프 연산자 기본 개념, `chain.invoke()` 사용법
**학습 목표**:
- LCEL 파이프 연산자(`|`)의 내부 동작 원리를 이해한다
- Runnable 인터페이스의 6가지 실행 메서드(`invoke`, `batch`, `stream`, `ainvoke`, `abatch`, `astream`)를 구분하여 사용할 수 있다
- `RunnablePassthrough`, `RunnableLambda`, `RunnableParallel`로 복잡한 체인을 조합할 수 있다
- LCEL 체인의 디버깅과 스트리밍 기법을 적용할 수 있다

## 왜 알아야 할까?

RAG 파이프라인은 단순히 "질문 → 답변"이 아닙니다. 실제로는 이런 흐름이죠:

1. 사용자 질문을 받아서
2. 벡터 DB에서 관련 문서를 검색하고
3. 검색 결과와 질문을 프롬프트에 조합한 뒤
4. LLM에게 전달하고
5. 응답을 파싱해서 반환

이 다섯 단계를 하나의 파이프라인으로 깔끔하게 연결하려면 LCEL이 필수입니다. LCEL 없이 이걸 구현하려면 각 단계를 일일이 호출하고, 에러 처리하고, 스트리밍 로직을 따로 만들어야 하거든요. LCEL을 쓰면 **한 줄의 체인 정의로 동기/비동기/스트리밍/배치가 전부 자동 지원**됩니다. 이것이 바로 LangChain이 다른 프레임워크와 차별화되는 핵심 기능이에요.

## 핵심 개념

### 개념 1: 파이프 연산자(`|`)의 비밀 — Unix 파이프에서 영감을 받다

> 💡 **비유**: 공장의 컨베이어 벨트를 떠올려 보세요. 첫 번째 작업자가 원재료를 다듬으면, 벨트가 자동으로 다음 작업자에게 전달하고, 다음 작업자는 조립하고, 또 다음 작업자에게 넘기죠. LCEL의 `|` 연산자가 바로 이 컨베이어 벨트입니다. 각 작업자(Runnable)는 자기 일만 하면 되고, 데이터 전달은 벨트(`|`)가 알아서 해줍니다.

LCEL의 파이프 연산자는 Unix의 파이프(`|`)와 같은 개념입니다. `ls | grep .py`처럼 왼쪽 명령의 출력이 오른쪽 명령의 입력이 되는 것이죠. Python에서는 `__or__` 매직 메서드를 오버라이드해서 이를 구현합니다.

내부적으로 `a | b`를 실행하면 Python은 `a.__or__(b)`를 호출합니다. LangChain의 모든 Runnable 클래스에는 이 메서드가 정의되어 있어서, 두 Runnable을 연결하면 자동으로 `RunnableSequence`가 생성됩니다.

```python
from langchain_core.runnables import RunnableSequence

# 이 두 표현은 완전히 동일합니다
chain_pipe = prompt | model | parser          # 파이프 연산자 방식
chain_explicit = RunnableSequence(             # 명시적 생성 방식 (내부 구현)
    first=prompt,
    middle=[model],
    last=parser
)
```

> ⚠️ **주의**: 위의 `RunnableSequence(first=..., middle=..., last=...)` 형태는 LangChain 내부 구현을 보여주기 위한 교육용 예시입니다. `first`, `middle`, `last` 파라미터는 공개 API가 아니라 내부 구현 세부사항이므로, 버전 업데이트 시 예고 없이 변경될 수 있습니다. **실제 코드에서는 항상 파이프 연산자(`|`)만 사용하세요.** 파이프 연산자가 공식적이고 안정적인 API입니다.

핵심은 **모든 LangChain 컴포넌트가 Runnable 인터페이스를 구현한다**는 것입니다. `ChatPromptTemplate`, `ChatOpenAI`, `StrOutputParser` 모두 Runnable이기 때문에 `|`로 자유롭게 연결할 수 있어요.

### 개념 2: Runnable 인터페이스 — 6가지 실행 메서드

> 💡 **비유**: Runnable 인터페이스는 만능 리모컨과 같습니다. TV가 어떤 브랜드든 리모컨의 "전원", "볼륨+", "채널" 버튼은 동일하죠. 마찬가지로 어떤 LangChain 컴포넌트든 동일한 6가지 메서드로 실행할 수 있습니다.

모든 Runnable은 다음 6가지 메서드를 제공합니다:

| 메서드 | 설명 | 용도 |
|--------|------|------|
| `invoke(input)` | 단일 입력 처리 (동기) | 일반적인 단건 호출 |
| `batch(inputs)` | 여러 입력 동시 처리 (동기) | 대량 문서 처리 |
| `stream(input)` | 결과를 토큰 단위로 스트리밍 (동기) | 실시간 UI 표시 |
| `ainvoke(input)` | 단일 입력 처리 (비동기) | 비동기 웹 서버 |
| `abatch(inputs)` | 여러 입력 동시 처리 (비동기) | 비동기 대량 처리 |
| `astream(input)` | 결과를 토큰 단위로 스트리밍 (비동기) | 비동기 실시간 UI |

```run:python
from langchain_core.runnables import RunnableLambda

# 간단한 Runnable 생성
add_exclaim = RunnableLambda(lambda x: x + "!")

# invoke: 단일 입력
result = add_exclaim.invoke("안녕하세요")
print(f"invoke 결과: {result}")

# batch: 여러 입력을 한 번에 처리
results = add_exclaim.batch(["안녕", "반가워", "잘가"])
print(f"batch 결과: {results}")
```

```output
invoke 결과: 안녕하세요!
batch 결과: ['안녕!', '반가워!', '잘가!']
```

`batch`는 내부적으로 스레드풀을 사용해서 병렬 처리합니다. I/O 바운드 작업(API 호출 등)에서는 순차 실행보다 훨씬 빠르죠. `stream`은 LLM 응답을 토큰 단위로 받을 때 특히 유용합니다:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 스트리밍: 토큰이 생성될 때마다 바로 출력
for chunk in model.stream("RAG란 무엇인가요?"):
    print(chunk.content, end="", flush=True)
```

> 🔥 **실무 팁**: 웹 애플리케이션에서 RAG를 사용할 때는 `stream` 또는 `astream`을 사용하세요. 사용자가 답변이 생성되는 과정을 실시간으로 볼 수 있어서 체감 응답 시간이 크게 줄어듭니다. ChatGPT가 글자가 타이핑되듯 나오는 것도 바로 이 스트리밍 덕분이에요.

### 개념 3: RunnablePassthrough — 데이터를 그대로 통과시키기

> 💡 **비유**: 고속도로 톨게이트를 생각해보세요. 하이패스 차량은 속도를 줄이지 않고 그대로 통과합니다. `RunnablePassthrough`가 바로 이 하이패스예요 — 입력 데이터를 아무 변형 없이 그대로 다음 단계로 넘겨줍니다.

`RunnablePassthrough`는 입력을 그대로 출력으로 전달하는 Runnable입니다. "이걸 왜 쓰지?"라고 생각할 수 있는데, RAG에서는 **검색 결과와 원래 질문을 동시에 전달**해야 하기 때문에 매우 중요합니다.

```run:python
from langchain_core.runnables import RunnablePassthrough

# 기본 동작: 입력을 그대로 반환
passthrough = RunnablePassthrough()
result = passthrough.invoke("이 값은 그대로 통과합니다")
print(f"결과: {result}")
```

```output
결과: 이 값은 그대로 통과합니다
```

더 강력한 기능은 `.assign()` 메서드입니다. 원래 입력은 그대로 유지하면서 **새로운 키를 추가**할 수 있거든요:

```run:python
from langchain_core.runnables import RunnablePassthrough

# assign: 원래 입력을 유지하면서 새 키 추가
chain = RunnablePassthrough.assign(
    length=lambda x: len(x["text"])  # "text" 키의 길이를 계산해서 추가
)

result = chain.invoke({"text": "Hello LCEL"})
print(f"결과: {result}")
# 원래의 "text" 키는 그대로 유지되고, "length" 키가 추가됨
```

```output
결과: {'text': 'Hello LCEL', 'length': 10}
```

RAG에서의 전형적인 사용 패턴을 미리 살펴볼까요? 다음 세션에서 본격적으로 구현하겠지만, 구조만 먼저 보겠습니다:

```python
from langchain_core.runnables import RunnablePassthrough

# RAG 체인의 핵심 패턴
rag_chain = (
    {
        "context": retriever | format_docs,   # 질문으로 문서 검색 후 포맷
        "question": RunnablePassthrough()      # 질문을 그대로 전달
    }
    | prompt   # context와 question을 프롬프트에 주입
    | model    # LLM 호출
    | parser   # 출력 파싱
)
```

여기서 `{"context": ..., "question": ...}` 딕셔너리 구문은 자동으로 `RunnableParallel`로 변환됩니다. `retriever`가 문서를 검색하는 동안 `RunnablePassthrough()`는 원래 질문을 그대로 넘기죠. 두 작업이 **동시에** 실행되므로 효율적입니다.

### 개념 4: RunnableLambda — 일반 함수를 Runnable로 변환

> 💡 **비유**: 해외 전자제품을 쓸 때 어댑터(변환 플러그)가 필요하죠? `RunnableLambda`는 일반 Python 함수를 LCEL 체인에 꽂을 수 있게 만들어주는 어댑터입니다.

아무리 LangChain이 다양한 컴포넌트를 제공해도, 커스텀 로직이 필요한 순간이 반드시 있습니다. 검색 결과를 특별한 형식으로 가공하거나, 입력을 전처리하거나, 출력에서 특정 부분만 추출하거나. 이때 `RunnableLambda`를 쓰면 어떤 Python 함수든 `|` 체인에 연결할 수 있습니다.

```run:python
from langchain_core.runnables import RunnableLambda

# 일반 함수를 Runnable로 변환
def format_docs(docs: list[str]) -> str:
    """검색된 문서 리스트를 하나의 문자열로 결합"""
    return "\n\n---\n\n".join(docs)

format_chain = RunnableLambda(format_docs)

# invoke로 실행
sample_docs = ["문서 1: RAG는 검색 증강 생성입니다.", "문서 2: LLM의 한계를 극복합니다."]
result = format_chain.invoke(sample_docs)
print(result)
```

```output
문서 1: RAG는 검색 증강 생성입니다.

---

문서 2: LLM의 한계를 극복합니다.
```

`RunnableLambda`의 장점은 일반 함수와 달리 `invoke`, `batch`, `stream` 등 Runnable 인터페이스 전체를 자동으로 지원한다는 점입니다:

```python
# 일반 함수는 파이프 체인에 직접 연결할 수 없음
# prompt | model | my_function  # ❌ TypeError

# RunnableLambda로 감싸면 가능
# prompt | model | RunnableLambda(my_function)  # ✅ 정상 동작
```

> ⚠️ **흔한 오해**: `RunnableLambda`에 전달하는 함수는 반드시 **하나의 인자**만 받아야 합니다. 여러 인자가 필요하면 딕셔너리로 묶어서 전달하세요. `lambda x: func(x["a"], x["b"])` 패턴을 기억하세요.

### 개념 5: RunnableParallel — 여러 작업을 동시에 실행

> 💡 **비유**: 요리를 할 때 밥을 짓는 동안 반찬을 동시에 만들죠? 밥이 다 될 때까지 기다렸다가 반찬을 시작하면 시간이 두 배로 걸릴 겁니다. `RunnableParallel`은 여러 요리를 동시에 진행해서 전체 조리 시간을 단축하는 셰프입니다.

`RunnableParallel`은 여러 Runnable을 **병렬로** 실행하고, 각각의 결과를 딕셔너리로 묶어서 반환합니다. RAG에서는 여러 검색 소스를 동시에 조회하거나, 질문과 컨텍스트를 동시에 준비할 때 사용합니다.

```run:python
from langchain_core.runnables import RunnableParallel, RunnableLambda

# 두 가지 처리를 동시에 실행
chain = RunnableParallel(
    upper=RunnableLambda(lambda x: x.upper()),     # 대문자 변환
    length=RunnableLambda(lambda x: len(x)),       # 길이 계산
    reversed=RunnableLambda(lambda x: x[::-1])     # 문자열 뒤집기
)

result = chain.invoke("hello lcel")
print(f"결과: {result}")
```

```output
결과: {'upper': 'HELLO LCEL', 'length': 10, 'reversed': 'lecl olleh'}
```

편리한 단축 문법도 있습니다. **딕셔너리를 파이프에 넣으면 자동으로 `RunnableParallel`로 변환**됩니다:

```python
# 이 두 표현은 동일합니다
# 명시적 방식
chain = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)

# 딕셔너리 단축 문법 (더 자주 사용)
chain = {
    "context": retriever,
    "question": RunnablePassthrough()
}
```

### 개념 6: 체인 조합 패턴과 디버깅

실전에서 LCEL 체인을 디버깅할 때는 두 가지 방법이 유용합니다.

**방법 1: 중간 단계에 로깅 삽입**

```python
from langchain_core.runnables import RunnableLambda

def log_step(data):
    """중간 데이터를 확인하는 디버깅용 함수"""
    print(f"[DEBUG] 데이터 타입: {type(data)}, 값: {str(data)[:100]}")
    return data  # 데이터를 그대로 반환 (통과)

debug = RunnableLambda(log_step)

# 체인 중간에 삽입해서 데이터 흐름 확인
chain = prompt | debug | model | debug | parser
```

**방법 2: `ConsoleCallbackHandler`로 전체 실행 추적**

```python
from langchain_core.callbacks import ConsoleCallbackHandler

# 체인 실행 시 콜백 핸들러 전달
result = chain.invoke(
    {"question": "RAG란?"},
    config={"callbacks": [ConsoleCallbackHandler()]}
)
```

`ConsoleCallbackHandler`는 체인의 각 단계가 시작/종료될 때마다 콘솔에 상세 정보를 출력합니다. 어떤 단계에서 에러가 발생했는지, 각 단계의 입출력이 무엇인지 한눈에 파악할 수 있어요.

> 🔥 **실무 팁**: 프로덕션 환경에서는 `ConsoleCallbackHandler` 대신 [LangSmith](https://smith.langchain.com/)를 사용하세요. 환경 변수 `LANGSMITH_TRACING=true`만 설정하면 모든 LCEL 체인의 실행이 자동으로 트레이싱됩니다. 체인 각 단계의 입출력, 지연시간, 토큰 사용량까지 웹 UI에서 확인할 수 있습니다.

## 실습: 직접 해보기

지금까지 배운 모든 개념을 종합하여, 간단한 "문서 기반 질의응답 체인"을 LCEL로 구축해봅시다. 아직 벡터 DB 연동 전이므로, 하드코딩된 문서 리스트를 사용합니다.

```python
# 필요한 패키지 설치
# pip install langchain-core langchain-openai python-dotenv

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_openai import ChatOpenAI

load_dotenv()  # .env 파일에서 OPENAI_API_KEY 로드

# ── 1단계: 컴포넌트 준비 ──

# 가상의 문서 저장소 (실제로는 벡터 DB를 사용)
DOCUMENTS = {
    "RAG": "RAG(Retrieval-Augmented Generation)는 외부 지식 소스에서 관련 정보를 "
           "검색하여 LLM의 응답을 보강하는 기법입니다. 할루시네이션을 줄이고 "
           "최신 정보를 반영할 수 있습니다.",
    "임베딩": "임베딩(Embedding)은 텍스트를 고차원 벡터로 변환하는 과정입니다. "
              "의미가 비슷한 텍스트는 벡터 공간에서 가까이 위치합니다.",
    "청킹": "청킹(Chunking)은 긴 문서를 작은 단위로 분할하는 과정입니다. "
            "적절한 청크 크기는 검색 품질에 큰 영향을 미칩니다.",
}


def fake_retriever(question: str) -> list[str]:
    """키워드 기반 간이 검색기 (실제로는 벡터 검색 사용)"""
    results = []
    for keyword, doc in DOCUMENTS.items():
        if keyword.lower() in question.lower() or keyword in question:
            results.append(doc)
    # 매칭 결과가 없으면 전체 반환
    return results if results else list(DOCUMENTS.values())


def format_docs(docs: list[str]) -> str:
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    return "\n\n".join(
        f"[문서 {i+1}] {doc}" for i, doc in enumerate(docs)
    )


# ── 2단계: 프롬프트 정의 ──
prompt = ChatPromptTemplate.from_template(
    """다음 컨텍스트를 기반으로 질문에 답변하세요.

컨텍스트:
{context}

질문: {question}

답변:"""
)

# ── 3단계: 모델과 파서 ──
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# ── 4단계: LCEL 체인 조합 ──
# RunnableLambda로 커스텀 함수를 Runnable로 변환
retrieve = RunnableLambda(fake_retriever)
format_step = RunnableLambda(format_docs)

# 전체 RAG 체인 구성
rag_chain = (
    {
        # RunnableParallel: 검색과 질문 전달을 동시에 실행
        "context": retrieve | format_step,    # 검색 → 포맷팅
        "question": RunnablePassthrough(),     # 질문을 그대로 전달
    }
    | prompt    # 컨텍스트와 질문을 프롬프트에 주입
    | model     # LLM 호출
    | parser    # 문자열로 파싱
)

# ── 5단계: 실행 ──
# invoke: 단건 실행
answer = rag_chain.invoke("RAG란 무엇인가요?")
print("=== 단건 실행 결과 ===")
print(answer)
print()

# batch: 여러 질문을 한 번에 처리
questions = ["임베딩이 뭔가요?", "청킹은 왜 필요한가요?"]
answers = rag_chain.batch(questions)
print("=== 배치 실행 결과 ===")
for q, a in zip(questions, answers):
    print(f"Q: {q}")
    print(f"A: {a}")
    print()

# stream: 토큰 단위 스트리밍
print("=== 스트리밍 결과 ===")
for chunk in rag_chain.stream("RAG의 장점은?"):
    print(chunk, end="", flush=True)
print()
```

이 실습에서 주목할 포인트:

1. **`RunnableLambda`로 일반 함수를 체인에 연결**: `fake_retriever`와 `format_docs`를 감쌌습니다
2. **딕셔너리 단축 문법**: `{...}`가 자동으로 `RunnableParallel`로 변환됩니다
3. **`RunnablePassthrough()`로 질문 보존**: 검색 작업과 병렬로 원래 질문을 전달합니다
4. **동일한 체인으로 `invoke`, `batch`, `stream` 모두 가능**: 코드를 바꿀 필요 없이 실행 방식만 바꾸면 됩니다

## 더 깊이 알아보기

### LCEL의 탄생 — SQLAlchemy에서 영감을 받다

LCEL은 2023년 8월 1일, LangChain 블로그를 통해 공식 발표되었습니다. LangChain 팀이 LCEL을 만든 이유는 명확했는데요 — 기존의 체인 시스템이 세 가지 심각한 문제를 갖고 있었기 때문입니다.

첫째, **조합성의 부재**였습니다. `LLMChain`, `ConversationalRetrievalChain` 같은 사전 제작 체인들은 서로 조합하기가 어려웠습니다. `SequentialChain`이라는 것이 존재했지만 "그다지 사용하기 좋지 않았다"고 LangChain 팀 스스로 인정했죠.

둘째, **일관성 없는 인터페이스** 문제였습니다. 각 체인이 독자적인 구현을 가지고 있어서 "모든 체인에 대해 공통 인터페이스를 적용하고, 배치·스트리밍·비동기를 동일하게 지원하기가 어려웠다"고 합니다.

셋째, **숨겨진 설정** — 프롬프트가 "다소 숨겨져 있고 변경하기 어려웠다"는 점이었습니다.

흥미로운 것은 LCEL이라는 이름 자체가 **SQLAlchemy의 Expression Language**에서 영감을 받았다는 점입니다. SQLAlchemy가 SQL 쿼리를 Python 표현식으로 조합할 수 있게 한 것처럼, LCEL은 LLM 체인을 Python의 `|` 연산자로 선언적으로 조합할 수 있게 만든 것이죠. Harrison Chase와 LangChain 팀은 이를 "체인을 진정으로 조합하는 선언적 방법"이라고 설명했습니다.

### RunnableSequence와 RunnableParallel — 두 개의 기둥

LCEL의 내부를 들여다보면, 모든 체인은 결국 두 가지 조합 프리미티브(composition primitive)로 구성됩니다:

- **`RunnableSequence`**: `a | b | c` — 순차 실행. 앞 단계의 출력이 다음 단계의 입력
- **`RunnableParallel`**: `{"x": a, "y": b}` — 병렬 실행. 동일한 입력을 여러 Runnable에 동시 전달

이 두 프리미티브를 중첩하면 아무리 복잡한 파이프라인도 표현할 수 있습니다. 마치 레고 블록의 기본 부품 두 가지로 어떤 형태든 만들 수 있는 것처럼요.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "`|` 연산자를 쓰면 모든 단계가 병렬로 실행된다"고 생각하는 분이 많습니다. 사실 `|`는 **순차(sequential)** 실행입니다. 앞 단계가 끝나야 다음 단계가 시작돼요. 병렬 실행은 `RunnableParallel`(또는 딕셔너리 문법)을 사용해야 합니다. `a | b`는 "a 다음 b", `{"x": a, "y": b}`는 "a와 b를 동시에" — 이 차이를 꼭 기억하세요.

> 💡 **알고 계셨나요?**: LCEL 체인은 `chain.get_graph().print_ascii()`로 전체 실행 그래프를 ASCII 아트로 시각화할 수 있습니다. 복잡한 체인의 구조를 파악할 때 매우 유용한 디버깅 도구인데, 의외로 모르는 분이 많습니다.

> 🔥 **실무 팁**: `RunnableLambda`에 전달하는 함수가 비동기(`async def`)이면, `ainvoke`/`astream` 호출 시 자동으로 비동기로 실행됩니다. 동기 함수와 비동기 함수를 하나의 체인에 섞어도 LangChain이 알아서 처리하니, 비동기 전환이 필요할 때 체인 전체를 다시 작성할 필요가 없어요.

> ⚠️ **흔한 오해**: `RunnablePassthrough()`와 `RunnablePassthrough.assign()`은 다릅니다. 전자는 입력을 그대로 반환하고, 후자는 입력을 유지하면서 **새로운 키-값 쌍을 추가**합니다. RAG 체인에서는 보통 `assign`을 사용해 검색 결과를 기존 입력에 덧붙이는 패턴이 많습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `\|` (파이프 연산자) | 두 Runnable을 **순차적으로** 연결. 내부적으로 `RunnableSequence` 생성 |
| Runnable 인터페이스 | `invoke`, `batch`, `stream` + 비동기 버전 3가지 = 총 6가지 실행 메서드 |
| `RunnablePassthrough` | 입력을 그대로 통과시키는 Runnable. `.assign()`으로 새 키 추가 가능 |
| `RunnableLambda` | 일반 Python 함수를 Runnable로 변환하는 어댑터 |
| `RunnableParallel` | 여러 Runnable을 병렬 실행. 딕셔너리 `{...}` 단축 문법 지원 |
| `RunnableSequence` | `a \| b`의 결과로 생성되는 순차 실행 체인 |
| RAG 체인 패턴 | `{"context": retriever, "question": RunnablePassthrough()} \| prompt \| model \| parser` |
| 디버깅 | `RunnableLambda(log_fn)` 삽입 또는 `ConsoleCallbackHandler` 사용 |

## 다음 섹션 미리보기

LCEL의 조합 문법을 마스터했으니, 다음 세션에서는 드디어 **실제 벡터 데이터베이스와 연결하여 Retriever를 구성**합니다. 앞서 [챕터 6](06-벡터-데이터베이스-기초-chromadb로-시작하기/01-벡터-데이터베이스란-왜-필요한가.md)에서 배운 ChromaDB를 LangChain의 `VectorStoreRetriever`로 감싸고, 이번에 배운 LCEL 패턴(`RunnablePassthrough` + `RunnableParallel`)을 활용해 진짜 작동하는 RAG 체인을 완성합니다.

## 참고 자료

- [LangChain Expression Language (LCEL) 공식 개념 문서](https://python.langchain.com/docs/concepts/lcel/) — LCEL의 설계 철학과 핵심 개념을 설명하는 공식 문서
- [LangChain Expression Language 발표 블로그](https://blog.langchain.com/langchain-expression-language/) — 2023년 8월 LCEL 탄생 배경과 설계 동기를 설명하는 공식 블로그 포스트
- [LCEL Interface — LangChain OpenTutorial](https://langchain-opentutorial.gitbook.io/langchain-opentutorial/01-basic/07-lcel-interface) — invoke, batch, stream 등 Runnable 인터페이스의 실습 예제를 제공하는 커뮤니티 튜토리얼
- [LangChain Expression Language Explained — Pinecone](https://www.pinecone.io/learn/series/langchain/langchain-expression-language/) — RunnablePassthrough, RunnableLambda, RunnableParallel의 상세 설명과 RAG 패턴 예제
- [Runnable API Reference](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.Runnable.html) — Runnable 인터페이스의 전체 메서드 목록과 타입 정의

---
### 🔗 Related Sessions
- [lcel](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
- [chatmodel](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
- [chatprompttemplate](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
- [stroutputparser](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
- [langchain-core](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
- [langchain-openai](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
