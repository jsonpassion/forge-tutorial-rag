# FastAPI로 RAG API 서빙

> 개발 환경의 RAG 파이프라인을 프로덕션 API로 변환하여 누구나 사용할 수 있는 서비스로 만드는 방법을 배웁니다.

## 개요

지금까지 우리는 Jupyter 노트북이나 스크립트에서 RAG 파이프라인을 실행해왔습니다. 하지만 실제 서비스에서는 웹 앱, 모바일 앱, 다른 시스템이 HTTP 요청으로 RAG를 호출할 수 있어야 하죠. 이 세션에서는 FastAPI를 사용해 RAG 파이프라인을 REST API로 감싸고, 비동기 처리와 스트리밍 응답까지 구현합니다. 나아가 [Ch16에서 배운 에이전틱 RAG](ch16/session1)처럼 LangGraph 기반의 복잡한 그래프를 서빙할 때의 스트리밍 패턴과 상태 관리 전략도 다룹니다.

**선수 지식**: [Ch18 Session 18.1: RAG 파이프라인 디버깅](ch18/session1)에서 다룬 파이프라인 구조와 디버깅 기법, [Ch8 Session 8.1: 기본 RAG 파이프라인 구축](ch08/session1)에서 만든 LangChain 기반 RAG 체인, [Ch16: 에이전틱 RAG](ch16/session1)에서 배운 LangGraph 기반 에이전트 구조
**학습 목표**:
- FastAPI로 RAG 파이프라인을 REST API 엔드포인트로 래핑할 수 있다
- Pydantic v2를 사용해 요청/응답 스키마를 설계할 수 있다
- `async/await`와 `StreamingResponse`로 실시간 토큰 스트리밍을 구현할 수 있다
- LangGraph 기반 에이전틱 RAG를 `astream_events`로 서빙하고 상태를 관리할 수 있다
- 프로덕션 수준의 프로젝트 구조를 설계할 수 있다

## 왜 알아야 할까?

여러분이 만든 RAG 시스템이 아무리 정확하고 빠르더라도, 노트북 안에 갇혀 있으면 여러분만 쓸 수 있습니다. 프론트엔드 개발자가 "그 RAG 기능, API로 호출하고 싶은데요"라고 하면 어떻게 할까요?

실제로 기업에서 AI 프로젝트가 PoC(개념 증명)에서 프로덕션으로 넘어가지 못하는 가장 흔한 이유 중 하나가 바로 **서빙(Serving)** 문제입니다. 모델은 잘 동작하는데, 그것을 안정적인 API로 제공하는 방법을 모르는 거죠. 특히 [Ch16에서 다룬 에이전틱 RAG](ch16/session1)처럼 LangGraph 그래프가 여러 도구를 호출하며 분기하는 복잡한 파이프라인은, 단순한 LCEL 체인보다 서빙 난이도가 훨씬 높습니다. FastAPI는 Python 생태계에서 가장 인기 있는 비동기 웹 프레임워크로, LLM 기반 애플리케이션 서빙에 특히 적합합니다. 비동기 처리, 자동 문서 생성, 타입 검증까지 — RAG API 서빙에 필요한 모든 것을 제공하거든요.

## 핵심 개념

### 개념 1: FastAPI 프로젝트 구조 설계

> 💡 **비유**: RAG API를 레스토랑이라고 생각해보세요. 주방(RAG 파이프라인)이 아무리 훌륭해도, 홀(API 서버)이 체계적이지 않으면 손님(클라이언트)에게 좋은 서비스를 제공할 수 없습니다. 메뉴판(API 문서), 주문 시스템(요청 스키마), 서빙 동선(라우터 구조)이 잘 설계되어야 하죠.

프로덕션 RAG API는 단일 파일이 아닌 모듈 기반 구조로 설계해야 합니다. 각 관심사를 분리하면 유지보수와 테스트가 훨씬 쉬워집니다.

> 📊 **그림 1**: FastAPI RAG 프로젝트의 디렉토리 구조

```mermaid
graph TD
    A["📁 rag-api/"] --> B["📄 main.py<br/>앱 진입점"]
    A --> C["📁 app/"]
    C --> D["📁 api/"]
    C --> E["📁 core/"]
    C --> F["📁 services/"]
    C --> G["📁 schemas/"]
    D --> D1["📄 routes.py<br/>엔드포인트 정의"]
    D --> D2["📄 deps.py<br/>의존성 주입"]
    E --> E1["📄 config.py<br/>환경 설정"]
    E --> E2["📄 lifespan.py<br/>앱 생명주기"]
    F --> F1["📄 rag_chain.py<br/>RAG 파이프라인"]
    F --> F2["📄 vector_store.py<br/>벡터 DB 연결"]
    G --> G1["📄 request.py<br/>요청 모델"]
    G --> G2["📄 response.py<br/>응답 모델"]
```

핵심 디렉토리 역할을 살펴보겠습니다:

- **`app/api/`**: HTTP 엔드포인트와 라우터를 정의합니다. 비즈니스 로직은 여기에 넣지 않습니다.
- **`app/core/`**: 설정, 앱 생명주기(lifespan), 미들웨어 등 앱 전반의 기반 코드입니다.
- **`app/services/`**: RAG 체인, 벡터 스토어 연결 등 실제 비즈니스 로직이 위치합니다.
- **`app/schemas/`**: Pydantic 모델로 정의한 요청/응답 스키마입니다.

```python
# app/core/config.py — 환경 설정 관리
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정 — .env 파일에서 자동 로드"""
    
    app_name: str = "RAG API"
    openai_api_key: str  # 필수 — 없으면 시작 시 에러
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4

    model_config = {"env_file": ".env"}


settings = Settings()
```

> 🔥 **실무 팁**: `pydantic_settings`의 `BaseSettings`를 사용하면 환경 변수를 타입 안전하게 관리할 수 있습니다. `.env` 파일에 `OPENAI_API_KEY=sk-...`만 넣어두면 자동으로 읽히거든요. 절대 코드에 API 키를 하드코딩하지 마세요!

### 개념 2: RAG 파이프라인을 REST API로 래핑

> 💡 **비유**: RAG 파이프라인을 API로 래핑하는 것은, 집에서 만든 요리를 배달 용기에 담는 것과 같습니다. 요리(파이프라인) 자체는 바뀌지 않지만, 외부 소비자가 사용할 수 있도록 표준화된 형태(HTTP 인터페이스)로 포장하는 거죠.

먼저 RAG 서비스를 클래스로 캡슐화합니다:

```python
# app/services/rag_chain.py — RAG 파이프라인 서비스
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings


class RAGService:
    """RAG 파이프라인을 관리하는 서비스 클래스"""

    def __init__(self):
        # 임베딩 모델 초기화
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        # 벡터 스토어 연결
        self.vector_store = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings,
        )
        # 리트리버 설정
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": settings.top_k}
        )
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
        # RAG 체인 구성
        self.chain = self._build_chain()

    def _build_chain(self):
        """LCEL로 RAG 체인 구성"""
        prompt = ChatPromptTemplate.from_template(
            "다음 컨텍스트를 기반으로 질문에 답하세요.\n\n"
            "컨텍스트:\n{context}\n\n"
            "질문: {question}\n\n"
            "답변:"
        )

        # 문서 리스트를 텍스트로 변환하는 헬퍼
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    async def query(self, question: str) -> dict:
        """질문에 대한 RAG 응답 반환"""
        # ainvoke로 비동기 실행
        answer = await self.chain.ainvoke(question)
        
        # 소스 문서도 함께 반환
        docs = await self.retriever.ainvoke(question)
        sources = [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in docs
        ]
        return {"answer": answer, "sources": sources}
```

그리고 앱 생명주기(lifespan)에서 RAG 서비스를 초기화합니다. FastAPI의 `lifespan` 패턴을 사용하면 앱 시작 시 한 번만 무거운 리소스를 로드하고, 종료 시 정리할 수 있습니다:

```python
# app/core/lifespan.py — 앱 생명주기 관리
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.services.rag_chain import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 관리"""
    # 시작 시: RAG 서비스 초기화 (벡터 DB 로드 포함)
    app.state.rag_service = RAGService()
    print("✅ RAG 서비스 초기화 완료")
    
    yield  # 앱 실행 중
    
    # 종료 시: 리소스 정리
    print("🔄 RAG 서비스 종료")
```

> 📊 **그림 2**: FastAPI 앱의 요청 처리 흐름

```mermaid
sequenceDiagram
    participant C as 클라이언트
    participant F as FastAPI
    participant V as 스키마 검증<br/>(Pydantic)
    participant R as RAG 서비스
    participant DB as 벡터 DB
    participant LLM as LLM API

    C->>F: POST /api/query
    F->>V: 요청 바디 검증
    V-->>F: 검증 통과
    F->>R: query(question)
    R->>DB: 유사도 검색
    DB-->>R: 관련 문서 반환
    R->>LLM: 프롬프트 + 컨텍스트
    LLM-->>R: 생성된 답변
    R-->>F: answer + sources
    F-->>C: JSON 응답
```

이제 라우터를 정의합니다:

```python
# app/api/routes.py — API 엔드포인트 정의
from fastapi import APIRouter, Request, HTTPException

from app.schemas.request import QueryRequest
from app.schemas.response import QueryResponse

router = APIRouter(prefix="/api", tags=["RAG"])


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: Request, body: QueryRequest):
    """RAG 파이프라인에 질문을 전달하고 답변을 반환합니다."""
    rag_service = request.app.state.rag_service
    
    try:
        result = await rag_service.query(body.question)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 처리 실패: {str(e)}")
```

### 개념 3: Pydantic v2로 요청/응답 스키마 설계

> 💡 **비유**: API 스키마는 주문서와 같습니다. 손님(클라이언트)이 "아메리카노 한 잔, 아이스, 톨 사이즈"라고 정확히 적어야 하고, 영수증(응답)에도 주문 내용과 가격이 명확히 표시되어야 하잖아요. 스키마가 명확하면 소통 오류가 줄어듭니다.

Pydantic v2는 Rust로 작성된 코어 덕분에 v1 대비 5~50배 빠른 검증 성능을 제공합니다. FastAPI는 Pydantic v2를 기본으로 사용하므로, 스키마를 정의하면 자동으로 입력 검증과 OpenAPI 문서 생성이 이루어집니다:

```python
# app/schemas/request.py — 요청 스키마
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """RAG 질의 요청"""
    question: str = Field(
        ...,  # 필수 필드
        min_length=1,
        max_length=1000,
        description="RAG 시스템에 전달할 질문",
        examples=["LangChain의 LCEL이란 무엇인가요?"],
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="검색할 문서 수",
    )
    stream: bool = Field(
        default=False,
        description="스트리밍 응답 여부",
    )
```

```python
# app/schemas/response.py — 응답 스키마
from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """검색된 소스 문서 정보"""
    content: str = Field(description="문서 내용 미리보기")
    metadata: dict = Field(default_factory=dict, description="문서 메타데이터")


class QueryResponse(BaseModel):
    """RAG 질의 응답"""
    answer: str = Field(description="생성된 답변")
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="참조한 소스 문서 목록",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "LCEL은 LangChain Expression Language의 약자로...",
                    "sources": [
                        {
                            "content": "LCEL은 파이프 연산자(|)를 사용하여...",
                            "metadata": {"source": "langchain_docs.pdf", "page": 42},
                        }
                    ],
                }
            ]
        }
    }
```

스키마를 이렇게 정의하면 FastAPI가 자동으로 `/docs` (Swagger UI)와 `/redoc` 페이지를 생성합니다. 프론트엔드 개발자가 별도 문서 없이도 API를 이해하고 테스트할 수 있죠.

### 개념 4: 비동기 엔드포인트와 StreamingResponse

RAG API에서 가장 시간이 오래 걸리는 부분은 LLM의 응답 생성입니다. 사용자가 10초 넘게 빈 화면을 바라보게 하는 것보다, 토큰이 생성되는 대로 하나씩 보여주는 게 훨씬 좋은 경험이겠죠? 이것이 바로 **스트리밍 응답**입니다.

> 📊 **그림 3**: 일반 응답 vs 스트리밍 응답 비교

```mermaid
flowchart LR
    subgraph 일반["일반 응답 (한 번에)"]
        A1["요청"] --> B1["⏳ 대기 5~15초"]
        B1 --> C1["전체 답변 수신"]
    end
    
    subgraph 스트리밍["스트리밍 응답 (토큰 단위)"]
        A2["요청"] --> B2["첫 토큰 ~0.5초"]
        B2 --> C2["토큰 연속 수신"]
        C2 --> D2["완료"]
    end
```

FastAPI의 `StreamingResponse`와 LangChain의 `astream`을 조합하면 토큰 단위 스트리밍을 구현할 수 있습니다:

```python
# app/api/routes.py에 스트리밍 엔드포인트 추가
import json
from fastapi.responses import StreamingResponse


async def _stream_rag_response(rag_service, question: str):
    """RAG 응답을 Server-Sent Events로 스트리밍"""
    # 먼저 소스 문서를 검색해서 전송
    docs = await rag_service.retriever.ainvoke(question)
    sources = [
        {"content": doc.page_content[:200], "metadata": doc.metadata}
        for doc in docs
    ]
    # 소스 정보를 첫 이벤트로 전송
    yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"

    # LLM 응답을 토큰 단위로 스트리밍
    async for chunk in rag_service.chain.astream(question):
        yield f"data: {json.dumps({'type': 'token', 'data': chunk}, ensure_ascii=False)}\n\n"

    # 스트림 종료 신호
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/query/stream")
async def query_rag_stream(request: Request, body: QueryRequest):
    """RAG 응답을 스트리밍으로 반환합니다 (SSE)."""
    rag_service = request.app.state.rag_service
    
    return StreamingResponse(
        _stream_rag_response(rag_service, body.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
```

여기서 핵심은 `async for chunk in rag_service.chain.astream(question)` 부분입니다. LangChain의 LCEL 체인은 `astream` 메서드를 기본 제공하므로, 별도 콜백 없이도 토큰 단위 스트리밍이 가능합니다.

> ⚠️ **흔한 오해**: `StreamingResponse`를 쓰면 자동으로 스트리밍이 된다고 생각하기 쉽습니다. 하지만 내부 함수가 실제로 **제너레이터(generator)** 여야 합니다. 일반 함수에서 전체 결과를 한번에 반환하면 `StreamingResponse`로 감싸도 스트리밍 효과가 없어요.

### 개념 5: LangGraph 에이전틱 RAG 서빙 — astream_events와 상태 관리

> 💡 **비유**: 앞서 만든 LCEL 체인이 "한 줄로 쭉 이어진 컨베이어 벨트"라면, [Ch16에서 배운 LangGraph 에이전틱 RAG](ch16/session1)는 "여러 갈래로 분기하는 물류 센터"에 가깝습니다. 물류 센터에서는 어떤 택배가 어디를 거쳐가는지 추적해야 하고, 동시에 여러 라인이 돌아가니 상태 관리가 핵심이죠.

LCEL 체인의 `astream`은 최종 LLM 출력만 토큰 단위로 스트리밍합니다. 하지만 에이전틱 RAG에서는 그래프가 **검색 → 판단 → 도구 호출 → 재검색 → 답변 생성**처럼 여러 노드를 거치는데, 사용자에게 "지금 어떤 단계를 실행 중인지"를 실시간으로 알려줘야 좋은 경험을 제공할 수 있습니다. 이럴 때 필요한 것이 바로 `astream_events`입니다.

> 📊 **그림 5**: LCEL astream vs LangGraph astream_events 비교

```mermaid
flowchart TD
    subgraph LCEL["LCEL 체인 — astream"]
        L1["사용자 질문"] --> L2["리트리버"]
        L2 --> L3["LLM"]
        L3 -->|"토큰 스트리밍"| L4["클라이언트"]
    end

    subgraph LG["LangGraph 에이전트 — astream_events"]
        G1["사용자 질문"] --> G2["라우터 노드"]
        G2 -->|"이벤트: 노드 시작"| G3["검색 도구"]
        G3 -->|"이벤트: 도구 결과"| G4["평가 노드"]
        G4 -->|"이벤트: 조건 분기"| G5["재검색 or 답변"]
        G5 -->|"이벤트: LLM 토큰"| G6["클라이언트"]
    end
```

#### astream_events로 세밀한 스트리밍 구현

`astream_events`는 LangGraph 그래프 내부에서 발생하는 **모든 이벤트**를 실시간으로 방출합니다. 노드 시작/종료, 도구 호출, LLM 토큰 생성 등 각 단계를 개별 이벤트로 받을 수 있죠:

```python
# app/services/agentic_rag.py — LangGraph 에이전틱 RAG 서비스
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

from app.core.config import settings


class AgentState(TypedDict):
    """에이전트 그래프의 상태 스키마"""
    messages: Annotated[list, add_messages]
    retrieved_docs: list[str]
    need_more_search: bool


class AgenticRAGService:
    """LangGraph 기반 에이전틱 RAG 서비스"""

    def __init__(self, retriever, tools):
        self.retriever = retriever
        self.tools = tools
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        ).bind_tools(tools)
        # 그래프를 한 번만 컴파일 — 매 요청마다 재컴파일하지 않음
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """에이전틱 RAG 그래프 구성 (Ch16에서 배운 패턴)"""
        builder = StateGraph(AgentState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("grade", self._grade_node)
        builder.add_node("generate", self._generate_node)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges(
            "grade",
            lambda s: "retrieve" if s["need_more_search"] else "generate",
        )
        builder.add_edge("generate", END)
        return builder.compile()

    async def _retrieve_node(self, state: AgentState):
        # 검색 로직 (생략)
        ...

    async def _grade_node(self, state: AgentState):
        # 문서 관련성 평가 (생략)
        ...

    async def _generate_node(self, state: AgentState):
        # LLM 답변 생성 (생략)
        ...
```

이제 핵심인 스트리밍 엔드포인트입니다. `astream_events`는 `version="v2"`를 지정해야 하며, 각 이벤트의 `event` 필드로 이벤트 종류를, `name`으로 어떤 노드/도구에서 발생했는지를 구분합니다:

```python
# app/api/routes.py — 에이전틱 RAG 스트리밍 엔드포인트
import json
from fastapi.responses import StreamingResponse


async def _stream_agentic_response(agentic_service, question: str, thread_id: str):
    """LangGraph 에이전틱 RAG를 astream_events로 스트리밍"""
    # thread_id로 대화 상태를 격리 (동시 사용자 지원)
    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=question)]}

    async for event in agentic_service.graph.astream_events(
        input_state, config=config, version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "")

        # 노드 실행 시작 → 클라이언트에 현재 단계 알림
        if kind == "on_chain_start" and name in ("retrieve", "grade", "generate"):
            yield f"data: {json.dumps({'type': 'step', 'data': f'{name} 단계 실행 중...'}, ensure_ascii=False)}\n\n"

        # 도구 호출 완료 → 어떤 도구가 어떤 결과를 반환했는지
        elif kind == "on_tool_end":
            tool_output = event["data"].output if hasattr(event["data"], "output") else str(event["data"])
            yield f"data: {json.dumps({'type': 'tool_result', 'tool': name, 'data': tool_output[:300]}, ensure_ascii=False)}\n\n"

        # LLM 토큰 스트리밍 → 최종 답변 생성 단계
        elif kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                yield f"data: {json.dumps({'type': 'token', 'data': token}, ensure_ascii=False)}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/query/agent/stream")
async def query_agentic_stream(request: Request, body: QueryRequest):
    """LangGraph 에이전틱 RAG 응답을 스트리밍합니다."""
    agentic_service = request.app.state.agentic_rag_service

    # thread_id로 동시 요청의 상태를 격리
    import uuid
    thread_id = body.thread_id if hasattr(body, "thread_id") and body.thread_id else str(uuid.uuid4())

    return StreamingResponse(
        _stream_agentic_response(agentic_service, body.question, thread_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
```

#### 상태 관리 전략: 동시 사용자와 대화 지속성

에이전틱 RAG를 서빙할 때 가장 까다로운 부분이 **상태 관리**입니다. LCEL 체인은 상태가 없어서(stateless) 요청마다 독립적으로 처리하면 되지만, LangGraph 그래프는 노드 간에 상태를 공유하고, 멀티턴 대화에서는 이전 대화 맥락까지 유지해야 하거든요.

> 📊 **그림 6**: 에이전틱 RAG 상태 관리 아키텍처

```mermaid
flowchart TD
    U1["사용자 A<br/>thread_id: abc"] -->|"요청"| API["FastAPI"]
    U2["사용자 B<br/>thread_id: xyz"] -->|"요청"| API

    API --> CS["체크포인터<br/>(상태 저장소)"]

    CS -->|"thread abc 상태 로드"| G1["그래프 실행<br/>(사용자 A)"]
    CS -->|"thread xyz 상태 로드"| G2["그래프 실행<br/>(사용자 B)"]

    G1 -->|"상태 저장"| CS
    G2 -->|"상태 저장"| CS
```

LangGraph는 `thread_id` 기반의 체크포인터(Checkpointer)로 이 문제를 해결합니다. 프로덕션에서는 인메모리 대신 영속적 저장소를 사용해야 합니다:

```python
# app/core/lifespan.py — 에이전틱 RAG 포함 버전
from contextlib import asynccontextmanager
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.services.rag_chain import RAGService
from app.services.agentic_rag import AgenticRAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 관리"""
    # 기본 RAG 서비스
    app.state.rag_service = RAGService()

    # 에이전틱 RAG — PostgreSQL 체크포인터로 상태 영속화
    async with AsyncPostgresSaver.from_conn_string(
        "postgresql://user:pass@localhost/rag_state"
    ) as checkpointer:
        await checkpointer.setup()  # 테이블 자동 생성
        agentic_service = AgenticRAGService(
            retriever=app.state.rag_service.retriever,
            tools=[...],  # 검색, 계산 등 도구 목록
        )
        # 컴파일된 그래프에 체크포인터 연결
        agentic_service.graph = agentic_service._build_graph()
        # compile 시 checkpointer 전달
        agentic_service.graph = agentic_service.graph  # 이미 compile됨
        app.state.agentic_rag_service = agentic_service
        print("✅ 에이전틱 RAG 서비스 초기화 완료")

        yield

    print("🔄 서비스 종료")
```

> ⚠️ **흔한 오해**: "LangGraph 그래프를 매 요청마다 `compile()`하면 되겠지"라고 생각하기 쉽습니다. 하지만 그래프 컴파일은 비용이 큰 작업이에요. **반드시 `lifespan`에서 한 번만 컴파일**하고, `thread_id`로 각 사용자의 상태를 격리하세요. 그래프 구조는 공유하되 상태만 분리하는 패턴이 핵심입니다.

> 🔥 **실무 팁**: 에이전틱 RAG는 도구를 여러 번 호출하면서 지연 시간이 누적됩니다. 예를 들어 검색 → 평가 → 재검색 → 답변으로 4단계를 거치면 각 LLM 호출이 2~3초씩, 총 8~12초가 걸릴 수 있죠. `astream_events`로 각 단계의 진행 상황을 실시간 전송하면, 사용자가 "지금 검색 중...", "문서 평가 중..."처럼 진행 상태를 볼 수 있어 체감 대기 시간이 크게 줄어듭니다.

## 실습: 직접 해보기

이제 모든 조각을 합쳐서 완전한 RAG API 서버를 만들어보겠습니다. 먼저 필요한 패키지를 설치합니다:

```python
# requirements.txt
# fastapi[standard]>=0.115.0
# uvicorn[standard]>=0.32.0
# pydantic-settings>=2.5.0
# langchain-openai>=0.3.0
# langchain-chroma>=0.2.0
# langchain-core>=0.3.0
# langgraph>=0.2.0
# langgraph-checkpoint-postgres>=0.2.0  # 에이전틱 RAG 상태 관리용
```

메인 앱 파일입니다:

```python
# main.py — FastAPI 앱 진입점
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.lifespan import lifespan
from app.api.routes import router


def create_app() -> FastAPI:
    """FastAPI 앱 팩토리"""
    app = FastAPI(
        title="RAG API",
        description="검색 증강 생성(RAG) 기반 질의응답 API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS 설정 — 프론트엔드에서 호출 허용
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    app.include_router(router)

    # 헬스체크 엔드포인트
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app


app = create_app()
```

서버를 실행하고 테스트해봅시다:

```run:python
# 서버 실행 예시 (실제로는 터미널에서 실행)
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 클라이언트 테스트 코드
import json

# 1. 일반 요청 예시
request_body = {
    "question": "RAG에서 청킹이 중요한 이유는?",
    "top_k": 4,
    "stream": False,
}
print("=== 일반 요청 ===")
print(f"POST /api/query")
print(f"Body: {json.dumps(request_body, ensure_ascii=False, indent=2)}")
print()

# 2. 응답 예시
response = {
    "answer": "청킹은 긴 문서를 검색에 적합한 단위로 나누는 과정입니다...",
    "sources": [
        {
            "content": "텍스트 청킹은 RAG 파이프라인의 핵심 전처리 단계로...",
            "metadata": {"source": "rag_guide.pdf", "page": 15},
        }
    ],
}
print("=== 응답 ===")
print(json.dumps(response, ensure_ascii=False, indent=2))
```

```output
=== 일반 요청 ===
POST /api/query
Body: {
  "question": "RAG에서 청킹이 중요한 이유는?",
  "top_k": 4,
  "stream": false
}

=== 응답 ===
{
  "answer": "청킹은 긴 문서를 검색에 적합한 단위로 나누는 과정입니다...",
  "sources": [
    {
      "content": "텍스트 청킹은 RAG 파이프라인의 핵심 전처리 단계로...",
      "metadata": {
        "source": "rag_guide.pdf",
        "page": 15
      }
    }
  ]
}
```

스트리밍 응답을 클라이언트에서 소비하는 코드도 살펴보겠습니다:

```python
# 스트리밍 클라이언트 예시 (httpx 사용)
import httpx


async def consume_stream():
    """SSE 스트리밍 응답을 소비하는 클라이언트"""
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/api/query/stream",
            json={"question": "RAG란 무엇인가요?"},
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    
                    if data["type"] == "sources":
                        print(f"📚 소스 {len(data['data'])}건 수신")
                    elif data["type"] == "token":
                        print(data["data"], end="", flush=True)
                    elif data["type"] == "done":
                        print("\n✅ 스트리밍 완료")
```

에이전틱 RAG 스트리밍 클라이언트는 추가 이벤트 타입을 처리해야 합니다:

```python
# 에이전틱 RAG 스트리밍 클라이언트
async def consume_agentic_stream():
    """astream_events 기반 에이전틱 RAG 스트리밍 소비"""
    async with httpx.AsyncClient(timeout=60.0) as client:  # 타임아웃 여유 있게
        async with client.stream(
            "POST",
            "http://localhost:8000/api/query/agent/stream",
            json={"question": "2024년 노벨 물리학상 수상자는?"},
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])

                if data["type"] == "step":
                    # 현재 실행 단계 표시
                    print(f"\n🔄 {data['data']}")
                elif data["type"] == "tool_result":
                    # 도구 실행 결과 요약
                    print(f"🔧 [{data['tool']}] {data['data'][:100]}...")
                elif data["type"] == "token":
                    # 최종 답변 토큰
                    print(data["data"], end="", flush=True)
                elif data["type"] == "done":
                    print("\n✅ 에이전틱 RAG 완료")
```

전체 프로젝트 구조를 정리하면 다음과 같습니다:

```run:python
# 완성된 프로젝트 구조 확인
structure = """
rag-api/
├── main.py                  # 앱 진입점, create_app()
├── requirements.txt         # 의존성 목록
├── .env                     # 환경 변수 (API 키 등)
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # /query, /query/stream, /query/agent/stream
│   │   └── deps.py          # 의존성 주입 (인증, rate limit 등)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── lifespan.py      # 앱 시작/종료 시 리소스 관리
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py       # QueryRequest
│   │   └── response.py      # QueryResponse, SourceDocument
│   └── services/
│       ├── __init__.py
│       ├── rag_chain.py     # RAGService (LCEL 기반)
│       ├── agentic_rag.py   # AgenticRAGService (LangGraph 기반)
│       └── vector_store.py  # 벡터 DB 연결 관리
└── chroma_db/               # ChromaDB 영속화 디렉토리
"""
print(structure)
```

```output

rag-api/
├── main.py                  # 앱 진입점, create_app()
├── requirements.txt         # 의존성 목록
├── .env                     # 환경 변수 (API 키 등)
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py        # /query, /query/stream, /query/agent/stream
│   │   └── deps.py          # 의존성 주입 (인증, rate limit 등)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings (pydantic-settings)
│   │   └── lifespan.py      # 앱 시작/종료 시 리소스 관리
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request.py       # QueryRequest
│   │   └── response.py      # QueryResponse, SourceDocument
│   └── services/
│       ├── __init__.py
│       ├── rag_chain.py     # RAGService (LCEL 기반)
│       ├── agentic_rag.py   # AgenticRAGService (LangGraph 기반)
│       └── vector_store.py  # 벡터 DB 연결 관리
└── chroma_db/               # ChromaDB 영속화 디렉토리

```

> 📊 **그림 4**: RAG API의 전체 요청-응답 아키텍처

```mermaid
flowchart TD
    Client["🖥️ 클라이언트<br/>(웹, 모바일, 다른 서비스)"]
    
    subgraph FastAPI["FastAPI 서버"]
        MW["미들웨어<br/>(CORS, 로깅)"]
        Val["Pydantic 검증"]
        Route["라우터<br/>(/api/query)"]
    end
    
    subgraph Service["서비스 레이어"]
        RAG["RAG 서비스<br/>(LCEL)"]
        Agent["에이전틱 RAG<br/>(LangGraph)"]
        Ret["리트리버"]
        Chain["LCEL 체인"]
    end
    
    VDB["벡터 DB<br/>(ChromaDB)"]
    LLM["LLM API<br/>(OpenAI)"]
    CP["체크포인터<br/>(PostgreSQL)"]
    
    Client -->|"HTTP POST"| MW
    MW --> Val
    Val --> Route
    Route --> RAG
    Route --> Agent
    RAG --> Ret
    Agent --> Ret
    Agent -->|"상태 저장/로드"| CP
    Ret -->|"유사도 검색"| VDB
    VDB -->|"관련 문서"| Ret
    RAG --> Chain
    Agent -->|"그래프 실행"| LLM
    Chain -->|"프롬프트 + 컨텍스트"| LLM
    LLM -->|"생성 답변"| Chain
    Chain -->|"JSON / SSE"| Client
```

## 더 깊이 알아보기

### FastAPI의 탄생 이야기

FastAPI를 만든 **Sebastián Ramírez**는 콜롬비아 출신의 독학 개발자입니다. 7살 때부터 독학으로 프로그래밍과 영어를 배운 그는, 수년간 클라우드 소프트웨어를 만들면서 기존 Python 웹 프레임워크들의 한계를 절감했거든요. 코드 에디터의 자동 완성, 인라인 에러 알림 같은 개발자 경험(DX)을 최우선으로 하는 프레임워크를 만들고 싶었습니다.

낮에는 본업을 하고 밤과 주말에 FastAPI를 개발하며, 눈이 경련을 일으키고 손가락 하나가 마비될 정도로 몰두했다고 합니다. 그리고 **2018년 크리스마스 이브**, 전 세계 개발자에게 선물하는 마음으로 GitHub에 첫 릴리스를 공개했죠. OpenAPI와 JSON Schema 표준을 철저히 따르면서도 Python의 타입 힌트를 활용해 코드만으로 문서가 자동 생성되는 — 그가 꿈꿨던 프레임워크가 탄생한 순간이었습니다.

지금 FastAPI는 GitHub 스타 80,000개 이상을 보유한 Python 생태계 최고의 웹 프레임워크 중 하나가 되었고, 특히 ML/AI API 서빙 분야에서 사실상 표준으로 자리 잡았습니다.

### 왜 Flask나 Django가 아닌 FastAPI인가?

RAG API에 FastAPI가 특히 적합한 이유가 있습니다. Flask는 동기 처리가 기본이라 LLM API 호출을 기다리는 동안 다른 요청을 처리하지 못합니다. Django는 강력하지만 RAG API에는 불필요한 기능(ORM, 템플릿 엔진 등)이 너무 많죠. FastAPI는 **비동기 네이티브**이면서도 가벼워서, LLM 호출이 많은 RAG 서비스에 딱 맞습니다. 실제로 Uvicorn 단일 워커에서도 분당 10,000건 이상의 요청을 처리할 수 있다는 벤치마크 결과도 있습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "FastAPI의 `async def`를 쓰면 무조건 빨라진다." — 그렇지 않습니다! 내부에서 호출하는 라이브러리가 비동기를 지원하지 않으면(`requests` 등), 오히려 이벤트 루프를 블로킹해서 더 느려질 수 있습니다. LangChain의 `ainvoke`, `astream`이나 `httpx.AsyncClient`처럼 비동기를 지원하는 라이브러리를 사용하세요. 동기 코드만 있다면 `def`(일반 함수)로 정의하는 게 오히려 낫습니다 — FastAPI가 자동으로 별도 스레드에서 실행해주거든요.

> 💡 **알고 계셨나요?**: FastAPI의 이름에서 "Fast"는 두 가지 의미를 가집니다. 하나는 높은 성능(NodeJS, Go에 견줄 수 있는 속도), 다른 하나는 빠른 개발 속도입니다. 타입 힌트 기반 자동 문서 생성 덕분에 API 문서를 따로 작성할 필요가 없으니, 개발 속도가 200~300% 빨라진다고 Sebastián Ramírez는 말합니다.

> 🔥 **실무 팁**: 프로덕션에서는 반드시 `lifespan`으로 리소스를 관리하세요. 매 요청마다 벡터 스토어를 로드하면 응답 시간이 수 초씩 늘어납니다. `app.state`에 RAG 서비스를 캐싱하면 첫 요청만 느리고 이후는 밀리초 단위로 응답할 수 있습니다. 또한, `uvicorn --workers 4` 같이 멀티 워커를 띄울 때는 각 워커가 독립적인 RAG 서비스 인스턴스를 가진다는 점을 기억하세요.

> 🔥 **실무 팁**: CORS 설정에서 `allow_origins=["*"]`는 개발 환경에서만 사용하세요. 프로덕션에서는 `allow_origins=["https://your-app.com"]`처럼 허용할 도메인을 명시해야 보안 취약점을 방지할 수 있습니다.

> 🔥 **실무 팁**: 에이전틱 RAG에서 타임아웃 설정은 필수입니다. 도구를 3~4번 연속 호출하면 총 지연이 30초를 넘길 수 있으므로, 클라이언트의 `httpx.AsyncClient(timeout=60.0)`과 서버의 `uvicorn --timeout-keep-alive 65` 모두 넉넉하게 잡아야 합니다. Nginx 같은 리버스 프록시를 쓴다면 `proxy_read_timeout`도 함께 늘려주세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| FastAPI | Python 비동기 웹 프레임워크. 타입 힌트 기반 자동 문서 생성, 높은 성능 |
| `lifespan` | 앱 시작/종료 시 리소스(벡터 스토어, LLM 클라이언트, 그래프)를 관리하는 패턴 |
| Pydantic v2 스키마 | 요청/응답 데이터 구조를 정의하고 자동 검증. `BaseModel` 상속 |
| `async def` 엔드포인트 | 비동기 처리로 LLM 호출 대기 중에도 다른 요청 처리 가능 |
| `StreamingResponse` | SSE(Server-Sent Events)로 LLM 토큰을 실시간 스트리밍 |
| `astream` | LangChain LCEL 체인의 비동기 스트리밍 메서드 |
| `astream_events` | LangGraph 그래프의 노드/도구/LLM 이벤트를 세밀하게 스트리밍 |
| `thread_id` + 체크포인터 | LangGraph 에이전트의 대화 상태를 사용자별로 격리·영속화하는 패턴 |
| 프로젝트 구조 | `api/` (라우터), `services/` (비즈니스 로직), `schemas/` (데이터 모델), `core/` (설정) 분리 |
| `app.state` | FastAPI 앱 인스턴스에 공유 리소스(RAG 서비스)를 저장하는 속성 |

## 다음 섹션 미리보기

API 서버가 준비되었으니, 이제 그 뒤에서 데이터를 관리하는 방법을 알아볼 차례입니다. 다음 세션 **[20.2: 인덱스 관리와 데이터 파이프라인](ch20/session2)**에서는 문서의 추가·수정·삭제를 자동으로 감지하는 증분 인덱싱(Incremental Indexing)을 구축합니다. 지금 만든 FastAPI 앱이 항상 최신 데이터를 기반으로 응답할 수 있도록, 벡터 인덱스를 효율적으로 유지·갱신하는 전략을 배우게 됩니다.

## 참고 자료

- [FastAPI 공식 문서 — Tutorial](https://fastapi.tiangolo.com/tutorial/) - FastAPI의 기초부터 고급 기능까지 단계별로 배울 수 있는 공식 가이드
- [FastAPI 공식 문서 — StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) - 스트리밍 응답의 공식 사용법과 예제
- [LangGraph — Streaming](https://langchain-ai.github.io/langgraph/how-tos/streaming-tokens/) - LangGraph의 `astream_events`를 포함한 다양한 스트리밍 패턴 공식 가이드
- [LangGraph — Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) - 체크포인터 기반 상태 관리와 멀티턴 대화 지원 공식 문서
- [LangChain RAG Documentation](https://docs.langchain.com/oss/python/langchain/rag) - LangChain 기반 RAG 파이프라인 구축 공식 문서
- [Azure RAG Solution Design and Evaluation Guide](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-solution-design-and-evaluation-guide) - 프로덕션 RAG 시스템 설계에 대한 Microsoft의 아키텍처 가이드
- [FastAPI Best Practices (GitHub)](https://github.com/zhanymkanov/fastapi-best-practices) - 스타트업에서 검증된 FastAPI 개발 관례와 베스트 프랙티스 모음
- [Building a RAG System with LangChain and FastAPI (DataCamp)](https://www.datacamp.com/tutorial/building-a-rag-system-with-langchain-and-fastapi) - LangChain + FastAPI로 RAG 시스템을 구축하는 실습 튜토리얼

---
### 🔗 Related Sessions
- [lcel](../08-기본-rag-파이프라인-구축-langchain으로-첫-rag-앱-만들기/01-langchain-v1-핵심-개념과-설정.md) (prerequisite)
