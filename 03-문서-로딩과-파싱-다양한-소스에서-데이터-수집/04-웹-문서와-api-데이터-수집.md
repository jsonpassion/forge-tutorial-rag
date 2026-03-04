# 웹 문서와 API 데이터 수집

> 웹 페이지 크롤링부터 REST API 응답까지, 인터넷 위의 모든 데이터를 RAG 파이프라인에 투입하는 방법을 배웁니다.

## 개요

이 섹션에서는 웹에 존재하는 다양한 데이터를 수집하여 LangChain의 Document 객체로 변환하는 방법을 학습합니다. 단일 웹 페이지 로딩부터 사이트 전체 크롤링, REST API 응답 처리까지 다루며, 책임감 있는 크롤링을 위한 윤리적 가이드라인도 함께 살펴봅니다.

**선수 지식**:
- [Session 3.1: 문서 로딩 기초](ch03/session_3_1.md)에서 배운 Document 객체(page_content + metadata) 구조
- [Session 3.2: PDF 문서 처리](ch03/session_02.md)에서 배운 content_type 메타데이터 패턴
- [Session 3.3: Unstructured.io](ch03/session_3_3.md)에서 배운 `partition_html()` 함수
- Python의 기본 HTTP 요청 개념 (requests 라이브러리)

**학습 목표**:
- WebBaseLoader를 사용하여 웹 페이지의 텍스트를 Document로 변환할 수 있다
- RecursiveUrlLoader와 SitemapLoader로 다수의 웹 페이지를 효율적으로 수집할 수 있다
- REST API 응답 데이터를 Document 객체로 변환하여 RAG 파이프라인에 투입할 수 있다
- robots.txt와 rate limiting 등 크롤링 윤리를 이해하고 준수할 수 있다

## 왜 알아야 할까?

지금까지 우리는 로컬 파일(텍스트, CSV, PDF)을 로딩하는 방법을 배웠습니다. 하지만 현실의 RAG 시스템이 다루는 지식은 대부분 **웹에 존재**합니다. 공식 문서, 기술 블로그, API 레퍼런스, 뉴스 기사, 위키 — 이 모든 것이 URL 하나로 접근 가능한 웹 데이터죠.

실제로 기업 내부 RAG 시스템을 구축할 때 가장 많이 요청받는 데이터 소스가 바로 **사내 위키**, **Confluence 페이지**, **API 문서**입니다. "우리 회사 기술 문서를 전부 RAG에 넣어주세요"라는 요청은 곧 "웹 페이지 수백~수천 개를 자동으로 수집하세요"라는 의미거든요.

이번 섹션을 마치면, 단일 URL부터 사이트 전체까지 자유자재로 수집하고, REST API에서 받아온 JSON 데이터까지 RAG 파이프라인에 투입할 수 있게 됩니다.

## 핵심 개념

### 개념 1: WebBaseLoader — 웹 페이지 한 장 가져오기

> 💡 **비유**: WebBaseLoader는 웹 브라우저에서 "Ctrl+A → Ctrl+C"를 하는 것과 같습니다. 페이지의 텍스트 내용을 통째로 복사해서 Document 객체에 담아주죠. 단, 사람 눈에 보이는 예쁜 레이아웃 대신, 순수한 텍스트만 가져옵니다.

WebBaseLoader는 LangChain에서 가장 기본적인 웹 로더입니다. 내부적으로 `requests` 라이브러리로 HTML을 가져오고, `BeautifulSoup`으로 텍스트를 추출합니다. [Session 3.1](ch03/session_3_1.md)에서 배운 TextLoader가 로컬 파일을 읽었다면, WebBaseLoader는 **URL을 읽는 TextLoader**라고 생각하면 됩니다.

```python
# 필요한 패키지 설치
# pip install langchain-community beautifulsoup4

from langchain_community.document_loaders import WebBaseLoader

# 단일 URL 로딩
loader = WebBaseLoader("https://docs.python.org/3/tutorial/index.html")
docs = loader.load()
```

```run:python
# Document 객체 확인
print(f"문서 수: {len(docs)}")
print(f"메타데이터: {docs[0].metadata}")
print(f"내용 미리보기: {docs[0].page_content[:200]}")
```

```output
문서 수: 1
메타데이터: {'source': 'https://docs.python.org/3/tutorial/index.html', 'title': 'The Python Tutorial', 'language': 'en'}
내용 미리보기: The Python Tutorial — Python 3.13.2 documentation...
```

하지만 기본 설정으로는 네비게이션 바, 푸터, 사이드바 등 **불필요한 텍스트**도 함께 가져옵니다. BeautifulSoup의 파싱 옵션(`bs_kwargs`)을 사용하면 원하는 영역만 추출할 수 있습니다.

```python
import bs4

# <article> 태그 내용만 추출
loader = WebBaseLoader(
    web_paths=["https://docs.python.org/3/tutorial/index.html"],
    bs_kwargs={
        "parse_only": bs4.SoupStrainer("article")  # article 태그만 파싱
    }
)
docs = loader.load()
```

여러 URL을 동시에 로딩할 수도 있습니다. `requests_per_second` 매개변수로 초당 요청 수를 제어하여, 대상 서버에 부담을 주지 않도록 합니다.

```python
# 여러 URL 동시 로딩 (초당 2회 요청 제한)
loader = WebBaseLoader(
    web_paths=[
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html",
        "https://docs.python.org/3/tutorial/datastructures.html",
    ],
    requests_per_second=2  # 서버 부담 방지
)
docs = loader.load()
```

> ⚠️ **흔한 오해**: WebBaseLoader가 브라우저처럼 JavaScript를 실행한다고 생각하는 분이 많습니다. 실제로는 **정적 HTML만 가져옵니다**. React, Vue 같은 SPA(Single Page Application)의 동적 콘텐츠는 WebBaseLoader로 수집할 수 없습니다. 이 경우 `AsyncChromiumLoader`(Playwright 기반)나 `FireCrawlLoader` 같은 헤드리스 브라우저 기반 로더를 사용해야 합니다.

### 개념 2: RecursiveUrlLoader — 링크를 따라가며 재귀적으로 수집

> 💡 **비유**: RecursiveUrlLoader는 도서관에서 책 한 권을 읽다가, 참고문헌에 나온 다른 책도 찾아 읽고, 그 책의 참고문헌에 나온 책도 읽는 것과 같습니다. 시작 URL에서 출발해 페이지 안의 링크를 따라가며 연쇄적으로 수집하죠.

문서 사이트나 위키처럼 페이지가 서로 링크로 연결된 경우, 일일이 URL을 지정하기란 비현실적입니다. RecursiveUrlLoader는 시작 URL에서 출발하여 내부 링크를 자동으로 탐색하며, `max_depth` 매개변수로 탐색 깊이를 제한합니다.

```python
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
import re

# HTML에서 텍스트만 깔끔하게 추출하는 함수
def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

loader = RecursiveUrlLoader(
    url="https://docs.python.org/3/tutorial/",
    max_depth=2,              # 2단계 깊이까지만 탐색
    extractor=bs4_extractor,  # 커스텀 텍스트 추출기
    prevent_outside=True,     # 외부 도메인 링크 무시
    exclude_dirs=[            # 제외할 경로 패턴
        "https://docs.python.org/3/library/",
        "https://docs.python.org/3/reference/",
    ],
)
docs = loader.load()
```

```run:python
print(f"수집된 문서 수: {len(docs)}")
# 수집된 URL 확인
for doc in docs[:5]:
    print(f"  - {doc.metadata['source']}")
```

```output
수집된 문서 수: 17
  - https://docs.python.org/3/tutorial/
  - https://docs.python.org/3/tutorial/appetite.html
  - https://docs.python.org/3/tutorial/interpreter.html
  - https://docs.python.org/3/tutorial/introduction.html
  - https://docs.python.org/3/tutorial/controlflow.html
```

핵심 매개변수를 정리하면 다음과 같습니다:

| 매개변수 | 설명 | 기본값 |
|----------|------|--------|
| `max_depth` | 재귀 탐색 깊이 제한 | 2 |
| `prevent_outside` | 외부 도메인 링크 차단 | `False` |
| `exclude_dirs` | 제외할 URL 경로 목록 | `()` |
| `link_regex` | 수집할 링크의 정규식 패턴 | `None` |
| `extractor` | HTML → 텍스트 변환 함수 | `None` |
| `use_async` | 비동기 로딩 활성화 | `False` |
| `timeout` | 요청 타임아웃 (초) | `None` |

> 🔥 **실무 팁**: `max_depth`를 3 이상으로 설정하면 수집되는 페이지 수가 기하급수적으로 증가합니다. 대형 문서 사이트에서 `max_depth=3`으로 설정했다가 수만 페이지를 수집하게 된 사례가 많습니다. **반드시 `exclude_dirs`와 함께 사용하고, 먼저 `max_depth=1`로 테스트**하여 수집 범위를 확인하세요.

### 개념 3: SitemapLoader — 사이트맵으로 체계적 수집

> 💡 **비유**: RecursiveUrlLoader가 미로를 탐험하듯 링크를 따라가는 방식이라면, SitemapLoader는 **건물 안내도**를 먼저 받아보는 것과 같습니다. 사이트맵(sitemap.xml)은 웹사이트가 "우리 사이트에는 이런 페이지들이 있습니다"라고 정리해둔 목록이거든요.

대부분의 체계적인 웹사이트는 `sitemap.xml` 파일을 제공합니다. SitemapLoader는 이 파일을 먼저 읽고, 그 안에 나열된 URL을 동시에 수집합니다. RecursiveUrlLoader처럼 링크를 추적할 필요가 없으니, 누락 없이 사이트의 모든 페이지를 수집할 수 있죠.

```python
from langchain_community.document_loaders.sitemap import SitemapLoader

# 사이트맵 기반 로딩
loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    requests_per_second=2,  # 초당 요청 제한
)
docs = loader.load()
```

`filter_urls` 매개변수로 특정 패턴의 URL만 선별적으로 수집할 수 있습니다. 정규식도 지원하므로 매우 유연합니다.

```python
# 특정 경로의 문서만 수집
loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    filter_urls=[
        "https://api.python.langchain.com/en/latest/core"  # core 모듈만
    ],
    requests_per_second=2,
)
docs = loader.load()
```

네비게이션이나 헤더 같은 불필요한 요소를 제거하려면, 커스텀 파싱 함수를 전달합니다.

```python
from bs4 import BeautifulSoup

def remove_nav_and_header(content: BeautifulSoup) -> str:
    """네비게이션과 헤더 요소를 제거하고 본문만 추출합니다."""
    for element in content.find_all(["nav", "header", "footer"]):
        element.decompose()  # 해당 요소 완전 제거
    return content.get_text(separator="\n", strip=True)

loader = SitemapLoader(
    web_path="https://api.python.langchain.com/sitemap.xml",
    parsing_function=remove_nav_and_header,
    requests_per_second=2,
)
```

**RecursiveUrlLoader vs SitemapLoader — 언제 무엇을 쓸까?**

| 기준 | RecursiveUrlLoader | SitemapLoader |
|------|-------------------|---------------|
| 사이트맵 필요 여부 | 불필요 | `sitemap.xml` 필수 |
| 수집 범위 제어 | `max_depth`, `exclude_dirs` | `filter_urls` |
| 누락 가능성 | 링크가 없으면 누락 | 사이트맵에 있으면 모두 수집 |
| 적합한 상황 | 사이트맵 없는 소규모 사이트 | 체계적인 문서 사이트 |
| 속도 | 순차적 (기본) | 동시 수집 (기본 2req/s) |

### 개념 4: API 데이터를 Document로 변환

> 💡 **비유**: 웹 페이지 크롤링이 서점에서 책을 직접 읽는 것이라면, API 호출은 **도서관 사서에게 "이런 책 있나요?"라고 물어보는 것**과 같습니다. 사서(API)는 깔끔하게 정리된 정보(JSON)를 돌려주죠.

RAG 시스템은 웹 페이지만 다루지 않습니다. GitHub Issues, Jira 티켓, Notion 페이지, Slack 메시지 등 다양한 **API 기반 데이터 소스**를 활용해야 할 때가 많습니다. API 응답(주로 JSON)을 LangChain의 Document 객체로 변환하는 패턴을 알아봅시다.

[Session 3.1](ch03/session_3_1.md)에서 배운 것처럼 Document는 `page_content`와 `metadata`로 구성됩니다. API 응답의 핵심 텍스트를 `page_content`에, 부가 정보를 `metadata`에 매핑하면 됩니다. 이때 [Session 3.2](ch03/session_02.md)에서 배운 `content_type` 메타데이터 패턴을 동일하게 적용하면, 웹/API에서 수집한 문서와 PDF에서 수집한 문서를 하나의 파이프라인에서 일관되게 관리할 수 있습니다.

```python
import requests
from langchain_core.documents import Document

def fetch_github_issues(owner: str, repo: str, count: int = 10) -> list[Document]:
    """GitHub 이슈를 Document 객체 리스트로 변환합니다."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(
        url,
        params={"state": "open", "per_page": count},
        headers={"Accept": "application/vnd.github.v3+json"},
        timeout=10,
    )
    response.raise_for_status()  # HTTP 에러 시 예외 발생

    documents = []
    for issue in response.json():
        # 제목과 본문을 page_content로 결합
        content = f"# {issue['title']}\n\n{issue.get('body', '') or ''}"

        # 부가 정보를 metadata로 저장
        metadata = {
            "source": issue["html_url"],
            "issue_number": issue["number"],
            "author": issue["user"]["login"],
            "labels": [label["name"] for label in issue["labels"]],
            "created_at": issue["created_at"],
            "state": issue["state"],
        }
        documents.append(Document(page_content=content, metadata=metadata))

    return documents
```

```run:python
# GitHub 이슈를 Document로 변환
docs = fetch_github_issues("langchain-ai", "langchain", count=3)
print(f"수집된 이슈 수: {len(docs)}")
print(f"첫 번째 이슈: #{docs[0].metadata['issue_number']}")
print(f"라벨: {docs[0].metadata['labels']}")
print(f"내용 미리보기: {docs[0].page_content[:100]}...")
```

```output
수집된 이슈 수: 3
첫 번째 이슈: #30521
라벨: ['bug', 'investigate']
내용 미리보기: # ChatOpenAI streaming with structured output raises BadRequestError

## Checked other resources...
```

페이지네이션이 있는 API는 여러 번 요청을 보내며 Document를 누적해야 합니다. 이 패턴을 함수로 정리해두면 어떤 API든 재사용할 수 있습니다.

```python
def fetch_paginated_api(
    base_url: str,
    params: dict | None = None,
    headers: dict | None = None,
    content_key: str = "content",
    max_pages: int = 5,
) -> list[Document]:
    """페이지네이션이 있는 API에서 Document를 수집합니다."""
    documents = []
    params = params or {}
    page = 1

    while page <= max_pages:
        params["page"] = page
        response = requests.get(
            base_url,
            params=params,
            headers=headers or {},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        items = data.get("results", data.get("items", []))
        if not items:
            break  # 더 이상 데이터 없음

        for item in items:
            doc = Document(
                page_content=str(item.get(content_key, "")),
                metadata={
                    "source": base_url,
                    "page": page,
                    "api_response": True,
                },
            )
            documents.append(doc)

        page += 1

    return documents
```

### 개념 5: 책임감 있는 크롤링 — robots.txt와 Rate Limiting

> 💡 **비유**: 도서관에 가면 "조용히 해주세요", "음식 반입 금지" 같은 규칙이 있죠? 웹사이트도 마찬가지입니다. `robots.txt`는 "우리 사이트에서 이 구역은 크롤링하지 마세요"라는 규칙이고, Rate Limiting은 "한 번에 너무 많이 가져가지 마세요"라는 규칙입니다.

**robots.txt 확인하기**

`robots.txt`는 웹사이트 루트에 위치한 텍스트 파일로, 크롤러가 접근해도 되는 경로와 접근 금지 경로를 명시합니다. 크롤링 전에 반드시 확인해야 합니다.

```python
from urllib.robotparser import RobotFileParser

def check_robots_txt(url: str, user_agent: str = "*") -> tuple[bool, float | None]:
    """robots.txt를 확인하여 해당 URL의 크롤링 가능 여부와 권장 크롤링 간격을 반환합니다."""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    rp = RobotFileParser()
    rp.set_url(robots_url)
    rp.read()

    can_fetch = rp.can_fetch(user_agent, url)
    crawl_delay = rp.crawl_delay(user_agent)

    return can_fetch, crawl_delay
```

```run:python
# robots.txt 확인 예시
can_fetch, delay = check_robots_txt("https://docs.python.org/3/tutorial/")
print(f"크롤링 허용: {can_fetch}")
print(f"권장 크롤링 간격: {delay}초")
```

```output
크롤링 허용: True
권장 크롤링 간격: None초
```

**Rate Limiting 구현하기**

서버에 과도한 부하를 주지 않으려면, 요청 사이에 적절한 간격을 두어야 합니다. LangChain의 웹 로더들은 `requests_per_second` 매개변수를 제공하지만, 직접 API를 호출할 때는 rate limiter를 구현해야 합니다.

```python
import time
from typing import Generator

def rate_limited_fetch(
    urls: list[str],
    requests_per_second: float = 1.0,
) -> Generator[Document, None, None]:
    """초당 요청 수를 제한하며 URL을 순차적으로 수집합니다."""
    interval = 1.0 / requests_per_second

    for url in urls:
        start_time = time.time()

        # robots.txt 확인
        can_fetch, crawl_delay = check_robots_txt(url)
        if not can_fetch:
            print(f"⚠️ robots.txt에 의해 차단됨: {url}")
            continue

        # Crawl-delay 존재 시 해당 간격 적용
        actual_interval = max(interval, crawl_delay or 0)

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        yield Document(
            page_content=response.text,
            metadata={"source": url, "status_code": response.status_code},
        )

        # Rate limiting 적용
        elapsed = time.time() - start_time
        if elapsed < actual_interval:
            time.sleep(actual_interval - elapsed)
```

**크롤링 윤리 체크리스트**:

1. **robots.txt 확인**: 크롤링 전에 반드시 `robots.txt`를 읽고 규칙을 준수합니다
2. **요청 간격 조절**: 최소 1~2초 간격을 두고, `Crawl-delay` 지시가 있으면 따릅니다
3. **User-Agent 명시**: 크롤러의 정체를 밝히는 것이 좋은 관행입니다
4. **이용 약관 확인**: 웹사이트의 Terms of Service에서 스크래핑 관련 조항을 확인합니다
5. **캐싱 활용**: 동일 페이지를 반복 크롤링하지 않도록 결과를 캐싱합니다

## 실습: 직접 해보기

실제 기술 문서 사이트에서 데이터를 수집하고, API 응답과 결합하여 하나의 Document 컬렉션을 만들어 봅시다.

```python
"""
실습: 웹 문서 + API 데이터를 결합한 Document 수집 파이프라인
필요 패키지: pip install langchain-community beautifulsoup4 lxml requests
"""
import time
import requests
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader


# ── 1단계: robots.txt 검증 유틸리티 ──
def is_crawlable(url: str) -> tuple[bool, float | None]:
    """URL의 크롤링 허용 여부와 권장 간격을 확인합니다."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
    except Exception:
        return True, None  # robots.txt를 읽을 수 없으면 허용으로 간주
    return rp.can_fetch("*", url), rp.crawl_delay("*")


# ── 2단계: 웹 문서 수집 (WebBaseLoader) ──
def collect_web_docs(urls: list[str]) -> list[Document]:
    """여러 URL에서 웹 문서를 수집합니다."""
    # 크롤링 가능 여부 사전 확인
    allowed_urls = []
    for url in urls:
        can_fetch, _ = is_crawlable(url)
        if can_fetch:
            allowed_urls.append(url)
        else:
            print(f"⚠️ 크롤링 차단: {url}")

    if not allowed_urls:
        return []

    # BeautifulSoup으로 본문만 추출 (article 또는 main 태그)
    loader = WebBaseLoader(
        web_paths=allowed_urls,
        requests_per_second=1,  # 서버 부담 최소화
    )
    docs = loader.load()

    # 메타데이터에 수집 시각 추가
    for doc in docs:
        doc.metadata["collected_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        doc.metadata["content_type"] = "web_page"

    return docs


# ── 3단계: GitHub API에서 이슈 수집 ──
def collect_github_issues(
    owner: str, repo: str, count: int = 5
) -> list[Document]:
    """GitHub 리포지토리의 최근 이슈를 Document로 수집합니다."""
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(
        url,
        params={"state": "open", "per_page": count, "sort": "created"},
        headers={"Accept": "application/vnd.github.v3+json"},
        timeout=10,
    )
    response.raise_for_status()

    documents = []
    for issue in response.json():
        if "pull_request" in issue:
            continue  # PR은 제외

        body = issue.get("body") or "(내용 없음)"
        content = f"# {issue['title']}\n\n{body}"

        documents.append(Document(
            page_content=content,
            metadata={
                "source": issue["html_url"],
                "issue_number": issue["number"],
                "author": issue["user"]["login"],
                "labels": [l["name"] for l in issue["labels"]],
                "created_at": issue["created_at"],
                "content_type": "github_issue",
                "collected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        ))

    return documents


# ── 4단계: 통합 수집 및 결과 확인 ──
def main():
    # 웹 문서 수집
    web_urls = [
        "https://docs.python.org/3/tutorial/introduction.html",
        "https://docs.python.org/3/tutorial/controlflow.html",
    ]
    web_docs = collect_web_docs(web_urls)
    print(f"✅ 웹 문서 수집 완료: {len(web_docs)}건")

    # GitHub 이슈 수집
    issue_docs = collect_github_issues("langchain-ai", "langchain", count=3)
    print(f"✅ GitHub 이슈 수집 완료: {len(issue_docs)}건")

    # 모든 문서 통합
    all_docs = web_docs + issue_docs
    print(f"\n📊 전체 수집 결과: {len(all_docs)}건")
    print("-" * 50)

    for i, doc in enumerate(all_docs):
        content_type = doc.metadata.get("content_type", "unknown")
        source = doc.metadata.get("source", "N/A")
        length = len(doc.page_content)
        print(f"[{i+1}] {content_type:15s} | {length:6,}자 | {source[:60]}")

    return all_docs


if __name__ == "__main__":
    all_docs = main()
```

실행하면 다음과 비슷한 결과를 볼 수 있습니다:

```output
✅ 웹 문서 수집 완료: 2건
✅ GitHub 이슈 수집 완료: 3건

📊 전체 수집 결과: 5건
--------------------------------------------------
[1] web_page        | 12,847자 | https://docs.python.org/3/tutorial/introduction.html
[2] web_page        |  8,523자 | https://docs.python.org/3/tutorial/controlflow.html
[3] github_issue    |  1,245자 | https://github.com/langchain-ai/langchain/issues/30521
[4] github_issue    |    892자 | https://github.com/langchain-ai/langchain/issues/30518
[5] github_issue    |    634자 | https://github.com/langchain-ai/langchain/issues/30515
```

이렇게 수집한 Document들은 다음 장([Chapter 4: 텍스트 청킹 전략](ch04/session_4_1.md))에서 배울 텍스트 분할기를 거쳐 벡터 데이터베이스에 저장하게 됩니다.

## 더 깊이 알아보기

### 웹 크롤러의 탄생 — 1993년, Matthew Gray의 Wanderer

웹 크롤링의 역사는 월드 와이드 웹의 역사만큼이나 오래되었습니다. 1993년 6월, MIT의 물리학과 학생이었던 **Matthew Gray**는 흥미로운 질문을 던졌습니다. **"지금 웹에는 사이트가 몇 개나 있을까?"**

당시 웹은 아직 갓 태어난 기술이었습니다. 같은 해 4월에 CERN이 월드 와이드 웹 소프트웨어를 퍼블릭 도메인으로 공개했고, Mosaic 브라우저가 첫 버전을 선보인 시점이었죠. Gray는 이 작은 웹의 크기를 측정하기 위해 Perl로 **World Wide Web Wanderer**를 작성했습니다.

Wanderer는 웹 페이지를 자동으로 방문하고 링크를 따라가며 새로운 사이트를 발견하는 프로그램이었는데, 이것이 **역사상 최초의 웹 크롤러(웹 로봇)**였습니다. Gray는 Wanderer가 수집한 정보를 바탕으로 **Wandex**라는 인덱스도 만들었는데, 이는 최초의 웹 데이터베이스 중 하나였습니다.

놀랍게도 Wanderer가 처음 측정한 웹의 크기는 약 **130개 사이트**에 불과했습니다. Wanderer는 1995년까지 웹의 성장을 추적했고, 그 과정에서 크롤러가 서버에 과도한 부하를 준다는 불만이 생겨나기 시작했습니다. 이것이 바로 1994년에 `robots.txt` 표준이 탄생하게 된 직접적인 계기였습니다.

### robots.txt — Martijn Koster의 선견지명

Wanderer의 등장 이후 여러 웹 크롤러가 생겨나면서, 서버 관리자들의 불만이 커졌습니다. 1994년, 네덜란드 출신 웹 개발자 **Martijn Koster**는 크롤러와 서버 간의 에티켓을 정하자는 제안을 했고, 이것이 **Robots Exclusion Protocol** — 흔히 `robots.txt`로 불리는 표준의 시작이었습니다.

30년이 지난 지금도 거의 모든 웹사이트가 이 표준을 사용하고 있다는 것은, 단순하지만 효과적인 약속의 힘을 보여줍니다. 우리가 RAG를 위해 크롤링할 때도 이 약속을 지키는 것은 기본 중의 기본이겠죠?

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "WebBaseLoader로 어떤 웹 페이지든 수집할 수 있다"고 생각하기 쉽습니다. 하지만 JavaScript로 렌더링되는 SPA(React, Vue, Angular 앱), 로그인이 필요한 페이지, CAPTCHA가 있는 사이트, IP 기반 접근 제한이 있는 사이트 등은 WebBaseLoader로 수집할 수 없습니다. JavaScript 렌더링이 필요하면 `AsyncChromiumLoader`, 인증이 필요하면 API를 직접 호출하는 방식을 고려하세요.

> 💡 **알고 계셨나요?**: 구글의 첫 번째 크롤러 **Googlebot**은 1996년에 탄생했는데, Larry Page와 Sergey Brin이 스탠포드 대학원 기숙사 방에서 운영했습니다. 크롤러가 너무 많은 대역폭을 잡아먹어서 스탠포드 네트워크가 한 번 다운된 적이 있다고 합니다. 지금도 `requests_per_second`를 적절히 설정해야 하는 이유를 30년 전에 이미 증명한 셈이죠.

> 🔥 **실무 팁**: 프로덕션 RAG 시스템에서 웹 데이터를 수집할 때는 **캐싱 레이어**를 반드시 추가하세요. 같은 페이지를 매번 새로 크롤링하면 불필요한 네트워크 비용이 발생하고, 서버에도 부담을 줍니다. 수집 결과를 로컬 파일이나 데이터베이스에 저장해두고, 일정 주기(예: 24시간)마다 갱신하는 것이 일반적인 패턴입니다. `metadata`에 `collected_at` 타임스탬프를 넣어두면 "이 데이터가 언제 수집된 것인지" 추적할 수 있어 디버깅에도 유용합니다.

> 🔥 **실무 팁**: SitemapLoader를 사용할 때, 사이트맵이 `sitemap-index.xml`처럼 여러 하위 사이트맵으로 구성된 경우가 있습니다. SitemapLoader는 이를 자동으로 처리하지만, 대형 사이트는 수만 개의 URL이 포함될 수 있으므로, 반드시 `filter_urls`로 범위를 좁히세요.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| WebBaseLoader | BeautifulSoup 기반으로 단일/다수 URL의 정적 HTML을 Document로 변환 |
| RecursiveUrlLoader | 시작 URL에서 링크를 재귀적으로 추적하며 수집. `max_depth`로 깊이 제한 |
| SitemapLoader | `sitemap.xml`에 나열된 URL을 동시 수집. 누락 없는 체계적 크롤링 |
| API → Document | `requests`로 JSON 응답을 받아 `Document(page_content=..., metadata=...)`로 변환 |
| robots.txt | 웹사이트가 크롤러에게 접근 규칙을 알려주는 표준. `RobotFileParser`로 확인 |
| requests_per_second | 초당 요청 수 제한. 서버 부하 방지를 위해 1~2로 설정 권장 |
| bs_kwargs | WebBaseLoader에서 BeautifulSoup 파싱 옵션 지정. 특정 태그만 추출 가능 |
| prevent_outside | RecursiveUrlLoader에서 외부 도메인 링크 탐색을 차단하는 옵션 |
| filter_urls | SitemapLoader에서 특정 URL 패턴만 선별적으로 수집하는 옵션 |

## 다음 섹션 미리보기

지금까지 Chapter 3에서 텍스트 파일, PDF, Unstructured.io, 그리고 웹/API까지 다양한 소스에서 데이터를 수집하는 방법을 배웠습니다. 다음 [Session 3.5](ch03/session_3_5.md)에서는 이 모든 것을 통합하여 **실전 문서 전처리 파이프라인**을 구축합니다. 여러 형식의 문서가 혼재된 상황에서 자동으로 형식을 감지하고, 적절한 로더를 선택하여, 깔끔한 Document 컬렉션으로 만드는 실전 워크플로를 다루게 됩니다. [Chapter 4: 텍스트 청킹 전략](ch04/session_4_1.md)으로 넘어가기 전에 꼭 알아야 할 통합 실습이니 기대해 주세요!

## 참고 자료

- [WebBaseLoader 공식 문서](https://docs.langchain.com/oss/python/integrations/document_loaders/web_base) — WebBaseLoader의 전체 매개변수와 사용법을 확인할 수 있는 공식 가이드
- [RecursiveUrlLoader 공식 문서](https://docs.langchain.com/oss/python/integrations/document_loaders/recursive_url) — 재귀 크롤링 로더의 매개변수, 보안 고려사항, 커스텀 extractor 사용법
- [SitemapLoader 공식 문서](https://docs.langchain.com/oss/python/integrations/document_loaders/sitemap) — 사이트맵 기반 수집의 필터링, 커스텀 파싱 함수, 로컬 사이트맵 지원
- [Google robots.txt 명세 가이드](https://developers.google.com/crawling/docs/robots-txt/robots-txt-spec) — robots.txt 표준의 공식 해석과 구현 가이드
- [Unstructured.io GitHub](https://github.com/Unstructured-IO/unstructured) — `partition_html(url=...)` 직접 URL 파싱 지원. [Session 3.3](ch03/session_3_3.md)에서 배운 Unstructured를 웹 데이터에 활용 가능
- [LangChain RAG 공식 문서](https://docs.langchain.com/oss/python/langchain/rag) — 문서 수집부터 검색까지 RAG 전체 파이프라인의 공식 레퍼런스

---
### 🔗 Related Sessions
- [document](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [page_content](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [metadata](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [lazy_load](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [partition_html](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/03-unstructuredio-범용-문서-파싱-엔진.md) (prerequisite)
- [textloader](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
