# Unstructured.io — 범용 문서 파싱 엔진

> 25가지 이상의 문서 형식을 하나의 함수로 파싱하는 오픈소스 ETL 엔진

## 개요

이 섹션에서는 RAG 파이프라인의 문서 전처리 단계에서 가장 널리 사용되는 오픈소스 라이브러리인 Unstructured.io를 깊이 있게 다룹니다. 앞서 [3.1: 문서 로딩 기초](03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md)에서 LangChain Document Loaders의 기본 구조를 배웠고, [3.2: PDF 문서 처리](03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md)에서 PDF 파싱의 어려움과 다양한 라이브러리를 비교했습니다. 이번에는 그 모든 것을 하나로 통합하는 범용 파싱 엔진을 만나봅니다.

**선수 지식**: Document 객체(page_content + metadata) 구조, PDF 파싱의 기본 개념, `UnstructuredPDFLoader`의 기본 사용법
**학습 목표**:
- Unstructured.io의 아키텍처와 `partition` 함수의 동작 원리를 이해한다
- 15가지 Element 타입(Title, NarrativeText, Table 등)의 역할과 활용법을 익힌다
- 파싱 전략(`fast`, `hi_res`, `ocr_only`, `auto`)의 차이를 이해하고 상황에 맞게 선택할 수 있다
- LangChain의 `UnstructuredLoader`와 연동하여 RAG 파이프라인에 통합할 수 있다

## 왜 알아야 할까?

실무에서 RAG 시스템을 구축할 때, 데이터는 절대 하나의 형식으로 오지 않습니다. 영업팀은 PDF 보고서를, 개발팀은 Markdown 문서를, 법무팀은 Word 파일을, 고객지원팀은 이메일(.eml)을 보내죠. 앞서 배운 것처럼 각 형식마다 별도의 로더를 사용할 수도 있지만, 형식이 10개, 20개로 늘어나면 어떻게 될까요?

Unstructured.io는 이 문제를 근본적으로 해결합니다. **하나의 `partition()` 함수**로 PDF, HTML, Word, PowerPoint, 이메일, 이미지 등 25가지 이상의 형식을 자동으로 감지하고 파싱합니다. 마치 만능 번역기처럼, 어떤 문서 형식이 들어오든 동일한 구조의 Element 객체로 변환해주는 거죠.

특히 RAG에서 Unstructured.io가 중요한 이유는, 단순히 텍스트만 추출하는 게 아니라 **문서의 의미 구조를 보존**한다는 점입니다. 제목은 `Title`로, 본문은 `NarrativeText`로, 표는 `Table`로 분류하여 추출하기 때문에, 이후 청킹(Chunking)이나 메타데이터 필터링 단계에서 훨씬 정교한 처리가 가능합니다.

## 핵심 개념

### 개념 1: Unstructured.io의 아키텍처 — 만능 분류기

> 💡 **비유**: Unstructured.io는 **우체국의 자동 분류 시스템**과 같습니다. 편지, 소포, 등기, 국제 우편 등 다양한 우편물이 들어오면, 자동으로 종류를 식별하고 적절한 처리 라인으로 보내죠. `partition()` 함수가 바로 이 자동 분류기 역할을 합니다.

Unstructured.io의 핵심 아이디어는 단순합니다. **어떤 형식의 문서든 `partition()` 하나로 처리한다**는 것이죠.

내부적으로는 파일 형식을 자동 감지한 뒤(`libmagic` 사용), 형식에 맞는 전용 파서로 라우팅합니다:

| 문서 형식 | 전용 파서 함수 | 핵심 의존성 |
|-----------|---------------|------------|
| PDF | `partition_pdf` | pdfminer, detectron2 |
| Word (.docx) | `partition_docx` | python-docx |
| Word (.doc) | `partition_doc` | LibreOffice |
| PowerPoint | `partition_pptx` | python-pptx |
| HTML | `partition_html` | lxml |
| Markdown | `partition_md` | 내장 파서 |
| 이메일 (.eml) | `partition_email` | 내장 파서 |
| 이미지 (PNG/JPG) | `partition_image` | Tesseract OCR |
| CSV/TSV | `partition_csv` / `partition_tsv` | 내장 파서 |
| EPUB | `partition_epub` | Pandoc |

먼저 설치부터 해보겠습니다:

```python
# 기본 설치 (텍스트 기반 문서만)
# pip install unstructured

# 모든 문서 형식 지원 (PDF, 이미지, OCR 포함)
# pip install "unstructured[all-docs]"

# LangChain 연동 패키지
# pip install langchain-unstructured
```

가장 기본적인 사용법은 놀라울 정도로 간단합니다:

```run:python
from unstructured.partition.auto import partition

# partition()은 파일 형식을 자동으로 감지합니다
elements = partition(filename="report.pdf")

# 추출된 Element 개수 확인
print(f"추출된 Element 수: {len(elements)}")

# 각 Element의 타입과 내용 미리보기
for element in elements[:5]:
    print(f"[{type(element).__name__}] {str(element)[:80]}...")
```
```output
추출된 Element 수: 47
[Title] 2024년 4분기 매출 보고서...
[NarrativeText] 본 보고서는 2024년 4분기 실적을 요약한 문서입니다. 전년 동기 대비 매출이...
[ListItem] 1. 국내 매출: 전년 대비 15% 증가...
[Table] 구분 | Q3 | Q4 | 증감률 매출액 | 500억 | 575억 | +15%...
[NarrativeText] 해외 시장에서의 성장은 특히 동남아 지역에서 두드러졌으며, 베트남과 인도네시아...
```

보이시나요? 파일 확장자를 보고 자동으로 PDF 파서를 선택했고, 제목, 본문, 리스트, 표까지 **의미 단위로** 분류해서 추출했습니다. 이것이 단순한 텍스트 추출과 Unstructured.io의 가장 큰 차이점입니다.

### 개념 2: Element 타입 체계 — 문서의 DNA

> 💡 **비유**: Element 타입은 **레고 블록의 종류**와 같습니다. 레고 세트를 열면 기둥 블록, 판 블록, 바퀴, 창문 등 다양한 종류가 있죠. 각 블록은 고유한 역할이 있고, 이들을 조합해야 의미 있는 구조물이 됩니다. 문서도 마찬가지로, Title, NarrativeText, Table 같은 다양한 Element가 모여 하나의 문서를 구성합니다.

Unstructured.io는 15가지 핵심 Element 타입을 제공합니다:

| Element 타입 | 역할 | 예시 |
|-------------|------|------|
| `Title` | 제목, 소제목 | "3장: 검색 엔진의 원리" |
| `NarrativeText` | 잘 구성된 본문 텍스트 | 여러 문장으로 된 설명 단락 |
| `ListItem` | 목록 항목 | "- 첫 번째 단계: 데이터 수집" |
| `Table` | 표 데이터 | HTML 형태로 보존 |
| `Header` | 문서 머리글 | 페이지 상단 반복 텍스트 |
| `Footer` | 문서 바닥글 | 페이지 하단 반복 텍스트 |
| `FigureCaption` | 그림 설명 | "그림 3.1: 시스템 아키텍처" |
| `Image` | 이미지 참조 | 이미지 메타데이터 |
| `Formula` | 수학 수식 | 수식 텍스트 |
| `Address` | 주소 정보 | "서울특별시 강남구..." |
| `EmailAddress` | 이메일 주소 | "user@example.com" |
| `CodeSnippet` | 코드 블록 | 프로그래밍 코드 |
| `PageBreak` | 페이지 구분 | 페이지 경계 표시 |
| `PageNumber` | 페이지 번호 | "- 42 -" |
| `UncategorizedText` | 미분류 텍스트 | 분류하기 어려운 짧은 텍스트 |

각 Element는 공통 속성을 가지고 있습니다:

```run:python
from unstructured.partition.auto import partition

elements = partition(filename="report.pdf")

# 첫 번째 Element 상세 정보 확인
element = elements[0]

print(f"타입: {type(element).__name__}")
print(f"텍스트: {element.text}")
print(f"Element ID: {element.id}")
print(f"메타데이터 키: {list(element.metadata.to_dict().keys())}")
```
```output
타입: Title
텍스트: 2024년 4분기 매출 보고서
Element ID: a1b2c3d4e5f6
메타데이터 키: ['filename', 'file_directory', 'filetype', 'page_number', 'languages', 'parent_id', 'category_depth']
```

특히 RAG에서 유용한 메타데이터 필드들이 있습니다:

- **`page_number`**: 원본 문서의 페이지 번호. 검색 결과에 출처를 표시할 때 필수
- **`parent_id`**: 부모 Element와의 계층 관계. Title 아래의 NarrativeText를 연결
- **`category_depth`**: 제목의 깊이(h1=0, h2=1...). 청킹 시 섹션 경계를 판단하는 데 활용
- **`text_as_html`**: Table Element의 경우 HTML 형태로 표 구조를 보존
- **`languages`**: 감지된 언어. 다국어 문서 처리에 유용
- **`coordinates`**: Element의 문서 내 위치(바운딩 박스). `hi_res` 전략에서 제공

### 개념 3: 파싱 전략 — 속도와 정확도의 트레이드오프

> 💡 **비유**: 파싱 전략을 선택하는 것은 **음식점에서 메뉴를 고르는 것**과 비슷합니다. 패스트푸드(`fast`)는 빠르지만 맛의 깊이가 부족하고, 파인 다이닝(`hi_res`)은 시간이 걸리지만 정교하며, 뷔페(`ocr_only`)는 특수한 상황(스캔 문서)에 최적화되어 있죠. `auto`는 셰프 추천 메뉴로, 상황에 맞는 최선을 골라줍니다.

Unstructured.io는 PDF와 이미지 처리를 위해 4가지 파싱 전략을 제공합니다:

**1. `fast` 전략 — 속도 우선**

```python
from unstructured.partition.pdf import partition_pdf

# fast: pdfminer로 텍스트를 추출한 뒤 partition_text로 분류
elements = partition_pdf(
    filename="report.pdf",
    strategy="fast"  # 가장 빠름, 텍스트 선택 가능한 PDF에 적합
)
```

- 내부적으로 `pdfminer`로 텍스트를 추출한 뒤 `partition_text`로 Element를 분류합니다
- **장점**: 매우 빠름, 추가 모델 불필요
- **단점**: 레이아웃 정보 활용 불가, 표 구조 인식 제한
- **적합**: 텍스트가 잘 추출되는 일반 PDF (논문, 보고서 등)

**2. `hi_res` 전략 — 정확도 우선**

```python
# hi_res: detectron2로 레이아웃을 분석한 뒤 Element를 분류
elements = partition_pdf(
    filename="complex_report.pdf",
    strategy="hi_res",              # 레이아웃 분석 모델 사용
    infer_table_structure=True,     # 표 구조를 HTML로 보존
    languages=["kor", "eng"]        # OCR 언어 지정
)
```

- `detectron2` (Facebook AI의 컴퓨터 비전 모델)로 문서 레이아웃을 분석합니다
- **장점**: 표, 이미지, 다단 레이아웃 등 복잡한 구조 정확히 인식
- **단점**: 느림 (GPU 권장), 추가 모델 설치 필요
- **적합**: 표가 많은 재무보고서, 복잡한 레이아웃의 기술 문서

**3. `ocr_only` 전략 — 스캔 문서 전용**

```python
# ocr_only: Tesseract OCR로 이미지에서 텍스트를 추출
elements = partition_pdf(
    filename="scanned_document.pdf",
    strategy="ocr_only",           # OCR만 사용
    languages=["kor"]              # 한국어 OCR
)
```

- Tesseract OCR로 이미지에서 텍스트를 추출한 뒤 분류합니다
- **장점**: 스캔 문서나 이미지 기반 PDF에서 텍스트 추출 가능
- **단점**: OCR 정확도에 의존, 텍스트 기반 PDF에선 비효율적
- **적합**: 스캔된 계약서, 영수증, 손글씨 문서

**4. `auto` 전략 — 자동 선택 (기본값)**

```python
# auto: 문서 특성에 따라 최적 전략을 자동으로 선택
elements = partition_pdf(
    filename="any_document.pdf",
    strategy="auto"  # 기본값 — 대부분의 경우 이것으로 충분
)
```

- 문서의 특성을 분석하여 자동으로 전략을 선택합니다
- 텍스트가 추출 가능하면 `fast`, 표 추출이 필요하면 `hi_res`, 이미지 기반이면 `ocr_only`
- **적합**: 문서 형식을 미리 알 수 없는 경우, 대량 문서 일괄 처리

전략 비교를 정리하면:

| 전략 | 속도 | 정확도 | 표 인식 | 스캔 문서 | 의존성 |
|------|------|--------|---------|----------|--------|
| `fast` | ⚡⚡⚡ | ★★☆ | 제한적 | ✗ | pdfminer |
| `hi_res` | ⚡ | ★★★ | ✓ (HTML) | ✓ | detectron2, Tesseract |
| `ocr_only` | ⚡⚡ | ★★☆ | ✗ | ✓ | Tesseract |
| `auto` | 가변 | 가변 | 자동 | 자동 | 가변 |

### 개념 4: LangChain 연동 — UnstructuredLoader

Unstructured.io의 Element를 직접 다루는 것도 좋지만, RAG 파이프라인에서는 LangChain의 Document 객체로 변환하는 것이 편리합니다. `langchain-unstructured` 패키지가 이 다리를 놓아줍니다.

```python
from langchain_unstructured import UnstructuredLoader

# 단일 파일 로드
loader = UnstructuredLoader(
    file_path="report.pdf",
    strategy="hi_res",               # 파싱 전략 지정
    languages=["kor", "eng"],         # OCR 언어
)

docs = loader.load()
# 각 Element가 하나의 Document 객체로 변환됨
```

[3.1절](03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md)에서 배운 `Document(page_content, metadata)` 구조를 기억하시나요? `UnstructuredLoader`는 각 Element의 텍스트를 `page_content`로, Element 타입과 메타데이터를 `metadata`로 매핑합니다.

```run:python
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(file_path="report.pdf")
docs = loader.load()

# Document 객체의 구조 확인
doc = docs[0]
print(f"page_content: {doc.page_content[:60]}...")
print(f"metadata keys: {list(doc.metadata.keys())}")
print(f"category: {doc.metadata.get('category', 'N/A')}")
print(f"filetype: {doc.metadata.get('filetype', 'N/A')}")
```
```output
page_content: 2024년 4분기 매출 보고서...
metadata keys: ['source', 'category', 'filetype', 'page_number', 'languages', 'parent_id', 'category_depth']
category: Title
filetype: application/pdf
```

> 🔥 **실무 팁**: `langchain-unstructured` 패키지의 `UnstructuredLoader`는 구버전의 `langchain-community`에 있던 `UnstructuredFileLoader`를 대체합니다. 새로운 프로젝트에서는 반드시 `langchain-unstructured`를 사용하세요. 구버전의 `mode="elements"` 파라미터 대신, 새 버전에서는 기본적으로 각 Element가 별도의 Document로 변환됩니다.

## 실습: 직접 해보기

다양한 형식의 문서를 `partition()`으로 통합 처리하고, Element 타입별로 분석하는 실전 코드를 작성해보겠습니다.

```python
"""
Unstructured.io 실습: 다양한 문서 형식 통합 파싱
필요 패키지: pip install "unstructured[all-docs]" langchain-unstructured
"""
from collections import Counter
from unstructured.partition.auto import partition


def analyze_document(filepath: str, strategy: str = "auto") -> dict:
    """문서를 파싱하고 Element 통계를 반환합니다."""
    
    # 1. 문서 파싱 — 파일 형식 자동 감지
    elements = partition(
        filename=filepath,
        strategy=strategy,
        languages=["kor", "eng"]  # 한국어 + 영어 OCR 지원
    )
    
    # 2. Element 타입별 통계
    type_counts = Counter(type(el).__name__ for el in elements)
    
    # 3. 결과 출력
    print(f"\n📄 파일: {filepath}")
    print(f"   전략: {strategy}")
    print(f"   총 Element 수: {len(elements)}")
    print(f"   Element 타입 분포:")
    for elem_type, count in type_counts.most_common():
        print(f"     - {elem_type}: {count}개")
    
    return {
        "elements": elements,
        "type_counts": type_counts,
        "total": len(elements)
    }


def extract_tables(elements: list) -> list[str]:
    """Table Element만 추출하여 HTML 형태로 반환합니다."""
    from unstructured.documents.elements import Table
    
    tables = []
    for el in elements:
        if isinstance(el, Table):
            # text_as_html 메타데이터에 표 구조가 HTML로 보존됨
            html = el.metadata.text_as_html
            if html:
                tables.append(html)
            else:
                tables.append(el.text)  # HTML이 없으면 텍스트로 대체
    return tables


def build_rag_documents(elements: list) -> list[dict]:
    """Element를 RAG에 적합한 Document 형태로 변환합니다."""
    from unstructured.documents.elements import (
        Title, NarrativeText, ListItem, Table
    )
    
    documents = []
    current_title = ""
    
    for el in elements:
        # 제목 추적 — 이후 Element의 메타데이터로 활용
        if isinstance(el, Title):
            current_title = el.text
            continue
        
        # 본문, 리스트, 표만 RAG Document로 변환
        if isinstance(el, (NarrativeText, ListItem, Table)):
            doc = {
                "content": el.text,
                "metadata": {
                    "type": type(el).__name__,
                    "section_title": current_title,
                    "page_number": el.metadata.page_number,
                    "source": el.metadata.filename,
                }
            }
            
            # Table인 경우 HTML 표현도 보존
            if isinstance(el, Table) and el.metadata.text_as_html:
                doc["metadata"]["table_html"] = el.metadata.text_as_html
            
            documents.append(doc)
    
    return documents


# 실행 예시
if __name__ == "__main__":
    # 1. PDF 분석
    result = analyze_document("report.pdf", strategy="hi_res")
    
    # 2. 표 추출
    tables = extract_tables(result["elements"])
    print(f"\n📊 추출된 표: {len(tables)}개")
    
    # 3. RAG Document 변환
    rag_docs = build_rag_documents(result["elements"])
    print(f"📝 RAG Document: {len(rag_docs)}개")
    
    # 4. 샘플 확인
    if rag_docs:
        sample = rag_docs[0]
        print(f"\n--- 첫 번째 Document ---")
        print(f"섹션: {sample['metadata']['section_title']}")
        print(f"타입: {sample['metadata']['type']}")
        print(f"내용: {sample['content'][:100]}...")
```

이 코드의 핵심 포인트를 짚어보겠습니다:

1. **`partition()`에 파일 경로만 전달**하면 형식 자동 감지 → 적절한 파서 호출 → Element 반환까지 한 번에 처리됩니다
2. **Element 타입별 필터링**으로 표만 따로 추출하거나, 본문 텍스트만 모을 수 있습니다
3. **`Title`을 추적하여 `section_title` 메타데이터를 추가**하면, 검색 결과에서 "이 단락이 어느 섹션에 속하는지" 보여줄 수 있습니다 — 이는 이후 [Ch4: 텍스트 청킹 전략](04-텍스트-청킹-전략-문서-분할과-최적화/01-청킹의-중요성과-기본-원리.md)에서 더 자세히 다룹니다

## 더 깊이 알아보기

### Unstructured.io의 탄생 스토리

Unstructured.io의 창업 이야기는 꽤 극적입니다. 창업자 Brian Raymond는 CIA 정보 분석관 출신으로, 이후 미국 국가안전보장회의(NSC)에서 이라크 담당 디렉터를 역임했습니다. 이후 AI 기업 Primer에서 일하면서, 고객들이 항상 같은 문제에 부딪히는 것을 목격했습니다 — **"AI/ML 솔루션을 도입하고 싶지만, 데이터가 사용 불가능한 형식에 갇혀 있다."**

10년간 NLP 업계에서 이 문제를 반복적으로 경험한 Raymond는 2022년 7월, 드디어 직접 해결하기로 결심하고 Unstructured.io를 설립합니다. 처음에는 커스텀 NER이나 관계 추출 모델에 깨끗한 훈련 데이터를 제공하는 오픈소스 툴킷을 만들 계획이었죠.

그런데 운명의 타이밍이 맞아 떨어졌습니다. 오픈소스 라이브러리의 초기 버전을 2022년 9월에 출시한 지 불과 몇 주 뒤, ChatGPT가 세상에 등장한 겁니다. 갑자기 수천 명의 개발자가 "자신의 데이터와 대화하고 싶다"며 몰려들었고, Unstructured 라이브러리는 그 수요에 완벽하게 부합했습니다. 이후 $65M 이상의 투자를 유치하며 RAG 생태계의 핵심 인프라로 자리잡았습니다.

### partition() vs 개별 파서 — 언제 무엇을 쓸까?

`partition()` 함수는 내부적으로 `libmagic`을 사용하여 파일의 MIME 타입을 감지하고, 적절한 전용 파서 함수(`partition_pdf`, `partition_html` 등)로 라우팅합니다. 대부분의 경우 `partition()`만 사용하면 충분하지만, 전용 파서를 직접 호출하면 해당 형식 고유의 파라미터에 접근할 수 있습니다:

```python
# 이메일 전용 파라미터 — partition_email만 지원
from unstructured.partition.email import partition_email

elements = partition_email(
    filename="newsletter.eml",
    content_source="text/html",       # HTML 본문 우선 추출
    process_attachments=True,         # 첨부파일도 재귀적으로 파싱
    include_headers=True              # 발신자, 수신자 정보 포함
)

# XML 전용 파라미터 — partition_xml만 지원
from unstructured.partition.xml import partition_xml

elements = partition_xml(
    filename="data.xml",
    xml_keep_tags=True,               # XML 태그 정보 보존
    xml_path="/catalog/book"          # 특정 XPath만 추출
)
```

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Unstructured.io는 PDF 전용 도구다" — 아닙니다! PDF 파싱이 가장 유명하지만, 실제로는 25가지 이상의 형식을 지원합니다. Word, PowerPoint, HTML, 이메일, 이미지, CSV, EPUB 등 거의 모든 문서 형식을 처리할 수 있습니다. 심지어 최근에는 MP3, MP4 같은 오디오/비디오 파일도 지원합니다.

> 💡 **알고 계셨나요?**: `hi_res` 전략에서 사용하는 `detectron2`는 Facebook AI Research(FAIR)에서 개발한 객체 감지 모델입니다. 원래 사진 속 사물(사람, 자동차 등)을 인식하기 위해 만들어졌는데, Unstructured.io 팀이 이를 **문서 레이아웃 분석**에 적용한 것이죠. 제목, 본문, 표, 이미지 영역을 "객체"로 감지하는 발상의 전환이 핵심이었습니다.

> 🔥 **실무 팁**: `"unstructured[all-docs]"`를 설치하면 모든 의존성이 포함되지만 패키지 크기가 상당합니다. 프로덕션에서는 필요한 형식만 선택적으로 설치하세요. 예: PDF만 필요하면 `"unstructured[pdf]"`, 이미지 OCR이 필요하면 `"unstructured[tesseract]"`로 설치할 수 있습니다. 또한 `hi_res` 전략은 GPU가 있으면 10배 이상 빨라지므로, 대량의 PDF를 처리할 때는 GPU 환경을 고려하세요.

> 🔥 **실무 팁**: `partition()`으로 추출한 Element 중 `Header`와 `Footer`는 RAG에 넣지 않는 것이 좋습니다. "페이지 1/15", "Confidential" 같은 반복 텍스트가 검색 품질을 떨어뜨리거든요. Element 타입 필터링으로 쉽게 제거할 수 있습니다:
> ```python
> from unstructured.documents.elements import Header, Footer, PageNumber
> clean = [el for el in elements if not isinstance(el, (Header, Footer, PageNumber))]
> ```

## 핵심 정리

| 개념 | 설명 |
|------|------|
| `partition()` | 파일 형식을 자동 감지하여 적절한 파서로 라우팅하는 범용 파싱 함수 |
| Element 타입 | Title, NarrativeText, Table 등 15가지 의미 단위로 문서 구조를 보존 |
| `fast` 전략 | pdfminer 기반, 빠르지만 레이아웃 분석 없음. 텍스트 기반 PDF에 적합 |
| `hi_res` 전략 | detectron2 기반 레이아웃 분석, 정확하지만 느림. 표/이미지가 많은 문서에 적합 |
| `ocr_only` 전략 | Tesseract OCR 전용. 스캔 문서나 이미지 기반 PDF에 적합 |
| `auto` 전략 | 문서 특성에 따라 최적 전략을 자동 선택 (기본값) |
| `text_as_html` | Table Element의 메타데이터에 표 구조가 HTML로 보존됨 |
| `parent_id` / `category_depth` | Element 간 계층 관계를 추적. 청킹 시 섹션 경계 판단에 활용 |
| `UnstructuredLoader` | `langchain-unstructured` 패키지의 LangChain 연동 로더 |

## 다음 섹션 미리보기

이번 섹션에서 Unstructured.io로 문서를 **의미 단위의 Element**로 분해하는 방법을 배웠습니다. 하지만 실무에서는 한 가지 형식의 파일만 처리하는 경우가 드뭅니다. 다음 섹션 **"웹 데이터와 API 소스 — WebBaseLoader와 실시간 데이터 수집"**에서는 웹 페이지, API 응답 등 동적 데이터 소스에서 RAG 데이터를 수집하는 방법을 다룹니다. LangChain의 `WebBaseLoader`와 `RecursiveUrlLoader` 등을 활용하여, 정적 파일 중심이었던 데이터 수집 범위를 동적 웹 데이터까지 확장하는 전략을 살펴봅니다.

## 참고 자료

- [Unstructured.io GitHub Repository](https://github.com/Unstructured-IO/unstructured) - 오픈소스 코드와 설치 가이드, 지원 형식 전체 목록을 확인할 수 있습니다
- [Unstructured 공식 문서 — Partitioning](https://docs.unstructured.io/open-source/core-functionality/partitioning) - partition 함수의 전체 파라미터와 전략별 상세 설명
- [Unstructured 공식 문서 — Document Elements](https://docs.unstructured.io/open-source/concepts/document-elements) - 15가지 Element 타입과 메타데이터 필드의 완전한 레퍼런스
- [LangChain Unstructured Integration](https://docs.langchain.com/oss/python/integrations/document_loaders/unstructured_file) - LangChain에서 UnstructuredLoader를 사용하는 방법과 예제
- [How We Got Started — Unstructured.io Blog](https://unstructured.io/blog/how-we-got-started) - Brian Raymond의 창업 스토리와 Unstructured.io의 탄생 배경
- [Choose a Partitioning Strategy — Unstructured Docs](https://unstructured.readthedocs.io/en/main/best_practices/strategies.html) - fast, hi_res, ocr_only, auto 전략의 상세 비교와 선택 가이드

---
### 🔗 Related Sessions
- [document](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [page_content](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [metadata](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [pypdfloader](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md) (prerequisite)
- [pdfplumberloader](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md) (prerequisite)
- [unstructuredpdfloader](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md) (prerequisite)
- [pdfminer.six](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md) (prerequisite)
- [content_type](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/02-pdf-문서-처리-텍스트-추출과-레이아웃-분석.md) (prerequisite)
