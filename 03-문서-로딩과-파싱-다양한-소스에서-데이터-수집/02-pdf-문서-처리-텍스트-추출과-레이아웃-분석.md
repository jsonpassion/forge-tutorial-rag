# PDF 문서 처리 — 텍스트 추출과 레이아웃 분석

> PDF는 RAG 파이프라인에서 가장 흔하면서도 가장 까다로운 문서 형식입니다. 이 섹션에서는 PDF의 구조적 특성을 이해하고, Python 라이브러리별 파싱 전략을 비교하며, 테이블과 복잡한 레이아웃을 효과적으로 처리하는 방법을 학습합니다.

## 개요

앞서 [문서 로딩 기초 — LangChain Document Loaders](ch03/session_01.md)에서 `Document` 객체의 구조와 다양한 로더의 기본 사용법을 배웠습니다. 이번 섹션에서는 실무에서 가장 빈번하게 마주치는 **PDF 문서**에 집중합니다. PDF는 "보이는 대로" 렌더링하기 위해 설계된 포맷이라, 텍스트를 "읽히는 순서대로" 추출하는 것이 생각보다 훨씬 어렵습니다.

**선수 지식**: Session 3.1에서 다룬 `Document` 객체(`page_content`, `metadata`), `load()` / `lazy_load()` 패턴
**학습 목표**:
- PDF 포맷이 텍스트 추출에 어려운 이유를 구조적으로 이해한다
- pypdf, pdfplumber, pdfminer.six 세 라이브러리의 특성과 장단점을 비교할 수 있다
- LangChain의 `PyPDFLoader`, `PDFPlumberLoader`를 사용하여 PDF를 Document 객체로 변환할 수 있다
- 테이블, 멀티컬럼, 헤더/푸터가 포함된 복잡한 PDF의 처리 전략을 수립할 수 있다

## 왜 알아야 할까?

기업 문서의 상당수는 PDF로 존재합니다. 연구 논문, 재무 보고서, 법률 문서, 기술 매뉴얼 — 이 모든 것이 PDF죠. RAG 시스템을 구축할 때 "PDF를 얼마나 잘 파싱하느냐"가 곧 **검색 품질의 천장**을 결정합니다.

문제는 PDF 파싱이 잘못되면 그 영향이 파이프라인 전체에 전파된다는 것입니다. 텍스트가 뒤섞이면 → 청킹이 엉망이 되고 → 임베딩 품질이 떨어지고 → 검색 결과가 부정확해집니다. 말 그대로 "Garbage In, Garbage Out"이거든요. 그래서 PDF 파싱은 RAG 파이프라인의 **첫 번째 관문이자 가장 중요한 전처리 단계**입니다.

## 핵심 개념

### 개념 1: PDF는 왜 파싱이 어려울까?

> 💡 **비유**: PDF를 읽는 것은 마치 **완성된 퍼즐 사진만 보고 각 조각의 순서를 맞추는 것**과 같습니다. 우리 눈에는 완성된 문서가 보이지만, 내부적으로는 텍스트 조각들이 좌표(x, y)에 "뿌려진" 상태입니다.

PDF(Portable Document Format)는 **화면 표시와 인쇄**를 위해 설계된 포맷입니다. HTML처럼 "이 텍스트는 제목이고, 이 다음은 본문이야"라는 **의미 구조(semantic structure)** 를 갖고 있지 않죠. 대신 "이 글자를 (72, 540) 좌표에, 12pt 폰트로 그려라"라는 **렌더링 명령**의 집합입니다.

이것이 만드는 핵심적인 어려움들을 살펴볼까요?

**1. 읽기 순서(Reading Order) 문제**

멀티컬럼 레이아웃의 PDF에서 텍스트를 단순히 위에서 아래로 추출하면, 왼쪽 컬럼 1줄 → 오른쪽 컬럼 1줄이 번갈아 섞이게 됩니다.

```
┌─────────────────────────────────┐
│ 왼쪽 컬럼 A     │ 오른쪽 컬럼 B   │
│ 왼쪽 컬럼 A-2   │ 오른쪽 컬럼 B-2 │
│ 왼쪽 컬럼 A-3   │ 오른쪽 컬럼 B-3 │
└─────────────────────────────────┘

❌ 단순 추출: "왼쪽 컬럼 A 오른쪽 컬럼 B 왼쪽 컬럼 A-2 ..."
✅ 올바른 추출: "왼쪽 컬럼 A 왼쪽 컬럼 A-2 왼쪽 컬럼 A-3 오른쪽 컬럼 B ..."
```

**2. 테이블 구조 손실**

PDF에는 "테이블"이라는 개념이 없습니다. 셀 안의 텍스트와 선(line) 객체가 독립적으로 존재할 뿐이에요. 테이블을 인식하려면 선과 텍스트의 좌표를 분석해서 행/열 구조를 **역추론**해야 합니다.

**3. 헤더/푸터/페이지 번호**

모든 페이지에 반복되는 헤더, 푸터, 페이지 번호가 본문 텍스트에 섞여 들어옵니다. "제3장 데이터 분석 42"처럼 의미 없는 텍스트가 중간에 끼어들죠.

**4. 이미지 속 텍스트**

스캔된 PDF나 이미지로 삽입된 텍스트는 일반 텍스트 추출로는 전혀 읽을 수 없습니다. OCR(Optical Character Recognition)이 필요합니다.

> ⚠️ **흔한 오해**: "PDF에서 텍스트를 복사-붙여넣기 할 수 있으니까 프로그래밍으로도 쉽게 추출할 수 있겠지?"라고 생각하기 쉽지만, 실제로 PDF 뷰어의 텍스트 선택도 내부적으로 복잡한 좌표 계산을 수행합니다. 심지어 복사한 텍스트의 순서가 틀리는 경우도 흔하죠.

### 개념 2: Python PDF 파싱 라이브러리 비교

> 💡 **비유**: PDF 파싱 라이브러리를 고르는 것은 **도구 상자에서 적합한 공구를 선택하는 것**과 같습니다. 나사를 돌리는데 망치를 쓰면 안 되듯이, PDF의 특성에 따라 적합한 라이브러리가 다릅니다.

Python에서 PDF를 다루는 대표적인 라이브러리 세 가지를 비교해보겠습니다.

#### pypdf — 가볍고 빠른 범용 도구

[pypdf](https://github.com/py-pdf/pypdf)는 순수 Python으로 작성된 PDF 라이브러리입니다. 외부 의존성이 없어서 설치가 간편하고, 텍스트 추출부터 PDF 병합/분할까지 다양한 기능을 제공합니다.

```python
# pypdf 기본 텍스트 추출
from pypdf import PdfReader

reader = PdfReader("report.pdf")

# 각 페이지에서 텍스트 추출
for i, page in enumerate(reader.pages):
    text = page.extract_text()  # 페이지의 텍스트 추출
    print(f"--- 페이지 {i + 1} ---")
    print(text[:200])  # 처음 200자만 미리보기
```

**장점**: 설치 간편(순수 Python), 빠른 속도, 활발한 유지보수
**단점**: 복잡한 레이아웃이나 테이블 추출에 취약

#### pdfplumber — 테이블 추출의 강자

[pdfplumber](https://github.com/jsvine/pdfplumber)는 pdfminer.six 위에 구축된 라이브러리로, **테이블 추출**에 특화되어 있습니다. PDF 페이지의 모든 문자, 선, 사각형 등의 좌표 정보에 접근할 수 있고, 이를 기반으로 테이블 구조를 자동 감지합니다.

```python
# pdfplumber로 테이블 추출
import pdfplumber

with pdfplumber.open("financial_report.pdf") as pdf:
    page = pdf.pages[0]  # 첫 번째 페이지

    # 테이블 자동 감지 및 추출
    tables = page.extract_tables()
    for table in tables:
        for row in table:
            print(row)  # 각 행을 리스트로 반환

    # 텍스트도 추출 가능
    text = page.extract_text()
```

**장점**: 뛰어난 테이블 추출(96% 인식률), 시각적 디버깅 도구 제공, 좌표 기반 영역 선택 가능
**단점**: pypdf보다 느림, 스캔된 PDF 미지원

#### pdfminer.six — 정밀한 레이아웃 분석

[pdfminer.six](https://github.com/pdfminer/pdfminer.six)는 PDF의 텍스트 위치, 폰트, 크기 등 **저수준 정보**에 접근할 수 있는 라이브러리입니다. 텍스트의 정확한 좌표와 스타일 정보가 필요한 경우에 적합합니다.

```python
# pdfminer.six 간단 사용법
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTChar

# 간단한 텍스트 추출
text = extract_text("research_paper.pdf")

# 레이아웃 분석 — 각 텍스트 요소의 위치와 폰트 정보 접근
for page_layout in extract_pages("research_paper.pdf"):
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            # 텍스트와 바운딩 박스(좌표) 정보
            print(f"텍스트: {element.get_text()[:50]}")
            print(f"위치: ({element.x0:.1f}, {element.y0:.1f})")
```

**장점**: 가장 정밀한 레이아웃 정보, 폰트/크기 감지로 제목 구분 가능
**단점**: API가 복잡, 단독으로 테이블 추출 어려움

세 라이브러리를 한눈에 비교해볼까요?

| 기준 | pypdf | pdfplumber | pdfminer.six |
|------|-------|------------|-------------|
| 설치 용이성 | ⭐⭐⭐ (순수 Python) | ⭐⭐ (pdfminer 의존) | ⭐⭐ |
| 텍스트 추출 속도 | 빠름 | 보통 | 느림 |
| 테이블 추출 | ❌ 약함 | ✅ 우수 | ❌ 약함 |
| 레이아웃 분석 | 기본적 | 좋음 | 매우 정밀 |
| 멀티컬럼 처리 | 보통 | 좋음 | 좋음 |
| 메타데이터 접근 | ✅ | ✅ | ✅ |
| 암호화 PDF | ✅ | ❌ | ❌ |
| 주요 사용처 | 일반 텍스트 PDF | 테이블 중심 PDF | 레이아웃 분석 필요 시 |

### 개념 3: LangChain PDF 로더 활용

> 💡 **비유**: LangChain의 PDF 로더는 **자동 번역기**와 같습니다. 내부적으로 위의 라이브러리들을 사용하지만, 결과물을 항상 Session 3.1에서 배운 `Document` 객체로 표준화해줍니다.

LangChain은 다양한 PDF 파싱 라이브러리를 감싸는 **통일된 인터페이스**를 제공합니다. 어떤 로더를 쓰든 결과는 동일한 `Document(page_content=..., metadata=...)` 형태입니다.

#### PyPDFLoader — 가장 기본적인 선택

```python
from langchain_community.document_loaders import PyPDFLoader

# PDF 로드 — 페이지별로 Document 객체 생성
loader = PyPDFLoader("report.pdf")
documents = loader.load()

# 결과 확인
for doc in documents[:2]:
    print(f"페이지: {doc.metadata['page']}")       # 메타데이터에 페이지 번호 포함
    print(f"내용: {doc.page_content[:100]}...")     # 추출된 텍스트
    print(f"출처: {doc.metadata['source']}")        # 파일 경로
    print()
```

`PyPDFLoader`는 내부적으로 `pypdf`를 사용하며, 기본적으로 **페이지 단위**로 `Document` 객체를 생성합니다. `mode="single"` 옵션을 주면 전체 PDF를 하나의 Document로 만들 수도 있습니다.

#### PDFPlumberLoader — 테이블이 있는 PDF에 추천

```python
from langchain_community.document_loaders import PDFPlumberLoader

# pdfplumber 기반 로더 — 테이블/컬럼 처리에 유리
loader = PDFPlumberLoader("financial_report.pdf")
documents = loader.load()

# pdfplumber의 텍스트 추출 결과가 page_content에 담김
print(documents[0].page_content[:300])
```

> 🔥 **실무 팁**: PDF에 테이블이 많다면 `PDFPlumberLoader`가 기본 `PyPDFLoader`보다 훨씬 나은 결과를 줍니다. 하지만 속도가 더 느리므로, 먼저 PDF의 특성을 파악한 뒤 적합한 로더를 선택하세요. 단순 텍스트 위주 PDF → `PyPDFLoader`, 테이블/컬럼 PDF → `PDFPlumberLoader`가 일반적인 가이드라인입니다.

#### UnstructuredPDFLoader — 올인원 솔루션

복잡한 PDF(스캔 문서, 혼합 레이아웃)에는 [Unstructured.io](https://github.com/Unstructured-IO/unstructured)를 활용하는 `UnstructuredPDFLoader`가 좋은 선택입니다.

```python
from langchain_community.document_loaders import UnstructuredPDFLoader

# strategy 옵션: "fast"(빠름), "hi_res"(고해상도, 테이블/이미지 인식)
loader = UnstructuredPDFLoader(
    "complex_document.pdf",
    mode="elements",       # 제목, 본문, 테이블 등 요소별로 분리
    strategy="hi_res",     # 레이아웃 분석 모델 사용
)
documents = loader.load()

# 각 요소의 카테고리 확인
for doc in documents[:5]:
    category = doc.metadata.get("category", "unknown")
    print(f"[{category}] {doc.page_content[:80]}")
```

Unstructured의 `hi_res` 전략은 내부적으로 레이아웃 분석 모델을 사용해서 제목(Title), 본문(NarrativeText), 테이블(Table) 등을 **자동 분류**합니다. `mode="elements"`와 함께 사용하면 요소 타입별로 분리된 Document를 얻을 수 있어, 이후 청킹 전략에서 큰 이점이 됩니다.

### 개념 4: 테이블 처리 전략

> 💡 **비유**: PDF에서 테이블을 추출하는 것은 **벽돌담 사진에서 각 벽돌의 크기를 재는 것**과 같습니다. 눈으로는 행과 열이 보이지만, 컴퓨터에게는 좌표에 흩어진 텍스트와 선분일 뿐이거든요.

RAG 시스템에서 테이블 데이터의 처리는 특히 중요합니다. 재무 보고서의 수치, 연구 논문의 실험 결과 등 핵심 정보가 테이블에 담겨 있는 경우가 많으니까요.

**전략 1: 마크다운 테이블로 변환**

테이블을 마크다운 형식으로 변환하면 LLM이 구조를 이해하기 쉽습니다.

```python
import pdfplumber

def extract_tables_as_markdown(pdf_path: str) -> list[str]:
    """PDF에서 테이블을 추출하여 마크다운 형식으로 변환"""
    markdown_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()

            for table_idx, table in enumerate(tables):
                if not table or not table[0]:
                    continue

                # 첫 행을 헤더로 사용
                headers = table[0]
                md = "| " + " | ".join(str(h) for h in headers) + " |\n"
                md += "| " + " | ".join("---" for _ in headers) + " |\n"

                # 나머지 행을 데이터로 추가
                for row in table[1:]:
                    md += "| " + " | ".join(str(c) for c in row) + " |\n"

                markdown_tables.append(md)

    return markdown_tables
```

**전략 2: 테이블을 별도 Document로 분리**

테이블과 본문 텍스트를 별도의 Document 객체로 관리하면, 검색 시 테이블 데이터에 대한 정확한 매칭이 가능합니다.

```python
from langchain_core.documents import Document

def create_table_documents(
    pdf_path: str,
    markdown_tables: list[str]
) -> list[Document]:
    """마크다운 테이블을 별도 Document 객체로 생성"""
    docs = []
    for i, table_md in enumerate(markdown_tables):
        doc = Document(
            page_content=table_md,
            metadata={
                "source": pdf_path,
                "content_type": "table",     # 테이블임을 표시
                "table_index": i,
            }
        )
        docs.append(doc)
    return docs
```

> 🔥 **실무 팁**: 테이블 Document의 `metadata`에 `"content_type": "table"`을 추가해두면, 나중에 메타데이터 필터링으로 "테이블 데이터에서만 검색"하는 것이 가능해집니다. 이 기법은 [검색 품질 향상](10-검색-품질-향상-유사도-검색과-메타데이터-필터링/01-유사도-검색-심화-top-k와-임계값-최적화.md)에서 자세히 다룹니다.

### 개념 5: 헤더/푸터 제거와 전처리

실무에서 PDF를 파싱할 때 반드시 처리해야 하는 것이 반복적인 헤더, 푸터, 페이지 번호입니다. 이들이 본문에 섞이면 청킹과 검색 품질에 부정적인 영향을 미칩니다.

```python
import re

def clean_pdf_text(text: str) -> str:
    """PDF 추출 텍스트에서 일반적인 노이즈를 제거"""
    # 페이지 번호 패턴 제거 (예: "- 42 -", "Page 3", "3/10")
    text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)

    # 연속된 빈 줄을 하나로 축소
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 하이픈으로 끊어진 단어 복원 (예: "docu-\nment" → "document")
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    return text.strip()
```

```run:python
import re

def clean_pdf_text(text: str) -> str:
    """PDF 추출 텍스트에서 일반적인 노이즈를 제거"""
    text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
    text = re.sub(r'\n\s*Page\s+\d+\s*\n', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()

# 테스트: 노이즈가 포함된 PDF 추출 텍스트
raw_text = """제3장 데이터 분석 방법론

본 연구에서는 다양한 데이터를 수집하여 분석하였으며, docu-
ment 처리 파이프라인을 구축하였다.

- 42 -

다음 단계에서는 전처리를 수행하였다.

Page 43

결과는 다음과 같다."""

cleaned = clean_pdf_text(raw_text)
print(cleaned)
```

```output
제3장 데이터 분석 방법론

본 연구에서는 다양한 데이터를 수집하여 분석하였으며, document 처리 파이프라인을 구축하였다.

다음 단계에서는 전처리를 수행하였다.

결과는 다음과 같다.
```

페이지 번호("- 42 -", "Page 43")가 제거되고, 줄 바꿈으로 끊어진 단어("docu-ment")가 복원된 것을 확인할 수 있습니다.

## 실습: 직접 해보기

실제로 여러 PDF 로더를 비교해보는 완전한 코드입니다. 테스트용 PDF를 만들어서 각 로더의 결과를 비교해봅시다.

```python
"""
PDF 로더 비교 실습
필요 패키지: pip install pypdf pdfplumber pdfminer.six langchain-community
"""
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader


def clean_text(text: str) -> str:
    """PDF 추출 텍스트 정리"""
    text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    return text.strip()


def compare_loaders(pdf_path: str) -> None:
    """PyPDFLoader와 PDFPlumberLoader의 결과를 비교"""
    print(f"📄 PDF 파일: {pdf_path}\n")

    # 1. PyPDFLoader
    print("=" * 50)
    print("🔹 PyPDFLoader (pypdf 기반)")
    print("=" * 50)
    pypdf_loader = PyPDFLoader(pdf_path)
    pypdf_docs = pypdf_loader.load()

    for doc in pypdf_docs[:2]:  # 처음 2페이지만
        cleaned = clean_text(doc.page_content)
        print(f"\n[페이지 {doc.metadata['page']}]")
        print(f"글자 수: {len(cleaned)}")
        print(f"미리보기: {cleaned[:200]}...")
        print(f"메타데이터: {doc.metadata}")

    # 2. PDFPlumberLoader
    print("\n" + "=" * 50)
    print("🔹 PDFPlumberLoader (pdfplumber 기반)")
    print("=" * 50)
    plumber_loader = PDFPlumberLoader(pdf_path)
    plumber_docs = plumber_loader.load()

    for doc in plumber_docs[:2]:  # 처음 2페이지만
        cleaned = clean_text(doc.page_content)
        print(f"\n[페이지 {doc.metadata.get('page', 'N/A')}]")
        print(f"글자 수: {len(cleaned)}")
        print(f"미리보기: {cleaned[:200]}...")
        print(f"메타데이터: {doc.metadata}")

    # 3. 결과 비교
    print("\n" + "=" * 50)
    print("📊 비교 요약")
    print("=" * 50)
    print(f"PyPDFLoader    — 문서 수: {len(pypdf_docs)}, "
          f"총 글자: {sum(len(d.page_content) for d in pypdf_docs)}")
    print(f"PDFPlumberLoader — 문서 수: {len(plumber_docs)}, "
          f"총 글자: {sum(len(d.page_content) for d in plumber_docs)}")


# 실행
# compare_loaders("your_document.pdf")  # 본인의 PDF 경로를 지정하세요
```

테이블이 포함된 PDF를 처리하는 고급 실습도 해볼까요?

```python
"""
테이블 포함 PDF 처리 파이프라인
필요 패키지: pip install pdfplumber langchain-core
"""
import pdfplumber
from langchain_core.documents import Document


def process_pdf_with_tables(pdf_path: str) -> list[Document]:
    """텍스트와 테이블을 분리하여 Document 리스트로 반환"""
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # 1) 테이블 영역 감지
            tables = page.extract_tables()
            table_bboxes = [
                table.bbox for table in page.find_tables()
            ] if page.find_tables() else []

            # 2) 테이블 영역을 제외한 텍스트 추출
            if table_bboxes:
                # 테이블 영역을 잘라낸 페이지에서 텍스트 추출
                filtered_page = page
                for bbox in table_bboxes:
                    filtered_page = filtered_page.outside_bbox(bbox)
                body_text = filtered_page.extract_text() or ""
            else:
                body_text = page.extract_text() or ""

            # 3) 본문 텍스트를 Document로 생성
            if body_text.strip():
                documents.append(Document(
                    page_content=body_text.strip(),
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": "text",
                    }
                ))

            # 4) 각 테이블을 마크다운으로 변환하여 별도 Document 생성
            for table_idx, table in enumerate(tables):
                if not table or not table[0]:
                    continue
                # 마크다운 테이블 변환
                headers = [str(h) if h else "" for h in table[0]]
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join("---" for _ in headers) + " |\n"
                for row in table[1:]:
                    cells = [str(c) if c else "" for c in row]
                    md += "| " + " | ".join(cells) + " |\n"

                documents.append(Document(
                    page_content=md,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": "table",
                        "table_index": table_idx,
                    }
                ))

    return documents


# 사용 예시
# docs = process_pdf_with_tables("annual_report.pdf")
# text_docs = [d for d in docs if d.metadata["content_type"] == "text"]
# table_docs = [d for d in docs if d.metadata["content_type"] == "table"]
# print(f"텍스트 문서: {len(text_docs)}개, 테이블 문서: {len(table_docs)}개")
```

```run:python
# 로더 선택 가이드를 코드로 구현
def recommend_loader(has_tables: bool, has_scanned: bool, has_columns: bool) -> str:
    """PDF 특성에 따라 적합한 로더를 추천"""
    if has_scanned:
        return "UnstructuredPDFLoader (strategy='hi_res') — OCR 지원"
    elif has_tables and has_columns:
        return "PDFPlumberLoader — 테이블 + 레이아웃 분석"
    elif has_tables:
        return "PDFPlumberLoader — 테이블 추출 특화"
    elif has_columns:
        return "UnstructuredPDFLoader (strategy='fast') — 레이아웃 인식"
    else:
        return "PyPDFLoader — 빠르고 간편한 기본 선택"

# 다양한 PDF 시나리오 테스트
scenarios = [
    ("일반 텍스트 문서", False, False, False),
    ("재무 보고서 (테이블 포함)", True, False, False),
    ("학술 논문 (2단 컬럼)", False, False, True),
    ("스캔된 계약서", False, True, False),
    ("복잡한 기술 매뉴얼", True, False, True),
]

for name, tables, scanned, columns in scenarios:
    rec = recommend_loader(tables, scanned, columns)
    print(f"📄 {name}")
    print(f"   → {rec}\n")
```

```output
📄 일반 텍스트 문서
   → PyPDFLoader — 빠르고 간편한 기본 선택

📄 재무 보고서 (테이블 포함)
   → PDFPlumberLoader — 테이블 추출 특화

📄 학술 논문 (2단 컬럼)
   → UnstructuredPDFLoader (strategy='fast') — 레이아웃 인식

📄 스캔된 계약서
   → UnstructuredPDFLoader (strategy='hi_res') — OCR 지원

📄 복잡한 기술 매뉴얼
   → PDFPlumberLoader — 테이블 + 레이아웃 분석
```

## 더 깊이 알아보기

### PDF의 탄생 — 존 워녹의 "카멜롯 프로젝트"

1990년 여름, Adobe 공동 창업자 **존 워녹(John Warnock)** 은 6페이지짜리 백서를 작성했습니다. 코드명은 **"카멜롯 프로젝트(The Camelot Project)"**. 그가 해결하고 싶었던 문제는 단 하나였습니다: *"서로 다른 컴퓨터, 서로 다른 운영체제에서 동일한 문서를 정확하게 볼 수 있게 하자."*

당시에는 Mac에서 만든 문서를 Windows에서 열면 폰트가 깨지고, 레이아웃이 뒤틀리는 것이 일상이었거든요. 워녹은 이미 Adobe가 만든 PostScript(프린터 제어 언어)를 단순화한 "Interchange PostScript"라는 개념을 구상했고, 이것이 발전하여 1993년 6월 15일 **PDF(Portable Document Format)** 와 Adobe Acrobat이 공식 출시되었습니다.

아이러니하게도, PDF가 "어떤 화면에서든 동일하게 보이도록" 설계되었기 때문에 내부 구조가 **시각적 렌더링에 최적화**되었고, 이것이 오늘날 우리가 텍스트 추출에 고생하는 근본적인 이유입니다. PDF는 "사람이 읽기 위한" 포맷이지, "기계가 읽기 위한" 포맷이 아니었던 거죠.

> 💡 **알고 계셨나요?** 존 워녹이 PDF 개발의 영감을 얻은 것은 1985년, 자신의 연방 세금 신고서(1040 양식)를 PostScript로 다시 코딩한 경험이었습니다. 스티브 잡스가 그해 Apple LaserWriter를 공개할 때, 무대에서 출력한 문서 중 하나가 바로 워녹의 세금 신고서였다고 합니다.

### 왜 PDF 파싱은 "해결된 문제"가 아닌가?

2008년에 PDF는 ISO 32000-1 국제 표준이 되었지만, 30년 넘게 축적된 다양한 PDF 생성 도구들(각각 미묘하게 다른 방식으로 PDF를 생성)과 스캔 문서, 디지털 서명, 멀티미디어 내장 등 끊임없이 확장된 기능 때문에, "모든 PDF를 완벽하게 파싱하는" 도구는 아직 존재하지 않습니다. 이것이 여러 라이브러리를 상황에 맞게 선택해야 하는 이유입니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "PDF 로더 하나만 잘 고르면 모든 PDF를 완벽하게 처리할 수 있다." 실제로는 PDF의 종류(텍스트 기반 vs 스캔, 단일 컬럼 vs 멀티컬럼, 테이블 유무)에 따라 최적의 도구가 달라집니다. 프로덕션 환경에서는 PDF 특성을 먼저 분석하고 적합한 로더를 동적으로 선택하는 전략이 필요합니다.

> 💡 **알고 계셨나요?** pdfplumber의 테이블 인식률이 96%에 달한다는 연구 결과가 있습니다. 하지만 이는 선이 명확하게 그려진 테이블에 해당하며, 선 없이 공백으로만 구분된 테이블에서는 인식률이 크게 떨어집니다. `pdfplumber`의 `table_settings` 파라미터를 조정하면 이런 경우에도 품질을 개선할 수 있습니다.

> 🔥 **실무 팁**: RAG 파이프라인에서 PDF를 처리할 때, 파싱 결과를 **반드시 육안으로 검증**하세요. 특히 첫 번째 배치를 처리할 때 다음을 확인하면 좋습니다:
> 1. 텍스트 순서가 올바른지 (멀티컬럼이 섞이지 않았는지)
> 2. 테이블 데이터가 구조를 유지하는지
> 3. 헤더/푸터가 본문에 섞이지 않았는지
> 4. 특수 문자나 수식이 깨지지 않았는지

> 🔥 **실무 팁**: 대량의 PDF를 처리할 때는 `lazy_load()`를 활용하세요. Session 3.1에서 배운 것처럼, `load()` 대신 `lazy_load()`를 사용하면 메모리를 절약하며 수백 개의 PDF를 순차적으로 처리할 수 있습니다.
> ```python
> loader = PyPDFLoader("large_report.pdf")
> for doc in loader.lazy_load():  # 한 페이지씩 처리
>     processed = clean_text(doc.page_content)
>     # 청킹, 임베딩 등 후속 처리
> ```

## 핵심 정리

| 개념 | 설명 |
|------|------|
| PDF 파싱의 어려움 | PDF는 시각적 렌더링용 포맷이라 텍스트 순서, 테이블 구조, 의미 정보가 내부에 없음 |
| pypdf | 순수 Python, 빠르고 가벼움. 일반 텍스트 PDF에 적합. 최신 버전은 [공식 문서](https://pypdf.readthedocs.io/) 참고 |
| pdfplumber | 테이블 추출 특화, 좌표 기반 영역 선택. 테이블/컬럼 PDF에 적합 |
| pdfminer.six | 정밀한 레이아웃/폰트 분석. 저수준 PDF 정보 접근 필요 시 사용. 최신 버전은 [공식 문서](https://pdfminersix.readthedocs.io/) 참고 |
| PyPDFLoader | LangChain의 기본 PDF 로더. pypdf 기반, 페이지별 Document 생성 |
| PDFPlumberLoader | LangChain의 테이블 특화 PDF 로더. pdfplumber 기반 |
| UnstructuredPDFLoader | 레이아웃 분석 모델 기반 올인원 로더. 스캔/복잡한 PDF에 적합 |
| 테이블 → 마크다운 변환 | 테이블을 마크다운 형식으로 변환하면 LLM이 구조를 이해하기 쉬움 |
| `content_type` 메타데이터 | 테이블/텍스트를 구분하는 메타데이터로, 이후 필터링 검색에 활용 |

## 다음 섹션 미리보기

이번 섹션에서는 PDF라는 가장 까다로운 형식을 다루는 방법을 배웠습니다. 다음 섹션에서는 **Unstructured.io — 범용 문서 파싱 엔진**을 다룹니다. PDF를 포함한 25가지 이상의 형식을 하나의 `partition()` 함수로 처리하는 Unstructured.io를 깊이 살펴봅니다. Session 3.2에서 배운 `UnstructuredPDFLoader`의 내부 엔진을 깊이 파헤쳐 봅시다.

## 참고 자료

- [pypdf 공식 문서](https://pypdf.readthedocs.io/) — pypdf의 전체 API와 텍스트 추출 가이드
- [pdfplumber GitHub](https://github.com/jsvine/pdfplumber) — 테이블 추출 예제와 시각적 디버깅 도구 사용법이 잘 정리된 README
- [pdfminer.six 공식 문서](https://pdfminersix.readthedocs.io/) — 레이아웃 분석과 저수준 PDF 요소 접근 방법 안내
- [LangChain PyPDFLoader 문서](https://docs.langchain.com/oss/python/integrations/document_loaders/pypdfloader) — LangChain PDF 로더의 공식 가이드와 파라미터 설명
- [Unstructured.io GitHub](https://github.com/Unstructured-IO/unstructured) — 복잡한 문서 파싱을 위한 오픈소스 ETL 솔루션. `partition_pdf` 함수의 다양한 전략 설명
- [Extracting Data from PDFs — Challenges in RAG Applications](https://unstract.com/blog/pdf-hell-and-practical-rag-applications/) — PDF 파싱이 RAG 파이프라인에 미치는 영향과 실전 해결책을 다룬 블로그

---
### 🔗 Related Sessions
- [document](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [page_content](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [metadata](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
- [lazy_load](../03-문서-로딩과-파싱-다양한-소스에서-데이터-수집/01-문서-로딩-기초-langchain-document-loaders.md) (prerequisite)
