"""OpenAI API: strict CV extraction to Arabic profile JSON + vision OCR for scanned PDFs."""

from __future__ import annotations

import base64
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIError, APITimeoutError, OpenAI, RateLimitError

load_dotenv(Path(__file__).resolve().parent / ".env")

from utils import MISSING_AR, parse_json_response, validate_and_normalize

CV_EXTRACTION_PROMPT = """You are a **strict CV information extraction and Arabic normalization engine**.

Your task is to extract ONLY the following fields from the CV text:

* name
* job_title
* summary_ar
* experience_ar
* skills_ar

Return the result in **Arabic only**.

## CRITICAL RULES

1. Extract information ONLY from the CV text.
2. Do NOT hallucinate.
3. Do NOT infer missing facts.
4. Do NOT add achievements, adjectives, or interpretations.
5. Do NOT paraphrase the meaning of the content — **except for `skills_ar` only**, where you must **generalize** tools and platforms into **high-level professional capability labels** (see section E). That exception does not apply to name, job_title, summary_ar, or experience_ar.
6. If a field is missing, return "غير مذكور".
7. Output must be valid JSON only.
8. Do not include markdown fences.
9. Do not return any explanation outside the JSON.

## ARABIC OUTPUT NORMALIZATION RULES

The final output must be written in **clear, professional Arabic**.

### Important:

* If the CV is in English, translate the extracted content into Arabic faithfully.
* Do NOT leave full English phrases in the final output unless they are proper nouns, company names, product names, framework names, model names, or official technical terms that should remain as-is — **except `skills_ar`**, which must never be a list of English tool names (see section E).
* Do NOT produce awkward Arabic transliterations letter-by-letter.
* Prefer **natural Arabic technical naming** where appropriate (outside of `skills_ar` executive labels).

### Apply these formatting rules:

#### A) Names

* Keep the person's name as originally written if it is a foreign proper name.
* Do not invent an Arabic version of the person's name unless the CV already contains it.

#### B) Job Titles

Translate job titles into professional Arabic, for example:

* Software Engineer → مهندس برمجيات
* Senior Software Engineer → مهندس برمجيات أول
* AI Engineer → مهندس ذكاء اصطناعي
* Machine Learning Engineer → مهندس تعلم آلة
* Backend Engineer → مهندس برمجيات خلفية
* Data Engineer → مهندس بيانات

Use the closest accurate Arabic title without adding seniority not mentioned in the CV.

#### C) Summary

* Write the summary in Arabic using ONLY explicit facts from the CV.
* Do not embellish.
* Do not add praise words.
* Do not add inferred years of experience.
* Preserve factual meaning exactly.

#### D) Experience

* Translate each experience item into Arabic faithfully.
* Keep company names as originally written.
* Keep dates as they are, but format surrounding text in Arabic.
* Keep technical product/framework/model names in their standard form if translating them would be unnatural.
* Do NOT leave full explanatory English sentences inside the Arabic output.

#### E) Skills (`skills_ar`) — **executive / corporate profile style only**

`skills_ar` must **NOT** be a raw list of tools, frameworks, libraries, cloud services, APIs, databases, or product names.

**Do NOT return** entries like: بايثون، سي شارب، PostgreSQL، AWS، GCP، Django، ASP.NET، FastAPI، Flask، Git، Docker، Jira، Qdrant، LangChain، PyTorch، TensorFlow، OpenCV، YOLOv11، Kubernetes، SQL، Bash، GitLab CI/CD، or any similar technology keyword.

**Instead**, infer **only high-level professional Arabic capability labels** suitable for **executive profile slides** (short noun phrases, business-friendly wording).

Sources you may use (only facts present in the CV):

* Dedicated skills section
* Work experience bullets
* Project descriptions

**Rules for `skills_ar`:**

1. Do **not** output individual tools, products, frameworks, or cloud brands as standalone skills.
2. **Generalize** the technical stack into **concise Arabic labels** (e.g. تطوير البرمجيات، الحوسبة السحابية، هندسة البيانات).
3. Output **between 5 and 8** items inclusive. Each item: **one short Arabic label**, not a sentence.
4. **No duplicates** (no synonymous repeats).
5. **Do not hallucinate** categories: every label must be **clearly supported** by what the candidate actually did or listed.
6. Prefer wording similar in tone to: إدارة المنتجات، الحوكمة، إدارة البيانات، تطوير البرمجيات، الذكاء الاصطناعي والتحليلات، تصور البيانات ولوحات المعلومات، إدارة المشاريع، الأتمتة، البنية المؤسسية، تجربة المستخدم، تحليل الأعمال، التحول الرقمي، هندسة البيانات، الرؤية الحاسوبية، معالجة الصور، تكامل الأنظمة، تطوير الحلول التقنية — **only when** the CV supports them.

**Mapping guidance (examples — apply only when the CV supports the underlying work):**

* Python / C# / Django / FastAPI / Flask / ASP.NET → **تطوير البرمجيات**
* PostgreSQL / SQL / DynamoDB / S3 / Qdrant / vector databases → **إدارة البيانات** or **هندسة البيانات**
* AWS / GCP / Kubernetes / Docker / Podman / EC2 / ECS / EKS / Anthos → **الحوسبة السحابية** or **البنية التحتية التقنية**
* PyTorch / TensorFlow / LangChain / Pydantic.AI / Semantic Kernel / LLMs / RAG → **الذكاء الاصطناعي** or **حلول الذكاء الاصطناعي** or **الذكاء الاصطناعي التوليدي**
* OpenCV / YOLO / SAM / image processing / object detection / segmentation → **الرؤية الحاسوبية** or **معالجة الصور**
* Git / GitLab CI/CD / Bash / Airflow / automation tooling → **الأتمتة** or **تكامل الأنظمة** or **تطوير الحلول التقنية**
* Jira / delivery coordination → **إدارة المشاريع**

**Bad `skills_ar` example:**

["بايثون", "FastAPI", "Docker", "Kubernetes", "PyTorch", "OpenCV"]

**Good `skills_ar` example:**

["تطوير البرمجيات", "الحوسبة السحابية", "هندسة البيانات", "الذكاء الاصطناعي", "الرؤية الحاسوبية", "إدارة المشاريع"]

### KEY NORMALIZATION GOAL

* **experience_ar** and **summary_ar**: accurate Arabic, facts only; technical names may appear where needed for clarity.
* **skills_ar** only: **executive-style** high-level Arabic capabilities — **never** a technology keyword list.

## OUTPUT JSON SCHEMA

Return ONLY this JSON:

{
"name": "",
"job_title": "",
"summary_ar": "",
"experience_ar": [],
"skills_ar": []
}

## EXTRA QUALITY RULES

* **skills_ar**: only **5–8** short **Arabic** capability labels; **no** tools, frameworks, clouds, or product names; executive/corporate tone; no duplicates.
* experience_ar must be a clean array of bullet-style Arabic strings (factual; tools may appear in context).
* summary_ar must be one concise Arabic paragraph (explicit facts only).
* Preserve company names exactly as written where they appear outside skills_ar.
"""

VISION_OCR_BATCH_PROMPT = """These images are consecutive pages of one CV/resume (in order: first image = page 1, next = page 2, etc.).
Transcribe ALL visible text exactly as printed (Arabic, English, or mixed). Preserve line breaks.
Include names, dates, job titles, companies, skills, and bullet points.
Between each page's text, output a single line containing exactly: ---PAGE---
Do not summarize or describe the layout. Plain text only — no markdown, no JSON."""

_DEFAULT_MODEL = "gpt-5.2"
_VISION_MAX_IMAGES_PER_REQUEST = 6
_MAX_AUTO_RETRY_WAIT_S = 90.0


def _openai_json_schema_strict() -> dict[str, Any]:
    """JSON Schema for OpenAI structured outputs (strict mode)."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "job_title": {"type": "string"},
            "summary_ar": {"type": "string"},
            "experience_ar": {"type": "array", "items": {"type": "string"}},
            "skills_ar": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string"},
            },
        },
        "required": [
            "name",
            "job_title",
            "summary_ar",
            "experience_ar",
            "skills_ar",
        ],
    }


def _format_openai_error(exc: BaseException) -> RuntimeError:
    msg = str(exc)
    if "429" in msg or isinstance(exc, RateLimitError):
        return RuntimeError(
            "OpenAI rate limit (429): too many requests or quota exceeded.\n\n"
            "• Wait a minute and try again, or check usage at https://platform.openai.com/usage\n"
            "• Try a different `OPENAI_MODEL` if your plan limits certain models.\n"
            f"Details: {exc}"
        )
    return RuntimeError(f"OpenAI API error: {exc}")


def _retry_after_seconds_openai(exc: BaseException) -> float | None:
    blob = str(exc).lower()
    m = re.search(r"try again in ([\d.]+)\s*s", blob)
    if m:
        return min(float(m.group(1)), _MAX_AUTO_RETRY_WAIT_S)
    m2 = re.search(r"retry after (\d+)", blob)
    if m2:
        return min(float(m2.group(1)), _MAX_AUTO_RETRY_WAIT_S)
    return None


def _should_retry_openai(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError)):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503):
            return True
    return False


def call_openai_with_retry(
    generate_fn: Callable[[], Any],
    max_retries: int = 5,
    *,
    retry_on_429: bool = True,
) -> Any:
    attempts = max(1, max_retries)
    delay = 2.0
    for attempt in range(attempts):
        try:
            return generate_fn()
        except Exception as e:
            if retry_on_429 and _should_retry_openai(e) and attempt < attempts - 1:
                wait = _retry_after_seconds_openai(e)
                if wait is None or wait <= 0:
                    wait = delay + random.uniform(0, 2)
                wait = min(wait, 120.0)
                print(f"OpenAI transient error, retrying in {wait:.1f}s... ({e!s})")
                time.sleep(wait)
                delay = min(delay * 2, 45.0)
                continue
            raise


def _chat_completion_json(
    client: OpenAI,
    model: str,
    user_text: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": user_text,
        }
    ]

    def _try_structured(use_strict_schema: bool) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_completion_tokens": 8192,
        }
        if use_strict_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "cv_arabic_profile",
                    "strict": True,
                    "schema": _openai_json_schema_strict(),
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    def _call() -> Any:
        try:
            return _try_structured(True)
        except (RateLimitError, APITimeoutError):
            raise
        except APIError as e:
            if getattr(e, "status_code", None) == 429:
                raise
            err = str(e).lower()
            if "json_schema" in err or "response_format" in err or "unsupported" in err:
                return _try_structured(False)
            raise
        except Exception as e:
            err = str(e).lower()
            if "json_schema" in err or "response_format" in err or "unsupported" in err:
                return _try_structured(False)
            raise

    completion = call_openai_with_retry(_call, max_retries=5, retry_on_429=True)
    choice = completion.choices[0]
    content = (choice.message.content or "").strip()
    if not content:
        raise RuntimeError("Empty response from OpenAI.")
    return content


def extract_profile(cv_text: str) -> dict[str, Any]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    model = (os.environ.get("OPENAI_MODEL") or _DEFAULT_MODEL).strip()
    client = OpenAI(api_key=api_key)
    user_content = f"{CV_EXTRACTION_PROMPT}\n\n---\n\nCV TEXT:\n{cv_text}"

    try:
        raw = _chat_completion_json(client, model, user_content)
    except Exception as e:
        raise _format_openai_error(e) from e

    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        raise ValueError("OpenAI returned JSON that is not an object.")

    return validate_and_normalize(parsed)


def extract_text_from_scanned_pdf_via_openai(
    data: bytes,
    *,
    diagnostic_lines: list[str] | None = None,
) -> str:
    def log(msg: str) -> None:
        if diagnostic_lines is not None:
            diagnostic_lines.append(msg)
        print(msg)

    load_dotenv(Path(__file__).resolve().parent / ".env")

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        log("OpenAI vision: OPENAI_API_KEY is empty.")
        return ""

    try:
        import fitz  # PyMuPDF
    except ImportError:
        log("OpenAI vision: install pymupdf (pip install pymupdf).")
        return ""

    model = (
        (os.environ.get("OPENAI_VISION_MODEL") or "").strip()
        or (os.environ.get("OPENAI_MODEL") or "").strip()
        or _DEFAULT_MODEL
    )

    try:
        dpi = max(144, min(int(os.environ.get("OPENAI_VISION_DPI", "200")), 300))
    except ValueError:
        dpi = 200

    doc = fitz.open(stream=data, filetype="pdf")
    all_pngs: list[bytes] = []
    try:
        total = len(doc)
        cap_raw = (os.environ.get("OPENAI_VISION_MAX_PAGES") or "").strip()
        if cap_raw:
            try:
                cap = int(cap_raw)
                n_pages = total if cap <= 0 else min(total, cap)
            except ValueError:
                n_pages = total
        else:
            n_pages = total

        for i in range(n_pages):
            pix = doc[i].get_pixmap(dpi=dpi, alpha=False)
            all_pngs.append(pix.tobytes("png"))
    finally:
        doc.close()

    if not all_pngs:
        log("OpenAI vision: PDF has no pages to render.")
        return ""

    log(
        f"OpenAI vision: {len(all_pngs)} page image(s), dpi={dpi}, "
        f"up to {_VISION_MAX_IMAGES_PER_REQUEST} images per request, model={model}"
    )

    client = OpenAI(api_key=api_key)
    combined: list[str] = []

    for batch_start in range(0, len(all_pngs), _VISION_MAX_IMAGES_PER_REQUEST):
        batch_pngs = all_pngs[batch_start : batch_start + _VISION_MAX_IMAGES_PER_REQUEST]
        batch_idx = batch_start // _VISION_MAX_IMAGES_PER_REQUEST + 1
        page_range = f"{batch_start + 1}-{batch_start + len(batch_pngs)}"

        content: list[dict[str, Any]] = [
            {"type": "text", "text": VISION_OCR_BATCH_PROMPT},
        ]
        for png in batch_pngs:
            b64 = base64.standard_b64encode(png).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

        def _call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=8192,
            )

        try:
            comp = call_openai_with_retry(_call, max_retries=5, retry_on_429=True)
            text = (comp.choices[0].message.content or "").strip()
            if text:
                combined.append(text)
                log(f"Batch {batch_idx} (pages {page_range}): OK, chars={len(text)}")
            else:
                log(f"Batch {batch_idx}: empty response from OpenAI.")
        except Exception as e:
            log(f"Batch {batch_idx} (pages {page_range}): error: {type(e).__name__}: {e}")

    return "\n\n".join(combined).strip()


def empty_profile() -> dict[str, Any]:
    return {
        "name": MISSING_AR,
        "job_title": MISSING_AR,
        "summary_ar": MISSING_AR,
        "experience_ar": [],
        "skills_ar": [],
    }
