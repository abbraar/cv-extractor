"""
Post-process skills_ar: map tools/tech to high-level Arabic capability labels,
drop raw technology keywords, dedupe, cap at 8 items.
"""

from __future__ import annotations

import re
from typing import Final

# Longest phrases first for substring matching (casefold keys).
_TECH_PHRASES: list[tuple[str, str]] = [
    ("gitlab ci/cd", "الأتمتة"),
    ("semantic kernel", "حلول الذكاء الاصطناعي"),
    ("pydantic.ai", "حلول الذكاء الاصطناعي"),
    ("pydantic ai", "حلول الذكاء الاصطناعي"),
    ("machine learning", "الذكاء الاصطناعي"),
    ("deep learning", "الذكاء الاصطناعي"),
    ("computer vision", "الرؤية الحاسوبية"),
    ("image processing", "معالجة الصور"),
    ("object detection", "الرؤية الحاسوبية"),
    ("image segmentation", "معالجة الصور"),
    ("vector databases", "هندسة البيانات"),
    ("vector database", "هندسة البيانات"),
    ("data engineering", "هندسة البيانات"),
    ("asp.net", "تطوير البرمجيات"),
    ("asp .net", "تطوير البرمجيات"),
    (".net", "تطوير البرمجيات"),
    ("node.js", "تطوير البرمجيات"),
    ("react.js", "تطوير البرمجيات"),
    ("vue.js", "تطوير البرمجيات"),
    ("ci/cd", "تكامل الأنظمة"),
    ("apache airflow", "الأتمتة"),
    # Arabic tool names → executive categories
    ("بوستجري إس كيو إل", "هندسة البيانات"),
    ("خدمات أمازون السحابية", "الحوسبة السحابية"),
    ("منصة جوجل السحابية", "الحوسبة السحابية"),
    ("سي شارب", "تطوير البرمجيات"),
    ("إطار فاست", "تطوير البرمجيات"),
    ("إطار فلاسك", "تطوير البرمجيات"),
    ("بايثون", "تطوير البرمجيات"),
    ("دوكر", "البنية التحتية التقنية"),
    ("جيتلاب", "الأتمتة"),
    ("جيرا", "إدارة المشاريع"),
    ("جيت", "الأتمتة"),
    ("كوبرنيتيس", "البنية التحتية التقنية"),
    ("لانج تشين", "حلول الذكاء الاصطناعي"),
    # English phrases & brands
    ("postgresql", "هندسة البيانات"),
    ("tensorflow", "الذكاء الاصطناعي"),
    ("kubernetes", "البنية التحتية التقنية"),
    ("langchain", "حلول الذكاء الاصطناعي"),
    ("fastapi", "تطوير البرمجيات"),
    ("opencv", "الرؤية الحاسوبية"),
    ("pytorch", "الذكاء الاصطناعي"),
    ("yolov11", "الرؤية الحاسوبية"),
    ("yolov8", "الرؤية الحاسوبية"),
    ("yolo", "الرؤية الحاسوبية"),
    ("qdrant", "هندسة البيانات"),
    ("ollama", "حلول الذكاء الاصطناعي"),
    ("dynamodb", "هندسة البيانات"),
    ("mongodb", "هندسة البيانات"),
    ("postgres", "هندسة البيانات"),
    ("mysql", "هندسة البيانات"),
    ("django", "تطوير البرمجيات"),
    ("flask", "تطوير البرمجيات"),
    ("terraform", "البنية التحتية التقنية"),
    ("ansible", "الأتمتة"),
    ("jenkins", "الأتمتة"),
    ("airflow", "الأتمتة"),
    ("snowflake", "هندسة البيانات"),
    ("bigquery", "هندسة البيانات"),
    ("redshift", "هندسة البيانات"),
    ("anthos", "الحوسبة السحابية"),
    ("podman", "البنية التحتية التقنية"),
    ("typescript", "تطوير البرمجيات"),
    ("javascript", "تطوير البرمجيات"),
    ("react", "تطوير البرمجيات"),
    ("angular", "تطوير البرمجيات"),
    ("vue", "تطوير البرمجيات"),
    ("dotnet", "تطوير البرمجيات"),
    ("csharp", "تطوير البرمجيات"),
    ("gitlab", "الأتمتة"),
    ("github", "الأتمتة"),
    ("bitbucket", "الأتمتة"),
    ("bash", "الأتمتة"),
    ("powershell", "الأتمتة"),
    ("jira", "إدارة المشاريع"),
    ("confluence", "إدارة المشاريع"),
    ("docker", "البنية التحتية التقنية"),
    ("aws", "الحوسبة السحابية"),
    ("gcp", "الحوسبة السحابية"),
    ("azure", "الحوسبة السحابية"),
    ("ec2", "الحوسبة السحابية"),
    ("ecs", "الحوسبة السحابية"),
    ("eks", "الحوسبة السحابية"),
    ("s3", "الحوسبة السحابية"),
    ("lambda", "الحوسبة السحابية"),
    ("sql", "إدارة البيانات"),
    ("nosql", "هندسة البيانات"),
    ("redis", "هندسة البيانات"),
    ("kafka", "هندسة البيانات"),
    ("spark", "هندسة البيانات"),
    ("pandas", "هندسة البيانات"),
    ("numpy", "الذكاء الاصطناعي"),
    ("scikit-learn", "الذكاء الاصطناعي"),
    ("sklearn", "الذكاء الاصطناعي"),
    ("llm", "الذكاء الاصطناعي التوليدي"),
    ("llms", "الذكاء الاصطناعي التوليدي"),
    ("rag", "حلول الذكاء الاصطناعي"),
    ("python", "تطوير البرمجيات"),
    ("git", "الأتمتة"),
    ("k8s", "البنية التحتية التقنية"),
    ("nodejs", "تطوير البرمجيات"),
    ("node", "تطوير البرمجيات"),
    ("spring boot", "تطوير البرمجيات"),
    ("springboot", "تطوير البرمجيات"),
    ("ruby on rails", "تطوير البرمجيات"),
    ("rails", "تطوير البرمجيات"),
    ("laravel", "تطوير البرمجيات"),
]

# Deduplicate phrase keys keeping first category (longer patterns already prioritized).
_seen_phrase: set[str] = set()
_ORDERED_PHRASES: list[tuple[str, str]] = []
for ph, cat in sorted(_TECH_PHRASES, key=lambda x: len(x[0]), reverse=True):
    if ph in _seen_phrase:
        continue
    _seen_phrase.add(ph)
    _ORDERED_PHRASES.append((ph, cat))

# Whole-token match for short codes (casefold key).
_TECH_TOKENS: dict[str, str] = {
    "c#": "تطوير البرمجيات",
    "c++": "تطوير البرمجيات",
    "go": "تطوير البرمجيات",
    "rust": "تطوير البرمجيات",
    "php": "تطوير البرمجيات",
    "java": "تطوير البرمجيات",
    "kotlin": "تطوير البرمجيات",
    "swift": "تطوير البرمجيات",
    "sql": "إدارة البيانات",
    "nosql": "هندسة البيانات",
    "etl": "هندسة البيانات",
    "elt": "هندسة البيانات",
    "api": "تطوير الحلول التقنية",
    "rest": "تطوير الحلول التقنية",
    "graphql": "تطوير الحلول التقنية",
    "grpc": "تطوير الحلول التقنية",
    "ml": "الذكاء الاصطناعي",
    "ai": "الذكاء الاصطناعي",
    "nlp": "الذكاء الاصطناعي",
    "cv": "الرؤية الحاسوبية",
    "gpu": "الذكاء الاصطناعي",
    "cuda": "الذكاء الاصطناعي",
}

# Merge near-duplicate executive labels (normalized key → preferred Arabic).
_SYNONYM_CANONICAL: dict[str, str] = {
    "الذكاء الاصطناعي التوليدي": "الذكاء الاصطناعي",
    "حلول الذكاء الاصطناعي": "الذكاء الاصطناعي",
    "الذكاء الاصطناعي والتحليلات": "الذكاء الاصطناعي",
}

_MAX_SKILLS: Final[int] = 8

_AR_LETTER_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")


def _is_mostly_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ar = sum(1 for c in letters if _AR_LETTER_RE.match(c))
    return (ar / len(letters)) >= 0.55


def _canonical_executive_label(label: str) -> str:
    s = label.strip()
    if not s:
        return s
    k = re.sub(r"\s+", " ", s).casefold()
    return _SYNONYM_CANONICAL.get(k, s)


def _extract_categories_from_text(s_cf: str) -> list[str]:
    """Return executive categories implied by tech phrases/tokens in s_cf."""
    found: list[str] = []
    remaining = s_cf
    for phrase, cat in _ORDERED_PHRASES:
        if phrase in remaining:
            found.append(cat)
            remaining = remaining.replace(phrase, " ")
    # Token scan for isolated English/short codes
    for tok, cat in _TECH_TOKENS.items():
        pat = rf"(?<![a-z0-9]){re.escape(tok)}(?![a-z0-9])"
        if re.search(pat, remaining):
            found.append(cat)
    # Dedupe categories while preserving order
    seen_cat: set[str] = set()
    out: list[str] = []
    for c in found:
        k = c.casefold()
        if k not in seen_cat:
            seen_cat.add(k)
            out.append(c)
    return out


def _looks_like_raw_tool_only_label(s: str) -> bool:
    """Heuristic: single short Latin token + digits/symbols, no Arabic."""
    t = s.strip()
    if not t or _is_mostly_arabic(t):
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9+#.\-]{0,40}", t, re.I):
        return True
    if re.fullmatch(r"[a-z0-9][a-z0-9+#.\-]*\s*/\s*[a-z0-9][a-z0-9+#.\-]*", t, re.I):
        return True
    return False


def normalize_skills_ar_for_executive_profile(items: list[str]) -> list[str]:
    """
    Map tools/tech to high-level Arabic labels, drop bare technology keywords,
    dedupe, cap at 8. Does not invent skills (no padding to 5).
    """
    out: list[str] = []
    seen: set[str] = set()

    def add_label(lab: str) -> None:
        if len(out) >= _MAX_SKILLS:
            return
        c2 = _canonical_executive_label(lab)
        k = c2.casefold()
        if k in seen:
            return
        seen.add(k)
        out.append(c2)

    for raw in items:
        if len(out) >= _MAX_SKILLS:
            break
        s = re.sub(r"\s+", " ", raw.strip())
        if not s:
            continue
        s_cf = s.casefold()

        cats = _extract_categories_from_text(s_cf)
        if cats:
            for c in cats:
                add_label(c)
                if len(out) >= _MAX_SKILLS:
                    break
            continue

        if _looks_like_raw_tool_only_label(s):
            continue
        if _is_mostly_arabic(s):
            add_label(s)

    return out[:_MAX_SKILLS]
