"""Validation and normalization for Gemini CV extraction output."""

from __future__ import annotations

import json
import re
from typing import Any

from skills_executive import normalize_skills_ar_for_executive_profile

MISSING_AR = "غير مذكور"

# Canonical Arabic / brand spellings — does not add facts, only unifies wording.
TERM_REPLACEMENTS: dict[str, str] = {
    "Computer Vision": "الرؤية الحاسوبية",
    "Image Processing": "معالجة الصور",
    "Object Detection": "اكتشاف الكائنات",
    "Image Segmentation": "تجزئة الصور",
    "Vector Databases": "قواعد البيانات المتجهية",
    "FastAPI": "FastAPI",
    "Flask": "Flask",
    "LangChain": "LangChain",
    "PyTorch": "PyTorch",
    "TensorFlow": "TensorFlow",
    "OpenCV": "OpenCV",
    "Kubernetes": "Kubernetes",
    "Docker": "دوكر",
    "Git": "جيت",
    "Jira": "جيرا",
    "Python": "بايثون",
    "AWS": "خدمات أمازون السحابية",
    "GCP": "منصة جوجل السحابية",
}


def _normalize_whitespace(value: str) -> str:
    if value == MISSING_AR:
        return value
    s = value.strip()
    if not s:
        return ""
    return re.sub(r"\s+", " ", s)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = phrase.split()
    if len(parts) == 1:
        tok = re.escape(parts[0])
        return re.compile(
            rf"(?<![A-Za-z0-9]){tok}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    inner = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(
        rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def _apply_term_normalization(text: str) -> str:
    """Apply longest-first replacements; ASCII tokens use non-alphanumeric boundaries."""
    if not text or text == MISSING_AR:
        return text
    out = text
    for source, target in sorted(
        TERM_REPLACEMENTS.items(),
        key=lambda kv: len(kv[0]),
        reverse=True,
    ):
        pat = _phrase_pattern(source)
        out = pat.sub(target, out)
    return _normalize_whitespace(out)


def _coerce_string(value: Any) -> str:
    if value is None:
        return MISSING_AR
    if isinstance(value, str):
        s = value.strip()
        return s if s else MISSING_AR
    return str(value).strip() or MISSING_AR


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    if not isinstance(value, list):
        return [_coerce_string(value)]
    out: list[str] = []
    for item in value:
        out.append(_coerce_string(item))
    return out


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse JSON from model output; strip markdown fences if present."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    return json.loads(text)


def validate_and_normalize(data: dict[str, Any]) -> dict[str, Any]:
    """
    Allowed keys only; null/empty strings → غير مذكور; whitespace cleanup;
    term normalization on summary/experience; executive-style skills post-process.
    """
    name = _normalize_whitespace(_coerce_string(data.get("name"))) or MISSING_AR
    job_title = _normalize_whitespace(_coerce_string(data.get("job_title"))) or MISSING_AR

    summary_raw = _coerce_string(data.get("summary_ar"))
    if summary_raw == MISSING_AR:
        summary_ar = MISSING_AR
    else:
        summary_ar = _apply_term_normalization(_normalize_whitespace(summary_raw)) or MISSING_AR

    experience_ar: list[str] = []
    for line in _coerce_string_list(data.get("experience_ar")):
        if line == MISSING_AR:
            continue
        norm = _apply_term_normalization(_normalize_whitespace(line))
        if norm and norm != MISSING_AR:
            experience_ar.append(norm)

    skills_raw: list[str] = []
    for s in _coerce_string_list(data.get("skills_ar")):
        if s == MISSING_AR:
            continue
        w = _normalize_whitespace(s)
        if w and w != MISSING_AR:
            skills_raw.append(w)
    skills_ar = normalize_skills_ar_for_executive_profile(skills_raw)

    return {
        "name": name if name else MISSING_AR,
        "job_title": job_title if job_title else MISSING_AR,
        "summary_ar": summary_ar,
        "experience_ar": experience_ar,
        "skills_ar": skills_ar,
    }
