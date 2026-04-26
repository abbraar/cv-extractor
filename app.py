"""Streamlit UI: CV → Arabic corporate profile fields via OpenAI."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from extractor import (
    extract_text_from_bytes,
    get_last_pdf_vision_diagnostic,
    pdf_ocr_dependencies_ok,
    pdf_openai_vision_ready,
)
from openai_service import extract_profile

# Streamlit’s process cwd may not be this folder; load .env next to app.py.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _inject_streamlit_secrets_into_environ() -> None:
    """Streamlit Community Cloud: keys live in App settings → Secrets (TOML), not in the repo."""
    try:
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:

        class StreamlitSecretNotFoundError(Exception):
            pass

    keys = (
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_VISION_MODEL",
        "OPENAI_VISION_MAX_PAGES",
        "OPENAI_VISION_DPI",
        "TESSERACT_CMD",
    )
    # Local dev: no secrets.toml → `key in st.secrets` parses files and raises.
    try:
        secrets = st.secrets
        for key in keys:
            if key not in secrets:
                continue
            val = secrets[key]
            if val is None:
                continue
            s = str(val).strip()
            if not s:
                continue
            if not os.environ.get(key):
                os.environ[key] = s
    except StreamlitSecretNotFoundError:
        return
    except FileNotFoundError:
        return


st.set_page_config(page_title="CV Extractor (Arabic Profile)", layout="wide")
_inject_streamlit_secrets_into_environ()
st.title("CV Extractor (Arabic Profile)")

uploaded = st.file_uploader(
    "Upload a CV",
    type=["pdf", "docx", "txt"],
    help="PDF: text layer, then Tesseract (optional), then OpenAI vision if OPENAI_API_KEY is set.",
)

if st.button("Extract", type="primary"):
    if not uploaded:
        st.warning("Please upload a file first.")
    else:
        try:
            raw = uploaded.getvalue()
            name_lower = uploaded.name.lower()
            if name_lower.endswith(".pdf"):
                with st.spinner(
                    "Reading PDF (text layer → Tesseract if installed → OpenAI vision for scans)…"
                ):
                    text = extract_text_from_bytes(raw, uploaded.name)
            else:
                text = extract_text_from_bytes(raw, uploaded.name)
            if not text:
                pdf_hint = ""
                if name_lower.endswith(".pdf"):
                    o_ready = pdf_openai_vision_ready()
                    tess_ok = pdf_ocr_dependencies_ok()
                    if o_ready:
                        pdf_hint = (
                            "**Scanned PDF:** OpenAI vision did not return usable text. "
                            "Check `OPENAI_API_KEY`, model name (`OPENAI_MODEL` / `OPENAI_VISION_MODEL`), "
                            "billing/limits at https://platform.openai.com — or open **Technical details** below.\n\n"
                        )
                    elif tess_ok:
                        pdf_hint = (
                            "**Scanned PDF:** Tesseract ran but returned no text. "
                            "Install **Arabic** language data (`ara`) for Arabic CVs, or set **`OPENAI_API_KEY`** "
                            "and `pip install pymupdf` to use **OpenAI vision**.\n\n"
                        )
                    else:
                        pdf_hint = (
                            "**Scanned PDF (image-only):** Do one of the following:\n"
                            "- **Recommended:** `pip install pymupdf`, set **`OPENAI_API_KEY`** in `.env` — "
                            "the app will read scans with **OpenAI vision**.\n"
                            "- **Or** install Tesseract (https://github.com/tesseract-ocr/tesseract), "
                            "`pip install pytesseract Pillow`; on Windows common install paths are auto-tried "
                            "or set `TESSERACT_CMD`.\n\n"
                        )
                st.error(
                    "Could not read any text from this file.\n\n"
                    + pdf_hint
                    + "**Other causes:**\n"
                    "- **PDF:** Corrupt, password-protected, or blank pages.\n"
                    "- **DOCX / TXT:** Empty file or wrong type (e.g. `.doc` renamed to `.docx`)."
                )
                diag = get_last_pdf_vision_diagnostic()
                if name_lower.endswith(".pdf") and diag:
                    with st.expander("Technical details (OpenAI vision / PDF)", expanded=True):
                        st.code(diag, language=None)
            else:
                with st.spinner("Extracting with OpenAI…"):
                    data = extract_profile(text)

                st.subheader("الاسم")
                st.write(data["name"])

                st.subheader("المسمى الوظيفي")
                st.write(data["job_title"])

                st.subheader("نبذة عامة")
                st.write(data["summary_ar"])

                st.subheader("الخبرات المهنية")
                exp = data["experience_ar"]
                if exp:
                    for line in exp:
                        st.markdown(f"- {line}")
                else:
                    st.write("غير مذكور")

                st.subheader("مهارات")
                skills = data["skills_ar"]
                if skills:
                    for s in skills:
                        st.markdown(f"- {s}")
                else:
                    st.write("غير مذكور")

        except ValueError as e:
            st.error(str(e))
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            if os.environ.get("DEBUG"):
                raise

st.caption(
    "Made by Abrar"
)
