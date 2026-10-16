import os
import re
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


def extract_sections_from_pdf(reader: PdfReader, file_name: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF while tracking structural sections
    and course hierarchy across pages (e.g. Foundation, Diploma, Degree levels,
    and individual courses like DBMS, Python, Java, Machine Learning, etc.).
    
    This ensures continuation pages (e.g., page 32 containing a grading formula)
    retain the context of which course they belong to (e.g., DBMS).
    """
    pages_data = []

    current_level = "General Guidelines & Schedules"
    current_course = "General Overview"

    level_pattern = re.compile(
        r"(?:^|\n)\s*(Foundation\s+level\s+courses|Diploma\s+level\s+courses|Degree\s+level\s+courses)",
        re.IGNORECASE,
    )

    # Identifies course headings across pages
    course_pattern = re.compile(
        r"(?:^|\n)\s*(\d{1,2}\.\s+(?:Mathematics|English|Computational|Statistics|"
        r"Machine Learning|Business|Tools|Database|Application|Programming|"
        r"System commands|Software|Deep Learning|AI|Strategies|Industry|Speech|"
        r"Design|Financial|Introduction to Big Data|Linear Models|Special topics|"
        r"Corporate Finance|Computer|Game Theory|Algorithms|Compiler|Sequential|"
        r"Operating Systems|Discrete)[^\n]{0,90})",
        re.IGNORECASE,
    )

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text(extraction_mode="layout") or ""
        if len(raw_text.strip()) < 20:
            raw_text = page.extract_text() or ""

        # Normalize line spacing while preserving tabular line breaks
        lines = [" ".join(l.split()) for l in raw_text.split("\n")]
        cleaned_text = "\n".join([l for l in lines if l])

        # Track program level
        lvl_match = level_pattern.search(cleaned_text)
        if lvl_match:
            current_level = " ".join(lvl_match.group(1).split())

        # Track course sections (past table of contents on page 9+)
        if page_idx >= 9:
            matches = list(course_pattern.finditer(cleaned_text))
            if matches:
                c_cand = " ".join(matches[-1].group(1).split())
                if len(c_cand) > 5:
                    current_course = c_cand
        elif page_idx in [1, 2, 3, 4, 5, 6]:
            if "OPPE SCHEDULE" in cleaned_text.upper():
                current_course = "OPPE Exam Schedule"
            elif "QUIZ" in cleaned_text.upper() and "SCHEDULE" in cleaned_text.upper():
                current_course = "Quiz Schedule"
            elif "ASSIGNMENT DEADLINES" in cleaned_text.upper():
                current_course = "Assignment Deadlines & Bonus Policy"
        elif page_idx in [7, 8]:
            current_course = "Table of Contents & Suggested Pathways"

        if cleaned_text.strip():
            pages_data.append({
                "text": cleaned_text,
                "metadata": {
                    "source": file_name,
                    "page": page_idx,
                    "file_type": "pdf",
                    "level": current_level,
                    "section": current_course,
                },
            })

    return pages_data


def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
    """Loads PDF with section-aware extraction."""
    reader = PdfReader(str(file_path))
    return extract_sections_from_pdf(reader, file_path.name)


def load_text_or_md(file_path: Path) -> List[Dict[str, Any]]:
    """Loads plain text (.txt) or Markdown (.md) files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()

    if not content:
        return []

    return [{
        "text": content,
        "metadata": {
            "source": file_path.name,
            "page": 1,
            "file_type": file_path.suffix.lstrip("."),
            "level": "Document",
            "section": file_path.stem.replace("_", " ").title(),
        },
    }]


def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Scans a directory and loads all supported (.pdf, .md, .txt) documents."""
    root = Path(folder_path)
    if not root.exists():
        raise FileNotFoundError(f"Documents folder not found: {folder_path}")

    supported_extensions = {".pdf", ".md", ".txt"}
    all_loaded_segments = []

    for file_path in sorted(root.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            print(f"📄 Loading: {file_path.name}...")
            ext = file_path.suffix.lower()

            if ext == ".pdf":
                segments = load_pdf(file_path)
                print(f"   └── Extracted {len(segments)} pages with section tracking.")
            else:
                segments = load_text_or_md(file_path)
                print(f"   └── Loaded {len(segments)} text document.")

            all_loaded_segments.extend(segments)

    return all_loaded_segments
