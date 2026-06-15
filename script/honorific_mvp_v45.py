"""
Honorific Suspicious-Case Generator MVP (v4.5)

This script implements a practical HITL-oriented pipeline for Korean honorific review.
It is designed as a sentence-level suspicious-case generator, not as a final grading engine.

Version 4.5 focus:
- Keep the detection logic practical and review-oriented
- Reduce obvious over-triggering for style-shift cases
- Separate strong issues from low-confidence review cases
- Preserve raw diagnostic output while supporting professor-facing review output
- Prefer human-readable explanations over complex document-level scoring

Core philosophy:
- Prefer practical review value over theoretical completeness
- Prefer conservative fallback over overconfident mislabeling
- Prefer sentence-level explainability over complex automatic judgment
- Prefer readable code and explicit comments over clever shortcuts
"""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from kiwipiepy import Kiwi


# ============================================================
# 1. Core Labels and Issue Schema
# ============================================================

# Subject group labels
SUBJECT_SELF = "SELF"
SUBJECT_ADDRESSEE = "ADDRESSEE"
SUBJECT_THIRD_OR_NOUN = "THIRD_OR_NOUN"
SUBJECT_OMITTED = "OMITTED"
SUBJECT_UNKNOWN = "UNKNOWN"

# Style labels
STYLE_POLITE = "POLITE"
STYLE_PLAIN = "PLAIN"
STYLE_CASUAL = "CASUAL"
STYLE_ARCHAIC = "ARCHAIC"
STYLE_UNKNOWN = "UNKNOWN"

# Issue codes
ISSUE_E01 = "E01_SELF_HONORIFIC_SUBJECT"
ISSUE_E02 = "E02_DOC_STYLE_SHIFT"
ISSUE_E03 = "E03_ARCHAIC_OR_ODD_STYLE"
ISSUE_E04 = "E04_SHORT_OR_BROKEN_OUTPUT"
ISSUE_E05 = "E05_UNKNOWN_ENDING"
ISSUE_E06 = "E06_QUOTE_OR_COMPLEX_CLAUSE"

# Severity labels
SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"

ISSUE_SEVERITY_MAP: Dict[str, str] = {
    ISSUE_E01: SEVERITY_ERROR,
    ISSUE_E02: SEVERITY_WARNING,
    ISSUE_E03: SEVERITY_WARNING,
    ISSUE_E04: SEVERITY_ERROR,
    ISSUE_E05: SEVERITY_WARNING,
    ISSUE_E06: SEVERITY_WARNING,
}

ISSUE_BRIEF_REASON_MAP: Dict[str, str] = {
    ISSUE_E01: "1st-person subject with honorific predicate",
    ISSUE_E02: "Sentence style differs from document-dominant style",
    ISSUE_E03: "Archaic or unusual sentence style",
    ISSUE_E04: "Broken or abnormally short output",
    ISSUE_E05: "Sentence-final style could not be classified",
    ISSUE_E06: "Quoted or complex clause; low-confidence auto judgment",
}


# ============================================================
# 2. Configuration Dataclass
# ============================================================

def get_issue_severity(issue_code: str) -> str:
    """
    Return the fixed severity label for a given issue code.
    """
    return ISSUE_SEVERITY_MAP.get(issue_code, SEVERITY_WARNING)


def get_issue_brief_reason(issue_code: str) -> str:
    """
    Return a short review-friendly explanation for a given issue code.
    """
    return ISSUE_BRIEF_REASON_MAP.get(issue_code, "Review required")

def get_review_priority(row: IssueRecord) -> str:
    """
    Assign human-review priority for review CSV.

    FOCUS:
    - Strong E02 style-shift candidates in news where
      dominant document style is POLITE but the sentence is CASUAL

    NORMAL:
    - All other cases
    """
    if (
        row.issue_code == ISSUE_E02
        and row.domain == "news"
        and row.dominant_doc_style == STYLE_POLITE
        and row.style_label == STYLE_CASUAL
    ):
        return "FOCUS"

    return "NORMAL"

@dataclass
class PipelineConfig:
    """
    Central configuration object for lightweight heuristic rules.

    Why a dataclass is useful here:
    - It gathers rule lists in one reviewable place.
    - It prevents long hardcoded branches from spreading everywhere.
    - It makes later tuning easier after sample inspection.
    """

    # Common first-person subject candidates.
    self_subject_forms: Sequence[str] = field(default_factory=lambda: [
        "나", "내", "내가", "나는",
        "저", "제", "제가", "저는",
        "우리", "우리가", "우리는",
        "저희", "저희가", "저희는",
    ])

    # Common addressee-like pronouns.
    addressee_forms: Sequence[str] = field(default_factory=lambda: [
        "너", "네", "니", "당신", "자네", "그대"
    ])

    # Polite endings used in modern conversational or formal Korean.
    polite_endings: Sequence[str] = field(default_factory=lambda: [
        "습니다", "ㅂ니다", "니다",
        "습니까", "ㅂ니까",
        "세요", "셔요",
        "어요", "아요", "여요",
        "예요", "에요",
        "까요", "네요",
        "입니다", "됩니다",
        "죠", "지요",
        "요"
    ])

    # Plain narrative / descriptive endings.
    plain_endings: Sequence[str] = field(default_factory=lambda: [
        "다", "는다", "ㄴ다",
        "었다", "았다", "ㅆ다",
        "했다", "였다"
    ])

    # Casual conversational endings.
    #
    # Important note:
    # We intentionally move '-네' here for MVP purposes.
    # In traditional grammar discussions, '-네' may be analyzed differently in some contexts.
    # However, in practical modern conversational data, treating it as ARCHAIC creates too many false positives.
    casual_endings: Sequence[str] = field(default_factory=lambda: [
        "아", "어", "지", "니", "냐",
        "었어", "았어",
        "겠지", "거야", "잖아",
        "줘", "래",
        "ㄹ게", "을게",
        "야", "이야",
        "대", "더라",
        "데", "ㄴ데", "는데",
        "네"
    ])

    # Archaic endings should remain intentionally narrow.
    # We exclude '-네' here because it causes too many false positives in modern data.
    archaic_final_ef_forms: Sequence[str] = field(default_factory=lambda: [
        "게", "세", "오", "소", "는가"
    ])

    # Quote-like/reporting patterns.
    #
    # These patterns must never force E05 by themselves.
    # They are only used as fallback explanations when style classification or predicate extraction already failed.
    quote_like_patterns: Sequence[str] = field(default_factory=lambda: [
        '"', "'", "“", "”", "‘", "’",
        "고 말했다", "라고 말했다",
        "고 했다", "라고 했다",
        "고 물었다", "라고 물었다",
        "고 생각했다", "라고 생각했다",
        "고 다짐했다", "라고 다짐했다"
    ])

    # Broken-output heuristics.
    broken_text_min_chars: int = 20
    broken_text_min_sentences: int = 1
    broken_special_char_ratio: float = 0.35

    # If too few valid styles are found, dominant style remains unknown.
    dominant_style_min_valid_sentences: int = 2

    # Fragment heuristics.
    # These are intentionally rough because the goal is review support, not strict parsing.
    fragment_min_token_count: int = 2
    fragment_max_token_count_for_short_np: int = 8


# ============================================================
# 3. Record Dataclasses
# ============================================================

@dataclass
class DocumentRecord:
    """
    One document-model pair.
    """
    doc_id: str
    model: str
    text: str
    domain: str = ""


@dataclass
class SentenceAnalysis:
    """
    Intermediate sentence-level analysis result.

    This object is produced before issue tagging.
    The separation is useful because:
    - NLP analysis can be debugged independently from issue logic
    - the same sentence analysis can later support multiple issue rules
    """
    doc_id: str
    model: str
    sent_id: int
    sentence: str
    subject_text: str
    subject_group: str
    predicate_text: str
    ending_text: str
    final_ef_text: str
    final_ec_text: str
    style_label: str
    has_si: bool
    fragment_like: bool
    complex_or_unknown: bool
    complex_reason: str
    char_span_start: int
    char_span_end: int


@dataclass
class IssueRecord:
    """
    Final CSV-exportable suspicious case record.
    """
    doc_id: str
    model: str
    domain: str
    sent_id: int
    sentence: str
    subject_text: str
    subject_group: str
    predicate_text: str
    ending_text: str
    style_label: str
    dominant_doc_style: str
    issue_code: str
    severity: str
    reason: str
    char_span_start: int
    char_span_end: int

RAW_CSV_COLUMNS: List[str] = list(IssueRecord.__dataclass_fields__.keys())

REVIEW_CSV_COLUMNS: List[str] = [
    "doc_id",
    "model",
    "domain",
    "sent_id",
    "sentence",
    "issue_code",
    "severity",
    "review_priority",
    "brief_reason",
    "subject_text",
    "predicate_text",
    "ending_text",
    "style_label",
    "dominant_doc_style",
]

@dataclass
class ReferenceStyleProfile:
    """
    Document-level style profile estimated from the human reference text.
    """
    doc_id: str
    domain: str
    dominant_style: str
    valid_sentence_count: int
    style_counts: Dict[str, int]
    style_ratios: Dict[str, float]
    allowed_styles: List[str]

# ============================================================
# 4. Configuration Builder
# ============================================================

def build_default_config() -> PipelineConfig:
    """
    Build the default MVP configuration.

    Keeping this as a dedicated function makes code review simpler.
    """
    return PipelineConfig()


# ============================================================
# 5. Loading and Flattening
# ============================================================

def load_tgt_doc_json(path: str) -> Dict[str, Dict[str, str]]:
    """
    Load the raw target-document JSON.

    This function intentionally does one thing only:
    read JSON from disk and return it.
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be a dictionary at the top level.")

    return data

def load_reference_metadata_json(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load filtered reference metadata JSON.

    Expected structure:
    {
      "0": {
        "original_doc_id": "...",
        "domain": "news",
        "src_text": "...",
        "reference_text": "..."
      },
      ...
    }
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference metadata JSON not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Reference metadata JSON must be a dictionary at the top level.")

    return data

def flatten_records(
    data: Dict[str, Dict[str, str]],
    doc_domain_map: Optional[Dict[str, str]] = None
) -> List[DocumentRecord]:
    """
    Convert nested JSON into a flat list of DocumentRecord objects.

    Example:
    {
        "0": {"ModelA": "...", "ModelB": "..."}
    }
    ->
    [
        DocumentRecord(doc_id="0", model="ModelA", text="..."),
        DocumentRecord(doc_id="0", model="ModelB", text="...")
    ]
    """
    records: List[DocumentRecord] = []

    for doc_id, model_map in data.items():
        if not isinstance(model_map, dict):
            continue

        domain = ""
        if doc_domain_map is not None:
            domain = doc_domain_map.get(str(doc_id), "")

        for model_name, text in model_map.items():
            if not isinstance(text, str):
                text = str(text)

            records.append(
                DocumentRecord(
                    doc_id=str(doc_id),
                    model=str(model_name),
                    text=text,
                    domain=domain,
                )
            )

    return records



# ============================================================
# 6. Text Normalization and Broken Output Detection
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text conservatively.

    The purpose of this function is stability, not rewriting.
    It should help parsing without changing meaning.
    """
    normalized = html.unescape(text)

    # Normalize smart quotes for more stable downstream handling.
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")

    # Remove control characters but preserve readable spacing.
    normalized = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", normalized)

    # Normalize line breaks and repeated spaces.
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def estimate_special_char_ratio(text: str) -> float:
    """
    Estimate noisy-character ratio for rough broken-output detection.

    This is intentionally simple because broken-output detection is only
    a front-end safety screen, not the main research task.
    """
    if not text:
        return 1.0

    special_chars = re.findall(r"[^0-9A-Za-z가-힣\s\.\,\!\?\-\'\"$$$$$$$$$$$$\{\}]", text)
    return len(special_chars) / max(len(text), 1)


def detect_broken_output(text: str, config: PipelineConfig) -> Optional[str]:
    """
    Detect clearly broken document-level outputs.

    Return:
    - None if the document looks acceptable enough for sentence-level analysis
    - A reason string if the document appears broken
    """
    stripped = text.strip()

    if not stripped:
        return "Document output is empty."

    if len(stripped) < config.broken_text_min_chars:
        return f"Document output is too short ({len(stripped)} chars)."

    sentence_punct_count = sum(stripped.count(p) for p in [".", "!", "?", "다", "요"])
    if sentence_punct_count < config.broken_text_min_sentences:
        return "Document output has too little sentence-like punctuation."

    special_ratio = estimate_special_char_ratio(stripped)
    if special_ratio > config.broken_special_char_ratio:
        return f"Document output has a high noisy-character ratio ({special_ratio:.2f})."

    return None


# ============================================================
# 7. Sentence Splitting
# ============================================================

def split_sentences(text: str) -> List[str]:
    """
    Split a document into sentence-like units.

    This splitter is intentionally conservative.
    The MVP does not aim for perfect segmentation.
    """
    if not text:
        return []

    working = text.replace("\n", " ")
    pieces = re.split(r"(?<=[\.\!\?])\s+", working)

    sentences: List[str] = []
    for piece in pieces:
        part = piece.strip()
        if part:
            sentences.append(part)

    # Fallback for long passages where punctuation splitting is weak.
    if len(sentences) <= 1 and len(text) > 120:
        fallback_parts = re.split(r"\s{2,}", text)
        fallback_sentences = [p.strip() for p in fallback_parts if p.strip()]
        if len(fallback_sentences) > len(sentences):
            sentences = fallback_sentences

    return sentences


# ============================================================
# 8. Kiwi Wrapper
# ============================================================

def parse_with_kiwi(sentence: str, kiwi: Kiwi) -> Any:
    """
    Wrap Kiwi tokenization.

    This function intentionally contains no business logic.
    """
    return kiwi.tokenize(sentence)


# ============================================================
# 9. Jamo Normalization Helpers
# ============================================================

def normalize_jamo_text(text: str) -> str:
    """
    Normalize leading jongseong-like jamo forms into compatibility jamo.

    Why this function matters:
    A parser may emit endings such as:
    - 'ᆯ게'
    - 'ᆫ다'
    - 'ᆸ니다'

    But our style dictionaries are usually written as:
    - 'ㄹ게'
    - 'ㄴ다'
    - 'ㅂ니다'

    Without normalization, practically valid endings may be mislabeled as UNKNOWN,
    which then inflates E05.
    """
    if not text:
        return text

    mapping = {
        "ᆯ": "ㄹ",
        "ᆫ": "ㄴ",
        "ᆷ": "ㅁ",
        "ᆸ": "ㅂ",
        "ᆼ": "ㅇ",
        "ᆮ": "ㄷ",
        "ᆺ": "ㅅ",
        "ᆨ": "ㄱ",
    }

    normalized = text
    for src, tgt in mapping.items():
        normalized = normalized.replace(src, tgt)

    return normalized


# ============================================================
# 10. Subject Extraction
# ============================================================

def classify_subject_surface(surface: str, config: PipelineConfig) -> str:
    """
    Classify a subject candidate from its surface form.

    This is intentionally shallow.
    The goal is to support practical E01 detection, not full syntactic analysis.
    """
    normalized = surface.strip()

    if not normalized:
        return SUBJECT_OMITTED

    if any(form == normalized or normalized.startswith(form) for form in config.self_subject_forms):
        return SUBJECT_SELF

    if any(form == normalized or normalized.startswith(form) for form in config.addressee_forms):
        return SUBJECT_ADDRESSEE

    return SUBJECT_THIRD_OR_NOUN


def extract_subject_info(sentence: str, tokens: Any, config: PipelineConfig) -> Tuple[str, str]:
    """
    Extract a likely subject candidate and map it into a simple subject group.

    Conservative policy:
    - Prefer explicit noun/pronoun + marker combinations
    - Fall back to first-person pronouns if visible
    - Otherwise return OMITTED or UNKNOWN rather than over-claiming
    """
    if not tokens:
        return "", SUBJECT_UNKNOWN

    for idx, tok in enumerate(tokens):
        tag = getattr(tok, "tag", "")
        form = getattr(tok, "form", "")

        # Subject/topic markers often follow noun-like forms.
        if tag in {"JKS", "JX"} and idx > 0:
            prev_tok = tokens[idx - 1]
            prev_form = getattr(prev_tok, "form", "")
            prev_tag = getattr(prev_tok, "tag", "")

            if prev_tag.startswith("N") or prev_tag == "NP":
                subject_text = f"{prev_form}{form}"
                subject_group = classify_subject_surface(subject_text, config)
                return subject_text, subject_group

    # Weak fallback for first-person pronouns.
    for tok in tokens:
        form = getattr(tok, "form", "")
        tag = getattr(tok, "tag", "")
        if tag == "NP" and any(form == base for base in config.self_subject_forms):
            return form, SUBJECT_SELF

    has_noun_like = any(
        getattr(tok, "tag", "").startswith("N") or getattr(tok, "tag", "") == "NP"
        for tok in tokens
    )

    if has_noun_like:
        return "", SUBJECT_UNKNOWN

    return "", SUBJECT_OMITTED


# ============================================================
# 11. Predicate and Ending Extraction
# ============================================================

def token_text_span(sentence: str, surface: str) -> Tuple[int, int]:
    """
    Find a conservative character span for a surface string inside the sentence.

    This is only a fallback span finder.
    It does not need to be perfect for MVP review use.
    """
    if not surface:
        return -1, -1

    start = sentence.rfind(surface)
    if start == -1:
        return -1, -1

    end = start + len(surface)
    return start, end


def find_last_predicate_index(tokens: Any) -> Optional[int]:
    """
    Find the last predicate-like token index.

    We search backward because sentence-final predicate material is most useful
    for style classification and honorific review.
    """
    predicate_tags = {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}

    for idx in range(len(tokens) - 1, -1, -1):
        tag = getattr(tokens[idx], "tag", "")
        if tag in predicate_tags:
            return idx

    return None


def postprocess_predicate_text(text: str) -> str:
    """
    Apply a minimal readability-oriented postprocessing step.

    The goal is not perfect conjugation restoration.
    The goal is only to avoid obviously awkward outputs when possible.
    """
    if not text:
        return text

    replacements = {
        "하었다": "했다",
        "하였": "했",
    }

    processed = text
    for src, tgt in replacements.items():
        processed = processed.replace(src, tgt)

    return processed


def find_final_ef_text(tokens: Any) -> str:
    """
    Find the sentence-final EF token near the end.

    We do not require the literal last token to be EF because punctuation,
    closing quotes, or closing brackets may appear after it.
    """
    closing_tags = {"SF", "SE", "SSO", "SSC", "SP", "SY"}

    idx = len(tokens) - 1
    while idx >= 0:
        tag = getattr(tokens[idx], "tag", "")
        if tag in closing_tags:
            idx -= 1
            continue
        break

    while idx >= 0:
        tag = getattr(tokens[idx], "tag", "")
        form = getattr(tokens[idx], "form", "")
        if tag == "EF":
            return normalize_jamo_text(form)

        # Stop if a lexical token blocks the search before EF is found.
        if tag.startswith("N") or tag in {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}:
            break

        idx -= 1

    return ""


def find_final_ec_text(tokens: Any) -> str:
    """
    Find the sentence-final EC token near the end.

    Why this matters:
    If a clause ends with EC rather than EF, it is likely non-final or truncated.
    Such cases should not be aggressively style-classified.
    """
    closing_tags = {"SF", "SE", "SSO", "SSC", "SP", "SY"}

    idx = len(tokens) - 1
    while idx >= 0:
        tag = getattr(tokens[idx], "tag", "")
        if tag in closing_tags:
            idx -= 1
            continue
        break

    while idx >= 0:
        tag = getattr(tokens[idx], "tag", "")
        form = getattr(tokens[idx], "form", "")
        if tag == "EC":
            return normalize_jamo_text(form)

        if tag.startswith("N") or tag in {"VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA"}:
            break

        idx -= 1

    return ""


def extract_predicate_info(sentence: str, tokens: Any) -> Tuple[str, str, str, str, bool, int, int]:
    """
    Extract practical predicate information for review.

    Returns:
    - predicate_text
    - ending_text
    - final_ef_text
    - final_ec_text
    - has_si
    - char_span_start
    - char_span_end

    Design note:
    This function is intentionally heuristic.
    It is not trying to build a full syntactic analysis.
    """
    if not tokens:
        return "", "", "", "", False, -1, -1

    last_pred_idx = find_last_predicate_index(tokens)
    final_ef_text = find_final_ef_text(tokens)
    final_ec_text = find_final_ec_text(tokens)

    if last_pred_idx is None:
        return "", "", final_ef_text, final_ec_text, False, -1, -1

    predicate_parts: List[str] = []
    ending_parts: List[str] = []
    has_si = False

    for idx in range(last_pred_idx, len(tokens)):
        tok = tokens[idx]
        form = getattr(tok, "form", "")
        tag = getattr(tok, "tag", "")

        # Keep stem and nearby ending-zone tokens.
        if idx == last_pred_idx:
            predicate_parts.append(form)
        elif tag in {"EP", "EF", "EC", "ETN", "ETM", "JX", "SF", "SE", "SSO", "SSC"}:
            predicate_parts.append(form)
        elif tag.startswith("E"):
            predicate_parts.append(form)
        else:
            break

        if tag == "EP" and form in {"시", "으시"}:
            has_si = True

        if tag.startswith("E"):
            ending_parts.append(normalize_jamo_text(form))

    predicate_text = "".join(predicate_parts).strip()
    predicate_text = postprocess_predicate_text(predicate_text)

    ending_text = "".join(ending_parts).strip()
    ending_text = normalize_jamo_text(ending_text)

    if not predicate_text:
        predicate_text = getattr(tokens[last_pred_idx], "form", "")

    span_start, span_end = token_text_span(sentence, predicate_text)

    return (
        predicate_text,
        ending_text,
        final_ef_text,
        final_ec_text,
        has_si,
        span_start,
        span_end,
    )


# ============================================================
# 12. Fragment-Like and Clause-Type Helpers
# ============================================================

def is_fragment_like_sentence(sentence: str, tokens: Any, predicate_text: str, config: PipelineConfig) -> bool:
    """
    Detect whether a sentence looks more like a fragment than a full predicate-bearing sentence.

    Typical examples:
    - titles
    - slogans
    - noun phrase lists
    - short coordinated fragments

    We keep this heuristic conservative because we do not want to overclassify.
    """
    if not sentence.strip():
        return False

    # If a clear predicate exists, do not call it a fragment.
    if predicate_text:
        return False

    if not tokens or len(tokens) < config.fragment_min_token_count:
        return False

    noun_like_count = sum(
        1 for tok in tokens
        if getattr(tok, "tag", "").startswith("N") or getattr(tok, "tag", "") == "NP"
    )

    comma_like = sentence.count(",") + sentence.count("·") + sentence.count("•")

    # A short, noun-heavy, predicate-free unit is a good fragment candidate.
    if noun_like_count >= 2 and len(tokens) <= config.fragment_max_token_count_for_short_np:
        return True

    # Repeated listing punctuation is also a strong fragment hint.
    if comma_like >= 1 and noun_like_count >= 2 and not predicate_text:
        return True

    return False


# ============================================================
# 13. Style Classification
# ============================================================

def ending_matches(ending_text: str, candidates: Sequence[str]) -> bool:
    """
    Check whether an ending string matches one of the candidate endings.

    We allow exact match or suffix-style match to tolerate parser variation.
    """
    if not ending_text:
        return False

    for cand in candidates:
        if ending_text == cand or ending_text.endswith(cand):
            return True

    return False


def classify_style(
    sentence: str,
    ending_text: str,
    final_ef_text: str,
    final_ec_text: str,
    predicate_text: str,
    config: PipelineConfig
) -> str:
    """
    Classify sentence style into one of five practical labels.

    Key policies:
    - Rescue obvious modern conversational endings before fallback
    - Use sentence-final EF as the strongest evidence when available
    - Avoid treating connective EC endings as sentence-final style evidence
    - Avoid greedy archaic detection based on raw substring matches
    """

    # --------------------------------------------------------
    # 1. If the sentence appears to end with EC and lacks EF,
    # we do not aggressively style-classify it here.
    #
    # This protects non-final clauses such as:
    # - ...시고
    # - ...는데
    # - ...고
    # --------------------------------------------------------
    if final_ec_text and not final_ef_text:
        return STYLE_UNKNOWN

    # --------------------------------------------------------
    # 2. Rescue casual promise/intention endings first.
    #
    # These are very important because they were previously
    # misrouted into UNKNOWN/E05 or false archaic buckets.
    # --------------------------------------------------------
    if final_ef_text in {"ㄹ게", "을게"}:
        return STYLE_CASUAL

    # Surface-form fallback for common final casual promise forms.
    # This is intentionally limited and only used as a practical rescue path.
    if sentence.strip().endswith(("할게.", "갈게.", "올게.", "줄게.", "볼게.", "먹을게.", "할게", "갈게", "올게", "줄게", "볼게", "먹을게")):
        return STYLE_CASUAL

    # --------------------------------------------------------
    # 3. Polite detection
    # --------------------------------------------------------
    if ending_matches(final_ef_text, config.polite_endings):
        return STYLE_POLITE
    if ending_matches(ending_text, config.polite_endings):
        return STYLE_POLITE
    if predicate_text.endswith((
        "습니다", "ㅂ니다", "니다",
        "요", "세요", "셔요",
        "까요", "네요",
        "예요", "에요",
        "입니다", "됩니다",
        "죠", "지요"
    )):
        return STYLE_POLITE

    # --------------------------------------------------------
    # 4. Archaic detection
    #
    # Important:
    # We only trust narrow sentence-final EF evidence here.
    # We intentionally exclude modern conversational '-네' from archaic detection.
    # --------------------------------------------------------

    if final_ef_text in config.archaic_final_ef_forms:
        return STYLE_ARCHAIC

    # --------------------------------------------------------
    # 5. Plain detection
    # --------------------------------------------------------
    if ending_matches(final_ef_text, config.plain_endings):
        return STYLE_PLAIN
    if ending_matches(ending_text, config.plain_endings):
        return STYLE_PLAIN
    if predicate_text.endswith((
        "다", "는다", "ㄴ다",
        "었다", "았다", "했다", "였다"
    )):
        return STYLE_PLAIN

    # --------------------------------------------------------
    # 6. Casual detection
    # --------------------------------------------------------
    if ending_matches(final_ef_text, config.casual_endings):
        return STYLE_CASUAL
    if ending_matches(ending_text, config.casual_endings):
        return STYLE_CASUAL
    if predicate_text.endswith((
        "아", "어", "지", "니", "냐",
        "었어", "았어",
        "겠지", "거야", "잖아",
        "줘", "래",
        "야", "이야",
        "대", "더라",
        "데", "ㄴ데", "는데",
        "네"
    )):
        return STYLE_CASUAL

    return STYLE_UNKNOWN


# ============================================================
# 14. Complex or Unknown Detection
# ============================================================

def has_quote_like_pattern(sentence: str, config: PipelineConfig) -> bool:
    """
    Check whether the sentence contains explicit quote/report patterns.

    This is deliberately narrow.
    We do not want broad triggers such as every occurrence of '다고'.
    """
    for pattern in config.quote_like_patterns:
        if pattern in sentence:
            return True
    return False


def detect_complex_or_unknown(
    sentence: str,
    tokens: Any,
    predicate_text: str,
    style_label: str,
    fragment_like: bool,
    final_ef_text: str,
    final_ec_text: str,
    config: PipelineConfig
) -> Tuple[bool, str]:

    """
    Detect whether the sentence should fall into a low-confidence fallback path.

    Current policy:
    - If a style is already known and predicate extraction succeeded, do not assign a fallback issue.
    - If the sentence is fragment-like, explain it as such.
    - If the sentence ends with EC rather than EF, treat it as non-final or truncated.
    - Quote/report patterns matter only when other analysis has already failed.
    """

    if not tokens:
        return True, "Parser returned no tokens."

    # Fragment-like units are not necessarily wrong,
    # but they are not comfortably style-classifiable either.
    if fragment_like:
        return True, "Fragment-like sentence without a clear predicate."

    # If we have a non-final clause ending in EC without EF,
    # we avoid strong style judgment and keep it as a cautious fallback.
    if final_ec_text and not final_ef_text:
        return True, f"Sentence ends with EC ('{final_ec_text}'), not EF."

    # If predicate extraction failed, the sentence remains uncertain.
    if not predicate_text:
        if has_quote_like_pattern(sentence, config):
            return True, "Predicate extraction failed in a quote/report-like sentence."
        return True, "Predicate extraction failed."

    # If style is already known, do not downgrade it to E05.
    if style_label != STYLE_UNKNOWN:
        return False, ""

    # At this point the style classification failed after normalization and fallback rules.
    if has_quote_like_pattern(sentence, config):
        return True, "Style classification failed in a quote/report-like sentence."

    return True, "Style classification is unknown after normalization."


# ============================================================
# 15. Sentence Analysis Orchestration
# ============================================================

def analyze_sentence(
    doc_id: str,
    model: str,
    sent_id: int,
    sentence: str,
    kiwi: Kiwi,
    config: PipelineConfig
) -> SentenceAnalysis:
    """
    Analyze one sentence and produce one structured intermediate analysis object.

    This function acts as the center of sentence-level analysis.
    It collects:
    - parser output
    - subject candidate
    - predicate and ending evidence
    - style label
    - fallback/complexity explanation
    """
    tokens = parse_with_kiwi(sentence, kiwi)

    subject_text, subject_group = extract_subject_info(sentence, tokens, config)

    (
        predicate_text,
        ending_text,
        final_ef_text,
        final_ec_text,
        has_si,
        span_start,
        span_end,
    ) = extract_predicate_info(sentence, tokens)

    fragment_like = is_fragment_like_sentence(
        sentence=sentence,
        tokens=tokens,
        predicate_text=predicate_text,
        config=config,
    )

    style_label = classify_style(
        sentence=sentence,
        ending_text=ending_text,
        final_ef_text=final_ef_text,
        final_ec_text=final_ec_text,
        predicate_text=predicate_text,
        config=config,
    )

    complex_or_unknown, complex_reason = detect_complex_or_unknown(
        sentence=sentence,
        tokens=tokens,
        predicate_text=predicate_text,
        style_label=style_label,
        fragment_like=fragment_like,
        final_ef_text=final_ef_text,
        final_ec_text=final_ec_text,
        config=config,
    )

    return SentenceAnalysis(
        doc_id=doc_id,
        model=model,
        sent_id=sent_id,
        sentence=sentence,
        subject_text=subject_text,
        subject_group=subject_group,
        predicate_text=predicate_text,
        ending_text=final_ef_text if final_ef_text else ending_text,
        final_ef_text=final_ef_text,
        final_ec_text=final_ec_text,
        style_label=style_label,
        has_si=has_si,
        fragment_like=fragment_like,
        complex_or_unknown=complex_or_unknown,
        complex_reason=complex_reason,
        char_span_start=span_start,
        char_span_end=span_end,
    )


def analyze_document(
    record: DocumentRecord,
    kiwi: Kiwi,
    config: PipelineConfig
) -> Tuple[List[SentenceAnalysis], Optional[str]]:
    """
    Analyze a whole document-model pair.

    Returns:
    - sentence-level analyses
    - optional broken-output reason
    """
    cleaned_text = normalize_text(record.text)
    broken_reason = detect_broken_output(cleaned_text, config)

    sentences = split_sentences(cleaned_text)
    analyses: List[SentenceAnalysis] = []

    for sent_id, sentence in enumerate(sentences):
        analysis = analyze_sentence(
            doc_id=record.doc_id,
            model=record.model,
            sent_id=sent_id,
            sentence=sentence,
            kiwi=kiwi,
            config=config,
        )
        analyses.append(analysis)

    return analyses, broken_reason


# ============================================================
# 16. Document-Level Style Aggregation
# ============================================================

def get_dominant_doc_style(rows: List[SentenceAnalysis], config: PipelineConfig) -> str:
    """
    Determine the dominant style of the document.

    UNKNOWN is ignored as much as possible.
    """
    valid_styles = [
        row.style_label
        for row in rows
        if row.style_label != STYLE_UNKNOWN
    ]

    if len(valid_styles) < config.dominant_style_min_valid_sentences:
        return STYLE_UNKNOWN

    counter = Counter(valid_styles)
    return counter.most_common(1)[0][0]

def build_reference_style_profile(
    doc_id: str,
    ref_row: Dict[str, Any],
    kiwi: Kiwi,
    config: PipelineConfig
) -> ReferenceStyleProfile:
    """
    Analyze the human reference text and build a simple document-level style profile.
    """
    domain = str(ref_row.get("domain", "")).strip().lower()
    reference_text = str(ref_row.get("reference_text", "") or "")

    cleaned_text = normalize_text(reference_text)
    sentences = split_sentences(cleaned_text)

    analyses: List[SentenceAnalysis] = []
    for sent_id, sentence in enumerate(sentences):
        analysis = analyze_sentence(
            doc_id=str(doc_id),
            model="REFERENCE",
            sent_id=sent_id,
            sentence=sentence,
            kiwi=kiwi,
            config=config,
        )
        analyses.append(analysis)

    valid_styles = [
        row.style_label
        for row in analyses
        if row.style_label in {STYLE_PLAIN, STYLE_POLITE, STYLE_CASUAL, STYLE_ARCHAIC}
    ]

    counter = Counter(valid_styles)
    valid_n = sum(counter.values())

    if valid_n == 0:
        style_counts = {
            STYLE_PLAIN: 0,
            STYLE_POLITE: 0,
            STYLE_CASUAL: 0,
            STYLE_ARCHAIC: 0,
        }
        style_ratios = {
            STYLE_PLAIN: 0.0,
            STYLE_POLITE: 0.0,
            STYLE_CASUAL: 0.0,
            STYLE_ARCHAIC: 0.0,
        }
        dominant_style = STYLE_UNKNOWN
        allowed_styles: List[str] = []
    else:
        style_counts = {
            STYLE_PLAIN: counter.get(STYLE_PLAIN, 0),
            STYLE_POLITE: counter.get(STYLE_POLITE, 0),
            STYLE_CASUAL: counter.get(STYLE_CASUAL, 0),
            STYLE_ARCHAIC: counter.get(STYLE_ARCHAIC, 0),
        }
        style_ratios = {
            style: style_counts[style] / valid_n
            for style in style_counts
        }
        dominant_style = counter.most_common(1)[0][0]

        allowed_styles = []
        for style in [STYLE_PLAIN, STYLE_POLITE, STYLE_CASUAL, STYLE_ARCHAIC]:
            count = style_counts[style]
            ratio = style_ratios[style]

            # Allow a style if it has non-trivial support in the human reference.
            if count >= 2 and ratio >= 0.10:
                allowed_styles.append(style)
            elif ratio >= 0.25:
                allowed_styles.append(style)

    return ReferenceStyleProfile(
        doc_id=str(doc_id),
        domain=domain,
        dominant_style=dominant_style,
        valid_sentence_count=valid_n,
        style_counts=style_counts,
        style_ratios=style_ratios,
        allowed_styles=allowed_styles,
    )


def build_reference_style_profile_map(
    reference_metadata: Dict[str, Dict[str, Any]],
    kiwi: Kiwi,
    config: PipelineConfig
) -> Dict[str, ReferenceStyleProfile]:
    profile_map: Dict[str, ReferenceStyleProfile] = {}

    for doc_id, ref_row in reference_metadata.items():
        if not isinstance(ref_row, dict):
            continue
        profile = build_reference_style_profile(
            doc_id=str(doc_id),
            ref_row=ref_row,
            kiwi=kiwi,
            config=config,
        )
        profile_map[str(doc_id)] = profile

    return profile_map

def get_reference_ratio(profile: Optional[ReferenceStyleProfile], style: str) -> float:
    if profile is None:
        return 0.0
    return float(profile.style_ratios.get(style, 0.0))


def get_hybrid_e02_decision(
    domain: str,
    dominant_style: str,
    sentence_style: str,
    reference_profile: Optional[ReferenceStyleProfile],
) -> Optional[Tuple[Optional[str], str]]:
    """
    Hybrid E02 decision:
    - return None -> no E02
    - return (severity, reason) -> keep E02 with that severity

    Design:
    - domain = prior / default policy
    - reference profile = evidence / document-specific adjustment
    """
    domain = (domain or "").strip().lower()

    if dominant_style == STYLE_UNKNOWN or sentence_style == STYLE_UNKNOWN:
        return None

    if dominant_style == sentence_style:
        return None

    triad = {STYLE_PLAIN, STYLE_POLITE, STYLE_CASUAL}
    pair = {dominant_style, sentence_style}

    allowed_styles = set(reference_profile.allowed_styles) if reference_profile is not None else set()
    sentence_ratio = get_reference_ratio(reference_profile, sentence_style)
    dominant_ratio = get_reference_ratio(reference_profile, dominant_style)

    # --------------------------------------------------------
    # 1) Strong skip zone:
    # if the sentence style is explicitly supported by human reference
    # and the domain naturally allows style variation.
    # --------------------------------------------------------
    if domain in {"literary", "social"}:
        if pair <= triad and sentence_style in allowed_styles:
            return None

    if domain == "news":
        if pair <= {STYLE_PLAIN, STYLE_POLITE} and sentence_style in allowed_styles:
            return None

    # --------------------------------------------------------
    # 2) Domain-level softening:
    # use WARNING when the domain plausibly allows style variation,
    # even if reference evidence is not strong enough for full skip.
    # --------------------------------------------------------
    if domain == "literary":
        if pair <= triad:
            return (
                SEVERITY_WARNING,
                (
                    f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}. "
                    f"Domain='literary' allows broad style variation; "
                    f"reference support={sentence_ratio:.2f}."
                ),
            )

    if domain == "social":
        if pair <= triad:
            return (
                SEVERITY_WARNING,
                (
                    f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}. "
                    f"Domain='social' allows moderate style variation; "
                    f"reference support={sentence_ratio:.2f}."
                ),
            )

    if domain == "news":
        if pair <= {STYLE_PLAIN, STYLE_POLITE}:
            return (
                SEVERITY_WARNING,
                (
                    f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}. "
                    f"Domain='news' partially allows PLAIN/POLITE mixing; "
                    f"reference support={sentence_ratio:.2f}."
                ),
            )
        if sentence_style == STYLE_CASUAL and sentence_ratio >= 0.20:
            return (
                SEVERITY_WARNING,
                (
                    f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}. "
                    f"Domain='news' is usually strict, but the human reference contains noticeable CASUAL usage "
                    f"({sentence_ratio:.2f})."
                ),
            )

    # --------------------------------------------------------
    # 3) Reference-evidence fallback:
    # even outside the domain defaults, if the human reference strongly
    # supports both styles, we downgrade to WARNING.
    # --------------------------------------------------------
    if sentence_style in allowed_styles and dominant_style in allowed_styles:
        return (
            SEVERITY_WARNING,
            (
                f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}. "
                f"Human reference supports both styles "
                f"(dominant_ratio={dominant_ratio:.2f}, sentence_ratio={sentence_ratio:.2f})."
            ),
        )

    # --------------------------------------------------------
    # 4) Default fallback behavior for unresolved style-shift cases
    # --------------------------------------------------------

    if is_strong_style_shift(dominant_style, sentence_style):
        return (
            SEVERITY_ERROR,
            f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}."
        )

    if is_weak_style_shift(dominant_style, sentence_style):
        return (
            SEVERITY_WARNING,
            f"Dominant document style is {dominant_style}, but this sentence is {sentence_style}."
        )

    return None

# ============================================================
# 17. Issue Detection
# ============================================================

def is_strong_style_shift(dominant_style: str, sentence_style: str) -> bool:
    """
    Determine whether a style shift is strong enough for ERROR.

    Strong shifts:
    - POLITE <-> PLAIN
    - POLITE <-> CASUAL
    """
    strong_pairs = {
        (STYLE_POLITE, STYLE_PLAIN),
        (STYLE_PLAIN, STYLE_POLITE),
        (STYLE_POLITE, STYLE_CASUAL),
        (STYLE_CASUAL, STYLE_POLITE),
    }
    return (dominant_style, sentence_style) in strong_pairs


def is_weak_style_shift(dominant_style: str, sentence_style: str) -> bool:
    """
    Determine whether a style shift is weaker and should be WARNING.

    Weak shifts:
    - PLAIN <-> CASUAL
    """
    weak_pairs = {
        (STYLE_PLAIN, STYLE_CASUAL),
        (STYLE_CASUAL, STYLE_PLAIN),
    }
    return (dominant_style, sentence_style) in weak_pairs


def detect_issue_for_sentence(
    row: SentenceAnalysis,
    dominant_style: str,
    domain: str,
    reference_profile: Optional[ReferenceStyleProfile],
) -> Optional[Tuple[str, str, str]]:

    """
    Return one primary issue for a sentence if needed.

    Priority order:
    1. E01_SELF_HONORIFIC_SUBJECT
    2. E03_ARCHAIC_OR_ODD_STYLE
    3. E02_DOC_STYLE_SHIFT
    4. E05_UNKNOWN_ENDING / E06_QUOTE_OR_COMPLEX_CLAUSE

    Why E05/E06 remain last:
    - They are fallback categories for low-confidence cases
    - More specific sentence-local signals should win first
    - E05 is used for ending-classification failure
    - E06 is used for quotation or complex-clause review cases
    """

    # Priority 1: strongest sentence-local explicit rule.
    if row.subject_group == SUBJECT_SELF and row.has_si:
        return (
            ISSUE_E01,
            get_issue_severity(ISSUE_E01),
            "SELF subject is combined with the honorific marker -si-."
        )

    # Priority 2: narrow archaic warning.
    if row.style_label == STYLE_ARCHAIC:
        return (
            ISSUE_E03,
            get_issue_severity(ISSUE_E03),
            "Sentence-final EF indicates an archaic or unusual style."
        )

    # Priority 3: style shift relative to document dominant style.
    if dominant_style != STYLE_UNKNOWN and row.style_label != STYLE_UNKNOWN:
        if row.style_label != dominant_style:
            hybrid_decision = get_hybrid_e02_decision(
                domain=domain,
                dominant_style=dominant_style,
                sentence_style=row.style_label,
                reference_profile=reference_profile,
            )
            if hybrid_decision is not None:
                severity, reason = hybrid_decision
                return (
                    ISSUE_E02,
                    severity,
                    reason,
                )

    # Priority 4: final fallback for low-confidence cases.
    if row.complex_or_unknown:
        complex_reason_lower = (row.complex_reason or "").lower()

        if any(
            keyword in complex_reason_lower
            for keyword in ["quote", "quoted", "quotation", "reporting", "complex", "clause"]
        ):
            return (
                ISSUE_E06,
                get_issue_severity(ISSUE_E06),
                row.complex_reason or "Sentence contains a quotation or a complex clause."
            )

        return (
            ISSUE_E05,
            get_issue_severity(ISSUE_E05),
            row.complex_reason or "Sentence-final style could not be classified with confidence."
        )

    return None


def build_broken_output_issue(record: DocumentRecord, reason: str) -> IssueRecord:
    """
    Build a document-level broken-output issue row.

    We use sent_id = -1 because E04 is not sentence-specific.
    """
    return IssueRecord(
        doc_id=record.doc_id,
        model=record.model,
        domain=record.domain,
        sent_id=-1,
        sentence=record.text[:500],
        subject_text="",
        subject_group=SUBJECT_UNKNOWN,
        predicate_text="",
        ending_text="",
        style_label=STYLE_UNKNOWN,
        dominant_doc_style=STYLE_UNKNOWN,
        issue_code=ISSUE_E04,
        severity=get_issue_severity(ISSUE_E04),
        reason=reason,
        char_span_start=-1,
        char_span_end=-1,
    )


def build_issue_record(
    row: SentenceAnalysis,
    dominant_style: str,
    issue_code: str,
    severity: str,
    reason: str,
    domain: str,
) -> IssueRecord:
    """
    Convert sentence analysis plus issue metadata into one final export row.
    """
    return IssueRecord(
        doc_id=row.doc_id,
        model=row.model,
        domain=domain,
        sent_id=row.sent_id,
        sentence=row.sentence,
        subject_text=row.subject_text,
        subject_group=row.subject_group,
        predicate_text=row.predicate_text,
        ending_text=row.ending_text,
        style_label=row.style_label,
        dominant_doc_style=dominant_style,
        issue_code=issue_code,
        severity=severity,
        reason=reason,
        char_span_start=row.char_span_start,
        char_span_end=row.char_span_end,
    )


# ============================================================
# 18. Pipeline Runner
# ============================================================

def filter_records(
    records: List[DocumentRecord],
    sample_doc_ids: Optional[List[str]] = None,
    sample_models: Optional[List[str]] = None
) -> List[DocumentRecord]:
    """
    Filter records for sample-first execution.

    This is important because the workflow should remain:
    sample first -> inspect -> tune -> full run later
    """
    filtered = records

    if sample_doc_ids is not None:
        sample_doc_ids_set = set(sample_doc_ids)
        filtered = [r for r in filtered if r.doc_id in sample_doc_ids_set]

    if sample_models is not None:
        sample_models_set = set(sample_models)
        filtered = [r for r in filtered if r.model in sample_models_set]

    return filtered


def run_pipeline(
    records: List[DocumentRecord],
    kiwi: Kiwi,
    config: PipelineConfig,
    reference_profile_map: Optional[Dict[str, ReferenceStyleProfile]] = None,
    sample_doc_ids: Optional[List[str]] = None,
    sample_models: Optional[List[str]] = None
) -> List[IssueRecord]:

    """
    Run the end-to-end suspicious-case generation pipeline.
    """
    selected_records = filter_records(records, sample_doc_ids, sample_models)
    issue_rows: List[IssueRecord] = []

    print(f"[INFO] Total records after filtering: {len(selected_records)}")

    domain_counter = Counter(r.domain or "UNKNOWN_DOMAIN" for r in selected_records)
    print(f"[INFO] Record domain distribution: {dict(domain_counter)}")

    for idx, record in enumerate(selected_records, start=1):
        print(
            f"[INFO] Processing {idx}/{len(selected_records)} - "
            f"doc_id={record.doc_id}, model={record.model}, domain={record.domain}"
        )

        analyses, broken_reason = analyze_document(record, kiwi, config)

        if broken_reason is not None:
            issue_rows.append(build_broken_output_issue(record, broken_reason))

        dominant_style = get_dominant_doc_style(analyses, config)
        reference_profile = None
        if reference_profile_map is not None:
            reference_profile = reference_profile_map.get(record.doc_id)

        for row in analyses:
            result = detect_issue_for_sentence(
                row=row,
                dominant_style=dominant_style,
                domain=record.domain,
                reference_profile=reference_profile,
            )
            if result is None:
                continue

            issue_code, severity, reason = result
            issue_rows.append(
                build_issue_record(
                    row=row,
                    dominant_style=dominant_style,
                    issue_code=issue_code,
                    severity=severity,
                    reason=reason,
                    domain=record.domain,
                )
            )

    issue_counter = Counter((r.issue_code, r.severity) for r in issue_rows)
    print("[INFO] Issue summary:")
    for key, value in sorted(issue_counter.items()):
        print(f" - {key}: {value}")

    return issue_rows



# ============================================================
# 19. CSV Export
# ============================================================

def export_raw_csv(rows: List[IssueRecord], out_path: str) -> None:
    """
    Export the full raw suspicious-case table for internal analysis.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    print(f"[INFO] Raw CSV exported to: {output_path}")


def build_review_row(row: IssueRecord) -> Dict[str, Any]:
    """
    Convert a raw IssueRecord into a compact review-friendly row.
    """
    return {
        "doc_id": row.doc_id,
        "model": row.model,
        "domain": row.domain,
        "sent_id": row.sent_id,
        "sentence": row.sentence,
        "issue_code": row.issue_code,
        "severity": row.severity,
        "review_priority": get_review_priority(row),
        "brief_reason": get_issue_brief_reason(row.issue_code),
        "subject_text": row.subject_text,
        "predicate_text": row.predicate_text,
        "ending_text": row.ending_text,
        "style_label": row.style_label,
        "dominant_doc_style": row.dominant_doc_style,
    }


def export_review_csv(rows: List[IssueRecord], out_path: str) -> None:
    """
    Export a compact review-oriented CSV for professor-facing inspection.
    """
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    review_rows = [build_review_row(row) for row in rows]
    review_rows.sort(
    key=lambda r: (
        0 if r["review_priority"] == "FOCUS" else 1,
        0 if r["severity"] == SEVERITY_ERROR else 1,
        str(r["doc_id"]),
        str(r["model"]),
        int(r["sent_id"]),
    )
)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_CSV_COLUMNS)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    print(f"[INFO] Review CSV exported to: {output_path}")


# ============================================================
# 20. Utility for Model Inspection
# ============================================================

def print_available_models(data: Dict[str, Dict[str, str]], doc_id: str) -> None:
    """
    Print available model names for a specific document.

    This is helpful because sample_models must match exact JSON keys.
    """
    if doc_id not in data:
        print(f"[WARN] doc_id={doc_id} not found in JSON.")
        return

    model_names = list(data[doc_id].keys())
    print(f"[INFO] Available models for doc_id={doc_id}:")
    for name in model_names:
        print(f" - {name}")


# ============================================================
# 21. Main Entry Point
# ============================================================

def main() -> None:
    """
    Main execution entry.

    Current behavior:
    - Run either the full dataset or selected doc_ids
    - Export both raw and review CSV outputs
    """
    # --------------------------------------------------------
    # Input paths
    # --------------------------------------------------------
    input_path = "data/wmt25/tgt_doc.json"
    reference_metadata_path = "data/wmt25/reference_ko_general_nospeech.json"

    # --------------------------------------------------------
    # Sample mode controls
    # --------------------------------------------------------
    sample_doc_ids: Optional[List[str]] = None
    sample_models: Optional[List[str]] = None

    version_tag = "v45"

    # --------------------------------------------------------
    # Build config and parser
    # --------------------------------------------------------
    config = build_default_config()
    kiwi = Kiwi()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    data = load_tgt_doc_json(input_path)
    print_available_models(data, doc_id="0")

    reference_metadata = load_reference_metadata_json(reference_metadata_path)
    print(f"[INFO] Loaded reference metadata for {len(reference_metadata)} docs.")

    doc_domain_map = {
        str(doc_id): str(row.get("domain", "")).strip().lower()
        for doc_id, row in reference_metadata.items()
        if isinstance(row, dict)
    }

    reference_profile_map = build_reference_style_profile_map(
        reference_metadata=reference_metadata,
        kiwi=kiwi,
        config=config,
    )
    print(f"[INFO] Built reference style profiles for {len(reference_profile_map)} docs.")

    records = flatten_records(data, doc_domain_map=doc_domain_map)
    print(f"[INFO] Loaded {len(records)} total document-model records.")

    # --------------------------------------------------------
    # Run pipeline + export
    # --------------------------------------------------------
    if sample_doc_ids is None:
        raw_output_path = f"script/output_all/honorific_mvp_{version_tag}_all_docs_raw.csv"
        review_output_path = f"script/output_all/honorific_mvp_{version_tag}_all_docs_review.csv"

        issue_rows = run_pipeline(
            records=records,
            kiwi=kiwi,
            config=config,
            reference_profile_map=reference_profile_map,
            sample_doc_ids=None,
            sample_models=sample_models,
        )

        export_raw_csv(issue_rows, raw_output_path)
        export_review_csv(issue_rows, review_output_path)
        print(f"[INFO] Suspicious-case rows: {len(issue_rows)}")
        print(f"[INFO] Saved raw: {raw_output_path}")
        print(f"[INFO] Saved review: {review_output_path}")

    else:
        for doc_id in sample_doc_ids:
            raw_output_path = f"script/output_samples/honorific_mvp_{version_tag}_doc{doc_id}_raw.csv"
            review_output_path = f"script/output_samples/honorific_mvp_{version_tag}_doc{doc_id}_review.csv"

            print("\n" + "=" * 60)
            print(f"[INFO] Running single-doc export for doc_id={doc_id}")
            print("=" * 60)

            issue_rows = run_pipeline(
                records=records,
                kiwi=kiwi,
                config=config,
                reference_profile_map=reference_profile_map,
                sample_doc_ids=[doc_id],
                sample_models=sample_models,
            )

            export_raw_csv(issue_rows, raw_output_path)
            export_review_csv(issue_rows, review_output_path)
            print(f"[INFO] Suspicious-case rows for doc_id={doc_id}: {len(issue_rows)}")
            print(f"[INFO] Saved raw: {raw_output_path}")
            print(f"[INFO] Saved review: {review_output_path}")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()
