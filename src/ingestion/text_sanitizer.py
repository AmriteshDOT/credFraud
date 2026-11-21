import re
from dataclasses import dataclass

# Fast deterministic regex patterns (paired with Microsoft Presidio NER in prod) : Presidio later but feels slow for UPI like transaction?
AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
PHONE_RE = re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b")
INJECTION_RE = re.compile(
    r"(ignore\s+previous|system\s+override|disregard\s+instructions|approve\s+immediately)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SanitizedText:
    clean_text: str
    pii_redacted: bool
    injection_flagged: bool


def sanitize_untrusted_text(raw_text: str) -> SanitizedText:
    """Scrubs PII and flags prompt-injection patterns in UPI remarks or Hinglish chats."""
    text = raw_text.strip()
    injection_flagged = bool(INJECTION_RE.search(text))

    redacted = AADHAAR_RE.sub("[AADHAAR_REDACTED]", text)
    redacted = PAN_RE.sub("[PAN_REDACTED]", redacted)
    redacted = PHONE_RE.sub("[PHONE_REDACTED]", redacted)
    pii_redacted = redacted != text

    return SanitizedText(
        clean_text=redacted,
        pii_redacted=pii_redacted,
        injection_flagged=injection_flagged,
    )
