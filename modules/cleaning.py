import re
from datetime import datetime


def clean_num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "").replace("₹", "").replace("%", "")
    if s in {"", "-", "--", "na", "n/a", "None", "N.A."}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def clean_text(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s and s not in {"-", "--"} else None


def slugify(text: str) -> str:
    text = clean_text(text) or ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def parse_date_safe(v):
    if not v:
        return None
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).date().isoformat()
        except Exception:
            pass
    try:
        return datetime.fromisoformat(str(v).strip()).date().isoformat()
    except Exception:
        return None
