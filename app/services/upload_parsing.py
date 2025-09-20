import io
import json
import csv
from typing import List


def _ensure_list_of_strings(values) -> List[str]:
    if isinstance(values, list):
        items = []
        for v in values:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                items.append(s)
        return items
    return []


def parse_texts_from_upload(filename: str, content: bytes) -> List[str]:
    name = (filename or "").lower()
    text = content.decode("utf-8-sig", errors="ignore") if isinstance(content, (bytes, bytearray)) else str(content)

    # Try JSON when suggested by extension or first non-space char
    tstrip = text.lstrip()
    if name.endswith(".json") or (tstrip.startswith("{") or tstrip.startswith("[")):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "texts" in data:
                return _ensure_list_of_strings(data.get("texts"))
            if isinstance(data, list):
                return _ensure_list_of_strings(data)
        except Exception:
            pass

    # Try CSV (prefer 'text' or 'comment' field)
    if name.endswith(".csv") or "," in text.splitlines()[0:2][0] if text else False:
        try:
            sio = io.StringIO(text)
            reader = csv.DictReader(sio)
            if reader.fieldnames:
                pick = None
                lower = [f.lower() for f in reader.fieldnames]
                for cand in ("text", "comment", "review", "message"):
                    if cand in lower:
                        pick = reader.fieldnames[lower.index(cand)]
                        break
                if pick:
                    return _ensure_list_of_strings([row.get(pick, "") for row in reader])
            # Fallback to first column
            sio.seek(0)
            reader2 = csv.reader(sio)
            return _ensure_list_of_strings([row[0] for row in reader2 if row])
        except Exception:
            pass

    # Plain text: one comment per line
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]
