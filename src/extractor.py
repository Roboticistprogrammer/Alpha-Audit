"""extractor.py

Read Binance-style futures trade-history PDFs and write normalized JSON rows.

Rows omit personal identifiers (ledger uid, buyer/maker flags). Lines are merged
to recover Trade / Order IDs split across PDF line wraps.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

_ROW_START = re.compile(r"^\d{8}\s+\d{2}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+")
_SKIP_LINE_PREFIXES = (
    "www.binance",
    "Page ",
    "Futures Trade",
    "Uid Time Symbol",
)
_SKIP_LINE_STARTSWITH = ("Name:", "Address:", "User ID:", "Email:", "Period(")


def _merge_wrapped_lines(text: str) -> List[str]:
    """Join continuation lines so each logical trade row starts with uid + datetime."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    merged: List[str] = []
    buf: Optional[str] = None
    for ln in lines:
        if ln.startswith(_SKIP_LINE_PREFIXES):
            continue
        if any(ln.startswith(p) for p in _SKIP_LINE_STARTSWITH):
            continue
        if _ROW_START.match(ln):
            if buf is not None:
                merged.append(buf)
            buf = ln
        elif buf is not None:
            buf = f"{buf} {ln}"
    if buf is not None:
        merged.append(buf)
    return merged


def _normalize_fee_usdt(line: str) -> str:
    return re.sub(r"([\d.]+)USDT\b", r"\1 USDT", line)


def parse_trade_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse one merged trade row into a dict. Returns None if the line is junk."""
    line = _normalize_fee_usdt(line)
    parts = line.split()
    try:
        usdt_idx = parts.index("USDT")
    except ValueError:
        return None
    if len(parts) < usdt_idx + 6:
        return None

    symbol = parts[3]
    side = parts[4]
    if side not in ("BUY", "SELL"):
        return None

    fee = float(parts[usdt_idx - 1])
    realized_profit = float(parts[usdt_idx + 1])
    rest = parts[usdt_idx + 4 :]
    if len(rest) < 2:
        return None

    order_id = rest[-1]
    trade_id = "".join(rest[:-1])

    mid = parts[5:usdt_idx - 1]
    if len(mid) < 3:
        return None
    price = float(mid[0])
    quantity = float(mid[1])
    amount = float(mid[2])

    date_token, time_token = parts[1], parts[2]
    dt_str = f"{date_token} {time_token}"

    return {
        "datetime": dt_str,
        "symbol": symbol,
        "side": side,
        "price": price,
        "quantity": quantity,
        "amount": amount,
        "fee": fee,
        "realized_profit": realized_profit,
        "trade_id": trade_id,
        "order_id": order_id,
    }


def _load_pdf(path: Path):
    return pdfplumber.open(path)


def extract_pdf(pdf_path: str, output_dir: str = "data/processed") -> str:
    """Extract trading rows from *pdf_path* and write JSON (no name/address rows).

    Parameters
    ----------
    pdf_path: str
        Path to the source PDF.
    output_dir: str
        Directory for the JSON file (created if missing).

    Returns
    -------
    str
        Absolute path to the written JSON file.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    with _load_pdf(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in _merge_wrapped_lines(text):
                parsed = parse_trade_line(raw)
                if parsed is not None:
                    rows.append(parsed)

    json_path = out_dir / (pdf_path.stem + ".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    return str(json_path)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/raw/journal.pdf"
    out = extract_pdf(src)
    print(out)
