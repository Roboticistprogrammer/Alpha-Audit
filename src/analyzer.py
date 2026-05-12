"""analyzer.py

Call OpenRouter chat completions to audit organized trade cycles against
generic risk and execution guidelines. Writes structured JSON for Streamlit.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_GUIDELINES = textwrap.dedent(
    """\
    Use these execution and risk-management norms when judging behavior:
    - Define risk before entry: position size vs account risk, stop placement, invalidation.
    - Avoid averaging down into a losing thesis without a new, written setup trigger.
    - Prefer scaling out at planned targets rather than ad-hoc full reversals.
    - Watch fee drag vs edge on small or frequent churning trades.
    - Avoid oversized leverage relative to planned stop distance.
    - Separate planned trades from impulsive / revenge-style sequences after losses.
    - Note excessive concentration in one symbol or correlated themes.
    - Flag unclear exit logic (no stop, widened stop after entry, moving targets only).
    """
).strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Parse first JSON object from model output (handles markdown fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(text[start : end + 1])


def _compact_cycles(symbol: str, cycles: List[Dict[str, Any]], max_cycles: int = 12) -> str:
    lines: List[str] = []
    for c in cycles[:max_cycles]:
        status = c.get("status", "closed")
        legs = c.get("legs") or []
        summary_parts = []
        for lg in legs:
            summary_parts.append(
                f"{lg.get('side')} q={lg.get('quantity')} @ {lg.get('avg_price')} "
                f"rp={lg.get('realized_profit')} fee={lg.get('fee')} t={lg.get('datetime')}"
            )
        lines.append(
            f"- cycle #{c.get('cycle_index')} [{status}] "
            f"rp_total={c.get('total_realized_profit')} fee={c.get('total_fee')} "
            f"{c.get('start_time')} → {c.get('end_time')}\n  "
            + "\n  ".join(summary_parts)
        )
    if len(cycles) > max_cycles:
        lines.append(f"... ({len(cycles) - max_cycles} more cycles omitted)")
    return "\n".join(lines)


def _call_openrouter(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    timeout: float = 120.0,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional attribution for OpenRouter rankings
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "alpha-audit"),
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(OPENROUTER_URL, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected OpenRouter payload: {data!r}") from e


def analyze_symbol(
    symbol: str,
    cycles: List[Dict[str, Any]],
    *,
    model: str,
    api_key: str,
) -> Dict[str, Any]:
    system = (
        "You are a trading execution reviewer. Answer ONLY with valid JSON, no prose outside JSON. "
        "Ground every mistake in the supplied cycle data (reference timestamps or legs)."
    )
    user = textwrap.dedent(
        f"""\
        Symbol: {symbol}

        Guidelines:
        {DEFAULT_GUIDELINES}

        Organized trade cycles (aggregated partial fills, net-zero chunks where possible):
        {_compact_cycles(symbol, cycles)}

        Return JSON with this shape:
        {{
          "symbol": "{symbol}",
          "summary": "one paragraph",
          "mistakes": [
            {{
              "title": "short label",
              "severity": "low|medium|high",
              "guideline": "which norm was breached",
              "evidence": "quote cycle index / times / sides",
              "suggestion": "concrete fix"
            }}
          ],
          "notes": "optional extra context"
        }}
        """
    )
    raw = _call_openrouter(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model,
        api_key=api_key,
    )
    parsed = _extract_json_object(raw)
    parsed["symbol"] = symbol
    return parsed


def analyze_organized_file(
    organized_json: str | Path,
    output_path: Optional[str | Path] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    delay_s: float = 0.4,
) -> str:
    """Run analysis per symbol; write combined JSON. Returns output path."""
    organized_json = Path(organized_json).resolve()
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY in the environment or .env")

    model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    with organized_json.open(encoding="utf-8") as f:
        payload = json.load(f)

    cycles_map: Dict[str, List[Dict[str, Any]]] = payload.get("trade_cycles_by_symbol") or {}
    results: List[Dict[str, Any]] = []

    for symbol in sorted(cycles_map.keys()):
        cycles = cycles_map[symbol] or []
        if not cycles:
            continue
        for attempt in range(3):
            try:
                results.append(analyze_symbol(symbol, cycles, model=model, api_key=api_key))
                break
            except Exception as e:
                if attempt == 2:
                    results.append(
                        {
                            "symbol": symbol,
                            "error": str(e),
                            "mistakes": [],
                            "summary": "Analysis failed for this symbol.",
                        }
                    )
                else:
                    time.sleep(1.5 * (attempt + 1))
        time.sleep(delay_s)

    out: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source_organized": str(organized_json),
        "guidelines_version": "default_v1",
        "results": results,
    }

    if output_path is None:
        output_path = organized_json.parent.parent / "analysis" / (organized_json.stem + "_analysis.json")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze organized trades via OpenRouter")
    parser.add_argument(
        "organized",
        nargs="?",
        default="data/organized/journal_organized.json",
        help="Path to *_organized.json from utils/organizer.py",
    )
    parser.add_argument("-o", "--output", default=None, help="Output analysis JSON path")
    parser.add_argument("-m", "--model", default=None, help="OpenRouter model id")
    args = parser.parse_args()

    path = analyze_organized_file(args.organized, args.output, model=args.model)
    print(path)
