"""Streamlit dashboard for Alpha-Audit analysis output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


def _load_analysis(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _mistakes_df(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for block in results:
        sym = block.get("symbol", "?")
        if block.get("error"):
            rows.append(
                {
                    "symbol": sym,
                    "severity": "high",
                    "title": "Analysis error",
                    "guideline": "",
                    "evidence": block.get("error"),
                    "suggestion": "Retry with a valid API key or smaller payload.",
                }
            )
            continue
        for m in block.get("mistakes") or []:
            rows.append(
                {
                    "symbol": sym,
                    "severity": m.get("severity", ""),
                    "title": m.get("title", ""),
                    "guideline": m.get("guideline", ""),
                    "evidence": m.get("evidence", ""),
                    "suggestion": m.get("suggestion", ""),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="Alpha-Audit", layout="wide")
    st.title("Alpha-Audit")
    st.caption("Trading journal review from organized cycles + OpenRouter analysis.")

    default_glob = "data/analysis/*_analysis.json"
    path_str = st.text_input(
        "Analysis JSON path",
        value="data/analysis/journal_organized_analysis.json",
        help="Output file produced by src/analyzer.py",
    )
    path = Path(path_str).expanduser()
    if not path.is_file():
        st.warning(f"File not found: {path}. Run the extractor, organizer, and analyzer first.")
        st.code(
            "python src/extractor.py data/raw/journal.pdf\n"
            "python utils/organizer.py\n"
            "export OPENROUTER_API_KEY=...\n"
            "python src/analyzer.py",
            language="bash",
        )
        return

    data = _load_analysis(path)
    st.subheader("Run metadata")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", data.get("model", "—"))
    c2.metric("Generated (UTC)", data.get("generated_at", "—")[:19])
    c3.metric("Symbols reviewed", len(data.get("results") or []))

    st.divider()
    results = data.get("results") or []

    st.subheader("Summaries")
    for block in results:
        sym = block.get("symbol", "?")
        with st.expander(f"{sym}", expanded=False):
            st.write(block.get("summary", "—"))
            if block.get("notes"):
                st.caption(block["notes"])

    st.subheader("Mistakes")
    df = _mistakes_df(results)
    if df.empty:
        st.info("No mistakes listed (or analysis blocks empty).")
    else:
        severity_order = {"high": 0, "medium": 1, "low": 2}
        df["_sev"] = df["severity"].str.lower().map(lambda s: severity_order.get(s, 9))
        df = df.sort_values("_sev").drop(columns=["_sev"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Severity counts")
        counts = df.groupby("severity").size().reset_index(name="count")
        st.bar_chart(counts.set_index("severity"))


if __name__ == "__main__":
    main()
