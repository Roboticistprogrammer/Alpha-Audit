"""organizer.py

Consolidate raw PDF rows into:
1) fills aggregated by Order ID (partial fills merged)
2) trade cycles per symbol (position net-zero slices in time order)

Export JSON suitable for LLM analysis (Streamlit / analyzer).
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_EPS = 1e-8


def _parse_dt(s: str) -> datetime:
    # Binance PDF uses YY-MM-DD HH:MM:SS (e.g. 26-05-10 21:02:27)
    return datetime.strptime(s.strip(), "%y-%m-%d %H:%M:%S")


def load_processed_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if df.empty:
        df["dt"] = pd.Series(dtype="datetime64[ns]")
        return df
    df["dt"] = df["datetime"].map(_parse_dt)
    return df.sort_values("dt").reset_index(drop=True)


def aggregate_order_fills(df: pd.DataFrame) -> pd.DataFrame:
    """Merge partial fills that share the same order_id and symbol."""
    if df.empty:
        return df

    rows: List[Dict[str, Any]] = []
    for (order_id, symbol), group in df.groupby(["order_id", "symbol"], sort=False):
        qty = float(group["quantity"].sum())
        amt = float(group["amount"].sum())
        fee = float(group["fee"].sum())
        rp = float(group["realized_profit"].sum())
        avg_price = amt / qty if qty else float(group["price"].iloc[0])
        ts = pd.Timestamp(group["dt"].min()).to_pydatetime().isoformat()
        rows.append(
            {
                "datetime": ts,
                "symbol": symbol,
                "side": group["side"].iloc[0],
                "quantity": qty,
                "amount": amt,
                "avg_price": avg_price,
                "fee": fee,
                "realized_profit": rp,
                "order_id": order_id,
                "trade_ids": sorted({str(x) for x in group["trade_id"].tolist()}),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("datetime").reset_index(drop=True)


def _signed_qty(row: pd.Series) -> float:
    q = float(row["quantity"])
    return q if row["side"] == "BUY" else -q


def _leg_dict_from_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "datetime": row["datetime"],
        "symbol": row["symbol"],
        "side": row["side"],
        "quantity": float(row["quantity"]),
        "amount": float(row["amount"]),
        "avg_price": float(row["avg_price"]),
        "fee": float(row["fee"]),
        "realized_profit": float(row["realized_profit"]),
        "order_id": row["order_id"],
        "trade_ids": list(row["trade_ids"]) if isinstance(row["trade_ids"], list) else row["trade_ids"],
    }


def trade_cycles_by_symbol(agg: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Split each symbol into consecutive net-zero position segments."""
    cycles_by_symbol: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if agg.empty:
        return cycles_by_symbol

    for sym, g in agg.groupby("symbol", sort=False):
        position = 0.0
        legs: List[Dict[str, Any]] = []
        cycle_idx = 0
        for _, row in g.sort_values("datetime").iterrows():
            legs.append(_leg_dict_from_row(row))
            position += _signed_qty(row)
            if abs(position) < _EPS:
                cycles_by_symbol[sym].append(
                    {
                        "cycle_index": cycle_idx,
                        "symbol": sym,
                        "legs": legs,
                        "total_realized_profit": sum(float(x["realized_profit"]) for x in legs),
                        "total_fee": sum(float(x["fee"]) for x in legs),
                        "start_time": legs[0]["datetime"],
                        "end_time": legs[-1]["datetime"],
                    }
                )
                cycle_idx += 1
                legs = []
        if legs:
            cycles_by_symbol[sym].append(
                {
                    "cycle_index": cycle_idx,
                    "symbol": sym,
                    "legs": legs,
                    "status": "open",
                    "total_realized_profit": sum(float(x["realized_profit"]) for x in legs),
                    "total_fee": sum(float(x["fee"]) for x in legs),
                    "start_time": legs[0]["datetime"],
                    "end_time": legs[-1]["datetime"],
                }
            )
    return cycles_by_symbol


def organize(
    processed_json: str | Path,
    output_path: Optional[str | Path] = None,
) -> str:
    """Build aggregated fills + trade cycles JSON. Returns output path."""
    processed_json = Path(processed_json).resolve()
    df = load_processed_json(processed_json)
    agg = aggregate_order_fills(df)
    cycles = trade_cycles_by_symbol(agg)

    payload = {
        "source": str(processed_json),
        "aggregated_fill_count": int(len(agg)),
        "raw_fill_count": int(len(df)),
        "aggregated_fills": json.loads(
            agg.to_json(orient="records", date_format="iso", default_handler=str)
        ),
        "trade_cycles_by_symbol": {k: v for k, v in cycles.items()},
    }

    if output_path is None:
        output_path = processed_json.parent.parent / "organized" / (processed_json.stem + "_organized.json")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(output_path)


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/processed/journal.json"
    print(organize(src))
