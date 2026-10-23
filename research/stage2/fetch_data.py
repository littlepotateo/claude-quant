"""Download free daily (and recent hourly) index data from Yahoo Finance and
run a data-quality report.

Usage: python research/stage2/fetch_data.py
Writes research/stage2/data/raw/*.csv and research/stage2/results/data_quality.json
"""

import datetime as dt
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

SYMBOLS = {"N225": "^N225", "DAX": "^GDAXI", "NDX": "^NDX", "VIX": "^VIX"}
# Exchange time zones, used to stamp hourly bars in local session time.
TZ = {"N225": "Asia/Tokyo", "DAX": "Europe/Berlin", "NDX": "America/New_York", "VIX": "America/New_York"}
UA = {"User-Agent": "Mozilla/5.0"}


def _chart(symbol: str, params: str) -> dict:
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?{params}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)["chart"]["result"][0]
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))


def _frame(res: dict, tz: str) -> pd.DataFrame:
    q = res["indicators"]["quote"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz)
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=idx)
    return df.dropna(subset=["open", "close"])


def fetch_daily(name: str) -> pd.DataFrame:
    res = _chart(SYMBOLS[name], f"period1=946684800&period2={int(time.time())}&interval=1d")
    df = _frame(res, TZ[name])
    df.index = df.index.date
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")]


def fetch_hourly(name: str) -> pd.DataFrame:
    df = _frame(_chart(SYMBOLS[name], "range=730d&interval=60m"), TZ[name])
    df.index.name = "time"
    return df


def quality(name: str, df: pd.DataFrame) -> dict:
    prev_close = df["close"].shift(1)
    flat_open = (df["open"] - prev_close).abs() < 1e-9
    oc = df["close"] / df["open"] - 1
    # Defective-open stretches: rolling share of days where open == prior close.
    roll = flat_open.astype(float).rolling(60, min_periods=60).mean()
    bad = roll[roll > 0.2]
    clean_from = None
    if len(bad):
        clean_from = str(bad.index[-1])
    return {
        "rows": int(len(df)),
        "first": str(df.index[0]),
        "last": str(df.index[-1]),
        "open_equals_prev_close": int(flat_open.sum()),
        "open_to_close_moves_over_8pct": int((oc.abs() > 0.08).sum()),
        "zero_range_days": int(((df["high"] - df["low"]).abs() < 1e-9).sum()),
        "defective_open_stretch_ends": clean_from,
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report = {"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "daily": {}, "hourly": {}}
    for name in SYMBOLS:
        d = fetch_daily(name)
        d.to_csv(RAW / f"{name}_daily.csv")
        report["daily"][name] = quality(name, d)
        if name != "VIX":
            h = fetch_hourly(name)
            h.to_csv(RAW / f"{name}_hourly.csv")
            report["hourly"][name] = {"rows": int(len(h)), "first": str(h.index[0]), "last": str(h.index[-1])}
        time.sleep(1)
    # Latest FX for converting micro-contract P&L into SGD (held constant in the simulation).
    fx = {}
    for key, sym in {"EURUSD": "EURUSD=X", "USDJPY": "JPY=X", "USDSGD": "SGD=X"}.items():
        fx[key] = float(_chart(sym, "range=5d&interval=1d")["meta"]["regularMarketPrice"])
    report["fx"] = fx
    (RESULTS / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    import urllib.parse  # noqa: F401  (used by _chart)
    main()
