"""Tests for research/stage2/backtest.py: no lookahead in signals, and cost /
compounding arithmetic. Run: python -m pytest research/stage2/test_backtest.py"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402

DATES = pd.bdate_range("2010-01-04", periods=30)


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(DATES)))
    open_ = close * (1 + rng.normal(0, 0.005, len(DATES)))
    df = pd.DataFrame({"open": open_, "close": close}, index=DATES)
    df["prev_close"] = df["close"].shift(1)
    df["oc"] = df["close"] / df["open"] - 1
    df["gap"] = df["open"] / df["prev_close"] - 1
    return df


def _data() -> dict:
    return {"N225": _frame(1), "DAX": _frame(2), "NDX": _frame(3), "VIX": _frame(4) * 0 + 18}


def test_nikkei_signal_is_previous_nasdaq_session():
    data = _data()
    legs = bt.build_legs(data)
    t = DATES[10]
    assert legs["N225"].loc[t, "signal"] == pytest.approx(data["NDX"].loc[DATES[9], "oc"])


def test_changing_today_nasdaq_does_not_change_today_nikkei_signal():
    data = _data()
    before = bt.build_legs(data)["N225"].loc[DATES[10], "signal"]
    data["NDX"].loc[DATES[10], "oc"] = 0.5
    assert bt.build_legs(data)["N225"].loc[DATES[10], "signal"] == before


def test_dax_signal_is_same_day_nikkei():
    data = _data()
    legs = bt.build_legs(data)
    assert legs["DAX"].loc[DATES[5], "signal"] == pytest.approx(data["N225"].loc[DATES[5], "oc"])


def test_nasdaq_signal_ignores_dax_close():
    """The DAX closes after the Nasdaq opens, so its close must not move the signal."""
    data = _data()
    before = bt.build_legs(data)["NDX"].loc[DATES[12], "signal"]
    data["DAX"].loc[DATES[12], ["close", "oc"]] = [1e6, 0.9]
    assert bt.build_legs(data)["NDX"].loc[DATES[12], "signal"] == pytest.approx(before)


def test_nasdaq_signal_is_nikkei_plus_dax_gap():
    data = _data()
    legs = bt.build_legs(data)
    t = DATES[12]
    expected = data["N225"].loc[t, "oc"] + data["DAX"].loc[t, "gap"]
    assert legs["NDX"].loc[t, "signal"] == pytest.approx(expected)


def test_vix_uses_prior_close():
    data = _data()
    data["VIX"]["close"] = np.arange(len(DATES), dtype=float)
    legs = bt.build_legs(data)
    assert legs["DAX"].loc[DATES[8], "vix_prev"] == 7.0


def test_cost_arithmetic():
    leg = pd.DataFrame({"ret": [0.01, 0.01]}, index=DATES[:2])
    pos = pd.Series([1.0, -2.0], index=DATES[:2])
    pnl = bt.leg_pnl(leg, pos, cost_bps=5)
    assert pnl.iloc[0] == pytest.approx(0.01 - 0.0005)
    assert pnl.iloc[1] == pytest.approx(-0.02 - 0.0010)


def test_sequential_compounding():
    a = pd.Series([0.10], index=DATES[:1])
    b = pd.Series([0.10], index=DATES[:1])
    assert bt.combine({"a": a, "b": b}).iloc[0] == pytest.approx(0.21)
    assert bt.combine({"a": a, "b": b}, "parallel").iloc[0] == pytest.approx(0.10)


def test_vix_bands():
    v = pd.Series([10, 15, 19.9, 25, 45, np.nan])
    assert list(bt.vix_scale(v)) == [1.0, 0.75, 0.75, 0.5, 0.25, 1.0]


def test_threshold_and_expansion_rules():
    leg = pd.DataFrame({"signal": [0.001, -0.004, 0.01], "nikkei_up": [True, True, False],
                        "vix_prev": 18.0, "ret": 0.0}, index=DATES[:3])
    assert list(bt.positions(leg, "threshold", "DAX")) == [0.0, -1.0, 1.0]
    assert list(bt.positions(leg, "expansion", "DAX")) == [2.0, -1.0, 1.0]


def test_max_drawdown():
    r = pd.Series([0.10, -0.50, 0.20], index=DATES[:3])
    assert bt.max_drawdown(r) == pytest.approx(-0.5)
