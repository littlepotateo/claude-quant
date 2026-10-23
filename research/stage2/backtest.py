"""Stage 2 backtest: does the Nikkei -> DAX -> Nasdaq session lead-lag idea
from claude-hedge survive real index data and trading costs?

Every rule, threshold and pass criterion here was fixed before results were
looked at (see research/stage2/README.md). Session returns are cash-index
open-to-close, used as a proxy for the futures sessions.

Usage: python research/stage2/backtest.py   (after fetch_data.py)
"""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"

START = pd.Timestamp("2001-01-01")  # NDX opens are defective before late 2000
PERIODS = {
    "in_sample": (START, pd.Timestamp("2016-12-31")),
    "out_of_sample": (pd.Timestamp("2017-01-01"), pd.Timestamp("2024-12-31")),
    "hold_out": (pd.Timestamp("2025-01-01"), pd.Timestamp("2100-01-01")),
}
COSTS_BPS = [0, 2, 5, 10]  # round trip per session trade
BASE_COST_BPS = 5
THRESHOLD = 0.0025
VIX_BANDS = [(15, 1.0), (20, 0.75), (30, 0.5), (np.inf, 0.25)]
LEGS = ["N225", "DAX", "NDX"]
# Micro contracts: multiplier per index point, currency.
CONTRACTS = {
    "NDX": ("MNQ (CME)", 2.0, "USD"),
    "DAX": ("Micro-DAX FDXS (Eurex)", 1.0, "EUR"),
    "N225": ("Nikkei 225 Micro (OSE)", 10.0, "JPY"),
}
ACCOUNT_SGD = 5000.0
SEED = 7


# ---------------------------------------------------------------- data

def load_daily(name: str, fetch_date: pd.Timestamp) -> pd.DataFrame:
    df = pd.read_csv(RAW / f"{name}_daily.csv", parse_dates=["date"]).set_index("date").sort_index()
    df = df[df.index < fetch_date]  # drop a session that may still be trading
    df["prev_close"] = df["close"].shift(1)
    df["oc"] = df["close"] / df["open"] - 1
    df["gap"] = df["open"] / df["prev_close"] - 1
    # Yahoo sometimes reports open == prior close (no real open): no trade that day.
    bad_open = (df["open"] - df["prev_close"]).abs() < 1e-9
    df.loc[bad_open, ["oc", "gap"]] = np.nan
    return df


def _asof_before(target: pd.DatetimeIndex, src: pd.Series, max_days: int = 5) -> pd.Series:
    """Value of src on the latest date strictly before each target date."""
    s = src.dropna()
    left = pd.DataFrame({"date": target})
    right = pd.DataFrame({"src_date": s.index, "val": s.values})
    m = pd.merge_asof(left, right, left_on="date", right_on="src_date",
                      allow_exact_matches=False, direction="backward",
                      tolerance=pd.Timedelta(days=max_days))
    return pd.Series(m["val"].values, index=target)


def build_legs(data: dict) -> dict:
    """One frame per traded session: trade return, signal known before the open,
    prior VIX close and whether the Nikkei session that day closed up."""
    n, d, q, v = data["N225"], data["DAX"], data["NDX"], data["VIX"]
    legs = {}

    # Nikkei day t: signal = Nasdaq session that closed before Tokyo opened.
    f = pd.DataFrame({"ret": n["oc"]})
    f["signal"] = _asof_before(f.index, q["oc"])
    legs["N225"] = f

    # DAX day t: signal = Nikkei session the same date (closes before Frankfurt opens).
    f = pd.DataFrame({"ret": d["oc"]})
    f["signal"] = n["oc"].reindex(f.index)
    legs["DAX"] = f

    # Nasdaq day t: Nikkei same date + DAX overnight gap. The DAX close is after the
    # Nasdaq open, so the full DAX session would be lookahead.
    f = pd.DataFrame({"ret": q["oc"]})
    nik, gap = n["oc"].reindex(f.index), d["gap"].reindex(f.index)
    f["signal"] = nik.fillna(0) + gap.fillna(0)
    f.loc[nik.isna() & gap.isna(), "signal"] = np.nan
    legs["NDX"] = f

    for name, f in legs.items():
        f["vix_prev"] = _asof_before(f.index, v["close"])
        f["nikkei_up"] = (n["oc"].reindex(f.index) > 0) if name != "N225" else False
        legs[name] = f[(f.index >= START) & f["ret"].notna()]
    return legs


# ---------------------------------------------------------------- rules

def vix_scale(vix: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=vix.index)
    lo = -np.inf
    for hi, mult in VIX_BANDS:
        out[(vix >= lo) & (vix < hi)] = mult
        lo = hi
    return out.fillna(1.0)


def positions(leg: pd.DataFrame, rule: str, name: str) -> pd.Series:
    sig = leg["signal"]
    if rule == "always_long":
        return pd.Series(1.0, index=leg.index)
    pos = np.sign(sig).fillna(0.0)
    if rule == "threshold":
        pos = pos.where(sig.abs() > THRESHOLD, 0.0)
    elif rule == "vix":
        pos = pos * vix_scale(leg["vix_prev"])
    elif rule == "expansion":
        if name != "N225":
            pos = pos.where(~((pos > 0) & leg["nikkei_up"]), pos * 2)
    elif rule != "lead_lag":
        raise ValueError(rule)
    return pos


def leg_pnl(leg: pd.DataFrame, pos: pd.Series, cost_bps: float) -> pd.Series:
    return pos * leg["ret"] - pos.abs() * cost_bps / 1e4


def combine(pnls: dict, how: str = "sequential") -> pd.Series:
    df = pd.DataFrame(pnls)
    if how == "sequential":  # full equity rolls through each session in turn
        return (1 + df.fillna(0)).prod(axis=1) - 1
    return df.fillna(0).sum(axis=1) / len(pnls)  # capital split evenly


# ---------------------------------------------------------------- metrics

def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def metrics(r: pd.Series, traded: pd.Series | None = None) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 20:
        return {"days": n}
    years = max((r.index[-1] - r.index[0]).days / 365.25, 1 / 252)
    ppy = n / years
    sd = r.std()
    out = {
        "days": n,
        "total_return": float((1 + r).prod() - 1),
        "ann_return": float((1 + r).prod() ** (1 / years) - 1),
        "ann_vol": float(sd * np.sqrt(ppy)),
        "sharpe": float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0,
        "t_stat": float(r.mean() / sd * np.sqrt(n)) if sd > 0 else 0.0,
        "max_drawdown": max_drawdown(r),
    }
    t = r[traded.reindex(r.index).fillna(False)] if traded is not None else r[r != 0]
    out["trades"] = int(len(t))
    out["hit_rate"] = float((t > 0).mean()) if len(t) else float("nan")
    return out


def slice_period(s: pd.Series, period: str) -> pd.Series:
    a, b = PERIODS[period]
    return s[(s.index >= a) & (s.index <= b)]


def breakeven_bps(legs: dict, pos: dict, period: str) -> float:
    gross = sum(slice_period(pos[k] * legs[k]["ret"], period).sum() for k in pos)
    turnover = sum(slice_period(pos[k].abs(), period).sum() for k in pos)
    return float(gross / turnover * 1e4) if turnover else float("nan")


# ---------------------------------------------------------------- experiments

def run_rule(legs: dict, rule: str, which: list, cost_bps: float, how: str = "sequential"):
    pos = {k: positions(legs[k], rule, k) for k in which}
    pnl = {k: leg_pnl(legs[k], pos[k], cost_bps) for k in which}
    r = pnl[which[0]] if len(which) == 1 else combine(pnl, how)
    traded = pd.DataFrame({k: pos[k] != 0 for k in which}).any(axis=1)
    return r, pos, traded


def random_null(legs: dict, which: list, period: str, cost_bps: float, sims: int = 1000) -> np.ndarray:
    """Sharpe of random +/-1 positions on the same sessions the lead-lag rule trades."""
    rng = np.random.default_rng(SEED)
    base = {k: positions(legs[k], "lead_lag", k) for k in which}
    out = np.empty(sims)
    for i in range(sims):
        pnl = {}
        for k in which:
            p = base[k].abs() * rng.choice([-1.0, 1.0], size=len(base[k]))
            pnl[k] = leg_pnl(legs[k], pd.Series(p, index=base[k].index), cost_bps)
        r = pnl[which[0]] if len(which) == 1 else combine(pnl)
        out[i] = metrics(slice_period(r, period))["sharpe"]
    return out


def bootstrap(r: pd.Series, sims: int = 2000) -> dict:
    rng = np.random.default_rng(SEED)
    x = r.dropna().values
    years = (r.index[-1] - r.index[0]).days / 365.25
    ppy = len(x) / years
    sh, dd = np.empty(sims), np.empty(sims)
    for i in range(sims):
        s = rng.choice(x, size=len(x), replace=True)
        sh[i] = s.mean() / s.std() * np.sqrt(ppy)
        eq = np.cumprod(1 + s)
        dd[i] = (eq / np.maximum.accumulate(eq) - 1).min()
    return {"sharpe_p2_5": float(np.percentile(sh, 2.5)), "sharpe_p50": float(np.percentile(sh, 50)),
            "sharpe_p97_5": float(np.percentile(sh, 97.5)), "max_dd_p50": float(np.percentile(dd, 50)),
            "max_dd_p5": float(np.percentile(dd, 5))}


def var_cvar(r: pd.Series, confidence: float = 0.95) -> dict:
    """Same definitions as RiskCalculator.calculate_var / calculate_cvar in the
    claude-hedge repo, without scipy."""
    r = r.dropna()
    z = NormalDist().inv_cdf(1 - confidence)
    thr = np.percentile(r, (1 - confidence) * 100)
    return {"var_parametric": float(r.mean() + z * r.std()), "cvar_historical": float(r[r <= thr].mean())}


def hourly_crosscheck(daily_legs: dict) -> dict:
    """Recompute the Nasdaq leg with true session timing from 60-minute bars:
    signal = Nikkei session + DAX from its open to 14:00 Berlin (always before the
    Nasdaq open). Compare with the daily proxy on the same dates."""
    def hourly(name, tz):
        h = pd.read_csv(RAW / f"{name}_hourly.csv")
        h["time"] = pd.to_datetime(h["time"], utc=True).dt.tz_convert(tz)
        h["date"] = h["time"].dt.tz_localize(None).dt.normalize()
        return h

    n = hourly("N225", "Asia/Tokyo")
    nik = n.groupby("date").apply(lambda g: g["close"].iloc[-1] / g["open"].iloc[0] - 1, include_groups=False)
    d = hourly("DAX", "Europe/Berlin")
    early = d[d["time"].dt.hour < 14]  # bars starting 09:00..13:00, last one ends 14:00
    dax_am = early.groupby("date").apply(lambda g: g["close"].iloc[-1] / g["open"].iloc[0] - 1, include_groups=False)

    leg = daily_legs["NDX"]
    dates = leg.index[leg.index.isin(dax_am.index)]
    sig_h = nik.reindex(dates).fillna(0) + dax_am.reindex(dates)
    pos_h = np.sign(sig_h).fillna(0)
    pos_d = positions(leg, "lead_lag", "NDX").reindex(dates)
    ret = leg["ret"].reindex(dates)
    r_h = pos_h * ret - pos_h.abs() * BASE_COST_BPS / 1e4
    r_d = pos_d * ret - pos_d.abs() * BASE_COST_BPS / 1e4
    return {
        "first": str(dates[0].date()), "last": str(dates[-1].date()), "days": int(len(dates)),
        "position_agreement": float((pos_h == pos_d).mean()),
        "hourly_timing": metrics(r_h), "daily_proxy": metrics(r_d),
    }


def small_account(data: dict, legs: dict, fx: dict, start: str = "2017-01-01",
                  cost_bps: float = BASE_COST_BPS) -> tuple[pd.DataFrame, dict]:
    """One whole micro contract per session in the lead-lag direction, on a S$5,000
    account. P&L in index points x multiplier, converted at today's FX (held fixed)."""
    to_usd = {"USD": 1.0, "EUR": fx["EURUSD"], "JPY": 1 / fx["USDJPY"]}
    usd_to_sgd = fx["USDSGD"]
    rows, eq = [], ACCOUNT_SGD
    events = []
    for k in LEGS:
        name, mult, ccy = CONTRACTS[k]
        px = data[k].reindex(legs[k].index)
        pos = positions(legs[k], "lead_lag", k)
        notional_sgd = px["open"] * mult * to_usd[ccy] * usd_to_sgd
        pnl_sgd = pos * (px["close"] - px["open"]) * mult * to_usd[ccy] * usd_to_sgd
        cost_sgd = pos.abs() * notional_sgd * cost_bps / 1e4
        events.append(pd.DataFrame({"leg": k, "pnl": pnl_sgd - cost_sgd, "notional": notional_sgd, "pos": pos}))
    ev = pd.concat(events).sort_index()
    ev = ev[ev.index >= pd.Timestamp(start)]
    order = {"N225": 0, "DAX": 1, "NDX": 2}
    ev = ev.assign(o=ev["leg"].map(order)).sort_values(["o"]).sort_index(kind="stable")
    ruined_on = None
    peak, worst = eq, 0.0
    start_lev = {k: float(ev.loc[ev["leg"] == k, "notional"].iloc[0] / ACCOUNT_SGD) for k in LEGS}
    for t, row in ev.iterrows():
        if row["pos"] == 0:
            continue
        if eq <= 0:
            ruined_on = str(t.date())
            break
        eq += row["pnl"]
        peak = max(peak, eq)
        worst = min(worst, eq / peak - 1)
        rows.append((t, eq))
    curve = pd.DataFrame(rows, columns=["date", "equity_sgd"]).groupby("date").last()
    latest = {k: float(data[k]["close"].iloc[-1] * CONTRACTS[k][1] * to_usd[CONTRACTS[k][2]] * usd_to_sgd) for k in LEGS}
    summary = {
        "start_sgd": ACCOUNT_SGD, "end_sgd": float(eq), "min_sgd": float(curve["equity_sgd"].min()),
        "max_drawdown": float(worst), "ruined_on": ruined_on,
        "notional_per_contract_today_sgd": latest,
        "leverage_today": {k: latest[k] / ACCOUNT_SGD for k in LEGS},
        "leverage_at_start": start_lev, "start": start, "cost_bps": cost_bps, "fx": fx,
        "contracts": {k: CONTRACTS[k][0] for k in LEGS},
    }
    return curve, summary


# ---------------------------------------------------------------- main

def main() -> None:
    dq = json.loads((RESULTS / "data_quality.json").read_text())
    fetch_date = pd.Timestamp(dq["fetched_utc"][:10])
    data = {k: load_daily(k, fetch_date) for k in LEGS + ["VIX"]}
    legs = build_legs(data)

    tests = [(rule, which) for rule in ["lead_lag", "threshold", "vix"]
             for which in (["N225"], ["DAX"], ["NDX"], LEGS)] + [("expansion", LEGS)]
    n_tests = len(tests)
    z_crit = NormalDist().inv_cdf(1 - 0.05 / (2 * n_tests))

    results = []
    for rule, which in tests:
        label = f"{rule}:{'+'.join(which) if len(which) == 1 else 'combined'}"
        row = {"id": label, "rule": rule, "legs": which, "by_period": {}, "cost_sensitivity": {}}
        for period in PERIODS:
            r, pos, traded = run_rule(legs, rule, which, BASE_COST_BPS)
            row["by_period"][period] = metrics(slice_period(r, period), traded)
            row["by_period"][period]["breakeven_bps"] = breakeven_bps(legs, pos, period)
        for c in COSTS_BPS:
            r, _, traded = run_rule(legs, rule, which, c)
            row["cost_sensitivity"][str(c)] = metrics(slice_period(r, "out_of_sample"), traded)["sharpe"]
        results.append(row)

    # Baselines
    baselines = {}
    for k in LEGS:
        r, _, tr = run_rule(legs, "always_long", [k], BASE_COST_BPS)
        r0, _, _ = run_rule(legs, "always_long", [k], 0)
        bh = data[k]["close"].pct_change()
        bh = bh[bh.index >= START]
        baselines[k] = {p: {"always_long_oc": metrics(slice_period(r, p), tr),
                            "always_long_oc_gross": metrics(slice_period(r0, p), tr),
                            "buy_hold": metrics(slice_period(bh, p))} for p in PERIODS}
    r_al, _, tr_al = run_rule(legs, "always_long", LEGS, BASE_COST_BPS)
    baselines["combined_always_long"] = {p: metrics(slice_period(r_al, p), tr_al) for p in PERIODS}
    nulls = {"+".join(w): random_null(legs, w, "out_of_sample", BASE_COST_BPS) for w in (["N225"], ["DAX"], ["NDX"], LEGS)}
    null = nulls["+".join(LEGS)]

    # Pass/fail, stated in the README before running
    for row in results:
        ins, oos, ho = (row["by_period"][p] for p in ("in_sample", "out_of_sample", "hold_out"))
        if len(row["legs"]) == 1:
            al = baselines[row["legs"][0]]["out_of_sample"]["always_long_oc"]["sharpe"]
        else:
            al = baselines["combined_always_long"]["out_of_sample"]["sharpe"]
        checks = {
            "oos_sharpe_over_0_5": oos["sharpe"] > 0.5,
            "beats_always_long_oos": oos["sharpe"] > al,
            "beats_random_95pct_oos": oos["sharpe"] > float(np.percentile(nulls["+".join(row["legs"])], 95)),
            "significant_in_sample_bonferroni": abs(ins["t_stat"]) > z_crit and ins["t_stat"] > 0,
            "holds_in_hold_out": ho.get("sharpe", -1) > 0,
        }
        row["checks"] = checks
        row["survives"] = all(checks.values())

    # Sequential vs parallel: is "sequential compounding" more than leverage?
    seq = {}
    for how in ("sequential", "parallel"):
        r, _, _ = run_rule(legs, "always_long", LEGS, BASE_COST_BPS, how)
        seq[how] = metrics(slice_period(r, "out_of_sample"))
    r_par, _, _ = run_rule(legs, "always_long", LEGS, BASE_COST_BPS, "parallel")
    seq["parallel_x3"] = metrics(slice_period(3 * r_par, "out_of_sample"))

    # Main rule detail
    r_main, pos_main, tr_main = run_rule(legs, "lead_lag", LEGS, BASE_COST_BPS)
    oos_main = slice_period(r_main, "out_of_sample")
    yearly = r_main.groupby(r_main.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly_al = r_al.groupby(r_al.index.year).apply(lambda x: (1 + x).prod() - 1)

    curve, acct = small_account(data, legs, dq["fx"])
    curve0, acct0 = small_account(data, legs, dq["fx"], cost_bps=0)
    acct["no_costs"] = {k: acct0[k] for k in ("end_sgd", "min_sgd", "max_drawdown", "ruined_on")}

    # What lookahead would look like: Nasdaq signal using the full same-day DAX session,
    # which closes after the Nasdaq opens. Not tradable; shown to make the timing point.
    cheat = legs["NDX"].copy()
    cheat["signal"] = data["N225"]["oc"].reindex(cheat.index).fillna(0) + \
        (data["DAX"]["close"] / data["DAX"]["prev_close"] - 1).reindex(cheat.index).fillna(0)
    lookahead_demo = {p: metrics(slice_period(leg_pnl(cheat, positions(cheat, "lead_lag", "NDX"), BASE_COST_BPS), p))["sharpe"]
                      for p in PERIODS}

    out = {
        "generated_from": dq["fetched_utc"],
        "sample": {k: [str(v.index[0].date()), str(v.index[-1].date()), int(len(v))] for k, v in legs.items()},
        "periods": {k: [str(a.date()), str(min(b, max(v.index[-1] for v in legs.values())).date())] for k, (a, b) in PERIODS.items()},
        "n_tests": n_tests, "bonferroni_z": z_crit, "base_cost_bps": BASE_COST_BPS,
        "tests": results, "baselines": baselines,
        "random_null_oos_sharpe": {k: {"p5": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                                       "p95": float(np.percentile(v, 95))} for k, v in nulls.items()},
        "sequential_vs_parallel_oos": seq,
        "main_rule_bootstrap_oos": bootstrap(oos_main),
        "main_rule_var_cvar_oos": var_cvar(oos_main),
        "hourly_crosscheck_ndx": hourly_crosscheck(legs),
        "small_account": acct,
        "lookahead_demo_ndx_sharpe": lookahead_demo,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    eq = pd.DataFrame({
        "lead_lag": (1 + r_main).cumprod(),
        "threshold": (1 + run_rule(legs, "threshold", LEGS, BASE_COST_BPS)[0]).cumprod(),
        "vix": (1 + run_rule(legs, "vix", LEGS, BASE_COST_BPS)[0]).cumprod(),
        "expansion": (1 + run_rule(legs, "expansion", LEGS, BASE_COST_BPS)[0]).cumprod(),
        "always_long": (1 + r_al).cumprod(),
    })
    eq.to_csv(RESULTS / "equity_combined_5bps.csv", float_format="%.6f")
    pd.DataFrame({"lead_lag": yearly, "always_long": yearly_al}).to_csv(RESULTS / "yearly_combined_5bps.csv", float_format="%.6f")
    curve.to_csv(RESULTS / "small_account_sgd.csv", float_format="%.2f")

    print(f"tests={n_tests} bonferroni z={z_crit:.2f}  random-null OOS Sharpe p95: "
          + ", ".join(f"{k}={v['p95']:.2f}" for k, v in out["random_null_oos_sharpe"].items()))
    for row in results:
        b = row["by_period"]
        print(f"{row['id']:<22} IS {b['in_sample']['sharpe']:+.2f} (t {b['in_sample']['t_stat']:+.1f})  "
              f"OOS {b['out_of_sample']['sharpe']:+.2f}  HO {b['hold_out'].get('sharpe', float('nan')):+.2f}  "
              f"BE {b['out_of_sample']['breakeven_bps']:+.1f}bp  survives={row['survives']}")


if __name__ == "__main__":
    main()
