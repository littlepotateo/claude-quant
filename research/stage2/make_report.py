"""Build research/stage2/report.html from results/*.json and *.csv.

Usage: python research/stage2/make_report.py   (after backtest.py)
"""

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"

m = json.loads((RES / "metrics.json").read_text())
eq = pd.read_csv(RES / "equity_combined_5bps.csv", parse_dates=["date"]).set_index("date")
yearly = pd.read_csv(RES / "yearly_combined_5bps.csv").set_index("date")
acct_curve = pd.read_csv(RES / "small_account_sgd.csv", parse_dates=["date"]).set_index("date")
fetch_date = pd.Timestamp(m["generated_from"][:10])

NAMES = {"N225": "Nikkei 225", "DAX": "DAX", "NDX": "Nasdaq 100"}
RULES = {"lead_lag": "Lead-lag", "threshold": "Lead-lag + 0.25% threshold",
         "vix": "Lead-lag + VIX scaling", "expansion": "Conditional expansion"}
CHECK_LABELS = {
    "oos_sharpe_over_0_5": "Out-of-sample Sharpe > 0.5",
    "beats_always_long_oos": "Beats always-long",
    "beats_random_95pct_oos": "Beats 95% of random",
    "significant_in_sample_bonferroni": "Significant in-sample",
    "holds_in_hold_out": "Positive in hold-out",
}


def pct(x, d=1):
    return f"{x * 100:+.{d}f}%" if x is not None and not (isinstance(x, float) and math.isnan(x)) else "–"


def num(x, d=2):
    return f"{x:+.{d}f}"


def sgd(x):
    return f"S${x:,.0f}" if x >= 0 else f"−S${-x:,.0f}"


def cls(x):
    return "neg" if x < 0 else "pos"


# ---------------------------------------------------------------- derived numbers

def intraday_overnight():
    rows = []
    for k in bt.LEGS:
        d = bt.load_daily(k, fetch_date)
        d = d[d.index >= bt.START].dropna(subset=["oc", "gap"])
        rows.append((k, float(np.prod(1 + d["oc"])), float(np.prod(1 + d["gap"]))))
    return rows


tests = {t["id"]: t for t in m["tests"]}
main = tests["lead_lag:combined"]
best = max(m["tests"], key=lambda t: t["by_period"]["out_of_sample"]["sharpe"])
n_pass = sum(t["survives"] for t in m["tests"])
acct = m["small_account"]
seqp = m["sequential_vs_parallel_oos"]
hx = m["hourly_crosscheck_ndx"]
look = m["lookahead_demo_ndx_sharpe"]
io = intraday_overnight()
periods = m["periods"]


# ---------------------------------------------------------------- charts (inline SVG)

def equity_chart():
    W, H, L, R, T, B = 760, 330, 54, 170, 14, 30
    series = [("lead_lag", "Lead-lag", "--s1"), ("threshold", "+ threshold", "--s2"),
              ("vix", "+ VIX scaling", "--s3"), ("expansion", "Cond. expansion", "--s4"),
              ("always_long", "Always long", "--s5")]
    lo = float(np.log10(eq.min().min()))
    hi = float(np.log10(eq.max().max()))
    lo, hi = math.floor(lo), math.ceil(hi)
    t0, t1 = eq.index[0], eq.index[-1]
    X = lambda t: L + (W - L - R) * (t - t0).days / (t1 - t0).days
    Y = lambda v: T + (H - T - B) * (1 - (math.log10(v) - lo) / (hi - lo))
    out = []
    for name, (a, b) in periods.items():
        a, b = max(pd.Timestamp(a), t0), min(pd.Timestamp(b), t1)
        fill = "var(--shade)" if name == "out_of_sample" else ("var(--shade2)" if name == "hold_out" else "none")
        out.append(f'<rect x="{X(a):.1f}" y="{T}" width="{X(b) - X(a):.1f}" height="{H - T - B}" fill="{fill}"/>')
        label = {"in_sample": "In-sample", "out_of_sample": "Out-of-sample", "hold_out": "Hold-out"}[name]
        out.append(f'<text x="{(X(a) + X(b)) / 2:.1f}" y="{T + 12}" text-anchor="middle" class="ax">{label}</text>')
    for e in range(lo, hi + 1):
        y = Y(10 ** e)
        lab = f"{10 ** e:g}×"
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{lab}</text>')
    for yr in range(2002, t1.year + 1, 4):
        x = X(pd.Timestamp(f"{yr}-01-01"))
        out.append(f'<text x="{x:.1f}" y="{H - 10}" text-anchor="middle" class="ax">{yr}</text>')
    ends = []
    for col, label, tok in series:
        s = eq[col].dropna().iloc[::3]
        d = "M" + "L".join(f"{X(t):.1f} {Y(v):.1f}" for t, v in s.items())
        out.append(f'<path d="{d}" fill="none" stroke="var({tok})" stroke-width="1.6"/>')
        ends.append([Y(eq[col].dropna().iloc[-1]), label, tok, eq[col].dropna().iloc[-1]])
    ends.sort()
    last = -99
    for e in ends:  # keep end labels from overlapping
        e[0] = max(e[0], last + 15)
        last = e[0]
        val = f"{e[3]:.2f}×" if e[3] >= 0.01 else "<0.01×"
        out.append(f'<text x="{W - R + 6}" y="{e[0] + 4:.1f}" class="lab" fill="var({e[2]})">{e[1]} {val}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Growth of 1 unit for each rule, log scale, 2001 to 2026">{"".join(out)}</svg>'


def cost_chart():
    W, H, L, R, T, B = 760, 250, 54, 220, 14, 34
    rows = [("lead_lag:combined", "Lead-lag, combined", "--s1"), ("threshold:combined", "+ threshold, combined", "--s2"),
            ("vix:combined", "+ VIX, combined", "--s3"), ("threshold:N225", "Nikkei after Nasdaq, threshold", "--s4")]
    costs = bt.COSTS_BPS
    vals = [[tests[i]["cost_sensitivity"][str(c)] for c in costs] for i, _, _ in rows]
    lo = math.floor(min(min(v) for v in vals) * 2) / 2
    hi = max(1.0, math.ceil(max(max(v) for v in vals) * 2) / 2)
    X = lambda c: L + (W - L - R) * c / costs[-1]
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    out = []
    v = lo
    while v <= hi + 1e-9:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="ax">{v:.1f}</text>')
        v += 0.5
    out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(0.5):.1f}" y2="{Y(0.5):.1f}" stroke="var(--fg)" stroke-dasharray="5 4"/>'
               f'<text x="{W - R}" y="{Y(0.5) - 6:.1f}" text-anchor="end" class="ax">pass line: Sharpe 0.5</text>')
    for c in costs:
        out.append(f'<text x="{X(c):.1f}" y="{H - 14}" text-anchor="middle" class="ax">{c} bps</text>')
    out.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 1}" text-anchor="middle" class="ax">round-trip cost per session trade</text>')
    ends = []
    for (i, label, tok), vs in zip(rows, vals):
        d = "M" + "L".join(f"{X(c):.1f} {Y(v):.1f}" for c, v in zip(costs, vs))
        out.append(f'<path d="{d}" fill="none" stroke="var({tok})" stroke-width="2"/>')
        out += [f'<circle cx="{X(c):.1f}" cy="{Y(v):.1f}" r="3" fill="var({tok})"/>' for c, v in zip(costs, vs)]
        ends.append([Y(vs[-1]), label, tok])
    ends.sort()
    last = -99
    for e in ends:
        e[0] = max(e[0], last + 15)
        last = e[0]
        out.append(f'<text x="{W - R + 8}" y="{e[0] + 4:.1f}" class="lab" fill="var({e[2]})">{e[1]}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Out-of-sample Sharpe ratio against trading cost">{"".join(out)}</svg>'


def account_chart():
    W, H, L, R, T, B = 760, 220, 64, 30, 30, 30
    s = acct_curve["equity_sgd"]
    t0 = pd.Timestamp(acct["start"])
    t1 = s.index[-1]
    lo, hi = min(0, s.min()), max(acct["start_sgd"], s.max())
    hi = math.ceil(hi / 1000) * 1000
    X = lambda t: L + (W - L - R) * (t - t0).days / max((t1 - t0).days, 1)
    Y = lambda v: T + (H - T - B) * (1 - (v - lo) / (hi - lo))
    out = []
    for v in range(0, int(hi) + 1, 1000):
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="ax">S${v:,}</text>')
    for mth in pd.date_range(t0, t1, freq="3MS"):
        if X(mth) > W - R - 30:
            continue
        out.append(f'<text x="{X(mth):.1f}" y="{H - 10}" text-anchor="middle" class="ax">{mth:%b %Y}</text>')
    pts = [(t0, acct["start_sgd"])] + list(s.items())
    d = "M" + "L".join(f"{X(t):.1f} {Y(v):.1f}" for t, v in pts)
    area = d + f"L{X(t1):.1f} {Y(lo):.1f}L{X(t0):.1f} {Y(lo):.1f}Z"
    out.append(f'<path d="{area}" fill="var(--bad-soft)"/><path d="{d}" fill="none" stroke="var(--bad)" stroke-width="1.8"/>')
    out.append(f'<circle cx="{X(t1):.1f}" cy="{Y(s.iloc[-1]):.1f}" r="4" fill="var(--bad)"/>'
               f'<text x="{W - R}" y="{T - 12}" text-anchor="end" class="lab" fill="var(--bad)">Account gone, {pd.Timestamp(acct["ruined_on"]):%-d %b %Y}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="S$5,000 account equity trading one micro contract per session">{"".join(out)}</svg>'


# ---------------------------------------------------------------- tables

def tests_table():
    head = ("<tr><th>Rule</th><th>Market</th><th>In-sample Sharpe (t)</th><th>Out-of-sample Sharpe</th>"
            "<th>Hold-out Sharpe</th><th>Break-even cost</th><th>Checks passed</th><th>Verdict</th></tr>")
    body = []
    for t in m["tests"]:
        b = t["by_period"]
        market = NAMES[t["legs"][0]] if len(t["legs"]) == 1 else "All three, in sequence"
        passed = sum(t["checks"].values())
        tip = html.escape("; ".join(f"{CHECK_LABELS[k]}: {'yes' if v else 'no'}" for k, v in t["checks"].items()))
        body.append(
            f"<tr><td>{RULES[t['rule']]}</td><td>{market}</td>"
            f"<td class='{cls(b['in_sample']['sharpe'])}'>{num(b['in_sample']['sharpe'])} <span class='mute'>({num(b['in_sample']['t_stat'], 1)})</span></td>"
            f"<td class='{cls(b['out_of_sample']['sharpe'])}'>{num(b['out_of_sample']['sharpe'])}</td>"
            f"<td class='{cls(b['hold_out']['sharpe'])}'>{num(b['hold_out']['sharpe'])}</td>"
            f"<td>{b['out_of_sample']['breakeven_bps']:.1f} bps</td>"
            f"<td title='{tip}'><span class='dots'>{''.join('<i class=on></i>' if v else '<i></i>' for v in t['checks'].values())}</span> {passed}/5</td>"
            f"<td><span class='pill {'ok' if t['survives'] else 'fail'}'>{'Survives' if t['survives'] else 'Fails'}</span></td></tr>")
    return f"<table>{head}{''.join(body)}</table>"


def io_table():
    rows = "".join(
        f"<tr><td>{NAMES[k]}</td><td class='{cls(a - 1)}'>{a:.2f}×</td><td class='{cls(b - 1)}'>{b:.2f}×</td></tr>" for k, a, b in io)
    return ("<table><tr><th>Index, 2001–2026</th><th>During sessions (open → close)</th>"
            f"<th>Overnight (close → next open)</th></tr>{rows}</table>")


def yearly_table():
    rows = []
    for y, r in yearly.iterrows():
        rows.append(f"<tr><td>{y}</td><td class='{cls(r['lead_lag'])}'>{pct(r['lead_lag'])}</td>"
                    f"<td class='{cls(r['always_long'])}'>{pct(r['always_long'])}</td></tr>")
    return ("<table><tr><th>Year</th><th>Lead-lag, all three</th><th>Always long, all three</th></tr>"
            f"{''.join(rows)}</table>")


def account_table():
    rows = []
    for k in bt.LEGS:
        rows.append(f"<tr><td>{acct['contracts'][k]}</td><td>{sgd(acct['notional_per_contract_today_sgd'][k])}</td>"
                    f"<td>{acct['leverage_today'][k]:.1f}×</td></tr>")
    return ("<table><tr><th>Smallest contract</th><th>Exposure per contract today</th><th>Times a S$5,000 account</th></tr>"
            f"{''.join(rows)}</table>")


# ---------------------------------------------------------------- page

b_main = main["by_period"]
b_best = best["by_period"]
tm = m["random_null_oos_sharpe"]
css = (ROOT / "report_style.css").read_text()

page = f"""<title>Session Lead-Lag Backtest</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{css}</style>
<main>
<header class="top">
  <p class="eyebrow">ClaudeHedge research · Stage 2 · data to {html.escape(periods['hold_out'][1])}</p>
  <h1>The Nikkei → DAX → Nasdaq session idea does not survive real data</h1>
  <p class="lede">I rebuilt the strategy from free Yahoo index data, 2001–2026, with trading costs.
  <b>{n_pass} of {m['n_tests']}</b> pre-registered tests passed. The rule closest to the repo's description lost money out-of-sample. A S$5,000 account trading it with micro futures would have been wiped out by
  {pd.Timestamp(acct['ruined_on']):%B %Y}.</p>
</header>

<section class="kpis" aria-label="Headline results">
  <div><b class="neg">{num(b_main['out_of_sample']['sharpe'])}</b><span>Out-of-sample Sharpe, lead-lag across all three markets, 5 bps cost (2017–2024)</span></div>
  <div><b>{b_main['out_of_sample']['breakeven_bps']:.1f} bps</b><span>Cost per trade at which it breaks even. Realistic cost is about 2–5 bps.</span></div>
  <div><b class="{cls(b_best['out_of_sample']['sharpe'])}">{num(b_best['out_of_sample']['sharpe'])}</b><span>Best out-of-sample Sharpe of any rule ({RULES[best['rule']].lower()}, {NAMES[best['legs'][0]] if len(best['legs']) == 1 else 'combined'}). Pass line is 0.5.</span></div>
  <div><b class="neg">{pd.Timestamp(acct['ruined_on']):%b %Y}</b><span>When a S$5,000 account trading one micro contract per session hit zero</span></div>
</section>

<section>
  <h2>Why it fails: the gains happen while these markets are closed</h2>
  <p>The strategy only holds positions during each market's session, from open to close. Since 2001, almost all of the long-run rise in these indices came between one session's close and the next session's open. A strategy that is flat overnight starts from that handicap. It needs a real directional edge just to break even, and the signal isn't strong enough.</p>
  <div class="tw">{io_table()}</div>
  <p class="note">Growth multiples compound each daily return. Days with a defective Yahoo open are excluded, so the products differ slightly from the raw index change.</p>
</section>

<section>
  <h2>Every rule, tested the same way</h2>
  <p>The rules, threshold, VIX bands, cost and pass criteria were fixed before any result was seen. Each signal uses only prices known before the trade opens. A rule had to pass all five checks: out-of-sample Sharpe above 0.5 after 5 bps, beating always-long, beating 95% of 1,000 random-direction runs, an in-sample t-stat above {m['bonferroni_z']:.2f} (the 5% level after a Bonferroni correction for {m['n_tests']} tests), and staying positive in the 2025–2026 hold-out. Hover the dots to see which checks each rule passed.</p>
  <div class="tw">{tests_table()}</div>
  <p class="note">Periods: in-sample {periods['in_sample'][0]} to {periods['in_sample'][1]}; out-of-sample {periods['out_of_sample'][0]} to {periods['out_of_sample'][1]}; hold-out {periods['hold_out'][0]} to {periods['hold_out'][1]}. Sharpe ratios use daily returns after 5 bps per session round trip. Break-even cost is the out-of-sample gross profit per unit traded.</p>
</section>

<section>
  <h2>Growth of 1 unit, all three markets in sequence</h2>
  <div class="chart">{equity_chart()}</div>
  <p class="note">Log scale, after 5 bps per session trade. "Always long" buys every session at the open and sells at the close. It loses because of the overnight effect above plus three cost charges a day. Shaded: out-of-sample (darker) and hold-out.</p>
</section>

<section>
  <h2>Costs decide it</h2>
  <p>Before costs, the combined lead-lag rule has an out-of-sample Sharpe of {num(main['cost_sensitivity']['0'])}. At 2 bps per trade it is already negative. Trading three sessions a day multiplies the cost drag, and a weak signal can't carry it.</p>
  <div class="chart">{cost_chart()}</div>
</section>

<section class="split">
  <div>
    <h2>The one lead worth noting</h2>
    <p><b>Trading the Nikkei session in the direction of the previous Nasdaq session</b>, only when the Nasdaq move was bigger than 0.25%, is the strongest result:</p>
    <ul>
      <li>In-sample Sharpe {num(tests['threshold:N225']['by_period']['in_sample']['sharpe'])}, t-stat {num(tests['threshold:N225']['by_period']['in_sample']['t_stat'], 1)}. That's below the {m['bonferroni_z']:.2f} needed once you account for {m['n_tests']} tests.</li>
      <li>Out-of-sample Sharpe {num(tests['threshold:N225']['by_period']['out_of_sample']['sharpe'])}, under the 0.5 pass line. It breaks even at {tests['threshold:N225']['by_period']['out_of_sample']['breakeven_bps']:.1f} bps.</li>
      <li>Hold-out Sharpe {num(tests['threshold:N225']['by_period']['hold_out']['sharpe'])} over {tests['threshold:N225']['by_period']['hold_out']['days']} days. That's encouraging but short, and one good stretch is what chance produces when you test 13 rules.</li>
    </ul>
    <p>This is the US-to-Japan spillover documented in academic work. Most of it shows up in the Nikkei's opening gap, before you can trade. What's left is thin.</p>
  </div>
  <div>
    <h2>What would have made it look good</h2>
    <p>Using the full same-day DAX session as the Nasdaq signal turns the Nasdaq leg into a Sharpe of <b>{num(look['in_sample'])}</b> in-sample and <b>{num(look['out_of_sample'])}</b> out-of-sample. That signal isn't tradable: the DAX closes two hours after the Nasdaq opens. A timing slip this small is enough to produce an impressive-looking backtest, which is why the tests above check that every signal comes before its trade.</p>
    <p>The hourly data confirms the timing matters, and that the honest version fails. Rebuilding the Nasdaq leg from 60-minute bars over {hx['first']} to {hx['last']}, using DAX prices only up to 14:00 Berlin, gives a Sharpe of {num(hx['hourly_timing']['sharpe'])}, against {num(hx['daily_proxy']['sharpe'])} for the daily proxy. The two versions take the same side {hx['position_agreement'] * 100:.0f}% of the time.</p>
  </div>
</section>

<section>
  <h2>"Sequential compounding" is leverage</h2>
  <p>The README says rolling the full account through three sessions a day is 4.75× more capital-efficient than splitting it. Out-of-sample, the sequential version has the same Sharpe as an even split held at three times the size:</p>
  <div class="tw"><table><tr><th>Always long, 2017–2024, 5 bps</th><th>Annual return</th><th>Volatility</th><th>Sharpe</th><th>Max drawdown</th></tr>
  <tr><td>Full account through each session in turn</td><td class="neg">{pct(seqp['sequential']['ann_return'])}</td><td>{seqp['sequential']['ann_vol'] * 100:.1f}%</td><td class="neg">{num(seqp['sequential']['sharpe'])}</td><td class="neg">{pct(seqp['sequential']['max_drawdown'])}</td></tr>
  <tr><td>Account split three ways</td><td class="neg">{pct(seqp['parallel']['ann_return'])}</td><td>{seqp['parallel']['ann_vol'] * 100:.1f}%</td><td class="neg">{num(seqp['parallel']['sharpe'])}</td><td class="neg">{pct(seqp['parallel']['max_drawdown'])}</td></tr>
  <tr><td>Split three ways, at 3× size</td><td class="neg">{pct(seqp['parallel_x3']['ann_return'])}</td><td>{seqp['parallel_x3']['ann_vol'] * 100:.1f}%</td><td class="neg">{num(seqp['parallel_x3']['sharpe'])}</td><td class="neg">{pct(seqp['parallel_x3']['max_drawdown'])}</td></tr></table></div>
  <p class="note">Compounding magnifies whatever the edge is. With a negative edge it magnifies losses.</p>
</section>

<section>
  <h2>What it means for a S$5,000 account</h2>
  <p>Futures come in whole contracts, so you can't size down to match a small account. One contract of each micro future today is a very different bet:</p>
  <div class="tw">{account_table()}</div>
  <p>Trading the combined lead-lag rule with one contract per session from January 2017, at 5 bps cost and today's exchange rates, the account fell to zero in {pd.Timestamp(acct['ruined_on']):%B %Y}. With no costs at all it still hit zero, in {pd.Timestamp(acct['no_costs']['ruined_on']):%B %Y}. One contract is a fixed bet, so as the account shrinks the leverage rises, and losses speed up.</p>
  <div class="chart">{account_chart()}</div>
  <p class="note">Margin calls aren't modelled; a broker would have closed positions earlier. Index levels in 2017 were lower, so the starting exposure was {acct['leverage_at_start']['NDX']:.1f}× for MNQ and {acct['leverage_at_start']['DAX']:.1f}× for Micro-DAX.</p>
</section>

<section class="split">
  <div>
    <h2>Recommendation</h2>
    <ul>
      <li><b>Don't trade this strategy</b>, and don't pay for futures data to test it further. It fails the criteria agreed before the test, and with whole contracts the account is too small to size it safely.</li>
      <li><b>If you want to keep researching</b>, the only thread left is the Nikkei-after-Nasdaq rule. Treat it as a new hypothesis, test it on genuinely new data (futures, not cash index), and paper trade it before any money goes in.</li>
      <li><b>Remember the overnight effect.</b> Strategies that are flat overnight give up most of the market's historical return. This is general information, not personal financial advice.</li>
    </ul>
  </div>
  <div>
    <h2>Main rule, out-of-sample risk</h2>
    <div class="tw"><table>
      <tr><td>Bootstrap Sharpe, 95% range</td><td class="neg">{num(m['main_rule_bootstrap_oos']['sharpe_p2_5'])} to {num(m['main_rule_bootstrap_oos']['sharpe_p97_5'])}</td></tr>
      <tr><td>Median bootstrap max drawdown</td><td class="neg">{pct(m['main_rule_bootstrap_oos']['max_dd_p50'])}</td></tr>
      <tr><td>Daily VaR 95% (parametric)</td><td class="neg">{pct(m['main_rule_var_cvar_oos']['var_parametric'], 2)}</td></tr>
      <tr><td>Daily CVaR 95% (historical)</td><td class="neg">{pct(m['main_rule_var_cvar_oos']['cvar_historical'], 2)}</td></tr>
      <tr><td>Random-direction Sharpe, 5–95%</td><td>{num(tm['N225+DAX+NDX']['p5'])} to {num(tm['N225+DAX+NDX']['p95'])}</td></tr>
    </table></div>
  </div>
</section>

<section>
  <h2>Year by year</h2>
  <details><summary>Show the yearly returns, all three markets, 5 bps</summary><div class="tw">{yearly_table()}</div></details>
</section>

<section class="method">
  <h2>Method and limits</h2>
  <ul>
    <li><b>Data:</b> Yahoo Finance daily bars for ^N225, ^GDAXI, ^NDX and ^VIX, 2000–2026, fetched {html.escape(m['generated_from'])}, plus 60-minute bars for 2023–2026. The Nasdaq series starts 2001 to skip defective opens in 2000. The 44 days where Yahoo reports an open equal to the prior close are not traded.</li>
    <li><b>Signals:</b> the Nikkei uses the previous Nasdaq session. The DAX uses the same day's Nikkei session. The Nasdaq uses the same day's Nikkei session plus the DAX opening gap. VIX scaling uses the prior VIX close.</li>
    <li><b>Prices:</b> cash-index open and close are a proxy. Futures trade around them, and real fills will differ.</li>
    <li><b>Contracts:</b> MNQ is $2 × Nasdaq 100 (CME), Micro-DAX is €1 × DAX (Eurex), Nikkei 225 Micro is ¥10 × Nikkei (OSE). FX is held at EURUSD {acct['fx']['EURUSD']}, USDJPY {acct['fx']['USDJPY']}, USDSGD {acct['fx']['USDSGD']}.</li>
    <li><b>Reproduce:</b> <code>research/stage2/</code> in the repo: <code>fetch_data.py</code>, <code>backtest.py</code>, <code>make_report.py</code>, and 11 unit tests including lookahead checks.</li>
  </ul>
</section>
</main>
"""

(ROOT / "report.html").write_text(page)
print("wrote", ROOT / "report.html", len(page), "bytes")
