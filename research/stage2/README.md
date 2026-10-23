# Stage 2: session lead-lag backtest on free index data

Tests the claude-hedge idea (trade Nikkei -> DAX -> Nasdaq sessions in sequence) on real data with trading costs. The extracted repo's own return files were not used. The audit found the "live" file copies the backtest on every overlapping day.

## Rerun

```
pip install pandas numpy pytest
python research/stage2/fetch_data.py      # downloads Yahoo daily + hourly bars, FX; writes data_quality.json
python research/stage2/backtest.py        # writes results/metrics.json and CSVs
python -m pytest research/stage2/test_backtest.py
```

The data in `data/raw/` is the snapshot used for the report (fetched 2026-10-23 UTC). Re-fetching will extend it.

## What is tested (fixed before running)

Each session's return is the cash index open to close, as a proxy for futures. Signals use only information known before the session opens:

| Trade | Signal |
|---|---|
| Nikkei, day t | Nasdaq open-to-close, previous US session |
| DAX, day t | Nikkei open-to-close, same date |
| Nasdaq, day t | Nikkei same date + DAX overnight gap (the DAX close is after the Nasdaq open) |

**Rules:**
- Lead-lag (trade in the signal's direction).
- Lead-lag + 0.25% threshold.
- Lead-lag + VIX scaling (VIX <15: 1.0, 15-20: 0.75, 20-30: 0.5, >30: 0.25).
- The repo's "conditional expansion" (double DAX/Nasdaq longs after an up Nikkei session).

Each rule is tested per leg and combined, giving 13 tests and a Bonferroni z of 2.89.

**Periods:**
- In-sample: 2001-2016.
- Out-of-sample: 2017-2024.
- Hold-out: 2025 to the present.

**Costs:** 5 bps per session round trip is the base case. 0, 2 and 10 bps are run as a sensitivity check.

**Pass (all required):**
- Out-of-sample Sharpe above 0.5 after 5 bps.
- Beats always-long open-to-close out-of-sample.
- Beats 95% of 1,000 random-direction runs out-of-sample.
- In-sample t-stat above the Bonferroni threshold.
- Positive Sharpe in the hold-out.

## Limits

- Index open/close prices are not tradable fills; futures trade around them.
- Daily data can't do the exact 24-hour handoff. `hourly_crosscheck_ndx` redoes the Nasdaq leg with true timing on 2023-12 to 2026-10 60-minute bars.
- The small-account simulation uses whole micro contracts (MNQ $2, Micro-DAX EUR 1, OSE Nikkei Micro JPY 10 per point) and today's FX held constant.
- No margin calls are modelled. The account is simply marked as ruined when equity reaches zero.
