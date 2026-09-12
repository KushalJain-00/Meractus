<div align="center">

# Mercatus Arena

### Algorithmic Trading Strategy — TechVerse 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Adaptive Z-Score Mean Reversion + Ensemble Voting**

[Strategy Report (PDF)](trading_report.pdf) · [Quick Start](#quick-start) · [Strategy](#strategy) · [Architecture](#architecture)

</div>

---

## Overview

A rule-based algorithmic trading strategy built for the Mercatus Arena competition. The system combines three independent signal generators through a weighted ensemble, with the core innovation being **volatility-adaptive z-score thresholds** that dynamically adjust entry quality based on market regime.

**Key Results (Backtest):**

| Metric | Value |
|--------|-------|
| Sharpe Ratio | **7.51** |
| Total Return | **+78.1%** |
| Max Drawdown | **-5.8%** |
| Win Rate | **49.9%** |
| Profit Factor | **12.51** |

---

## Quick Start

### Prerequisites

- Python 3.10+
- No external dependencies required (stdlib only for competition runner)

### Installation

```bash
git clone https://github.com/KushalJain-00/Meractus.git
cd Meractus
pip install -r requirements.txt
```

### Run Modes

```bash
# Offline simulation (fast, no server)
python3 run_live.py --mode simulate

# Mock server test (validates HTTP + WebSocket pipeline)
python3 run_live.py --mode mock

# Live competition (requires config.json with credentials)
cp config.json.example config.json
# Edit config.json with your API key
python3 run_live.py --mode live config.json

# Use organizer's runner (recommended for competition)
python3 run_strategy.py config.json
```

### Generate Report

```bash
python3 report.py
# Outputs trading_report.pdf
```

---

## Strategy

### Signal Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ENSEMBLE VOTER                     │
│                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌───────────┐ │
│   │ Mean Revert  │  │  Composite   │  │ Momentum  │ │
│   │    (55%)     │  │    (25%)     │  │   (20%)   │ │
│   │              │  │              │  │           │ │
│   │ Adaptive Z   │  │  RSI + MACD  │  │ EMA Cross │ │
│   │ + RSI Filter │  │  Crossover   │  │  (10/30)  │ │
│   └──────┬───────┘  └──────┬───────┘  └─────┬─────┘ │
│          │                 │                 │        │
│          └────────┬────────┘                 │        │
│                   │                          │        │
│              score > 0.3  →  BUY             │        │
│              score < -0.2 →  SELL            │        │
└─────────────────────────────────────────────────────┘
```

### Core Innovation: Adaptive Z-Score

Traditional mean reversion uses fixed thresholds. Our approach adapts to volatility regimes in real-time:

```
adaptive_z = -2.0 + (vol_percentile - 0.5) × 1.5
```

| Volatility Regime | Percentile | Threshold | Behavior |
|-------------------|------------|-----------|----------|
| High volatility | > 80% | -1.5 | More entries (opportunities in chaos) |
| Normal volatility | 20-80% | -2.0 | Baseline |
| Low volatility | < 20% | -2.5 | Stricter (higher quality signals) |

This is **parameter-free** — the percentile ranking automatically adapts to any market condition.

### Signal Formulas

**Mean Reversion (55% weight):**
```
z_score = (Price - SMA(20)) / StdDev(20)
BUY when: z < adaptive_z AND RSI(14) < 34
SELL when: z > 0 (back to mean)
```

**Composite (25% weight):**
```
MACD = EMA(12) - EMA(26)
BUY when: MACD crosses above 0 AND RSI(14) < 70
SELL when: MACD crosses below 0
```

**Momentum (20% weight):**
```
BUY when: EMA(10) crosses above EMA(30) AND RSI(14) < 74
SELL when: EMA(10) crosses below EMA(30)
```

---

## Risk Management

| Layer | Mechanism | Parameter |
|-------|-----------|-----------|
| **Position Sizing** | Equity-weighted with volatility scalar | Max 20% per symbol |
| **Cash Reserve** | Never deploy full cash | 10% reserve |
| **Trailing Stop** | Exit at -5% from peak | After 60 bars |
| **Hard Stop** | Exit at -5% from entry | Immediate |
| **Circuit Breaker** | Pause after 3 consecutive losses | 300 bar cooldown |
| **Stale Exit** | Force sell if held too long | 500 bars |
| **DD Scaling** | Reduce size during drawdowns | Linear to 25% at -15% DD |
| **Regime Guard** | Skip mean reversion on trending symbols | ADX-based |
| **Flash Crash** | Buy 10%+ dips for recovery | 60 bar lookback |

---

## Architecture

```
meractus/
├── strategy.py           # V2 production strategy
├── api_client.py         # REST + WebSocket client
├── run_live.py           # Runner (simulate/mock/live)
├── dashboard.py          # Rich terminal dashboard
├── mock_server.py        # Mock server for testing
├── report.py             # PDF report generator
├── config.json.example   # Config template
├── trading_report.pdf    # Competition report
├── requirements.txt      # Dependencies
├── docs/                 # Reference documentation
│   ├── User Guide.pdf
│   ├── USER_WORKFLOW.pdf
│   ├── trained_model_params.json
│   ├── dataset_meta.json
│   └── backtest_results.json
└── results/              # Version comparison data
    ├── v0_baseline.json
    ├── v1_adaptive_z.json
    └── ...
```

### File Descriptions

| File | Purpose | Lines |
|------|---------|-------|
| `strategy.py` | Core strategy with 3 signal generators, ensemble voter, and risk management | ~360 |
| `api_client.py` | Mercatus REST/WS client with rate limiting (60 trades/min, 300 API/min), JWT refresh, auto-reconnect | ~460 |
| `run_live.py` | Async runner supporting offline simulation, mock server testing, and live competition | ~210 |
| `dashboard.py` | Real-time terminal dashboard using Rich library | ~300 |
| `mock_server.py` | Full mock server (REST + WebSocket) for end-to-end testing | ~180 |
| `report.py` | PDF report generator with 12+ matplotlib charts | ~1200 |

---

## Competition Rules

| Rule | Value | Our Handling |
|------|-------|--------------|
| Starting Capital | $10,000,000 | — |
| Position Type | Long-only | No short logic |
| Fees/Slippage | Zero | Exact fill prices |
| Trade Rate Limit | 60/min | Built-in RateLimiter |
| API Rate Limit | 300/min | Built-in RateLimiter |
| API Freeze | T-15 min | Bot stops trading |
| Flash Crashes | -30% to +30% | Trailing stops + circuit breaker |
| Scoring | PnL (50%) + Code (25%) + Report (25%) | Optimized for all three |

---

## Results

### Version Comparison

| Version | Return | Sharpe | Max DD | Trades | Win Rate |
|---------|--------|--------|--------|--------|----------|
| V0 Baseline | +67.6% | 5.38 | -7.6% | 3,032 | 49.1% |
| **V1 Adaptive Z** | **+78.1%** | **7.51** | **-5.8%** | **2,960** | **49.9%** |
| V1 + P1 | +81.8% | 5.80 | -8.1% | 4,699 | 49.9% |

### Improvement Breakdown

| Improvement | Return Impact | Sharpe Impact | Verdict |
|-------------|---------------|---------------|---------|
| Adaptive z-score | +10.5% | +2.1 | **Breakthrough** |
| DD scaling | +1.2% | -0.3 | Marginal |
| Circuit breaker | +0.8% | -0.1 | Marginal |
| Regime guard | +0.9% | +0.1 | Marginal |
| Stale exit | +0.8% | -0.2 | Marginal |
| ATR stops | -30% | -3.7 | Rejected |
| Staged exit | -55% | -1.4 | Rejected |
| Kelly criterion | -78% | -5.0 | Rejected |

---

## Development

### Testing

```bash
# Run mock server + strategy end-to-end
python3 run_live.py --mode mock

# Generate fresh report
python3 report.py
```

### Adding Improvements

1. Create a new version in `strategy.py`
2. Backtest against `paper_dataset(1).csv`
3. Save results to `results/vN_name.json`
4. Regenerate report with `python3 report.py`

---

## License

MIT

---

<div align="center">

**Built for TechVerse 2026 · Mercatus Arena**

</div>
