# NIFTY Intelligence Terminal

A quantitative market-research and live market-intelligence platform for **NIFTY 50**, built to combine real-time market data, multi-timeframe analysis, feature engineering, machine-learning research, backtesting, signal evaluation, and risk controls in a single system.

The project focuses on a core principle:

> **The chart, quantitative analysis, and model inference must operate on the same canonical market state.**

This avoids inconsistencies caused by using different data feeds or timestamps for visualization and analysis.

---

## Overview

The NIFTY Intelligence Terminal is designed as an event-driven research and market-analysis platform for Indian equity indices.

The system supports:

- Live and historical NIFTY 50 market data
- Canonical market-state construction
- 1-minute market-data ingestion
- Multi-timeframe candle aggregation
- Feature engineering and price-action analysis
- Machine-learning research pipelines
- BUY / SELL / WAIT decision policies
- Probability calibration
- Historical replay and backtesting
- Walk-forward evaluation
- Signal lifecycle tracking
- Paper-trade and prediction tracking
- Data-quality validation
- Risk and release controls
- Real-time dashboard visualization

The platform is primarily a **quantitative research and decision-support system**, not an automated trading system.

---

## System Architecture

```text
                    Market Data Provider
                            |
                            v
                   Data Ingestion Layer
                            |
              Validation / Sequencing / Deduplication
                            |
                            v
                  Canonical Market State
                            |
             +--------------+--------------+
             |                             |
             v                             v
      Candle Engine                 Market Snapshots
             |                             |
      1m / 5m / 15m / 1h                  |
             |                             |
             +-------------+---------------+
                           |
                           v
                    Feature Engine
                           |
              +------------+-------------+
              |                          |
              v                          v
       Price Action                ML / Research
         Analysis                     Models
              |                          |
              +------------+-------------+
                           |
                           v
                 Calibration / Policy
                           |
                           v
                  BUY / SELL / WAIT
                           |
              +------------+-------------+
              |                          |
              v                          v
        Signal Tracking             Dashboard
              |
              v
      Replay / Evaluation
```

---

## Market Data Pipeline

The market-data layer is responsible for creating a consistent representation of market activity before any feature computation or prediction occurs.

The pipeline performs:

1. Provider data ingestion
2. Timestamp normalization
3. Validation and sequencing
4. Duplicate detection
5. Candle construction
6. Market-state snapshot creation
7. Feature computation
8. Research/model inference
9. Signal-policy evaluation
10. Dashboard publication

The current implementation integrates **Angel One SmartAPI** for NIFTY market-data acquisition.

Sensitive API credentials are kept outside the repository using environment variables.

---

## Historical Data

The project includes a historical acquisition and validation pipeline for NIFTY 50.

The current research dataset covers approximately:

```text
January 2025 - August 2026
```

with more than **150,000 one-minute market observations** processed.

Canonical candles are generated for:

| Timeframe | Purpose |
|---|---|
| 1 minute | Base market-data resolution |
| 5 minutes | Primary analysis / inference timeframe |
| 15 minutes | Intraday context |
| 1 hour | Higher-timeframe context |

Historical data passes through session normalization and data-quality checks before being used for research.

The pipeline detects conditions including:

- Missing minute intervals
- Consecutive data gaps
- Duplicate observations
- Out-of-order events
- Session inconsistencies
- Candle revisions

Research datasets can be rejected when quality thresholds are not satisfied.

---

## Multi-Timeframe Analysis

Instead of evaluating a single candle interval independently, the terminal maintains multiple synchronized views of the market.

```text
1m  -> short-term market structure
5m  -> primary decision timeframe
15m -> intraday context
1h  -> broader market regime
```

This allows short-term signals to be evaluated against higher-timeframe market conditions.

---

## Canonical Market State

A major architectural goal of the project is preventing the chart and analytical engine from observing different versions of the market.

The system therefore constructs a **canonical market-state snapshot**.

The same timestamped state is used by:

- Charts
- Feature computation
- Price-action analysis
- Model inference
- Signal generation
- Historical replay
- Prediction tracking

This also improves reproducibility when investigating historical predictions.

---

## Feature Engineering

The feature pipeline transforms market observations into structured inputs for quantitative research.

The architecture supports features derived from areas such as:

- Returns and momentum
- Volatility
- Candle structure
- Trend behaviour
- Multi-timeframe relationships
- Market regime
- Price action
- Cross-market context
- Derivatives context

Feature snapshots are versioned so that historical predictions can be reproduced using the same feature definitions.

---

## Price-Action Analysis

The terminal includes a dedicated price-action research layer.

It is designed to analyse market structure independently from the machine-learning pipeline and provide additional context for quantitative decisions.

This enables research into combinations of:

```text
Market Data
     +
Quantitative Features
     +
Price Action
     +
Market Context
     ↓
Decision Policy
```

---

## Machine-Learning Research

The ML subsystem provides infrastructure for experimenting with directional market models.

The research workflow includes:

```text
Historical Market Data
        ↓
Feature Generation
        ↓
Label Construction
        ↓
Chronological Dataset
        ↓
Training
        ↓
Walk-Forward Evaluation
        ↓
Probability Calibration
        ↓
Decision Policy
```

The project intentionally separates:

- Model prediction
- Probability calibration
- Trading/decision policy

A model producing a directional probability does not automatically produce a trade signal.

---

## Time-Series Validation

Financial datasets cannot safely be evaluated using ordinary random train/test splitting because observations are chronologically dependent.

The research framework therefore uses chronological evaluation and supports walk-forward testing.

This helps reduce:

- Look-ahead bias
- Data leakage
- Unrealistic evaluation
- Overfitting to historical periods

Model-selection gates evaluate more than prediction accuracy alone.

---

## Signal Engine

The decision layer produces three possible states:

```text
BUY
SELL
WAIT
```

`WAIT` is intentionally treated as a valid decision.

A directional model prediction does not automatically become an actionable signal.

Signals may depend on:

- Model output
- Calibrated probability
- Market regime
- Feature validity
- Data quality
- Risk constraints
- Release-readiness conditions

---

## Risk Controls

The platform follows a fail-safe approach.

If required market state, feature data, model state, or integrity checks are unavailable, the system should prefer:

```text
WAIT / SIGNAL UNAVAILABLE
```

rather than generating an unsupported trading decision.

The project also includes a **live signal kill switch**, allowing signal generation to remain disabled until required research and release gates are satisfied.

---

## Backtesting & Evaluation

The research framework evaluates strategies and models using metrics beyond simple accuracy or win rate.

Examples include:

- Trade count
- Win rate
- Expectancy
- Profit factor
- Net points
- Drawdown
- Risk-normalized performance
- Calibration quality

Research candidates must satisfy predefined evaluation gates before they can be considered for further testing.

A candidate failing a risk or statistical-confidence requirement is not promoted simply because its historical return is positive.

---

## Probability Calibration

Raw machine-learning probabilities are not assumed to represent reliable market probabilities.

The calibration subsystem evaluates and transforms model outputs before they can be used by the decision policy.

The system separates:

```text
Raw Model Score
       ↓
Probability Calibration
       ↓
Policy Evaluation
       ↓
BUY / SELL / WAIT
```

Precise probability displays can remain disabled until calibration requirements are satisfied.

---

## Prediction & Signal Tracking

Predictions and signals can be associated with their underlying market state and research configuration.

The tracking architecture includes concepts such as:

- Market snapshot
- Feature version
- Model version
- Calibration version
- Signal-policy version
- Prediction outcome
- Signal lifecycle
- Paper-trade events

This provides an auditable connection between a historical decision and the information available when that decision was generated.

---

## Data Integrity & Reproducibility

Reproducibility is an important design goal of the project.

Historical datasets and research artifacts can be identified using metadata and cryptographic hashes.

This helps answer questions such as:

> Which exact dataset, features, model and policy produced this result?

rather than relying only on filenames or mutable datasets.

---

## Cross-Market & Derivatives Research

The architecture contains research components for incorporating additional market context beyond NIFTY spot prices.

This includes work around:

- BANK NIFTY
- India VIX
- Cross-market relationships
- Derivatives context
- Conditional directional research

These components are used as research inputs rather than assumed trading signals.

---

## Technology Stack

### Backend

- Python
- FastAPI
- Pydantic
- SQLite
- REST APIs
- WebSockets

### Frontend

- React
- TypeScript
- Next.js / Vite

### Market Data

- Angel One SmartAPI
- Historical and live market-data pipelines

### Quantitative Research

- Python-based feature engineering
- Time-series datasets
- Walk-forward evaluation
- Probability calibration
- Historical replay

### Infrastructure

- Git / GitHub
- Cloudflare Workers
- Automated validation and testing

---

## Repository Structure

```text
stock-intelligence/
│
├── app/                  # Application routes and UI
├── config/               # Market/calendar configuration
├── contracts/            # Versioned data contracts and schemas
├── db/                   # Database definitions
├── docs/                 # Research and architecture documentation
├── public/               # Static frontend assets
├── scripts/              # Validation and research utilities
│
├── services/
│   └── backend/
│       ├── src/
│       │   └── nifty_terminal/
│       │       ├── api/
│       │       ├── calibration/
│       │       ├── candles/
│       │       ├── context/
│       │       ├── dashboard/
│       │       ├── derivatives/
│       │       ├── features/
│       │       ├── hardening/
│       │       ├── history/
│       │       ├── ingestion/
│       │       ├── ml/
│       │       ├── price_action/
│       │       ├── providers/
│       │       ├── research/
│       │       ├── runtime/
│       │       ├── shadow/
│       │       ├── signals/
│       │       ├── snapshots/
│       │       └── tracking/
│       │
│       └── tests/
│
├── src/                  # Frontend components and clients
├── tests/                # Frontend / integration tests
├── worker/               # Market gateway / Worker
│
├── .env.example
├── package.json
└── README.md
```

---

## Environment Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

On Windows Command Prompt:

```bat
copy .env.example .env
```

Provider credentials must only be placed in the local `.env` file.

Never commit:

```text
.env
API keys
Client codes
PINs
TOTP secrets
Authentication tokens
```

The repository intentionally contains only empty credential placeholders in `.env.example`.

---

## Running the Project

Install the JavaScript dependencies:

```bash
npm install
```

The Python backend is located under:

```text
services/backend/
```

Create and activate a Python virtual environment and install the backend dependencies according to the backend project configuration.

For development, the architecture supports a local frontend and backend, with the frontend consuming the market API and WebSocket stream.

---

## Testing

The repository contains automated tests covering major components of the platform, including:

- Market-data ingestion
- Candle generation
- Historical acquisition
- Feature snapshots
- ML datasets
- Calibration
- Price action
- Signal policies
- Market snapshots
- Tracking
- Dashboard behaviour
- Research pipelines
- Data validation
- Provider integration

These tests are used to enforce consistency between the research and live-analysis infrastructure.

---

## Current Status

The project is under active development and research.

Implemented infrastructure includes market-data acquisition, historical processing, canonical candle generation, feature pipelines, research workflows, price-action analysis, signal-policy infrastructure, replay/evaluation components, tracking, and dashboard components.

Live trading and automatic order execution are **not enabled**.

Research models must pass statistical, risk, data-quality, and release-readiness gates before they can be considered suitable for live signal use.

---

## Research Principles

The project follows several rules intended to keep financial research realistic:

1. **No look-ahead information** in historical predictions.
2. **Chronological evaluation** instead of random time-series splitting.
3. **Same canonical data** for visualization and quantitative analysis.
4. **No signal when required data is invalid or unavailable.**
5. **Risk-adjusted evaluation** rather than relying only on win rate.
6. **Reproducible datasets and model lineage.**
7. **WAIT is a valid decision.**
8. **Historical performance is not assumed to represent future performance.**

---

## Motivation

This project was built to explore the intersection of:

**Financial Markets + Quantitative Research + Machine Learning + Software Engineering**

Rather than treating market prediction as a standalone ML classification problem, the terminal focuses on the complete research lifecycle:

```text
Data
 ↓
Validation
 ↓
Market State
 ↓
Features
 ↓
Research
 ↓
Calibration
 ↓
Decision Policy
 ↓
Risk Controls
 ↓
Tracking
 ↓
Evaluation
```

The goal is to build a rigorous framework for studying short-term behaviour in Indian equity indices while maintaining clear separation between experimental research and live trading decisions.

---

## Disclaimer

This project is intended for **educational and quantitative research purposes only**.

It does not constitute financial or investment advice. Historical or simulated results do not guarantee future performance. Automated trading and official live trading signals are not enabled by default.

---

## Author

**Sahil Yadav**  
B.Tech Computer Science & Engineering  
Indian Institute of Technology Goa
