# Step 8 Professional Dashboard Contract

The Step 8 dashboard is a working trading-intelligence surface, not a mock
trading screen. Every number, label, level, marker, and explanation must come
from a versioned backend contract tied to the chart's canonical market state.

## Snapshot synchronization

After each market-state WebSocket update, the browser requests analysis for the
exact `snapshot_id` and `candle_revision_checksum` shown by the chart. Analysis
is rendered only when all four identities match:

1. Market-state snapshot
2. Analysis view snapshot
3. Signal snapshot
4. Candle-revision checksum used by the feature/model input

Any missing or mismatched value produces `SYNCING ANALYSIS`, suppresses the
signal, hides precise probabilities and trade levels, and removes active chart
overlays. The chart may retain the last known candles during a stale or
disconnected state, but it must visibly say that new signals are disabled.

## First viewport

The first viewport keeps the candlestick chart continuously visible beside the
decision engine. It includes:

- NIFTY 50 spot and NSE identity
- 5m signal, 15m context, and 1H regime selectors
- current data status and age
- latest finalized price and candle time
- finalized/developing candle distinction
- EMA 20 and EMA 50 visual toggles
- support, resistance, entry, stop, targets, and signal overlays
- BUY, SELL, or WAIT with calibrated probability only when allowed
- UP, DOWN, and NEITHER probability distribution
- entry zone, stop, three targets, reward/risk, evidence, contradiction, and invalidation

NIFTY spot volume remains explicitly unavailable.

## Chart overlay policy

The chart uses TradingView Lightweight Charts with canonical backend candles.
EMA overlays are calculated from the finalized candles already received by the
browser and are visual aids only. Official inference remains server-side.
Support/resistance, entry/stop/targets, and markers are drawn only from the
synchronized `analysis-view.v1` response.

No chart screenshot or rendered pixel is an ML input.

## Context panels

Market regime, trend, momentum, volatility, news/events, historical analogs,
and data integrity all have explicit unavailable states. News requires source,
publication time, and arrival time. Historical analogs require approved feature
snapshots and indexed history. Empty fields are never populated with sample
market claims.

## Responsive and accessibility behavior

The desktop layout prioritizes chart width while keeping analysis visible.
Tablet stacks the analysis below the chart; mobile stacks every section with
touch-sized controls. Buttons expose pressed states, focus rings remain visible,
and status text does not rely on color alone.

## Current truth state

The repository still has no configured licensed feed or approved live model and
calibration artifact. The production dashboard therefore displays `LIVE
ANALYSIS UNAVAILABLE`, WAIT, blank probabilities, and unavailable context.
