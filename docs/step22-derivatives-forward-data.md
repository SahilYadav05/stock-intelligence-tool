# Step 22 — derivatives forward-data ledger

The expanded Step 20 and Step 21 studies show that NIFTY price, BANKNIFTY, India
VIX, and causal price action do not yet establish a reliable tradable edge. More
threshold searches on those same diagnostic folds are prohibited.

Step 22 adds a read-only, append-only Angel One collector for information not
present in the price-only history:

- nearest NIFTY futures price, traded volume, open interest, buy/sell quantity,
  and normalized book imbalance;
- provider NIFTY put/call ratio;
- nearest-expiry ATM implied volatility, 25-delta put/call IV skew, and an
  option-volume put/call ratio derived from the returned surface.

The collector resolves current unexpired contracts from the official instrument
master on every run, hashes every snapshot, and inserts it idempotently into
`data/derivatives-forward.sqlite3`. It exposes no order method, keeps the global
live-signal kill switch on, and cannot turn a snapshot into an official signal.

Run one collection with:

```text
npm run collect:derivatives:windows
```

This data is forward-only because Angel One documents historical OI for live F&O
contracts; the current instrument master cannot honestly reconstruct expired
contract history. A future model experiment should start only after enough
point-in-time sessions have accumulated and must reserve a later untouched block.

