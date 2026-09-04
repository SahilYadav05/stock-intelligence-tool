# Step 24 — causal derivatives features and readiness gate

Step 23 established that the existing price, Bank Nifty, India VIX, and chart
structure inputs can weakly identify future movement but cannot select direction
reliably. Its diagnostic direction AUC was 0.5047 and no policy passed.

Step 24 adds information unavailable to those studies. New collections synchronize
NIFTY spot and nearest-future quotes so the ledger records futures basis in points
and basis points. The point-in-time feature builder then derives within-session,
same-contract changes in spot, futures, basis, volume, open interest, order-book
imbalance, provider PCR, ATM IV, 25-delta IV skew, and option-volume PCR.

Deltas reset across sessions, contract rolls, and gaps longer than 15 minutes. Only
weekday snapshots between 09:15 and 15:30 IST can count toward readiness. The next
model experiment is blocked until the append-only ledger contains at least 3,000
complete core rows across 60 sessions and 90 calendar days, with options context
complete on at least 80% of regular-session snapshots. These thresholds prevent a
small, freshly collected sample from producing another misleading win rate.

Check progress with:

```text
npm run audit:derivatives:windows
```

The readiness report is research-only and cannot enable an official signal or order.
