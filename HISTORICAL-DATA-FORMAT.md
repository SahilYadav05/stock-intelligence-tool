# Historical provider CSV format

Only import data you are licensed or otherwise authorized to store and use for
research. The adapter does not scrape a charting website and does not infer
missing prices.

## Required columns

```text
provider_bar_id
provider_revision
opens_at
closes_at
open
high
low
close
volume
finalized_at
source_watermark
```

Rules:

- Timestamps must be ISO 8601 with an explicit timezone.
- Rows represent finalized one-minute bars.
- `provider_bar_id` must be stable and unique per provider revision.
- Corrections reuse the candle opening time with a strictly higher
  `provider_revision`.
- NIFTY 50 spot `volume` must be blank.
- Missing minutes remain missing and produce a degraded quality verdict.
- Rows outside the requested interval or NSE session are rejected.
- The source file SHA-256 becomes part of the immutable dataset identity.

## Exchange calendar exceptions

Copy `config/nse-calendar.example.json` to a private working file and populate
holidays and special sessions from an authoritative exchange calendar.

Example shape only:

```json
{
  "holidays": ["YYYY-MM-DD"],
  "special_sessions": {
    "YYYY-MM-DD": {
      "open": "HH:MM",
      "close": "HH:MM"
    }
  }
}
```

Do not invent holiday or special-session dates. An incomplete calendar may
correctly cause a dataset to be marked degraded rather than silently accepted.
