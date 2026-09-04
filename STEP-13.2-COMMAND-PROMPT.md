# Step 13.2 — Exact Muhurat provider-observation exclusions

Angel One returned four NIFTY index observations on 21 October 2025 at 11:17,
11:40, 14:45 and 14:46 IST. All four are outside NSE's official 13:45–14:45
continuous Muhurat trading session. They are not canonical tradable minute bars.

This patch authorizes exclusion of only those four exact minutes. Every other
unknown out-of-session timestamp remains a hard failure.

## Install from Windows Command Prompt

Stop the backend and run:

```cmd
cd /d C:\Users\sy771\Downloads\stock-intelligence
copy "config\nse-calendar-through-2026-08-25.json" "config\nse-calendar-through-2026-08-25.pre-step13.2.json"
tar -xf "%USERPROFILE%\Downloads\step13.2-calendar-hotfix.zip" -C .
npm.cmd run check:step13
npm.cmd run test:windows
```

Expected structural result:

```text
Step 13 check passed (7 required paths).
```

Run the acquisition again:

```cmd
npm.cmd run acquire:history:angelone:windows
```

The repeat download is necessary because rejected provider rows were not stored
in SQLite. Do not delete the prior reports; they are immutable audit records.

After completion, continue only if both fields are true:

```text
research_quality_accepted: true
training_research_ready: true
```

If not, share only the non-secret report sections previously listed. Never
share `.env`, credentials, tokens, the database or licensed raw market data.
