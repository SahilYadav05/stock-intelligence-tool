# Step 1 — Windows setup and verification

This guide covers only Stage 3, Step 1: project foundation and repository architecture.

The supplied Step 1 project archive contains the complete contents of every
source, configuration, test, and documentation file. Using the archive is safer
than manually reconstructing framework-managed files with Notepad.

## 1. Install prerequisites

Open PowerShell as your normal Windows user.

```powershell
winget install OpenJS.NodeJS.LTS
winget install Python.Python.3.12
winget install Git.Git
```

Close PowerShell, open it again, and verify:

```powershell
node --version
npm --version
python --version
git --version
```

Required minimums:

- Node.js 22.13
- Python 3.12
- npm 10

## 2. Extract the complete project

Download `nifty-intelligence-terminal-step-1.zip` and place it in your Downloads folder.

```powershell
New-Item -ItemType Directory -Force "$HOME\Projects" | Out-Null
Expand-Archive -Path "$HOME\Downloads\nifty-intelligence-terminal-step-1.zip" -DestinationPath "$HOME\Projects" -Force
Set-Location "$HOME\Projects\nifty-intelligence-terminal"
```

Confirm that you are in the correct directory:

```powershell
Get-ChildItem
```

You should see `app`, `contracts`, `services`, `src`, `scripts`, `tests`, and `package.json`.

## 3. Create the local environment file

```powershell
Copy-Item ".env.example" ".env" -Force
notepad ".env"
```

For Step 1, keep `MARKET_DATA_PROVIDER` blank. Do not place any real provider
credentials in `.env.example` or any frontend variable beginning with
`NEXT_PUBLIC_`.

## 4. Install frontend dependencies

```powershell
npm install
```

This installs only open-source packages declared in the lockfile. No paid service is required.

## 5. Run all Step 1 checks

```powershell
npm run check:foundation
npm run lint:windows
npm run test:windows
```

Expected results include:

```text
Foundation check passed (7 required paths).
pass 1
fail 0
```

Run the Python foundation tests:

```powershell
$env:PYTHONPATH="services/backend/src"
python -m unittest discover -s services/backend/tests -p "test_*.py"
```

Expected result:

```text
Ran 2 tests
OK
```

## 6. Start the local website

```powershell
npm run dev:windows
```

Open the local URL printed in PowerShell. Stop the server with `Ctrl+C`.

The Step 1 page must show:

- `NIFTY Intelligence Terminal`
- `Step 1 of 10`
- `Market data — Not connected`
- `Model — Not trained`
- `Official signal — Unavailable`
- `LIVE ANALYSIS UNAVAILABLE`

It must not show fabricated prices, probabilities, signals, accuracy, or performance.

## 7. Open every Step 1-owned file in Notepad

These commands open the complete files supplied in the archive:

```powershell
notepad ".env.example"
notepad ".editorconfig"
notepad ".gitattributes"
notepad ".gitignore"
notepad "README.md"
notepad "package.json"
notepad "app\layout.tsx"
notepad "app\page.tsx"
notepad "app\globals.css"
notepad "src\config\project.ts"
notepad "contracts\README.md"
notepad "scripts\check-foundation.mjs"
notepad "services\backend\pyproject.toml"
notepad "services\backend\src\nifty_terminal\__init__.py"
notepad "services\backend\src\nifty_terminal\settings.py"
notepad "services\backend\tests\test_settings.py"
notepad "tests\rendered-html.test.mjs"
```

Do not replace these files with partial snippets. If an edit is requested in a
later step, the entire replacement file will be supplied.

## 8. Optional Git initialization

Only run this if the extracted folder is not already a Git repository:

```powershell
git init
git add .
git commit -m "Complete Stage 3 Step 1 foundation"
```

If Git requests your identity, configure it with your own name and email before retrying the commit.

## Step 1 completion condition

Step 1 is complete when the frontend checks, production build, rendered-page
test, and Python tests all pass. Do not add market-provider code, WebSockets,
candles, models, indicators, or signals until Step 2 is explicitly started.
