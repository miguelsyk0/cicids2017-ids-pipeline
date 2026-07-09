# CIC-IDS2017 Network Intrusion Detection — ITE 17 Final Project

A CRISP-DM-based analytics project using the CICIDS2017 dataset to classify
network traffic as benign or malicious, built with SQL (Azure SQL Database),
Power BI, and Python.

## Project Overview

- **Dataset:** CICIDS2017 (Canadian Institute for Cybersecurity) —
  `MachineLearningCSV` files, ~2.8M network flow records, 78+ features,
  covering DoS, DDoS, PortScan, Brute Force, Botnet, Web Attacks, and
  Infiltration traffic vs. benign traffic.
- **Methodology:** CRISP-DM (Business Understanding → Data Understanding →
  Data Preparation → EDA/Visualization → Modeling → Evaluation)
- **Stack:** Azure SQL Database, Power BI (Import mode), Python
  (pandas, scikit-learn), Streamlit (prototype)

## Repository Structure

```
.
├── data/           # Local CSVs (raw dataset) — gitignored, not committed
├── powerbi/        # Power BI .pbix dashboard file(s)
├── python/         # All Python scripts (import, cleaning, EDA, ML, Streamlit)
│   ├── db_connection.py         # Shared Azure SQL connection module
│   └── import_raw_to_azuresql.py # Raw CSV -> Azure SQL import script
├── sql/            # T-SQL scripts (schema, cleaning, star schema build)
├── .env.example    # Template for local environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

## One-Time Setup

1. **Clone the repo and install dependencies**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   pip install -r requirements.txt
   ```

2. **Install the ODBC Driver 18 for SQL Server** (system-level driver, not
   a pip package — required for Python to talk to Azure SQL):
   - Windows: usually preinstalled; otherwise download from Microsoft
   - Mac: `brew install msodbcsql18` (via the `microsoft/mssql-release` tap)
   - Linux: follow Microsoft's apt/yum instructions for `msodbcsql18`

3. **Set up your local environment file**
   ```bash
   cp .env.example .env
   ```
   Fill in the real Azure SQL credentials in `.env` (shared with the team
   privately — Discord/private doc, **never** post them in GitHub or chat
   that gets logged publicly). `.env` is gitignored and will never be
   committed.

4. **Get your IP added to the Azure SQL firewall**
   Azure blocks all external connections by default. Ask the database owner
   (see below) to add your public IP under:
   Azure Portal → SQL Server → Networking → Firewall rules.
   If your IP changes often (switching wifi/mobile data), let them know —
   you'll need to re-add it.

5. **Verify your connection works**
   ```bash
   cd python
   python db_connection.py
   ```
   You should see `Connection successful. Test query returned: (1,)`.
   If this fails, check (in order): `.env` values are correct, ODBC driver
   is installed, your IP is whitelisted, and the Azure connection policy is
   set to **Proxy** (not Redirect — some school/home networks block the
   redirect port range and cause a Post-Login timeout).

## Team Conventions

- **Database owner:** one designated teammate applies schema changes
  (`CREATE TABLE`, `ALTER`, cleaning scripts) to avoid conflicting edits.
  Everyone else connects read/query-only unless actively contributing a
  reviewed script.
- **SQL scripts are version-controlled** in `sql/` — every cleaning/
  transformation step should exist as a saved `.sql` file, not just run
  ad hoc in SSMS/Azure Data Studio, so the BRD's Data Preparation section
  has a reproducible trail.
- **Credentials never go in code.** All scripts pull connection details
  from `.env` via `python/db_connection.py` — don't hardcode server names,
  usernames, or passwords in any script you commit.
- **Power BI in Import mode**, not DirectQuery, for defense-day reliability.
  Keep a local `.pbix` with data already imported as a fallback in case
  venue wifi is unreliable.

## Current Pipeline Status

- [x] Raw CICIDS2017 CSVs imported as-is into `raw_flows` (Azure SQL),
      all columns as text, no cleaning applied yet
- [ ] SQL data cleaning (blank row removal, type casting, `Infinity`/NaN
      handling, deduplication)
- [ ] Star schema build (`fact_network_flows` + `dim_label`, `dim_protocol`,
      `dim_time`, `dim_port`)
- [ ] EDA + Power BI dashboard
- [ ] ML model development (classification)
- [ ] Streamlit prototype
- [ ] BRD documentation
- [ ] Project defense prep

## Dataset Source

CICIDS2017, Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html