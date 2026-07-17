"""
Import CICIDS2017 MachineLearningCSV files into Azure SQL Database, AS-IS.

WHAT THIS SCRIPT DOES (and does NOT do)
----------------------------------------
Same philosophy as the Supabase version: this is a pure "load raw data" step.
No cleaning, no dropping, no fixing values. The only transformation is making
column headers SQL-safe. Cleaning happens later in a separate T-SQL script.

WHY THIS SCRIPT LOOKS DIFFERENT FROM THE SUPABASE VERSION
------------------------------------------------------------
Azure SQL Database (SQL Server) is a different engine from Postgres, so:
  - It uses pyodbc instead of psycopg2
  - It has a hard limit of 2,100 parameters PER STATEMENT (much stricter
    than Postgres's 65,535). With ~80 columns, that's only ~26 rows per
    multi-row INSERT -- too small to be practical.
  - Instead of method="multi", this script uses pyodbc's fast_executemany
    feature, which batches rows efficiently at the driver level without
    hitting that per-statement parameter limit.

BEFORE YOU RUN THIS
--------------------
1. Install the ODBC Driver 18 for SQL Server (this is a system driver, not
   a pip package):
   - Windows: usually already present, or download from Microsoft's site
     ("ODBC Driver 18 for SQL Server download")
   - Mac: brew install msodbcsql18 (via the microsoft/mssql-release tap)
   - Linux: follow Microsoft's apt/yum instructions for msodbcsql18

2. Install Python dependencies:
   pip install pandas sqlalchemy pyodbc

3. In the Azure portal, go to your SQL Server resource -> Networking ->
   Firewall rules -> add a rule allowing your (and your teammates') public
   IP address(es). Without this, the connection will be refused entirely.
   For a class project, it's common (though not best practice) to
   temporarily allow a wide IP range so every teammate can connect without
   individually adding IPs -- just be aware this is less secure.

4. Get your connection details from the Azure portal (SQL Database ->
   Overview -> Connection strings), you'll need:
   - Server name: yourserver.database.windows.net
   - Database name
   - Username / password (SQL authentication)

5. Set CSV_FOLDER and the connection details below.

6. Run:
   python import_raw_to_azuresql.py
"""

import os
import re
import glob
import pandas as pd
import pyodbc
from urllib.parse import quote_plus
from sqlalchemy import create_engine, event

# ------------------------------------------------------------------
# CONFIG -- EDIT THESE
# ------------------------------------------------------------------
AZURE_SERVER = "cic-ids2017-sql.database.windows.net"
AZURE_DATABASE = "cic-ids2017"
AZURE_USERNAME = "cic-ids2017-admin"
AZURE_PASSWORD = "ZZs5^@x63%J'dnT"
ODBC_DRIVER = "ODBC Driver 18 for SQL Server"  # must match what you installed

CSV_FOLDER = r"C:\Users\Academics\Downloads\GeneratedLabelledFlows\TrafficLabelling"  # folder containing the 8 CSVs

TABLE_NAME = "raw_flows"
READ_CHUNK_SIZE = 20000   # rows read from CSV at a time
INSERT_BATCH_SIZE = 1000  # rows per executemany batch (fast_executemany
                          # handles this efficiently -- not limited by the
                          # 2,100 parameter cap since it's not building one
                          # giant multi-row VALUES statement)

# ------------------------------------------------------------------
# Map filenames to source day/file, purely as metadata (not a data "fix")
# ------------------------------------------------------------------
FILENAME_TO_DAY = {
    "monday-workinghours.pcap_iscx.csv": "Monday",
    "tuesday-workinghours.pcap_iscx.csv": "Tuesday",
    "wednesday-workinghours.pcap_iscx.csv": "Wednesday",
    "thursday-workinghours-morning-webattacks.pcap_iscx.csv": "Thursday-Morning",
    "thursday-workinghours-afternoon-infileteration.pcap_iscx.csv": "Thursday-Afternoon",
    "friday-workinghours-morning.pcap_iscx.csv": "Friday-Morning",
    "friday-workinghours-afternoon-portscan.pcap_iscx.csv": "Friday-Afternoon-PortScan",
    "friday-workinghours-afternoon-ddos.pcap_iscx.csv": "Friday-Afternoon-DDoS",
}


def sql_safe_column_name(col: str) -> str:
    """Make a header SQL-safe WITHOUT changing its meaning. No data cleanup here."""
    col = col.strip()
    col = col.lower()
    col = re.sub(r"[^\w]+", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_")
    return col


def build_engine():
    # URL-encode username/password so special characters (@, %, #, /, etc.)
    # in your Azure SQL password don't get misinterpreted as part of the
    # connection URL structure (this was the cause of the earlier
    # "server not found" error -- part of the password was leaking into
    # where the hostname should be).
    safe_username = quote_plus(AZURE_USERNAME)
    safe_password = quote_plus(AZURE_PASSWORD)

    connection_string = (
        f"mssql+pyodbc://{safe_username}:{safe_password}"
        f"@{AZURE_SERVER}:1433/{AZURE_DATABASE}"
        f"?driver={ODBC_DRIVER.replace(' ', '+')}"
        f"&Encrypt=yes&TrustServerCertificate=no"
    )
    engine = create_engine(connection_string, fast_executemany=True)

    # Belt-and-suspenders: explicitly enable fast_executemany on the pyodbc
    # cursor as well, since driver/dialect versions vary in how reliably
    # they pick this up automatically.
    @event.listens_for(engine, "before_cursor_execute")
    def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
        if executemany:
            cursor.fast_executemany = True

    return engine


def main():
    engine = build_engine()

    csv_files = sorted(glob.glob(os.path.join(CSV_FOLDER, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {CSV_FOLDER}. Check the path.")
        return

    print(f"Found {len(csv_files)} CSV files to import (raw, no cleaning).\n")

    first_chunk_overall = True
    total_rows = 0

    for filepath in csv_files:
        print(f"Reading {os.path.basename(filepath)} ...")

        reader = pd.read_csv(
            filepath,
            chunksize=READ_CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            encoding="latin1",
        )

        base = os.path.basename(filepath).lower()
        source_day = FILENAME_TO_DAY.get(base, "Unknown")

        for chunk in reader:
            chunk.columns = [sql_safe_column_name(c) for c in chunk.columns]
            chunk["source_file"] = base
            chunk["source_day"] = source_day

            if_exists_mode = "replace" if first_chunk_overall else "append"

            chunk.to_sql(
                TABLE_NAME,
                engine,
                if_exists=if_exists_mode,
                index=False,
                chunksize=INSERT_BATCH_SIZE,
                method=None,  # let pyodbc's fast_executemany do the batching
            )

            total_rows += len(chunk)
            first_chunk_overall = False
            print(f"  -> inserted {len(chunk)} rows (running total: {total_rows})")

    print(f"\nDone. {total_rows} total rows loaded into '{TABLE_NAME}' in Azure SQL.")
    print("All columns are stored as text (NVARCHAR). Cleaning/casting happens next in T-SQL.")


if __name__ == "__main__":
    main()