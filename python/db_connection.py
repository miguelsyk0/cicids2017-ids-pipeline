"""
db_connection.py
----------------
Reusable Azure SQL Database connection module for the ITE 17 IDS project.

Why this exists:
    Every script that touches the database (staging imports, cleaning jobs,
    feature fetches for ML, Streamlit prototype, etc.) needs the exact same
    connection setup. Centralizing it here means:
      - One place to fix connection issues (e.g. the Proxy/redirect port
        issue we hit on campus/home networks)
      - One place to manage credentials via .env (never hardcoded)
      - Consistent use of fast_executemany for bulk inserts

Libraries used:
    - os            -> reading environment variables
    - dotenv         -> loads .env file into environment variables
    - urllib.parse   -> quote_plus() safely URL-encodes the connection
                        string (needed because SQL passwords often contain
                        special characters like @, #, ! that break URLs)
    - sqlalchemy     -> the actual DB engine/connection pooling layer. We use
                        SQLAlchemy (not raw pyodbc) because pandas.read_sql()
                        and pandas.to_sql() both expect a SQLAlchemy engine.
    - pyodbc         -> the low-level ODBC driver SQLAlchemy uses under the
                        hood to talk to Azure SQL (installed as a dependency,
                        not imported directly here, but required system-wide)
"""

import os
from dotenv import load_dotenv
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Load variables from .env into the environment.
# .env should live in the project root and NEVER be committed to GitHub.
load_dotenv()


def get_connection_string() -> str:
    """
    Builds the ODBC connection string for Azure SQL Database using
    credentials stored in .env.

    Required .env variables:
        AZURE_SQL_SERVER    e.g. "your-server.database.windows.net"
        AZURE_SQL_DATABASE  e.g. "ids_project_db"
        AZURE_SQL_USERNAME  SQL authentication username
        AZURE_SQL_PASSWORD  SQL authentication password
        AZURE_SQL_DRIVER    e.g. "ODBC Driver 18 for SQL Server"

    Returns:
        str: a fully encoded SQLAlchemy-compatible connection URL
    """
    server = os.getenv("AZURE_SQL_SERVER")
    database = os.getenv("AZURE_SQL_DATABASE")
    username = os.getenv("AZURE_SQL_USERNAME")
    password = os.getenv("AZURE_SQL_PASSWORD")
    driver = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    if not all([server, database, username, password]):
        raise EnvironmentError(
            "Missing one or more required .env variables: "
            "AZURE_SQL_SERVER, AZURE_SQL_DATABASE, AZURE_SQL_USERNAME, AZURE_SQL_PASSWORD"
        )

    # quote_plus escapes special characters in the ODBC string so it
    # doesn't break the connection URL (e.g. '@' or '#' in a password).
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER=tcp:{server},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    connection_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_str)}"
    return connection_url


def get_engine(echo: bool = False) -> Engine:
    """
    Creates and returns a SQLAlchemy engine connected to Azure SQL.

    Uses fast_executemany=True so that bulk INSERT operations (e.g. pushing
    cleaned pandas DataFrames back into SQL) run in batches instead of
    row-by-row, which matters given the ~2.83M row CICIDS2017 dataset.

    Parameters:
        echo (bool): if True, SQLAlchemy logs every SQL statement it runs.
                     Useful for debugging, noisy for normal use. Default False.

    Returns:
        sqlalchemy.engine.Engine
    """
    connection_url = get_connection_string()
    engine = create_engine(
        connection_url,
        fast_executemany=True,
        echo=echo,
        pool_pre_ping=True,   # checks connection is alive before using it;
                              # prevents "stale connection" errors on reruns
    )
    return engine


def test_connection() -> bool:
    """
    Quick sanity check to confirm the DB is reachable before running a
    full script. Prints the SQL Server version if successful.

    Returns:
        bool: True if connection succeeded, False otherwise.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT @@VERSION"))
            version = result.scalar()
            print("Connected successfully.")
            print(version)
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    # Run this file directly (`python db_connection.py`) to test the connection
    test_connection()