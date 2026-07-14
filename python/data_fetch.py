"""
data_fetch.py
-------------
Reusable data-fetching module for pulling cleaned / star-schema data out of
Azure SQL into pandas DataFrames for the ML pipeline (Isolation Forest,
XGBoost, rule-based layer) and for Power BI-adjacent exports.

Why this is a separate module from db_connection.py:
  db_connection.py only knows how to CONNECT.
  data_fetch.py knows how to ASK FOR DATA in the shapes the rest of the
  pipeline needs (full table, filtered query, chunked for memory safety).
  Keeping fetch logic here means the ML notebooks and the Streamlit app
  can both import from this file without duplicating query strings.

Libraries used:
  - pandas        -> read_sql_query() loads query results directly into
                     DataFrames (and can return a chunked generator)
  - sqlalchemy    -> text() wraps raw SQL safely so we can pass
                     parameters instead of f-string interpolation
  - db_connection -> our own module, supplies the engine
"""

import pandas as pd
from sqlalchemy import text
from db_connection import get_engine

def fetch_query(query: str, params:dict = None, chunksize: int = None):
  """
  pd.read_sql_query() Runs an arbitrary SQL query and returns the result as a DataFrame
  (or a generator of DataFrames if chunksize is set).

  Parameters:
      query (str): raw SQL query string. Use :param_name placeholders for
                   parameters (see `params`), NOT f-string interpolation
                   of user-provided values, to avoid SQL injection.
      params (dict, optional): values matched to :param_name placeholders.
      chunksize (int, optional): if set, returns a generator yielding
                   DataFrames of this many rows at a time. Use this for
                   the full 2.83M-row fetch to avoid loading everything
                   into memory at once.

  Returns:
      pd.DataFrame, or a generator of pd.DataFrame if chunksize is set.

  Example:
      df = fetch_query(
          "SELECT TOP 1000 * FROM fact_flows WHERE label = :label",
          params={"label": "DDoS"}
      )
  """
  engine = get_engine()
  conn = engine.connect()
  if chunksize:
      return pd.read_sql_query(text(query), conn, params=params, chunksize=chunksize)
  with conn:
      return pd.read_sql_query(text(query), conn, params=params)


def fetch_table(
      table_name: str,
      columns: list = None,
      where: str = None,
      limit: int = None
) -> pd.DataFrame:
  """
  Convenience wrapper to fetch a full table (or a filtered subset) without
  hand-writing SQL every time.

  Parameters:
      table_name (str): name of the table/view to fetch from,
                         e.g. "fact_flows", "dim_protocol"
      columns (list[str], optional): specific columns to select.
                         Defaults to all columns (*).
      where (str, optional): raw SQL WHERE clause condition, without the
                         "WHERE" keyword, e.g. "label != 'BENIGN'"
      limit (int, optional): caps rows returned. Useful for quick sampling
                         before committing to a full 2.83M-row run.

  Returns:
      pd.DataFrame

  Example:
      # Quick sample of attack flows only
      df = fetch_table("fact_flows", where="label != 'BENIGN'", limit=5000)
  """
  col_str = ", ".join(columns) if columns else "*"
  top_clause = f"TOP {limit}" if limit else ""
  query = f"SELECT {top_clause}{col_str} FROM {table_name}"
  if where:
      query+= f" WHERE {where}"
  return fetch_query(query)

def fetch_training_data(chunksize: int = 100_000, view_name: str = "vw_ml_training_data"):
  """
  Fetches the full cleaned dataset in chunks, shaped for the ML training
  pipeline (i.e. the star schema already joined back into one flat table).

  Assumes a SQL view (default "vw_ml_training_data") exists that joins
  fact_flows with its dimension tables into one flat feature table --
  keeping that join logic in SQL rather than pandas, consistent with the
  "all cleaning/joining deferred to SQL" approach used for staging.

  Parameters:
      chunksize (int): rows per chunk. Default 100,000. With 78 columns,
                        this is a safe starting point on a typical student
                        laptop; lower it if you hit memory errors.
      view_name (str): name of the flat training view/table to pull from.

  Returns:
      Generator yielding pd.DataFrame chunks. Concatenate with pd.concat()
      only if your machine can hold the full ~2.83M x 78 table in memory.

  Example:
      chunks = fetch_training_data(chunksize=200_000)
      full_df = pd.concat(chunks, ignore_index=True)
  """
  query = f"SELECT * FROM {view_name}"
  return fetch_query(query, chunksize=chunksize)

if __name__ == "__main__":
  sample = fetch_table("dbo.cic_typed", limit=10)
  print(sample.head())
  print(f"Shape: {sample.shape}")