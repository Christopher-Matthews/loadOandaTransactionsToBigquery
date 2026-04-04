#!/usr/bin/env python3
"""
Daily sync: fetch up to 100 new OANDA transactions after the latest id in BigQuery,
load into staging, MERGE into oanda.transactions (skip existing ids).
Requires: pip install google-cloud-bigquery pandas requests python-dotenv pyarrow
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

PROJECT_ID = "bold-artifact-312304"
DATASET_ID = "oanda"
TABLE_ID = "transactions"
STAGING_ID = "transactions_staging"

SCRIPT_DIR = Path(__file__).resolve().parent


def _credentials_path() -> str:
    return os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", str(SCRIPT_DIR / "ba.json"))


def bq_client() -> bigquery.Client:
    return bigquery.Client.from_service_account_json(_credentials_path())


def transactions_schema() -> list[bigquery.SchemaField]:
    return [
        bigquery.SchemaField("id", "INTEGER"),
        bigquery.SchemaField("accountID", "STRING"),
        bigquery.SchemaField("userID", "INTEGER"),
        bigquery.SchemaField("time", "DATETIME"),
        bigquery.SchemaField("batchID", "INTEGER"),
        bigquery.SchemaField("orderID", "INTEGER"),
        bigquery.SchemaField("tradeID", "INTEGER"),
        bigquery.SchemaField("tradeOpened_tradeID", "INTEGER"),
        bigquery.SchemaField("requestID", "INTEGER"),
        bigquery.SchemaField("tradeCloseTransactionID", "INTEGER"),
        bigquery.SchemaField("positionFill", "STRING"),
        bigquery.SchemaField("closedTradeID", "INTEGER"),
        bigquery.SchemaField("type", "STRING"),
        bigquery.SchemaField("accountBalance", "FLOAT"),
        bigquery.SchemaField("pl", "FLOAT"),
        bigquery.SchemaField("instrument", "STRING"),
        bigquery.SchemaField("units", "INTEGER"),
        bigquery.SchemaField("requestedUnits", "INTEGER"),
        bigquery.SchemaField("tradeOpened_units", "INTEGER"),
        bigquery.SchemaField("price", "FLOAT"),
        bigquery.SchemaField("halfSpreadCost", "FLOAT"),
        bigquery.SchemaField("fullVWAP", "FLOAT"),
        bigquery.SchemaField("reason", "STRING"),
        bigquery.SchemaField("fullPrice_timestamp", "DATETIME"),
        bigquery.SchemaField("tradeOpened_initialMarginRequired", "FLOAT"),
        bigquery.SchemaField("_loadedAt", "TIMESTAMP"),
    ]


def ensure_dataset(client: bigquery.Client) -> None:
    ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    try:
        client.get_dataset(ref)
    except NotFound:
        client.create_dataset(bigquery.Dataset(ref))


def ensure_table(client: bigquery.Client, table_id: str) -> None:
    ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID).table(table_id)
    try:
        client.get_table(ref)
    except NotFound:
        client.create_table(bigquery.Table(ref, schema=transactions_schema()))


def max_stored_transaction_id(client: bigquery.Client, account_id: str) -> int:
    sql = f"""
    SELECT COALESCE(MAX(id), 0) AS max_id
    FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
    WHERE accountID = @acct
    """
    job = client.query(
        sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("acct", "STRING", account_id),
            ]
        ),
    )
    row = next(iter(job.result()))
    return int(row.max_id)


def oanda_last_transaction_id(api_base: str, account_id: str, headers: dict) -> int:
    r = requests.get(
        f"{api_base.rstrip('/')}/v3/accounts/{account_id}/transactions",
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return int(r.json()["lastTransactionID"])


def fetch_transaction_range(
    api_base: str, account_id: str, headers: dict, from_id: int, to_id: int
) -> dict:
    r = requests.get(
        f"{api_base.rstrip('/')}/v3/accounts/{account_id}/transactions/"
        f"idrange?from={from_id}&to={to_id}",
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def truncate_timestamp_to_seconds(ts: str | None) -> str:
    if not ts or not isinstance(ts, str):
        return ts or ""
    return ts.split(".")[0] if "." in ts else ts


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    out: dict = {}
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk, sep))
        else:
            out[nk] = v
    return out


def transactions_dataframe(payload: dict) -> pd.DataFrame:
    transactions = payload.get("transactions") or []
    main_records: list[dict] = []

    for txn in transactions:
        txn_copy = txn.copy()
        txn_copy.pop("tradesClosed", None)
        full_price = txn_copy.get("fullPrice")
        if full_price:
            simple = {k: v for k, v in full_price.items() if k not in ("bids", "asks")}
            fp_flat = {f"fullPrice_{k}": v for k, v in simple.items()}
        else:
            fp_flat = {}
        txn_copy.pop("fullPrice", None)

        flat: dict = {}
        for key, value in txn_copy.items():
            if isinstance(value, dict):
                flat.update(flatten_dict(value, parent_key=key))
            else:
                flat[key] = value
        flat.update(fp_flat)
        main_records.append(flat)

    if not main_records:
        return pd.DataFrame()

    df = pd.DataFrame(main_records)
    cols = [
        "id",
        "accountID",
        "userID",
        "time",
        "batchID",
        "orderID",
        "tradeID",
        "tradeOpened_tradeID",
        "requestID",
        "tradeCloseTransactionID",
        "positionFill",
        "closedTradeID",
        "type",
        "accountBalance",
        "pl",
        "instrument",
        "units",
        "requestedUnits",
        "tradeOpened_units",
        "price",
        "halfSpreadCost",
        "fullVWAP",
        "reason",
        "fullPrice_timestamp",
        "tradeOpened_initialMarginRequired",
    ]
    df = df.reindex(columns=cols)
    df["userID"] = df["userID"].fillna(0)
    df["batchID"] = df["batchID"].fillna(0)
    df["type"] = df["type"].fillna("").astype(str)
    df["accountID"] = df["accountID"].fillna("").astype(str)

    df["fullPrice_timestamp"] = df["fullPrice_timestamp"].fillna("1111-11-11T11:11:11")
    df["orderID"] = df["orderID"].fillna(0)
    df["instrument"] = df["instrument"].fillna("")
    df["closedTradeID"] = df["closedTradeID"].fillna(0)
    df["tradeCloseTransactionID"] = df["tradeCloseTransactionID"].fillna(0)
    df["requestID"] = df["requestID"].fillna(0)
    df["tradeOpened_initialMarginRequired"] = df["tradeOpened_initialMarginRequired"].fillna(0)
    df["positionFill"] = df["positionFill"].fillna("")
    df["tradeOpened_tradeID"] = df["tradeOpened_tradeID"].fillna(0)
    df["tradeOpened_units"] = df["tradeOpened_units"].fillna(0)
    df["tradeID"] = df["tradeID"].fillna(0)
    df["fullVWAP"] = df["fullVWAP"].fillna(0)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").round().fillna(0).astype("Int64")
    df["requestedUnits"] = df["requestedUnits"].fillna(0)
    df["price"] = df["price"].fillna(0)
    df["pl"] = df["pl"].fillna(0)
    df["halfSpreadCost"] = df["halfSpreadCost"].fillna(0)
    df["reason"] = df["reason"].fillna("")
    df["accountBalance"] = df["accountBalance"].fillna(-1.0)

    df["time"] = df["time"].map(truncate_timestamp_to_seconds)
    df["fullPrice_timestamp"] = df["fullPrice_timestamp"].map(truncate_timestamp_to_seconds)

    loaded = datetime.now(timezone.utc)
    df["_loadedAt"] = loaded

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["fullPrice_timestamp"] = pd.to_datetime(df["fullPrice_timestamp"], errors="coerce")

    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"])
    df["id"] = df["id"].astype("int64")
    int_cols = [
        "userID",
        "batchID",
        "orderID",
        "tradeID",
        "tradeOpened_tradeID",
        "requestID",
        "tradeCloseTransactionID",
        "closedTradeID",
        "units",
        "requestedUnits",
        "tradeOpened_units",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    float_cols = [
        "accountBalance",
        "pl",
        "price",
        "halfSpreadCost",
        "fullVWAP",
        "tradeOpened_initialMarginRequired",
    ]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")

    return df


def load_dataframe_to_staging(client: bigquery.Client, df: pd.DataFrame) -> None:
    if df.empty:
        return
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{STAGING_ID}"
    job = client.load_table_from_dataframe(
        df,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            schema=transactions_schema(),
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        ),
    )
    job.result()


def merge_staging_into_transactions(client: bigquery.Client) -> None:
    fields = [f.name for f in transactions_schema()]
    insert_cols = ", ".join(f"`{c}`" for c in fields)
    values = ", ".join(f"S.`{c}`" for c in fields)
    sql = f"""
    MERGE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` T
    USING `{PROJECT_ID}.{DATASET_ID}.{STAGING_ID}` S
    ON T.id = S.id
    WHEN NOT MATCHED THEN
      INSERT ({insert_cols})
      VALUES ({values})
    """
    client.query(sql).result()


def main() -> None:
    load_dotenv(SCRIPT_DIR / ".env")

    token = os.environ["OANDA_ACCESS_TOKEN"]
    account_id = os.environ["OANDA_ACCOUNT_ID"]
    api_base = os.environ.get("OANDA_API_BASE") or "https://api-fxtrade.oanda.com"

    headers = {"Authorization": f"Bearer {token}"}
    client = bq_client()

    ensure_dataset(client)
    ensure_table(client, TABLE_ID)
    ensure_table(client, STAGING_ID)

    max_id = max_stored_transaction_id(client, account_id)
    start = max_id + 1
    last_api = oanda_last_transaction_id(api_base, account_id, headers)

    if start > last_api:
        return

    end = min(start + 99, last_api)
    payload = fetch_transaction_range(api_base, account_id, headers, start, end)
    df = transactions_dataframe(payload)

    if df.empty:
        return

    staging_ref = f"`{PROJECT_ID}.{DATASET_ID}.{STAGING_ID}`"
    client.query(f"TRUNCATE TABLE {staging_ref}").result()
    load_dataframe_to_staging(client, df)
    merge_staging_into_transactions(client)


if __name__ == "__main__":
    main()
