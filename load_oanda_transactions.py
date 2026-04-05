#!/usr/bin/env python3
"""
Daily sync: load new OANDA transactions after the latest id in BigQuery in batches of 300,
repeated until caught up with the API: TRUNCATE staging, load batch, MERGE into
oanda.transactions (skip existing ids). Repeats until no new ids remain.

Uses only the standard library plus requests, google-cloud-bigquery, and python-dotenv
(no pandas/numpy/pyarrow) so macOS code-signing issues with scientific wheels do not apply.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

PROJECT_ID = "bold-artifact-312304"
DATASET_ID = "oanda"
TABLE_ID = "transactions"
STAGING_ID = "transactions_staging"
BATCH_SIZE = 300

SCRIPT_DIR = Path(__file__).resolve().parent


def _clean_env_value(name: str, required: bool = True) -> str:
    raw = os.environ.get(name)
    if raw is None:
        if required:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return ""

    value = raw.strip()
    # Allow users to paste quoted values in .env/GitHub Secrets.
    if (
        len(value) >= 2
        and ((value[0] == "'" and value[-1] == "'") or (value[0] == '"' and value[-1] == '"'))
    ):
        value = value[1:-1].strip()

    if required and not value:
        raise RuntimeError(f"Environment variable {name} is empty after trimming.")
    return value


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


def _flatten_transaction(txn: dict) -> dict:
    txn_copy = dict(txn)
    txn_copy.pop("tradesClosed", None)
    full_price = txn_copy.get("fullPrice")
    if full_price and isinstance(full_price, dict):
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
    return flat


def _to_int(x: Any, default: int = 0) -> int:
    if x is None or x == "":
        return default
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return default


def _to_float(x: Any, default: float = 0.0) -> float:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_bq_datetime(s: str | None) -> datetime | None:
    """Parse OANDA-style timestamp into naive UTC wall clock for BigQuery DATETIME."""
    if not s or not isinstance(s, str):
        return None
    s = truncate_timestamp_to_seconds(s.strip())
    if not s:
        return None
    s_iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s_iso)
    except ValueError:
        try:
            return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalize_row(flat: dict) -> dict[str, Any] | None:
    raw_id = flat.get("id")
    if raw_id is None:
        return None
    try:
        tid = int(round(float(raw_id)))
    except (TypeError, ValueError):
        return None

    def g(key: str, default: Any = None) -> Any:
        v = flat.get(key)
        return default if v is None else v

    fp_raw = g("fullPrice_timestamp")
    if fp_raw is None or fp_raw == "":
        fp_s = "1111-11-11T11:11:11"
    else:
        fp_s = truncate_timestamp_to_seconds(str(fp_raw)) or "1111-11-11T11:11:11"

    time_raw = g("time")
    time_s = truncate_timestamp_to_seconds(str(time_raw)) if time_raw is not None else ""

    ab = g("accountBalance")
    if ab is None:
        account_balance = -1.0
    else:
        account_balance = _to_float(ab, -1.0)

    return {
        "id": tid,
        "accountID": str(g("accountID", "") or ""),
        "userID": _to_int(g("userID"), 0),
        "time": _parse_bq_datetime(time_s) if time_s else None,
        "batchID": _to_int(g("batchID"), 0),
        "orderID": _to_int(g("orderID"), 0),
        "tradeID": _to_int(g("tradeID"), 0),
        "tradeOpened_tradeID": _to_int(g("tradeOpened_tradeID"), 0),
        "requestID": _to_int(g("requestID"), 0),
        "tradeCloseTransactionID": _to_int(g("tradeCloseTransactionID"), 0),
        "positionFill": str(g("positionFill", "") or ""),
        "closedTradeID": _to_int(g("closedTradeID"), 0),
        "type": str(g("type", "") or ""),
        "accountBalance": account_balance,
        "pl": _to_float(g("pl"), 0.0),
        "instrument": str(g("instrument", "") or ""),
        "units": _to_int(g("units"), 0),
        "requestedUnits": _to_int(g("requestedUnits"), 0),
        "tradeOpened_units": _to_int(g("tradeOpened_units"), 0),
        "price": _to_float(g("price"), 0.0),
        "halfSpreadCost": _to_float(g("halfSpreadCost"), 0.0),
        "fullVWAP": _to_float(g("fullVWAP"), 0.0),
        "reason": str(g("reason", "") or ""),
        "fullPrice_timestamp": _parse_bq_datetime(fp_s),
        "tradeOpened_initialMarginRequired": _to_float(
            g("tradeOpened_initialMarginRequired"), 0.0
        ),
        "_loadedAt": datetime.now(timezone.utc),
    }


def transaction_records(payload: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for txn in payload.get("transactions") or []:
        if not isinstance(txn, dict):
            continue
        norm = _normalize_row(_flatten_transaction(txn))
        if norm is not None:
            rows.append(norm)
    return rows


def _is_missing_json_value(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return True
    return False


def rows_for_bigquery_json(normalized: list[dict[str, Any]]) -> list[dict[str, object]]:
    type_by_col = {f.name: f.field_type for f in transactions_schema()}
    out: list[dict[str, object]] = []

    for row in normalized:
        rec: dict[str, object] = {}
        for name, val in row.items():
            ft = type_by_col.get(name, "STRING")
            if _is_missing_json_value(val):
                rec[name] = None
                continue

            if ft == "TIMESTAMP":
                if not isinstance(val, datetime):
                    rec[name] = None
                    continue
                ts = val
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
                rec[name] = ts.strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC"
            elif ft == "DATETIME":
                if not isinstance(val, datetime):
                    rec[name] = None
                else:
                    rec[name] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif ft == "INTEGER":
                rec[name] = _to_int(val, 0)
            elif ft == "FLOAT":
                try:
                    fv = float(val)
                    if math.isnan(fv) or math.isinf(fv):
                        rec[name] = None
                    else:
                        rec[name] = fv
                except (TypeError, ValueError):
                    rec[name] = None
            else:
                rec[name] = str(val)
        out.append(rec)
    return out


def load_records_to_staging(client: bigquery.Client, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{STAGING_ID}"
    rows = rows_for_bigquery_json(records)
    job = client.load_table_from_json(
        rows,
        table_ref,
        job_config=bigquery.LoadJobConfig(
            schema=transactions_schema(),
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
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

    token = _clean_env_value("OANDA_ACCESS_TOKEN")
    account_id = _clean_env_value("OANDA_ACCOUNT_ID")
    api_base = _clean_env_value("OANDA_API_BASE", required=False) or "https://api-fxtrade.oanda.com"

    headers = {"Authorization": f"Bearer {token}"}
    client = bq_client()

    ensure_dataset(client)
    ensure_table(client, TABLE_ID)
    ensure_table(client, STAGING_ID)

    start = max_stored_transaction_id(client, account_id) + 1
    staging_ref = f"`{PROJECT_ID}.{DATASET_ID}.{STAGING_ID}`"

    while True:
        last_api = oanda_last_transaction_id(api_base, account_id, headers)
        if start > last_api:
            break

        end = min(start + BATCH_SIZE - 1, last_api)
        payload = fetch_transaction_range(api_base, account_id, headers, start, end)
        records = transaction_records(payload)

        if not records:
            break

        client.query(f"TRUNCATE TABLE {staging_ref}").result()
        load_records_to_staging(client, records)
        merge_staging_into_transactions(client)
        print(f"Rows added to transactions: {len(records)}")

        start = end + 1


if __name__ == "__main__":
    main()
