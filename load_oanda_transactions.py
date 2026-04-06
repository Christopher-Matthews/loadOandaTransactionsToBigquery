#!/usr/bin/env python3
"""
Incremental OANDA transactions sync with event-leg expansion.

Flow (unchanged pattern):
- Pull transaction id ranges in batches.
- TRUNCATE staging.
- Load batch rows into staging.
- MERGE staging -> production (insert-only dedupe).

Schema approach:
- Single table, but ORDER_FILL events can expand into multiple logical rows:
  - OPEN legs (from tradeOpened)
  - CLOSE legs (one row per tradesClosed[] item)
  - OTHER for all remaining transaction events
- Missing fields remain NULL (no dummy placeholders).
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
        bigquery.SchemaField("transactionRowKey", "STRING"),
        bigquery.SchemaField("id", "INTEGER"),
        bigquery.SchemaField("accountID", "STRING"),
        bigquery.SchemaField("userID", "INTEGER"),
        bigquery.SchemaField("time", "DATETIME"),
        bigquery.SchemaField("eventType", "STRING"),
        bigquery.SchemaField("openTime", "DATETIME"),
        bigquery.SchemaField("closeTime", "DATETIME"),
        bigquery.SchemaField("realizedPL", "FLOAT"),
        bigquery.SchemaField("batchID", "INTEGER"),
        bigquery.SchemaField("orderID", "INTEGER"),
        bigquery.SchemaField("linkedOrderID", "INTEGER"),
        bigquery.SchemaField("tradeID", "INTEGER"),
        bigquery.SchemaField("linkedTradeID", "INTEGER"),
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
        bigquery.SchemaField("rawTransactionJson", "JSON"),
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


def oanda_last_transaction_id(api_base: str, account_id: str, headers: dict[str, str]) -> int:
    r = requests.get(
        f"{api_base.rstrip('/')}/v3/accounts/{account_id}/transactions",
        headers=headers,
        timeout=60,
    )
    r.raise_for_status()
    return int(r.json()["lastTransactionID"])


def fetch_transaction_range(
    api_base: str, account_id: str, headers: dict[str, str], from_id: int, to_id: int
) -> dict[str, Any]:
    r = requests.get(
        f"{api_base.rstrip('/')}/v3/accounts/{account_id}/transactions/"
        f"idrange?from={from_id}&to={to_id}",
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def _to_int(x: Any, default: int | None = None) -> int | None:
    if x is None or x == "":
        return default
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return default


def _to_float(x: Any, default: float | None = None) -> float | None:
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _to_str(x: Any, default: str | None = None) -> str | None:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _parse_bq_datetime(s: str | None) -> datetime | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    # OANDA often sends nanosecond precision; truncate to seconds for DATETIME.
    if "." in s:
        s = s.split(".", 1)[0] + ("Z" if s.endswith("Z") else "")

    s_iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s_iso)
    except ValueError:
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            return dt
        except ValueError:
            return None

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _base_row(txn: dict[str, Any], loaded_at: datetime) -> dict[str, Any] | None:
    tid = _to_int(txn.get("id"))
    if tid is None:
        return None

    t_dt = _parse_bq_datetime(_to_str(txn.get("time")))
    full_price = txn.get("fullPrice") if isinstance(txn.get("fullPrice"), dict) else None
    full_price_ts = _parse_bq_datetime(_to_str(full_price.get("timestamp"))) if full_price else None
    trade_opened = txn.get("tradeOpened") if isinstance(txn.get("tradeOpened"), dict) else None

    return {
        "transactionRowKey": None,
        "id": tid,
        "accountID": _to_str(txn.get("accountID")),
        "userID": _to_int(txn.get("userID")),
        "time": t_dt,
        "eventType": None,
        "openTime": None,
        "closeTime": None,
        "realizedPL": None,
        "batchID": _to_int(txn.get("batchID")),
        "orderID": _to_int(txn.get("orderID")),
        "linkedOrderID": _to_int(txn.get("orderID")),
        "tradeID": _to_int(txn.get("tradeID")),
        "linkedTradeID": _to_int(txn.get("tradeID")),
        "tradeOpened_tradeID": _to_int(trade_opened.get("tradeID")) if trade_opened else None,
        "requestID": _to_int(txn.get("requestID")),
        "tradeCloseTransactionID": _to_int(txn.get("tradeCloseTransactionID")),
        "positionFill": _to_str(txn.get("positionFill")),
        "closedTradeID": _to_int(txn.get("closedTradeID")),
        "type": _to_str(txn.get("type")),
        "accountBalance": _to_float(txn.get("accountBalance")),
        "pl": _to_float(txn.get("pl")),
        "instrument": _to_str(txn.get("instrument")),
        "units": _to_int(txn.get("units")),
        "requestedUnits": _to_int(txn.get("requestedUnits")),
        "tradeOpened_units": _to_int(trade_opened.get("units")) if trade_opened else None,
        "price": _to_float(txn.get("price")),
        "halfSpreadCost": _to_float(txn.get("halfSpreadCost")),
        "fullVWAP": _to_float(txn.get("fullVWAP")),
        "reason": _to_str(txn.get("reason")),
        "fullPrice_timestamp": full_price_ts,
        "tradeOpened_initialMarginRequired": _to_float(trade_opened.get("initialMarginRequired"))
        if trade_opened
        else None,
        "rawTransactionJson": txn,
        "_loadedAt": loaded_at,
    }


def _open_leg_row(base: dict[str, Any], txn: dict[str, Any]) -> dict[str, Any] | None:
    trade_opened = txn.get("tradeOpened") if isinstance(txn.get("tradeOpened"), dict) else None
    if not trade_opened:
        return None

    trade_id = _to_int(trade_opened.get("tradeID"))
    row = dict(base)
    row["eventType"] = "OPEN"
    row["openTime"] = base.get("time")
    row["linkedTradeID"] = trade_id
    row["tradeID"] = trade_id
    row["tradeOpened_tradeID"] = trade_id
    row["tradeOpened_units"] = _to_int(trade_opened.get("units"))
    row["tradeOpened_initialMarginRequired"] = _to_float(trade_opened.get("initialMarginRequired"))
    if _to_float(trade_opened.get("price")) is not None:
        row["price"] = _to_float(trade_opened.get("price"))
    if _to_float(trade_opened.get("halfSpreadCost")) is not None:
        row["halfSpreadCost"] = _to_float(trade_opened.get("halfSpreadCost"))
    key_trade = trade_id if trade_id is not None else "na"
    row["transactionRowKey"] = f"{base['id']}:OPEN:{key_trade}"
    return row


def _close_leg_rows(base: dict[str, Any], txn: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    closed = txn.get("tradesClosed")
    if not isinstance(closed, list):
        return rows

    for idx, item in enumerate(closed):
        if not isinstance(item, dict):
            continue
        trade_id = _to_int(item.get("tradeID"))
        row = dict(base)
        row["eventType"] = "CLOSE"
        row["closeTime"] = base.get("time")
        row["linkedTradeID"] = trade_id
        row["closedTradeID"] = trade_id
        row["tradeID"] = trade_id
        row["realizedPL"] = _to_float(item.get("realizedPL"))
        row["pl"] = _to_float(item.get("realizedPL"))
        if _to_int(item.get("units")) is not None:
            row["units"] = _to_int(item.get("units"))
        if _to_float(item.get("price")) is not None:
            row["price"] = _to_float(item.get("price"))
        if _to_float(item.get("halfSpreadCost")) is not None:
            row["halfSpreadCost"] = _to_float(item.get("halfSpreadCost"))

        key_trade = trade_id if trade_id is not None else "na"
        row["transactionRowKey"] = f"{base['id']}:CLOSE:{key_trade}:{idx}"
        rows.append(row)
    return rows


def _other_event_row(base: dict[str, Any], txn: dict[str, Any]) -> dict[str, Any]:
    row = dict(base)
    row["eventType"] = "OTHER"

    if row.get("linkedTradeID") is None:
        row["linkedTradeID"] = _to_int(txn.get("closedTradeID"))
    if row.get("tradeCloseTransactionID") is None:
        row["tradeCloseTransactionID"] = _to_int(txn.get("tradeCloseTransactionID"))

    t = row.get("type") or "UNKNOWN"
    row["transactionRowKey"] = f"{base['id']}:OTHER:{t}"
    return row


def transaction_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    loaded_at = datetime.now(timezone.utc)

    for txn in payload.get("transactions") or []:
        if not isinstance(txn, dict):
            continue

        base = _base_row(txn, loaded_at)
        if base is None:
            continue

        tx_type = _to_str(txn.get("type"))
        if tx_type == "ORDER_FILL":
            open_row = _open_leg_row(base, txn)
            close_rows = _close_leg_rows(base, txn)

            if open_row is not None:
                out.append(open_row)
            if close_rows:
                out.extend(close_rows)
            if open_row is None and not close_rows:
                out.append(_other_event_row(base, txn))
        else:
            out.append(_other_event_row(base, txn))

    return out


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
                ts = val.astimezone(timezone.utc) if val.tzinfo else val.replace(tzinfo=timezone.utc)
                rec[name] = ts.strftime("%Y-%m-%d %H:%M:%S.%f") + " UTC"
            elif ft == "DATETIME":
                if not isinstance(val, datetime):
                    rec[name] = None
                else:
                    rec[name] = val.strftime("%Y-%m-%d %H:%M:%S")
            elif ft == "INTEGER":
                rec[name] = _to_int(val)
            elif ft == "FLOAT":
                rec[name] = _to_float(val)
            elif ft == "JSON":
                rec[name] = val if isinstance(val, (dict, list, str, int, float, bool)) else None
            else:
                rec[name] = _to_str(val)
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
    ON T.transactionRowKey = S.transactionRowKey
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

        print(f"Loaded transaction range {start}-{end}: {len(records)} rows")
        start = end + 1


if __name__ == "__main__":
    main()
