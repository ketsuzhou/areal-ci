#!/usr/bin/env python3
"""Read and write rows in the Supabase `query_bank` table.

Configuration is read from environment variables:

    SUPABASE_URL
    SUPABASE_ANON_KEY
    SUPABASE_SERVICE_ROLE_KEY  # optional, only used with --service-role

Examples:

    uv run python query_bank_client.py list --limit 5
    uv run python query_bank_client.py get --eq id 123
    uv run python query_bank_client.py insert --data '{"query":"hello","answer":"world"}'
    uv run python query_bank_client.py update --eq id 123 --data '{"answer":"updated"}'
    uv run python query_bank_client.py delete --eq id 123
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_TABLE = "query_bank"


def _json_scalar(value: str) -> Any:
    """Parse JSON scalars when possible, otherwise keep the original string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_json(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)


def _client(use_service_role: bool, explicit_key: str | None) -> Any:
    try:
        from supabase import create_client
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing Python package `supabase`. Run with the project environment, "
            "for example: uv run python query_bank_client.py ..."
        ) from exc

    url = os.getenv("SUPABASE_URL")
    if not url:
        raise SystemExit("Missing SUPABASE_URL")

    if explicit_key:
        key = explicit_key
    elif use_service_role:
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not key:
            raise SystemExit("Missing SUPABASE_SERVICE_ROLE_KEY")
    else:
        key = os.getenv("SUPABASE_ANON_KEY")
        if not key:
            raise SystemExit("Missing SUPABASE_ANON_KEY")

    return create_client(url, key)


def _apply_eq_filters(query: Any, eq_filters: list[list[str]] | None) -> Any:
    for column, value in eq_filters or []:
        query = query.eq(column, _json_scalar(value))
    return query


def _require_filter(args: argparse.Namespace) -> None:
    if not args.eq:
        raise SystemExit("Refusing to modify all rows. Add at least one --eq COLUMN VALUE filter.")


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def cmd_list(client: Any, args: argparse.Namespace) -> None:
    query = client.table(args.table).select(args.select)
    query = _apply_eq_filters(query, args.eq)

    if args.order:
        column = args.order.removeprefix("-")
        query = query.order(column, desc=args.order.startswith("-"))
    if args.limit is not None:
        query = query.limit(args.limit)
    if args.offset is not None:
        query = query.offset(args.offset)

    _print_json(query.execute().data)


def cmd_get(client: Any, args: argparse.Namespace) -> None:
    if not args.eq:
        raise SystemExit("get requires at least one --eq COLUMN VALUE filter")

    query = client.table(args.table).select(args.select)
    query = _apply_eq_filters(query, args.eq).limit(args.limit)
    _print_json(query.execute().data)


def cmd_insert(client: Any, args: argparse.Namespace) -> None:
    payload = _load_json(args.data)
    _print_json(client.table(args.table).insert(payload).execute().data)


def cmd_upsert(client: Any, args: argparse.Namespace) -> None:
    payload = _load_json(args.data)
    query = client.table(args.table).upsert(payload, on_conflict=args.on_conflict)
    _print_json(query.execute().data)


def cmd_update(client: Any, args: argparse.Namespace) -> None:
    _require_filter(args)
    payload = _load_json(args.data)
    query = client.table(args.table).update(payload)
    query = _apply_eq_filters(query, args.eq)
    _print_json(query.execute().data)


def cmd_delete(client: Any, args: argparse.Namespace) -> None:
    _require_filter(args)
    query = client.table(args.table).delete()
    query = _apply_eq_filters(query, args.eq)
    _print_json(query.execute().data)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--table", default=DEFAULT_TABLE, help=f"Table name. Default: {DEFAULT_TABLE}")
    parser.add_argument(
        "--service-role",
        action="store_true",
        help="Use SUPABASE_SERVICE_ROLE_KEY instead of SUPABASE_ANON_KEY.",
    )
    parser.add_argument("--key", help="Explicit Supabase API key. Overrides environment keys.")


def _add_read_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--eq",
        action="append",
        nargs=2,
        metavar=("COLUMN", "VALUE"),
        help="Equality filter. Can be repeated. VALUE may be a JSON scalar.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_common(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Read rows from the table.")
    list_parser.add_argument("--select", default="*", help="Columns to select. Default: *")
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--offset", type=int)
    list_parser.add_argument("--order", help="Column to order by. Prefix with '-' for descending.")
    _add_read_filters(list_parser)
    list_parser.set_defaults(func=cmd_list)

    get_parser = subparsers.add_parser("get", help="Read matching rows.")
    get_parser.add_argument("--select", default="*", help="Columns to select. Default: *")
    get_parser.add_argument("--limit", type=int, default=1)
    _add_read_filters(get_parser)
    get_parser.set_defaults(func=cmd_get)

    insert_parser = subparsers.add_parser("insert", help="Insert one object or a list of objects.")
    insert_parser.add_argument("--data", required=True, help="JSON object/list, or @path/to/file.json")
    insert_parser.set_defaults(func=cmd_insert)

    upsert_parser = subparsers.add_parser("upsert", help="Insert or update one object or a list of objects.")
    upsert_parser.add_argument("--data", required=True, help="JSON object/list, or @path/to/file.json")
    upsert_parser.add_argument("--on-conflict", help="Comma-separated conflict target columns.")
    upsert_parser.set_defaults(func=cmd_upsert)

    update_parser = subparsers.add_parser("update", help="Update rows matching --eq filters.")
    update_parser.add_argument("--data", required=True, help="JSON object, or @path/to/file.json")
    _add_read_filters(update_parser)
    update_parser.set_defaults(func=cmd_update)

    delete_parser = subparsers.add_parser("delete", help="Delete rows matching --eq filters.")
    _add_read_filters(delete_parser)
    delete_parser.set_defaults(func=cmd_delete)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    client = _client(args.service_role, args.key)
    args.func(client, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
