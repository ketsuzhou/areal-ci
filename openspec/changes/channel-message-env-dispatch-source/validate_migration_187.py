#!/usr/bin/env python3
"""Structural validation for migration 187 (build check).

This sandbox has no `go`/`pssql`/live Postgres, so the real multica build
(`make migrate-up`, `go test`) cannot run here. This script validates the
migration artifacts structurally: the up migration admits `env_dispatch`, the
down migration rewrites `env_dispatch` rows to `multica` and re-tightens the
CHECK, statements are balanced and semicolon-terminated. Green end-to-end
verification (apply migration on a live server, re-run the client) is recorded
as a deferred manual step in the verification report.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # change dir -> changes -> openspec -> areal
BASE = REPO_ROOT / "multica/server/migrations/187_channel_message_env_dispatch_source"
errors: list[str] = []


def check(label: str, cond: bool) -> None:
    if not cond:
        errors.append(label)


up = (BASE.with_suffix(".up.sql")).read_text()
down = (BASE.with_suffix(".down.sql")).read_text()

check("up: ALTER TABLE channel_message", "ALTER TABLE channel_message" in up)
check("up: DROP CONSTRAINT IF EXISTS channel_message_source_check",
      "DROP CONSTRAINT IF EXISTS channel_message_source_check" in up)
check("up: ADD CONSTRAINT channel_message_source_check",
      "ADD CONSTRAINT channel_message_source_check" in up)
check("up: CHECK admits env_dispatch",
      re.search(r"CHECK\s*\(\s*source IN \(\s*'multica'\s*,\s*'lark'\s*,\s*'env_dispatch'\s*\)\s*\)", up) is not None)
check("down: UPDATE env_dispatch -> multica",
      re.search(r"UPDATE channel_message SET source = 'multica' WHERE source = 'env_dispatch'", down) is not None)
add_part = down.split("ADD CONSTRAINT", 1)[1] if "ADD CONSTRAINT" in down else ""
check("down: re-tighten CHECK excludes env_dispatch",
      re.search(r"CHECK\s*\(\s*source IN \(\s*'multica'\s*,\s*'lark'\s*\)\s*\)", add_part) is not None
      and "'env_dispatch'" not in add_part)
for name, txt in (("up", up), ("down", down)):
    check(f"{name}: balanced parens", txt.count("(") == txt.count(")"))
    check(f"{name}: statements end with semicolon", txt.strip().endswith(";"))

if errors:
    print("FAIL:", "; ".join(errors))
    sys.exit(1)
print("PASS: migration 187 up/down structurally valid - up admits env_dispatch, "
      "down rewrites+re-tightens, balanced, semicolon-terminated")
