# Upstream AReaL Catch-Up (v1.0.2 → v2.0.0) — Design

**Date:** 2026-07-02
**Status:** Approved (brainstorming complete), pending spec review
**Base commit:** `bf9b3c3b` (2026-03-17, ~v1.0.2)
**Target:** `upstream/main` (v2.0.0, `9cf6852d`)
**Working branch:** `catchup/upstream-v2.0.0` (off `master`)

## Context

The local AReaL fork at `/workspaces/leagent/backend/areal/` was created as a fresh
git repo ("Initial commit: AReaL project", `ec8b2a4b`, 2026-03-25) seeded with an
upstream snapshot. It has **no shared git history** with `areal-project/AReaL`.
Since then, 463 local commits have layered on substantial customizations
(SuperNode Phase 1a, env-dispatch, MulticaEnvDispatchClient, swe_lego_issue_runner,
tree_search, plus the `multica/`, `River2.0/`, `customized_areal/` directories).

Upstream advanced 210 commits (v1.0.2 → v2.0.0), touching 981 files. The user wants
to catch up on upstream changes without reuniting git histories.

### Divergence measurements

- 981 files changed upstream since base
- 1156 files changed locally since initial commit
- **539 files changed on both sides** (conflict surface)
- 288 of those 539 are inside `areal/` (the rest are in `tests/`, `examples/`,
  `docs/`, `pyproject.toml`, `Dockerfile`, `uv.lock`, etc.)
- Local modifications inside `areal/`: 244 files modified, 160 added, 0 deleted
  across 45 local commits
- 442 upstream-changed files have no local modification (safe to take directly)
- 617 local-changed files have no upstream modification (pure customizations,
  untouchable)

### Upstream change areas (by commit count)

| Area | Commits | Notes |
| --- | --- | --- |
| PPO | 29 | CISPO loss surrogate, reuse_train_logp, variable-size trajectory reward normalization |
| Megatron | 12 | CP-safe vocab stats, MoE config, MTP head for Qwen3.6 |
| vLLM | 8 | frequency_penalty, stop forwarding, fixes |
| V2.0 structural reorg | 7 | 5 experimental modules moved into `areal/v2/` |
| FSDP | 6 | |
| AgentService | 4 | OpenClaw, Hermes examples |
| SWE-bench RL | 3 | New workflow, large |
| Docs | 11 | |
| Test/CI | 16 | |
| SGLang / Reward / Infra | 3 | |
| Other (feat / fix / chore) | 111 | |

## Goal

Bring the local fork up to upstream v2.0.0 for the `areal/` core, `tests/`,
`examples/`, `docs/`, and shared config files — preserving all local
customizations — through a conflict-walk that the user reviews at each
conflicted file.

## Non-goals

- **Reuniting git histories.** This catch-up does NOT make future
  `git merge upstream/main` work. Future catch-ups repeat the same patch-apply
  process unless histories are later reunited (separate, larger effort).
- **Updating dependency versions** (`sglang`, `vllm`, `megatron-core`) unless
  the user explicitly decides to at conflict time. Local pins are preserved by
  default.
- **Pulling the `multica/`, `River2.0/`, `customized_areal/`, `openspec/`,
  `.claude/`, `.kiro/`, `.codex/`, `.opencode/`, `.pi/`, `.superpowers/`,
  `.agents/` directories.** These are pure local additions with no upstream
  counterpart; the patch does not touch them.
- **GPU / distributed integration tests.** Run only the GPU-free subset of
  `pytest` as a smoke check. Flag the rest as skipped.

## Approach

### 1. Prep the working tree

- Verify `master` is clean except for the known untracked tooling dirs and the
  two modified `customized_areal/tree_search/README*` files.
- Stash or commit the two README modifications before applying the patch so
  they don't get clobbered or interleaved with catch-up commits. (Stash is
  fine; they are unrelated to upstream.)
- Create and switch to `catchup/upstream-v2.0.0` off `master`.

### 2. Generate the comprehensive patch

```
git diff bf9b3c3b..upstream/main > /tmp/catchup.patch
```

This single patch captures the full delta from the local base to upstream HEAD.

### 3. Apply with 3-way merge

```
git apply --3way /tmp/catchup.patch
```

`--3way` uses the blob hashes embedded in the patch to do a real three-way
merge per file. Files that don't conflict apply cleanly; files that do land
with standard conflict markers in the working tree.

We use `git apply --3way` rather than `git merge`/`git rebase` because the
repo has no shared history with upstream — `--allow-unrelated-histories` merge
would produce the same conflict surface but with worse tooling and a fake
merge commit that misrepresents the relationship.

### 4. Walk conflicts by area, stop at each

Group conflicted files by area and resolve in this order (per user choice,
PPO first):

1. **PPO** — `areal/algorithm/`, `areal/engine/` PPO paths, `tests/` PPO tests
2. **Megatron** — `areal/engine/megatron_utils/`
3. **vLLM** — `areal/engine/vllm_engine.py` and adapters
4. **FSDP** — `areal/engine/fsdp_utils/`
5. **V2.0 reorg** — moves into `areal/v2/` (judgment-heavy; surface for user
   decision per file)
6. **SWE-bench RL** — mostly new files, lower conflict risk
7. **AgentService** — `areal/agent_service/` (new), examples
8. **CLI** — `areal/cli/` additions
9. **Docs** — `docs/`, `README.md`
10. **Test/CI** — `.github/`, `tests/` non-PPO
11. **Config** — `pyproject.toml`, `pyproject.vllm.toml`, `Dockerfile`,
    `uv.lock`, `.pre-commit-config.yaml`, `.gitignore` (default: keep local
    pins/config, take upstream only on user signal)
12. **Other** — anything remaining

For each conflicted file, the workflow is:

1. Show the user: local side (`ours`), upstream side (`theirs`), and a
   proposed resolution with reasoning.
2. User confirms or redirects.
3. Write the resolution, `git add` the file.
4. Move to the next file in the same area.
5. When an area is fully resolved, commit with a message like
   `feat(areal): catch up <area> upstream (v1.0.2 → v2.0.0)`.

### 5. Commit in logical chunks

One commit per area. Final history on `catchup/upstream-v2.0.0`:

```
feat(areal): catch up PPO upstream (CISPO, reuse_train_logp, ...)
feat(areal): catch up Megatron upstream (CP-safe vocab, MoE, MTP, ...)
feat(areal): catch up vLLM upstream (frequency_penalty, stop, ...)
feat(areal): catch up FSDP upstream
feat(areal): catch up V2.0 structural reorg (areal/v2/)
feat(areal): catch up SWE-bench RL workflow
feat(areal): catch up agent_service (OpenClaw, Hermes)
feat(areal): catch up CLI subcommands
docs(areal): catch up docs/README to v2.0.0
test(areal): catch up tests/CI to v2.0.0
chore(areal): catch up config (pyproject/Dockerfile/uv.lock) — local pins kept
```

The user can squash before merging to `master`.

### 6. Sanity checks

- After each area: `uv run ruff check <touched files>` (format-only; do not
  auto-fix during the walk to avoid masking real conflicts).
- After all areas: `uv run pytest tests/ -k "not gpu and not distributed"`
  (or whatever the GPU-free subset is — confirm against
  `tests/conftest.py` markers at run time).
- Flag GPU/distributed tests as skipped, not failed.

## Risks

- **V2.0 reorg is judgment-heavy.** Upstream moves 5 experimental modules into
  `areal/v2/`. If any of those modules are among the 244 locally-modified
  `areal/` files, the resolution requires moving the file AND reapplying local
  mods to the new path. These get surfaced individually for the user's call
  before I proceed.
- **Dependency versions.** Upstream bumps `sglang`, `vllm`, `megatron-core`
  in `pyproject.toml` and `Dockerfile`. Local has its own pins (and a separate
  `pyproject.vllm.toml` flow). Default resolution: keep local pins, skip
  upstream version bumps. User decides at conflict time.
- **`uv.lock` will not regenerate cleanly** after a partial catch-up. We
  accept a noisy lockfile on the catch-up branch and regenerate at the end
  only if the user wants.
- **Scale.** 288+ conflicts in `areal/` alone. Realistically hours of
  back-and-forth across sessions, not minutes. The conflict walk is resumable:
  `catchup/upstream-v2.0.0` is a normal branch; conflicts already resolved
  and committed stay resolved.
- **No history reunification.** Future catch-ups repeat this process. If the
  user later wants `git merge upstream/main` to work, that's a separate
  one-time effort (rebuild local history on top of upstream, or vice versa).

## Success criteria

- `catchup/upstream-v2.0.0` branch exists, off `master`, with the catch-up as
  a series of area-scoped commits.
- All conflicts resolved with user sign-off; no conflict markers remain in
  the working tree.
- `uv run ruff check .` passes on touched files.
- GPU-free `pytest` subset passes or fails for reasons unrelated to the
  catch-up (pre-existing failures flagged separately).
- Local customizations (`multica/`, `River2.0/`, `customized_areal/`,
  `openspec/`, SuperNode, env-dispatch, tree_search) are intact and
  unchanged by the catch-up.
- User reviews the final diff before the branch is considered done.

## Out of scope (deferred)

- Merging `catchup/upstream-v2.0.0` to `master` — user's call, separate step.
- Updating local dependency versions to match upstream — deferred unless user
  explicitly asks during the config-area conflict walk.
- Reuniting git histories with upstream — separate, larger effort.
- Adapting local customizations to upstream API changes the catch-up may
  introduce (e.g., if upstream renamed a function the local code calls).
  These surface as local test failures post-catch-up and get handled as
  follow-up work, not as part of the catch-up itself.
