# Upstream AReaL Catch-Up (v1.0.2 → v2.0.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Catch up the local AReaL fork to upstream `areal-project/AReaL` v2.0.0 via a comprehensive 3-way patch from base `bf9b3c3b` to `upstream/main`, resolving conflicts area-by-area with user sign-off at each file.

**Architecture:** Single `git diff bf9b3c3b..upstream/main` patch applied with `git apply --3way`. No shared git history with upstream (fork was a fresh init), so merge/rebase are not viable. Conflicts walked by area (PPO first, then Megatron, vLLM, FSDP, V2.0 reorg, SWE, AgentService, CLI, Docs, Test/CI, Config), one commit per area on branch `catchup/upstream-v2.0.0` off `master`.

**Tech Stack:** git, `git apply --3way`, `uv run ruff check`, `uv run pytest`.

## Global Constraints

- **Base commit:** `bf9b3c3b` (2026-03-17, ~v1.0.2). **Target:** `upstream/main` (v2.0.0, `9cf6852d`).
- **Working branch:** `catchup/upstream-v2.0.0`, created off `master`. Never push to `master` directly.
- **Local customizations preserved untouched:** `multica/`, `River2.0/`, `customized_areal/`, `openspec/`, `.claude/`, `.kiro/`, `.codex/`, `.opencode/`, `.pi/`, `.superpowers/`, `.agents/`, SuperNode code, env-dispatch, tree_search, query_bank_client.py.
- **Dependency versions:** default-keep local pins for `sglang`, `vllm`, `megatron-core`. Upstream version bumps in `pyproject.toml`, `pyproject.vllm.toml`, `Dockerfile`, `uv.lock` are NOT applied unless user explicitly approves at Task 11.
- **Stop at each conflict:** show user local side, upstream side, proposed resolution. User confirms before `git add`.
- **Ruff:** `uv run ruff check <touched files>` after each area (check only; no auto-fix during walk).
- **Tests:** GPU-free subset of `uv run pytest tests/` after all areas. GPU/distributed tests flagged as skipped, not failed. No `pytest` markers for GPU in this repo — filter by name pattern: `-k "not gpu and not distributed and not multi_node"`.
- **Conflict markers:** none may remain in the working tree at commit time. Verify with `git diff --check` before each commit.
- **Commit message style:** Conventional Commits, `feat(areal): catch up <area> upstream (v1.0.2 → v2.0.0)`, ~72-char subject.
- **No history reunification:** this catch-up does not make future `git merge upstream/main` work. Out of scope.

---

## File Structure

This plan does not create new files (other than the patch file in `/tmp`). It modifies existing files in `areal/`, `tests/`, `examples/`, `docs/`, and root config files by applying upstream's delta. The patch file `/tmp/catchup.patch` is generated in Task 1 and consumed in Task 2.

**Conflict surface (539 files changed on both sides):**

| Directory | Conflicted files | Tasks |
| --- | --- | --- |
| `areal/algorithm/`, `areal/engine/` PPO paths | ~30 | Task 3 (PPO) |
| `areal/engine/megatron_utils/` | ~15 | Task 4 (Megatron) |
| `areal/engine/vllm_engine.py`, `areal/engine/vllm_utils/` | ~10 | Task 5 (vLLM) |
| `areal/engine/fsdp_utils/` | ~8 | Task 6 (FSDP) |
| `areal/experimental/` → `areal/v2/` moves | ~20 | Task 7 (V2.0 reorg) |
| `areal/swe/`, `examples/swe/` (new) | ~5 | Task 8 (SWE-bench) |
| `areal/agent_service/` (new), `examples/agent_service/` | ~3 | Task 9 (AgentService) |
| `areal/cli/` | ~5 | Task 10 (CLI) |
| `docs/`, `README.md` | ~36 | Task 11 (Docs) |
| `tests/`, `.github/` | ~78 | Task 12 (Test/CI) |
| `pyproject.toml`, `pyproject.vllm.toml`, `Dockerfile`, `uv.lock`, `.pre-commit-config.yaml`, `.gitignore` | ~6 | Task 13 (Config) |
| Remaining `areal/` files | ~288 total overlap; remainder here | Task 14 (Other) |

(Counts are approximate; the patch application determines the exact set.)

---

### Task 1: Prep working tree and generate patch

**Files:**
- Stash: `customized_areal/tree_search/README.md`, `customized_areal/tree_search/agents/README.md` (modified, unrelated to catch-up)
- Create: `/tmp/catchup.patch`
- Create branch: `catchup/upstream-v2.0.0` off `master`

**Interfaces:**
- Consumes: `upstream/main` remote (already fetched), base commit `bf9b3c3b`
- Produces: `/tmp/catchup.patch` (consumed by Task 2), branch `catchup/upstream-v2.0.0`

- [ ] **Step 1: Verify upstream remote and base commit**

Run:
```bash
cd /workspaces/leagent/backend/areal
git remote -v | grep upstream
git log -1 --format="%h %s" bf9b3c3b
git log -1 --format="%h %s" upstream/main
git merge-base HEAD upstream/main || echo "NO COMMON ANCESTOR (expected)"
```
Expected: `upstream` remote listed; `bf9b3c3b` resolves to "chore(deps): bump sglang, vllm, megatron-core..."; `upstream/main` resolves to a v2.0.0 commit; "NO COMMON ANCESTOR (expected)".

- [ ] **Step 2: Stash unrelated working-tree modifications**

Run:
```bash
git stash push -m "pre-catchup: tree_search README mods" \
  customized_areal/tree_search/README.md \
  customized_areal/tree_search/agents/README.md
git status --short
```
Expected: no `M` entries for the two README files. Untracked tooling dirs (`.claude/`, `.kiro/`, etc.) remain — they are not touched by the patch.

- [ ] **Step 3: Create catch-up branch off master**

Run:
```bash
git checkout -b catchup/upstream-v2.0.0 master
git status --short
git log --oneline -1
```
Expected: on branch `catchup/upstream-v2.0.0`; clean working tree (ignoring untracked tooling dirs); HEAD at master's tip.

- [ ] **Step 4: Generate the comprehensive patch**

Run:
```bash
git diff bf9b3c3b..upstream/main > /tmp/catchup.patch
wc -l /tmp/catchup.patch
grep -c "^diff --git" /tmp/catchup.patch
```
Expected: patch file with ~981 `diff --git` entries (one per changed file), hundreds of thousands of lines. Record the exact `diff --git` count for Task 2 verification.

- [ ] **Step 5: Commit prep metadata (optional)**

No commit needed — branch is at master's tip, patch is in `/tmp` (not tracked). Proceed to Task 2.

---

### Task 2: Apply patch with 3-way merge

**Files:**
- Modify: all files in the conflict surface (per File Structure table)
- Input: `/tmp/catchup.patch`

**Interfaces:**
- Consumes: `/tmp/catchup.patch` from Task 1, branch `catchup/upstream-v2.0.0`
- Produces: working tree with clean applies + conflicted files (with conflict markers). List of conflicted file paths (consumed by Tasks 3–14).

- [ ] **Step 1: Apply the patch with 3-way merge**

Run:
```bash
cd /workspaces/leagent/backend/areal
git apply --3way /tmp/catchup.patch 2>&1 | tee /tmp/apply.log
echo "exit: $?"
```
Expected: `git apply --3way` exits non-zero if any conflict cannot be auto-3way-merged, but partial application still occurs. The `2>&1 | tee` captures which files conflicted. Common output lines: `error: patch failed: <file>:<line>` and `Using index info to reconstruct a base tree...` and `Falling back to patching base and 3-way merge...` and `CONFLICT (content): Merge conflict in <file>`.

Note: `git apply --3way` does NOT use the index for conflict resolution the way `git merge` does. If `--3way` fails outright on some files, fall back to `--reject` to get `.rej` files for manual application:
```bash
git apply --3way --reject /tmp/catchup.patch 2>&1 | tee /tmp/apply.log
```

- [ ] **Step 2: Inventory the conflict surface**

Run:
```bash
# Files with conflict markers in the working tree
grep -rEl "^<<<<<<< |^>>>>>>> " --include="*.py" --include="*.toml" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.txt" --include="*.sh" --include="*.cfg" --include="*.ini" . 2>/dev/null | grep -v "^./.venv" | grep -v "^./.venv-test" | grep -v "^./multica" | grep -v "^./River2.0" | grep -v "^./customized_areal" | grep -v "^./openspec" | grep -v "^./.claude" | grep -v "^./.kiro" | grep -v "^./.codex" | grep -v "^./.opencode" | grep -v "^./.pi" | grep -v "^./.superpowers" | grep -v "^./.agents" | sort > /tmp/conflicted.txt
wc -l /tmp/conflicted.txt

# Files with .rej files (patch segments that didn't apply at all)
find . -name "*.rej" -not -path "./.venv/*" -not -path "./.venv-test/*" 2>/dev/null | sort > /tmp/rejected.txt
wc -l /tmp/rejected.txt

# Files cleanly applied (no conflict markers, no .rej)
git status --short | grep "^M " | wc -l
```
Expected: `/tmp/conflicted.txt` with 100–400 files (the 539 overlap minus those that 3-way merged cleanly); `/tmp/rejected.txt` small or empty; a few hundred `M ` entries in `git status`.

- [ ] **Step 3: Bucket conflicted files by area**

Run:
```bash
cd /workspaces/leagent/backend/areal
# PPO
grep -E "^./areal/(algorithm|engine/.*ppo|engine/.*train)" /tmp/conflicted.txt > /tmp/conflict-ppo.txt || true
# Megatron
grep -E "^./areal/engine/megatron" /tmp/conflicted.txt > /tmp/conflict-megatron.txt || true
# vLLM
grep -E "^./areal/engine/vllm" /tmp/conflicted.txt > /tmp/conflict-vllm.txt || true
# FSDP
grep -E "^./areal/engine/fsdp" /tmp/conflicted.txt > /tmp/conflict-fsdp.txt || true
# V2 reorg (experimental moves)
grep -E "^./areal/experimental" /tmp/conflicted.txt > /tmp/conflict-v2.txt || true
# SWE
grep -E "^./areal/swe|^./examples/swe" /tmp/conflicted.txt > /tmp/conflict-swe.txt || true
# AgentService
grep -E "^./areal/agent_service|^./examples/agent_service" /tmp/conflicted.txt > /tmp/conflict-agent.txt || true
# CLI
grep -E "^./areal/cli" /tmp/conflicted.txt > /tmp/conflict-cli.txt || true
# Docs
grep -E "^./docs|^./README.md|^./ROADMAP.md" /tmp/conflicted.txt > /tmp/conflict-docs.txt || true
# Test/CI
grep -E "^./tests|^./.github" /tmp/conflicted.txt > /tmp/conflict-test.txt || true
# Config
grep -E "^./pyproject|^./Dockerfile|^./uv.lock|^./.pre-commit|^./.gitignore|^./CLAUDE.md|^./AGENTS.md" /tmp/conflicted.txt > /tmp/conflict-config.txt || true
# Other (remaining areal/)
cat /tmp/conflicted.txt | grep -vxF -f <(cat /tmp/conflict-ppo.txt /tmp/conflict-megatron.txt /tmp/conflict-vllm.txt /tmp/conflict-fsdp.txt /tmp/conflict-v2.txt /tmp/conflict-swe.txt /tmp/conflict-agent.txt /tmp/conflict-cli.txt /tmp/conflict-docs.txt /tmp/conflict-test.txt /tmp/conflict-config.txt 2>/dev/null) > /tmp/conflict-other.txt || true

for a in ppo megatron vllm fsdp v2 swe agent cli docs test config other; do
  echo "$a: $(wc -l < /tmp/conflict-$a.txt 2>/dev/null || echo 0)"
done
```
Expected: per-area counts summing to the total in `/tmp/conflicted.txt`. Record these counts.

- [ ] **Step 4: Do NOT commit yet**

The working tree now has clean applies (staged via `git apply`'s index updates? — no, `git apply --3way` modifies the working tree and index but does not commit). Conflicts remain as conflict markers in files. We commit per-area in Tasks 3–14 after resolving.

Verify state:
```bash
git status --short | head -20
git diff --check 2>&1 | head -20
```
Expected: many `M ` and `UU ` (unmerged) entries; `git diff --check` lists lines with conflict markers.

---

### Task 3: Resolve PPO conflicts

**Files:**
- Modify: every file in `/tmp/conflict-ppo.txt` (typically `areal/algorithm/ppo.py`, `areal/algorithm/cispo.py` if added, `areal/engine/train_engine.py` PPO paths, `areal/api/algorithm_api.py`, `tests/` PPO tests)

**Interfaces:**
- Consumes: `/tmp/conflict-ppo.txt` from Task 2, working-tree conflicts
- Produces: resolved PPO files staged in index; one commit on `catchup/upstream-v2.0.0`

**Upstream PPO changes to look for (from commit log):**
- `feat(ppo): add CISPO loss surrogate (MiniMax-M1)` — new `areal/algorithm/cispo.py` or new function in `ppo.py`
- `feat(ppo): add reuse_train_logp proximal logp method` — new config flag, new code path in train step
- `fix(ppo): handle variable-size trajectory groups in reward normalization` — change to reward normalization
- `fix(ppo): coerce ppo_n_minibatches to 1 for reuse_train_logp` — config validation
- `test(ppo): update singleton leave-one-out group expectation` — test expectation change

- [ ] **Step 1: Show the user the PPO conflict list**

Run:
```bash
cat /tmp/conflict-ppo.txt
```
Show the user the file list. For each file, the workflow below applies.

- [ ] **Step 2: For each file in /tmp/conflict-ppo.txt, surface the conflict**

For the first file (e.g. `areal/algorithm/ppo.py`), run:
```bash
f="areal/algorithm/ppo.py"  # replace with actual path from list
git diff --check "$f" | head -20
grep -n "^<<<<<<< \|^=======\|^>>>>>>> " "$f" | head -30
```
Then show the user:
1. The conflicted region(s) with `<<<<<<<`, `=======`, `>>>>>>>` markers (use `Read` on the file around those line numbers).
2. Local side (`=======` to `>>>>>>>`): what the local fork has, and why (grep local git log: `git log --oneline ec8b2a4..HEAD -- "$f"`).
3. Upstream side (`<<<<<<<` to `=======`): what upstream changed, and why (grep upstream log: `git log --oneline bf9b3c3b..upstream/main -- "$f"`).
4. Proposed resolution: combine both, preferring upstream's new feature where it adds one (CISPO, reuse_train_logp) and preserving local customizations where they exist. State the reasoning in 1–2 sentences.

- [ ] **Step 3: Get user sign-off on the resolution**

Ask the user: "File `<path>`: proposed resolution is `<one-line summary>`. Approve, redirect, or show me more detail?"
Do not write until the user approves.

- [ ] **Step 4: Write the resolution**

Edit the file to remove conflict markers and apply the agreed resolution. Use `Edit` for small changes or `Write` for full rewrites.

- [ ] **Step 5: Verify no markers remain in this file**

Run:
```bash
f="areal/algorithm/ppo.py"  # actual path
grep -n "^<<<<<<< \|^=======\|^>>>>>>> " "$f" && echo "MARKERS REMAIN" || echo "clean"
```
Expected: `clean`.

- [ ] **Step 6: Stage the file**

Run:
```bash
git add "$f"
```

- [ ] **Step 7: Repeat Steps 2–6 for every file in /tmp/conflict-ppo.txt**

Iterate. For each file, surface → sign-off → write → verify → stage. Do not batch — the user approved "stop and ask at each conflict".

- [ ] **Step 8: Run ruff on touched PPO files**

Run:
```bash
uv run ruff check $(cat /tmp/conflict-ppo.txt | sed 's|^\./||')
```
Expected: no errors (warnings acceptable). If errors, fix them inline (the upstream code should already be ruff-clean; errors likely mean a misresolution).

- [ ] **Step 9: Verify no conflict markers anywhere in PPO files**

Run:
```bash
for f in $(cat /tmp/conflict-ppo.txt | sed 's|^\./||'); do
  grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null
done
```
Expected: no output.

- [ ] **Step 10: Commit the PPO area**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up PPO upstream (v1.0.2 → v2.0.0)

Bring in upstream PPO changes: CISPO loss surrogate (MiniMax-M1),
reuse_train_logp proximal logp method, variable-size trajectory
group reward normalization, ppo_n_minibatches coercion, singleton
leave-one-out test expectation. Local customizations preserved.

Conflicts resolved per-file with sign-off.
EOF
)"
```
Expected: one commit on `catchup/upstream-v2.0.0`.

---

### Task 4: Resolve Megatron conflicts

**Files:**
- Modify: every file in `/tmp/conflict-megatron.txt` (typically `areal/engine/megatron_utils/`, `areal/engine/megatron_engine.py`)

**Interfaces:**
- Consumes: `/tmp/conflict-megatron.txt`
- Produces: resolved Megatron files staged; one commit

**Upstream Megatron changes to look for:**
- `feat(megatron): add CP-safe vocab stats and MoE config support`
- `feat(megatron): make MTP head opt-in to support Qwen3.6 MoE RL`

- [ ] **Step 1: Show user the Megatron conflict list**

Run: `cat /tmp/conflict-megatron.txt`

- [ ] **Step 2: For each file, surface conflict (local side, upstream side, proposed resolution)**

Same workflow as Task 3 Step 2. Use:
```bash
git log --oneline ec8b2a4..HEAD -- "$f"   # local why
git log --oneline bf9b3c3b..upstream/main -- "$f"  # upstream why
```

- [ ] **Step 3: Get user sign-off**

Same as Task 3 Step 3.

- [ ] **Step 4: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 5: Repeat for every file in /tmp/conflict-megatron.txt**

- [ ] **Step 6: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-megatron.txt | sed 's|^\./||')`

- [ ] **Step 7: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-megatron.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`
Expected: no output.

- [ ] **Step 8: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up Megatron upstream (v1.0.2 → v2.0.0)

CP-safe vocab stats, MoE config support, opt-in MTP head for
Qwen3.6 MoE RL. Local customizations preserved.
EOF
)"
```

---

### Task 5: Resolve vLLM conflicts

**Files:**
- Modify: every file in `/tmp/conflict-vllm.txt` (typically `areal/engine/vllm_engine.py`, `areal/engine/vllm_utils/`)

**Interfaces:**
- Consumes: `/tmp/conflict-vllm.txt`
- Produces: resolved vLLM files staged; one commit

**Upstream vLLM changes to look for:**
- `fix(vllm): forward frequency_penalty and stop in generation requests`

- [ ] **Step 1: Show user the vLLM conflict list**

Run: `cat /tmp/conflict-vllm.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3.

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-vllm.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-vllm.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-vllm.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up vLLM upstream (v1.0.2 → v2.0.0)

Forward frequency_penalty and stop in generation requests.
Local customizations preserved.
EOF
)"
```

---

### Task 6: Resolve FSDP conflicts

**Files:**
- Modify: every file in `/tmp/conflict-fsdp.txt` (typically `areal/engine/fsdp_utils/`, `areal/engine/fsdp_engine.py`)

**Interfaces:**
- Consumes: `/tmp/conflict-fsdp.txt`
- Produces: resolved FSDP files staged; one commit

- [ ] **Step 1: Show user the FSDP conflict list**

Run: `cat /tmp/conflict-fsdp.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3.

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-fsdp.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-fsdp.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-fsdp.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up FSDP upstream (v1.0.2 → v2.0.0)

Local customizations preserved.
EOF
)"
```

---

### Task 7: Resolve V2.0 structural reorg conflicts

**Files:**
- Modify: every file in `/tmp/conflict-v2.txt` (typically `areal/experimental/` modules being moved to `areal/v2/`)
- Possibly Create: new paths under `areal/v2/`

**Interfaces:**
- Consumes: `/tmp/conflict-v2.txt`
- Produces: resolved V2 reorg files staged; one commit

**Upstream V2.0 changes to look for:**
- `refactor: move 5 experimental modules into areal/v2 for 2.0 release`
- `feat: support v2 weight update disk mode for lora RL`

**⚠️ Highest-risk task.** Upstream moves files; if any of those files are among the 244 locally-modified `areal/` files, the resolution requires moving the file to its new upstream path AND reapplying local mods to that new path. Surface each such file to the user before proceeding.

- [ ] **Step 1: Show user the V2 reorg conflict list**

Run: `cat /tmp/conflict-v2.txt`

- [ ] **Step 2: Cross-reference moved files against local mods**

For each file in `/tmp/conflict-v2.txt`, check whether local has modified it:
```bash
for f in $(cat /tmp/conflict-v2.txt | sed 's|^\./||'); do
  local_commits=$(git log --oneline ec8b2a4..HEAD -- "$f" 2>/dev/null | wc -l)
  upstream_commits=$(git log --oneline bf9b3c3b..upstream/main -- "$f" 2>/dev/null | wc -l)
  echo "$f  local=$local_commits  upstream=$upstream_commits"
done
```
Files with `local > 0` are the judgment-heavy ones. Flag them to the user first.

- [ ] **Step 3: For each file, surface conflict, propose resolution (move + reapply mods), get sign-off**

For files upstream moved (path A → path B):
1. Show user the old path (`areal/experimental/X.py`) and new path (`areal/v2/X.py`).
2. Show local mods to the old path (`git log --oneline ec8b2a4..HEAD -- areal/experimental/X.py` and `git diff ec8b2a4..HEAD -- areal/experimental/X.py`).
3. Show upstream's changes to the file at the new path.
4. Proposed resolution: create the new path with upstream's content + local mods reapplied; delete the old path. State reasoning.
5. User sign-off required.

For files upstream modified in place (no move): same workflow as Task 3 Step 2.

- [ ] **Step 4: Write resolution, verify clean, stage**

For moves: `git rm` the old path, `Write` the new path with merged content, `git add` the new path. For in-place: same as Task 3 Steps 4–6.

- [ ] **Step 5: Repeat for every file in /tmp/conflict-v2.txt**

- [ ] **Step 6: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-v2.txt | sed 's|^\./||') areal/v2/`

- [ ] **Step 7: Verify no markers remain and old paths are gone**

Run:
```bash
for f in $(cat /tmp/conflict-v2.txt | sed 's|^\./||'); do
  grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null
done
# Verify experimental modules that should have moved are gone
ls areal/experimental/ 2>/dev/null
```

- [ ] **Step 8: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up V2.0 structural reorg (v1.0.2 → v2.0.0)

Move 5 experimental modules into areal/v2/. Support v2 weight update
disk mode for lora RL. Local mods reapplied to new paths.
EOF
)"
```

---

### Task 8: Resolve SWE-bench RL conflicts

**Files:**
- Modify: every file in `/tmp/conflict-swe.txt`
- Possibly Create: new files under `areal/swe/`, `examples/swe/`

**Interfaces:**
- Consumes: `/tmp/conflict-swe.txt`
- Produces: resolved SWE files staged; one commit

**Upstream SWE changes to look for:**
- `feat(swe): add SWE-bench RL training workflow`
- `feat(swe): add SWE agent proxy, message preprocessors and tool-call support`
- `chore(swe): remove unused SWE SFT dataset loader`
- `fix(openai): parse tool_call arguments from JSON string to dict before chat template`

Mostly new files; lower conflict risk.

- [ ] **Step 1: Show user the SWE conflict list**

Run: `cat /tmp/conflict-swe.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3.

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-swe.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-swe.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-swe.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up SWE-bench RL upstream (v1.0.2 → v2.0.0)

SWE-bench RL training workflow, SWE agent proxy, message
preprocessors, tool-call support. Local customizations preserved.
EOF
)"
```

---

### Task 9: Resolve AgentService conflicts

**Files:**
- Modify: every file in `/tmp/conflict-agent.txt`
- Possibly Create: new files under `areal/agent_service/`, `examples/agent_service/`

**Interfaces:**
- Consumes: `/tmp/conflict-agent.txt`
- Produces: resolved AgentService files staged; one commit

**Upstream changes to look for:**
- `feat(agent_service): add agent service with OpenClaw and Hermes examples`

Mostly new files.

- [ ] **Step 1: Show user the AgentService conflict list**

Run: `cat /tmp/conflict-agent.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3.

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-agent.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-agent.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-agent.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up agent_service upstream (v1.0.2 → v2.0.0)

Agent service with OpenClaw and Hermes examples. Local
customizations preserved.
EOF
)"
```

---

### Task 10: Resolve CLI conflicts

**Files:**
- Modify: every file in `/tmp/conflict-cli.txt` (typically `areal/cli/`)

**Interfaces:**
- Consumes: `/tmp/conflict-cli.txt`
- Produces: resolved CLI files staged; one commit

**Upstream CLI changes to look for:**
- `feat(cli): add training service cli`
- `feat(cli): add agent service cli`
- `feat(cli): add inference service cli`
- `feat(cli): add experimental cli scaffold for service-style subcommands`

- [ ] **Step 1: Show user the CLI conflict list**

Run: `cat /tmp/conflict-cli.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3.

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-cli.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-cli.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-cli.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up CLI upstream (v1.0.2 → v2.0.0)

Training, agent, inference service CLIs. Experimental CLI scaffold
for service-style subcommands. Local customizations preserved.
EOF
)"
```

---

### Task 11: Resolve Docs conflicts

**Files:**
- Modify: every file in `/tmp/conflict-docs.txt` (typically `docs/`, `README.md`, `ROADMAP.md`)

**Interfaces:**
- Consumes: `/tmp/conflict-docs.txt`
- Produces: resolved docs files staged; one commit

- [ ] **Step 1: Show user the Docs conflict list**

Run: `cat /tmp/conflict-docs.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3. For docs, prefer upstream's content for upstream-only features (v2.0, SWE, AgentService) and preserve local additions (customization guides, SuperNode docs, env-dispatch docs).

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-docs.txt**

- [ ] **Step 5: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-docs.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 6: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
docs(areal): catch up docs/README to v2.0.0

Upstream v2.0 docs (algorithms, customization, CLI reference).
Local customization docs preserved.
EOF
)"
```

---

### Task 12: Resolve Test/CI conflicts

**Files:**
- Modify: every file in `/tmp/conflict-test.txt` (typically `tests/`, `.github/`)

**Interfaces:**
- Consumes: `/tmp/conflict-test.txt`
- Produces: resolved test/CI files staged; one commit

- [ ] **Step 1: Show user the Test/CI conflict list**

Run: `cat /tmp/conflict-test.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3. For tests, prefer upstream's expectation for upstream features; preserve local tests for local features (SuperNode, env-dispatch).

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-test.txt**

- [ ] **Step 5: Run ruff on touched test files**

Run: `uv run ruff check $(cat /tmp/conflict-test.txt | sed 's|^\./||' | grep '\.py$')`

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-test.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
test(areal): catch up tests/CI to v2.0.0

Upstream test updates (PPO, Megatron, vLLM, SWE, AgentService).
Local tests for SuperNode, env-dispatch, tree_search preserved.
EOF
)"
```

---

### Task 13: Resolve Config conflicts

**Files:**
- Modify: every file in `/tmp/conflict-config.txt` (typically `pyproject.toml`, `pyproject.vllm.toml`, `Dockerfile`, `uv.lock`, `.pre-commit-config.yaml`, `.gitignore`, `CLAUDE.md`, `AGENTS.md`)

**Interfaces:**
- Consumes: `/tmp/conflict-config.txt`
- Produces: resolved config files staged; one commit

**⚠️ Default resolution: keep local.** Upstream bumps `sglang`, `vllm`, `megatron-core` versions and changes the Dockerfile structure. Local has its own pins and a separate `pyproject.vllm.toml` flow. Unless the user explicitly approves taking an upstream version bump, keep local.

- [ ] **Step 1: Show user the Config conflict list**

Run: `cat /tmp/conflict-config.txt`

- [ ] **Step 2: For each file, surface conflict with default recommendation**

For each file, show:
1. Local side and upstream side.
2. Default recommendation: keep local (for `pyproject.toml`, `pyproject.vllm.toml`, `Dockerfile`, `uv.lock`); take upstream (for `.gitignore`, `.pre-commit-config.yaml` if upstream adds entries that don't conflict); merge (for `CLAUDE.md`, `AGENTS.md` — preserve local customizations, add upstream's new sections).
3. Ask user to confirm or override.

- [ ] **Step 3: Get user sign-off per file**

Same as Task 3 Step 3. The user must explicitly approve any dependency version bump.

- [ ] **Step 4: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 5: Repeat for every file in /tmp/conflict-config.txt**

- [ ] **Step 6: Verify no markers remain**

Run: `for f in $(cat /tmp/conflict-config.txt | sed 's|^\./||'); do grep -l "^<<<<<<< \|^>>>>>>> " "$f" 2>/dev/null; done`

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore(areal): catch up config (pyproject/Dockerfile/uv.lock) — local pins kept

Local dependency pins preserved by default. Upstream additions to
.gitignore, .pre-commit-config.yaml, CLAUDE.md, AGENTS.md merged.
EOF
)"
```

---

### Task 14: Resolve remaining areal/ conflicts

**Files:**
- Modify: every file in `/tmp/conflict-other.txt` (remaining `areal/` files not bucketed into Tasks 3–10)

**Interfaces:**
- Consumes: `/tmp/conflict-other.txt`
- Produces: resolved remaining files staged; one commit

- [ ] **Step 1: Show user the remaining conflict list**

Run: `cat /tmp/conflict-other.txt`

- [ ] **Step 2: For each file, surface conflict and get sign-off**

Same workflow as Task 3 Steps 2–3. Group by subdirectory if the list is long (e.g. `areal/api/`, `areal/dataset/`, `areal/reward/`, `areal/utils/`, `areal/infra/`).

- [ ] **Step 3: Write resolution, verify clean, stage**

Same as Task 3 Steps 4–6.

- [ ] **Step 4: Repeat for every file in /tmp/conflict-other.txt**

- [ ] **Step 5: Run ruff on touched files**

Run: `uv run ruff check $(cat /tmp/conflict-other.txt | sed 's|^\./||')`

- [ ] **Step 6: Verify no markers remain anywhere in the repo**

Run:
```bash
git diff --check 2>&1 | head -20
grep -rEl "^<<<<<<< |^>>>>>>> " --include="*.py" --include="*.toml" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.txt" --include="*.sh" . 2>/dev/null | grep -v "^./.venv" | grep -v "^./.venv-test" | grep -v "^./multica" | grep -v "^./River2.0" | grep -v "^./customized_areal" | grep -v "^./openspec" | grep -v "^./.claude" | grep -v "^./.kiro" | grep -v "^./.codex" | grep -v "^./.opencode" | grep -v "^./.pi" | grep -v "^./.superpowers" | grep -v "^./.agents"
```
Expected: both empty.

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
feat(areal): catch up remaining areal/ upstream (v1.0.2 → v2.0.0)

Resolve remaining conflicts in areal/api, areal/dataset, areal/reward,
areal/utils, areal/infra. Local customizations preserved.
EOF
)"
```

---

### Task 15: Final verification

**Files:**
- Verify: entire working tree, all commits on `catchup/upstream-v2.0.0`

**Interfaces:**
- Consumes: branch `catchup/upstream-v2.0.0` with all area commits
- Produces: verification report; branch ready for user review

- [ ] **Step 1: Verify no conflict markers anywhere**

Run:
```bash
cd /workspaces/leagent/backend/areal
git diff --check 2>&1 | head
grep -rEl "^<<<<<<< |^>>>>>>> " --include="*.py" --include="*.toml" --include="*.md" --include="*.yaml" --include="*.yml" --include="*.txt" --include="*.sh" . 2>/dev/null | grep -vE "^\./(\.venv|\.venv-test|multica|River2.0|customized_areal|openspec|\.claude|\.kiro|\.codex|\.opencode|\.pi|\.superpowers|\.agents)"
```
Expected: both empty.

- [ ] **Step 2: Verify local customizations intact**

Run:
```bash
# Local-only directories still present
ls -d multica River2.0 customized_areal openspec .agents 2>/dev/null
# Local customization files still present
ls customized_areal/tree_search/ 2>/dev/null | head
ls areal/workflow/ 2>/dev/null | grep -i supernode
git log --oneline catchup/upstream-v2.0.0 | grep -iE "supernode|env-dispatch|tree_search|multica" | head
```
Expected: local dirs present; local customization files present; local commits still in history below the catch-up commits.

- [ ] **Step 3: Run ruff on the whole areal/ package**

Run:
```bash
uv run ruff check areal/ tests/ 2>&1 | tail -30
```
Expected: no errors (warnings acceptable). Fix any errors introduced by misresolutions.

- [ ] **Step 4: Run GPU-free pytest subset**

Run:
```bash
uv run pytest tests/ -k "not gpu and not distributed and not multi_node" --timeout=120 -q 2>&1 | tail -50
```
Expected: passes or fails for pre-existing reasons. Flag any new failures introduced by the catch-up (e.g. `ImportError` from upstream API changes the local code calls — these are follow-up work, not catch-up blockers).

- [ ] **Step 5: Show the user the catch-up branch summary**

Run:
```bash
git log --oneline master..catchup/upstream-v2.0.0
git diff --stat master..catchup/upstream-v2.0.0 | tail -5
```
Show the user the commit list and total stat. Ask for final review.

- [ ] **Step 6: Restore stashed README modifications**

Run:
```bash
git stash pop
git status --short
```
Expected: the two `customized_areal/tree_search/README*` modifications restored. If they conflict with anything (they shouldn't — they were stashed before the catch-up and the catch-up didn't touch them), resolve manually.

- [ ] **Step 7: Report to user**

Report:
- Branch `catchup/upstream-v2.0.0` ready with N commits.
- All conflicts resolved with sign-off.
- Ruff clean on `areal/` and `tests/`.
- GPU-free pytest results (pass / fail-with-pre-existing-reasons / fail-with-new-reasons).
- Local customizations intact.
- Next step: user reviews the diff; squash if desired; merge to `master` on user's schedule (out of scope for this plan).

---

## Self-Review

**1. Spec coverage:**
- Spec §"Approach → 1. Prep the working tree" → Task 1 ✓
- Spec §"Approach → 2. Generate the comprehensive patch" → Task 1 Step 4 ✓
- Spec §"Approach → 3. Apply with 3-way merge" → Task 2 ✓
- Spec §"Approach → 4. Walk conflicts by area, stop at each" → Tasks 3–14, with PPO first per user choice ✓
- Spec §"Approach → 5. Commit in logical chunks" → one commit per task (area) ✓
- Spec §"Approach → 6. Sanity checks" → per-area ruff (Steps in each task) + final pytest (Task 15) ✓
- Spec §"Risks → V2.0 reorg judgment-heavy" → Task 7 Step 2 cross-references moved files against local mods ✓
- Spec §"Risks → Dependency versions" → Task 13 default-keep-local ✓
- Spec §"Risks → uv.lock noisy" → Task 13 Step 2 default-keep-local-uv.lock ✓
- Spec §"Success criteria" → Task 15 verifies each criterion ✓
- Spec §"Out of scope" → not implemented (correctly) ✓

**2. Placeholder scan:** No "TBD", "TODO", "implement later". Each step has exact commands. Commit messages are full templates. The one judgment-call placeholder is "propose a resolution" in conflict-walk steps — this is inherent to the task (the resolution depends on the actual conflict content, which is only known after applying the patch) and is bounded by the "show user local side / upstream side / proposed resolution / get sign-off" workflow.

**3. Type consistency:** No types/functions defined (this is a patch-apply plan, not a code-implementation plan). File-path variables (`$f`) used consistently. Conflict-list filenames (`/tmp/conflict-<area>.txt`) consistent across tasks.

**4. Scope check:** Single subsystem (the catch-up). No decomposition needed.

Plan is internally consistent and covers the spec.
