# SWE-Lego Docker × Multica Remote Mode × AReaL RL — Design

**Date:** 2026-06-29
**Status:** Approved design — ready for implementation planning
**Builds on:** `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md` (the approved cloud-only DAG-RL design)
**Research basis:** `customized_areal/tree_search/agents/reward/swe_data_pipeline_comparison.md`

## 1. Goal

Enable AReaL to train RL over SWE issues using **SWE-Lego docker containers** as agent
environments, launched via **multica remote mode**, with a hybrid **SWE-Lego verifier** producing
the reward. For each issue:

1. AReal calls one atomic multica API that creates a fresh multica project and forks the repo's
   git history **before the issue** into it (SWE-Lego anti-hacking: history after the issue date
   is deleted).
2. Multica boots `group_size` daemon-in-docker sandboxes from a SWE-Lego image and starts one
   agent run per sandbox (remote mode).
3. AReal drives the existing DAG-RL branching machinery within each lane (snapshot + issue fork +
   `agent_start_branch` at `(task_id, seq)`).
4. When each lane's terminal run finishes, the hybrid verifier produces a reward, which is backed
   up through the lane's delegation DAG and **actually shapes training** (Phase 3 is load-bearing,
   not deferred).

## 2. Locked decisions

1. **Scope = per-issue rollout + mid-run branching.** The full `multica-dag-rl` machinery applies
   (snapshot-at-frontier, `ForkIssueSubtree` at `task_message.seq`, DAG credit backup), retargeted
   to SWE-Lego docker containers.
2. **Multi-agent = group_size parallel root agents + delegation DAG.** `group_size` independent
   root agents per issue form the GRPO rollout group; each root may spawn a multica delegation
   DAG (`issue.parent_issue_id`); branching happens within each lane.
3. **Multica owns docker.** AReal never calls docker directly; it calls one atomic multica endpoint
   plus the existing cloud-runtime sandbox endpoints.
4. **Daemon-in-docker (model A).** Each SWE-Lego container runs a multica daemon inside it. The
   daemon spawns the agent CLI, parses stdout, and POSTs `task_message` batches — producing the
   `seq` that mid-run `(task_id, seq)` branching cuts at.
5. **SWE-Lego anti-hacking fork.** The repo is checked out at `base_commit` and git history after
   `issue_date` is **deleted** via `git filter-repo --commit-cutoff`, so an agent cannot `git log`
   or `git blame` its way to the future fix. Enforced once at image-build time, inherited by every
   forked sandbox.
6. **Hybrid verifier + reward.** `ObjectiveVerifier` runs `FAIL_TO_PASS` / `PASS_TO_PASS` tests
   (Skywork-SWE empty-vs-gold two-run model) for a binary pass/fail; `AgenticVerifier` generative
   critic (yes/no token probability) covers what tests cannot; SWE-Lego semi-resolved partial
   credit (located the bug but did not fully fix) provides a dense signal. Reward = weighted blend.
7. **One atomic multica endpoint.** `POST /api/v1/swe-lego/issues` composes project creation + SWE-Lego
   image build + base sandbox boot + `group_size` forks + agent-run enqueue. AReal is thin.
8. **Image build runs on a Fleet node** via the existing `/api/v1/nodes` exec path. The multica
   server never shells out to `docker build` locally.
9. **Phase 3 DAG credit backup is load-bearing in v1.** The terminal verifier reward is distributed
   along DAG edges so sub-agent runs that contributed to a successful root get credit; the existing
   `TreeAdvantageComputer` is extended (not replaced) to consume per-node credit. This is in-scope
   and ships in v1 — it is not deferred, not merely logged.

## 3. Architecture

```
AReaL side (Python):
  ├── Per-issue orchestration loop (NEW)
  │   └── customized_areal/tree_search/agents/swe_lego_issue_runner.py
  │       POST /api/v1/swe-lego/issues → { project_id, base_sandbox_id, agent_run_ids[] }
  ├── ForkableEnvironment provider — NEW class (retarget)
  │   └── customized_areal/tree_search/agents/environment.py
  │       FleetSandboxProvider (existing, Fleet/Daytona)
  │       MulticaSweLegoProvider (NEW — calls multica cloud-runtime proxy)
  ├── BranchMaterializer — UNCHANGED
  │   └── customized_areal/tree_search/agents/integration.py
  │       (snapshot + ForkIssueSubtree + agent_start_branch; provider is injected)
  ├── Hybrid SWE-Lego verifier (NEW)
  │   └── customized_areal/tree_search/agents/reward/swe_lego_verifier.py
  │       = ObjectiveVerifier (FAIL_TO_PASS / PASS_TO_PASS empty-vs-gold)
  │       + AgenticVerifier generative critic
  │       + semi-resolved partial-credit shaping
  ├── DAG credit backup + advantage — IN-SCOPE, load-bearing (Phase 3)
  │   └── customized_areal/tree_search/dag/backup.py + credit.py (extended)
  │       TreeAdvantageComputer extended to consume per-node credit
  └── RL session wiring — UNCHANGED (db_bridge rl_start/set_reward/end_session)

Multica side (Go):
  ├── POST /api/v1/swe-lego/issues  (NEW, atomic orchestration)
  │   └── internal/handler/swe_lego_issue.go
  │       composes: CreateProject + image build/truncate + boot base sandbox
  │                 + fork × group_size + enqueue agent tasks
  ├── SWE-Lego image builder + history truncation (NEW)
  │   └── internal/service/swe_lego_image.go
  │       checkout base_commit → git filter-repo --commit-cutoff → docker build
  │       runs ON A FLEET NODE via /api/v1/nodes exec
  ├── Build-node lifecycle (NEW, thin)
  │   └── internal/service/swe_lego_build_node.go
  │       picks a swe-lego-build tagged node, ships build script, returns image ref
  └── Existing endpoints, reused as-is:
      ├── POST /api/projects
      ├── POST /api/issues  (+ acceptance_criteria, fail_to_pass, pass_to_pass metadata)
      ├── POST /api/issues/{id}/fork + DELETE  (Phase 1 ForkIssueSubtree)
      └── POST /api/v1/sandboxes/{id}/snapshot | /fork | DELETE  (cloud-runtime proxy,
          already present at server/internal/handler/cloud_runtime.go:108-127)

db_bridge:
  └── UNCHANGED — rl_start_session / rl_set_reward / rl_end_session / agent_start_branch
```

### Key invariants

1. **One multica project per issue.** All `group_size` agents and their delegation DAGs live in the
   same project, against the same forked-issue root.
2. **Docker ownership stays in multica.** AReal never calls docker directly; it calls the one
   `POST /api/v1/swe-lego/issues` endpoint plus the existing sandbox snapshot/fork endpoints.
3. **The `ForkableEnvironment` seam is the only thing areal's branching code sees.** Swapping
   `FleetSandboxProvider` → `MulticaSweLegoProvider` is the entire env-side change; `BranchMaterializer`
   is untouched.
4. **Anti-hacking is enforced at image-build time.** Git history after `issue_date` is deleted once,
   when the SWE-Lego image is built on the Fleet build-node — not at fork time. Every forked sandbox
   inherits the truncated history.

## 4. Multica side

### 4.1 The atomic endpoint

```
POST /api/v1/swe-lego/issues
Authorization: Bearer <MULTICA_API_KEY>   (areal-side service key)
X-Workspace-ID: <ws>
Content-Type: application/json

{
  "repo_url":            "https://github.com/psf/requests.git",
  "base_commit":         "abc123…",               // parent of the fixing PR's merge
  "issue_date":          "2025-03-14T09:30:00Z",   // history after this is deleted
  "issue_text":          "When retrying HTTPS…",   // becomes the issue body
  "issue_title":         "Retry leaks…",
  "acceptance_criteria": "The RetryAdapter must…",
  "fail_to_pass":        ["tests/test_retry.py::test_leak"],
  "pass_to_pass":        ["tests/test_retry.py::test_basic"],
  "group_size":          4,
  "agent_config_id":     "<uuid>",                 // which multica agent to run
  "base_image":          "swe-lego/python:3.11"   // optional; default from config
}
```

**Response (201):**
```json
{
  "project_id":              "<uuid>",
  "issue_id":                "<uuid>",
  "image_id":                "<sha256>",
  "build_node_id":           "<uuid>",             // Fleet node that built the image
  "base_sandbox_id":         "<uuid>",             // the one daemon-in-docker sandbox
  "base_sandbox_runtime_id": "<uuid>",              // daemon's runtime row
  "agent_run_ids":           ["<uuid>", … x group_size]
}
```

**Atomicity contract.** The endpoint either returns 201 with all `group_size` agent runs enqueued,
or it rolls back: deletes the base sandbox, deletes the project, and leaves the built image in the
build-node cache for reuse (image build is the expensive step; rebuilding across retries is wasteful).
The areal side treats any non-201 as "this issue did not start" and does not proceed to
branching/verifier.

**Auth.** Reuses the existing workspace + bearer-token model that `cloud_runtime.go` and `issue.go`
already enforce. The areal caller carries a service-key PAT scoped to the training workspace.

### 4.2 Endpoint sequence

```
1. CreateProject(workspace, name="swe-lego/<repo>-<base_commit[:8]>")
2. CreateIssue(project, title, body=issue_text, acceptance_criteria)
   → store fail_to_pass / pass_to_pass as issue metadata
3. swe_lego_image.BuildOrReuse(repo_url, base_commit, issue_date, base_image)
   → image_id, build_node_id   (see 4.3; cached by (repo_url, base_commit, issue_date, base_image))
4. CloudRuntime: boot one container from image_id ON THE SAME build_node_id
   (image lives on that node's docker daemon; not in a registry in v1)
   → base_sandbox_id, base_sandbox_runtime_id
   (the daemon self-registers its runtime on boot, as it does today)
5. for i in range(group_size):
     forked = POST /api/v1/sandboxes/fork { source_sandbox_id: base_sandbox_id }
     enqueue agent task on issue_id, bound to forked sandbox's runtime
     → agent_run_ids[i]
6. return 201 with all IDs
```

Step 5 forks the **base sandbox** (not a snapshot of a running agent) so all `group_size` agents start
from identical, pristine, history-truncated state — matching SWE-bench/Skywork-SWE rollout-group
semantics. The base sandbox stays alive as the fork source for the lifetime of the issue's training
episode; it's deleted at episode end.

### 4.3 SWE-Lego image builder + history truncation

New file `internal/service/swe_lego_image.go`. The build is **idempotent and cached** by
`(repo_url, base_commit, issue_date, base_image)` — repeated issues against the same triple reuse
the image. The build runs **on a Fleet build-node**, not on the multica server host.

```
BuildOrReuse(repo_url, base_commit, issue_date, base_image):
  cache_key = sha256(repo_url + base_commit + issue_date + base_image)

  # 1. Pick a build-node and check its local docker image cache.
  node = pick_build_node()    # /api/v1/nodes selection, swe-lego-build tagged
  if exec(node, ["docker", "image", "inspect", "swe-lego:"+cache_key]).ok:
      return image_ref("swe-lego:"+cache_key, node)

  # 2. Ship the build script to the node and run it there via /api/v1/nodes/exec.
  #    Inside the node:
  #      a. clone shallow-extended to base_commit
  #         git clone --filter=blob:none <repo_url> /tmp/build
  #         cd /tmp/build && git fetch origin <base_commit> && git checkout <base_commit>
  #      b. SWE-Lego anti-hacking: delete history after issue_date
  #         cutoff_commit = git rev-list -1 --before=<issue_date> HEAD
  #         git filter-repo --replace-ref refs/heads/main:<cutoff_commit> \
  #                         --commit-cutoff <cutoff_commit>
  #      c. docker build -t swe-lego:<cache_key> -f Dockerfile.swe-lego .
  exec(node, build_script(repo_url, base_commit, issue_date, base_image, cache_key))
  return image_ref("swe-lego:"+cache_key, node)
```

**Why `git filter-repo --commit-cutoff` (not `git checkout` alone).** A plain checkout of
`base_commit` leaves the full reflog and all future commits reachable from `.git/refs/*` and
`ORIG_HEAD`/`FETCH_HEAD`. SWE-Lego's anti-hacking rule requires the *history itself* to be gone,
so an agent that runs `git log` or `git blame` cannot see the future fix.
`git filter-repo --commit-cutoff` is the supported way to physically delete commits and rewrite
refs. `filter-repo` is preferred over the legacy `git filter-branch`.

**Image locality.** The built image lives **on the build node's docker daemon**, not in a registry.
Step 4 of §4.2 must boot the base sandbox **on the same node** that built the image. The endpoint
records `build_node_id` in the response so step 4 targets it. A registry-backed pull is a future
concern (deferred for v1).

**Dockerfile skeleton** (`multica/server/internal/service/swe_lego_image.Dockerfile.tmpl`):

```dockerfile
FROM <base_image>                      # e.g. swe-lego/python:3.11
COPY repo/ /workspace/repo             # the truncated checkout
WORKDIR /workspace/repo
RUN pip install -e . 2>/dev/null || true   # best-effort install (Skywork pattern)
COPY multica-daemon /usr/local/bin/multica-daemon
ENV MULTICA_DAEMON_AUTO_REGISTER=1
CMD ["multica-daemon", "run"]           # daemon self-registers, then claims tasks
```

The daemon binary is built from the existing multica daemon source and copied in at image-build
time — it's the same binary that runs locally today, just inside the container.

### 4.4 Build-node lifecycle

`pick_build_node()` reuses the existing Fleet `/api/v1/nodes` path — `cloud_runtime.go` already
proxies `CreateCloudRuntimeNode` / `ExecCloudRuntimeNode`. A node is picked by the
`swe-lego-build` tag. For v1, one tagged build-node is assumed; multi-node scheduling is a future
concern.

**Failure handling.** If the exec build fails (clone fails, filter-repo fails, docker build fails),
the endpoint returns 502 with a structured error and does **not** create the project/issue — the
build is the first expensive step, so failing fast before project creation keeps the rollback
surface small. The build-node's docker image cache retains any partial artifacts for diagnosis;
they are not promoted to the cache key on failure.

### 4.5 Existing endpoints reused as-is

- `POST /api/projects` (`server/internal/handler/project.go:221`) — create the per-issue project.
- `POST /api/issues` (`server/internal/handler/issue.go:2070`) — create the root issue, with
  `acceptance_criteria` and `fail_to_pass`/`pass_to_pass` stored as issue metadata.
- `POST /api/issues/{id}/fork` + `DELETE /api/issues/{id}/fork` (Phase 1 `ForkIssueSubtree`) — the
  issue-subtree fork at `(task_id, seq)`, used by mid-run branching.
- `POST /api/v1/sandboxes/{id}/snapshot` + `POST /api/v1/sandboxes/fork` + `DELETE`
  (`server/internal/handler/cloud_runtime.go:108-127`) — the sandbox snapshot/fork/cleanup the
  areal-side `MulticaSweLegoProvider` calls.
- `ClaimTaskByRuntime` / `ReportTaskMessages` (`server/internal/handler/daemon.go`) — the daemon's
  claim-and-report loop, unchanged; the daemon inside the SWE-Lego container uses these exactly as
  a local daemon does today.

## 5. AReal side

### 5.1 Per-issue orchestration loop

New file `customized_areal/tree_search/agents/swe_lego_issue_runner.py`. Top-level entry called by
the training episode loop, once per issue:

```python
async def run_swe_lego_issue(
    *,
    issue: SweLegoIssue,          # repo_url, base_commit, issue_date, tests, …
    group_size: int,
    agent_config_id: str,
    multica: MulticaSweLegoClient,
    env: ForkableEnvironment,     # = MulticaSweLegoProvider(multica)
    branch_materializer: BranchMaterializer,
    verifier: SweLegoVerifier,
    rl_session: RlSessionWriter,  # db_bridge client
) -> SweLegoIssueResult:
    # 1. Atomically set up the issue env + start group_size agents.
    setup = await multica.create_swe_lego_issue(
        repo_url=issue.repo_url, base_commit=issue.base_commit,
        issue_date=issue.issue_date, issue_text=issue.issue_text,
        acceptance_criteria=issue.acceptance_criteria,
        fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
        group_size=group_size, agent_config_id=agent_config_id,
    )
    # setup = { project_id, issue_id, base_sandbox_id, agent_run_ids[], … }

    # 2. Open one RL session per agent_run (group_size sessions).
    sessions = [
        await rl_session.start(agent_run_id=rid, issue_id=setup.issue_id)
        for rid in setup.agent_run_ids
    ]

    # 3. Let the episode loop drive branching within each lane.
    #    select_branch_candidate / BranchMaterializer.materialize operate per-lane,
    #    using `env` (= MulticaSweLegoProvider) for snapshot+fork and the
    #    existing ForkIssueSubtree for the issue subtree. The materializer is
    #    UNCHANGED — only the injected provider class differs.
    ...

    # 4. When each agent_run finishes, run the hybrid verifier against its
    #    sandbox + transcript, write reward via rl_set_reward.
    results = await asyncio.gather(*[
        verifier.verify_and_reward(
            agent_run_id=rid, sandbox_id=sbx, session_id=sid,
            fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
            transcript=..., acceptance_criteria=issue.acceptance_criteria,
        )
        for rid, sbx, sid in zip(setup.agent_run_ids, forked_sandbox_ids, sessions)
    ])

    # 5. Cleanup: delete forked sandboxes + the base sandbox + the project.
    await multica.cleanup_swe_lego_issue(setup.project_id, setup.base_sandbox_id, ...)
    return SweLegoIssueResult(per_agent=results, ...)
```

**Branching within a lane.** Step 3 is where the existing DAG-RL branching machinery engages.
`select_branch_candidate` picks a `(task_id, seq)` within a lane; `BranchMaterializer.materialize`
(unchanged) does snapshot → `ForkIssueSubtree` → `agent_start_branch`. The only difference from the
approved design is that `env` is `MulticaSweLegoProvider`, not `FleetSandboxProvider`. The verifier
runs once per *terminal* agent run (the leaf of each lane's branch tree), and rewards back up
through the DAG via the existing credit/backup path.

### 5.2 `MulticaSweLegoProvider` — the retargeted `ForkableEnvironment`

New class in `customized_areal/tree_search/agents/environment.py`, sibling to
`FleetSandboxProvider`. Same Protocol (`snapshot`/`fork`/`restore`/`cleanup`), different backend:

```python
class MulticaSweLegoProvider:
    """ForkableEnvironment backed by multica's cloud-runtime proxy.

    Calls the EXISTING endpoints that cloud_runtime.go already exposes:
      POST /api/v1/sandboxes/{id}/snapshot  → SnapshotResult
      POST /api/v1/sandboxes/fork           → ForkResult
      POST /api/v1/sandboxes/{id}/restore   → None   (no-op if Fleet lacks it)
      DELETE /api/v1/sandboxes/{id}         → None   (idempotent on 404)

    Identical surface to FleetSandboxProvider so BranchMaterializer is unchanged.
    base_url / api_key default to MULTICA_BASE_URL / MULTICA_API_KEY.
    The concurrency semaphore (§3.3 of the approved design) is preserved.
    """

    async def snapshot(self, sandbox_id: str) -> SnapshotResult: ...
    async def fork(self, *, source_sandbox_id=None, snapshot_id=None) -> ForkResult: ...
    async def restore(self, sandbox_id: str) -> None: ...
    async def cleanup(self, sandbox_id: str) -> None: ...
```

**Why a new class instead of reusing `FleetSandboxProvider`.** The endpoints are the *same shape*
but the *base URL and auth* differ (multica proxy vs. Fleet direct). Keeping two classes makes the
boundary explicit and lets each carry its own config/env without runtime branching. The Protocol
(`ForkableEnvironment`) is the only thing `BranchMaterializer` sees, so injecting either is a
one-line change at the call site.

### 5.3 Hybrid SWE-Lego verifier

New file `customized_areal/tree_search/agents/reward/swe_lego_verifier.py`. Composes three layers,
matching the SWE-Lego/R2E-Gym hybrid recipe from the research doc:

```python
class SweLegoVerifier:
    """Hybrid verifier: objective tests + generative critic + semi-resolved.

    Layers (in order, each can short-circuit):
      1. ObjectiveVerifier  — run FAIL_TO_PASS + PASS_TO_PASS tests in the
         sandbox (Skywork-SWE empty-vs-gold two-run model). Binary pass/fail.
         Exec via the MulticaSweLegoProvider's exec path or the cloud-runtime
         /api/v1/nodes/exec proxy.
      2. AgenticVerifier    — generative critic over the agent transcript +
         acceptance_criteria. yes/no token probability → score in [0,1].
         Used when tests are inconclusive or to weight the objective result.
      3. semi-resolved      — partial credit when the agent located the bug
         (e.g. failing test count reduced) but did not fully fix. Dense signal
         that shapes intermediate steps via per_step_signals.
    """

    async def verify_and_reward(self, *, agent_run_id, sandbox_id, session_id,
                                 fail_to_pass, pass_to_pass, transcript,
                                 acceptance_criteria) -> VerifierResult:
        obj = await self._objective.run_tests(sandbox_id, fail_to_pass, pass_to_pass)
        gen = await self._critic.score(transcript, acceptance_criteria, obj)
        partial = self._semi_resolved_credit(obj)
        reward = self._blend(obj, gen, partial)
        await self._rl.set_reward(session_id=session_id, reward=reward)
        return VerifierResult(
            success=obj.fully_passes, reward=reward, source="hybrid",
            rationale=..., per_step_signals=partial.signals,
        )
```

**Anti-cheating discipline.** The verifier reads only the test results and the agent transcript —
never the gold patch, never git history (which is already truncated at image-build time). The
objective check executes tests in a way that does not leak the `test_patch` to the agent's
environment.

**Reward blending.** The objective layer short-circuits when it is decisive:
`FAIL_TO_PASS` fully passes → reward `1.0` (regardless of critic/semi-resolved); `FAIL_TO_PASS`
fully fails AND no failing-test-count reduction → reward `0.0`. In the mixed middle (some
`FAIL_TO_PASS` pass, or failing-test count reduced but not to zero), the blend is
`0.7 * objective + 0.2 * generative + 0.1 * semi_resolved`. Weights are configurable; the
short-circuit thresholds are not (they are the anti-cheating anchor).

### 5.4 DAG credit backup — load-bearing in v1

Decision 9 commits v1 to actually training on DAG-distributed credit, not just logging it. This
means:

1. **The Phase 3 work from `multica-dag-rl-design.md` is in-scope.** DAG-aware hybrid reward backup
   + advantage computer (`customized_areal/tree_search/dag/backup.py` + `credit.py`), per-node
   credit at fan-in joins. Task 10 of the approved design.
2. **`TreeAdvantageComputer` is extended, not replaced.** It keeps its GRPO-normalize-one-reward-
   per-episode behavior but is extended to consume the per-node credit from `credit.py` instead of
   broadcasting a flat episode reward to all turns.
3. **The verifier's terminal reward is distributed along DAG edges** (delegation / mention /
   completion) so sub-agent runs that contributed to a successful root get credit, and ones that
   didn't get less. Fan-in joins: the verifier assigns per-agent credit explicitly (decision 8 of
   the approved design) — no fixed sum/mean/max aggregation.
4. **The generative critic's `per_step_signals` shape intermediate steps**, distinct from the
   terminal backup. This is the dense signal that turns a sparse terminal reward into per-step
   credit.
5. **Training export carries per-node credit.** `/export_trajectories` returns tensor data whose
   per-token advantage / lossmask reflects DAG-distributed credit, not a flat episode reward. This
   is where "shapes training" becomes concrete.

### 5.5 What stays unchanged

- `BranchMaterializer` (`integration.py`) — snapshot + `ForkIssueSubtree` + `agent_start_branch`
  orchestration, with rollback. Untouched.
- `MulticaIssueForker` (`integration.py`) — the `POST /api/issues/{id}/fork` client. Untouched (the
  Phase 1 endpoint already exists).
- `db_bridge` RL session channels (`rl_start_session` / `rl_set_reward` / `rl_end_session` /
  `agent_start_branch`). Untouched.
- DAG model, `execution_dag.py`, GAE, advantage, harvest — the Phase 0-2 machinery. Untouched.
- The `Verifier` Protocol and `ObjectiveVerifier` base (`verifier.py`). Untouched; `SweLegoVerifier`
  composes them.

## 6. Data flow — end to end

```
Training episode loop (per issue)
  │
  ├─ run_swe_lego_issue(issue, group_size=N)
  │   ├─ multica.create_swe_lego_issue(...)
  │   │     ├─ build SWE-Lego image on Fleet build-node (history truncated)
  │   │     ├─ boot base sandbox (daemon-in-docker) on same node
  │   │     ├─ fork × N  →  N daemon-in-docker sandboxes
  │   │     └─ enqueue N agent tasks  →  N agent runs (remote mode)
  │   ├─ rl_session.start × N  →  N RL sessions
  │   │
  │   ├─ [per lane, in parallel]
  │   │   ├─ daemon claims task, runs agent, POSTs task_message batches (seq source)
  │   │   ├─ select_branch_candidate(task_id, seq)
  │   │   ├─ BranchMaterializer.materialize
  │   │   │     ├─ MulticaSweLegoProvider.snapshot(live sandbox)
  │   │   │     ├─ ForkIssueSubtree(issue, task_id, seq)
  │   │   │     └─ agent_start_branch(forked sandbox + forked issue, replay msgs≤seq)
  │   │   ├─ ... branching tree grows ...
  │   │   └─ terminal run finishes
  │   │
  │   ├─ [per lane terminal] SweLegoVerifier.verify_and_reward
  │   │     ├─ ObjectiveVerifier: empty-vs-gold test run  →  binary pass/fail
  │   │     ├─ AgenticVerifier: generative critic          →  score in [0,1]
  │   │     ├─ semi-resolved partial credit                →  dense signal
  │   │     ├─ blend → terminal reward
  │   │     └─ rl_set_reward(session_id, reward)
  │   │
  │   ├─ DAG credit backup (Phase 3, load-bearing)
  │   │   ├─ distribute terminal reward along DAG edges
  │   │   ├─ per-node credit at fan-in joins (explicit, no sum/mean/max)
  │   │   ├─ per_step_signals shape intermediate steps
  │   │   └─ TreeAdvantageComputer consumes per-node credit  →  per-token advantage
  │   │
  │   ├─ /export_trajectories  →  tensor data with DAG-distributed credit
  │   └─ cleanup: forked sandboxes + base sandbox + project
  │
  └─ PPO/GRPO step on the exported, credit-shaped trajectories
```

## 7. Error handling

- **Image build failure** (clone/filter-repo/docker build) → endpoint returns 502, no project
  created. Build-node cache retains partial artifacts for diagnosis; nothing promoted to the cache
  key.
- **Base sandbox boot failure** → endpoint rolls back the project (and the image stays cached for
  retry). Returns 503.
- **Fork failure** (one of `group_size`) → endpoint rolls back the successfully-forked sandboxes
  and the base sandbox, deletes the project. Returns 503. The image stays cached.
- **Agent run enqueue failure** → same rollback as fork failure.
- **Mid-run branching failure** → `BranchMaterializer` rolls back per its existing contract
  (`integration.py`): cleans up the forked sandbox and the forked issue, logs, and the lane
  continues without that branch. Other lanes are unaffected.
- **Verifier failure** (test execution crash, critic LLM error) → `SweLegoVerifier` falls back to a
  neutral reward (0.0) with `source="default"` and a rationale, never crashing the episode. This
  matches the existing `ObjectiveVerifier` exception-to-failure contract.
- **Reward write failure** (`rl_set_reward`) → logged; the trajectory is still exported with the
  verifier-computed reward attached, so a downstream re-run of the PPO step can recover.

## 8. Security and anti-cheating

- **Git history truncation** is enforced once at image-build time via `git filter-repo
  --commit-cutoff`. Every forked sandbox inherits the truncated history. An agent running `git log`
  or `git blame` cannot reach the future fix.
- **Verifier read discipline.** The verifier reads only test results and the agent transcript —
  never the gold patch, never `test_patch` source beyond the test names needed to run them.
- **Test execution isolation.** Tests run in the agent's sandbox but the `test_patch` (the gold
  tests that define `FAIL_TO_PASS`) is applied by the verifier in a way that does not leave it
  readable in the agent's working tree during the agent's run. (The empty-vs-gold two-run model
  applies `test_patch` only for the gold run, after the agent has finished.)
- **API key scope.** The areal-side service key is scoped to the training workspace; it cannot
  touch other workspaces' projects or sandboxes. Reuses the existing multica workspace authz.
- **No secrets in images.** The SWE-Lego image contains only the repo, the daemon binary, and the
  test runner. No API keys are baked in; the daemon receives its `MULTICA_TOKEN` at runtime via the
  existing daemon-bootstrap path.

## 9. Testing strategy

- **`MulticaSweLegoProvider` contract tests** — fake httpx transport exercises
  snapshot/fork/restore/cleanup ordering, error paths, and the concurrency semaphore, without
  touching a real multica or Fleet. Mirrors the existing `FleetSandboxProvider` contract tests.
- **`swe_lego_image.BuildOrReuse` Go tests** — fake build-node exec asserts: cache hit short-
  circuits; cache miss runs clone → filter-repo → docker build in order; `--commit-cutoff` is
  computed from `issue_date`; a second call with the same triple reuses the image. History-
  truncation is asserted by inspecting the script shipped to the node (no real git operations in
  unit tests).
- **`swe_lego_issue` handler Go tests** — mock the service layer: happy path returns 201 with all
  IDs; image-build failure returns 502 with no project; fork failure rolls back; auth/workspace
  checks enforce. Follows the `parseUUIDOrBadRequest` / loader convention from `multica/CLAUDE.md`.
- **`SweLegoVerifier` Python tests** — mock `ObjectiveVerifier` and `AgenticVerifier`: objective
  fully passes → reward 1.0, short-circuit; objective fully fails → reward 0.0; mixed → blend;
  critic raises → neutral fallback. Semi-resolved credit asserted on a fixture where failing-test
  count drops but not to zero.
- **DAG credit backup tests** — extend the existing `test_backup.py` to assert that a multi-node
  DAG distributes the terminal reward along edges and that `TreeAdvantageComputer` produces
  per-token advantage reflecting per-node credit (not a flat broadcast). Fan-in join credit is
  explicit per the approved design.
- **End-to-end** — one issue at `group_size=2` against a real multica + Fleet build-node + cloud
  sandbox stack. Asserts: image built with truncated history; two daemon-in-docker sandboxes
  boot; agents run; verifier produces a reward; `/export_trajectories` returns DAG-credit-shaped
  tensors. Integration tests requiring multi-node hardware are skipped with an explanation when
  unavailable (per `backend/areal/CLAUDE.md`).
- **Anti-cheating test** — a canary agent run that attempts `git log --after=<issue_date>` in the
  sandbox asserts the history is unreachable. This is a regression guard on the truncation.

## 10. Out of scope (v1)

- **Registry-backed image distribution.** The built image lives on the build node's docker daemon.
  A registry pull so sandboxes can boot on any node is deferred.
- **Multi-node build scheduling.** One `swe-lego-build` tagged node is assumed.
- **Non-Python repos.** SWE-Lego docker images are Python-first in v1 (matching the research doc
  baseline). Multi-language support is a future concern.
- **Streaming critic.** The generative critic returns one score per terminal run; streaming
  per-step critic scores during the run is deferred.
- **Daemonless (thin docker) path.** Out of scope — decision 4 commits to daemon-in-docker.

## 11. References

### AReaL
- Approved DAG-RL design: `docs/superpowers/specs/2026-06-26-multica-dag-rl-design.md`
- DAG event codec design: `docs/superpowers/specs/2026-06-29-dag-event-codec-design.md`
- Event branch selection design: `docs/superpowers/specs/2026-06-29-event-branch-selection-design.md`
- OpenRouter remote rollout proxy design:
  `docs/superpowers/specs/2026-06-29-openrouter-remote-rollout-proxy-design.md`
- DAG RL package: `customized_areal/tree_search/agents/` (`environment.py`, `integration.py`,
  `verifier.py`, `branch_selection.py`, `execution_dag.py`, `agentic_verifier.py`)
- SWE data pipeline research: `customized_areal/tree_search/agents/reward/swe_data_pipeline_comparison.md`

### Multica
- Cloud-runtime proxy: `server/internal/handler/cloud_runtime.go:108-127`
- Project / issue handlers: `server/internal/handler/project.go:221`,
  `server/internal/handler/issue.go:2070`
- Daemon claim/report loop: `server/internal/handler/daemon.go` (`ClaimTaskByRuntime`,
  `ReportTaskMessages`)
- Routes: `server/cmd/server/router.go:722-810` (issues, projects), `:969-970` (sandbox
  snapshot/fork)
- Conventions: `multica/CLAUDE.md`

### Project rules
- Backend boundary: `.claude/rules/backend.md` ("Boundary Normalization Rule")
- Sandbox-layer types: `.claude/rules/code-quality.md` ("Sandbox-layer types stay behind the
  wrapper")
- Multi-tenancy: `.claude/rules/multi-tenancy.md`
