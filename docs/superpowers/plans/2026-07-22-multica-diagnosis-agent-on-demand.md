# Multica Diagnosis Agent On-Demand Implementation Plan

> **For Codex:** Execute this plan task by task. Use the `executing-plans` skill for inline execution, or `subagent-driven-development` if the user explicitly selects subagents. Apply TDD: write the failing test, verify the failure, implement the minimum code, then rerun the focused test. Load `commit-conventions` before every commit.

**Goal:** Replace the one-shot diagnosis prompt with one persistent, tool-less Pi session that receives only the DAG topology up front, fetches segment messages page by page through a capability-scoped Multica API, writes step rewards directly into the authoritative DAG store, and compacts its diagnosis memory after every completed segment.

**Architecture:** Multica owns topology, messages, progress, rewards, and recovery state. A loopback-only HTTP server exposes five diagnosis capabilities to one run. A trusted Pi extension translates model tool requests into authenticated calls to that server while Pi remains launched with `--no-tools`. The runner drives exactly one Pi turn per segment, validates server-side completion, forces compaction after every segment, and resumes from persisted checkpoints after interruption.

**Tech stack:** Go, SQLite/sqlc generated query companions, Pi RPC mode, TypeScript Pi extension, `net/http`, Go unit/integration tests, Markdown/Mermaid documentation.

**Design source:** `docs/superpowers/specs/2026-07-22-multica-diagnosis-agent-on-demand-design.md`

## Invariants to preserve

- Pi starts with the basic DAG topology and task context, not all segment messages.
- One Pi RPC process and one session are reused for the entire diagnosis run.
- There is exactly one scored segment per Pi turn.
- Segment messages are fetched with a server-issued opaque cursor; coverage is validated by Multica, not trusted from model text.
- Rewards are persisted incrementally and idempotently before a segment is marked complete.
- Normal compaction runs after every segment. At 80% context usage, Pi must compact at the next safe page boundary before fetching more messages.
- The compressed memory preserves evidence, scoring calibration, unresolved hypotheses, cross-segment dependencies, and useful future cues; it drops raw repetition and details with no expected value for later scoring.
- Pi keeps `--no-tools`; only a Multica-generated trusted extension is loaded explicitly.
- The capability token never appears in prompts, command-line arguments, logs, or persisted diagnosis memory.
- Existing terminal-task behavior remains soft-fail: diagnosis failure is recorded and logged but does not prevent the task from closing.

## Planned file map

### Create

- `multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.up.sql`
- `multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.down.sql`
- `multica/server/pkg/db/generated/interaction_dag_diagnosis.sql.go`
- `multica/server/internal/service/diagnosis_state.go`
- `multica/server/internal/service/diagnosis_state_test.go`
- `multica/server/internal/service/diagnosis_tool_server.go`
- `multica/server/internal/service/diagnosis_tool_server_test.go`
- `multica/server/internal/service/diagnosis_pi_extension.go`
- `multica/server/internal/service/diagnosis_pi_extension_test.go`
- `multica/server/internal/service/diagnosis_agent_integration_test.go`

### Modify

- `multica/server/pkg/db/queries/interaction_dag.sql`
- `multica/server/pkg/db/generated/interaction_dag.sql.go`
- `multica/server/pkg/db/queries/task_message.sql`
- `multica/server/pkg/db/generated/task_message.sql.go`
- `multica/server/pkg/agent/agent.go`
- `multica/server/pkg/agent/pi.go`
- `multica/server/pkg/agent/pi_test.go`
- `multica/server/pkg/agent/pi_rpc.go`
- `multica/server/pkg/agent/pi_rpc_test.go`
- `multica/server/internal/service/interaction_dag.go`
- `multica/server/internal/service/interaction_dag_test.go`
- `multica/server/internal/service/diagnosis_tools.go`
- `multica/server/internal/service/diagnosis_tools_test.go`
- `multica/server/internal/service/diagnosis_agent.go`
- `multica/server/internal/service/diagnosis_agent_test.go`
- `multica/server/internal/service/training.go`
- `multica/server/internal/service/training_test.go`
- `multica/server/internal/service/training_config.go`
- `multica/server/internal/service/training_config_test.go`
- `customized_areal/tree_search/agents/diagnose_agent_flow.md`

## Task 1: Persist resumable diagnosis run and segment state

**Files:**

- Create: `multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.up.sql`
- Create: `multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.down.sql`
- Modify: `multica/server/pkg/db/queries/interaction_dag.sql`
- Create: `multica/server/pkg/db/generated/interaction_dag_diagnosis.sql.go`
- Create: `multica/server/internal/service/diagnosis_state.go`
- Create: `multica/server/internal/service/diagnosis_state_test.go`

### Step 1: Write failing state-store tests

Add table-driven tests covering:

- creating a run snapshots `project_id`, `task_id`, ordered segment IDs, and status `running`;
- `StartSegment` is idempotent and records the expected message count;
- `RecordSegmentPage` accepts only the current cursor and monotonically increases fetched counts;
- `CompleteSegment` fails until message coverage and reward coverage are both complete;
- completing the same segment twice is idempotent;
- `LoadResumableRun` returns the first incomplete segment and its cursor;
- `CompleteRun` fails while any segment is incomplete;
- a stale run can be marked `failed` with a bounded error string.

Use the repository's existing temporary SQLite/migration test helper. Keep assertions at the service boundary so the schema can evolve without leaking SQL details into later tasks.

### Step 2: Run the focused test and confirm the expected failure

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisState' -count=1
```

Expected: compile failure because `DiagnosisStateStore`, `DiagnosisRunCheckpoint`, and `SegmentDiagnosisCheckpoint` do not exist.

### Step 3: Add migration 208

Create two tables:

- `interaction_dag_diagnosis_run`: run ID, project/task scope, topology hash, ordered segment snapshot JSON, status, current segment ordinal, Pi session ID when available, bounded error, timestamps.
- `interaction_dag_diagnosis_segment`: run ID plus segment ID primary key, ordinal, expected/fetched message counts, next opaque cursor, reward count, status, timestamps.

Add constraints for valid statuses and non-negative counters. Add an index for resumable runs by `(project_id, task_id, status, updated_at)`. The down migration drops only these two tables and their indexes.

### Step 4: Add explicit query source and generated companion

Add named SQL queries for:

- create/load/update/complete/fail run;
- create/load/update/complete segment checkpoint;
- list segment checkpoints in ordinal order.

Update `pkg/db/queries/interaction_dag.sql` as the canonical source. Because the repository currently carries manual generated-query drift, add the matching typed methods in `pkg/db/generated/interaction_dag_diagnosis.sql.go` and document that it mirrors the named queries. Do not run an unrelated full sqlc rewrite.

### Step 5: Implement the state-store boundary

In `diagnosis_state.go`, define:

```go
type DiagnosisRunCheckpoint struct {
    RunID                 string
    ProjectID             string
    TaskID                string
    TopologyHash          string
    OrderedSegmentIDs     []string
    CurrentSegmentOrdinal int
    Status                DiagnosisRunStatus
}

type SegmentDiagnosisCheckpoint struct {
    RunID               string
    SegmentID           string
    Ordinal             int
    ExpectedMessageCount int
    FetchedMessageCount int
    NextCursor          string
    RewardCount         int
    Status              SegmentDiagnosisStatus
}
```

Implement transactional transitions with compare-and-set predicates. Return typed errors for stale cursor, incomplete message coverage, incomplete reward coverage, topology mismatch, and invalid state transition. Bound persisted errors before writing them.

### Step 6: Run tests and inspect migration reversibility

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisState' -count=1
go test ./pkg/db/... -count=1
```

Expected: all focused tests pass and both migration directions parse under the repository migration harness.

### Step 7: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.up.sql multica/server/pkg/db/migrations/208_interaction_dag_diagnosis_state.down.sql multica/server/pkg/db/queries/interaction_dag.sql multica/server/pkg/db/generated/interaction_dag_diagnosis.sql.go multica/server/internal/service/diagnosis_state.go multica/server/internal/service/diagnosis_state_test.go
git commit -m "feat(multica): persist diagnosis progress"
```

## Task 2: Add cursor-paginated segment messages and coverage validation

**Files:**

- Modify: `multica/server/pkg/db/queries/task_message.sql`
- Modify: `multica/server/pkg/db/generated/task_message.sql.go`
- Modify: `multica/server/internal/service/interaction_dag.go`
- Modify: `multica/server/internal/service/diagnosis_tools.go`
- Modify: `multica/server/internal/service/diagnosis_tools_test.go`

### Step 1: Write failing paging tests

Add tests for `GetSegmentMessagePage` that verify:

- deterministic ordering by message timestamp and stable message ID tie-breaker;
- the first request uses an empty cursor;
- the response returns `Messages`, `NextCursor`, `FetchedCount`, `ExpectedCount`, and `Complete`;
- a byte limit never splits a message and still returns one oversized message as a single page;
- a turn limit stops at the configured boundary;
- replaying the same cursor returns the same page;
- a cursor from another run or segment is rejected;
- a mutated, expired, or malformed cursor is rejected;
- system/context messages outside the segment range cannot leak into the page.

### Step 2: Run the focused tests and confirm failure

```bash
cd multica/server
go test ./internal/service -run 'TestGetSegmentMessagePage|TestDiagnosisCursor' -count=1
```

Expected: compile failure because the paged API and cursor codec do not exist.

### Step 3: Add the database page query

Add a keyset-pagination query to `task_message.sql`, mirrored in the generated companion. It must accept task ID, inclusive segment time bounds, last timestamp, last message ID, and limit. Add a count query using the identical range predicates so `ExpectedCount` cannot disagree with page membership.

Do not use offset pagination: messages may be numerous and the checkpoint must resume without rescanning earlier rows.

### Step 4: Implement opaque scoped cursors

Define an internal cursor payload containing run ID, segment ID, last timestamp, last message ID, and page sequence. Sign it with an HMAC key created for the diagnosis tool-server lifetime. Base64url-encode the payload plus signature. Keep codec methods private to the service package.

### Step 5: Implement `SegmentMessagePage`

Add:

```go
type SegmentMessagePage struct {
    Messages      []DiagnosisMessage
    NextCursor    string
    FetchedCount  int
    ExpectedCount int
    Complete      bool
}
```

Use defaults of 20 turns and 24 KiB per page, both configurable later. After producing a page, atomically persist its `NextCursor` and cumulative fetched count through `DiagnosisStateStore`. Only return a page after the checkpoint update succeeds.

### Step 6: Run tests

```bash
cd multica/server
go test ./internal/service -run 'TestGetSegmentMessagePage|TestDiagnosisCursor' -count=1
go test ./pkg/db/... -count=1
```

Expected: paging, cursor isolation, and coverage tests pass.

### Step 7: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/pkg/db/queries/task_message.sql multica/server/pkg/db/generated/task_message.sql.go multica/server/internal/service/interaction_dag.go multica/server/internal/service/diagnosis_tools.go multica/server/internal/service/diagnosis_tools_test.go
git commit -m "feat(multica): page diagnosis segment messages"
```

## Task 3: Allow explicit trusted Pi extensions while tools stay disabled

**Files:**

- Modify: `multica/server/pkg/agent/agent.go`
- Modify: `multica/server/pkg/agent/pi.go`
- Modify: `multica/server/pkg/agent/pi_test.go`
- Modify: `multica/server/pkg/agent/pi_rpc.go`
- Modify: `multica/server/pkg/agent/pi_rpc_test.go`

### Step 1: Write failing argument-construction tests

Cover both one-shot Pi and Pi RPC argument builders:

- `DisableTools: true` still emits the existing no-tools restrictions;
- `TrustedExtensionPaths: []string{"/tmp/diagnosis.ts"}` emits exactly one explicit extension argument;
- ordinary configured/user extensions remain disabled;
- relative paths, missing files, directories, duplicate paths, and paths outside the caller-provided trusted directory are rejected;
- the trusted extension path is not accepted unless `DisableTools` is true;
- no secret environment value is copied into argv.

### Step 2: Run the focused tests and confirm failure

```bash
cd multica/server
go test ./pkg/agent -run 'TestBuildPiArgs.*TrustedExtension|TestBuildPiRPCArgs.*TrustedExtension' -count=1
```

Expected: compile failure because `ExecOptions.TrustedExtensionPaths` is absent.

### Step 3: Extend `ExecOptions`

Add:

```go
TrustedExtensionPaths []string
TrustedExtensionRoot  string
```

Document that these fields are for application-generated extensions only. Validate each path with `filepath.Abs`, `filepath.EvalSymlinks`, `os.Stat`, and `filepath.Rel` against the trusted root. Accept regular files only. Preserve the existing built-in Pi output-limit extension behavior.

### Step 4: Update both Pi argument builders

For the restricted profile, continue emitting `--no-extensions`, `--no-skills`, `--no-prompt-templates`, `--no-context-files`, `--no-approve`, and the Pi no-tools form. Then append only the validated explicit extension flags required by Pi. Apply the same restrictions in RPC mode so the two launch paths cannot diverge.

### Step 5: Run tests

```bash
cd multica/server
go test ./pkg/agent -run 'TestBuildPiArgs|TestBuildPiRPCArgs' -count=1
```

Expected: all argument and trust-boundary tests pass.

### Step 6: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/pkg/agent/agent.go multica/server/pkg/agent/pi.go multica/server/pkg/agent/pi_test.go multica/server/pkg/agent/pi_rpc.go multica/server/pkg/agent/pi_rpc_test.go
git commit -m "feat(agent): load scoped Pi extensions"
```

## Task 4: Expose explicit Pi RPC compaction and runtime controls

**Files:**

- Modify: `multica/server/pkg/agent/pi_rpc.go`
- Modify: `multica/server/pkg/agent/pi_rpc_test.go`

### Step 1: Extend the fake Pi RPC script and add failing tests

Make the test script recognize:

- `compact` with custom instructions;
- `set_auto_compaction` with `enabled: false`;
- `get_session_stats` and/or `get_state` outside an active prompt turn;
- interleaved prompt and control responses with distinct request IDs.

Test that:

- control calls do not consume or corrupt the current prompt response;
- compaction returns a typed result and propagates Pi errors;
- runtime stats are available between segment turns;
- closing the backend unblocks pending control calls;
- context cancellation removes the pending request without poisoning later calls.

### Step 2: Run the tests and confirm failure

```bash
cd multica/server
go test ./pkg/agent -run 'TestPiRPCBackend.*Compact|TestPiRPCBackend.*RuntimeStats|TestPiRPCBackend.*Control' -count=1
```

Expected: compile failure because the control methods and response router are missing.

### Step 3: Add typed control methods

Implement:

```go
type PiCompactionResult struct {
    Summary         string
    TokensBefore    int
    TokensAfter     int
}

func (b *PiRPCBackend) Compact(ctx context.Context, instructions string) (PiCompactionResult, error)
func (b *PiRPCBackend) SetAutoCompaction(ctx context.Context, enabled bool) error
func (b *PiRPCBackend) RuntimeStats(ctx context.Context, fallbackModel string) (*RuntimeTokenStats, error)
```

Refactor response routing from a single-current-turn assumption to a request-ID keyed pending map protected by a mutex. Keep streamed prompt events routed to the active execution while terminal control responses resolve their own waiter.

### Step 4: Disable Pi automatic compaction for diagnosis sessions

The diagnosis runner will call `SetAutoCompaction(false)` once after startup. This prevents an unstructured Pi auto-summary from racing with Multica's required after-every-segment compression. Other Pi RPC consumers retain their current default.

### Step 5: Run package tests and race detection

```bash
cd multica/server
go test ./pkg/agent -count=1
go test -race ./pkg/agent -run 'TestPiRPCBackend' -count=1
```

Expected: the full agent package and focused race suite pass.

### Step 6: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/pkg/agent/pi_rpc.go multica/server/pkg/agent/pi_rpc_test.go
git commit -m "feat(agent): control Pi RPC compaction"
```

## Task 5: Build the loopback capability-scoped diagnosis API

**Files:**

- Create: `multica/server/internal/service/diagnosis_tool_server.go`
- Create: `multica/server/internal/service/diagnosis_tool_server_test.go`
- Modify: `multica/server/internal/service/interaction_dag.go`
- Modify: `multica/server/internal/service/interaction_dag_test.go`

### Step 1: Write failing HTTP contract and security tests

Test all five endpoints:

- `POST /v1/get-segment-messages`
- `POST /v1/record-step-rewards`
- `GET /v1/diagnosis-progress`
- `POST /v1/finish-segment`
- `POST /v1/complete-diagnosis`

Cover valid calls plus:

- missing/incorrect bearer token returns 401 without reflecting the token;
- a valid token cannot access another run, project, task, or segment;
- the listener address is loopback and uses an ephemeral port;
- request bodies have a strict size cap and unknown JSON fields fail;
- rewards for steps outside the active segment fail;
- duplicate reward writes are idempotent;
- conflicting rewrites fail rather than silently replacing scores;
- `finish-segment` rejects incomplete message or reward coverage;
- `complete-diagnosis` rejects incomplete segments;
- access logs and returned errors contain no bearer token or raw prompt content.

### Step 2: Run the tests and confirm failure

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisToolServer|TestRecordStepRewardsIdempotent' -count=1
```

Expected: compile failure because `DiagnosisToolServer` does not exist.

### Step 3: Make step-reward persistence idempotent

Update `RecordStepRewards` so the unique identity is the existing DAG step identity. Within one transaction:

- insert missing rewards;
- accept an exact replay as success;
- reject a replay whose score or diagnosis payload conflicts;
- update the segment checkpoint reward count only after all writes succeed.

Keep the existing `/dag` assembly behavior unchanged and add regression coverage to `interaction_dag_test.go`.

### Step 4: Implement `DiagnosisToolServer`

Construct it with explicit dependencies: run checkpoint, topology snapshot, state store, message pager, DAG reward writer, HMAC cursor key, and a cryptographically random 32-byte bearer token. Bind with `net.Listen("tcp", "127.0.0.1:0")`. Expose the selected base URL and token to the caller separately.

Use `http.Server` with read-header, read, write, and idle timeouts. Apply constant-time token comparison, strict JSON decoding, structured error codes, request IDs, and redacted logging. The handler must derive project/task/run scope from server state rather than trusting request fields.

### Step 5: Implement the progress response

Return only data useful to the current run:

```go
type DiagnosisProgress struct {
    RunID                  string
    CurrentSegmentID       string
    CurrentSegmentOrdinal  int
    CompletedSegmentIDs    []string
    RemainingSegmentIDs    []string
    FetchedMessageCount    int
    ExpectedMessageCount   int
    RecordedRewardCount    int
    ExpectedRewardCount    int
}
```

Do not return the capability token, database IDs unrelated to the run, or raw messages from this endpoint.

### Step 6: Run tests and race detection

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisToolServer|TestRecordStepRewards|TestAssembleAssembledDag' -count=1
go test -race ./internal/service -run 'TestDiagnosisToolServer' -count=1
```

Expected: contract, isolation, idempotency, DAG regression, and race tests pass.

### Step 7: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/internal/service/diagnosis_tool_server.go multica/server/internal/service/diagnosis_tool_server_test.go multica/server/internal/service/interaction_dag.go multica/server/internal/service/interaction_dag_test.go
git commit -m "feat(multica): expose scoped diagnosis API"
```

## Task 6: Generate the trusted Pi diagnosis extension

**Files:**

- Create: `multica/server/internal/service/diagnosis_pi_extension.go`
- Create: `multica/server/internal/service/diagnosis_pi_extension_test.go`

### Step 1: Write failing generator tests

Generate the extension into a `t.TempDir()` and verify:

- the file is a regular file below the requested trusted root;
- permissions are owner-only where supported;
- it registers exactly five tools matching the server endpoints;
- schemas reject unknown properties and enforce score bounds;
- the base URL and token are read from environment variables at runtime, never embedded in source;
- HTTP requests set the bearer header, content type, timeout, and bounded response reader;
- server error codes are returned to Pi without token or response-body leakage;
- cleanup removes the generated file and does not touch sibling files.

### Step 2: Run the tests and confirm failure

```bash
cd multica/server
go test ./internal/service -run 'TestGenerateDiagnosisPiExtension' -count=1
```

Expected: compile failure because the extension generator does not exist.

### Step 3: Generate a fixed extension, not model-authored code

Implement a Go function that writes a reviewed TypeScript template with these Pi tool names:

- `multica_get_segment_messages`
- `multica_record_step_rewards`
- `multica_get_diagnosis_progress`
- `multica_finish_segment`
- `multica_complete_diagnosis`

The extension reads `MULTICA_DIAGNOSIS_API_URL` and `MULTICA_DIAGNOSIS_CAPABILITY_TOKEN` from its process environment. It must never expose a generic HTTP tool, filesystem access, shell execution, arbitrary URL selection, or arbitrary run identifiers.

### Step 4: Define strict tool schemas

The fetch tool accepts only `segment_id` and optional opaque `cursor`. The reward tool accepts only the active segment ID and a bounded array of step ID, score, and concise rationale. Finish accepts only the active segment ID plus a structured memory-update acknowledgement. Complete takes no scope override.

Keep response payloads compact. Do not include internal stack traces or unbounded server responses in tool results.

### Step 5: Run tests

```bash
cd multica/server
go test ./internal/service -run 'TestGenerateDiagnosisPiExtension' -count=1
```

Expected: generation, schema, secret-separation, and cleanup tests pass.

### Step 6: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/internal/service/diagnosis_pi_extension.go multica/server/internal/service/diagnosis_pi_extension_test.go
git commit -m "feat(multica): generate Pi diagnosis tools"
```

## Task 7: Replace the one-shot diagnoser with the persistent per-segment state machine

**Files:**

- Modify: `multica/server/internal/service/diagnosis_agent.go`
- Modify: `multica/server/internal/service/diagnosis_agent_test.go`
- Create: `multica/server/internal/service/diagnosis_agent_integration_test.go`

### Step 1: Replace old expectations with failing state-machine tests

Keep regression tests for topology ordering and segment selection, then add tests proving:

- the bootstrap prompt contains task context, scoring rubric, ordered topology, segment/step IDs, and message counts, but no segment message bodies;
- one `PiRPCBackend` instance is created and reused;
- auto-compaction is disabled once;
- exactly one Pi prompt turn is issued per incomplete segment;
- a segment turn instructs Pi to page until `Complete`, record every expected step reward, and finish the segment;
- the next segment is not prompted until the server checkpoint says the prior segment is complete;
- normal compaction occurs after every completed segment;
- restart skips completed segments and resumes the cursor for the first incomplete segment;
- Pi claiming success without server-side completion is retried at most twice, then fails the run;
- reward and checkpoint writes survive a runner failure;
- the old rich all-messages prompt builder is no longer called.

### Step 2: Run the focused tests and confirm failure

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisAgent.*Persistent|TestDiagnosisAgent.*Resume|TestDiagnosisAgent.*Compact' -count=1
```

Expected: failures because the runner still makes one backend call with all messages and returns rewards in memory.

### Step 3: Refactor constructor dependencies

Inject factories/interfaces for:

- persistent Pi RPC backend;
- diagnosis state store;
- capability tool server;
- trusted extension generator;
- runtime token stats.

Keep production construction in the existing service package. Tests use deterministic fakes rather than starting the real Pi binary.

### Step 4: Build the topology-only bootstrap prompt

Include:

- task objective and bounded task context;
- scoring definition and output invariants;
- ordered segment graph with segment IDs, predecessor/successor IDs, step IDs, roles, timestamps, and message counts;
- current persisted progress;
- explicit instruction that messages must be obtained only through `multica_get_segment_messages`;
- the memory compression contract below.

Do not include raw message content. Hash the canonical ordered topology and require it to match a resumable checkpoint.

### Step 5: Implement one-turn-per-segment orchestration

For each incomplete segment:

1. Start or resume its checkpoint.
1. Prompt Pi once with the segment ID, expected step IDs, current cursor status, and remaining topology cues.
1. Let Pi call the extension tools within that turn until it records rewards and calls `finish_segment`.
1. Read authoritative progress from Multica after the turn.
1. If incomplete, send at most two bounded repair turns for the same segment; do not advance.
1. Once complete, force compaction and proceed to the next segment.

`Diagnose` should return a `DiagnosisReport` containing run ID, completed segment count, and final status. It should no longer return the reward slice because rewards are already persisted.

### Step 6: Encode the detailed compaction contract

After every segment, call `Compact` with instructions that rebuild a bounded structured memory containing:

1. **Scoring calibration:** rubric interpretation, score anchors, and any calibration corrections learned from observed outputs.
1. **Verified evidence:** concise facts tied to segment and step IDs, including contradictions and causal dependencies.
1. **Per-segment conclusions:** final scores already written, short rationale, confidence, and whether the conclusion is stable or may be revised only through an explicit conflict path.
1. **Cross-segment state:** unresolved hypotheses, dependencies on future segments, repeated failure patterns, and behavioral changes across branches.
1. **Future-use index:** names, identifiers, claims, artifacts, or decision points likely to reappear in remaining topology, with the segment where each originated.
1. **Coverage ledger:** completed segments, current/remaining segment IDs, last committed cursor state, and server-confirmed reward coverage.

The compacted memory must remove:

- raw message transcripts already reduced to evidence;
- repeated tool responses and duplicate arguments;
- discarded hypotheses with no remaining dependency;
- stylistic detail, pleasantries, and execution chatter;
- information relevant only to already-finalized scoring and with no cross-segment value;
- capability tokens, API URLs, secrets, or executable instructions copied from messages.

Require provenance by segment/step ID for every retained factual claim. Require explicit `unknown` markers rather than invented details. The database remains authoritative if the compacted coverage ledger disagrees with server progress.

### Step 7: Enforce the 80% emergency boundary

Before each page fetch, the extension result exposes current page completion and the runner/extension can query runtime usage. When usage reaches or exceeds 80%:

- finish processing the page already returned;
- persist rewards that are complete for that page;
- do not fetch another page;
- return control from the current Pi turn with an explicit partial-segment checkpoint;
- compact using the same structured contract plus current segment cursor/evidence;
- resume the same segment in a new turn from the server-issued cursor.

Never compact halfway through a message page, never infer cursor advancement from model text, and never mark the segment complete during emergency compaction.

### Step 8: Add an end-to-end fake-Pi integration test

Use the real state store, message pager, HTTP capability server, generated extension artifact, and DAG reward writer with a scripted fake Pi RPC process. Exercise:

- two segments with multiple pages;
- reward persistence after each segment;
- forced after-segment compaction;
- emergency 80% compaction in the second segment;
- simulated crash and resume from the persisted cursor;
- final `/dag` assembly containing all step rewards.

Assert that the initial prompt contains none of the sentinel raw message bodies and that the capability token appears in neither captured argv nor prompts/log output.

### Step 9: Run focused and integration tests

```bash
cd multica/server
go test ./internal/service -run 'TestDiagnosisAgent' -count=1
go test ./internal/service -run 'TestDiagnosisAgentIntegration' -count=1
go test -race ./internal/service -run 'TestDiagnosisAgentIntegration' -count=1
```

Expected: persistent-session, compaction, restart, secret-isolation, and assembled-DAG assertions pass.

### Step 10: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/internal/service/diagnosis_agent.go multica/server/internal/service/diagnosis_agent_test.go multica/server/internal/service/diagnosis_agent_integration_test.go
git commit -m "feat(multica): diagnose DAG segments on demand"
```

## Task 8: Wire terminal training, configuration, observability, and docs

**Files:**

- Modify: `multica/server/internal/service/training.go`
- Modify: `multica/server/internal/service/training_test.go`
- Modify: `multica/server/internal/service/training_config.go`
- Modify: `multica/server/internal/service/training_config_test.go`
- Modify: `customized_areal/tree_search/agents/diagnose_agent_flow.md`

### Step 1: Write failing wiring and configuration tests

Update the fake `Diagnoser` contract to return `DiagnosisReport`. Test:

- terminal training calls diagnosis before critic/close as today;
- training no longer calls `RecordStepRewards` after `Diagnose` returns;
- an interrupted diagnosis can resume on the next terminal-routing attempt;
- diagnosis failure remains soft-fail and preserves already committed rewards;
- the feature-disabled path does not start Pi or the tool server;
- invalid page sizes, retry counts, or context percentages fail config validation;
- defaults are after-every-segment compaction, 80% emergency threshold, 20 turns/page, 24 KiB/page, and two repair attempts.

### Step 2: Run the tests and confirm failure

```bash
cd multica/server
go test ./internal/service -run 'TestRouteTerminalTrainingTask.*Diagnosis|TestTrainingConfig.*Diagnosis' -count=1
```

Expected: compile/assertion failures from the old `[]StepReward` diagnoser contract and missing config.

### Step 3: Add backwards-compatible configuration fields

Extend the existing diagnosis configuration with defaults; do not alter `areal/api/cli_args.py`. Add environment-backed fields for:

- page turn limit;
- page byte limit;
- emergency context percentage;
- incomplete-segment repair attempts;
- compaction policy, currently accepting only `after_every_segment`.

Validate ranges at startup. Keep existing diagnosis enablement/model/provider variables working.

### Step 4: Update terminal training wiring

Change `Diagnoser` to return `DiagnosisReport`. Remove the post-return bulk `RecordStepRewards` call. Log run ID, completed/total segments, compaction count, page count, and resume count without message bodies or secrets. Preserve ordering and soft-fail semantics.

### Step 5: Update the Mermaid architecture document

Revise `diagnose_agent_flow.md` to show:

- topology-only bootstrap;
- one persistent Pi session;
- five extension tools and loopback capability server;
- page/cursor loop;
- incremental DAG reward writes;
- normal after-every-segment compaction;
- 80% safe-page-boundary emergency compaction;
- restart from persisted diagnosis checkpoints;
- DB authority over Pi's compressed memory.

Also add a concise table of default limits and explain that `--no-tools` disables Pi's ordinary tools while Multica loads one explicit reviewed extension.

### Step 6: Run service and documentation checks

```bash
cd multica/server
gofmt -w internal/service/diagnosis_*.go internal/service/training.go internal/service/training_config.go pkg/agent/agent.go pkg/agent/pi.go pkg/agent/pi_rpc.go pkg/db/generated/interaction_dag_diagnosis.sql.go
go test ./pkg/agent ./pkg/db/... ./internal/service -count=1
go test -race ./internal/service -run 'TestDiagnosisAgentIntegration|TestDiagnosisToolServer' -count=1
cd ../..
UV_PYTHON=/home/vscode/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 uv run --no-project --with mdformat==0.7.17 --with mdformat-gfm --with mdformat-tables --with mdformat-frontmatter mdformat --check customized_areal/tree_search/agents/diagnose_agent_flow.md docs/superpowers/specs/2026-07-22-multica-diagnosis-agent-on-demand-design.md
```

Expected: all Go tests, race tests, formatting, and Markdown checks pass.

If Mermaid CLI is present, also render the diagram:

```bash
npx --yes @mermaid-js/mermaid-cli -i customized_areal/tree_search/agents/diagnose_agent_flow.md -o /tmp/multica-diagnosis-flow.svg
```

Expected: command exits successfully. If the CLI is unavailable, record that the Markdown parser check passed and report the render check as skipped.

### Step 7: Update the repository knowledge graph

Because code changed and `graphify-out/graph.json` exists, run:

```bash
graphify update .
```

Expected: the graph updates successfully. If `graphify` is not installed in the environment, report the skipped update explicitly; do not edit generated graph files by hand.

### Step 8: Run final repository-scope verification

```bash
cd multica/server
go test ./... -count=1
cd ../..
git diff --check
git status --short
```

Expected: tests pass, no whitespace errors, and status lists only intentional files. If integration suites require unavailable external services or hardware, run all unit suites, name each skipped suite, and include the exact environmental reason.

### Step 9: Commit

Load `.agents/skills/commit-conventions/SKILL.md`, then:

```bash
git add multica/server/internal/service/training.go multica/server/internal/service/training_test.go multica/server/internal/service/training_config.go multica/server/internal/service/training_config_test.go customized_areal/tree_search/agents/diagnose_agent_flow.md
git commit -m "docs(multica): document on-demand diagnosis"
```

## Completion checklist

- [ ] Initial Pi prompt contains no raw segment messages.
- [ ] Exactly one persistent Pi RPC process/session is used per diagnosis run.
- [ ] All message reads use scoped opaque cursors and server-side coverage checks.
- [ ] All rewards are persisted incrementally, idempotently, and visible in `/dag`.
- [ ] Every completed segment triggers structured compaction.
- [ ] 80% context usage stops at a safe page boundary and resumes the same segment.
- [ ] Restart resumes from persisted run, segment, cursor, and reward state.
- [ ] Pi launches with no ordinary tools and exactly one reviewed trusted extension.
- [ ] Capability secrets are absent from argv, prompts, logs, and compressed memory.
- [ ] Terminal routing preserves critic/close ordering and soft-fail behavior.
- [ ] Focused unit, integration, race, formatting, migration, and documentation checks pass.
- [ ] `graphify update .` succeeds or its tool-unavailable skip is recorded.
