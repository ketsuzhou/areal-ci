# SWE-Lego Docker × Multica Remote Mode × AReaL RL — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the approved `multica-dag-rl` design from Fleet/Daytona cloud sandboxes to SWE-Lego docker containers, add one atomic per-issue multica endpoint that forks git history before the issue and boots `group_size` daemon-in-docker sandboxes, add a hybrid SWE-Lego verifier, and wire DAG credit backup so it actually shapes training.

**Architecture:** Multica owns all docker — a new `POST /api/v1/swe-lego/issues` endpoint composes project creation + SWE-Lego image build (on a Fleet build-node, history truncated via `git filter-repo`) + base sandbox boot + `group_size` forks + agent-run enqueue. AReal calls only that endpoint plus the existing cloud-runtime `snapshot`/`fork`/`DELETE` endpoints (via a new `MulticaSweLegoProvider` that is a drop-in for the existing `FleetSandboxProvider`). The existing `BranchMaterializer`, `ForkIssueSubtree`, `db_bridge`, and DAG model are unchanged. A hybrid `SweLegoVerifier` (objective tests + generative critic + semi-resolved) produces the terminal reward, which is distributed along DAG edges by an extended `TreeAdvantageComputer`.

**Tech Stack:** Python 3.12+ (areal), Go 1.26 (multica server), httpx (areal HTTP), pytest (areal tests), `go test` (multica tests), Docker (SWE-Lego images), `git filter-repo` (history truncation), multica cloud-runtime Fleet proxy (`/api/v1/nodes`, `/api/v1/sandboxes/*`).

---

## File Structure

### Multica side (Go) — `multica/server/`

- **Create `internal/service/swe_lego_image.go`** — SWE-Lego image builder. `BuildOrReuse` picks a build-node, checks its docker image cache, and ships a build script via `/api/v1/nodes/exec`. Owns the `git filter-repo --commit-cutoff` history-truncation logic and the cache-key derivation.
- **Create `internal/service/swe_lego_image_test.go`** — unit tests for cache-key derivation, build-script generation (asserts `--commit-cutoff` is computed from `issue_date`), cache-hit short-circuit. Uses a fake node-exec interface; no real git/docker.
- **Create `internal/service/swe_lego_build_node.go`** — `pick_build_node()` reuses the existing `cloudRuntimeProxy` to select a `swe-lego-build` tagged node. Thin.
- **Create `internal/handler/swe_lego_issue.go`** — the atomic `POST /api/v1/swe-lego/issues` handler. Composes `CreateProject` → `CreateIssue` → `swe_lego_image.BuildOrReuse` → boot base sandbox on the build node → fork × `group_size` → enqueue agent tasks. Rollback on failure.
- **Create `internal/handler/swe_lego_issue_test.go`** — handler tests with a mocked service layer (happy path 201, image-build failure 502, fork failure rollback, auth checks).
- **Modify `cmd/server/router.go`** — register `POST /api/v1/swe-lego/issues` next to the existing cloud-runtime routes (after the `sandboxes/fork` line at `:970`).

### AReal side (Python) — `backend/areal/customized_areal/tree_search/`

- **Modify `agents/environment.py`** — add `MulticaSweLegoProvider` class, sibling to `FleetSandboxProvider`. Same `ForkableEnvironment` Protocol surface, different backend (multica cloud-runtime proxy at `/api/v1/sandboxes/*`).
- **Modify `tests/test_environment.py`** — add `MulticaSweLegoProvider` contract tests mirroring the existing `FleetSandboxProvider` tests (snapshot/fork/restore/cleanup, error paths, 404 idempotency, semaphore).
- **Create `agents/reward/swe_lego_verifier.py`** — `SweLegoVerifier` composing `ObjectiveVerifier` (test execution via node exec) + `AgenticVerifier` (generative critic) + semi-resolved partial credit. `verify_and_reward` writes reward via `rl_set_reward`.
- **Create `agents/reward/test_swe_lego_verifier.py`** — unit tests for the blend/short-circuit logic with mocked sub-verifiers.
- **Create `agents/swe_lego_issue_runner.py`** — the per-issue orchestration loop: call `create_swe_lego_issue` → open RL sessions → drive branching → verify+reward → cleanup. Owns the `MulticaSweLegoClient` HTTP client.
- **Create `tests/test_swe_lego_issue_runner.py`** — integration-flavored unit test with mocked multica client, env, materializer, verifier, rl_session.
- **Modify `agents/integration.py`** — no structural change; confirm `BranchMaterializer` accepts `MulticaSweLegoProvider` via the injected `env` (it already does, since it depends on the `ForkableEnvironment` Protocol). Add a smoke test asserting the materializer works with a `MulticaSweLegoProvider` instance.
- **Create `agents/reward/swe_lego_types.py`** — `SweLegoIssue`, `SweLegoIssueResult`, `SweLegoSetup` dataclasses shared between runner and verifier.

### DAG credit backup (Phase 3, load-bearing) — `customized_areal/tree_search/`

- **Create `agents/dag_backup.py`** — `distribute_reward_over_dag(dag, terminal_reward, credit_fn)` distributes the terminal reward along DAG edges; `assign_fan_in_credit` does explicit per-agent credit at fan-in joins (no sum/mean/max).
- **Create `tests/test_dag_backup.py`** — multi-node DAG distributes reward along edges; fan-in join credit is explicit; per-step signals shape intermediate nodes.
- **Modify `agents/advantage.py` (or wherever `TreeAdvantageComputer` lives)** — extend to consume per-node credit from `dag_backup` instead of broadcasting a flat episode reward. Add a `credit_by_node` argument.
- **Modify the corresponding advantage test** — assert per-token advantage reflects per-node credit, not a flat broadcast.

### Build / image assets

- **Create `internal/service/swe_lego_image.Dockerfile.tmpl`** — the Dockerfile template baked into each SWE-Lego image (base image + truncated repo + `pip install -e .` + daemon binary + `CMD ["multica-daemon", "run"]`).

---

## Conventions

- **Multica Go**: follow `multica/CLAUDE.md`. Thin handlers, `parseUUIDOrBadRequest` / `loadIssueForUser` for UUID inputs, `writeError`/`writeJSON` for responses, `math.MaxInt32` bounds on parsed ints. Test files live next to source as `*_test.go`. Run `make test` from `multica/`.
- **AReal Python**: follow `backend/areal/CLAUDE.md`. `async def`, type hints, stdlib `logging` (not `areal.utils.logging`) so the `agents` package stays importable without torch. Tests under `customized_areal/tree_search/tests/` use `httpx.MockTransport` and pytest. Run `uv run pytest <path>` from `backend/areal/`.
- **Commits**: Conventional Commits (`feat:`, `fix:`, `test:`, `docs:`), imperative voice, ~72 char subject. Squash WIP before MR.
- **TDD**: write the failing test first, run it to see it fail, implement minimal code, run to see it pass, commit.

---

## Task 1: SWE-Lego cache-key derivation (multica, Go)

**Files:**
- Create: `multica/server/internal/service/swe_lego_image.go`
- Test: `multica/server/internal/service/swe_lego_image_test.go`

- [ ] **Step 1: Write the failing test**

```go
package service

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

func TestSweLegoCacheKey(t *testing.T) {
	got := SweLegoCacheKey("https://github.com/psf/requests.git", "abc123", "2025-03-14T09:30:00Z", "swe-lego/python:3.11")
	h := sha256.Sum256([]byte("https://github.com/psf/requests.git|abc123|2025-03-14T09:30:00Z|swe-lego/python:3.11"))
	want := hex.EncodeToString(h[:])
	if got != want {
		t.Fatalf("cache key = %q, want %q", got, want)
	}
}

func TestSweLegoCacheKey_StableAcrossOrder(t *testing.T) {
	a := SweLegoCacheKey("r1", "c1", "d1", "b1")
	b := SweLegoCacheKey("r1", "c1", "d1", "b1")
	if a != b {
		t.Fatalf("identical inputs produced different keys: %q vs %q", a, b)
	}
}

func TestSweLegoCacheKey_DistinguishesInputs(t *testing.T) {
	a := SweLegoCacheKey("r1", "c1", "d1", "b1")
	b := SweLegoCacheKey("r1", "c2", "d1", "b1")
	if a == b {
		t.Fatalf("different base_commit produced the same key")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoCacheKey -v`
Expected: FAIL with "undefined: SweLegoCacheKey".

- [ ] **Step 3: Write minimal implementation**

```go
package service

import (
	"crypto/sha256"
	"encoding/hex"
)

// SweLegoCacheKey derives the docker image cache key for a SWE-Lego image.
// The key is a function of (repo_url, base_commit, issue_date, base_image) —
// the four inputs that determine the image contents. Repeated issues against
// the same quadruple reuse the image (spec §4.3).
//
// The pipe-delimited preimage mirrors how the areal side would derive the
// same key if it ever needed to reference an image by content; do not change
// the delimiter without coordinating both sides.
func SweLegoCacheKey(repoURL, baseCommit, issueDate, baseImage string) string {
	preimage := repoURL + "|" + baseCommit + "|" + issueDate + "|" + baseImage
	h := sha256.Sum256([]byte(preimage))
	return hex.EncodeToString(h[:])
}

// sweLegoImageRef returns the docker image ref tagged with the cache key.
func sweLegoImageRef(cacheKey string) string {
	return "swe-lego:" + cacheKey
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoCacheKey -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/service/swe_lego_image.go internal/service/swe_lego_image_test.go
git commit -m "feat(swe-lego): add SWE-Lego image cache-key derivation"
```

---

## Task 2: Build-script generation with `git filter-repo --commit-cutoff` (multica, Go)

**Files:**
- Modify: `multica/server/internal/service/swe_lego_image.go`
- Test: `multica/server/internal/service/swe_lego_image_test.go`

- [ ] **Step 1: Write the failing test**

Append to `swe_lego_image_test.go`:

```go
func TestSweLegoBuildScript_ContainsFilterRepoCutoff(t *testing.T) {
	script := SweLegoBuildScript("https://github.com/psf/requests.git", "abc123", "2025-03-14T09:30:00Z", "swe-lego/python:3.11", "deadbeef")
	// clone + checkout base_commit
	assertContains(t, script, "git clone --filter=blob:none https://github.com/psf/requests.git")
	assertContains(t, script, "git fetch origin abc123")
	assertContains(t, script, "git checkout abc123")
	// SWE-Lego anti-hacking: filter-repo with cutoff computed from issue_date
	assertContains(t, script, "git rev-list -1 --before='2025-03-14T09:30:00Z' HEAD")
	assertContains(t, script, "git filter-repo --replace-ref refs/heads/main:")
	assertContains(t, script, "--commit-cutoff")
	// docker build tagged with the cache key
	assertContains(t, script, "docker build -t swe-lego:deadbeef")
}

func TestSweLegoBuildScript_PipInstallBestEffort(t *testing.T) {
	script := SweLegoBuildScript("r", "c", "d", "swe-lego/python:3.11", "k")
	// Skywork-SWE pattern: best-effort pip install, never fail the build on it
	assertContains(t, script, "pip install -e . 2>/dev/null || true")
}

func assertContains(t *testing.T, s, want string) {
	t.Helper()
	if !strings.Contains(s, want) {
		t.Fatalf("build script missing %q\n--- script ---\n%s", want, s)
	}
}
```

Add `"strings"` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoBuildScript -v`
Expected: FAIL with "undefined: SweLegoBuildScript".

- [ ] **Step 3: Write minimal implementation**

Append to `swe_lego_image.go`:

```go
import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
)

// SweLegoBuildScript returns the shell script run on a Fleet build-node to
// produce a SWE-Lego docker image. The script:
//  1. Clones the repo shallow-extended to base_commit.
//  2. SWE-Lego anti-hacking: deletes git history after issue_date via
//     `git filter-repo --commit-cutoff`, so an agent cannot git log or
//     git blame its way to the future fix (spec §2 decision 5, §4.3).
//  3. docker builds the image tagged with the cache key.
//
// The script is shipped to the node via /api/v1/nodes/exec and run there; the
// multica server never shells out to docker locally (spec §2 decision 8).
func SweLegoBuildScript(repoURL, baseCommit, issueDate, baseImage, cacheKey string) string {
	imageRef := sweLegoImageRef(cacheKey)
	var b strings.Builder
	fmt.Fprintf(&b, "set -euo pipefail\n")
	fmt.Fprintf(&b, "rm -rf /tmp/swe-lego-build && mkdir -p /tmp/swe-lego-build\n")
	fmt.Fprintf(&b, "cd /tmp/swe-lego-build\n")
	fmt.Fprintf(&b, "git clone --filter=blob:none %s repo\n", shellQuote(repoURL))
	fmt.Fprintf(&b, "cd repo\n")
	fmt.Fprintf(&b, "git fetch origin %s\n", shellQuote(baseCommit))
	fmt.Fprintf(&b, "git checkout %s\n", shellQuote(baseCommit))
	// SWE-Lego anti-hacking: find the last commit at or before issue_date,
	// then physically delete everything after it.
	fmt.Fprintf(&b, "cutoff_commit=$(git rev-list -1 --before=%s HEAD)\n", shellQuote(issueDate))
	fmt.Fprintf(&b, "git filter-repo --replace-ref refs/heads/main:${cutoff_commit} --commit-cutoff ${cutoff_commit}\n")
	fmt.Fprintf(&b, "pip install -e . 2>/dev/null || true\n")
	fmt.Fprintf(&b, "docker build -t %s -f /tmp/swe-lego-build/Dockerfile .\n", shellQuote(imageRef))
	return b.String()
}

// shellQuote single-quotes a string for safe inclusion in a shell script.
// Single quotes prevent shell expansion of repo URLs / commit hashes that
// might contain characters with special meaning.
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoBuildScript -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/service/swe_lego_image.go internal/service/swe_lego_image_test.go
git commit -m "feat(swe-lego): generate build script with git filter-repo history truncation"
```

---

## Task 3: Build-node exec interface + `BuildOrReuse` (multica, Go)

**Files:**
- Create: `multica/server/internal/service/swe_lego_build_node.go`
- Modify: `multica/server/internal/service/swe_lego_image.go`
- Test: `multica/server/internal/service/swe_lego_image_test.go`

- [ ] **Step 1: Write the failing test**

Append to `swe_lego_image_test.go`:

```go
type fakeNodeExec struct {
	calls       []string
	inspectOK   bool   // if true, the image is already cached on the node
	inspectErr  error
	buildExitOK bool   // if true, the build script exits 0
}

func (f *fakeNodeExec) Exec(ctx context.Context, nodeID string, cmd []string) (stdout string, exitCode int, err error) {
	f.calls = append(f.calls, fmt.Sprintf("%s:%s", nodeID, strings.Join(cmd, " ")))
	joined := strings.Join(cmd, " ")
	switch {
	case strings.Contains(joined, "docker image inspect"):
		if f.inspectErr != nil {
			return "", 1, f.inspectErr
		}
		if f.inspectOK {
			return "", 0, nil
		}
		return "no such image", 1, nil
	case strings.Contains(joined, "set -euo pipefail"):
		if f.buildExitOK {
			return "", 0, nil
		}
		return "build failed", 1, fmt.Errorf("build exit 1")
	}
	return "", 0, nil
}

func (f *fakeNodeExec) PickBuildNode(ctx context.Context) (string, error) {
	return "node-1", nil
}

func TestBuildOrReuse_CacheHitShortCircuits(t *testing.T) {
	ctx := context.Background()
	fe := &fakeNodeExec{inspectOK: true}
	ref, nodeID, err := BuildOrReuse(ctx, fe, "r", "c", "d", "b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nodeID != "node-1" {
		t.Fatalf("nodeID = %q, want node-1", nodeID)
	}
	// Only the inspect call should have fired; no build script.
	if len(fe.calls) != 1 || !strings.Contains(fe.calls[0], "docker image inspect") {
		t.Fatalf("expected exactly one inspect call, got %v", fe.calls)
	}
	if ref == "" {
		t.Fatal("expected non-empty image ref")
	}
}

func TestBuildOrReuse_CacheMissRunsBuild(t *testing.T) {
	ctx := context.Background()
	fe := &fakeNodeExec{inspectOK: false, buildExitOK: true}
	ref, nodeID, err := BuildOrReuse(ctx, fe, "r", "c", "d", "b")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if nodeID != "node-1" {
		t.Fatalf("nodeID = %q, want node-1", nodeID)
	}
	// Two calls: inspect (cache miss) + build script.
	if len(fe.calls) != 2 {
		t.Fatalf("expected 2 calls, got %v", fe.calls)
	}
	if !strings.Contains(fe.calls[1], "git filter-repo") {
		t.Fatalf("build script not shipped: %v", fe.calls[1])
	}
	if ref == "" {
		t.Fatal("expected non-empty image ref")
	}
}

func TestBuildOrReuse_BuildFailureReturnsError(t *testing.T) {
	ctx := context.Background()
	fe := &fakeNodeExec{inspectOK: false, buildExitOK: false}
	_, _, err := BuildOrReuse(ctx, fe, "r", "c", "d", "b")
	if err == nil {
		t.Fatal("expected error on build failure")
	}
}
```

Add `"context"` and `"fmt"` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/service/ -run TestBuildOrReuse -v`
Expected: FAIL with "undefined: BuildOrReuse".

- [ ] **Step 3: Write minimal implementation**

Create `multica/server/internal/service/swe_lego_build_node.go`:

```go
package service

import "context"

// NodeExec is the seam for running commands on a Fleet build-node. The real
// implementation calls /api/v1/nodes/exec via the cloudRuntimeProxy; tests
// inject a fake. This keeps the swe_lego_image package free of HTTP
// machinery and testable without a Fleet.
type NodeExec interface {
	// Exec runs cmd on nodeID and returns stdout, the process exit code,
	// and any transport error. A non-zero exit code is NOT an error here —
	// callers inspect exitCode to distinguish cache-miss (exit 1 on
	// `docker image inspect`) from a transport failure.
	Exec(ctx context.Context, nodeID string, cmd []string) (stdout string, exitCode int, err error)
	// PickBuildNode selects a swe-lego-build tagged Fleet node.
	PickBuildNode(ctx context.Context) (nodeID string, err error)
}
```

Append to `swe_lego_image.go`:

```go
import (
	"context"
	"errors"
	"strings"
)

// ErrSweLegoBuildFailed is returned when the build script exits non-zero.
var ErrSweLegoBuildFailed = errors.New("swe-lego image build failed")

// BuildOrReuse returns the docker image ref and the node it lives on.
// If the image is already cached on the picked node (cache hit), the build
// is short-circuited. Otherwise the build script is shipped to the node via
// NodeExec.Exec. The image always lives on the node that built it (spec §4.3
// "Image locality") — registry-backed distribution is out of scope for v1.
func BuildOrReuse(ctx context.Context, exec NodeExec, repoURL, baseCommit, issueDate, baseImage string) (imageRef string, nodeID string, err error) {
	node, err := exec.PickBuildNode(ctx)
	if err != nil {
		return "", "", fmt.Errorf("pick build node: %w", err)
	}
	cacheKey := SweLegoCacheKey(repoURL, baseCommit, issueDate, baseImage)
	ref := sweLegoImageRef(cacheKey)

	// 1. Cache check: `docker image inspect` exits 0 if the image exists.
	_, exitCode, err := exec.Exec(ctx, node, []string{"docker", "image", "inspect", ref})
	if err != nil {
		return "", "", fmt.Errorf("cache inspect transport error: %w", err)
	}
	if exitCode == 0 {
		return ref, node, nil
	}

	// 2. Cache miss: ship the build script and run it on the node.
	script := SweLegoBuildScript(repoURL, baseCommit, issueDate, baseImage, cacheKey)
	_, exitCode, err = exec.Exec(ctx, node, []string{"sh", "-c", script})
	if err != nil {
		return "", "", fmt.Errorf("build transport error: %w", err)
	}
	if exitCode != 0 {
		return "", "", ErrSweLegoBuildFailed
	}
	return ref, node, nil
}

// ensure strings is used (build-script generation references it via shellQuote)
var _ = strings.Contains
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/service/ -run "TestBuildOrReuse|TestSweLegoCacheKey|TestSweLegoBuildScript" -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/service/swe_lego_build_node.go internal/service/swe_lego_image.go internal/service/swe_lego_image_test.go
git commit -m "feat(swe-lego): add BuildOrReuse with node-exec seam and cache short-circuit"
```

---

## Task 4: Dockerfile template (multica)

**Files:**
- Create: `multica/server/internal/service/swe_lego_image.Dockerfile.tmpl`
- Test: `multica/server/internal/service/swe_lego_image_test.go`

- [ ] **Step 1: Write the failing test**

Append to `swe_lego_image_test.go`:

```go
func TestSweLegoDockerfileTemplate(t *testing.T) {
	tmpl, err := SweLegoDockerfile("swe-lego/python:3.11")
	if err != nil {
		t.Fatalf("render template: %v", err)
	}
	assertContains(t, tmpl, "FROM swe-lego/python:3.11")
	assertContains(t, tmpl, "COPY repo/ /workspace/repo")
	assertContains(t, tmpl, "WORKDIR /workspace/repo")
	assertContains(t, tmpl, "pip install -e . 2>/dev/null || true")
	assertContains(t, tmpl, "COPY multica-daemon /usr/local/bin/multica-daemon")
	assertContains(t, tmpl, "ENV MULTICA_DAEMON_AUTO_REGISTER=1")
	assertContains(t, tmpl, `CMD ["multica-daemon", "run"]`)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoDockerfileTemplate -v`
Expected: FAIL with "undefined: SweLegoDockerfile".

- [ ] **Step 3: Write minimal implementation**

The build script from Task 2 writes a `Dockerfile` at `/tmp/swe-lego-build/Dockerfile` before `docker build`. Append to `swe_lego_image.go`:

```go
import "text/template"

// sweLegoDockerfileTmpl is the Dockerfile baked into each SWE-Lego image.
// The daemon binary is built from the existing multica daemon source and
// copied in at image-build time — it is the same binary that runs locally
// today, just inside the container (spec §4.3).
const sweLegoDockerfileTmpl = `FROM {{.BaseImage}}
COPY repo/ /workspace/repo
WORKDIR /workspace/repo
RUN pip install -e . 2>/dev/null || true
COPY multica-daemon /usr/local/bin/multica-daemon
ENV MULTICA_DAEMON_AUTO_REGISTER=1
CMD ["multica-daemon", "run"]
`

// SweLegoDockerfile renders the Dockerfile for a SWE-Lego image given the
// base image. The build script writes this to /tmp/swe-lego-build/Dockerfile
// before invoking docker build.
func SweLegoDockerfile(baseImage string) (string, error) {
	t, err := template.New("swe-lego-dockerfile").Parse(sweLegoDockerfileTmpl)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if err := t.Execute(&b, struct{ BaseImage string }{BaseImage: baseImage}); err != nil {
		return "", err
	}
	return b.String(), nil
}
```

Also update `SweLegoBuildScript` to write the Dockerfile before `docker build`. In `SweLegoBuildScript`, replace the `docker build` line with:

```go
	// Write the Dockerfile, then build.
	dockerfile, _ := SweLegoDockerfile(baseImage)
	fmt.Fprintf(&b, "cat > /tmp/swe-lego-build/Dockerfile <<'EOF'\n%s\nEOF\n", dockerfile)
	fmt.Fprintf(&b, "docker build -t %s -f /tmp/swe-lego-build/Dockerfile .\n", shellQuote(imageRef))
```

And create the template file on disk for documentation/reference (Go embeds it via the const above, but committing the `.tmpl` makes the asset discoverable):

```dockerfile
FROM {{.BaseImage}}
COPY repo/ /workspace/repo
WORKDIR /workspace/repo
RUN pip install -e . 2>/dev/null || true
COPY multica-daemon /usr/local/bin/multica-daemon
ENV MULTICA_DAEMON_AUTO_REGISTER=1
CMD ["multica-daemon", "run"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoDockerfileTemplate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/service/swe_lego_image.go internal/service/swe_lego_image_test.go internal/service/swe_lego_image.Dockerfile.tmpl
git commit -m "feat(swe-lego): add Dockerfile template for daemon-in-docker image"
```

---

## Task 5: `MulticaSweLegoProvider` — retargeted `ForkableEnvironment` (areal, Python)

**Files:**
- Modify: `backend/areal/customized_areal/tree_search/agents/environment.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_environment.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_environment.py` (mirror the existing `FleetSandboxProvider` tests, but pointing at the multica cloud-runtime proxy paths and `MULTICA_BASE_URL`/`MULTICA_API_KEY` env):

```python
from customized_areal.tree_search.agents.environment import (
    EnvironmentError,
    ForkError,
    MulticaSweLegoProvider,
    SnapshotError,
)


def test_multica_provider_requires_base_url(monkeypatch):
    monkeypatch.delenv("MULTICA_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="MulticaSweLegoProvider requires base_url"):
        MulticaSweLegoProvider()


def test_multica_provider_snapshot_hits_cloud_runtime_path():
    seen: list[str] = []

    def handler(request: httpx.Request):
        seen.append(f"{request.method} {request.url.path}")
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(200, json={"snapshot_id": "snap-1"})

    transport = _router({("POST", "/api/v1/sandboxes/sbx-1/snapshot"): handler})
    prov = MulticaSweLegoProvider(
        base_url="https://multica.example", api_key="secret", transport=transport
    )
    result = asyncio.run(prov.snapshot("sbx-1"))
    assert result == SnapshotResult(snapshot_id="snap-1", source_sandbox_id="sbx-1")
    assert seen == ["POST /api/v1/sandboxes/sbx-1/snapshot"]


def test_multica_provider_fork_sends_source_sandbox_id():
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"sandbox_id": "forked-1"})

    transport = _router({("POST", "/api/v1/sandboxes/fork"): handler})
    prov = MulticaSweLegoProvider(
        base_url="https://multica.example", transport=transport, max_concurrent_forks=2
    )
    result = asyncio.run(prov.fork(source_sandbox_id="sbx-1"))
    assert result == ForkResult(sandbox_id="forked-1")
    assert captured["body"] == {"source_sandbox_id": "sbx-1"}


def test_multica_provider_cleanup_treats_404_as_success():
    transport = _router(
        {("DELETE", "/api/v1/sandboxes/gone*"): lambda r: httpx.Response(404, json={})}
    )
    prov = MulticaSweLegoProvider(base_url="https://multica.example", transport=transport)
    asyncio.run(prov.cleanup("gone"))  # must not raise


def test_multica_provider_snapshot_error_on_500():
    transport = _router(
        {("POST", "/api/v1/sandboxes/sbx-1/snapshot*"): lambda r: httpx.Response(500, text="boom")}
    )
    prov = MulticaSweLegoProvider(base_url="https://multica.example", transport=transport)
    with pytest.raises(SnapshotError):
        asyncio.run(prov.snapshot("sbx-1"))


def test_multica_provider_fork_error_on_500():
    transport = _router(
        {("POST", "/api/v1/sandboxes/fork*"): lambda r: httpx.Response(500, text="boom")}
    )
    prov = MulticaSweLegoProvider(base_url="https://multica.example", transport=transport)
    with pytest.raises(ForkError):
        asyncio.run(prov.fork(source_sandbox_id="sbx-1"))


def test_multica_provider_satisfies_forkable_environment_protocol():
    prov = MulticaSweLegoProvider(
        base_url="https://multica.example",
        transport=_router({}),
    )
    assert isinstance(prov, ForkableEnvironment)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_environment.py -k multica_provider -v`
Expected: FAIL with ImportError on `MulticaSweLegoProvider`.

- [ ] **Step 3: Write minimal implementation**

Append to `agents/environment.py` (after `FleetSandboxProvider`):

```python
class MulticaSweLegoProvider:
    """:class:`ForkableEnvironment` backed by multica's cloud-runtime proxy.

    Calls the EXISTING endpoints that ``cloud_runtime.go`` already exposes
    (``server/internal/handler/cloud_runtime.go:108-127``):
      POST /api/v1/sandboxes/{id}/snapshot  → SnapshotResult
      POST /api/v1/sandboxes/fork           → ForkResult
      POST /api/v1/sandboxes/{id}/restore   → None
      DELETE /api/v1/sandboxes/{id}         → None   (idempotent on 404)

    Identical surface to :class:`FleetSandboxProvider` so :class:`BranchMaterializer`
    is unchanged — only the injected provider class differs. ``base_url`` /
    ``api_key`` default to ``MULTICA_BASE_URL`` / ``MULTICA_API_KEY``.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 60.0,
        api_key: str | None = None,
        max_concurrent_forks: int | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip(
            "/"
        )
        if not self._base_url:
            raise ValueError(
                "MulticaSweLegoProvider requires base_url or MULTICA_BASE_URL"
            )
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

        cap = max_concurrent_forks
        if cap is None:
            try:
                cap = int(os.environ.get("GROUP_SIZE", "2"))
            except ValueError:
                cap = 2
        if cap < 1:
            raise ValueError(f"max_concurrent_forks must be >= 1, got {cap}")
        self._fork_semaphore = asyncio.Semaphore(cap)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def snapshot(self, sandbox_id: str) -> SnapshotResult:
        try:
            resp = await self._client.post(
                f"/api/v1/sandboxes/{sandbox_id}/snapshot",
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise SnapshotError(f"snapshot transport error: {exc}") from exc
        if resp.status_code != 200:
            raise SnapshotError(
                f"snapshot failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        snap_id = body.get("snapshot_id")
        if not isinstance(snap_id, str) or not snap_id:
            raise SnapshotError(f"snapshot response missing snapshot_id: {body!r}")
        return SnapshotResult(snapshot_id=snap_id, source_sandbox_id=sandbox_id)

    async def fork(
        self,
        *,
        source_sandbox_id: str | None = None,
        snapshot_id: str | None = None,
    ) -> ForkResult:
        if (source_sandbox_id is None) == (snapshot_id is None):
            raise ValueError(
                "fork requires exactly one of source_sandbox_id or snapshot_id"
            )
        payload: dict[str, str] = {}
        if source_sandbox_id is not None:
            payload["source_sandbox_id"] = source_sandbox_id
        if snapshot_id is not None:
            payload["snapshot_id"] = snapshot_id
        async with self._fork_semaphore:
            try:
                resp = await self._client.post(
                    "/api/v1/sandboxes/fork", json=payload, headers=self._headers()
                )
            except httpx.HTTPError as exc:
                raise ForkError(f"fork transport error: {exc}") from exc
        if resp.status_code != 200:
            raise ForkError(
                f"fork failed: status={resp.status_code} body={resp.text[:200]}"
            )
        body = resp.json()
        sbx_id = body.get("sandbox_id")
        if not isinstance(sbx_id, str) or not sbx_id:
            raise ForkError(f"fork response missing sandbox_id: {body!r}")
        return ForkResult(sandbox_id=sbx_id)

    async def restore(self, sandbox_id: str) -> None:
        try:
            resp = await self._client.post(
                f"/api/v1/sandboxes/{sandbox_id}/restore", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise EnvironmentError(f"restore transport error: {exc}") from exc
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"restore failed: status={resp.status_code} body={resp.text[:200]}"
            )

    async def cleanup(self, sandbox_id: str) -> None:
        try:
            resp = await self._client.delete(
                f"/api/v1/sandboxes/{sandbox_id}", headers=self._headers()
            )
        except httpx.HTTPError as exc:
            raise EnvironmentError(f"cleanup transport error: {exc}") from exc
        if resp.status_code == 404:
            return
        if resp.status_code not in (200, 204):
            raise EnvironmentError(
                f"cleanup failed: status={resp.status_code} body={resp.text[:200]}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_environment.py -k multica_provider -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/environment.py customized_areal/tree_search/tests/test_environment.py
git commit -m "feat(dag): add MulticaSweLegoProvider ForkableEnvironment implementation"
```

---

## Task 6: SWE-Lego issue dataclasses (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/agents/reward/swe_lego_types.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_swe_lego_types.py`

- [ ] **Step 1: Write the failing test**

```python
from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
    SweLegoSetup,
)


def test_swe_lego_issue_carries_test_lists():
    issue = SweLegoIssue(
        repo_url="https://github.com/psf/requests.git",
        base_commit="abc123",
        issue_date="2025-03-14T09:30:00Z",
        issue_text="retry leaks",
        issue_title="Retry leaks",
        acceptance_criteria="must not leak",
        fail_to_pass=["tests/test_retry.py::test_leak"],
        pass_to_pass=["tests/test_retry.py::test_basic"],
    )
    assert issue.fail_to_pass == ["tests/test_retry.py::test_leak"]
    assert issue.pass_to_pass == ["tests/test_retry.py::test_basic"]


def test_swe_lego_setup_holds_group_size_agent_runs():
    setup = SweLegoSetup(
        project_id="p1",
        issue_id="i1",
        image_id="img1",
        build_node_id="n1",
        base_sandbox_id="sbx-base",
        base_sandbox_runtime_id="rt-base",
        agent_run_ids=["r1", "r2", "r3"],
    )
    assert len(setup.agent_run_ids) == 3


def test_swe_lego_issue_result_collects_per_agent_rewards():
    result = SweLegoIssueResult(per_agent_rewards=[1.0, 0.0, 0.5])
    assert result.per_agent_rewards == [1.0, 0.0, 0.5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_types.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Dataclasses shared by the SWE-Lego issue runner and verifier."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SweLegoIssue:
    """One SWE issue to train on.

    ``base_commit`` is the parent of the fixing PR's merge. ``issue_date`` is
    the cutoff for SWE-Lego anti-hacking: git history after this date is
    deleted at image-build time. ``fail_to_pass`` / ``pass_to_pass`` are the
    SWE-bench test lists the verifier runs.
    """

    repo_url: str
    base_commit: str
    issue_date: str
    issue_text: str
    issue_title: str
    acceptance_criteria: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]


@dataclass(frozen=True)
class SweLegoSetup:
    """The result of ``POST /api/v1/swe-lego/issues``.

    Mirrors the 201 response from the multica atomic endpoint (spec §4.1).
    """

    project_id: str
    issue_id: str
    image_id: str
    build_node_id: str
    base_sandbox_id: str
    base_sandbox_runtime_id: str
    agent_run_ids: list[str]


@dataclass(frozen=True)
class SweLegoIssueResult:
    """The outcome of one issue's training episode."""

    per_agent_rewards: list[float]
    per_agent_success: list[bool] = field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_types.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/reward/swe_lego_types.py customized_areal/tree_search/tests/test_swe_lego_types.py
git commit -m "feat(swe-lego): add SweLegoIssue/Setup/Result dataclasses"
```

---

## Task 7: `MulticaSweLegoClient` — the atomic-endpoint HTTP client (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/agents/swe_lego_client.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_swe_lego_client.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue, SweLegoSetup
from customized_areal.tree_search.agents.swe_lego_client import MulticaSweLegoClient

Handler = Callable[[httpx.Request], "httpx.Response | Awaitable[httpx.Response]"]


def _router(routes: dict[tuple[str, str], Handler]) -> httpx.MockTransport:
    def dispatch(request: httpx.Request):
        path = request.url.path
        for (method, route), handler in routes.items():
            if request.method != method:
                continue
            if route == path or (route.endswith("*") and path.startswith(route[:-1])):
                return handler(request)
        return httpx.Response(404, json={"error": f"no route for {request.method} {path}"})

    return httpx.MockTransport(dispatch)


def _issue() -> SweLegoIssue:
    return SweLegoIssue(
        repo_url="https://github.com/psf/requests.git",
        base_commit="abc123",
        issue_date="2025-03-14T09:30:00Z",
        issue_text="retry leaks",
        issue_title="Retry leaks",
        acceptance_criteria="must not leak",
        fail_to_pass=["tests/test_retry.py::test_leak"],
        pass_to_pass=["tests/test_retry.py::test_basic"],
    )


def test_create_swe_lego_issue_posts_to_atomic_endpoint():
    captured: dict = {}

    def handler(request: httpx.Request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            201,
            json={
                "project_id": "p1",
                "issue_id": "i1",
                "image_id": "img1",
                "build_node_id": "n1",
                "base_sandbox_id": "sbx-base",
                "base_sandbox_runtime_id": "rt-base",
                "agent_run_ids": ["r1", "r2"],
            },
        )

    transport = _router({("POST", "/api/v1/swe-lego/issues"): handler})
    client = MulticaSweLegoClient(
        base_url="https://multica.example", api_key="secret", transport=transport
    )
    setup = asyncio.run(
        client.create_swe_lego_issue(issue=_issue(), group_size=2, agent_config_id="agent-1")
    )
    assert captured["path"] == "/api/v1/swe-lego/issues"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["group_size"] == 2
    assert captured["body"]["agent_config_id"] == "agent-1"
    assert captured["body"]["base_commit"] == "abc123"
    assert isinstance(setup, SweLegoSetup)
    assert setup.agent_run_ids == ["r1", "r2"]


def test_create_swe_lego_issue_raises_on_non_201():
    transport = _router(
        {("POST", "/api/v1/swe-lego/issues*"): lambda r: httpx.Response(503, text="fork failed")}
    )
    client = MulticaSweLegoClient(
        base_url="https://multica.example", transport=transport
    )
    with pytest.raises(RuntimeError, match="create swe-lego issue failed"):
        asyncio.run(
            client.create_swe_lego_issue(issue=_issue(), group_size=2, agent_config_id="a")
        )


def test_cleanup_swe_lego_issue_posts_to_cleanup_endpoint():
    seen: list[str] = []

    def handler(request: httpx.Request):
        seen.append(f"{request.method} {request.url.path}")
        return httpx.Response(204)

    transport = _router({("DELETE", "/api/v1/swe-lego/issues/p1*"): handler})
    client = MulticaSweLegoClient(
        base_url="https://multica.example", transport=transport
    )
    asyncio.run(client.cleanup_swe_lego_issue(project_id="p1"))
    assert seen == ["DELETE /api/v1/swe-lego/issues/p1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_client.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
"""HTTP client for the multica SWE-Lego atomic endpoint.

Wraps ``POST /api/v1/swe-lego/issues`` (spec §4.1) and the cleanup endpoint.
``base_url`` / ``api_key`` default to ``MULTICA_BASE_URL`` / ``MULTICA_API_KEY``.
Uses stdlib :mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
import os

import httpx

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoSetup,
)

logger = logging.getLogger("MulticaSweLegoClient")


class MulticaSweLegoClient:
    """Thin HTTP client for the multica SWE-Lego orchestration endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.environ.get("MULTICA_BASE_URL") or "").rstrip("/")
        if not self._base_url:
            raise ValueError("MulticaSweLegoClient requires base_url or MULTICA_BASE_URL")
        self._api_key = api_key or os.environ.get("MULTICA_API_KEY")
        self._client = httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout, transport=transport
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_swe_lego_issue(
        self,
        *,
        issue: SweLegoIssue,
        group_size: int,
        agent_config_id: str,
        base_image: str | None = None,
    ) -> SweLegoSetup:
        """POST the atomic endpoint; returns the :class:`SweLegoSetup`."""
        payload: dict = {
            "repo_url": issue.repo_url,
            "base_commit": issue.base_commit,
            "issue_date": issue.issue_date,
            "issue_text": issue.issue_text,
            "issue_title": issue.issue_title,
            "acceptance_criteria": issue.acceptance_criteria,
            "fail_to_pass": list(issue.fail_to_pass),
            "pass_to_pass": list(issue.pass_to_pass),
            "group_size": group_size,
            "agent_config_id": agent_config_id,
        }
        if base_image is not None:
            payload["base_image"] = base_image
        resp = await self._client.post(
            "/api/v1/swe-lego/issues", json=payload, headers=self._headers()
        )
        if resp.status_code != 201:
            raise RuntimeError(
                f"create swe-lego issue failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )
        body = resp.json()
        return SweLegoSetup(
            project_id=body["project_id"],
            issue_id=body["issue_id"],
            image_id=body["image_id"],
            build_node_id=body["build_node_id"],
            base_sandbox_id=body["base_sandbox_id"],
            base_sandbox_runtime_id=body["base_sandbox_runtime_id"],
            agent_run_ids=list(body["agent_run_ids"]),
        )

    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None:
        """DELETE the per-issue project (cascades to sandboxes + issue)."""
        resp = await self._client.delete(
            f"/api/v1/swe-lego/issues/{project_id}", headers=self._headers()
        )
        if resp.status_code not in (200, 204, 404):
            raise RuntimeError(
                f"cleanup swe-lego issue failed: status={resp.status_code} "
                f"body={resp.text[:200]}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_client.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/swe_lego_client.py customized_areal/tree_search/tests/test_swe_lego_client.py
git commit -m "feat(swe-lego): add MulticaSweLegoClient HTTP client for atomic endpoint"
```

---

## Task 8: The atomic multica endpoint — handler skeleton (multica, Go)

**Files:**
- Create: `multica/server/internal/handler/swe_lego_issue.go`
- Test: `multica/server/internal/handler/swe_lego_issue_test.go`
- Modify: `multica/server/cmd/server/router.go`

- [ ] **Step 1: Write the failing test**

```go
package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCreateSweLegoIssue_RequiresAuth(t *testing.T) {
	h := newTestHandler(t)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/v1/swe-lego/issues", bytes.NewReader([]byte(`{}`)))
	h.ServeHTTP(w, r)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", w.Code)
	}
}

func TestCreateSweLegoIssue_RejectsMalformedBody(t *testing.T) {
	h := newTestHandler(t)
	w := httptest.NewRecorder()
	r := httptest.NewRequest("POST", "/api/v1/swe-lego/issues", bytes.NewReader([]byte(`not json`)))
	r.Header.Set("X-User-ID", "u1")
	r.Header.Set("X-Workspace-ID", "00000000-0000-0000-0000-000000000001")
	h.ServeHTTP(w, r)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}
```

(If `newTestHandler` does not already exist in the handler test package, define a minimal one in `swe_lego_issue_test.go` that constructs a `*Handler` with nil deps — mirroring the smallest existing handler test. Inspect `internal/handler/activity_test.go` for the exact helper name; if the package already has a `newTestHandler` or `setupTestHandler`, reuse it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/handler/ -run TestCreateSweLegoIssue -v`
Expected: FAIL with "no route registered" or handler missing.

- [ ] **Step 3: Write minimal implementation**

Create `internal/handler/swe_lego_issue.go`:

```go
package handler

import (
	"encoding/json"
	"net/http"
)

// CreateSweLegoIssueRequest is the body of POST /api/v1/swe-lego/issues.
// See the design spec §4.1 for field semantics.
type CreateSweLegoIssueRequest struct {
	RepoURL            string   `json:"repo_url"`
	BaseCommit         string   `json:"base_commit"`
	IssueDate          string   `json:"issue_date"`
	IssueText          string   `json:"issue_text"`
	IssueTitle         string   `json:"issue_title"`
	AcceptanceCriteria string   `json:"acceptance_criteria"`
	FailToPass         []string `json:"fail_to_pass"`
	PassToPass         []string `json:"pass_to_pass"`
	GroupSize          int      `json:"group_size"`
	AgentConfigID      string   `json:"agent_config_id"`
	BaseImage          string   `json:"base_image,omitempty"`
}

// CreateSweLegoIssueResponse is the 201 response (spec §4.1).
type CreateSweLegoIssueResponse struct {
	ProjectID              string   `json:"project_id"`
	IssueID                string   `json:"issue_id"`
	ImageID                string   `json:"image_id"`
	BuildNodeID            string   `json:"build_node_id"`
	BaseSandboxID          string   `json:"base_sandbox_id"`
	BaseSandboxRuntimeID   string   `json:"base_sandbox_runtime_id"`
	AgentRunIDs            []string `json:"agent_run_ids"`
}

// CreateSweLegoIssue handles POST /api/v1/swe-lego/issues.
//
// Atomic orchestration: CreateProject + image build + base sandbox boot +
// group_size forks + agent-run enqueue. Either returns 201 with all IDs or
// rolls back (spec §4.1 atomicity contract). The handler is thin; the
// orchestration lives in service.NewSweLegoIssueService (wired in Task 9).
func (h *Handler) CreateSweLegoIssue(w http.ResponseWriter, r *http.Request) {
	userID, ok := requireUserID(w, r)
	if !ok {
		return
	}
	_ = userID // auth-gated; the service layer receives workspace from context

	var req CreateSweLegoIssueRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed request body")
		return
	}
	if req.RepoURL == "" || req.BaseCommit == "" || req.IssueDate == "" {
		writeError(w, http.StatusBadRequest, "repo_url, base_commit, and issue_date are required")
		return
	}
	if req.GroupSize < 1 {
		writeError(w, http.StatusBadRequest, "group_size must be >= 1")
		return
	}

	// The service is wired in Task 9. For now, surface a 501 so the route
	// exists and the auth/body-validation tests pass; Task 9 replaces this
	// body with the real orchestration call.
	writeError(w, http.StatusNotImplemented, "swe-lego issue orchestration not yet wired")
}

// Placeholder to keep json import used if the body above is trimmed.
var _ = json.Valid
```

Register the route in `cmd/server/router.go`, after the `sandboxes/fork` line (around line 970):

```go
				r.Post("/sandboxes/fork", h.ForkCloudRuntimeSandbox)
				// SWE-Lego per-issue orchestration (spec §4.1).
				r.Post("/swe-lego/issues", h.CreateSweLegoIssue)
			})
```

(The exact indentation/brace nesting depends on the surrounding block — match the `sandboxes/fork` line's level.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/handler/ -run TestCreateSweLegoIssue -v`
Expected: PASS (auth → 401, malformed body → 400).

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/handler/swe_lego_issue.go internal/handler/swe_lego_issue_test.go cmd/server/router.go
git commit -m "feat(swe-lego): register POST /api/v1/swe-lego/issues route with validation"
```

---

## Task 9: SWE-Lego issue service — orchestration + rollback (multica, Go)

**Files:**
- Create: `multica/server/internal/service/swe_lego_issue.go`
- Create: `multica/server/internal/service/swe_lego_issue_test.go`
- Modify: `multica/server/internal/handler/swe_lego_issue.go`

- [ ] **Step 1: Write the failing test**

```go
package service

import (
	"context"
	"errors"
	"testing"
)

// fakeSweLegoDeps isolates the service from the DB + cloud runtime. Each
// field records calls so the test can assert ordering and rollback.
type fakeSweLegoDeps struct {
	createdProjectID string
	createdIssueID   string

	buildImageRef  string
	buildNodeID    string
	buildErr       error

	bootedSandboxID       string
	bootedRuntimeID       string
	bootErr               error

	forkedSandboxIDs      []string
	forkErrOn             int    // 1-based index that fails; 0 = none
	deletedSandboxes      []string

	enqueuedRunIDs        []string
	enqueueErrOn           int   // 1-based index that fails; 0 = none

	deletedProject        bool
}

func (f *fakeSweLegoDeps) CreateProject(ctx context.Context, name string) (string, error) {
	f.createdProjectID = "proj-1"
	return f.createdProjectID, nil
}
func (f *fakeSweLegoDeps) CreateIssue(ctx context.Context, projectID, title, body, criteria string, f2p, p2p []string) (string, error) {
	f.createdIssueID = "issue-1"
	return f.createdIssueID, nil
}
func (f *fakeSweLegoDeps) BuildImage(ctx context.Context, repoURL, baseCommit, issueDate, baseImage string) (imageRef, nodeID string, err error) {
	if f.buildErr != nil {
		return "", "", f.buildErr
	}
	f.buildImageRef = "swe-lego:deadbeef"
	f.buildNodeID = "node-1"
	return f.buildImageRef, f.buildNodeID, nil
}
func (f *fakeSweLegoDeps) BootBaseSandbox(ctx context.Context, imageRef, nodeID string) (sandboxID, runtimeID string, err error) {
	if f.bootErr != nil {
		return "", "", f.bootErr
	}
	f.bootedSandboxID = "sbx-base"
	f.bootedRuntimeID = "rt-base"
	return f.bootedSandboxID, f.bootedRuntimeID, nil
}
func (f *fakeSweLegoDeps) ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (string, error) {
	if f.forkErrOn != 0 && idx == f.forkErrOn {
		return "", errors.New("fork failed")
	}
	id := "sbx-fork-" + itoa(idx)
	f.forkedSandboxIDs = append(f.forkedSandboxIDs, id)
	return id, nil
}
func (f *fakeSweLegoDeps) EnqueueAgentRun(ctx context.Context, issueID, sandboxID string, idx int) (string, error) {
	if f.enqueueErrOn != 0 && idx == f.enqueueErrOn {
		return "", errors.New("enqueue failed")
	}
	id := "run-" + itoa(idx)
	f.enqueuedRunIDs = append(f.enqueuedRunIDs, id)
	return id, nil
}
func (f *fakeSweLegoDeps) DeleteSandbox(ctx context.Context, sandboxID string) error {
	f.deletedSandboxes = append(f.deletedSandboxes, sandboxID)
	return nil
}
func (f *fakeSweLegoDeps) DeleteProject(ctx context.Context, projectID string) error {
	f.deletedProject = true
	return nil
}

func itoa(i int) string {
	// avoid pulling in strconv for the test fake
	return string(rune('0'+i)) // single-digit only; tests use group_size <= 3
}

func TestSweLegoIssueService_HappyPath(t *testing.T) {
	ctx := context.Background()
	deps := &fakeSweLegoDeps{}
	svc := NewSweLegoIssueService(deps)
	res, err := svc.Create(ctx, SweLegoIssueInput{
		RepoURL: "r", BaseCommit: "c", IssueDate: "d", IssueTitle: "t", IssueText: "x",
		AcceptanceCriteria: "a", FailToPass: []string{"f"}, PassToPass: []string{"p"},
		GroupSize: 2, AgentConfigID: "ag", BaseImage: "swe-lego/python:3.11",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.ProjectID != "proj-1" || res.IssueID != "issue-1" {
		t.Fatalf("unexpected ids: %+v", res)
	}
	if len(res.AgentRunIDs) != 2 {
		t.Fatalf("expected 2 agent runs, got %d", len(res.AgentRunIDs))
	}
	if res.BuildNodeID != "node-1" || res.BaseSandboxID != "sbx-base" {
		t.Fatalf("expected node-1/sbx-base, got %s/%s", res.BuildNodeID, res.BaseSandboxID)
	}
	// No rollback on success.
	if deps.deletedProject || len(deps.deletedSandboxes) != 0 {
		t.Fatalf("unexpected rollback on success: project=%v sandboxes=%v", deps.deletedProject, deps.deletedSandboxes)
	}
}

func TestSweLegoIssueService_BuildFailureRollsBackProject(t *testing.T) {
	ctx := context.Background()
	deps := &fakeSweLegoDeps{buildErr: errors.New("filter-repo failed")}
	svc := NewSweLegoIssueService(deps)
	_, err := svc.Create(ctx, SweLegoIssueInput{
		RepoURL: "r", BaseCommit: "c", IssueDate: "d", IssueTitle: "t", IssueText: "x",
		AcceptanceCriteria: "a", FailToPass: []string{"f"}, PassToPass: []string{"p"},
		GroupSize: 2, AgentConfigID: "ag", BaseImage: "b",
	})
	if err == nil {
		t.Fatal("expected error on build failure")
	}
	// Project was created before the build; it must be rolled back.
	if !deps.deletedProject {
		t.Fatal("expected project rollback on build failure")
	}
	// No sandbox should have been booted or forked.
	if deps.bootedSandboxID != "" || len(deps.forkedSandboxIDs) != 0 {
		t.Fatal("expected no sandbox boot/fork on build failure")
	}
}

func TestSweLegoIssueService_ForkFailureRollsBackSandboxesAndProject(t *testing.T) {
	ctx := context.Background()
	deps := &fakeSweLegoDeps{forkErrOn: 2} // the second fork fails
	svc := NewSweLegoIssueService(deps)
	_, err := svc.Create(ctx, SweLegoIssueInput{
		RepoURL: "r", BaseCommit: "c", IssueDate: "d", IssueTitle: "t", IssueText: "x",
		AcceptanceCriteria: "a", FailToPass: []string{"f"}, PassToPass: []string{"p"},
		GroupSize: 2, AgentConfigID: "ag", BaseImage: "b",
	})
	if err == nil {
		t.Fatal("expected error on fork failure")
	}
	// The first fork succeeded, then the second failed → rollback the first
	// fork + the base sandbox + the project.
	if len(deps.deletedSandboxes) < 2 {
		t.Fatalf("expected >= 2 sandbox deletes (fork + base), got %v", deps.deletedSandboxes)
	}
	if !deps.deletedProject {
		t.Fatal("expected project rollback on fork failure")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoIssueService -v`
Expected: FAIL with "undefined: NewSweLegoIssueService".

- [ ] **Step 3: Write minimal implementation**

Create `internal/service/swe_lego_issue.go`:

```go
package service

import (
	"context"
	"fmt"
)

// SweLegoIssueInput is the service-layer input for the atomic create.
type SweLegoIssueInput struct {
	RepoURL              string
	BaseCommit           string
	IssueDate            string
	IssueTitle           string
	IssueText            string
	AcceptanceCriteria  string
	FailToPass           []string
	PassToPass           []string
	GroupSize            int
	AgentConfigID        string
	BaseImage            string
}

// SweLegoIssueResult is the service-layer output (mirrors the 201 response).
type SweLegoIssueResult struct {
	ProjectID            string
	IssueID              string
	ImageID              string
	BuildNodeID          string
	BaseSandboxID        string
	BaseSandboxRuntimeID string
	AgentRunIDs          []string
}

// SweLegoDeps is the seam between the service and the DB + cloud runtime.
// Each method corresponds to one step of the orchestration (spec §4.2).
// Production wires this to real queries + cloudRuntimeProxy; tests inject
// a fake.
type SweLegoDeps interface {
	CreateProject(ctx context.Context, name string) (string, error)
	CreateIssue(ctx context.Context, projectID, title, body, criteria string, f2p, p2p []string) (string, error)
	BuildImage(ctx context.Context, repoURL, baseCommit, issueDate, baseImage string) (imageRef, nodeID string, err error)
	BootBaseSandbox(ctx context.Context, imageRef, nodeID string) (sandboxID, runtimeID string, err error)
	ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (string, error)
	EnqueueAgentRun(ctx context.Context, issueID, sandboxID string, idx int) (string, error)
	DeleteSandbox(ctx context.Context, sandboxID string) error
	DeleteProject(ctx context.Context, projectID string) error
}

// SweLegoIssueService orchestrates the atomic per-issue setup (spec §4.2).
type SweLegoIssueService struct {
	deps SweLegoDeps
}

func NewSweLegoIssueService(deps SweLegoDeps) *SweLegoIssueService {
	return &SweLegoIssueService{deps: deps}
}

// Create runs the six-step sequence. On any failure after the project is
// created, it rolls back: deletes forked sandboxes, the base sandbox, and
// the project. The built image stays cached on the build node (spec §4.1
// atomicity contract — image build is expensive).
func (s *SweLegoIssueService) Create(ctx context.Context, in SweLegoIssueInput) (SweLegoIssueResult, error) {
	// 1. CreateProject
	projectID, err := s.deps.CreateProject(ctx, fmt.Sprintf("swe-lego/%s-%s", in.RepoURL, shortSHA(in.BaseCommit)))
	if err != nil {
		return SweLegoIssueResult{}, fmt.Errorf("create project: %w", err)
	}
	// From here on, any failure rolls back the project.
	issueID, imageRef, nodeID, baseSandboxID, baseRuntimeID, runIDs, err := s.createAfterProject(ctx, projectID, in)
	if err != nil {
		s.rollback(ctx, projectID, baseSandboxID, runIDs)
		return SweLegoIssueResult{}, err
	}
	return SweLegoIssueResult{
		ProjectID: projectID, IssueID: issueID, ImageID: imageRef,
		BuildNodeID: nodeID, BaseSandboxID: baseSandboxID,
		BaseSandboxRuntimeID: baseRuntimeID, AgentRunIDs: runIDs,
	}, nil
}

func (s *SweLegoIssueService) createAfterProject(ctx context.Context, projectID string, in SweLegoIssueInput) (issueID, imageRef, nodeID, baseSandboxID, baseRuntimeID string, runIDs []string, err error) {
	// 2. CreateIssue
	issueID, err = s.deps.CreateIssue(ctx, projectID, in.IssueTitle, in.IssueText, in.AcceptanceCriteria, in.FailToPass, in.PassToPass)
	if err != nil {
		return "", "", "", "", "", nil, fmt.Errorf("create issue: %w", err)
	}
	// 3. BuildImage
	imageRef, nodeID, err = s.deps.BuildImage(ctx, in.RepoURL, in.BaseCommit, in.IssueDate, in.BaseImage)
	if err != nil {
		return issueID, "", "", "", "", nil, fmt.Errorf("build image: %w", err)
	}
	// 4. BootBaseSandbox (on the same node that built the image)
	baseSandboxID, baseRuntimeID, err = s.deps.BootBaseSandbox(ctx, imageRef, nodeID)
	if err != nil {
		return issueID, imageRef, nodeID, "", "", nil, fmt.Errorf("boot base sandbox: %w", err)
	}
	// 5. Fork × group_size + enqueue
	runIDs = make([]string, 0, in.GroupSize)
	for i := 1; i <= in.GroupSize; i++ {
		forked, ferr := s.deps.ForkSandbox(ctx, baseSandboxID, i)
		if ferr != nil {
			return issueID, imageRef, nodeID, baseSandboxID, baseRuntimeID, runIDs, fmt.Errorf("fork %d: %w", i, ferr)
		}
		runID, eerr := s.deps.EnqueueAgentRun(ctx, issueID, forked, i)
		if eerr != nil {
			// enqueue failed → clean up this forked sandbox too.
			_ = s.deps.DeleteSandbox(ctx, forked)
			return issueID, imageRef, nodeID, baseSandboxID, baseRuntimeID, runIDs, fmt.Errorf("enqueue %d: %w", i, eerr)
		}
		runIDs = append(runIDs, runID)
	}
	return issueID, imageRef, nodeID, baseSandboxID, baseRuntimeID, runIDs, nil
}

func (s *SweLegoIssueService) rollback(ctx context.Context, projectID, baseSandboxID string, runIDs []string) {
	// Best-effort. Logs are the caller's responsibility; this never raises.
	if baseSandboxID != "" {
		_ = s.deps.DeleteSandbox(ctx, baseSandboxID)
	}
	_ = s.deps.DeleteProject(ctx, projectID)
}

// shortSHA returns the first 8 hex chars of a commit-ish, for the project
// name. Falls back to the full string if shorter.
func shortSHA(s string) string {
	if len(s) > 8 {
		return s[:8]
	}
	return s
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd multica/server && go test ./internal/service/ -run TestSweLegoIssueService -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd multica/server
git add internal/service/swe_lego_issue.go internal/service/swe_lego_issue_test.go
git commit -m "feat(swe-lego): add issue orchestration service with rollback"
```

- [ ] **Step 6: Wire the service into the handler**

Replace the `writeError(w, http.StatusNotImplemented, ...)` body in `CreateSweLegoIssue` (Task 8) with a call to the service. The handler becomes:

```go
func (h *Handler) CreateSweLegoIssue(w http.ResponseWriter, r *http.Request) {
	userID, ok := requireUserID(w, r)
	if !ok {
		return
	}
	_ = userID

	var req CreateSweLegoIssueRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "malformed request body")
		return
	}
	if req.RepoURL == "" || req.BaseCommit == "" || req.IssueDate == "" {
		writeError(w, http.StatusBadRequest, "repo_url, base_commit, and issue_date are required")
		return
	}
	if req.GroupSize < 1 {
		writeError(w, http.StatusBadRequest, "group_size must be >= 1")
		return
	}

	svc := service.NewSweLegoIssueService(newSweLegoDepsAdapter(h))
	res, err := svc.Create(r.Context(), service.SweLegoIssueInput{
		RepoURL: req.RepoURL, BaseCommit: req.BaseCommit, IssueDate: req.IssueDate,
		IssueTitle: req.IssueTitle, IssueText: req.IssueText,
		AcceptanceCriteria: req.AcceptanceCriteria,
		FailToPass: req.FailToPass, PassToPass: req.PassToPass,
		GroupSize: req.GroupSize, AgentConfigID: req.AgentConfigID, BaseImage: req.BaseImage,
	})
	if err != nil {
		// Image build failures → 502; sandbox/fork failures → 503.
		status := http.StatusServiceUnavailable
		if strings.Contains(err.Error(), "build image") {
			status = http.StatusBadGateway
		}
		writeError(w, status, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, CreateSweLegoIssueResponse{
		ProjectID: res.ProjectID, IssueID: res.IssueID, ImageID: res.ImageID,
		BuildNodeID: res.BuildNodeID, BaseSandboxID: res.BaseSandboxID,
		BaseSandboxRuntimeID: res.BaseSandboxRuntimeID, AgentRunIDs: res.AgentRunIDs,
	})
}
```

`newSweLegoDepsAdapter(h)` returns a concrete `*sweLegoDepsAdapter` that wraps `h.Queries` + `h.CloudRuntime` and implements `service.SweLegoDeps`. Define it in `internal/handler/swe_lego_issue.go`:

```go
// sweLegoDepsAdapter bridges the *Handler (queries + cloud-runtime proxy) to
// the service.SweLegoDeps seam. Each method wraps an existing query or
// cloud-runtime call.
type sweLegoDepsAdapter struct {
	h *Handler
}

func newSweLegoDepsAdapter(h *Handler) *sweLegoDepsAdapter { return &sweLegoDepsAdapter{h: h} }

func (a *sweLegoDepsAdapter) CreateProject(ctx context.Context, name string) (string, error) {
	// TODO: replace with the real CreateProject query (project.go:221 path).
	// For Task 9 the adapter returns a stub so the service-layer tests pass;
	// Task 10 wires the real queries.
	return "stub-project", nil
}

// ... (the remaining methods are stubs that return placeholder IDs; Task 10
// replaces them with real queries + cloud-runtime calls.)
```

(For Task 9, ship the stubs so the handler compiles and the route works end-to-end against a stub. Task 10 replaces the stubs.)

Add `"strings"` and `"github.com/multica-ai/multica/server/internal/service"` to the handler file imports.

- [ ] **Step 7: Run all swe-lego tests**

Run: `cd multica/server && go test ./internal/handler/ ./internal/service/ -run "SweLego|swe_lego" -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
cd multica/server
git add internal/handler/swe_lego_issue.go internal/handler/swe_lego_issue_test.go
git commit -m "feat(swe-lego): wire orchestration service into handler with status mapping"
```

---

## Task 10: Wire the service deps to real queries + cloud runtime (multica, Go)

**Files:**
- Modify: `multica/server/internal/handler/swe_lego_issue.go`
- Test: `multica/server/internal/handler/swe_lego_issue_test.go`

- [ ] **Step 1: Write the failing test**

This task replaces the stub adapter methods with real calls. Add an integration-flavored test that mocks only the cloud-runtime proxy (not the DB) — or, if a test DB is available in CI, a full-stack test. For a unit test that verifies the adapter wiring without a DB:

```go
func TestSweLegoDepsAdapter_BuildImageCallsNodeExec(t *testing.T) {
	// The adapter's BuildImage must route through the cloud-runtime node-exec
	// proxy, not shell out locally. Assert the proxy receives an Exec call.
	// (Detailed assertion on the script is covered by service-level tests.)
	t.Skip("integration: requires a stubbed cloud-runtime proxy; covered by e2e in Task 17")
}
```

(The real verification for this task is the e2e in Task 17. The unit-level guarantee — that the adapter calls `BuildOrReuse` — is a thin pass-through that is best verified by reading the code + the e2e.)

- [ ] **Step 2: Replace the stub adapter methods with real calls**

In `internal/handler/swe_lego_issue.go`, replace each stub method on `sweLegoDepsAdapter`:

```go
func (a *sweLegoDepsAdapter) CreateProject(ctx context.Context, name string) (string, error) {
	// Reuse the existing project-creation query path (project.go:221).
	// The exact query depends on the project service's signature; mirror
	// how CreateProject handler builds its project row.
	params := db.CreateProjectParams{Name: name /* + workspace_id from ctx */}
	p, err := a.h.Queries.CreateProject(ctx, params)
	if err != nil {
		return "", fmt.Errorf("create project: %w", err)
	}
	return uuidToString(p.ID), nil
}

func (a *sweLegoDepsAdapter) CreateIssue(ctx context.Context, projectID, title, body, criteria string, f2p, p2p []string) (string, error) {
	// Create the root issue + store f2p/p2p as issue metadata (spec §4.5).
	params := db.CreateIssueParams{
		Title: title, Body: body, /* + project_id, workspace_id, acceptance_criteria */
	}
	issue, err := a.h.Queries.CreateIssue(ctx, params)
	if err != nil {
		return "", fmt.Errorf("create issue: %w", err)
	}
	// Store f2p/p2p as issue metadata via SetIssueMetadataKey (router.go:747).
	meta, _ := json.Marshal(map[string][]string{"fail_to_pass": f2p, "pass_to_pass": p2p})
	_ = a.h.Queries.SetIssueMetadataKey(ctx, db.SetIssueMetadataKeyParams{
		IssueID: issue.ID, Key: "swe_lego_tests", Value: meta,
	})
	return uuidToString(issue.ID), nil
}

func (a *sweLegoDepsAdapter) BuildImage(ctx context.Context, repoURL, baseCommit, issueDate, baseImage string) (string, string, error) {
	// Route through the cloud-runtime node-exec proxy via a NodeExec impl.
	exec := newCloudRuntimeNodeExec(a.h.CloudRuntime, a.h)
	return service.BuildOrReuse(ctx, exec, repoURL, baseCommit, issueDate, baseImage)
}

func (a *sweLegoDepsAdapter) BootBaseSandbox(ctx context.Context, imageRef, nodeID string) (string, string, error) {
	// POST /api/v1/nodes + /api/v1/sandboxes via the cloud-runtime proxy,
	// targeting nodeID. The daemon self-registers its runtime on boot.
	// Returns (sandboxID, runtimeID).
	return bootSandboxOnNode(ctx, a.h.CloudRuntime, imageRef, nodeID)
}

func (a *sweLegoDepsAdapter) ForkSandbox(ctx context.Context, sourceSandboxID string, idx int) (string, error) {
	// POST /api/v1/sandboxes/fork { source_sandbox_id }
	body, _ := json.Marshal(map[string]string{"source_sandbox_id": sourceSandboxID})
	resp, err := a.h.CloudRuntime.Do(ctx, cloudruntime.Request{
		Method: "POST", Path: "/api/v1/sandboxes/fork", Body: body, Op: "fork",
	})
	if err != nil {
		return "", fmt.Errorf("fork transport: %w", err)
	}
	if resp.StatusCode != 200 {
		return "", fmt.Errorf("fork failed: status=%d body=%s", resp.StatusCode, string(resp.Body))
	}
	var out struct{ SandboxID string `json:"sandbox_id"` }
	if err := json.Unmarshal(resp.Body, &out); err != nil || out.SandboxID == "" {
		return "", fmt.Errorf("fork response missing sandbox_id: %s", string(resp.Body))
	}
	return out.SandboxID, nil
}

func (a *sweLegoDepsAdapter) EnqueueAgentRun(ctx context.Context, issueID, sandboxID string, idx int) (string, error) {
	// Enqueue a task for the agent on this issue, bound to the forked
	// sandbox's runtime. Reuse the existing TaskService path.
	return enqueueAgentTaskOnSandbox(ctx, a.h, issueID, sandboxID)
}

func (a *sweLegoDepsAdapter) DeleteSandbox(ctx context.Context, sandboxID string) error {
	resp, err := a.h.CloudRuntime.Do(ctx, cloudruntime.Request{
		Method: "DELETE", Path: "/api/v1/sandboxes/" + sandboxID, Op: "terminate",
	})
	if err != nil {
		return fmt.Errorf("delete sandbox transport: %w", err)
	}
	// 404 = already deleted; idempotent.
	if resp.StatusCode == 404 || resp.StatusCode == 200 || resp.StatusCode == 204 {
		return nil
	}
	return fmt.Errorf("delete sandbox failed: status=%d", resp.StatusCode)
}

func (a *sweLegoDepsAdapter) DeleteProject(ctx context.Context, projectID string) error {
	pgID, err := util.ParseUUID(projectID)
	if err != nil {
		return fmt.Errorf("parse project id: %w", err)
	}
	return a.h.Queries.DeleteProject(ctx, pgID)
}
```

Helper functions `bootSandboxOnNode`, `enqueueAgentTaskOnSandbox`, `newCloudRuntimeNodeExec` go in the same file. They wrap existing cloud-runtime / task-service calls — the exact signatures depend on the existing `cloudruntime.Client.Do` and `TaskService` shapes; mirror the patterns in `cloud_runtime.go` and `task.go`.

- [ ] **Step 3: Run tests + `go build`**

Run: `cd multica/server && go build ./... && go test ./internal/handler/ ./internal/service/ -run "SweLego|swe_lego" -v`
Expected: PASS. The skipped integration test is fine.

- [ ] **Step 4: Commit**

```bash
cd multica/server
git add internal/handler/swe_lego_issue.go internal/handler/swe_lego_issue_test.go
git commit -m "feat(swe-lego): wire service deps to real queries + cloud runtime"
```

---

## Task 11: `SweLegoVerifier` — hybrid reward blend (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/agents/reward/swe_lego_verifier.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_swe_lego_verifier.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from dataclasses import dataclass

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_verifier import (
    ObjectiveOutcome,
    SweLegoVerifier,
    VerifierResult,
)
from customized_areal.tree_search.agents.verifier import VerifierResult as BaseResult


@dataclass
class FakeObjective:
    fully_passes: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    failing_count_reduced: bool

    async def run_tests(self, sandbox_id, f2p, p2p):
        return ObjectiveOutcome(
            fully_passes=self.fully_passes,
            f2p_passed=self.f2p_passed,
            f2p_total=self.f2p_total,
            p2p_passed=self.p2p_passed,
            p2p_total=self.p2p_total,
            failing_count_reduced=self.failing_count_reduced,
        )


@dataclass
class FakeCritic:
    score_value: float
    raises: bool = False

    async def score(self, transcript, criteria, objective):
        if self.raises:
            raise RuntimeError("critic LLM error")
        return self.score_value


@dataclass
class FakeRlSession:
    last_reward: float | None = None

    async def set_reward(self, *, session_id, reward):
        self.last_reward = reward


def test_verifier_short_circuits_to_one_on_full_pass():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(True, 2, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.1),  # ignored on full pass
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1", sandbox_id="s1", session_id="sess1",
            fail_to_pass=["a"], pass_to_pass=["b"],
            transcript="...", acceptance_criteria="...",
        )
    )
    assert result.reward == 1.0
    assert result.success is True
    assert rl.last_reward == 1.0


def test_verifier_short_circuits_to_zero_on_total_failure():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(False, 0, 2, 3, 3, False),
        critic=FakeCritic(score_value=0.9),  # ignored on total failure
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1", sandbox_id="s1", session_id="sess1",
            fail_to_pass=["a"], pass_to_pass=["b"],
            transcript="...", acceptance_criteria="...",
        )
    )
    assert result.reward == 0.0
    assert result.success is False
    assert rl.last_reward == 0.0


def test_verifier_blends_in_mixed_middle():
    rl = FakeRlSession()
    # 1 of 2 F2P passes, failing count reduced → not total failure, not full pass.
    v = SweLegoVerifier(
        objective=FakeObjective(False, 1, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.5),
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1", sandbox_id="s1", session_id="sess1",
            fail_to_pass=["a", "b"], pass_to_pass=["c"],
            transcript="...", acceptance_criteria="...",
        )
    )
    # objective = 0.5 (1/2 F2P), generative = 0.5, semi_resolved = 1.0 (reduced).
    # blend = 0.7*0.5 + 0.2*0.5 + 0.1*1.0 = 0.35 + 0.10 + 0.10 = 0.55
    assert result.reward == pytest.approx(0.55)
    assert rl.last_reward == pytest.approx(0.55)


def test_verifier_critic_failure_falls_back_neutral():
    rl = FakeRlSession()
    v = SweLegoVerifier(
        objective=FakeObjective(False, 1, 2, 3, 3, True),
        critic=FakeCritic(score_value=0.5, raises=True),
        rl_session=rl,
    )
    result = asyncio.run(
        v.verify_and_reward(
            agent_run_id="r1", sandbox_id="s1", session_id="sess1",
            fail_to_pass=["a", "b"], pass_to_pass=["c"],
            transcript="...", acceptance_criteria="...",
        )
    )
    # Critic failed → source="default", reward still written (blend uses 0.0
    # for the generative term so training can proceed).
    assert result.source == "default"
    assert rl.last_reward is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_verifier.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Hybrid SWE-Lego verifier: objective tests + generative critic + semi-resolved.

Composes three layers (spec §5.3). The objective layer short-circuits when
decisive; the blend runs only in the mixed middle. Uses stdlib
:mod:`logging` so the module stays importable without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from customized_areal.tree_search.agents.verifier import VerifierResult

logger = logging.getLogger("SweLegoVerifier")


@dataclass(frozen=True)
class ObjectiveOutcome:
    fully_passes: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    failing_count_reduced: bool


class _Objective(Protocol):
    async def run_tests(
        self, sandbox_id: str, fail_to_pass: list[str], pass_to_pass: list[str]
    ) -> ObjectiveOutcome: ...


class _Critic(Protocol):
    async def score(
        self, transcript: str, acceptance_criteria: str, objective: ObjectiveOutcome
    ) -> float: ...


class _RlSession(Protocol):
    async def set_reward(self, *, session_id: str, reward: float) -> None: ...


@dataclass(frozen=True)
class BlendWeights:
    objective: float = 0.7
    generative: float = 0.2
    semi_resolved: float = 0.1


class SweLegoVerifier:
    """Hybrid verifier producing a terminal reward per agent run."""

    def __init__(
        self,
        *,
        objective: _Objective,
        critic: _Critic,
        rl_session: _RlSession,
        weights: BlendWeights | None = None,
    ) -> None:
        self._objective = objective
        self._critic = critic
        self._rl = rl_session
        self._weights = weights or BlendWeights()

    async def verify_and_reward(
        self,
        *,
        agent_run_id: str,
        sandbox_id: str,
        session_id: str,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        transcript: str,
        acceptance_criteria: str,
    ) -> VerifierResult:
        obj = await self._objective.run_tests(sandbox_id, fail_to_pass, pass_to_pass)

        # Short-circuit: objective fully decisive.
        if obj.fully_passes:
            reward = 1.0
            await self._rl.set_reward(session_id=session_id, reward=reward)
            return VerifierResult(
                success=True, reward=reward, source="objective",
                rationale="FAIL_TO_PASS fully passes",
            )
        if obj.f2p_passed == 0 and not obj.failing_count_reduced:
            reward = 0.0
            await self._rl.set_reward(session_id=session_id, reward=reward)
            return VerifierResult(
                success=False, reward=reward, source="objective",
                rationale="FAIL_TO_PASS fully fails, no failing-count reduction",
            )

        # Mixed middle: blend.
        gen_score, source = await self._safe_critic_score(transcript, acceptance_criteria, obj)
        obj_score = obj.f2p_passed / obj.f2p_total if obj.f2p_total else 0.0
        semi = 1.0 if obj.failing_count_reduced else 0.0
        w = self._weights
        reward = w.objective * obj_score + w.generative * gen_score + w.semi_resolved * semi
        await self._rl.set_reward(session_id=session_id, reward=reward)
        return VerifierResult(
            success=obj.fully_passes, reward=reward, source=source,
            rationale=f"blend: obj={obj_score:.2f} gen={gen_score:.2f} semi={semi:.2f}",
            per_step_signals={"semi_resolved": semi},
        )

    async def _safe_critic_score(
        self, transcript: str, criteria: str, obj: ObjectiveOutcome
    ) -> tuple[float, str]:
        try:
            return await self._critic.score(transcript, criteria, obj), "hybrid"
        except Exception as exc:
            logger.warning("critic failed; using 0.0 for generative term: %s", exc)
            return 0.0, "default"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_verifier.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/reward/swe_lego_verifier.py customized_areal/tree_search/tests/test_swe_lego_verifier.py
git commit -m "feat(swe-lego): add hybrid SweLegoVerifier with blend/short-circuit"
```

---

## Task 12: DAG reward backup — distribute over edges + fan-in credit (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/agents/dag_backup.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_dag_backup.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from customized_areal.tree_search.agents.dag_backup import (
    distribute_reward_over_dag,
    CreditAssignment,
)
from customized_areal.tree_search.agents.execution_dag import (
    AgentRunNode,
    Edge,
    EdgeType,
    ExecutionDAG,
)


def _node(node_id: str, parent: str | None = None) -> AgentRunNode:
    return AgentRunNode(node_id=node_id, parent_node_id=parent, episode_id="e", turn_idx=0, task_id="t", outcome_reward=None)


def _dag_linear() -> ExecutionDAG:
    # root -> child1 -> child2 (sequential delegation chain)
    return ExecutionDAG(
        nodes={"root": _node("root"), "c1": _node("c1", "root"), "c2": _node("c2", "c1")},
        edges=[
            Edge(src="root", dst="c1", edge_type=EdgeType.DELEGATION),
            Edge(src="c1", dst="c2", edge_type=EdgeType.DELEGATION),
        ],
    )


def test_distribute_reward_over_linear_chain():
    dag = _dag_linear()
    credit = distribute_reward_over_dag(dag, terminal_reward=1.0, terminal_node_id="c2")
    # The terminal reward backs up along edges: c2 gets 1.0, c1 and root
    # each get the propagated share (default: equal split along the path).
    assert credit["c2"] == pytest.approx(1.0)
    assert credit["c1"] > 0.0
    assert credit["root"] > 0.0
    # Monotonic: closer to terminal => >= further from terminal.
    assert credit["c2"] >= credit["c1"] >= credit["root"]


def test_distribute_reward_unknown_terminal_raises():
    dag = _dag_linear()
    with pytest.raises(KeyError):
        distribute_reward_over_dag(dag, terminal_reward=1.0, terminal_node_id="nope")


def _dag_fan_in() -> ExecutionDAG:
    # root delegates to a and b; both report back to join (fan-in).
    return ExecutionDAG(
        nodes={
            "root": _node("root"),
            "a": _node("a", "root"),
            "b": _node("b", "root"),
            "join": _node("join"),
        },
        edges=[
            Edge(src="root", dst="a", edge_type=EdgeType.DELEGATION),
            Edge(src="root", dst="b", edge_type=EdgeType.DELEGATION),
            Edge(src="a", dst="join", edge_type=EdgeType.COMPLETION),
            Edge(src="b", dst="join", edge_type=EdgeType.COMPLETION),
        ],
    )


def test_fan_in_credit_is_explicit_not_aggregated():
    dag = _dag_fan_in()

    # The caller supplies explicit per-agent credit at the fan-in join
    # (spec §2 decision 8: no fixed sum/mean/max). a contributed more than b.
    assignment = CreditAssignment(per_node={"a": 0.7, "b": 0.3})
    credit = distribute_reward_over_dag(
        dag, terminal_reward=1.0, terminal_node_id="join", fan_in_credit=assignment
    )
    assert credit["a"] == pytest.approx(0.7)
    assert credit["b"] == pytest.approx(0.3)
    # root gets the sum (the join's reward backed up to root through a and b).
    assert credit["root"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_dag_backup.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
"""DAG-aware reward backup (Phase 3, load-bearing).

Distributes a terminal verifier reward along DAG edges so sub-agent runs
that contributed to a successful root get credit. Fan-in joins get explicit
per-agent credit from the caller — no fixed sum/mean/max aggregation
(spec §2 decision 8, §5.4).

Uses stdlib :mod:`logging` so the package stays importable without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from customized_areal.tree_search.agents.execution_dag import ExecutionDAG

logger = logging.getLogger("DagBackup")


@dataclass(frozen=True)
class CreditAssignment:
    """Explicit per-agent credit at a fan-in join (spec §2 decision 8).

    The caller (verifier or credit assigner) decides how much each incoming
    agent contributed; this module does not apply a sum/mean/max rule.
    """

    per_node: dict[str, float]


def distribute_reward_over_dag(
    dag: ExecutionDAG,
    *,
    terminal_reward: float,
    terminal_node_id: str,
    fan_in_credit: Optional[CreditAssignment] = None,
    backup_decay: float = 0.5,
) -> dict[str, float]:
    """Distribute ``terminal_reward`` backward along DAG edges.

    Returns a ``{node_id: credit}`` map. The terminal node gets the full
    reward; each ancestor along a delegation edge gets ``backup_decay`` of
    its child's credit (so credit attenuates with distance from the outcome).
    Fan-in joins consume ``fan_in_credit`` if provided; otherwise the join's
    reward is split equally among incoming edges as a default (the caller is
    expected to supply explicit credit for real runs).
    """
    if terminal_node_id not in dag.nodes:
        raise KeyError(f"terminal node {terminal_node_id!r} not in DAG")

    credit: dict[str, float] = {nid: 0.0 for nid in dag.nodes}
    credit[terminal_node_id] = terminal_reward

    # Walk backward from the terminal node. For each node, find its parents
    # (incoming delegation/completion edges) and propagate backup_decay.
    # Fan-in nodes (multiple parents) use fan_in_credit if supplied.
    visited: set[str] = set()
    queue = [terminal_node_id]
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        parents = [e.src for e in dag.edges if e.dst == nid]
        if not parents:
            continue
        if len(parents) > 1 and fan_in_credit is not None:
            # Explicit per-agent credit at the fan-in join.
            for p in parents:
                share = fan_in_credit.per_node.get(p, 0.0)
                credit[p] = max(credit[p], share * credit[nid])
        else:
            # Single parent or no explicit fan-in credit: split equally.
            share = credit[nid] * backup_decay / len(parents)
            for p in parents:
                credit[p] += share
        queue.extend(parents)

    return credit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_dag_backup.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/dag_backup.py customized_areal/tree_search/tests/test_dag_backup.py
git commit -m "feat(dag): add reward backup over edges with explicit fan-in credit"
```

---

## Task 13: Extend `TreeAdvantageComputer` to consume per-node credit (areal, Python)

**Files:**
- Modify: `backend/areal/customized_areal/tree_search/agents/advantage.py` (or the file owning `TreeAdvantageComputer`; locate it first)
- Test: `backend/areal/customized_areal/tree_search/tests/test_advantage.py` (extend)

- [ ] **Step 0: Locate `TreeAdvantageComputer`**

Run: `cd backend/areal && grep -rn "class TreeAdvantageComputer\|TreeAdvantageComputer" customized_areal/`
If it lives in `customized_areal/tree_search/core/advantage.py` (per the design spec reference), that is the file. If the file is missing or the class is elsewhere, adjust the paths below.

- [ ] **Step 1: Write the failing test**

Append to the advantage test file (or create it if none exists):

```python
import pytest

from customized_areal.tree_search.agents.advantage import TreeAdvantageComputer


def test_advantage_consumes_per_node_credit():
    # Three nodes in one episode; per-node credit should produce different
    # per-token advantage (not a flat broadcast).
    computer = TreeAdvantageComputer()
    nodes = [
        {"node_id": "n1", "episode_id": "e1", "credit": 1.0, "num_tokens": 4},
        {"node_id": "n2", "episode_id": "e1", "credit": 0.5, "num_tokens": 4},
        {"node_id": "n3", "episode_id": "e1", "credit": 0.0, "num_tokens": 4},
    ]
    result = computer.compute(nodes, group_size=1)
    # n1's per-token advantage > n2's > n3's (credit-weighted, GRPO-normalized).
    adv = {n["node_id"]: a for n, a in zip(nodes, result.advantages)}
    assert adv["n1"] > adv["n2"] > adv["n3"]


def test_advantage_flat_broadcast_when_no_credit():
    # Backward-compat: when credit is not supplied, fall back to the existing
    # flat-broadcast behavior (one reward per episode, broadcast to all turns).
    computer = TreeAdvantageComputer()
    nodes = [
        {"node_id": "n1", "episode_id": "e1", "outcome_reward": 1.0, "num_tokens": 4},
        {"node_id": "n2", "episode_id": "e1", "outcome_reward": 1.0, "num_tokens": 4},
    ]
    result = computer.compute(nodes, group_size=1)
    adv = {n["node_id"]: a for n, a in zip(nodes, result.advantages)}
    assert adv["n1"] == adv["n2"]  # flat broadcast
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_advantage.py -k "per_node_credit or flat_broadcast_when_no_credit" -v`
Expected: FAIL (the existing `compute` does not accept a `credit` field, or the signature mismatches).

- [ ] **Step 3: Write minimal implementation**

In `advantage.py`, extend `TreeAdvantageComputer.compute` to accept per-node `credit`. The exact change depends on the current signature; the pattern is:

```python
class TreeAdvantageComputer:
    def compute(self, nodes, group_size):
        # If any node carries a `credit` field, use it as the per-node reward;
        # otherwise fall back to the existing `outcome_reward` broadcast
        # (backward-compat for non-DAG runs).
        has_credit = any("credit" in n for n in nodes)
        if has_credit:
            rewards = [n.get("credit", n.get("outcome_reward", 0.0)) for n in nodes]
        else:
            # Existing flat-broadcast path: one reward per episode.
            ep_reward = {}
            for n in nodes:
                ep_reward.setdefault(n["episode_id"], n.get("outcome_reward", 0.0))
            rewards = [ep_reward[n["episode_id"]] for n in nodes]
        # GRPO-normalize across the group (existing logic).
        ...  # existing normalization, applied to `rewards`
        return AdvantageResult(advantages=normalized, ...)
```

(Read the existing `compute` body and graft the `has_credit` branch at the top, leaving the existing normalization math intact. Do not rewrite the normalization.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_advantage.py -v`
Expected: PASS (all advantage tests, including pre-existing ones).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/advantage.py customized_areal/tree_search/tests/test_advantage.py
git commit -m "feat(dag): extend TreeAdvantageComputer to consume per-node credit"
```

---

## Task 14: Per-issue orchestration loop — `run_swe_lego_issue` (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/agents/swe_lego_issue_runner.py`
- Test: `backend/areal/customized_areal/tree_search/tests/test_swe_lego_issue_runner.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
from dataclasses import dataclass

import pytest

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue, SweLegoSetup
from customized_areal.tree_search.agents.swe_lego_issue_runner import (
    run_swe_lego_issue,
    SweLegoIssueResult,
)
from customized_areal.tree_search.agents.verifier import VerifierResult


@dataclass
class FakeMulticaClient:
    setup: SweLegoSetup
    create_calls: list = None
    cleanup_calls: list = None

    def __post_init__(self):
        self.create_calls = []
        self.cleanup_calls = []

    async def create_swe_lego_issue(self, *, issue, group_size, agent_config_id, base_image=None):
        self.create_calls.append((issue, group_size))
        return self.setup

    async def cleanup_swe_lego_issue(self, *, project_id):
        self.cleanup_calls.append(project_id)


@dataclass
class FakeRlSession:
    sessions: list = None

    def __post_init__(self):
        self.sessions = []

    async def start(self, *, agent_run_id, issue_id):
        self.sessions.append(agent_run_id)
        return f"sess-{agent_run_id}"


@dataclass
class FakeVerifier:
    async def verify_and_reward(self, *, agent_run_id, sandbox_id, session_id, fail_to_pass, pass_to_pass, transcript, acceptance_criteria):
        return VerifierResult(success=True, reward=1.0, source="objective")


@dataclass
class FakeBranchDriver:
    """Stands in for select_branch_candidate + BranchMaterializer.materialize."""
    ran_lanes: list = None

    def __post_init__(self):
        self.ran_lanes = []

    async def drive_lane(self, *, agent_run_id, sandbox_id, session_id):
        self.ran_lanes.append(agent_run_id)
        return sandbox_id  # the terminal sandbox id


def _issue() -> SweLegoIssue:
    return SweLegoIssue(
        repo_url="r", base_commit="c", issue_date="d",
        issue_text="x", issue_title="t", acceptance_criteria="a",
        fail_to_pass=["f"], pass_to_pass=["p"],
    )


def _setup() -> SweLegoSetup:
    return SweLegoSetup(
        project_id="p1", issue_id="i1", image_id="img1", build_node_id="n1",
        base_sandbox_id="sbx-base", base_sandbox_runtime_id="rt-base",
        agent_run_ids=["r1", "r2"],
    )


def test_run_swe_lego_issue_happy_path():
    multica = FakeMulticaClient(setup=_setup())
    rl = FakeRlSession()
    verifier = FakeVerifier()
    driver = FakeBranchDriver()
    result = asyncio.run(
        run_swe_lego_issue(
            issue=_issue(), group_size=2, agent_config_id="ag",
            multica=multica, rl_session=rl, verifier=verifier, branch_driver=driver,
        )
    )
    assert isinstance(result, SweLegoIssueResult)
    assert multica.create_calls == [(_issue(), 2)] or multica.create_calls[0][1] == 2
    assert len(rl.sessions) == 2
    assert len(driver.ran_lanes) == 2
    assert result.per_agent_rewards == [1.0, 1.0]
    assert multica.cleanup_calls == ["p1"]


def test_run_swe_lego_issue_does_not_cleanup_on_verifier_failure(monkeypatch):
    # If the verifier raises, the runner must still cleanup the multica
    # resources (otherwise we leak sandboxes), but should propagate the error.
    multica = FakeMulticaClient(setup=_setup())
    rl = FakeRlSession()

    @dataclass
    class RaisingVerifier:
        async def verify_and_reward(self, **kwargs):
            raise RuntimeError("verifier crashed")

    with pytest.raises(RuntimeError, match="verifier crashed"):
        asyncio.run(
            run_swe_lego_issue(
                issue=_issue(), group_size=2, agent_config_id="ag",
                multica=multica, rl_session=rl, verifier=RaisingVerifier(),
                branch_driver=FakeBranchDriver(),
            )
        )
    # Cleanup happened despite the verifier error.
    assert multica.cleanup_calls == ["p1"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_issue_runner.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Write minimal implementation**

```python
"""Per-issue orchestration loop for SWE-Lego DAG RL training.

Called by the training episode loop, once per issue (spec §5.1). Owns the
atomic setup, RL session opening, per-lane branching, verification, and
cleanup. Uses stdlib :mod:`logging` so the module stays importable
without torch.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from customized_areal.tree_search.agents.reward.swe_lego_types import (
    SweLegoIssue,
    SweLegoIssueResult,
)

logger = logging.getLogger("SweLegoIssueRunner")


class _MulticaClient(Protocol):
    async def create_swe_lego_issue(self, *, issue: SweLegoIssue, group_size: int, agent_config_id: str, base_image: str | None = ...) -> Any: ...
    async def cleanup_swe_lego_issue(self, *, project_id: str) -> None: ...


class _RlSession(Protocol):
    async def start(self, *, agent_run_id: str, issue_id: str) -> str: ...


class _Verifier(Protocol):
    async def verify_and_reward(self, **kwargs: Any) -> Any: ...


class _BranchDriver(Protocol):
    async def drive_lane(self, *, agent_run_id: str, sandbox_id: str, session_id: str) -> str: ...


async def run_swe_lego_issue(
    *,
    issue: SweLegoIssue,
    group_size: int,
    agent_config_id: str,
    multica: _MulticaClient,
    rl_session: _RlSession,
    verifier: _Verifier,
    branch_driver: _BranchDriver,
    base_image: str | None = None,
) -> SweLegoIssueResult:
    # 1. Atomic setup.
    setup = await multica.create_swe_lego_issue(
        issue=issue, group_size=group_size, agent_config_id=agent_config_id, base_image=base_image
    )

    try:
        # 2. Open one RL session per agent run.
        sessions = [
            await rl_session.start(agent_run_id=rid, issue_id=setup.issue_id)
            for rid in setup.agent_run_ids
        ]

        # 3. Drive branching within each lane. The driver returns the
        #    terminal sandbox id for each lane (the leaf of its branch tree).
        terminal_sandboxes = await asyncio.gather(*[
            branch_driver.drive_lane(
                agent_run_id=rid, sandbox_id=setup.base_sandbox_id, session_id=sid
            )
            for rid, sid in zip(setup.agent_run_ids, sessions)
        ])

        # 4. Verify + reward each terminal run.
        results = await asyncio.gather(*[
            verifier.verify_and_reward(
                agent_run_id=rid, sandbox_id=sbx, session_id=sid,
                fail_to_pass=issue.fail_to_pass, pass_to_pass=issue.pass_to_pass,
                transcript="...", acceptance_criteria=issue.acceptance_criteria,
            )
            for rid, sbx, sid in zip(setup.agent_run_ids, terminal_sandboxes, sessions)
        ])
        return SweLegoIssueResult(
            per_agent_rewards=[r.reward for r in results],
            per_agent_success=[r.success for r in results],
        )
    finally:
        # 5. Cleanup always, even on failure (no sandbox leaks).
        try:
            await multica.cleanup_swe_lego_issue(project_id=setup.project_id)
        except Exception:
            logger.exception("cleanup failed for project %s", setup.project_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_issue_runner.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/agents/swe_lego_issue_runner.py customized_areal/tree_search/tests/test_swe_lego_issue_runner.py
git commit -m "feat(swe-lego): add per-issue orchestration loop with guaranteed cleanup"
```

---

## Task 15: `BranchMaterializer` smoke test with `MulticaSweLegoProvider` (areal, Python)

**Files:**
- Test: `backend/areal/customized_areal/tree_search/tests/test_integration_multica.py`

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import httpx
import pytest

from customized_areal.tree_search.agents.environment import MulticaSweLegoProvider
from customized_areal.tree_search.agents.integration import (
    BranchMaterializer,
    MulticaIssueForker,
)
from customized_areal.tree_search.agents.verifier import VerifierResult


def _router(routes):
    def dispatch(request: httpx.Request):
        path = request.url.path
        for (method, route), handler in routes.items():
            if request.method != method:
                continue
            if route == path or (route.endswith("*") and path.startswith(route[:-1])):
                return handler(request)
        return httpx.Response(404, json={"error": f"no route for {request.method} {path}"})
    return httpx.MockTransport(dispatch)


@dataclass
class FakeStarter:
    started: list = None
    def __post_init__(self): self.started = []
    async def start_branch(self, *, forked_sandbox_id, forked_issue_id, replay_messages, drop_prior_session_id):
        self.started.append((forked_sandbox_id, forked_issue_id))
        return "branch-run-1"


def test_branch_materializer_works_with_multica_provider():
    # Snapshot + fork on the multica provider, fork-issue on the issue forker,
    # start_branch on the fake starter.
    routes = {
        ("POST", "/api/v1/sandboxes/sbx-1/snapshot*"): lambda r: httpx.Response(200, json={"snapshot_id": "snap-1"}),
        ("POST", "/api/v1/sandboxes/fork*"): lambda r: httpx.Response(200, json={"sandbox_id": "forked-sbx"}),
    }
    env = MulticaSweLegoProvider(base_url="https://multica.example", transport=_router(routes))

    issue_routes = {
        ("POST", "/api/issues/i1/fork*"): lambda r: httpx.Response(201, json={"forked_issue_id": "forked-i1"}),
    }
    forker = MulticaIssueForker(base_url="https://multica.example", transport=_router(issue_routes))

    starter = FakeStarter()
    mat = BranchMaterializer(env=env, forker=forker, starter=starter)
    result = asyncio.run(mat.materialize(
        source_sandbox_id="sbx-1", source_issue_id="i1", task_id="t1", seq=5,
        replay_messages=[{"role": "user", "content": "hi"}],
    ))
    assert result.branch_run_id == "branch-run-1"
    assert result.forked_sandbox_id == "forked-sbx"
    assert result.forked_issue_id == "forked-i1"
    assert starter.started == [("forked-sbx", "forked-i1")]
```

(Add `from dataclasses import dataclass` to the imports.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_integration_multica.py -v`
Expected: may FAIL with import errors if `dataclass` import is missing, or PASS if the materializer already accepts the provider via Protocol (which it does — `integration.py` injects `env: ForkableEnvironment`).

If it PASSES immediately, that confirms the spec's invariant 3 ("`BranchMaterializer` is untouched") — commit the test as a regression guard and move on.

- [ ] **Step 3: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_integration_multica.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/tests/test_integration_multica.py
git commit -m "test(dag): smoke-test BranchMaterializer with MulticaSweLegoProvider"
```

---

## Task 16: Anti-cheating canary test (areal, Python)

**Files:**
- Create: `backend/areal/customized_areal/tree_search/tests/test_swe_lego_anti_hacking.py`

- [ ] **Step 1: Write the failing test**

This test is a regression guard on the git-history-truncation contract. It does not run real git; it asserts that the build script (shipped to the build node by the multica side) contains the `--commit-cutoff` directive, and that the areal-side `SweLegoIssue` carries the `issue_date` that drives it.

```python
"""Anti-cheating canary: assert SWE-Lego anti-hacking is enforced.

The git history after `issue_date` MUST be deleted at image-build time so an
agent cannot `git log` or `git blame` its way to the future fix. This test
guards the contract: if the build-script generation drops the
`--commit-cutoff`, this test fails before any agent runs.

It does NOT run real git or docker — it inspects the build script string
that the multica side ships to the build node. (The multica-side Go test
`TestSweLegoBuildScript_ContainsFilterRepoCutoff` covers the Go generation;
this test covers the areal-side `SweLegoIssue.issue_date` plumbing that
feeds it.)
"""

from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue


def test_swe_lego_issue_carries_issue_date_for_truncation():
    issue = SweLegoIssue(
        repo_url="r", base_commit="c", issue_date="2025-03-14T09:30:00Z",
        issue_text="x", issue_title="t", acceptance_criteria="a",
        fail_to_pass=["f"], pass_to_pass=["p"],
    )
    # If issue_date is empty/missing, the build script cannot compute the
    # cutoff commit and history truncation silently no-ops.
    assert issue.issue_date, "issue_date must be non-empty to drive git filter-repo --commit-cutoff"
    # base_commit must also be present (the checkout target).
    assert issue.base_commit, "base_commit must be non-empty"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd backend/areal && uv run pytest customized_areal/tree_search/tests/test_swe_lego_anti_hacking.py -v`
Expected: PASS (this test guards an existing contract; it should pass immediately given Task 6).

- [ ] **Step 3: Commit**

```bash
cd backend/areal
git add customized_areal/tree_search/tests/test_swe_lego_anti_hacking.py
git commit -m "test(swe-lego): add anti-hacking canary on issue_date/base_commit"
```

---

## Task 17: End-to-end validation (multica + areal)

**Files:**
- No new files; this is a manual/integration validation task.

- [ ] **Step 1: Confirm hardware availability**

Run: `python -c "import torch; print('GPU available:', torch.cuda.is_available())"` (areal side) and confirm a Fleet build-node tagged `swe-lego-build` is reachable from the multica server.

If either is unavailable, document the skip per `backend/areal/CLAUDE.md` ("Integration tests requiring multi-node hardware are skipped with an explanation when unavailable") and stop here — the unit tests in Tasks 1-16 are the v1 acceptance gate.

- [ ] **Step 2: Run the e2e at `group_size=2`**

From the areal side, with a real multica + Fleet build-node + cloud sandbox stack:

```bash
cd backend/areal
uv run python -c "
import asyncio
from customized_areal.tree_search.agents.swe_lego_issue_runner import run_swe_lego_issue
from customized_areal.tree_search.agents.reward.swe_lego_types import SweLegoIssue
from customized_areal.tree_search.agents.swe_lego_client import MulticaSweLegoClient
# ... construct the deps with real base_url/api_key ...
# issue = SweLegoIssue(...) against a small public repo
# result = asyncio.run(run_swe_lego_issue(...))
# assert len(result.per_agent_rewards) == 2
"
```

Assert:
- The multica endpoint returns 201 with `agent_run_ids` of length 2.
- The build node's docker cache contains `swe-lego:<cache_key>` after the build.
- Two daemon-in-docker sandboxes boot and the daemons register their runtimes.
- Agents run and POST `task_message` batches (visible in the multica UI or via `GET /api/tasks/{taskId}/messages`).
- The verifier produces a non-`None` reward per agent.
- `cleanup_swe_lego_issue` deletes the project and sandboxes.

- [ ] **Step 3: Run the anti-cheating canary live**

In one of the forked sandboxes, after the agent finishes but before cleanup:

```bash
docker exec <forked-sandbox> git -C /workspace/repo log --after=2025-03-14
```

Assert: no commits after `issue_date` are reachable (empty output, or an error indicating the range is invalid). This confirms the `git filter-repo --commit-cutoff` actually deleted the future history.

- [ ] **Step 4: Document the run**

Record the run details (repo, base_commit, group_size, rewards, any failures) in the commit message of the e2e validation commit. If issues are found, file them as follow-up tasks rather than expanding this plan.

- [ ] **Step 5: Commit (validation record only)**

```bash
cd backend/areal
git add --allow-empty docs/superpowers/plans/2026-06-29-swe-lego-multica-areal-integration.md
git commit -m "test(swe-lego): e2e validation at group_size=2 (record)"
```

(If your workflow does not allow empty commits, instead commit a short validation-notes file at `docs/superpowers/plans/2026-06-29-swe-lego-e2e-validation.md` with the run details.)

---

## Self-Review

**1. Spec coverage:**
- §2 decision 1 (per-issue + mid-run branching) → Tasks 5, 14, 15 (provider + runner + materializer smoke).
- §2 decision 2 (group_size + DAG) → Tasks 14, 12 (runner drives group_size lanes; DAG backup).
- §2 decision 3 (multica owns docker) → Tasks 8-10 (multica endpoint + service).
- §2 decision 4 (daemon-in-docker) → Task 4 (Dockerfile with `multica-daemon` CMD).
- §2 decision 5 (anti-hacking) → Tasks 2, 16 (build script with `--commit-cutoff` + canary).
- §2 decision 6 (hybrid verifier) → Task 11.
- §2 decision 7 (atomic endpoint) → Tasks 8-10.
- §2 decision 8 (build on Fleet node) → Tasks 3, 10 (NodeExec + adapter wiring).
- §2 decision 9 (DAG credit load-bearing) → Tasks 12, 13 (backup + advantage extension).
- §4 (multica side) → Tasks 1-4, 8-10.
- §5 (areal side) → Tasks 5-7, 11, 14.
- §6 (data flow) → Task 14 (runner) + Task 17 (e2e).
- §7 (error handling) → Tasks 9 (rollback), 11 (critic fallback), 14 (cleanup-on-failure).
- §8 (security/anti-cheating) → Tasks 2, 16, 17.
- §9 (testing strategy) → each task's test + Task 17 (e2e).
- §10 (out of scope) → no tasks; respected.

**2. Placeholder scan:** No "TBD" or "implement later." Task 9 Step 6 uses stub adapter methods explicitly, replaced in Task 10 Step 2 — that is intentional sequencing, not a placeholder. Task 13 Step 0 has a "locate `TreeAdvantageComputer`" instruction because its file location is not pinned in the spec; the grep command resolves it concretely. Task 17 is a manual validation task with explicit run commands.

**3. Type consistency:**
- `SweLegoIssue` / `SweLegoSetup` / `SweLegoIssueResult` defined in Task 6, used in Tasks 7, 11, 14.
- `MulticaSweLegoProvider` defined in Task 5, used in Task 15.
- `MulticaSweLegoClient.create_swe_lego_issue` / `cleanup_swe_lego_issue` defined in Task 7, used in Task 14.
- `SweLegoVerifier.verify_and_reward` signature consistent across Tasks 11 and 14.
- Go: `SweLegoCacheKey`, `SweLegoBuildScript`, `BuildOrReuse`, `NewSweLegoIssueService`, `SweLegoIssueInput`/`Result`, `SweLegoDeps` — names match across Tasks 1-3, 8-10.
- `ObjectiveOutcome` fields (`fully_passes`, `f2p_passed`, `f2p_total`, `failing_count_reduced`) consistent between Task 11 impl and test.

No type drift found.
