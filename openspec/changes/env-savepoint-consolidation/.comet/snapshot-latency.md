# SWE-scale snapshot latency

**Status: not measured. Requires a human operator with Cube API access.**

This environment cannot run the measurement. There is no Cube base URL or credential in
the devcontainer (`multica/.env` carries none, and no `CUBE_*` variable is set), and the
Cube API is not reachable from here. Rather than fabricate a number, this file records
what the code assumes, what would invalidate it, and the exact procedure to settle it.

## What the design decision rests on

The prerequisite experiment (design doc, "Prerequisite experiment") measured
`POST /sandboxes/{id}/snapshots` at **1.2s**, with the source left `running` and its
processes undisturbed. Synchronous saving was chosen on that basis: at ~1s there is no
reason to complicate the client contract with an async save.

Cube snapshots are memory-level checkpoint/restore, not filesystem copies, so duration
should track the sandbox's **memory footprint**, not repository size on disk. That is
exactly why the 1.2s figure needs re-measuring: it came from an idle sandbox, and a SWE
agent mid-task holds a warm working set, a language server, and a build cache in RAM.

## The threshold that actually binds

Not the server's budget. Two code facts set the real ceiling:

1. **The AReaL client's HTTP timeout is 120s.** `MulticaEnvDispatchClient.__init__`
   defaults `timeout: float = 120.0`, and `create_env_dispatch` goes through `_request`
   on that shared `httpx.AsyncClient`. The server sets no `WriteTimeout`
   (`cmd/server/main.go:388`), so the client gives up first. When it does, it raises
   `network request failed` while the server keeps going and still creates the checkpoint
   and lanes — a dispatch that looks failed to the caller but succeeded on the server.

2. **Capture is serial, once per source sandbox.** `EnvCheckpointService.Create` loops
   `for _, ref := range in.SandboxRefs` and waits for each savepoint to reach `ready`
   before starting the next. Under the eager-all decision, a branch dispatch captures
   every ready sandbox in the source channel, so the wall clock is
   `roster_size × per_snapshot_duration`.

So the passing condition is:

```
per_snapshot_duration × roster_size < 120s
```

At the measured 1.2s an eight-agent roster costs ~10s and there is nothing to worry
about. At 20s per snapshot a six-agent roster lands exactly on the client's timeout.

The server-side `branchSavepointSaveTimeout` of 15 minutes is deliberately generous and
is **not** the constraint; it exists so the server does not abandon a slow capture that
the client is still waiting on, and because the later peer-provisioning path runs under a
five second deadline and cannot wait at all.

## Procedure

Run from a host that can reach the Cube API, against a sandbox warmed to realistic SWE
scale — a real repository cloned, dependencies installed, and an agent turn or two
already executed, so the memory footprint resembles production rather than a fresh boot.

```bash
# Substitute the project's Cube base URL and the SWE-Lego template.
SBX=$(curl -sS -X POST "$CUBE_URL/sandboxes" -H 'content-type: application/json' \
  -d '{"templateID":"<swe-lego-template>","timeout":3600}' | jq -r .sandboxID)

# Warm it: clone a large repository, install dependencies, run a build.
# Then record the memory footprint that the snapshot has to capture.
curl -sS -X POST "$CUBE_URL/sandboxes/$SBX/execute" -H 'content-type: application/json' \
  -d '{"language":"bash","code":"free -m; du -sh /workspace"}'

# A probe distinguishes "frozen then restored" from "killed": a continuous counter
# with a gap in timestamps means frozen; a restarted counter means killed.
curl -sS -X POST "$CUBE_URL/sandboxes/$SBX/execute" -H 'content-type: application/json' \
  -d '{"language":"bash","code":"nohup bash -c \"i=0; while true; do i=$((i+1)); echo \\$i \\$(date +%s) >> /tmp/probe.log; sleep 1; done\" >/dev/null 2>&1 &"}'

# The measurement.
time curl -sS -X POST "$CUBE_URL/sandboxes/$SBX/snapshots" \
  -H 'content-type: application/json' -d '{}'

# The source must still be running, and the probe must have kept its counter.
curl -sS "$CUBE_URL/sandboxes/$SBX" | jq -r .state
curl -sS -X POST "$CUBE_URL/sandboxes/$SBX/execute" -H 'content-type: application/json' \
  -d '{"language":"bash","code":"tail -5 /tmp/probe.log"}'
```

## Record here

| Field                                      | Value |
| ------------------------------------------ | ----- |
| Sandbox memory footprint at snapshot (MB)  |       |
| Repository size on disk                    |       |
| Wall-clock snapshot duration               |       |
| Source `state` immediately after           |       |
| Probe counter continuous (frozen, not killed)? |   |
| Largest roster size this supports (120s ÷ duration) | |

Then state plainly whether it supports or contradicts the synchronous-save decision.

## If the measurement comes back slow

Do not reach for an async save first. Two cheaper mitigations come before changing the
client contract:

- **Parallelize the capture loop.** It is serial today only because 1.2s made the
  ordering irrelevant. Capturing the roster concurrently turns
  `roster_size × duration` back into roughly `duration`, which removes the multiplier
  without touching the client. This is deliberately not implemented ahead of the
  measurement: it adds concurrent snapshot load on the Cube host and error aggregation
  across lanes, and the current evidence says it buys nothing.
- **Raise the client timeout.** A one-line change in `MulticaEnvDispatchClient`, but it
  makes every AReaL dispatch wait longer on unrelated failures, so it is a worse first
  move than removing the multiplier.
