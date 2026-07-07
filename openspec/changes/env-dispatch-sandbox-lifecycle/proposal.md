## Why

Env-dispatch training needs a concrete, resumable environment lifecycle so rollouts can save state at useful decision points and later resume without inventing a parallel sandbox provider. `lijiannankai@126.com`'s Multica sandbox work already provides the needed foundation through `sandbox_instance` rows, sandboxd jobs, Cube pause/resume/delete, node ownership, and offline-node force-delete behavior.

## What Changes

- Integrate sandbox-instance-backed environment handles into env-dispatch so rollout environments reference Multica `sandbox_instance` records instead of opaque sandbox ids where save/resume is required.
- Add pause-in-place environment save semantics that reuse the existing sandboxd `stop` job and Cube `/pause` behavior.
- Add resume-from-checkpoint semantics that reuse existing sandboxd `resume` jobs and runtime restart metadata.
- Add checkpoint storage and APIs for create/list/resume so AReaL can save and resume rollout environments.
- Extend the AReaL env-dispatch client to send per-agent environment intent and call resume-from-checkpoint APIs.
- Preserve existing sandboxd delete and unreachable-node force-delete behavior; no immutable sandbox fork/snapshot capability is promised in this change.

## Capabilities

### New Capabilities

- `env-dispatch-sandbox-lifecycle`: Env-dispatch environments use sandbox-instance lifecycle handles for save, resume, delete, and per-agent sandbox assignment.
- `env-checkpoint-resume`: Training rollouts can create, list, and resume pause-in-place environment checkpoints backed by sandboxd jobs.

### Modified Capabilities

- `training-session-lifecycle`: Training sessions preserve the env-dispatch sandbox lifecycle handle needed to save and resume the rollout environment.
- `critic-driven-training-signal`: Entropy/logprob data captured for critic-driven training can optionally trigger checkpoint creation at uncertain tool-call decisions.

## Impact

- **multica/server**: env-dispatch handler/service contracts, sandbox lifecycle service wrapper, checkpoint service/handler, migrations, sqlc queries, and trigger seams around trained rollout events.
- **sandboxd/Cube lifecycle**: reuse existing create, stop, resume, delete, reconfigure, local_ref, runtime metadata, websocket wakeup, node ownership, and force-delete flows.
- **AReaL client**: `customized_areal/tree_search/agents/swe_lego_client.py` gains per-agent env serialization and checkpoint create/list/resume helpers.
- **Existing specs**: extends training-session lifecycle and critic-driven entropy capture without changing their core session-open/session-close contracts.
- **Out of scope**: immutable sandbox snapshots/forks, concurrent branches from a live rollout, copy-on-write sandbox optimization, and implementing code during the Open phase.
