# pi verifier-rl extension

Slash commands + read-only transcript access for the **verifier agent** (a pi
agent on a fixed judge model) that reviews a finished multi-agent task and
assigns a reward per RL `session_id`.

## Commands / tools

| Command / tool                          | Effect                                                              |
| --------------------------------------- | ------------------------------------------------------------------- |
| `/rl/set_reward <session_id> <reward>`  | `POST <gateway>/rl/set_reward` `{ session_id, reward }`              |
| `/export_trajectories <session_id>`     | `POST <gateway>/export_trajectories` `{ session_id }` (terminal)    |
| `read_task_messages(task_id)` (tool)    | `GET <multica>/api/tasks/{task_id}/messages` — read-only transcript |

Reward-before-export ordering is the contract: the verifier sets rewards first;
the training loop (Python `AgenticVerifier`/harvest) calls `/export_trajectories`
afterward, which revokes the session.

## Configuration (no secrets hardcoded)

| Env var               | Purpose                          |
| --------------------- | -------------------------------- |
| `AREAL_GATEWAY_URL`   | AReaL inference gateway base URL |
| `AREAL_ADMIN_API_KEY` | Admin bearer key for the gateway |
| `MULTICA_BASE_URL`    | Multica base URL (transcripts)   |
| `MULTICA_API_KEY`     | Multica bearer key (optional)    |

The verifier runs on a fixed judge model and must **not** route its own LLM
calls through `areal/...`; these helpers only touch the RL control-plane
endpoints with the admin key.

## Placement

Copy this directory to the verifier agent's `.pi/extensions/` (project-local)
or `~/.pi/agent/extensions/` (global).

## Tests

```bash
node --test   # requires Node >= 23 (native TS type-stripping)
```
