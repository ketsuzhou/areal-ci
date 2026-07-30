# Move Tree Search Config from .env to YAML Config

## Summary

Move all `TREE_SEARCH_*` configuration values from `customized_areal/.env` into the
`TPFCConfig` dataclass (nested under `TreeBackupConfig`), so they are persisted in
`config.yaml` alongside other training config and restored on recovery.

## Motivation

- `.env` is a separate config channel from the YAML config system, causing duplication
  and manual sync burden (e.g., `cache_dir` exists in both YAML and `.env`)
- On recovery, `config.yaml` is the authoritative snapshot of the experiment — but tree
  search settings are missing from it
- The `TreeBackupConfig` dataclass already defines all tree search fields with correct
  defaults; it just isn't wired into the config hierarchy

## Design

### Data model

Add 2 fields to `TreeBackupConfig` in `customized_areal/tree_search/config.py`:

- `enabled: bool = True` — replaces `use_TreeSearchGroupedRolloutWorkflow` env var
- `max_reasoning_tokens: int = 1000` — replaces `TREE_SEARCH_MAX_REASONING_TOKENS` env
  var

Add `tree_search: TreeBackupConfig` to `TPFCConfig` in
`customized_areal/tpfc/tpfc_config.py`. Remove the 3 now-redundant flat fields:
`cache_dir`, `cache_mode`, `loss_mode`.

### Configuration flow

```
YAML config ──> TPFCConfig.tree_search (TreeBackupConfig)
                    │
                    ├──> config.yaml (auto-serialized, complete)
                    │
                    └──> remote_inf_engine.py._resolve_workflow()
                              reads self.config.tree_search.*
                              ──> TreeSearchGroupedRolloutWorkflow(...)
```

On recovery, `config.yaml` is loaded back into `TPFCConfig`, restoring all tree search
settings without any `.env` dependency.

### Code changes

1. **`customized_areal/tree_search/config.py`** — Add `enabled` and
   `max_reasoning_tokens` fields to `TreeBackupConfig`

1. **`customized_areal/tpfc/tpfc_config.py`** — Remove `cache_dir`, `cache_mode`,
   `loss_mode`; add
   `tree_search: TreeBackupConfig = field(default_factory=TreeBackupConfig)`

1. **`areal/infra/remote_inf_engine.py`** — Replace `os.getenv("TREE_SEARCH_*")` reads
   (lines 700-785) with `self.config.tree_search.*` attribute access. The
   `TreeSearchGroupedRolloutWorkflow` constructor call stays the same.

1. **Callers of removed fields** — Update references from `config.cache_dir` →
   `config.tree_search.checkpoint_dir`, `config.cache_mode` →
   `config.tree_search.cache_mode`, `config.loss_mode` → `config.tree_search.loss_mode`
   in training scripts and trainer.

1. **YAML config files** — Add `tree_search:` section with all fields. Remove top-level
   `cache_dir`, `cache_mode`, `loss_mode`.

1. **`customized_areal/.env`** — Remove `use_TreeSearchGroupedRolloutWorkflow` and all
   `TREE_SEARCH_*` lines. Non-tree-search entries (SUPABASE, DAYTONA, etc.) remain.

### train_id.json

`train_id.json` is unchanged — it stores the runtime-generated `TRAIN_ID` UUID, which is
runtime identity, not static configuration. It remains in the checkpoint directory
written by `_write_train_id_sidecar()`.

## Migration

Existing YAML configs must add a `tree_search:` block. Example:

```yaml
tree_search:
  enabled: true
  checkpoint_dir: customized_areal/tpfc/data/tree_cache
  advantage_mode: tree
  loss_mode: grpo
  cache_mode: cross_training
  rl_loss_weight: 1.0
  distill_loss_weight: 0.005
  max_reasoning_tokens: 1000
  teacher_provider: external
  teacher_base_url: http://localhost:8001
  teacher_model_name: ""
  teacher_top_k: 10
  teacher_max_retries: 3
  teacher_timeout: 60.0
  teacher_missing_logprob: -23.0
  diagnose_model_name: qwen/qwen3.5-397b-a17b
  diagnose_max_tokens: 1024
  diagnose_temperature: 0.0
  diagnose_base_url: http://10.254.244.168:8443/service-large-64-1775803465274/llm/v1
  diagnose_api_key: aH6867Z7ppZWqN7NQGsaPhSlMrk68AGQnr18Z66KsSL0864mkjaBP7Tr868pKvnCJcWa0VvHZDHk64MrKG5Z5gWrZGx88mrmAv7pk88p786gFB6XXr6MrtSckKs9QJsr
  strict_distill_json: true
```
