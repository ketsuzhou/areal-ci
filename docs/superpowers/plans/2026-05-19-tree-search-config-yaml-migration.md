# Move Tree Search Config from .env to YAML Config — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all TREE_SEARCH\_\* config from `.env` into `TPFCConfig.tree_search`
(TreeBackupConfig dataclass) so values are persisted in config.yaml and restored on
recovery.

**Architecture:** Add `TreeBackupConfig` as a nested field on `TPFCConfig`, pass it
through `workflow_kwargs` to `_resolve_workflow()` in `remote_inf_engine.py`, where it
replaces `os.getenv()` reads. The `.env` file is cleaned up afterward.

**Tech Stack:** Python 3.12+, dataclasses, OmegaConf/YAML

**Key design decision:** `_resolve_workflow()` runs inside `RemoteInfEngine` which holds
`InferenceEngineConfig` (the rollout subsection), not the full `TPFCConfig`. Rather than
adding tree_search fields to the core `InferenceEngineConfig`, we pass the
`TreeBackupConfig` through `workflow_kwargs` — the existing parameter that already
carries workflow-level config. It is extracted early in `_resolve_workflow` and excluded
from the inner workflow constructor calls.

______________________________________________________________________

### Task 1: Add `enabled` and `max_reasoning_tokens` to TreeBackupConfig

**Files:**

- Modify: `customized_areal/tree_search/config.py`

- [ ] **Step 1: Add two fields to TreeBackupConfig**

Add `enabled` and `max_reasoning_tokens` immediately after the existing `mode` field:

```python
@dataclass
class TreeBackupConfig:
    mode: CacheMode = CacheMode.OFF
    enabled: bool = True
    checkpoint_dir: str = ""
    advantage_mode: AdvantageMode = AdvantageMode.TREE
    loss_mode: LossMode = LossMode.GRPO
    max_reasoning_tokens: int = 1000
    rl_loss_weight: float = 1.0
    # ... rest of fields unchanged ...
```

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tree_search/config.py
git commit -m "feat(tree-search): add enabled and max_reasoning_tokens to TreeBackupConfig"
```

______________________________________________________________________

### Task 2: Add `tree_search` to TPFCConfig, remove flat fields

**Files:**

- Modify: `customized_areal/tpfc/tpfc_config.py`

- [ ] **Step 1: Import TreeBackupConfig and replace fields**

Add import and replace `cache_dir`, `cache_mode`, `loss_mode` with `tree_search`:

```python
"""Configuration for TPFC Agent training experiments."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path for imports when this file is imported directly
_project_root = Path(__file__).parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from areal.api.cli_args import PPOConfig
from customized_areal.tree_search.config import TreeBackupConfig


@dataclass
class AgentConfig:
    """Configuration for the TPFC agent."""

    trial_name: str = field(
        default="",
        metadata={"help": "Trial name for the agent."},
    )
    train_id: str = field(
        default="",
        metadata={"help": "Training run identifier."},
    )
    user_id: str = field(
        default="",
        metadata={"help": "User identifier."},
    )
    model_name: str | None = field(
        default=None,
        metadata={"help": "Name of the model to use."},
    )
    judge_model_name: str | None = field(
        default=None,
        metadata={"help": "Name of the judge model."},
    )
    judge_base_url: str | None = field(
        default=None,
        metadata={"help": "Base URL for the judge API."},
    )
    judge_api_key: str | None = field(
        default=None,
        metadata={"help": "API key for the judge model."},
    )


@dataclass
class TPFCConfig(PPOConfig):
    """Configuration for TPFC Agent training experiments.

    Extends PPOConfig with agent-specific settings for the TPFC agent
    workflow using the OpenAI-compatible proxy approach.
    """

    workflow: str = field(
        default="customized_areal.tpfc_agent.TPFCAgent",
        metadata={"help": "Path to the TPFC workflow class for training."},
    )
    eval_workflow: str = field(
        default="${workflow}",
        metadata={"help": "Path to the TPFC workflow class for evaluation."},
    )
    agent: AgentConfig = field(default_factory=AgentConfig)
    tree_search: TreeBackupConfig = field(
        default_factory=TreeBackupConfig,
        metadata={"help": "Tree search configuration (MCTS, cache, loss, teacher, diagnose)."},
    )
    assistant_marker: str = field(
        default="",
        metadata={"help": "Marker string identifying assistant turns in tree backup."},
    )
```

Changes from current:

- Remove `cache_dir`, `cache_mode`, `loss_mode` fields

- Add `from customized_areal.tree_search.config import TreeBackupConfig`

- Add `tree_search: TreeBackupConfig = field(default_factory=TreeBackupConfig)`

- [ ] **Step 2: Commit**

```bash
git add customized_areal/tpfc/tpfc_config.py
git commit -m "feat(tpfc): replace flat tree-search fields with nested TreeBackupConfig"
```

______________________________________________________________________

### Task 3: Update YAML config files

**Files:**

- Modify: `customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-opd.yaml`

- Modify:
  `customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml`

- Modify:
  `customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml`

- [ ] **Step 1: Update config_tpfc_Qwen3-5L-9B-opd.yaml**

Replace lines 7-10:

```yaml
# Tree search / cache configuration
cache_dir: customized_areal/tpfc/data/tree_cache
cache_mode: cross_training
loss_mode: distill
```

With:

```yaml
# Tree search configuration
tree_search:
  enabled: true
  checkpoint_dir: customized_areal/tpfc/data/tree_cache
  mode: cross_training
  loss_mode: distill
  advantage_mode: tree
  rl_loss_weight: 1.0
  distill_loss_weight: 0.005
  max_reasoning_tokens: 1000
  diagnose_model_name: qwen/qwen3.5-397b-a17b
  diagnose_base_url: http://10.254.244.168:8443/service-large-64-1775803465274/llm/v1
  diagnose_api_key: aH6867Z7ppZWqN7NQGsaPhSlMrk68AGQnr18Z66KsSL0864mkjaBP7Tr868pKvnCJcWa0VvHZDHk64MrKG5Z5gWrZGx88mrmAv7pk88p786gFB6XXr6MrtSckKs9QJsr
```

- [ ] **Step 2: Update config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml**

Replace lines 7-9:

```yaml
cache_dir: customized_areal/tpfc/data/tree_cache
cache_mode: cross_training
```

With:

```yaml
tree_search:
  enabled: true
  checkpoint_dir: customized_areal/tpfc/data/tree_cache
  mode: cross_training
  loss_mode: grpo
```

This config doesn't have `loss_mode` currently — it uses the default. Add
`loss_mode: grpo` explicitly for clarity.

- [ ] **Step 3: Update config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml**

Same as Step 2 — replace `cache_dir`/`cache_mode` lines with `tree_search:` block:

```yaml
tree_search:
  enabled: true
  checkpoint_dir: customized_areal/tpfc/data/tree_cache
  mode: cross_training
  loss_mode: grpo
```

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tpfc/configs/
git commit -m "feat(configs): migrate tree-search settings from flat fields to nested tree_search block"
```

______________________________________________________________________

### Task 4: Update train_tpfc_tree_search.py

**Files:**

- Modify: `customized_areal/tpfc/scripts/train_tpfc_tree_search.py`

- [ ] **Step 1: Replace config.cache_dir/cache_mode/loss_mode with config.tree_search**

Replace lines 107-154 (the "Build cache / tree backup configs from overrides" block):

```python
    # Build cache / tree backup configs from overrides
    tree_search = config.tree_search
    if not tree_search.checkpoint_dir:
        raise ValueError(
            "tree_search.checkpoint_dir must be set when using tree search training. "
            "Set it in the config YAML under tree_search.checkpoint_dir."
        )

    n_samples = config.gconfig.n_samples
    tree_mode = tree_search.mode
    loss_mode = tree_search.loss_mode

    cache_config = RolloutCacheConfig(
        cache_dir=tree_search.checkpoint_dir,
        enabled=True,
        n_samples=n_samples,
    )

    tree_backup_config = tree_search

    logger.info(
        "Cache config: dir=%s, n_samples=%d, tree_mode=%s, loss_mode=%s",
        tree_search.checkpoint_dir,
        n_samples,
        tree_backup_config.mode.value,
        tree_backup_config.loss_mode.value,
    )
```

Key changes:

- `config.cache_dir` → `config.tree_search.checkpoint_dir`

- `config.cache_mode` → `config.tree_search.mode` (already a `CacheMode` enum, no
  parsing needed)

- `config.loss_mode` → `config.tree_search.loss_mode` (already a `LossMode` enum, no
  parsing needed)

- `tree_backup_config = TreeBackupConfig(mode=..., checkpoint_dir=..., loss_mode=...)` →
  `tree_backup_config = tree_search` (use the full config directly)

- Remove the `TreeBackupConfig(...)` constructor call and the try/except blocks for enum
  parsing

- [ ] **Step 2: Pass tree_search_config through workflow_kwargs**

After the existing `workflow_kwargs` dict (around line 158), add the config:

```python
    # Build workflow kwargs
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=getattr(config.gconfig, "top_p", 1.0),
        max_completion_tokens=config.gconfig.max_new_tokens,
        tree_search_config=tree_backup_config,
    )
```

- [ ] **Step 3: Clean up unused imports**

Remove `LossMode` and `TreeBackupConfig` from the import block (line 24-29) since
they're no longer directly constructed:

Actually keep `TreeBackupConfig` — it's still referenced indirectly. And `CacheMode` and
`LossMode` are no longer needed for parsing. Remove them.

Change the import on lines 24-29 from:

```python
from customized_areal.tree_search.config import (
    CacheMode,
    LossMode,
    RolloutCacheConfig,
    TreeBackupConfig,
)
```

To:

```python
from customized_areal.tree_search.config import RolloutCacheConfig
```

Wait, `CacheMode` and `LossMode` are not used anywhere else in this file after the
changes. Let me verify... Looking at the full file again: lines 120-125 parse
`cache_mode` string → enum, and lines 134-140 parse `loss_mode` string → enum. With the
new code, these parsing blocks are removed. So `CacheMode` and `LossMode` imports can be
removed. `TreeBackupConfig` is also not directly constructed anymore.

Correct import:

```python
from customized_areal.tree_search.config import RolloutCacheConfig
```

- [ ] **Step 4: Commit**

```bash
git add customized_areal/tpfc/scripts/train_tpfc_tree_search.py
git commit -m "feat(train): read tree-search config from TPFCConfig.tree_search, pass via workflow_kwargs"
```

______________________________________________________________________

### Task 5: Update remote_inf_engine.py to read from workflow_kwargs

**Files:**

- Modify: `areal/infra/remote_inf_engine.py`

- [ ] **Step 1: Extract tree_search_config at top of \_resolve_workflow**

After the docstring/early section of `_resolve_workflow` (after line 594), insert
tree_search_config extraction:

```python
    def _resolve_workflow(
        self,
        workflow: WorkflowLike | None,
        workflow_kwargs: dict[str, Any] | None,
        group_size: int = 1,
        proxy_addr: str | None = None,
    ) -> RolloutWorkflow:
        resolved: RolloutWorkflow

        # Extract tree_search_config from workflow_kwargs so it is not
        # forwarded to inner workflow constructors (they don't accept it).
        tree_search_cfg = None
        if workflow_kwargs is not None and "tree_search_config" in workflow_kwargs:
            workflow_kwargs = dict(workflow_kwargs)
            tree_search_cfg = workflow_kwargs.pop("tree_search_config")

        # 0. None workflow = online mode (config-driven)
        # ... rest unchanged ...
```

- [ ] **Step 2: Replace the tree search wrapping block (lines 699-813)**

Replace the entire `if group_size > 1:` block's tree search section. The current code
spans lines 699-818. Replace the tree search branch (lines 700-813):

```python
        # Wrap with GroupedRolloutWorkflow if group_size > 1
        if group_size > 1:
            use_tree_search = (
                tree_search_cfg is not None and tree_search_cfg.enabled
            )
            if use_tree_search:
                self.logger.warning(
                    "use TreeSearchGroupedRolloutWorkflow"
                )

                from customized_areal.tree_search.config import (
                    AdvantageMode,
                    CacheMode,
                    LossMode,
                )
                from customized_areal.tree_search.tree_search_grouped_workflow import (
                    TreeSearchGroupedRolloutWorkflow,
                )

                resolved = TreeSearchGroupedRolloutWorkflow(
                    resolved,
                    group_size,
                    checkpoint_dir=tree_search_cfg.checkpoint_dir,
                    advantage_mode=tree_search_cfg.advantage_mode,
                    loss_mode=tree_search_cfg.loss_mode,
                    cache_mode=tree_search_cfg.mode,
                    tokenizer_path=self.config.tokenizer_path,
                    max_reasoning_tokens=tree_search_cfg.max_reasoning_tokens,
                    rl_loss_weight=tree_search_cfg.rl_loss_weight,
                    distill_loss_weight=tree_search_cfg.distill_loss_weight,
                    topk_distill=tree_search_cfg.topk_distill,
                    teacher_provider=tree_search_cfg.teacher_provider,
                    teacher_base_url=tree_search_cfg.teacher_base_url,
                    teacher_model_name=tree_search_cfg.teacher_model_name,
                    teacher_top_k=tree_search_cfg.teacher_top_k,
                    teacher_max_retries=tree_search_cfg.teacher_max_retries,
                    teacher_timeout=tree_search_cfg.teacher_timeout,
                    teacher_missing_logprob=tree_search_cfg.teacher_missing_logprob,
                    diagnose_model_name=tree_search_cfg.diagnose_model_name,
                    diagnose_max_tokens=tree_search_cfg.diagnose_max_tokens,
                    diagnose_temperature=tree_search_cfg.diagnose_temperature,
                    diagnose_base_url=tree_search_cfg.diagnose_base_url,
                    diagnose_api_key=tree_search_cfg.diagnose_api_key,
                    strict_distill_json=tree_search_cfg.strict_distill_json,
                )
            else:
                self.logger.warning(
                    "use GroupedRolloutWorkflow"
                )
                resolved = GroupedRolloutWorkflow(resolved, group_size, self.logger)
```

Key changes:

- `os.getenv("use_TreeSearchGroupedRolloutWorkflow", "False").lower() == "true"` →
  `tree_search_cfg is not None and tree_search_cfg.enabled`
- Remove the hardcoded `use_tree_search = True` line
- Remove the `load_dotenv` block (lines 708-716)
- Remove all `os.getenv("TREE_SEARCH_*")` reads (lines 727-786)
- Read all values from `tree_search_cfg.*` attributes
- The tokenizer_path fallback:
  `tree_search_cfg.checkpoint_dir or self.config.tokenizer_path` — wait, looking at the
  original code, `TREE_SEARCH_TOKENIZER_PATH` defaulted to `self.config.tokenizer_path`.
  There's no `tokenizer_path` on `TreeBackupConfig`. Let me check...

Looking at the original code (line 743-746):

```python
tokenizer_path = (
    os.getenv("TREE_SEARCH_TOKENIZER_PATH", "")
    or self.config.tokenizer_path
)
```

`TreeBackupConfig` doesn't have a `tokenizer_path` field. But the user did NOT list
`TREE_SEARCH_TOKENIZER_PATH` in their config values. They only listed:

- use_TreeSearchGroupedRolloutWorkflow
- TREE_SEARCH_CHECKPOINT_DIR
- TREE_SEARCH_ADVANTAGE_MODE
- TREE_SEARCH_LOSS_MODE
- TREE_SEARCH_CACHE_MODE
- TREE_SEARCH_RL_LOSS_WEIGHT
- TREE_SEARCH_DISTILL_LOSS_WEIGHT
- TREE_SEARCH_DIAGNOSE_MODEL_NAME
- TREE_SEARCH_DIAGNOSE_BASE_URL
- TREE_SEARCH_DIAGNOSE_API_KEY

The user didn't list all TREE_SEARCH\_\* vars. But we're moving ALL of them. So we need
to add `tokenizer_path` to `TreeBackupConfig` too.

Wait, `TREE_SEARCH_TOKENIZER_PATH` has a default of `""` and falls back to
`self.config.tokenizer_path`. This is the rollout engine's tokenizer path (for the
inference model). This isn't really a tree search config. It might make more sense to
keep it as `self.config.tokenizer_path` only.

Let me re-read the original code... Line 743-746:

```python
tokenizer_path = (
    os.getenv("TREE_SEARCH_TOKENIZER_PATH", "")
    or self.config.tokenizer_path
)
```

This means: use TREE_SEARCH_TOKENIZER_PATH env var, and if it's empty, fall back to the
rollout config's tokenizer_path. Since TREE_SEARCH_TOKENIZER_PATH was never set in .env
(it defaulted to ""), the effective value was always `self.config.tokenizer_path`.

So for the migration, we can simply use `self.config.tokenizer_path` directly — no need
to add `tokenizer_path` to TreeBackupConfig. The env var was essentially a dead
override.

Let me update the code in my plan accordingly.

- [ ] **Step 3: Remove unused imports at top of file**

The `from dotenv import load_dotenv` inside the method is removed (it was inside the
tree search block). Check if `import os` is still needed elsewhere — yes, it's used
throughout the file. No import changes at the top level needed.

- [ ] **Step 4: Commit**

```bash
git add areal/infra/remote_inf_engine.py
git commit -m "feat(engine): read tree-search config from workflow_kwargs instead of .env"
```

______________________________________________________________________

### Task 6: Clean up .env

**Files:**

- Modify: `customized_areal/.env`

- [ ] **Step 1: Remove tree search lines from .env**

Remove lines 24-35:

```
# Tree search workflow configuration
# When True, _resolve_workflow wraps with TreeSearchGroupedRolloutWorkflow instead of GroupedRolloutWorkflow
use_TreeSearchGroupedRolloutWorkflow=True
TREE_SEARCH_CHECKPOINT_DIR=customized_areal/tpfc/data/tree_cache
TREE_SEARCH_ADVANTAGE_MODE=TREE
TREE_SEARCH_LOSS_MODE=GRPO
TREE_SEARCH_CACHE_MODE=CROSS_TRAINING
TREE_SEARCH_RL_LOSS_WEIGHT=1.0
TREE_SEARCH_DISTILL_LOSS_WEIGHT=0.005
TREE_SEARCH_DIAGNOSE_MODEL_NAME=qwen/qwen3.5-397b-a17b
TREE_SEARCH_DIAGNOSE_BASE_URL=http://10.254.244.168:8443/service-large-64-1775803465274/llm/v1
TREE_SEARCH_DIAGNOSE_API_KEY=aH6867Z7ppZWqN7NQGsaPhSlMrk68AGQnr18Z66KsSL0864mkjaBP7Tr868pKvnCJcWa0VvHZDHk64MrKG5Z5gWrZGx88mrmAv7pk88p786gFB6XXr6MrtSckKs9QJsr
```

The file should end at line 22 (WORKSPACE_OPENAI_API_BASE).

- [ ] **Step 2: Commit**

```bash
git add customized_areal/.env
git commit -m "chore: remove tree-search config from .env (migrated to YAML configs)"
```

______________________________________________________________________

### Task 7: Verify

**Files:** None (verification only)

- [ ] **Step 1: Run pre-commit hooks**

```bash
pre-commit run --all-files
```

Expected: All hooks pass. Fix any formatting/linting issues.

- [ ] **Step 2: Verify no remaining references to removed fields**

```bash
grep -rn "config\.cache_dir\|config\.cache_mode\|config\.loss_mode" --include="*.py" | grep -v __pycache__
```

Expected: No output (all references migrated).

- [ ] **Step 3: Verify no remaining TREE_SEARCH\_ os.getenv calls**

```bash
grep -rn 'TREE_SEARCH_' --include="*.py" | grep -v __pycache__
```

Expected: No output (all env var reads migrated).

- [ ] **Step 4: Quick syntax check**

```bash
python -c "from customized_areal.tpfc.tpfc_config import TPFCConfig; print('TPFCConfig OK')"
python -c "from customized_areal.tree_search.config import TreeBackupConfig; tc = TreeBackupConfig(); print(f'enabled={tc.enabled}, max_reasoning_tokens={tc.max_reasoning_tokens}')"
```

Expected: Both commands print success messages without errors.
