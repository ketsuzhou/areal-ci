# RolloutControllerV2 tree search support

## Goal

Add a V2 config for TPFC tree search training, and add tree search
(`TreeSearchGroupedRolloutWorkflow`) support to `RolloutControllerV2._resolve_workflow`.

## Background

`RolloutControllerV2` (`areal/experimental/inference_service/controller/controller.py`)
is a parallel implementation to `RolloutController` that routes inference through a
gateway HTTP stack. It selects via `rollout._version: v2` in the config.

V1's `RemoteInfEngine._resolve_workflow` already supports tree search: when
`group_size > 1` and `tree_search_config.enabled`, it wraps the inner workflow with
`TreeSearchGroupedRolloutWorkflow` instead of `GroupedRolloutWorkflow`. V2's
`_resolve_workflow` is missing this switch — it always uses `GroupedRolloutWorkflow`.

Additionally, V2 passes `workflow_kwargs` (which includes `tree_search_config`) to
`InferenceServiceWorkflow` and agent constructors, neither of which accept that key.

## Changes

### 1. New config file

`customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search_v2.yaml`

Copy of the existing tree search config with two additions under `rollout`:

- `_version: v2`
- `admin_api_key: areal-admin-key`

### 2. Code change

**File:** `areal/experimental/inference_service/controller/controller.py` **Method:**
`RolloutControllerV2._resolve_workflow`

Three edits:

a. Extract `tree_search_config` from `workflow_kwargs` at method start (prevents leakage
into InferenceServiceWorkflow / agent constructors).

b. Replace the `GroupedRolloutWorkflow` wrapping in the online branch with a call to a
new `_wrap_grouped` helper that checks `tree_search_config.enabled`.

c. Same replacement in the agent branch.

### 3. New helper: `_wrap_grouped`

When `tree_search_cfg is not None and tree_search_cfg.enabled`, returns
`TreeSearchGroupedRolloutWorkflow`; otherwise returns `GroupedRolloutWorkflow`. Exact
port of V1 lines 707-745.

### Flow

```
V1: TPFCAgent → OpenAIProxyWorkflow → TreeSearchGroupedRolloutWorkflow
V2: TPFCAgent → InferenceServiceWorkflow → TreeSearchGroupedRolloutWorkflow
```

Both `InferenceServiceWorkflow` and `OpenAIProxyWorkflow` implement
`RolloutWorkflow.arun_episode()` returning
`dict[str, InteractionWithTokenLogpReward] | None`. The wrapping is transparent to
`TreeSearchGroupedRolloutWorkflow`.

## Verification

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main
uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py \
  --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search_v2.yaml
```

Should launch with `RolloutControllerV2` and log "use TreeSearchGroupedRolloutWorkflow"
(from the tree search workflow constructor) when `group_size > 1`.
