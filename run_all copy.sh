#!/bin/bash

  redis-cli -a lenovo_2025 FLUSHALL && cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers
  system --processes 8 --threads 8

  
# Worker - terminal 1
gnome-terminal -- bash -c ' cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 3        --threads 2  2>&1 | tee worker.log  ; exec bash'

# API server - terminal 2
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python api.py  2>&1 | tee api.log  ; exec bash'

# Training - terminal 3
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash'


# Worker - terminal 1
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 16 --threads 4   2>&1 | tee worker.log  ; exec bash'

# API server - terminal 2
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python api.py 2>&1 | tee api.log ; exec bash'

# Training - terminal 3
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash' 


# Training - terminal 3
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search_v2.yaml 2>&1 | tee training.log; exec bash' 

cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config  customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-opd.yaml   2>&1 | tee training_opd.log

cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config  customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B_tree_search_self_play.yaml 2>&1 | tee training_self_play.log


cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config  customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B_self_play_test.yaml 2>&1 | tee training_self_play.log

AREAL_LOG_LEVEL=DEBUG uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config  customized_areal/tree_search/training/configs/config_Qwen3-5L-9B-multica_v2.yaml 2>&1 | tee training_multica.log