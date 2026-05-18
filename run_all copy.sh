#!/bin/bash

  redis-cli -a lenovo_2025 FLUSHALL && cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers
  system --processes 8 --threads 8

  
# Worker - terminal 1
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 32 --threads 2; exec bash'

# API server - terminal 2
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python api.py; exec bash'

# Training - terminal 3
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash'


# Worker - terminal 1
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 8 --threads 8; exec bash'

# API server - terminal 2
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend && .venv/bin/python api.py; exec bash'

# Training - terminal 3
gnome-terminal -- bash -c 'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash' 