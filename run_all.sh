#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="/dfs/share-groups/letrain/zhoujie/le-agent-dev_new/backend"
AREAL_DIR="/dfs/share-groups/letrain/zhoujie/AReaL-main"
SESSION="run_all"

tmux new-session -d -s "$SESSION" -n worker
tmux set -g mouse on

tmux send-keys -t "$SESSION:worker" "cd $BACKEND_DIR && uv run python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 16 --threads 4 2>&1 | tee worker.log" Enter

tmux new-window -t "$SESSION" -n api
tmux send-keys -t "$SESSION:api" "cd $BACKEND_DIR && uv run python api.py 2>&1 | tee api.log" Enter

tmux new-window -t "$SESSION" -n training
tmux send-keys -t "$SESSION:training" "cd $AREAL_DIR && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml 2>&1 | tee training.log" Enter

tmux attach-session -t "$SESSION"
