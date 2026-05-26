



tmux new-session -d -s workers 'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 10 --threads 2 2>&1 | tee worker.log; exec bash'
tmux new-session -d -s api     'cd /dfs/share-groups/letrain/zhoujie/le-agent-dev/backend && .venv/bin/python api.py 2>&1 | tee api.log; exec bash'
tmux new-session -d -s train   'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash'
