



tmux new-session -d -s train   'cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml 2>&1 | tee training.log; exec bash'
