# TPFC Training Environment Setup

## 1. Clone the Repository

```bash
git clone https://gitlab.xpaas.lenovo.com/zhoujie22/areal
```

## 2. Copy Pre-built Wheels

Copy the pre-built CUDA/wheel packages into the project. These include custom builds of PyTorch, Triton, SGLang kernels, FlashInfer, and NVIDIA libraries that are required for the training environment.

```bash
cp -r /dfs/share-groups/foundationmodelgroup/LRM/zhoujie/wheels/ AReaL-main/wheels
```

## 3. Install Dependencies

```bash
uv sync
```

## 4. Configure Environment Variables

Copy or create the `.env` file at `customized_areal/.env`. The file must contain the following variables:

```env
# Supabase (database & auth)
SUPABASE_URL=<your-supabase-url>
SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
SUPABASE_DB_CONNECTION=<your-db-connection-string>
SUPABASE_JWT_SECRET=<your-jwt-secret>
SUPABASE_AUTH_EMAIL=<your-auth-email>
SUPABASE_AUTH_PASSWORD=<your-auth-password>

# OpenRouter API
OPENAI_API_KEY=<your-openrouter-key>
OPENAI_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=<your-openrouter-key>

# Workspace LLM API
WORKSPACE_OPENAI_API_KEY=<your-workspace-key>
WORKSPACE_OPENAI_API_BASE=<your-workspace-api-base>

# TPFC Agent
TPFC_USER_ID=<your-user-id>
TPFC_AGENT_ID=<your-agent-id>
REFRESH_TOKEN=<your-refresh-token>
```

See `customized_areal/.env` for the current working values.

## 5. Run Training

### Tree Search Training

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml
```

### OPD Training

```bash
cd /dfs/share-groups/letrain/zhoujie/AReaL-main && uv run customized_areal/tpfc/scripts/train_tpfc_tree_search.py --config customized_areal/tpfc/configs/config_tpfc_Qwen3-5L-9B-opd.yaml 2>&1 | tee training_opd.log
```

## Available Configs

| Config | Description |
|--------|-------------|
| `config_tpfc_Qwen3-5L-9B-Instruct_tree_search.yaml` | Tree search training |
| `config_tpfc_Qwen3-5L-9B-Instruct_tree_search_v2.yaml` | Tree search v2 |
| `config_tpfc_Qwen3-5L-9B-opd.yaml` | OPD training |
| `config_tpfc_Qwen3-VL-8B-Instruct_tree_search.yaml` | VL-8B tree search |
