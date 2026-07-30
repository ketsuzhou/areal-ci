# LEAgent Workspace Setup Guide

## 1. Install uv and Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11.14
```

## 2. Install Redis

```bash
apt-get update && apt-get install -y redis-server 2>&1 | tail -5
```

## 3. Clone LEAgent Repo

```bash
git pull https://gitlab.xpaas.lenovo.com/lrm/leagent
```

## 4. Configure Backend Environment

Contact **andong** to get the configuration for `le-agent-dev/backend/.env`.

## 5. Configure Auto-Start Services

Append the following to `/root/.bashrc`:

```bash
# Auto-start services on interactive shell (run once per session)
if [ -n "$PS1" ] && [ -z "$SERVICES_STARTED" ]; then
    export SERVICES_STARTED=1
    # Start code-server
    if ! pgrep -f "code-server" > /dev/null 2>&1; then
        export PASSWORD=123456
        nohup /dfs/share-read-only/code-server/bin/code-server \
            --auth=password \
            --bind-addr=0.0.0.0:8080 \
            --user-data-dir=/dfs/data/data/ \
            --extensions-dir=/dfs/data/data/ > /dev/null 2>&1 &
        echo "✓ Code-server started on port 8080 (password: 123456)"
    fi
    # Start sing-box proxy
    if [ -f /dfs/share-groups/foundationmodelgroup/LRM/proxy/sing-box.sh ]; then
        /dfs/share-groups/foundationmodelgroup/LRM/proxy/sing-box.sh start
    fi
    # Start Redis (if not already running)
    if ! redis-cli -a lenovo_2025 ping > /dev/null 2>&1; then
        redis-server --daemonize yes
        redis-cli CONFIG SET requirepass "lenovo_2025"
        redis-cli -a lenovo_2025 CONFIG SET maxmemory 10gb
        redis-cli -a lenovo_2025 CONFIG SET maxmemory-policy allkeys-lru
        echo "✓ Redis started (with password, maxmemory 10gb, policy allkeys-lru)"
    fi
fi
```

Then apply:

```bash
source ~/.bashrc
```

## 6. Start Workers and API

```bash
tmux new-session -d -s workers 'cd le-agent-dev/backend && .venv/bin/python -m dramatiq core.agents.worker core.app.workflow.worker core.triggers.worker core.billing.worker --queues agents sub_agents workflows triggers system --processes 32 --threads 2 2>&1 | tee worker.log; exec bash'
tmux new-session -d -s api     'cd le-agent-dev/backend && .venv/bin/python api.py 2>&1 | tee api.log; exec bash'
```
