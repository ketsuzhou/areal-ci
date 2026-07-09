# AReaL v2 代码解读

> 解读对象：`areal/v2/`（约 2.1 万行 Python），AReaL 2.0 的服务化重写。
> 生成时间：2026-07-08。所有 `file:line` 引用均基于该时点的代码。

---

## 一、v2 是什么

`areal/v2` 是 AReaL 2.0 —— 把旧的 `areal/experimental/openai` 代理**单体**拆成了**服务化架构**。它围绕 RL 训练闭环组织成 5 个顶层包，核心思想是：推理、训练、权重同步各自独立成服务，通过 HTTP/RPC 松耦合，本地由 CLI 统一拉起。

```
        ┌─────────────── agent_service ───────────────┐
        │  (agent 层：多轮/工具/session，套在推理之上)    │
        └───────────────────┬─────────────────────────┘
                            │ 调用方自带推理路由 handle
                            ▼
   rollout 产生      ┌─ inference_service ─┐    权重热推      ┌─ weight_update ─┐
  trajectory(token/ ─▶│ /chat/completions   │◀────────────────│  awex/NCCL/disk │
  logprob 捕获)      │  SessionData/trajectory│   新权重        │  FSDP↔SGLang    │
                     └──────────┬───────────┘                 └────────┬────────┘
                                │ trajectory refs                       │
                                ▼                                        │
                     ┌─ training_service ───────────────────────────────┘
                     │  消费 trajectory -> RL 梯度步 (FSDP2/Megatron)
                     └──────────────────────────────────────────────────
          cli：本地拉起并管理以上所有服务的 gateway/router/worker/data_proxy/guard
```

**RL 一次迭代的数据流**：

1. `inference_service` 对外提供 `/chat/completions`，把每个 assistant turn 的 token+logprob 捕获成 `SessionData` 里的一个 interaction；
2. `set_reward` 把 active→ready 关闭成一条 trajectory；`export_trajectories` 返回 `RTensor` 引用（不传字节）；
3. `training_service` 通过 data_proxy 拉取这些 tensor refs，在 worker 上的 `TrainEngine`（FSDP2/Megatron）跑一步 RL；
4. `weight_update` 通过 awex（NCCL P2P）把新权重**原地**推回 `inference_service`，不重启推理引擎；
5. 下一轮 rollout 用新权重。`agent_service` 是可选的上层，把推理包装成带多轮/工具的 agent。

---

## 二、所有服务共用的骨架（`areal/infra/rpc/`）

每个服务的角色划分高度一致，因为都建立在同一套共享基础设施上：

- **Guard**（`infra/rpc/guard/app.py`）：进程 supervisor。Flask app，提供 `/health` `/alloc_ports` `/fork` `/kill_forked_worker` `/set_env` `/configure`。`GuardState` 持有 experiment/trial 身份、已分配端口、fork 出的子进程，并通过 name_resolve（etcd3/NFS）做服务注册。每个服务的 `guard/` 都只是对它的薄封装。
- **rpc_server**（`infra/rpc/rpc_server.py`）= Guard + `data_bp`（数据蓝图，提供 `/data/*` 张量分片存取）+ `engine_bp`（引擎蓝图，暴露 TrainEngine/InferenceEngine RPC）的组合体。训练 worker 直接用这个。
- **RTensor**（`infra/rpc/rtensor.py`）：“远程张量引用”。张量本体留在 data_proxy，只传 `TensorShardInfo(shard_id, node_addr)`，通过 `/data/<shard_id>` `/data/batch` 解析、`/data/clear` 释放。这是 spec 里 `tensor_ref` 机制的实现，保证大张量不进训练进程。

于是每个服务都有同样的角色：**gateway**（对外入口）/ **router**（路由）/ **guard**（进程管理）/ **controller**（编排，是 trainer 侧库类，非长驻进程）/ **data_proxy**（数据/张量）/ **worker**（计算）。

---

## 三、逐服务解读

### 1. `inference_service/`（7.4k 行，核心，rollout 生产者）

**进程拓扑**（由 trainer 侧的 `RolloutControllerV2` `controller/controller.py:65` 编排 fork）：

- **guard**：进程管理，`/fork` 出其余角色。
- **inference backend**：sglang/vllm，按 dp-group 启动，多节点通过 rendezvous 端口协调（`controller.py:544-679`）。
- **router**（`router/app.py:181`）：**推理后端路由**，持有 `WorkerRegistry`/`SessionRegistry`/`ModelRegistry`/`GroupRegistry`（`router/state.py`），把 session 钉到某个 data-proxy worker。不转发流量，只答 `POST /route`。策略只有 `RoundRobinStrategy`，`least_busy` 抛 `NotImplementedError`（`strategies.py:38-43`）。
- **data_proxy**（`data_proxy/app.py:316`）：每个 dp-group 一个，拥有 `SessionStore`、OpenAI 兼容的 `/chat/completions`、RL 控制面（`/rl/start_session` `/rl/set_reward` `/export_trajectories`）、版本/暂停端点，并挂载 `/data/*` 张量存储。
- **gateway**（`gateway/app.py:82`）：薄代理，只持 admin key + router 地址，查 router 拿 worker_addr 后转发。

**核心 RL 数据模型**（`data_proxy/session.py`）：

- `SessionData`（`session.py:119`）：`_active_completions: InteractionCache` + `_ready_trajectories: OrderedDict[int, ReadyTrajectory]`。
- 每次 `/chat/completions` 追加一个 `InteractionWithTokenLogpReward`（token+logprob）（`app.py:643-694`）。
- `set_reward`（`session.py:229-269`）：给某个 interaction 记 reward，然后 `_mark_active_trajectory_ready_locked`（`:186-217`）把 active 关成一条 ready trajectory。**今天 active→ready 只通过 set_reward 触发**。
- `export_trajectories`（`app.py:713-754` → `session.py:310-358`）：按 `trajectory_id` 弹出 ready trajectory，`remove_session=False` 时 session 留活；若有 tensor 数据则 pad 后 `RTensor.remotize(...)` 返回引用，否则拼字符串。**已支持按 trajectory_id + remove_session=False**。
- 引用解析：`GET /data/<shard_id>`、`POST /data/batch`、`DELETE /data/clear`（来自 `data_blueprint.py`）。

**后端集成**：

- sglang 经 `areal_launch_server`（`sglang/launch_server.py`）换入 `areal_run_scheduler_process`（`sglang/scheduler.py:148`）并注册 `/awex/*`；vLLM 用自己的 `build_cmd_from_args`。
- `InfBridge`（`inf_bridge.py:32`）实现 `_AsyncGenerateEngine` 协议，带 **pause→abort→resubmit 循环 + token 累积**（`inf_bridge.py:200-264`）：暂停时睡眠，收到 `abort` 就补 input_ids、缩 `max_new_tokens` 重发，累积 token/logprob/**版本号**。这样权重更新期间能安全暂停生成。
- **awex** = AReaL Weight Exchange。`AwexSchedulerBridge`（`sglang/scheduler.py:20`）用 `setattr` 把 `awex_*` 方法组合到普通 SGLang Scheduler 上（不改继承），`/awex/*` 端点经 `RpcProxy`（ZMQ）派发到 scheduler。这就是**原地权重更新**通道。

**controller**（`controller/controller.py`，1807 行）：`RolloutControllerV2` 是 v1 `RolloutController` 的 duck-type 替代（非子类）。负责服务生命周期、注册、**online callback server**（收 `/callback/online_ready` 唤醒等待者，`:789-932`）、rollout API（submit/wait/rollout_batch）、控制面 fan-out（set_version/pause/offload/onload 并发广播到所有 data_proxy，`:1013-1455`）、workflow 包装（`controller/workflow.py`，离线 start_session→run→set_reward→export，或在线等 ready 回调→export）。

### 2. `training_service/`（3.7k 行，rollout 消费者）

**进程拓扑**（`GatewayTrainController` `controller/controller.py` 编排）：

- guard / worker（Flask，每个 rank 一个 `TrainEngine`，所有引擎调用串行进单线程 Queue，`:32-94`）/ router（FastAPI，`api_key→data_proxy` 注册表 + 健康轮询）/ data_proxy（FastAPI，**无状态**分区 fan-out 分发器）/ gateway（FastAPI 反向代理，按 bearer token 查 router 解析 data_proxy）。
- 注意 web 栈异构：gateway/router/data_proxy 用 FastAPI+uvicorn，worker 用 Flask+`app.run(threaded=True)`。

**数据摄入**：data_proxy **不存 trajectory**，只做 DP 感知的分区分发（`dispatcher.py:59-403`）。trajectory tensor 以 `RTensor` shard 形式留在 worker 上，由 worker 的 `data_bp` 蓝图提供。请求体是 `{"args":[...],"kwargs":{...}}`，args 是 `list[dict]` per-item trajectory。`dispatcher.dispatch()` 按 DP head 切分+合并返回单结果；`broadcast()` 广播到所有 worker。`topology.py` GET 各 worker `/topology` 构建 `WorkerTopology`。

**训练引擎**：**这里没有 RL 数学**。`worker/engine.py` 动态 import 一个 `TrainEngine` 子类，把每个路由委托给 `getattr(engine, method)`（`engine.py:54-111`）。真正的 FSDP2/Megatron step loop 在 `areal/engine/fsdp_engine.py`/`megatron_engine.py`。`GatewayTrainController` 模仿 v1 `TrainController` 接口（train_batch/forward_batch/save/load/...），但 `data_parallel_world_size` 硬编码 1、`cpu_group=None`（`controller.py:904-927`）——真实并行在 worker 引擎里。

**awex**：`worker/awex.py` 在每个 train worker 暴露 `/awex/*`，懒加载 `AwexFSDPAdapter`/`AwexMegatronAdapter`，这是 `WeightUpdateController` 调用的服务端端点。`update_weights`（`controller.py:1018-1034`）驱动权重 delta 回推理，并在前后 pause/continue 生成。

### 3. `weight_update/`（3.4k 行，训练→推理权重桥）

**三种传输模式**（`gateway/config.py:58,73`）：

- **awex**：NCCL P2P send/recv，不落盘（`app.py:602-618`）。
- **disk**：训练存盘，推理读盘（`app.py:620-702`）。
- **colocate**：训练/推理共享 GPU，权重经 CUDA-IPC handle 通过 gateway KV store 传递 + 推理侧独占 NCCL group（`app.py:517-600`）。

**拓扑**：`WeightUpdateController`（`controller/controller.py:23`，training 侧持有）spawn gateway 子进程，驱动 `/connect` `/update_weights` `/disconnect`。Gateway 是 FastAPI，带 `WeightMetaStore`（按 pair_name 的线程安全 KV，`:12`）+ `PairRegistry`。KV GET 故意不鉴权以便 worker 无凭据读 meta（`app.py:778-785`）。

**awex 适配器**（核心难点是参数布局转换）：

- `AwexSGLangAdapter`（推理侧，`sglang_adapter.py:49`）：`_unfuse_params`（`:130-228`）把 SGLang 融合的 `qkv_proj`/`gate_up_proj`/`experts.w13/w2` 拆回 HF 名以匹配训练侧；构建 recv 计划，加入 NCCL group。
- `AwexFSDPAdapter`（训练侧，`fsdp_adapter.py:41`）：提取 DTensor shard meta + 算全局 offset，构建 send ops。
- `AwexMegatronAdapter`（训练侧，`megatron_adapter.py:46`）：支持 DP/TP/PP/EP，经 `get_named_parameters`+`all_gather_param`+`convert_to_hf` 产出 per-expert HF 名。

**NCCL group**（`nccl_group.py`）：`init_custom_process_group` 允许创建额外的**主进程组**（非子组）；`setup_batch_isend_irecv` 做防挂起预热（配对 rank 跑一次小批量 send/recv + barrier + 校验）。

**关键缺口**：`AwexFSDPAdapter` **没有实现 colocate**（尽管 Protocol 要求）——只有 Megatron+SGLang 支持 colocate，这是真实 gap。另外 awex + LoRA 被直接拒绝（`app.py:251-260`）；默认 admin key 是硬编码的 `"areal-admin-key"`。

### 4. `agent_service/`（2.8k 行，推理之上的 agent 层）

对外三个面：WebSocket `/ws` + OpenAI 兼容 `/v1/responses` `/v1/chat/completions`（`gateway/app.py:90-114`）。在 RL 环里位于**推理侧**，不直接接触训练。

**拓扑**：gateway / router（session→proxy 粘性映射 + 轮询）/ data_proxy（持 `sessions` dict + idle reaper）/ worker（无状态，启动时 `import_from_string` 加载一个 `AgentRunnable`）/ guard / controller（库类）。

**请求生命周期**（WebSocket `req`）：gateway 验 token → 生成 run_id → 立即回 `accepted` → 查 router `/route` 拿 data_proxy → POST `/session/{key}/turn` → data_proxy 创建/复用 `_SessionData`，带 `session.history.copy()` 流式转发给 worker `/run` → worker 调 `agent.run(request, emitter=...)`。

- **结构化**（`AgentResponse`）：emitter 缓冲事件，worker 返回 `{summary,metadata,events}`，data_proxy 重建 history，gateway 重发 `delta`/`tool_call` event 帧 + `complete` res 帧。
- **透传**（`StreamResponse`）：worker 设 `x-areal-passthrough:1` 逐字节转发，data_proxy 不留 history（服务 `/v1/chat/completions`）。data_proxy 靠这个 header（而非 Content-Type）区分，避免把非流式 JSON 透传误判为结构化 turn。

**如何调推理**：**不直接调**。整个包没有 runtime import `inference_service`。调用方在自己已有的推理 session 上 mint 一个 `sk-sess-*` key，把 `inf_base_url`+`session_api_key`+`inf_model` 随请求传入（`gateway/bridge.py:109-115`），data_proxy 缓存到 session 并注入 `metadata['areal_inference']`（`data_proxy/app.py:79-98`），agent 实现自己读这个 metadata 路由 LLM 调用。所以 agent_service 是后端无关的。

**坑**：`QueueMode`（COLLECT/FOLLOWUP）和 `idempotencyKey` 被解析并转发，但**没有任何一层实现排队/去重**，实际上 delegated 给 agent 实现 —— 看起来是 deferred。所有状态内存态，重启即失。

### 5. `cli/`（4.1k 行，操作员入口）

`areal` click group 三个子命令：`agent` / `inf` / `train`（`cli.py:22-24`）。

- `areal agent run/stop/status/ps/logs`：一个 agent 服务 = 1 gateway + 1 router + N (worker, data_proxy) 对。
- `areal inf run/stop/status/ps/register/deregister/models/logs`：gateway + router + (sglang/vllm worker, data_proxy) 副本的模型注册表。
- `areal train run`：in-process importlib 调训练 driver，**无生命周期/状态跟踪**。

**进程生命周期**：`NamespacedStateStore`（`state.py:92`）根在 `$AREAL_HOME`（默认 `~/.areal`），按 namespace 存 `services/<svc>.json` 状态 + flock + 日志 + `current-service` 指针，原子写。“running” = gateway PID 存活。`spawn_process`（`process.py:49`）用 `subprocess.Popen(start_new_session=True)` detach（抗 SIGHUP）。launcher 通过 `python -m areal.v2.<service>.<role>` 拉起各角色。`ForegroundWatcher`（`watcher.py`）：SIGTERM→teardown，SIGHUP→detach。

**注意**：CLI 只拉起 gateway/router/worker/data_proxy，**不拉起 guard 或 controller**（controller 是 trainer 侧库类，guard 由 controller 在 trainer 进程里 fork）。

**配置**：`ConfigLoader` 读 `$AREAL_HOME/<namespace>/config.toml`，优先级 CLI flag > --config > config.toml > click 默认。K8s/Slurm scheduler 是 stub，只实现了 local。

---

## 四、当前状态与缺口（重要）

1. **`close_segment` 尚未实现** —— grep 确认 `areal/v2/` 里零出现。这正是 `openspec/changes/multica-v2-segment-dag-training` 设计文档要加的核心：让 active→ready 关闭**与 reward 解耦**，使一个 agent-run session 能切成多条独立可奖励的 trajectory（给 Multica 多智能体环境用）。好消息是**按 trajectory_id 导出 + `remove_session=False` 已经支持**（`session.py:310-358`），`ready_trajectories` 已是 OrderedDict，所以设计 D2 基本就绪，只差无奖励关闭这一步。
2. `least_busy` 路由策略 = `NotImplementedError`。
3. `AwexFSDPAdapter` 无 colocate 支持（真实 gap，Protocol 要求但未实现）。
4. agent_service 的 `QueueMode`/`idempotencyKey` 转发但不执行。
5. CLI 的 K8s/Slurm scheduler 是 stub，只有 local。
6. 默认 admin key 硬编码 `"areal-admin-key"`（全服务），生产必须覆盖。
7. 几个潜在 bug：awex connect 里 `gateway_addr` 绑 `0.0.0.0` 时复用 `master_addr` 可能不可达（`weight_update/gateway/app.py:333`）；training gateway `--router-addr` 默认 `localhost:8081` 而 `RouterConfig.port` 默认 `9081`（`training_service/gateway/__main__.py:19` vs `router/config.py:11`），只因 controller 总显式传参才不出事。

---

## 五、文件速查表

| 关注点 | 文件 |
|---|---|
| 共享进程管理 | `areal/infra/rpc/guard/app.py` |
| RTensor 远程张量 | `areal/infra/rpc/rtensor.py` |
| 组合 RPC server | `areal/infra/rpc/rpc_server.py` |
| 推理 session/trajectory 核心 | `inference_service/data_proxy/session.py` |
| 推理编排（1807 行） | `inference_service/controller/controller.py` |
| 推理暂停/重发循环 | `inference_service/inf_bridge.py` |
| SGLang awex 注入 | `inference_service/sglang/scheduler.py` |
| 训练 DP 分发 | `training_service/data_proxy/dispatcher.py` |
| 训练 worker 引擎 | `training_service/worker/{app,engine,awex}.py` |
| 权重传输网关 | `weight_update/gateway/app.py` |
| FSDP/Megatron/SGLang 适配器 | `weight_update/awex/{fsdp,megatron,sglang}_adapter.py` |
| NCCL group | `weight_update/nccl_group.py` |
| Agent WebSocket 协议 | `agent_service/protocol.py` |
| CLI 状态/生命周期 | `cli/state.py` `cli/lifecycle.py` `cli/process.py` |

---

## 六、相关设计文档

- `openspec/changes/multica-v2-segment-dag-training/` — `close_segment` + per-segment tensor-ref export + `AssembledDag` 的设计与任务（在建）。
- `openspec/changes/sub-project-g-multica-interaction-dag/` — 已被上一条 supersede 的旧 turn-index 方案（已归档）。
- `openspec/changes/env-dispatch-sandbox-lifecycle/` — env-dispatch 轮询端点与 sandbox 生命周期（与 Multica 集成相关）。
