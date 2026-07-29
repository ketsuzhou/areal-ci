# 真 AReaL 在线训练 · segment 产出链路交接

> 更新时间:2026-07-29 13:10 (UTC+8) 目标:真 AReaL 在线训练端到端跑通,`interaction_dag_segment`
> 表出现数据,触发一次训练。

参照skill: .areal/.agents/skills/debug-real-areal-training/SKILL.md

## 一句话现状

**数据链路已全线打通,训练能吃到真轨迹并完成权重更新;唯一剩下的缺口是奖励进不去, 导致更新是零梯度空转。** 第四层(`export_trajectories`
通道缺失)已合并部署;第五层 ("segment 是空壳")已随 `store: true` cherry-pick + 命令钉住训练容器一并解决。 2026-07-29
的验收跑出了 52 条 `tensor_ref` 齐全的 segment、两次训练步、权重 v1→v2, 但
`grad_norm=0`。根因与修法见文末「验收结论(2026-07-29)」和「下一个任务」。

## 已修复的三层(按发现顺序)

### 1. T10 idle sweep 判定不成立 —— 已修,已上线

PR #1300。用实时数据验证过:现在每个 project 的 `active` 都返回 0,判定正确。 但当时没解决问题,因为它够不着(见第 2 层)。

### 2. chat-session 任务跳过所有终结 seam —— 已修,已上线

**根因**:rollout agent 全部跑在 chat session 里,daemon inbox handler 对 chat-session 投递是自己 ack
的,不走 `completeTask`。而 segment 闭合、RL 路由 (`RouteTerminalTrainingTask`)、临时沙箱回收三件事全挂在
`completeTask` 上。 于是每个 rollout 任务终结时一个 segment 都不写,AReaL 侧 `get_dag` 返回 200 但 segments
为空,episode 死在 `assembled DAG must contain at least one segment`。 反复出现的孤儿沙箱堆积也是同一个缺口。

**证据**(45 分钟窗口):38 个 `interaction_dag_session_run`、**0 个 segment**; 38/38 有
`chat_session_id`、0/38 有 `issue_id`、38/38 带 `areal_proxy` 上下文; `task completed` 日志 **0
条**。

**修复**:PR #1325,抽出 `FinalizeTerminalTaskSideEffects`,在 chat-session 的
完成和失败路径上也调用。已合并,线上镜像 `ghcr.io/lrm-teams/multica-backend:sha-e8927b2`。

> 注意:我的合并提交 `bd7206e` 的 Deploy 被后续提交的并发组取消了,是后来的 `e8927b2` 把它带上线的。以后合并后要确认**实际部署的 sha
> 是否包含你的提交**, 用 `git merge-base --is-ancestor <你的 sha> <线上 sha>` 判断。

### 3. `store: false` 让交互不被记录 —— 已修,但只推了分支

**根因**:pi 用 OpenAI JS SDK 发请求时带 `store: false`,而
`areal/experimental/openai/client.py:641` 的缓存门是 `if is_omitted(store) or store:` —— 显式
false 就跳过记录,但补全照样返回 200。于是 session 里永远没有 interaction, `close_segment` 必然报 400
`{"detail":"No interactions in session"}`。

**修复**:`areal/v2/inference_service/data_proxy/app.py`,只要 bearer token 解析出了 RL session
就强制 `kwargs["store"] = True`。附带单元测试
`test_chat_completions_session_overrides_store_false`。

**状态**:

- 提交 `00b9fdcf`,已推到 `origin/fix/session-forces-store`(gitlab zhoujie22/areal)。 **尚未合并到
  master,也未开 MR。**
- GPU 机的训练 checkout 上是**就地打的补丁**(不是 git 拉的),
  `/dfs/share-groups/letrain/zhoujie/multica-intergrate/areal/v2/inference_service/data_proxy/app.py:697`。
  那台机器的 `.venv` 被 `uv sync` 清过包,同理这个未跟踪的改动也可能在下次同步时丢掉。

**验证**:用带 `store: false` 的探针走完整桥路,`close_segment` 返回 `interaction_count: 1` ——
修复前这个形状必然报 "No interactions"。

## 当前卡点(第 4 层):`export_trajectories` 通道缺失

`CloseSegmentForEvent`(`server/internal/service/interaction_dag.go:185`)的顺序是 close →
**export** → 写 segment 行。export 失败就不写行:

```
interaction_dag: export_trajectory <session>/0:
  arealrl: export_trajectories returned status 404: {"detail":"Not Found"}
```

这是裸的 FastAPI 404,即**路由不存在**,不是业务错误。定位结论:

- AReaL 网关**有**这个路由:`curl -X POST 127.0.0.1:17727/export_trajectories` 直连返回 200。
- multica 侧 stub(`ghcr.io/lrm-teams/multica-db-bridge:sha-2d507d7`)的 openapi 里 只有
  `/chat/completions`、`/responses`、`/rl/{start_session,set_reward,end_session,close_segment}`
  —— **没有 `/export_trajectories`**。
- `db_bridge/channels.py` 的 `CHANNELS` 注册表里根本没有这个通道。 协议文档把 `export_trajectories` 列为
  db_bridge 应有的能力,只是一直没实现。

> 副作用值得注意:`close_segment` 现在是**成功**的,它会消费掉活跃 segment, 随后 export 失败,这段轨迹就丢了,重试也捞不回来。

## 第四层的修复(已提交,PR #1336)

代码在 **`/tmp/multica-seam`**,multica 仓库的 worktree,分支
`feat/export-trajectories-channel`,提交 `081b353a2`,已 rebase 到 `upstream/dev` 的
`3b48175b2` 并推到 LRM-Teams。PR:https://github.com/LRM-Teams/multica/pull/1336 (base
`dev`)。三个文件的改动:

| 文件                                  | 改动                                                                                                                       |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `db_bridge/channels.py`               | 新增 `export_trajectories` Channel(group=`gateway`, POST `/export_trajectories`, kind=`json`, timeout 120s, concurrency 4) |
| `db_bridge/schema.sql`                | 表数组里加 `rpc_export_trajectories`                                                                                       |
| `db_bridge/tests/test_entrypoints.py` | `stub_channels("multica")` 的期望集合加上新通道                                                                            |

stub 和 executor 都是按 `CHANNELS` 自动派生路由/轮询表的(`stub_server.py:709` 的
`stub_channels(side)`、`executor.py:471` 的 `executor_channels(side)`),
所以加一条定义就够,不用改这两个文件。

`schema.sql` 是幂等的(`create table if not exists` 驱动数组),重跑只建新表。

**测试状态**:`216 passed, 1 failed, 1 skipped`。唯一失败是
`tests/test_config.py::test_minimal_from_env_applies_defaults` (`poll_interval_s` 得到
0.075 而非 1.0)。**已确认与本改动无关**:在干净的 `dev`
checkout(`/workspaces/leagent/backend/areal/multica/db_bridge`)上以同样方式跑, 失败一模一样。是上游把
`_DEFAULT_POLL_INTERVAL` 改成 0.075 后没同步这个断言, 值得单独开一个小 PR 修。不必再 stash 复验:

```bash
cd /workspaces/leagent/backend/areal/multica/db_bridge && \
  env -i PATH="$PATH" HOME="$HOME" ./.venv/bin/python -m pytest tests/test_config.py -q
```

## 剩余步骤

1. ~~提交并推 `feat/export-trajectories-channel`,开 PR 到 `dev`,合并。~~ **已完成**: CI 全绿(backend
   11:09Z / frontend 11:13Z)后,经 zhoujie22 授权 squash 合并, 合并提交 **`dc8a0e29`** 在 `dev`
   上,Deploy workflow (run 30355002324)已触发。
1. ~~把 `db_bridge/schema.sql` 应用到共享 Supabase,建出 `rpc_export_trajectories` 表。~~
   **已完成**(见下节"Supabase 建表")。
1. ~~等镜像部署到 `101.200.210.144`,确认 stub 的 openapi 里出现 `/export_trajectories`。~~
   **已完成**:Deploy run 30355002324 success,线上两个容器都是 `sha-dc8a0e2`
   (`multica-db-bridge-stub-multica-1` / `multica-backend-1`),stub openapi 9 条路由里 已有
   `/export_trajectories`。
1. ~~在 GPU 机上重启 `run_executor --side areal`。~~ **已完成**,细节见下节「executor 重启」。
   日志确认:`executor ready side=areal channels=[... 'export_trajectories'] workers=52`。
1. ~~把 AReaL 的 `fix/session-forces-store` 并进 GPU checkout 的正式分支。~~ **已完成**,
   但**没有整分支合**:该分支与当前 `master` 双向分叉 66 文件 / 9693 插入,合过来会拖进 大量无关改动。改为 **cherry-pick
   `00b9fdcf` → `41b0b992`** 落在 `master` 上, 正好带上 `app.py` 那 8 行和回归测试
   `tests/v2/inference_service/test_data_proxy_chat.py`。 先核对过 worktree 里的就地补丁与提交里的 hunk
   **逐字节一致**,才 `git checkout --` 丢弃 再 cherry-pick。app.py 现在是**跟踪状态**,不再怕被 checkout/reset
   清掉。
   > `master` 相对 `origin/master` 是 1 ahead / 10 behind,**没有 push**(推之前要先处理那 10 个提交)。
1. ~~先 `pkill -f half_gpu.py`,再重启训练。~~ **训练已启动**(2026-07-28 19:47 +08),
   `TRAIN_ID=exporttraj-1785239239`,启动前两块卡都清到 4 MiB / 0%,`HALFGPU_CLEARED`。
   **待验**:`interaction_dag_segment` 是否出现新行(基线:总 16 条,最新一条是 7-24)。
   保活占卡只在空闲期开,训练一启动就要让开(已写进 skill 的「GPU 保活」一节, 启动脚本的两个 pkill 循环也加上了 `half_gpu.py` 模式)。

## executor 重启(2026-07-28,踩过的坑都在这)

GPU 机的 executor 跑在 `/dfs/share-groups/letrain/zhoujie/multica-intergrate/multica` 这份
checkout 上,由 `db_bridge/run_dev.sh` 监管。**三个必须知道的约束**:

1. **`run_dev.sh` 同时监管 shell runner 和 executor**,而且有 `trap cleanup INT TERM EXIT`
   - `wait -n`——杀掉 executor 会让 `wait -n` 返回,cleanup 顺带把 shell runner 也杀了。 而 `rsh.py` 的远程
     shell 正是走那个 runner,**等于自断通道**。所以重启必须写成脚本文件、 用 `setsid nohup` 脱离后再执行,并在开头 `sleep 6`
     让触发它的那个 rsh job 先回报完。
1. **`.env.areal` 里没有 `UV_NO_SYNC`**,而 `run_dev.sh` 用 `uv run` 起进程——直接重启会触发 同步,把 venv
   里不在 lockfile 的 `supabase` 清掉。重启前 `export UV_NO_SYNC=1`。
1. 这份 checkout 有**本地独有的**未提交改动 `db_bridge/remote_shell.py` +
   `tests/test_remote_shell.py`(远程 shell 的退避重试,上游 `dev` 里并没有)。 **弄丢就再也连不上 GPU 机**。更新前先
   `cp -a` 备份,更新后 `cmp -s` 不一致就还原。

本次是纯 FF:HEAD 是 `origin/dev` 的严格祖先(0 ahead / 141 behind),且那两个本地改动的文件 在 HEAD
与目标之间**字节一致**,FF 不会碰它们,所以补丁自然存活(日志里没有 `RESTORED_` 行)。 FF 到
`4f646c780`,`CHANNEL_HITS=2`,`SUPABASE_OK`。脚本留在 GPU 机 `/tmp/restart_exec.sh`, 日志
`/tmp/exec_restart.log`。

> 另注:`dev` 上 **#1332 是在 #1336 之后合的**,所以 `dev` 头是 `4f646c780` 而不是我的 `dc8a0e29`。PR 号顺序 ≠
> 合并顺序,判断"我的提交在不在线上"只能用 `git merge-base --is-ancestor`,不能比 PR 号。

## Supabase 建表(已完成 2026-07-28)

`ssh root@82.157.184.89`(免密可用),Supabase 是自建 compose,库容器
`supabase_db_leagent-supabase`,`psql -U postgres -d postgres`。

**不要整份重跑 `schema.sql`**。文件头注释说它幂等,但建表循环里的
`create policy allow_service_role`(`schema.sql:145`)**没有 if-not-exists 守卫**
——循环会遍历所有表,跑到第一张已存在的表就报 `policy already exists` 整份回滚。 (`bridge_stream_chunks`
那段有守卫,建表循环这段漏了。)值得单独开个小 PR 修。

实际做法:把建表循环的 `tables` 数组缩成只有新表,DDL 完全复用原文,不动任何函数 (`bridge_claim_next` 等都是 table-agnostic
的,加通道不需要改函数):

```bash
cd /tmp/multica-seam/db_bridge   # 生成脚本见 /tmp/add_export_trajectories.sql
scp /tmp/add_export_trajectories.sql root@82.157.184.89:/tmp/
ssh root@82.157.184.89 'docker cp /tmp/add_export_trajectories.sql supabase_db_leagent-supabase:/tmp/add.sql \
  && docker exec supabase_db_leagent-supabase psql -U postgres -d postgres -v ON_ERROR_STOP=1 -f /tmp/add.sql'
# PostgREST 的 schema 缓存要手动踢一下,否则 REST 上看不到新表
ssh root@82.157.184.89 "docker exec supabase_db_leagent-supabase psql -U postgres -d postgres -c \"notify pgrst, 'reload schema';\""
```

验证结果(全绿):

- 列/类型/默认值与 `rpc_rl_close_segment` **逐列一致**(双向 `except` 都为空)。
- RLS 已启用,`allow_service_role` policy 存在,`service_role` 有全权限。
- 两个索引 `_user_status_created_idx` / `_status_completed_idx` 都在。
- PostgREST 可见:`GET /rest/v1/rpc_export_trajectories?select=id&limit=1` → 200(25ms)。
- `POST /rest/v1/rpc/bridge_claim_next` 带 `p_table=rpc_export_trajectories` → 200 `null`
  (无待认领行,函数在新表上工作正常)。

> 另注:库里还有一张 `rpc_agent_start_branch`,当前 `dev` 的 `schema.sql` 并不创建它,
> 是别的分支带外应用的。本次只新增表,没有碰它。 主 worktree 的 `db_bridge/.env.areal` 里 **`BRIDGE_USER_ID`
> 是空的**,手工探针要显式写 `c88b46a8-a8b5-4414-82fb-91398bd063b4`。

## 环境速查

| 组件               | 位置                                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| AReaL GPU 机       | 通过 `python3 /tmp/rsh.py run <tag> <timeout> '<cmd>'` 远程执行                                       |
| 训练 checkout      | `/dfs/share-groups/letrain/zhoujie/multica-intergrate`                                                |
| 启动脚本           | GPU 机 `/tmp/run_train_multica.sh`(改 `TRAIN_ID` 后重跑)                                              |
| inference gateway  | GPU 机 `127.0.0.1:17727`(`AREAL_GATEWAY_FIXED_BIND=0.0.0.0:17727`)                                    |
| multica 控制面     | `ssh zhoujie22@101.200.210.144`,库 `docker exec -i multica-postgres-1 psql -U multica -d multica_dev` |
| 沙箱侧 stub / cube | `10.110.158.143:9100` / `:3000`                                                                       |
| Supabase           | `82.157.184.89:54321`,凭据在 `multica/db_bridge/.env.areal`                                           |
| AReaL admin key    | GPU 机 `customized_areal/.env` 的 `AREAL_ADMIN_API_KEY`                                               |
| multica worktree   | `/tmp/multica-seam`(分支 `feat/export-trajectories-channel`)                                          |

`rsh.py` 需要先 `set -a && source multica/db_bridge/.env.areal && set +a`, `BRIDGE_USER_ID`
必须和 executor 的一致(`c88b46a8-a8b5-4414-82fb-91398bd063b4`), 否则行没人认领,探针会挂到超时。

### 队列上有两个 runner,发命令必须钉住目标

同一台物理机上跑着两个容器,共享 `/dfs` 和同一对 GPU,但网络/PID 命名空间独立:

| `runner_id`         | checkout             | IP              | `BRIDGE_USER_ID` | 角色         |
| ------------------- | -------------------- | --------------- | ---------------- | ------------ |
| `areal-box-1`       | `multica-intergrate` | `172.18.253.84` | `c88b46a8…`      | **训练机**   |
| `areal-box-leagent` | `le-agent-dev_new`   | `172.18.217.97` | `ae36de93…`      | leagent stub |

因为 `127.0.0.1:17727` 在两边指向不同进程,命令落错容器就会出现"训练在 A、data proxy 在 B",shard 一律 404(这正是之前
`KeyError: shard ... not found (field 'rewards')` 的 成因)。`areal_shell_claim_next` 现在按
`target_runner_id` 过滤,发命令时钉死目标:

```bash
export RSH_TARGET_RUNNER=areal-box-1
```

不钉 = 两个 runner 抢单,谁先 `for update skip locked` 抢到谁执行。只改 `AREAL_REMOTE_SHELL_RUNNER_ID`
没用:它只是认领后写回的归属标记,不参与筛选。

## GPU 保活(排查间隙必须做)

AReaL 那个 workspace **一小时没有 GPU 使用就会被平台下线**。训练停着排查时,必须挂上 `customized_areal/half_gpu.py`
占卡(约 100 MB / ~10% util):

```bash
cd /workspaces/leagent/backend/areal/multica/db_bridge
set -a && . .env.areal && set +a
python3 /tmp/rsh.py run halfgpu 180 'REPO=/dfs/share-groups/letrain/zhoujie/multica-intergrate
cd "$REPO" || exit 1
for g in 0 1; do
  setsid nohup "$REPO/.venv/bin/python" customized_areal/half_gpu.py --gpu $g \
    > /tmp/half_gpu_$g.log 2>&1 < /dev/null &
done'
```

用仓库 `.venv/bin/python` 直接起,别用 `uv run`(会触发同步,把 venv 里不在 lockfile 的包清掉)。

> **重启训练前必须先杀掉它**:启动脚本的 pkill 模式列表里**没有** `half_gpu`, 不手动清就会一直占着显存和 ~10%
> 算力跟训练抢。`pkill -f half_gpu.py`。

## 验收结论(2026-07-28 20:25 +08):第四层修好了,卡到第五层

**先纠正一件事**:`interaction_dag_segment` 在 **multica 控制面库** (`ssh zhoujie22@101.200.210.144`
→ `docker exec -i multica-postgres-1 psql -U multica -d multica_dev`),**不在**
`82.157.184.89` 那个 Supabase。后者只有 `rpc_*` 桥表。

### 第四层已验证通过

| 指标                             | 修复前        | 现在                                                 |
| -------------------------------- | ------------- | ---------------------------------------------------- |
| `interaction_dag_segment` 总数   | 16(最新 7-24) | **116**,新增 100 行在 11:53:50–12:11:59Z             |
| 同窗口 `session_run` : `segment` | 38 : 0        | **100 : 100(1:1)**                                   |
| `trajectory_id` / `tensor_ref`   | —             | 100/100 都有,6 个 shard 指向 `172.18.24.175:16822`   |
| `export_trajectories`            | 404           | 正常,`trainable=t`、`trajectory_source=areal_tensor` |

### 第五层(当时的卡点):segment 行是**空的**

> **2026-07-29 更正**:本节把 `start_seq=1 / end_seq=0` 当成"空壳"的证据,**这个判据是错的**,
> 后来白白误导了一轮排查。这两个字段是 `task_message.seq` 的回合区间
> (`endSeq = GetMaxTaskMessageSeq(agent_run_id)`),而 proxy 驱动的 rollout 根本不往 multica 的
> `task_message` 写行,所以 `areal_tensor` 源的**每一行**都是 `[1,0]`;同一处代码 还把 `trajectory` 硬编码成
> `[]`,载荷全在 `tensor_ref`。AReaL 侧对这两个字段的引用数是 **0**。 判空壳要看 `tensor_ref` 的六个字段是否齐全 + 训练日志的
> `Packed tree ... microbatch #tokens`。 当时真正的信号是同期那条
> `set_reward 400 No interactions in session`(见「下一个任务」)。

100/100 都是 `start_seq=1 / end_seq=0`。 (基线 16 行里有 6 行是真区间:end_seq 7 / 13 / 17 / 128 /
316。)

所以 AReaL 照旧死在 `assembled DAG must contain at least one segment`,训练日志里 **165
次**(19:53–20:12 +08)。后端同期每 20 秒一条
`training: idle sweep failed ... set_reward 400 {"detail":"No interactions in session"}`。
两边指向同一件事:**RL session 里没有 interaction**。

**已排除 layer-3 复发**:GPU 机 `app.py:697` 就是 `kwargs["store"] = True`,而 data_proxy 进程 19:51
(+08) 才起(在 cherry-pick 之后),**不是在跑旧代码**。所以下一步要查的是: rollout agent 到底有没有经 proxy 发补全、它的
bearer token 能不能解析出 RL session。

### 另一件独立的事:env dispatch 从 20:12 (+08) 开始挂

`create_env_dispatch` **40 次 500 → 20 次 502**。

- **502 段是部署造成的**:控制面容器现在是 `sha-2c609e6`(backend 12:15:57Z 重启), 502 空 body 就是 caddy
  打不通后端。**我的修复活着**: `git merge-base --is-ancestor dc8a0e29 2c609e6` 通过,stub openapi 里
  `/export_trajectories` 仍在。
- **20:12 那批 500 在重启之前**,更像本文档警告过的**孤儿沙箱把 cube 池撑爆**。

### GPU 上堆了一批僵尸进程

两卡 **0% util**,但常驻 41958 MiB / 72298 MiB。原因是堆了 **5 个 `run_train_multica.sh` 壳** + **4 个旧
`weight_update.gateway`** (17:16 / 17:33 / 18:10 / 18:34 起的,端口 11717 / 20475 / 17909 /
19968)。 现有 pkill 模式覆盖不到它们,**下次启动训练前要一并清掉**,否则白占显存。

## 当前运行状态

合并 + 部署:PR #1336 squash 合进 dev,合并提交 dc8a0e29,Deploy 跑绿,线上两个容器都是 sha-dc8a0e2,stub openapi
里 /export_trajectories 出来了。

half_gpu 规则已入 skill:写到 debug-real-areal-training/SKILL.md,新增「GPU 保活(只在空闲期占卡)」一节 + 启动脚本两个
pkill 循环都补了 half_gpu.py,判断口径是"训练在跑就不该有 half_gpu,训练停着就该有"。

executor:GPU 机 FF 到 4f646c780 重启完,日志里 channels=\[... 'export_trajectories'\]
workers=52,通道认了。

AReaL 那个补丁:原计划整分支合会拖进 66 文件无关改动,改成 cherry-pick 00b9fdcf(只有 app.py 那 8 行 + 回归测试)。master
现在 1 ahead,没 push,要不要推你说一声。

训练:19:47+08 起来了,TRAIN_ID=exporttraj-1785239239,启动前两卡已清到 4 MiB(half_gpu
按新规则停了)。interaction_dag_segment 验收基线是 16 条、最新一条 7-24,排了 12:14Z
复查有没有新行进来——那才是这条链路真正打通的证据。 顺带回你 13:13 那个问题:链路没通,一个 step 都没跑成。TRAIN_ID
exporttraj-1785239239 从 19:53+08 开始每个 episode 都失败,assembled DAG must contain at least
one segment 1566 次,外加 create_env_dispatch failed status=500/502 60 次。export_trajectories
这个通道本身是通的(表里有 done 的行),但 segment 还是空的。两块卡 41GB/72GB 常驻、util 0%。

这里有个反直觉的地方:训练在空转、GPU util 长期 0%,平台照样会判"一小时没有 GPU 使用"。所以 workspace 很可能就是这么被下线的——而
half_gpu 恰恰因为"训练在跑"被停了。这条我记下来了,保活口径下次改成看卡上有没有真实算力使用,而不是看训练进程在不在

## 三步分诊(segment 为空时)

1. **sweep 前提成不成立** —— 按 project 统计非终态任务数,`active` 应为 0:

```sql
select r.project_id, count(*) filter (where t.status not in ('acked','suppressed')) as active
from interaction_dag_session_run r
left join agent_inbox_event t on t.id = r.agent_run_id::uuid
where r.created_at > now() - interval '40 minutes' group by 1;
```

2. **seam 有没有被调用** ——
   `docker logs --since 45m multica-backend-1 | grep -c "task completed"`, 以及有没有
   `interaction_dag:` / `training:` 开头的 WARN。**完全没有日志**意味着
   代码没被执行,和"判定条件不成立"症状一模一样但根因完全不同。
1. **错误在哪一层** —— 按 `close_segment` → `export_trajectory` → 写行的顺序看报错前进到哪一步。

完整的排查手册见 `.agents/skills/debug-real-areal-training/SKILL.md`。

## 验收结论(2026-07-29 13:00 +08):数据链路通了,卡在奖励

本轮跑在训练容器 `areal-box-1`(命令用 `RSH_TARGET_RUNNER` 钉住,见「队列上有两个
runner」),`/tmp/train_multica.out` 是主日志(`training_multica.log` 只有 tee 的少量内容)。

| 验收项           | 结果                      | 证据                                                                                                                                                                                                                              |
| ---------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 收集两条完整轨迹 | **达成**                  | 52 条 `areal_tensor` segment,`tensor_ref` 六字段全齐且各有独立 shard;`node_addr` 全部是本 run 自己的 data proxy `172.18.253.84:31959`(无跨容器污染);训练实际消费 `Packed tree #microbatch: 2, microbatch #tokens: [25119, 25079]` |
| 触发一次参数更新 | **机制通了,但数值是空转** | 两次训练步 + `Weight update` v1→v2 + shard 写入 + 推理端重载都发生了,可 `grad_norm=0`、`advantages` 的 avg/max/min 全 0、`actor_loss=0`                                                                                           |

零梯度的因果链(已逐环坐实):

`idle-sweep` 先调 `CloseSegmentForEvent`(其内部第 (c) 步 `ExportTrajectory`,而
`arealSessionCloser` 的注释自己写着"AReaL v2 reclaims the session itself once the trajectory is
exported")→ session 被回收 → 随后的 `SetReward` 撞 400 `No interactions in session`(后端 8/8
次全失败)→ AReaL 侧 58 次 `does not have a reward set` → 组内奖励全 None(`final_reward` avg -0.035
/ max 0)→ GRPO 组内归一化后优势恒 0 → 梯度 0 → **权重版本涨了但参数没动**。

> 之前把这个 `set_reward` 顺序问题判为"非阻塞"是错的:它不阻塞 segment 落库, 但它恰好阻塞了唯一的学习信号。

## 下一个任务:修 idle-sweep 的 reward/close 顺序

改 `multica/server/internal/service/training.go` 的 `maybeSweepIdleTrainingSessions`: 把
`reward` 计算(现 ~537-540)和 `deps.Closer.SetReward`(现 ~541)整段**移到**
`deps.DAG.CloseSegmentForEvent`(现 ~525)**之前**。`reward` 只依赖 `task.Status` 和
`task.TerminalOutcome`,不依赖关段结果,没有依赖阻碍。另一条路径 `maybeCloseTrainingSession` 只调
`SetReward`、不关 segment,顺序本来就是对的,不用动。

这是 Go 改动,要重建并重启 `101.200.210.144` 上的 `multica-backend-1` 才生效。

**重跑前的前置条件**:cube 沙箱池会被上一轮泄漏的行占住(本轮 26 分钟就攒了 60 个, `busy=60` 而
`max_concurrency=1`)。按「沙箱池」一节先按 `startedAt` 删自己的沙箱、 再把 `sandbox_instance` 的陈旧行标成
`failed`,否则 env dispatch 直接 500。

**收尾状态(已完成)**:训练进程已全部停掉,两卡 0 MiB / 0% util、compute apps 归零, 只剩两个不占资源的 `<defunct>` 条目。

## 沙箱泄漏(2026-07-29 已修 + 已清理)

每一轮 rollout 都会泄漏**五种**资源:cube 沙箱、`sandbox_instance` 行、`agent_runtime` 行、env-dispatch
克隆出来的派生 agent(`env-<uuid>`)、以及 `environment_agent_sandbox` 绑定行。上一轮 26 分钟就攒出 909 个沙箱、1036
个 runtime。**三个互相独立的缺陷,缺一个都清不干净**,现已全部修掉:

1. **AReaL 侧从不释放 dispatch。** `multi_agent_workflow.arun_episode` 调了
   `create_env_dispatch` 却从不调 `cleanup_env_dispatch`,DagTimeout / 空 DAG / 成功 三条 return
   路径全泄漏。已改成 `try/finally` + 永不抛出的 `_release_dispatch`。
1. **就算调了,那个端点的回收段是死代码。** `deleteEnvDispatchChannelRollout` 先 `markDeleting` 把绑定从
   `ready` 刷成 `deleting`,紧接着回收循环的条件是 `if b.Status != "ready" { continue }` ——
   整段回收从不执行;函数随后删掉 channel/绑定/env,于是沙箱、runtime、派生 agent 变成再也找不到的孤儿。已让循环同时接受
   `deleting`,并补上"硬删已归档 agent 再删 runtime"(否则 `agent.runtime_id` 的 RESTRICT 让 runtime
   永远删不掉,而那个错误原本被 `_ =` 吞掉,所以泄漏一直不可见)。
1. **循环真跑起来后,删沙箱这步又必然失败。** 同一函数把 `lifecycle.Delete` 的 actor 硬编码成 `""`,而
   `sandbox_job.initiator_user_id` 是 NOT NULL,入队必报 `parse actor_user_id`; `Delete`
   只在**节点不可用**时才回退到强删,节点在线时 cube 沙箱原样留着。已把 `DeleteEnvDispatchChannel` 里本来就被 `_` 丢掉的
   `requireUserID` 结果接到这一步。

原有测试用的是 `pending` 绑定,压根进不了那个循环,所以前两个缺陷从没被发现;第三个是新回归测试
`TestChannelCleanupReclaimsReadyBindingOwnedResources`(用 `ready` 绑定)跑起来后才暴露的。

### 同类缺陷:通用终结钩子也删不掉沙箱(已修)

`ephemeralSandboxCleanerAdapter.DeleteSandboxInstance`(`env_sandbox_lifecycle_adapter.go`)同样传
`""` 当 actor,所以 `cleanup_on_terminal` 那条通用路径在节点在线时也一样删不掉沙箱。AReaL 不走这条(env-dispatch
显式把该钩子关掉了),但其它临时沙箱任务都走 —— 泄漏面比训练链路更大。

actor 只能就地解出来:调用点 `maybeCleanupEphemeralSandbox` 拿到的 `db.AgentInboxEvent`
**没有任何用户字段**。好在该适配器为了拿 node/template 本来就要先调一次 `GetSandboxInstanceRef`,而
`sandbox_instance.creator_user_id` 是 NOT NULL 且已被填进
`SandboxInstanceRef.CreatorUserID`,所以改用**沙箱创建者**当 actor 即可 —— 不动接口、不加迁移、不多查一次库。回归测试:
`TestEphemeralSandboxCleanerEnqueuesDeleteWithSandboxCreatorAsInitiator`。

### 另一个未修缺陷(清理时必须知道)

沙箱内 daemon 注册出来的 runtime,`owner_id` 是 **NULL**,而 `canDeleteRuntime` 要求
`OwnerID.Valid && owner == 调用者`(没有 owner/admin 兜底)。 **这类 runtime 永远无法通过 API 删除**,只能走
SQL。

### 清理手法(顺序错了会白干)

**必须先杀沙箱再删 runtime。** 沙箱活着时 daemon 会在 runtime 被删后立刻重新注册:上一轮删完 1036 个,60 秒内又冒出 124
个,数量正好等于当时存活的沙箱数。

1. 沙箱:`DELETE /api/sandboxes/{id}?workspace_slug=...`。节点在线时入队真正的 sandboxd `delete`
   job(返回 **202**);节点不可达时只强删行(返回 **204**),cube 里的沙箱会残留 ——
   靠状态码区分这两种结果。`sandbox_job.instance_id` 是 `ON DELETE CASCADE`,所以 job 完成后 job
   行随实例行一起消失,"查不到 delete job"不等于没执行。
1. 绑定:`DELETE /api/v1/env-dispatch/channels/{channel_id}` 清 `ready` 绑定。不做这步, 删 runtime
   会因 `environment_agent_sandbox.runtime_id` 的 `ON DELETE SET NULL` 撞该表
   CHECK(`status='ready'` 要求三个句柄都非空)而报 500。
1. runtime:先试 `DELETE /api/runtimes/{id}`,拿到 409 `runtime_has_active_agents` 再带
   `expected_active_agent_ids` 调 `POST /api/runtimes/{id}/archive-agents-and-delete`。
1. 收尾:`owner_id IS NULL` 的重注册孤儿只能用 SQL 删,记得同时用 `workspace_id` +
   `daemon_id <> 自己的 daemon` 双重限定。

**范围一定要按 workspace + creator 限定。** 按 `status='running'` 全表捞会捞到别人的沙箱 (上一轮就差点误删 `lrm-team`
里另一个用户的实例,靠 workspace 作用域的 404 挡住了)。
