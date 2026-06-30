# SWE 数据构造管线对比（可验证奖励 / Verifier 设计调研）

> 用途：为 Multica 多智能体 DAG RL 的 **agentic verifier / 可验证 rubric** 设计提供"现有方案调研"。
> 核心问题：把 GitHub issue / PR / commit 转成**可验证的测试 oracle + 可执行环境**，作为 RL 奖励信号。
> 维护：随调研更新；事实来源为各项目论文 / 数据集页面（见文末引用）。

## 一、主对比表

> 表很宽，建议横向滚动。按"实例来源/是否可验证/训练范式"三轴速览见第三节。

| 维度 | **SWE-smith** | **SWE-Factory** | **R2E-Gym** | **SWE-Fixer** | **SWE-bench-Extra** | **SWE-rebench** | **SWE-Lego** | **Skywork-SWE** |
|---|---|---|---|---|---|---|---|---|
| 出处 | SWE-bench team (NeurIPS'25) | DeepSoftwareAnalytics (FSE'26) | UC Berkeley (COLM'25) | InternLM/上海AI Lab | Nebius (24-12) | Nebius (2505.20411) | Huawei (2601.01426) | Skywork/Kunlun (2506.19290) |
| 核心目标 | 海量合成实例 | 自动"工厂"造可验证实例 | 程序化生成可执行环境 | 开源 SFT pipeline | 大规模数据+轨迹 | **RL 数据+防污染榜** | **SFT 极限+混合数据+生成式 verifier** | **数据规模律+统一执行套件** |
| **实例来源** | 现有仓库**注入 bug** | 真实 **issue+PR** | 真实 **commit** | 真实 **issue+PR** | 真实 **issue+PR** | 真实 **issue+PR** | **真实+合成混合** | 真实 **issue+PR** |
| **problem_statement** | LLM 生成 | 真实 issue | commit **回译** | 真实 issue | 真实 issue | 真实 issue | 真:真实/合成:LLM | 真实 issue |
| **测试 oracle** | 注入 bug 破坏的现有测试 | PR 测试+校验 | **自动生成测试** | 现有测试(仅后处理) | PR test_patch | PR 测试+验证 | 真:PR/合成:注入前后 | PR test_patch |
| **可执行环境** | ✅ Docker | ✅ Docker | ✅ | ❌ **无** | ✅ 容器 | ✅ 容器 | ✅ Docker | ✅ Docker(三级镜像) |
| **环境构建方式** | 单仓库建后批量注入 | **LLM 多智能体** | **程序化 SWE-Gen** | 不适用 | 通用安装脚本 | 交互式 setup agent+LLM 过滤 | 复用 SWE-rebench 仓库 | **统一默认配置+三级镜像缓存** |
| **验证 / grading** | F2P/P2P | **退出码**+自动 fail2pass | **混合验证器** | 语法+现有测试 | 执行式取 F2P | 执行式+LLM 评审 | F2P/P2P+防 Git hacking | **empty/gold 两跑取 F2P** |
| 语言 | Python(后拆分) | **多语言** | Python | Python | Python | Python(V2 多语言) | Python | Python |
| 规模 | 50K/128 仓 | 671 issue | ~8.1K | 331K→110K | 640K→**6.4K**+80K 轨迹 | **>21K**/3.5K 仓 | **32.1K**+18.1K 轨迹/3251 仓 | **10.2K**/2531 仓+8.2K 轨迹 |
| **训练范式** | RLVR/SFT 轨迹 | RLVR(造数据) | RLVR+测试时验证 | **SFT** | SFT+critic | **RL at scale** | **SFT-only**+生成式 verifier | **SFT**(留 RL 待做) |
| 需人写 issue/测试 | 否(全合成) | 是 | **否** | 是 | 是 | 是 | 部分 | 是 |
| **防污染** | 新仓库 | 持续爬新 PR | commit 级 | 排除 SWE-bench 仓 | 排除仓+fork | **滚动供新任务** | 防 Git hacking | 排除 SWE-bench Verified 仓 |
| 关键产物 | 数据+SWE-agent-LM | swe-factory 工具链 | 环境+验证器 | 模型+110K+code | 数据+轨迹 | 数据+滚动榜 | 数据+模型+**生成式 verifier** | **Skywork-SWE-32B**+数据 |

## 二、各项目构造管线详解（按"是否可验证 + 关键创新"）

### Skywork-SWE（Kunlun，arXiv 2506.19290）—— 统一执行套件 + 数据规模律
- **定位**：证明 SWE 任务存在**数据规模律**（log-linear，8K+ 轨迹仍未饱和）；强调 SWE-bench-Extra/SWE-Fixer 缺"可执行环境"或"严格测试校验"，自己三者兼备（可执行环境 + 验证过的单测 + **统一代码执行套件 Code Execution Suite**）。
- **三阶段管线**：
  - **A. 采集+预过滤**：GitHub API 爬 repo 元数据（**排除 SWE-bench Verified 仓**防泄漏，按 star 优先；151K→8.5K 仓）→ 收"closes/fixes/resolves #…"且改测试文件的已 merge PR（146K 实例）→ 安装级验证（回退 base_commit 跑安装，146K→23K）。
  - **B. 执行式验证+环境**：**统一默认配置**（Py3.9 + requirements.txt + pytest/hypothesis/mock + `[test]/[tests]/[dev]` 回退安装 + apt make/gcc + 统一 pytest）；**三级 Docker 镜像**(base/env/instance) 增量构建+缓存；**empty test**(只打 test_patch) + **gold test**(test_patch+code patch) 两跑，取"empty-FAIL→gold-PASS"为 FAIL_TO_PASS；只留非空 F2P（23K→**10,169**）。
  - **C. 轨迹生成**：多个 proprietary LLM 经 OpenHands v0.32 跑(≤100 turns)→ 只留最终补丁过全部测试的轨迹（→**8,209**）。
- **训练**：纯 SFT（Qwen2.5-Coder-32B-Instruct）。38.0% pass@1，TTS(Best-of-8 + OpenHands critic) → 47.0%。**显式把 RL 留作未来工作**（指出可执行环境正是 online RL 的 verified reward 来源）。
- **工程教训**：单一统一配置必然丢数据（无法覆盖所有仓库依赖）；镜像约 1.2GB/个，全量 11.9TB → 分 mini-batch rollout-验证-删镜像复用。

### SWE-rebench（Nebius，arXiv 2505.20411）—— SWE-bench-Extra 规模化后继
- 自动、可持续管线持续抽真实交互式 SWE 任务；交互式 setup agent 合成每仓库安装/测试流程，LLM 评审 ensemble 剔除不可靠实例。
- >21,000 条 Python 任务（3.5K 仓），**专为大规模 RL**；持续供新任务建**防污染滚动 benchmark**。

### SWE-Lego（Huawei，arXiv 2601.01426）—— 真+合成混合，SFT 极限 + 生成式 verifier
- **混合数据**：真实(取自 SWE-rebench 的 PR，深度) + 合成(SWE-smith 式 LLM Rewrite/AST 注入，广度)；32.1K 实例 + 18.1K 轨迹(14.1K resolved + 4.0K **semi-resolved**：定位对但没修对，回收当信号)。
- **生成式 verifier > 回归式**：预测 yes/no 用 token 概率算分，TTS@16 49.6%，超 OpenHands-Critic-32B(44.0%) 与 R2E-Gym-Verifier-14B(47.0%)。
- **防 Git hacking**：删 issue 日期后 git 历史；**step-level error masking**：报错 token 不计 loss。

### 其余项目（详见早期记录）
- **SWE-smith**：反向造 bug（破坏现有测试），50K/128 仓，量大但 issue 合成。
- **SWE-Factory**：LLM 多智能体建环境，退出码 grading + 自动 fail2pass(P=.92,R=1.0)，多语言。
- **R2E-Gym**：从 commit 回译生成测试（不需 issue/人写测试），**执行式+执行无关混合验证器**。
- **SWE-Fixer**：**无执行环境**，纯 SFT(检索+编辑)，黄金补丁当标签 + rationalization CoT。
- **SWE-bench-Extra**：真实 PR + 通用安装脚本，6.4K 实例 + 80K 轨迹。

## 三、范式抽象（"bug/测试从哪来"）

- **反向造 bug**：SWE-smith — 破坏好仓库现有测试，量大、issue 合成。
- **正向收真 PR + 自动建环境**：SWE-Factory(LLM agent) / SWE-bench-Extra(通用脚本) / **SWE-rebench**(setup agent+LLM 过滤,可持续) / **Skywork-SWE**(统一配置+三级镜像,规模律)。最贴真实 issue。
- **从 commit 合成测试**：R2E-Gym — 不需 issue/人写测试，混合验证器最贴 verifier 设计。
- **混合(真+合成)**：SWE-Lego — 真给深度、合成给广度，统一 schema；当前最完整 recipe。
- **无环境 SFT**：SWE-Fixer — 不造可验证奖励，仅贡献爬取技巧 + 无环境冷启动。

## 四、对 Multica verifier / RL 的取用建议

| 需求 | 选谁 | 贡献 |
|---|---|---|
| 先跑通"测试执行→奖励→GAE"链路 | **SWE-smith** | 环境保证可跑，零搭建 bootstrap |
| 现成、已验证、SWE-bench 格式真实数据 | **SWE-bench-Extra** | 6.4K 即用实例 + 80K 轨迹 |
| **大规模 RL 数据 + 防污染评测** | **SWE-rebench** | 21K RL-ready 任务 + 滚动榜（最贴 RL 目标） |
| **混合数据配方 + 生成式 critic 范式** | **SWE-Lego** | 真+合成;生成式 verifier 优于回归式(印证 Phase 3 生成式 critic);防 Git hacking;semi-resolved 部分信用 |
| **可复用的统一执行/验证套件** | **Skywork-SWE** | 统一默认配置 + 三级 Docker 镜像 + empty/gold 两跑取 F2P（直接抄给 `ObjectiveVerifier` 的执行端） |
| 扩到真实 issue 语义、多语言 | **SWE-Factory** | 退出码 grading + 自动 fail2pass |
| "客观测试 + LLM 判断"双验证器 | **R2E-Gym** | 混合验证器 = `ObjectiveVerifier`+`AgenticVerifier` 现成范式 |
| 无可执行测试子任务冷启动 | **SWE-Fixer** | GitHub events 爬取 + rationalization CoT |

**映射到本仓库现有结构**（`tree_search/agents/`）：
- `image_name`/容器 + `FAIL_TO_PASS`/`PASS_TO_PASS` → `verifier.py::ObjectiveVerifier` 的 `check` callable，返回测试通过率（**可验证 rubric 客观锚点**）。Skywork-SWE 的 empty/gold 两跑 + 统一配置可直接作为该执行端实现蓝本。
- 测试无法覆盖部分（协作分工、是否真懂 issue）→ `agentic_verifier.py::AgenticVerifier`（固定 judge）仲裁，与客观结果加权（R2E-Gym hybrid 模式）。
- **生成式 critic（Phase 3）** 设计借鉴 SWE-Lego：用 `yes/no`(或 `<score>N</score>`) token 概率算分，已证明优于回归打分头。
- **防 Git hacking** 对应 verifier 只读纪律：agent 不得读 git 历史/transcripts 反推答案。
- 终局 verifier reward 按 RL `session_id` 写回 → Phase 3 co-trained critic 经 GAE 分配到中间节点；**semi-resolved（定位对没修对）→ 部分信用**作稠密奖励来源。

## 五、引用

- SWE-smith — arXiv:2504.21798；`github.com/SWE-bench/SWE-smith`
- SWE-Factory — arXiv:2506.10954；`github.com/DeepSoftwareAnalytics/swe-factory`
- R2E-Gym — arXiv:2504.07164；`github.com/R2E-Gym/R2E-Gym`
- SWE-Fixer — arXiv:2501.05040；`github.com/InternLM/SWE-Fixer`
- SWE-bench-Extra — Nebius blog (2024-12)；`huggingface.co/datasets/nebius/SWE-bench-extra`
- SWE-rebench — arXiv:2505.20411；`huggingface.co/datasets/nebius/SWE-rebench`
- SWE-Lego — arXiv:2601.01426；`github.com/SWE-Lego/SWE-Lego`
- Skywork-SWE — *Unveiling Data Scaling Laws for Software Engineering in LLMs*, arXiv:2506.19290；`huggingface.co/Skywork/Skywork-SWE-32B`
- 基础 — SWE-bench, arXiv:2310.06770；`github.com/SWE-bench/SWE-bench`
