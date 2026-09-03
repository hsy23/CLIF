# DeltaServe-style vLLM 单副本原型：WSL 实测、差距评估与引入路线

研究日期：2026-08-27  
对象：CLIF 当前原型、vLLM 0.21.0、Qwen3-0.6B、Ubuntu 24.04 WSL、RTX 5070 Ti Laptop GPU（11.94 GiB）

## Executive Summary

结论先行：**当前实现可以运行 vLLM 推理，也可以运行 DeltaServe admission policy 的纯逻辑测试，但还不能执行任何真实的 vLLM 训推并发。** 因此目前不能证明“无干扰高效并发”，甚至还没有可测量干扰的 concurrent treatment path。按子系统工程完备度估算，现有代码约完成了 **15%–20% 的原型脚手架**；按“端到端完成一次训练 forward/backward、同时处理推理并更新 LoRA”这一能力口径，则是 **0%**。该百分比是本报告的工程启发式评分，不是论文指标。

现有工作的价值主要在三点：第一，`DeltaServeAdmissionController` 已表达 shortest-first、activation capacity、graph/eager latency 和 backward 阻塞等核心策略；第二，`vllm_adapter.py` 已明确 synthetic prefill request 和需要 fork 的 vLLM 边界；第三，`CLIFDeltaServeBridge` 的职责划分与“保留 CLIF 全局 launcher 和局部 batch scheduler、把训练执行下沉到 engine-backed backend”的方向一致。但这三部分尚未连入真实执行链路。[1][20][21][22]

本机基线确认 vLLM 0.21.0 能用已有的 Qwen3-0.6B 权重在 5070 Ti 上运行，测试期间没有重复下载模型。CUDA Graph 模式的短请求吞吐在 batch 1/4 分别为 235.824/316.622 token/s，而 eager 为 51.641/137.078 token/s，对应 4.57× 和 2.31× 的局部差距。[19] 这个结果不能泛化为 vLLM 的一般性能，但直接说明：如果 mixed FT row 迫使 batch 从 graph 降为 eager，错误的 admission model 会产生很大推理干扰。

更重要的是，当前 WSL 环境没有 NVIDIA MPS 可执行文件，PyTorch 跨进程 CUDA tensor 传递在本机组合上报 `invalid resource handle`。[19] DeltaServe 论文依赖 separate backward subprocess、CUDA MPS 和 shared GPU memory；因此论文忠实的隔离路径暂时无法在这台 WSL 上验证。[1][8][10][11] 建议先在同进程完成“功能正确”的 mixed forward + activation-backed LoRA backward，再迁移到原生 Linux 或修复 WSL IPC/MPS 后验证“低干扰高效”。

对“无干扰”的表述也需要校准。DeltaServe 的目标是 **bounded interference under inference SLOs**，不是零干扰；论文自身也报告非零推理开销，在一个轻负载设置下平均推理延迟约增加 22%。[1] 对 CLIF 更合理的验收目标是：在给定 TTFT/TPOT SLO 下保持高达标率，同时让 FT throughput 显著高于 idle-only baseline。

## Introduction

本研究回答三个问题：

1. 当前 WSL + RTX 5070 Ti + vLLM 0.21.0 环境是否能运行已有原型？
2. 当前代码距离 DeltaServe-style vLLM 单副本还缺哪些执行机制？
3. 是否已经或最终能做到推理与微调在 vLLM 上低干扰、高效率地并发？

评估把“能运行”分为四层，避免把策略 dry-run 误认为系统原型：

| 层级 | 成功定义 | 当前状态 |
|---|---|---|
| L0 环境 | CUDA、vLLM、模型权重可用 | 通过 |
| L1 策略 | admission 与状态机测试通过 | 通过 |
| L2 引擎 | synthetic FT row 实际进入 vLLM forward 并返回 activation | 未实现 |
| L3 训推共存 | backward/optimizer 与推理并发，可测 TTFT/TPOT 和 FT throughput | 未实现 |

DeltaServe 不是把现有训练循环简单放在 vLLM 旁边。论文的关键是在不接管 host engine 的请求队列、KV cache、采样和 forward 的前提下，拦截每轮调度结果，插入 prefill-only 的训练行，并把这些行的 residual-stream activations 交给独立 backward worker。[1] 因而，真正的集成点在 vLLM scheduler output、model runner 和 engine completion path，而不是 CLIF launcher 外层。

## Main Analysis

### 1. DeltaServe 实际需要的执行链

论文机制可归纳为一条跨进程流水线：

```text
CLIF local FT queue
        |
        v
vLLM Scheduler.schedule
  +-- inference batch owned by vLLM
  +-- SLO/admission decision
  +-- synthetic prefill-only FT rows
        |
        v
GPUModelRunner mixed forward (eager)
  +-- normal inference logits/KV/sampling
  +-- capture FT-row residual activations
        |
        v
activation ring buffer / shared GPU memory
        |
        v
separate backward worker under CUDA MPS
  +-- frozen base model
  +-- LoRA loss/backward/optimizer
  +-- yield at layer boundaries when inference needs GPU
```

其中 scheduler 使用最短样本优先策略，同时检查 activation buffer 和推理 SLO；latency model 需要区分 graph 与 eager 两组系数，因为加入训练行会改变执行模式。[1] backward 期间不再接纳新的 FT row，并可在 transformer layer 边界让出 GPU；必要时，FT-only forward 可被中止并重新排队，以保护 tail latency。[1]

vLLM 原生 LoRA 能力不能替代这条链。`max_loras` 只约束一个 inference batch 中可以并存的 LoRA 数量，并没有训练 loss、autograd 或 optimizer 接口。[7] 同样，vLLM 虽允许配置 custom scheduler class，但官方文档明确 scheduler interface 不是 public API，兼容性不保证。[6] 因此一个可控的研究原型应 pin vLLM 0.21.0 并维护小型 fork，而不是把核心逻辑仅做成外部 Python wrapper。

### 2. 当前代码实现了什么

`engine/deltaserve_core.py` 是无外部依赖的 policy core：[20]

- `FineTunePool` 能按序列长度和 sample id 稳定排序、claim、complete 和 requeue。
- `ActivationBuffer` 只记录 token-equivalent 容量。
- `LatencyModel` 计算 graph/eager 两类线性预测。
- `DeltaServeAdmissionController.admit` 能在 SLO budget 和容量约束下逐个接纳短样本。
- `begin_backward`/`finish_backward` 能阻止 backward 期间的新 admission，并在失败时 requeue。

四个单元测试全部通过，dry-run 能得到预期 admission 和 backward-blocking 结果。[19] 这些测试证明“编码的政策逻辑自洽”，但它们没有创建模型、tensor、loss、autograd graph、optimizer，也没有调用 vLLM engine。

`engine/vllm_adapter.py` 定义了 synthetic request marker、vLLM 版本检查和 fork points。[21] 但实测调用 `make_synthetic_prefill_request` 即在第一步失败：vLLM 0.21.0 的 `Request` 需要 `pooling_params`，当前适配器未传入。显式传 `pooling_params=None` 可以构造 request，因此这是小型 API 兼容性缺陷，不是架构阻塞。[19]

真正的架构阻塞是 vanilla `EngineCore` 不存在 `register_deltaserve_hooks`，而 `GPUModelRunner.execute_model` 被 `@torch.inference_mode()` 包裹。[5][12][19] `inference_mode` 禁用梯度跟踪并削减 autograd 开销，因此不能直接保留训练 backward 所需的图。[12] 当前 adapter 只是检查“所需 patched capability 是否存在”，并没有实现它。[21]

`engine/clif_bridge.py` 目前是接口骨架。[22] 静态检索未发现它被 `main.py`、`run.py` 或现有 `core` 调用；原 replica 路径仍通过 Hugging Face `model.generate` 做推理、通过 `local_train` 做本地训练。[19] 所以 CLIF 的 launcher/scheduler 与新 backend 尚未形成控制闭环。

### 3. WSL 与 Qwen3-0.6B 实测

环境探针结果如下：[19]

| 检查项 | 结果 | 判断 |
|---|---|---|
| WSL / CUDA | Ubuntu 24.04；CUDA available | 通过 |
| GPU | RTX 5070 Ti Laptop；11.94 GiB | 通过 |
| Python packages | torch 2.11.0+cu130；vLLM 0.21.0 | 通过 |
| 模型 | 使用已有 Qwen3-0.6B snapshot | 通过；未重复下载 |
| admission tests | 4/4 通过 | 仅 L1 |
| vLLM request adapter | 缺 `pooling_params` | 失败，但易修 |
| vLLM hook | `register_deltaserve_hooks=False` | 关键缺口 |
| CLIF import | 缺 pandas；peft/sklearn 不存在 | 失败 |
| MPS binaries | 不存在 | 论文忠实隔离路径阻塞 |
| CUDA IPC probe | `invalid resource handle` | 本机跨进程路径阻塞 |

推理-only 微基准如下。每个 measured request 生成 16 token；warm-up 不计入表格：[19]

| 模式 | Batch | Aggregate token/s | 相对 eager |
|---|---:|---:|---:|
| eager | 1 | 51.641 | 1.00× |
| CUDA graph | 1 | 235.824 | 4.57× |
| eager | 4 | 137.078 | 1.00× |
| CUDA graph | 4 | 316.622 | 2.31× |

这个基准只证明推理基础设施可用，不能作为 concurrent 性能结论。它也使用 WSL 的 `pin_memory=False` 路径，vLLM 明确提示这可能降低性能。[19] 下一阶段应固定 prompt/output 分布、并发度、warm-up 和重复次数，并测 TTFT、TPOT、request throughput、显存峰值与 FT samples/s。

### 4. 为什么现在不能测“训推无干扰并发”

一个有效的并发实验至少需要两条路径：control 是 inference-only，treatment 是同一模型、同一 workload 下 inference + real FT backward。当前只有 control。所谓 backward 只是 Python 状态位，activation buffer 只是整数计数，所以 treatment 根本不存在。[19][20]

另外，当前 WSL 缺少 DeltaServe 用于隔离的两个关键条件。NVIDIA MPS 的作用是让多个进程的 CUDA kernel 更有效地共享 GPU；CUDA IPC 则允许跨进程映射 GPU memory/event handle。[8][9][10] NVIDIA 文档列出 WSL 从 R510 驱动代际支持 legacy CUDA IPC API，但本机 PyTorch spawn probe 仍失败，说明“平台文档支持”不能替代对具体 driver/runtime/framework 组合的验证。[11][19]

因此有两种实现策略：

1. **同进程功能原型**：最快验证 scheduler injection、activation capture、LoRA backward 和 CLIF wiring；缺点是不能主张 DeltaServe 式进程隔离，autograd 与 inference 更容易相互干扰。
2. **原生 Linux / 修复 WSL 后的论文忠实原型**：使用 MPS + CUDA IPC + separate backward worker；工程成本更高，但才有资格评估论文所说的 bounded-interference co-serving。

### 5. 逐子系统差距

以下完成度是基于代码与实测的工程启发式评分，用于排序，不应解释为可发表指标。

| 子系统 | 当前证据 | 启发式完成度 | 到可运行原型还缺什么 |
|---|---|---:|---|
| Admission policy | 4 tests 通过；shortest-first/SLO/capacity/state | 60% | 用真实 vLLM metrics 校准；在线更新；处理真实 completion/abort |
| vLLM request adapter | marker 和 fork points 已定义；API 调用失败 | 10% | `pooling_params=None`；internal request lifecycle；无用户输出 |
| Scheduler injection | 只有设计说明 | 0% | patch `Scheduler.schedule`/`SchedulerOutput`；混合 batch；force eager |
| Activation capture | token 计数，不存 tensor | 0% | FT-row mask；逐层 residual capture；ring buffer；生命周期/显存回收 |
| Backward/optimizer | 只有 running flag | 0% | loss、LoRA backward、optimizer step、adapter versioning |
| MPS / CUDA IPC | 无实现；本机探针失败 | 0% | MPS daemon、shared handles、进程存活协议、错误恢复 |
| Preemption/yield | 只有 admission block | 0% | layer-boundary yield；FT-only forward abort/requeue |
| Latency model | 静态线性公式 | 10% | graph/eager 实测拟合、online refinement、TTFT/TPOT SLO budget |
| CLIF wiring | bridge class 未被调用 | 10% | launcher 生命周期、local scheduler queue、adapter promotion/completion |
| Interference benchmark | 只有 inference-only baseline | 20% | real treatment path、重复试验、置信区间和 SLO 报告 |

综合判断是 **脚手架 15%–20%，端到端 co-serving 0%**。完成度较高的 admission core 恰好是风险较低的一部分；最难、最影响论文结论的 execution backend 仍未开始。

### 6. Public 源码状态

截至 2026-08-27，本次检索没有找到可访问的 DeltaServe 官方 public repository。论文表达了开源意图，但 arXiv HTML 未给出代码链接；CatalyzeX 页面仍显示“Request Code”。[1][18] 这是“本次未找到”，不是对互联网上绝对不存在源码的证明。因此当前应以论文机制自主实现，同时保留未来对官方实现做差异对照的接口边界。

相关系统可以帮助校准设计空间：FlexLLM 研究 inference 与 PEFT co-serving，[14] MuxServe 研究多 LLM 的时空复用，[15] Punica 处理多租户 LoRA serving。[16] 但它们不能替代 DeltaServe 在现有 host engine 中插 synthetic training rows、捕获 activation 并用独立 backward process 的具体机制。

## Synthesis & Insights

### 对 CLIF 架构选择的判断

“保留 CLIF 的局部 batch 控制 scheduler 和全局 launcher，把 DeltaServe 作为 engine-backed execution backend”是合理方向，但职责需要收紧：

- CLIF launcher 继续负责跨 replica 资源分配、任务生命周期和 adapter promotion。
- CLIF local scheduler 提供 FT queue、优先级、全局 budget 与 policy intent。
- vLLM host scheduler 拥有最终每步 batch；DeltaServe backend 在这里做实时 SLO admission。
- training worker 只消费 backend 产出的 activation work item，不自行绕过 vLLM 抢 GPU。

如果 local scheduler 直接决定某个 FT batch“现在执行”，而不读取 vLLM 当前 prefill/decode/KV 状态，就会破坏 DeltaServe 的核心保证。正确的关系不是两个 scheduler 并列发 GPU 工作，而是 CLIF 给出候选和约束，vLLM-side admission 在每一 iteration 做最后裁决。

### “无干扰”应改成可测的 bounded interference

理论上，同一 GPU 上同时增加 forward token、切换 graph/eager、保存 activation、执行 backward，就不可能物理上零干扰。DeltaServe 的贡献是以 SLO model、抢占和 MPS 把干扰控制在可接受区间。[1] 本项目建议定义以下验收门槛，而不是使用“无干扰”作为不可证伪目标：

- 在预先声明的 TTFT/TPOT SLO 下，推理请求达标率至少 99%。
- concurrent 的 p50/p95/p99 TTFT 和 TPOT 全部报告相对 inference-only 的增量。
- FT throughput 必须显著高于 idle-only FT baseline，否则 mixed admission 没有系统价值。
- 任何达到上述结果的实验都必须包含 graph/eager 分层、warm-up、至少三次重复和显存峰值。

这些是建议的项目验收条件，不是 DeltaServe 论文原样给出的统一阈值。

### 本机最关键的两个先验结论

第一，graph/eager 差距足够大，不能先写一个固定比例的 latency estimator 再假定无害。应先从 vLLM iteration metrics 拟合本机 Qwen3-0.6B 的两组系数，然后再开放 mixed admission。[1][19]

第二，WSL 的 MPS/IPC 阻塞应与核心功能开发解耦。若一开始就把所有工作押在 separate process 上，可能长时间停留在环境问题；先用同进程路径证明 request lifecycle、activation correctness 和 LoRA update，再把同一 work-item protocol 替换为 IPC transport，风险更低。

## Counterevidence Register

| 可能支持更乐观判断的证据 | 为什么不足以改变本报告结论 |
|---|---|
| 四个 admission tests 全部通过 | 它们只运行纯 Python policy，不包含模型 forward、activation 或 backward。[19][20] |
| Qwen3-0.6B 已能在 vLLM 上高速推理 | 这只证明 inference control path；没有 concurrent FT treatment。[19] |
| vLLM 支持 custom scheduler | 官方同时警告该接口不是 public API；而且 scheduler class 本身不能绕过 model runner 的 `inference_mode`。[6][12] |
| vLLM 原生支持多 LoRA | 该能力是 serving-time adapter batching，不提供训练 backward/optimizer。[7] |
| NVIDIA 文档列出 WSL legacy CUDA IPC 支持 | 本机具体 PyTorch/driver 组合仍在 CUDA storage rebuild 时失败，必须以实测为准。[11][19] |
| DeltaServe 论文报告高 FT throughput 与 SLO compliance | 论文仍报告非零 inference overhead；它证明 bounded interference 可行，不证明物理零干扰。[1] |

## Claims-Evidence Table

| 关键结论 | 主要证据 | 置信度 |
|---|---|---|
| 当前实现不是端到端 co-serving backend | activation/backward 只是计数和状态；bridge 未接主路径；无 concurrent treatment。[19][20][21][22] | 高 |
| vLLM 0.21.0 需要 fork-level 执行改动 | EngineCore 无预期 hook；model runner 使用 `inference_mode`；scheduler API 不稳定。[5][6][12][19] | 高 |
| 本机 graph/eager 切换是显著干扰风险 | Qwen3-0.6B 本地 batch 1/4 比值为 4.57×/2.31×；论文模型区分两组系数。[1][19] | 中高；基准规模小 |
| 当前 WSL 不能验证论文忠实 separate-process 路径 | MPS 工具缺失且 CUDA IPC probe 报 `invalid resource handle`。[19] | 高；仅适用于当前环境 |
| 最终应追求 SLO 下 bounded interference | DeltaServe 机制和评估均以 SLO 为约束，并报告非零 overhead。[1] | 高 |
| 脚手架完成度约 15%–20% | 子系统逐项评分的综合工程判断。 | 中；启发式而非测量 |

## Limitations & Caveats

- 没有 real FT treatment path，因此本报告不能给出 concurrent TTFT/TPOT、FT throughput 或 SLO 达标率。
- 推理基准是 Qwen3-0.6B 的短输出微基准，样本数很小；数字只用于暴露 graph/eager 风险，不用于和论文硬件结果横向比较。
- WSL IPC 失败只代表当前 Windows driver、WSL、PyTorch 2.11/CUDA 13.0 组合。NVIDIA 文档仍列出 WSL legacy CUDA IPC 支持。[11][19]
- 没有找到 public DeltaServe repo 不等于它不存在或未来不会发布。[18]
- 当前 WSL venv 只满足 vLLM 子集，完整 CLIF 还缺 pandas、peft、sklearn；在补齐依赖前不能把 backend 接入现有主程序。[19]
- vLLM custom scheduler API 不稳定，后续升级 vLLM 需要重新审计 fork points。[6][17]

## Recommendations

建议按以下顺序实现，每一步都有可执行的验收条件。

### Milestone 0：让环境与 adapter 可复现

- 在现有 venv 安装并锁定 CLIF 必需依赖，至少使 `import main` 通过。
- 修复 `Request(..., pooling_params=None)`。
- 建立 vLLM 0.21.0 fork/patch branch，记录上游 commit。

验收：`import main`、现有 4 tests、synthetic request construction 全部通过；不下载新的模型副本。

### Milestone 1：真实 vLLM mixed-forward 最小闭环

- 在 `Scheduler.schedule` 后调用 DeltaServe admission。
- 扩展 `SchedulerOutput` 携带 FT request ids / token-row mapping。
- mixed batch 强制 eager；FT internal request 不产生用户输出、不污染正常 completion。
- 在 model runner 中区分 inference rows 与 FT rows。

验收：一次 inference request 和一个 synthetic FT row 进入同一实际 model forward；推理输出正确；FT row 能返回可核对 shape/dtype/layer id 的 activation tensor。

### Milestone 2：同进程 activation-backed LoRA update

- 冻结 base weights，仅允许目标 LoRA 参数求导。
- 对 FT rows 计算 next-token loss、backward 和 optimizer step。
- 建立 activation buffer 的真实 tensor 生命周期和 adapter version。

验收：固定样本训练一步后 loss/LoRA 参数发生可复现变化；inference-only 与 mixed-forward 均不泄漏显存；当前纯计数 buffer 被真实 buffer 取代。

### Milestone 3：隔离与抢占

- 优先在原生 Linux 验证 MPS + CUDA IPC；若坚持 WSL，先解决 `invalid resource handle` 并安装/提供 MPS tooling。
- separate backward worker 映射 base model、adapter 和 activation handles。
- 实现 layer-boundary yield，以及 FT-only forward abort/requeue。

验收：backward process 连续执行 100 个 step 无 handle 泄漏/死锁；推理到达时能观察到 yield；worker crash 后 sample 可 requeue。

### Milestone 4：在线 latency calibration 与 SLO admission

- 分别收集 graph/eager 的 per-iteration latency、prefill tokens、decode requests、KV tokens 和 FT tokens。
- 在线拟合并周期更新系数；只在 prediction margin 足够时接纳 FT row。

验收：held-out iteration latency prediction 有可报告误差；在固定 workload 下达到预设 99% SLO target；误差过大时自动回退 idle-only。

### Milestone 5：接入 CLIF launcher/local scheduler

- local scheduler 只提供 FT candidates 和 policy budget，不越过 vLLM-side final admission。
- launcher 管理 backend 生命周期、adapter promotion、结果/失败上报。

验收：现有 CLIF 工作流能选择单 replica backend，完成一次 local FT job，并保持原有非-DeltaServe 路径可用。

### Milestone 6：四组对照实验

固定同一推理 trace 和 FT dataset，至少运行：

1. inference-only；
2. FT-only；
3. inference + idle-only FT（DeltaServe-Temp 风格）；
4. inference + mixed FT + preemption。

输出 TTFT/TPOT p50/p95/p99、request throughput、FT samples/s 或 tokens/s、SLO attainment、GPU utilization、graph/eager iteration 比例、显存峰值和失败数。只有第 4 组相对第 1 组干扰受控、且相对第 3 组 FT throughput 明显增加时，才能声称实现了 DeltaServe-style efficient co-serving。

## Appendix: Methodology

本报告采用 Deep Research 工作流，证据优先级为：论文与官方文档/源码、可复现的本地实验、第三方代码可用性索引。技术机制只以论文、vLLM/PyTorch/NVIDIA 官方资料和本地代码/实验为依据；CatalyzeX 仅用于说明本次 public-code 检索状态。

本地检查包含：WSL CUDA/package probe、unittest、policy dry-run、vLLM Request/API introspection、EngineCore capability probe、model runner decorator inspection、CLIF import probe、Qwen3-0.6B graph/eager offline benchmark、MPS binary probe和 PyTorch CUDA IPC spawn probe。完整命令关键输出保存在 `test_results.md`。

仓库检查使用静态搜索确认现有 HF `model.generate`/`local_train` 路径、新 bridge 的引用关系，以及 admission/adapter 的实际实现边界。没有修改现有实现来“使测试通过”；新增内容只包括研究基准脚本和本报告证据包。

研究限制和负面结果均保留：适配器 TypeError、CLIF 依赖缺失、MPS binary 缺失和 CUDA IPC failure 没有被归类为通过。没有 concurrent treatment 就没有推断 concurrent 性能。

## Bibliography

[1] Karmakar, S., Wang, W., & Potti, N. (2026). [DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs](https://arxiv.org/abs/2607.28848)

[2] vLLM Project. (2026). [vLLM v0.21.0 source tree](https://github.com/vllm-project/vllm/tree/v0.21.0)

[3] vLLM Project. (2026). [vLLM v0.21.0 EngineCore](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/engine/core.py)

[4] vLLM Project. (2026). [vLLM v0.21.0 Scheduler](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/core/sched/scheduler.py)

[5] vLLM Project. (2026). [vLLM v0.21.0 GPU model runner](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/worker/gpu/model_runner.py)

[6] vLLM Project. (2026). [Scheduler configuration documentation](https://docs.vllm.ai/en/stable/api/vllm/config/scheduler/)

[7] vLLM Project. (2026). [LoRA configuration documentation](https://docs.vllm.ai/en/stable/api/vllm/config/lora/)

[8] NVIDIA. (2026). [When to Use MPS](https://docs.nvidia.com/deploy/mps/when-to-use-mps.html)

[9] NVIDIA. (2026). [Multi-Process Service documentation](https://docs.nvidia.com/deploy/mps/latest/index.html)

[10] NVIDIA. (2026). [CUDA Programming Guide: Interprocess Communication](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/inter-process-communication.html)

[11] NVIDIA. (2026). [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

[12] PyTorch Foundation. (2026). [inference_mode documentation](https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html)

[13] PyTorch Foundation. (2026). [Multiprocessing documentation](https://docs.pytorch.org/docs/stable/multiprocessing.html)

[14] Wang, Y. et al. (2024). [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789)

[15] Jiang, Z. et al. (2024). [MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving](https://arxiv.org/abs/2404.02015)

[16] Fang, G. et al. (2023). [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)

[17] vLLM Project. (2025). [RFC: Restructure Core Execution Loop](https://github.com/vllm-project/vllm/issues/23233)

[18] CatalyzeX. (2026). [DeltaServe code availability page](https://www.catalyzex.com/paper/deltaserve-host-agnostic-co-serving-of)

[19] Codex local experiment. (2026). [WSL / RTX 5070 Ti validation log](test_results.md)

[20] CLIF project. (2026). [DeltaServe admission core prototype](../../engine/deltaserve_core.py)

[21] CLIF project. (2026). [DeltaServe vLLM adapter prototype](../../engine/vllm_adapter.py)

[22] CLIF project. (2026). [CLIF DeltaServe bridge prototype](../../engine/clif_bridge.py)
