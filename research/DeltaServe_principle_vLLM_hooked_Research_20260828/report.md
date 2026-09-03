# DeltaServe-style vLLM 单副本原型：WSL 实测与差距评估

## Executive Summary

本轮结论需要分成“原理打通”和“性能达标”两层。第一层已经通过：在 RTX 5070 Ti Laptop GPU、PyTorch 2.11.0+cu130、vLLM 0.21.0 与本地 Qwen3-0.6B 上，synthetic FT request 与普通 inference prefill 被调度进同一次 vLLM model forward；final RMSNorm hook 只截取 FT row；独立 GPU 子进程读取共享 activation、共享 live `lm_head` base allocation 和共享 LoRA 参数，执行真实 cross-entropy、backward 与 AdamW 更新。运行时 PID 为 423，反向子进程 PID 为 557，forward loss 与 backward loss 均为 6.731245517730713，LoRA-B 的 L1 参数变化为 15712.62109375。[14][15][16]

这已经符合 DeltaServe 的三个核心功能原则：复用 host inference engine 的 forward、将 FT forward 合入 host batch、把 backward 放到独立 GPU subprocess。[1][14][15] 但它仍是 LM-head LoRA 的最小证明，不是完整 transformer-layer LoRA 训练。当前子进程没有映射每层 base weight 与保存每层 backward 所需的 activation，因此不能把“共享一个完整模型实例”理解为“已经支持 Q/K/V/O、MLP 等所有 LoRA target modules 的端到端训练”。[9][16]

第二层尚未通过：目前不能宣称“无干扰高效并发”。WSL 本轮没有启用 CUDA MPS；NVIDIA 文档说明，不使用 MPS 时，不同 CUDA context 通常按整卡 time slice 调度，而 MPS 才提供跨进程同时调度与有限 QoS 资源配额。[5][6] 此外，共训批次被强制 eager，hook 中还执行 full-vocabulary LM-head loss，并在派发 backward 前调用设备同步。DeltaServe 论文专门用 graph-aware latency model、TTFT/TPOT slack admission、activation capacity 和 backward-in-flight gate 来控制这些额外代价；当前实现只具备一个确定性合批屏障，没有 SLO 控制闭环。[1][7][15][16]

主要建议是保留现有原型作为 `engine-backed execution backend` 的阶段 A 证据，然后按“全层 LoRA hooks → 持久 backward worker 与双缓冲 → MPS/资源治理 → DeltaServe SLO scheduler → CUDA graph 与基准矩阵”的顺序推进。在这些步骤完成前，正确表述是“DeltaServe-principle functional prototype”，而不是“DeltaServe-equivalent high-efficiency backend”。

## Introduction

### Research Question

本报告评估当前代码距离 DeltaServe-style vLLM 单副本原型还有多远，并回答它是否已经能够在同一 GPU 上实现无干扰、高效率的训练与推理并发。范围限定为本轮新建的 hooked/VMM 路径，不把早期“vLLM 推理进程 + 另一份 Transformers base model 训练进程”的重复模型实现视为目标方案。

DeltaServe 的关键不是简单地让两个进程同时占用一张 GPU。论文描述的是：把 LoRA fine-tuning forward 折叠进 host inference engine 现有 forward，再由独立 GPU subprocess 补齐 gradient computation 和 adapter update；scheduler 在 host scheduling loop 内基于 SLO headroom 决定每步是否接纳 FT token。[1] vLLM V1 的 EngineCore 本身负责 scheduler、KV cache 与 GPU worker 协调，因此原型应尽量在这一边界内增加极小 hook，而不是建立第二套推理/训练执行栈。[2][10]

### Scope, Methodology, and Assumptions

研究采用三类证据。第一类是 DeltaServe、FlexLLM、LoRA、vLLM、S-LoRA 与 Punica 等论文，用于定义功能与性能目标。[1][9][10][11][12][13] 第二类是 vLLM、CUDA 与 PyTorch 官方文档，用于核验多进程、VMM、LoRA 与 GPU 调度边界。[2][3][4][5][6][7][8] 第三类是本机 WSL 运行结果、JSONL trace 和代码检查，用于证明当前实现实际执行了什么。[14][15][16]

本报告采用一个重要假设：当前阶段只要求功能原理一致，不要求复现论文吞吐或 SLO 数字。因此 LM-head LoRA 可以作为最小 end-to-end gradient path，但报告会明确把它与完整 all-layer LoRA 分开。另一个假设是 CLIF 原 scheduler 与 launcher 暂不合并；因此本轮只评价 engine backend，不评价 CLIF 的全局联邦训练语义。

## Main Analysis

### Finding 1: 三个核心功能路径已被真实 GPU 运行证明

DeltaServe 论文明确把 co-serving workflow 描述为“host forward 内折叠 FT forward + 独立 GPU subprocess backward”。[1] 当前实现不再加载第二份 Qwen base model。vLLM 仍是唯一执行 transformer trunk 的 engine；synthetic FT prompt 使用 `deltaserve-ft-` request ID 进入 vLLM scheduler，普通 inference prompt 同时进入。一个仅在 prototype 环境变量开启时生效的 scheduler barrier 会把孤立 FT request 保留一个 ingress cycle，使紧随其后的 inference prefill 能进入同一 scheduler output。

实测 trace 的 `merged_forward` 事件同时包含两个 request ID，scheduled token 分别为 37 和 7，二者 `num_computed_tokens` 都为 0，因此不是“一个先 prefill、另一个后 decode”的伪合批，而是同一次 prefill-carrying model forward。[15] hook 安装在 Qwen3/vLLM 模型 final RMSNorm 上，训练 flat slice 为 `[0, 37]`，其中 36 个 next-token pair 被复制到共享 activation buffer。forward loss 为 6.731245517730713。[15][16]

独立子进程 PID 557 通过 CUDA VMM 导出的 POSIX file descriptor 导入同一 physical allocation。NVIDIA 官方 VMM 流程就是 allocation export、handle transfer、receiver import、virtual-address reserve/map 和 access-right setup；这与当前 `cuMemCreate`、`cuMemExportToShareableHandle`、`cuMemImportFromShareableHandle`、`cuMemMap` 路径一致。[4][16] 子进程在相同 activation 与 live head weight 上重算 head logits 和 LoRA delta，backward loss 与 parent hook 的 forward loss 完全相等，随后参数发生非零变化并被 live vLLM logits path 读取。[14][15]

因此，对“有没有在真实 vLLM model forward 中合批”和“backward 是否真在另一个 GPU PID 中发生”两个问题，答案均为肯定。这个结论强于只看到两个进程都在运行，也强于早期重复模型 baseline，因为它有 batch request IDs、flat slice、allocation ID、PID、loss 与 parameter delta 的闭环证据。[14][15]

### Finding 2: 当前共享的是唯一 trunk 加上可导出的 live LM head，不是完整 all-layer backward graph

LoRA 的通常定义是在 transformer 的目标线性层旁注入低秩 A/B，并冻结 base weight。[9] 当前原型只在 LM head 上创建 rank-4 LoRA。vLLM trunk 对 FT row 和 inference row 的 base forward 是共享的，但 hook 只在 final RMSNorm 捕获最终 hidden states；子进程只需要映射 `lm_head.weight`、final activation、labels 与 head LoRA A/B，因此没有第二份完整 Qwen 权重。[16]

这一设计足以证明 DeltaServe 的执行分割：共享 host forward 产生 activation，外部 backward process 消耗 activation 并更新 adapter。但是它没有证明 full-model LoRA。若目标是 Qwen 常见的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，每层 LoRA-B 的梯度依赖该层输入，LoRA-A 与更早层 adapter 的梯度还依赖从后层传播回来的梯度。只保存 final hidden state 无法恢复这些梯度。完整实现必须在 layer boundary 捕获或重建必要 activation，并让 backward worker 读取每个 target layer 的 frozen base weight；同时要控制 buffer 生命周期，避免 host 下一批覆盖尚未消费的数据。

这也是当前方案与 Punica、S-LoRA 或 vLLM multi-LoRA inference 的差别。Punica 和 S-LoRA 解决的是多个 adapter 在共享 base inference 上的高效异构 batching；vLLM 也支持按 request 应用 adapter。[3][11][13] 它们提供 adapter execution kernel 与管理机制，但并不自动提供 training autograd graph。当前原型可以参考这些 kernel/adapter slot 机制来发布训练后的 LoRA，却仍需自行实现 training activation contract。

对“距离完整 DeltaServe-style 单副本”进行工程估计，当前大约完成了执行骨架的 55%-65%，但只完成了训练语义的 20%-30%。前者包括 scheduler marker、same-forward hook、VMM IPC、separate backward 与 adapter publish；后者目前只有 LM head 单步、单样本、单 adapter。这个百分比是基于剩余模块数量与风险的工程判断，不是论文给出的指标。

### Finding 3: 目前存在并发进程，但没有证明 GPU kernel 高效并行，更没有干扰隔离

本轮 backward 确实在独立 GPU process 中运行，但“进程并发”不等于“kernel 并行”。NVIDIA MPS 架构文档指出，不使用 MPS 时，不同进程的 CUDA context 通常按整卡 time slice 调度；MPS server 才能减少 context switching，并允许不同 client 的工作更直接地同时调度。[5] 本机探测结果为 MPS disabled，因此当前最合理的解释是 vLLM 与 backward worker 在同一 GPU 上竞争、切换，而不是可控地填充彼此空闲 SM。

即使改成单 context 多 stream，也不能承诺无干扰。CUDA 文档说明，stream priority 只是调度提示，不会抢占已经运行的 kernel，也不保证严格顺序；并发还会被 default stream、显式同步和资源不足阻断。[7] 当前 parent 在复制 activation 后调用 `torch.cuda.synchronize()`，确保 child 读取一致数据，但这同时把 forward-to-backward 边界做成 host blocking point。hook 内的 full-vocabulary `F.linear` 与 cross-entropy 也在 inference worker context 中执行，会直接增加该 prefill step 的 TTFT。

DeltaServe 对此并非假设“训练天然不会干扰推理”，而是使用 graph-aware latency model 和 SLO-aware admission 控制干扰：只有预测的 TTFT/TPOT slack、activation capacity 和 backward state允许时才接纳 FT token；activation hook 使 co-serving step 转为 eager，模型分别拟合 graph 与 eager latency。[1] 当前实现只有一条 one-shot batch barrier，既没有 arrival timestamp、SLO budget、decode debt，也没有 online model refinement。因此它能证明 correctness，不能证明 SLO compliance。

如果目标措辞是“无干扰”，还需要定义可检验门槛，例如 P99 TTFT 增幅不超过 5%、P99 TPOT 增幅不超过 3%、SLO violation rate 不上升，并在 burst、steady、decode-heavy、prefill-heavy 四类 workload 上测量。DeltaServe 的核心价值恰恰是利用可用 headroom，而不是在所有负载下强行共训。[1][12] 在没有这些实验前，结论必须是“可并发，但干扰未受控”。

### Finding 4: vLLM 集成点已足够小，但当前补丁仍是版本锁定实验件

vLLM V1 把 scheduler 放在 EngineCore，把 model execution 放在 GPU worker；这个边界适合在 scheduler 做 FT admission、在 model runner 做 eager decision 与 activation hook。[2] 当前 patch 只针对 vLLM 0.21.0 的两个源文件：scheduler 增加 prototype batch barrier，GPUModelRunner 增加 runtime 初始化、batch metadata 传递和 post-forward notification。复杂逻辑放在独立 `engine/deltaserve_vllm_runtime.py`，符合 vLLM 官方对 model-runner 公共热路径保持最小改动的工程方向。[2][16]

但补丁当前通过文本锚点写入 site-packages。它有版本检查、备份和 restore，适合本地验证，不适合长期维护。vLLM 任何 source movement 都会让 anchor fail-fast；这比静默打错位置安全，但仍不是可合并上游的 extension API。下一步应维护一个极小 fork commit，或把 hook interface 抽象成稳定 patch series，并在 CI 中对 pinned vLLM wheel/source commit 运行 patch-idempotence 与 GPU integration test。

异步调度在本次 one-shot test 中关闭。原因不是 DeltaServe 原理要求关闭，而是离线 API 会在两个逐条 `add_request` 之间让第一个请求提前进入 EngineCore。当前 barrier 解决了确定性 same-batch 证明，但生产实现应由 CLIF local scheduler 或 DeltaServe admission logic 直接原子构造 mixed batch，而不是依赖 request arrival 顺序。

### Finding 5: 最高风险不在 VMM，而在 activation contract、backward scheduling 与 SLO calibration

CUDA VMM 路径已通过独立 probe 和端到端 run，说明 WSL/5070 Ti 上可以绕过 legacy CUDA IPC 的 invalid-resource-handle 问题，把同一 allocation 映射到另一个 process。[4][15][16] PyTorch 官方也要求 CUDA tensor 跨进程使用 `spawn` 或 `forkserver`，本实现使用 `spawn`，与文档约束一致。[8]

剩余高风险首先是 all-layer activation contract：需要决定保存哪些 tensor、dtype、shape、layer ordering 和 microbatch boundary，并确保 vLLM inference-mode forward 不建立 autograd graph。第二是 backward worker 生命周期与缓冲区：当前子进程启动后约 2.4 秒才记录 `backward_started`，说明 one-shot spawn 开销远大于一次训练 step；生产路径必须预热持久 worker，使用至少双缓冲和 CUDA event/sequence number 做 ownership handoff。[15]

第三是 GPU arbitration。NVIDIA 文档说明 MPS 可以提供 active-thread percentage 或更细资源 provisioning，但这仍是上限，不是完全独占预留。[6] 若 WSL 环境无法稳定运行 MPS，可考虑把 backward 放到同一 worker process 的低优先级 non-blocking stream，或研究 CUDA green contexts；但这会偏离“独立 GPU subprocess”这一指定约束。更直接的路线是在原生 Linux 主机验证 MPS，然后保留 WSL 作为 correctness 环境。

第四是 graph-aware latency model。co-serving batch 因 hook 被迫 eager，而 inference-only batch仍应保留 CUDA graph；需要针对 Qwen3-0.6B 的 prefill tokens、decode tokens、batch size、graph/eager mode 建立离线系数，并在线修正。只有 scheduler 用这些预测控制 FT token admission，才能开始讨论“高效且满足 SLO”。[1][7]

## Synthesis & Insights

当前原型最有价值的成果不是 LM-head LoRA 本身，而是证明了一个可行的 ownership split：vLLM 继续拥有 token batching、KV cache、attention/trunk forward 和 sampling；DeltaServe backend 只拥有 synthetic FT metadata、activation export、backward worker 与 adapter version。这个边界与后续保留 CLIF local batch scheduler 和 global launcher 的方向兼容，因为 CLIF 可以决定“是否以及放多少 FT tokens”，backend 负责“如何在 host engine 内执行”。

第二个重要模式是：memory sharing 与 execution isolation 是两个独立问题。VMM 解决“不要复制 base allocation”；MPS/streams/scheduler 解决“两个执行方如何争用 SM 和 memory bandwidth”。当前前者已经有证据，后者基本未做。继续优化前应避免把两者混成一个指标，否则看到显存下降就误判性能干扰已解决。

第三个洞察是 adapter publication 应晚于完整 optimizer commit，但不必晚于整个 training job。当前共享 A/B 在 child 中原地更新，parent 在收到 `backward_finished` 后设置 `adapter_ready`。这一机制可以自然扩展成双版本 adapter slot：inference batch固定读取 committed version N，child 写 staging version N+1，完成后原子切换。这比让 vLLM 在 backward 中途读取同一 tensor 更安全，也更适合 CLIF round/aggregation 语义。

## Limitations & Caveats

### Counterevidence Register

一项可能削弱“没有高效并发”判断的证据是：即使未启用 MPS，驱动仍可能在不同 context 之间调度工作，并且当单个 workload 未占满 GPU 时可能观察到表面重叠。[5] 本报告没有把这种可能性解释为“完全串行”，而是解释为“没有可验证的同时调度和 QoS 保证”。只有 Nsight timeline 与 latency benchmark 才能确认本机实际 kernel overlap 比例。

另一项反向证据是 forward/backward loss 完全相等且 adapter 已成功发布，这说明 LM-head 路径不是 mock。[14][15] 但该证据只能支持最小 training path，不能外推到 all-layer LoRA；因此它提高了功能结论的置信度，不改变性能与完整训练语义的限制。

报告没有复现 DeltaServe 论文的生产 trace、2.9x throughput 或 100% SLO compliance；这些数字只描述论文系统，不适用于当前 5070 Ti 原型。[1] 本轮也没有运行长时间 burst benchmark，因此没有 P50/P95/P99 TTFT/TPOT。GPU 并发判断依据是官方调度语义、本机 MPS disabled 状态与代码中的同步点，而不是 Nsight Systems kernel timeline。[5][6][7]

当前训练只覆盖 LM head，不能用于判断 full-layer LoRA 的显存、吞吐或模型质量。训练数据是短 synthetic text，只有一个 optimizer step；参数非零变化证明反向链路有效，不证明收敛或 adapter quality。[14][15] vLLM 运行结束有 NCCL process-group cleanup warning，未影响结果 JSON，但应在长期 runner 中显式关闭 engine 与 process group。

另外，论文发布日期为 2026 年 7 月，公开页面描述了 vLLM、SGLang、S-LoRA 集成，但本轮检索未发现作者公开实现仓库，因此当前代码是依据论文接口与已安装 vLLM 0.21.0 源码自主实现。[1] 若后续出现官方代码，应优先做接口和 buffer protocol 对比，而不是继续猜测内部细节。

## Recommendations

第一阶段应把当前 LM-head proof 固化为回归基线：保留 patch idempotence test、VMM IPC probe、hooked end-to-end result，并在每次 vLLM 版本升级时验证 same-batch request IDs、allocation IDs、different PIDs、loss equality 与 parameter delta。

第二阶段扩展到一个 transformer block 的 LoRA，建议先只做 `q_proj` 与 `v_proj`。需要定义 layer hook 输出、保存 input activation、backward worker 重算该层必要 base op、adapter gradient 与 version commit。单层通过后再扩展所有 target modules，避免一次性实现整网而无法定位梯度错误。

第三阶段把 backward worker 改为 engine 初始化时完成 CUDA context、VMM import、optimizer 与 kernel warmup的持久进程。使用双 activation buffer、job sequence、CUDA event 或显式完成标志，消除当前 2.4 秒首任务启动空窗，并允许 inference 在 backward 活跃时继续推进。

第四阶段在原生 Linux 上启用并校准 MPS，分别测试无 MPS、MPS active-thread percentage、以及可能的 same-process low-priority stream。每种配置都记录 TTFT、TPOT、throughput、FT tokens/s、context-switch overhead 和显存。只有 P99 inference 指标满足预先定义的门槛，才称为“低干扰”。[5][6][7]

第五阶段实现 DeltaServe admission model：为 graph/eager、prefill/decode/mixed step 分别拟合 latency coefficients；每步按 earliest request 的剩余 TTFT budget、decode debt、activation capacity 与 backward busy 状态决定 FT token count。[1] 完成后再把 backend 接回 CLIF local scheduler；CLIF global launcher 只消费 committed adapter version，不参与 vLLM 热路径。

## Bibliography

[1] Chen, Jiaxuan et al. (2026). “DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning for LLMs.” arXiv. https://arxiv.org/abs/2607.28848 (Retrieved: 2026-08-28)

[2] vLLM Project (2026). “Architecture Overview.” vLLM Documentation. https://docs.vllm.ai/en/latest/design/arch_overview/ (Retrieved: 2026-08-28)

[3] vLLM Project (2026). “LoRA Adapters.” vLLM Documentation. https://docs.vllm.ai/en/stable/features/lora/ (Retrieved: 2026-08-28)

[4] NVIDIA (2026). “Virtual Memory Management.” CUDA Programming Guide. https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html (Retrieved: 2026-08-28)

[5] NVIDIA (2026). “Architecture.” Multi-Process Service Documentation. https://docs.nvidia.com/deploy/mps/architecture.html (Retrieved: 2026-08-28)

[6] NVIDIA (2026). “When to Use MPS.” Multi-Process Service Documentation. https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html (Retrieved: 2026-08-28)

[7] NVIDIA (2026). “Asynchronous Execution.” CUDA Programming Guide. https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (Retrieved: 2026-08-28)

[8] PyTorch Project (2026). “Multiprocessing package — torch.multiprocessing.” PyTorch Documentation. https://docs.pytorch.org/docs/stable/multiprocessing.html (Retrieved: 2026-08-28)

[9] Hu, Edward J. et al. (2021). “LoRA: Low-Rank Adaptation of Large Language Models.” arXiv. https://arxiv.org/abs/2106.09685 (Retrieved: 2026-08-28)

[10] Kwon, Woosuk et al. (2023). “Efficient Memory Management for Large Language Model Serving with PagedAttention.” arXiv. https://arxiv.org/abs/2309.06180 (Retrieved: 2026-08-28)

[11] Sheng, Ying et al. (2023). “S-LoRA: Serving Thousands of Concurrent LoRA Adapters.” arXiv. https://arxiv.org/abs/2311.03285 (Retrieved: 2026-08-28)

[12] Miao, Xupeng et al. (2024). “FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning.” arXiv. https://arxiv.org/abs/2402.18789 (Retrieved: 2026-08-28)

[13] Chen, Lequn et al. (2023). “Punica: Multi-Tenant LoRA Serving.” arXiv. https://arxiv.org/abs/2310.18547 (Retrieved: 2026-08-28)

[14] CLIF local experiment (2026). “Qwen3-0.6B DeltaServe Hooked Integration Result.” [hooked-result.json](../DeltaServe_vLLM_single_replica_Research_20260827/hooked-result.json) (Retrieved: 2026-08-28)

[15] CLIF local experiment (2026). “Qwen3-0.6B DeltaServe Hooked Runtime Trace.” [hooked-trace.jsonl](../DeltaServe_vLLM_single_replica_Research_20260827/hooked-trace.jsonl) (Retrieved: 2026-08-28)

[16] CLIF prototype source (2026). “DeltaServe vLLM Runtime Prototype.” [deltaserve_vllm_runtime.py](../../engine/deltaserve_vllm_runtime.py) (Retrieved: 2026-08-28)

## Methodology Appendix

研究模式为 standard。Phase 1 将问题拆成三个功能不变量与一个性能问题；Phase 2 将证据分为论文目标、官方 runtime 语义和本机 execution trace；Phase 3 检索并登记 16 个来源；Phase 4 对 same-forward、shared allocation、separate PID 与 GPU process scheduling 分别交叉核验；Phase 4.5 根据实测结果把原先“是否可行”的提纲调整为“功能已通、性能未证”；Phase 5 综合形成分阶段路线；Phase 6-7 对“无干扰”措辞做反证检查；Phase 8 生成报告、source registry、evidence store 与 claim ledger。

核心结论的置信度如下：C1“最小 DeltaServe 原理路径已通”为高置信度，由论文定义、代码和本机 trace 支持。[1][14][15][16] C2“当前不能宣称无干扰高效”为高置信度，由 NVIDIA 多进程调度文档、MPS/QoS 文档、CUDA stream 限制和代码同步点支持。[5][6][7][16] C3“完整 all-layer LoRA 仍需 activation contract 与每层 backward”为中高置信度，由 LoRA 定义、DeltaServe 设计与代码检查支持。[1][9][16]

### Claims-Evidence Table

| Claim ID | Major Claim | Supporting Sources | Confidence |
|---|---|---|---|
| C1 | LM-head 最小原理路径已经实现 same-forward、shared allocation 与 separate backward PID | [1], [14], [15], [16] | High |
| C2 | 当前尚不能宣称无干扰高效并发 | [5], [6], [7], [15], [16] | High |
| C3 | 完整 all-layer LoRA 仍需 layer activation contract 与更完整 backward worker | [1], [9], [11], [13], [16] | Medium-High |
| C4 | vLLM scheduler/model-runner 是合适但需要版本化维护的集成点 | [2], [3], [10], [16] | High |

来源构成为 6 篇学术论文、7 份官方技术文档和 3 份本地实验/源码证据。未使用新闻或二手博客来支撑核心技术结论。报告的主要局限是没有 Nsight kernel timeline 与长尾 latency benchmark，因此没有对效率作数值承诺。
