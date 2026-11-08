本课说明怎样可靠执行循环计算，并比较能力与成本。先用小规模参考运行核对训练状态和并行归约，再分析缓存、生成延迟与最终评测。各阶段使用同一套版本记录，便于定位收益和开销的来源。

[返回后训练研究工作台](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
> **阅读目标：**分别定义显存、计算量、耗时和评测的统计口径。先沿参考算法测量，再用逐题记录连接能力结果与实际成本。

# 两张 A100 的验证顺序

验证按三步展开：固定短前缀、短序列和小批次，建立单路径数值参考；比较两卡归约；最后扩大上下文和生成并发。模型规模由显存实测决定，历史 4B 配方不是必要前提。第一轮建议将最大深度设为 2，再逐步增加深度与长度。以下是待运行的验证设计。

官方 4B 示例中的 `data.dp=8` 与多卡启动规模相配。若单节点两卡使用 `TP=1`，应令 `DP=2`；若 `TP=2`，则 `DP=1`。还需相应调整全局样本数、token budget 和数据分配，不能只改启动卡数。[训练入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py) · [4B 示例](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml)

# 并行时哪些量求和

| 部分 | 源码入口 | 核对重点 |
| --- | --- | --- |
| 模型切分与主权重 | [tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 `apply_tensor_parallel`、`apply_fsdp` | TP 切分一份模型计算；FSDP2 沿 DP 网格分片；计算和归约精度分别配置 |
| 全局 token 分母 | [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) 的 `train_model_tp` | 在 DP 组汇总，TP 组内不重复计数 |
| 复制组件梯度 | [tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 `sync_replicated_grads` | 对本应相同的 TP 复制梯度求均值，减少副本数值漂移；词表分片梯度的 SUM 由词表并行算子的自动微分规则完成 |
| 自适应控制流 | [iter_label.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py) 与 [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) | 某卡没有活动 token 时仍保持约定的 collective 调用顺序 |
| 重计算与缓存 | [causal_cache.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py) 的 `_TaHCacheReadOnly` | backward 重算不应把 KV 再追加一次 |

该版本 `apply_fsdp` 保留 FP32 主参数，以 BF16 计算、FP32 归约。全局 token 分母和 DP 平均梯度的补偿推导见[一次参数更新](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647)；不要再次除以累积次数。[PyTorch 2.9 FSDP2](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html)用于补充底层机制。

先确定每项 collective 所属的通信组，再检查调用顺序。正 CE 改善量的 coverage 候选在 DP 组汇总；路由和 label occupancy 使用 `tah_sync_group`，固定版本将其限定为 TP 组，TP=1 时就是单 rank。即使某个 rank 没有活动 token，也必须遵循所属组约定的 collective 顺序。 [通信组](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py#L36) · [标签汇总](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L110) · [词表伴随规则](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L1)

## 为什么三种 TP 归约需要不同的反向规则

词表并行中的 collective 不能只看前向输出是否相同，还要明确下游代表一个逻辑损失还是多份不同的分片贡献。固定源码把这三种情形分别实现为独立 autograd 函数：[vocab_parallel.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L26)。

| 逻辑用途 | 前向 | 反向 | 固定源码 |
| --- | --- | --- | --- |
| 共享输入被不同词表分片使用 | I | SUM | `_CopyToVocabParallel` |
| 各分片贡献合成一个被复制的逻辑输出 | SUM | I | `_ReduceFromVocabParallel` |
| 合成输出被各分片不同的损失项使用 | SUM | SUM | `_SumAcrossVocabParallel` |
| 拼接词表分片并查询对应分片梯度 | all-gather concat | local slice | `_GatherFromVocabParallel` |

以共享标量 u 为例，不同词表分片产生不同损失项，因此复制输入的反向必须收集全部贡献：

```math
L(u)=\sum_{r=1}^{P}\ell_r(u)
\quad\Longrightarrow\quad
\frac{dL}{du}=\sum_{r=1}^{P}\ell_r'(u).
```

若 S 是各分片贡献之和，一个逻辑损失 ℓ(S) 的反向只需把同一导数送回每个贡献；但当下游是各分片不同的损失项之和时，还需要对这些导数求和。

```math
S=\sum_r s_r,\qquad
\frac{\partial\ell(S)}{\partial s_j}=\ell'(S),\qquad
\frac{\partial\sum_r\ell_r(S)}{\partial s_j}
=\sum_r\ell_r'(S).
```

表中的 I/SUM 建立在“一个逻辑值在多卡复制”的损失计数约定上，不能直接推广为任意多输出张量映射的伴随。具体到实现，`logsumexp_lean` 对应复制的 normalizer，而 `log_softmax` 的分片消费者需要另一条反向归约。全词表混合路径中的 `copy_to_shards` 用于共享 log-weight，负责聚合各词表分片对 gate 权重的贡献。 [normalizer 路径](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L111)、[混合权重复制](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L600)

最后的 `sync_replicated_grads` 对副本梯度取 MEAN，不能替代前面按词表贡献所需的 SUM。例如两分片局部导数为 1 与 2，逻辑梯度为 3；直接把局部导数平均会错成 1.5。若前面的伴随已使各副本都持有 3，再取平均仍是 3。这是代数核对，不是已执行的多 GPU 一致性测试。[TP 梯度同步](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py)

**检查 TP 转换后的实际配置。** runtime 拒绝同时启用 `vocab_parallel_ops` 与模型级 `memory_lean_fp32_reductions=True`。正常调用 `apply_tensor_parallel` 时，词表适配路径会在 `_apply_vocab_parallel` 中把这个模型级 flag 清为 `False`，所以配方原始值不足以判断 TP 能否执行。绕过适配器、手动组合这两个状态时，保护仍会触发。词表工具内部的 `logsumexp_lean` 属于另一层接口，需要与模型级开关分别记录。 [TP 配置转换](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py#L133)、[runtime 保护](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L392)。

# 显存与计算成本

## 静态训练状态

训练显存分别记录参数、梯度、优化器、活动值、KV、读出和通信工作区。以 FP32 参数、梯度与两个 FP32 Adam 状态作简化估算，分片前静态总量为：

```math
M_{\rm static}\approx16n_{\rm trainable}\ {\rm bytes}.
```

按上述简化估算，4B 个可训练参数对应的静态训练状态约占 64 GB（十进制），还未计入 BF16 工作副本、活动值和临时缓冲。计算每卡占用时，应区分各状态实际采用的 shard／replicate 维度，不能直接除以总 GPU 数。

## KV 与混合累加器

```math
M_{\rm KV,live}\approx
\sum_{\ell}2b_\ell H_{{\rm KV},\ell}^{\rm local}d_{h,\ell}N_{{\rm live},\ell}.
```

`b` 为每元素字节数，`H_KV^local` 为每卡实际保存的 KV head 数，`d_h` 为每头维度；`N_live` 计实际位置与深度条目，循环下不等于文字长度。GQA 使用 KV head 数。已分配池由容量决定，减少逻辑活 KV 不一定降低 allocated 显存。

miniSGL 还为请求槽分配 hidden、remaining 及 mix 模式下完整词表累加器。若最大运行请求数为 `R`、hidden 维为 `H`、词表为 `V`，仅混合累加器就约为 `b_mix(R+1)V` 字节，精度受 `TAH_MIX_FP32` 影响，需与 KV 另列。[状态池](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L260)

## 有效计算量与内核开销

先按阶段拆分成本：训练包括主前向、无梯度 lookahead、反向、重计算、教师评分与同步，生成包括 prefill 和 decode。下面只估算 attention 中 QK 与 AV 的算法 FLOPs：一次乘加计两次 FLOPs，并只计算有效、可见的 query–key 配对。内核实际执行还可能包含 padding 或被遮蔽 tile 的额外工作，因此应与这个估算分别记录：

```math
F_{\rm attn}\approx4H_qd_h\sum_{q\in\mathcal A}|\mathcal V(q)|.
```

`H_q` 为 query head 数，`A` 是单层实际执行查询集合，`V(q)` 是其可见 KV 集合；跨层、轮次求和后，另加投影、MLP、softmax 与词表读出。稀疏执行还包含索引和调度成本。平均深度没有刻画可见 KV 长度、探测和利用率，因此不能直接换算成 FLOPs 或延迟。

**分别登记逻辑次数与后端实际工作。** 论文用 updater 的 m−1 次、gate 的 min(m,M−1) 次描述逻辑依赖。当前 MiniSGL fused decode 图则先在全批计算 updater，再以 `where(iterN_mask, updated, base)` 选择输入，并无条件执行 decider。因此，首轮未采用的 updater 结果和末轮后丢弃的 gate 结果，也可能消耗矩阵乘法与内核工作；估计后端成本时需要把这些执行计入。 [fused graph 源码](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/graph.py#L226) · [生成页的后端核对](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)。训练端同样应分别记录真实活动集合、已存状态集合、missing 触发的探测，以及为 capture/side-next 额外执行的 updater；集合并集不等于某层真实继续集合，详见[内部循环的辅助探测](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811)。

# 从参考路径到优化路径

先对照 SDPA 参考路径与 Triton 路径的可见性、概率和梯度，再检查缓存与后端优化。先固定执行深度比较数值，随后放开 gate 比较阈值边缘的路由变化。一个内核更快，只有在相同数学目标和硬件测量条件下才是可归因的系统收益。

## 概率接近不保证阈值路由和最终输出接近

固定同一个前驱状态，设两个后端 gate logits 之差不超过 ε，阈值 τ 严格在 0 与 1 之间。由 sigmoid 单调性，以下 margin 条件保证本次继续/停止决策一致：

```math
|a-b|\le\varepsilon,\qquad
\left|a-\log\frac{\tau}{1-\tau}\right|>\varepsilon
\quad\Longrightarrow\quad
\mathbf1[\sigma(a)>\tau]=\mathbf1[\sigma(b)>\tau].
```

只有此前路由仍相同，才能沿执行顺序继续应用此条件；路径一旦分叉，后续 KV 和隐状态也可能不同，不能继续假定比较的是同一状态。

实际停止轮吸收剩余混合质量，进一步带来阈值两侧的输出不连续。取两轮分布 q₁=(0.9,0.1)、q₂=(0.1,0.9)，阈值 0.5：若继续概率 c 小于阈值，D=1，输出就是 q₁；若 c 大于阈值，D=2，输出为 (1−c)q₁+cq₂。c 从 0.499999 变成 0.500001 时，变化仅 2×10⁻⁶，但两个输出的总变差距离约为 0.4000008。[阈值决策](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L252)、[实际停止吸收尾质量](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L562)

因此后端一致性应先固定路线比较条件概率，再放开实际 gate，记录每一步 margin、决定和输出差异。随机 gate 在同一个状态下若使用同一个均匀随机数 U，可证明两次 Bernoulli 决策不同的概率是两个继续概率之差的绝对值；但全局 seed 相同不保证不同批调度下每个请求消耗同一个 U。

“策略版本”至少包含骨干、updater、decider、tokenizer/模板、mask 与位置约定、mixture_mode、threshold/max_iter、warmup/随机选项/实际 eval decider、温度/截断，以及数值精度与后端。仅权重 hash 相同不足以证明策略相同。特别是 `even_mix` 在不同后端的深度约定需逐项核验，不能因模式名字相同就省略执行语义。

## 服务负载也是自适应策略的条件

固定 MiniSGL 的 DUO 页预算保护按 continue_indices 的批次顺序保留前 keep_n 行，超出者使用当前轮已经采样的候选。因此在相同权重、同一问题和相同 gate 规则下，其他请求占用的页数与本请求在批内的位置也能改变它是否继续。这是明确的控制流依赖，不是已经测得的负载下性能结论。[页预算保护](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L600-L625)

令 s 为模型状态，b 为服务状态，u 为已声明的随机输入；实际路由应表示为二级决策：

```math
\begin{aligned}
c^{\rm raw}&=\operatorname{Decide}_{\theta}(s,u),\\
c^{\rm exec}&=\operatorname{Admit}(c^{\rm raw},b),\\
(s',b')&=\operatorname{Step}_{\theta}(s,b,c^{\rm exec},u).
\end{aligned}
```

只有在不触发资源覆盖且批次算子/RNG 影响得到控制时，才可以省略 b，声称在测量同一个仅依赖模型状态的策略。扩容带来的吞吐变化和强停带来的生成分布变化需要分别记录。将强停当作自主 gate 的停止，会同时污染路由校准、平均深度和成本—能力归因。

| 记录对象 | 至少保留的字段 | 用来排除的混淆 |
| --- | --- | --- |
| 模型决策 | gate logit／概率、阈值或随机事件、深度上限、warmup／override | 未继续究竟来自模型还是配置 |
| 服务覆盖 | 批次顺序、请求 ID、可用循环页槽、raw 与 exec 掩码、覆盖原因 | 资源不足被误记为自主早退 |
| 读出与随机性 | remaining、停止分支、温度／截断、逻辑请求与随机抽样对应 | 同 seed 被误当成同一随机实验 |
| 后续状态 | 各深度 KV 条目、位置与有效性、请求结束和释放事件 | 同文字前缀被误当成同一模型状态 |

后续运行按三组配对进行：先在资源充足且固定路由时验证后端数值；再放开模型 gate、维持无资源强停；最后固定请求集合改变并发和排序，单列资源覆盖率与逐题能力—成本。前两组提供模型与内核的对照，第三组测量服务策略本身。当前这里只把测量协议与字段落入工程文档，尚未执行 GPU 服务压力测试。[当前 token 的强停界及序列条件](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)

## 计时与吞吐口径

[torch.profiler 2.9](https://docs.pytorch.org/docs/2.9/profiler.html)用于定位 attention、MLP、词表读出、通信和 CPU 调度。

测量固定硬件、dtype、内核、长度、并发和路由；

加载、首次编译与 warmup 单列。

端到端时间采用 GPU 同步边界并取多卡最大耗时；CUDA events 用于另外标明的设备区间。下式 `D` 为 DP 副本数，全局监督数只跨 DP 汇总。

```math
\begin{aligned}
T_{\rm update}&=\max_r\Delta t_r,\\
N_{\rm shifted}^{\rm global}&=\sum_{r=1}^{D}N_{r,\rm shifted},\\
{\rm supervised\ tokens/s}&=\frac{N_{\rm shifted}^{\rm global}}{T_{\rm update}}.
\end{aligned}
```

## 算法 9 更新窗口的性能测量

**过程** MeasureWindow(protocol)

**输入** 包含固定样本、版本、生效配置、预热规则及测量起止点的测量协议。

**输出** 各窗口耗时、各卡显存峰值、监督计数、深度、KV 与阶段耗时的原始记录和统计。

![算法 9 操作与数学公式](visualize:fde1_bGliZmlsZV8wOXdZMnZGNzM5NVVpTUhYYVNGR2RB_FileDrive_45288dfe06608191af94334ba8699511)

源码现有 `peak_mem_gib` 为 rank 0 的 `max_memory_allocated/2³⁰`，训练循环内未逐更新 reset；它是自最后一次重置以来的峰值，不能直接当作本次更新峰值。上述算法是拟补测量协议。[日志位置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L808)

| 测量对象 | 拟记录的量 |
| --- | --- |
| 一次训练更新 | 监督 token/s、更新时长、各阶段时间、峰值 allocated／reserved 显存、GPU·h |
| 单请求生成 | TTFT、decode ms/token、端到端时间及 p50／p95 |
| 服务吞吐 | 固定并发下 requests/s 与 output tokens/s |
| 循环与缓存 | prefill／decode 各深度计数、KV 条目、主路径／探测成本 |
| 版本切换 | actor 导出、传输、rollout 重载时间与概率差异 |

# 保存模型与恢复训练分开验收

[simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) 的 `save_model_tp` 导出 BF16 模型和循环组件；当 `save_only_model=false` 时，DCP 另存主权重和优化器状态。可推理的导出与可接续的训练状态分别检查；想复现后续轨迹，还要核对 scheduler、随机数、采样器、数据游标与配置是否完整恢复。

使用同一批次比较“连续执行”和“保存后重载继续”的下一次损失、参数变化与数据身份，声明比较精度和容差。没有这些记录时，只报告模型可加载或训练可继续，不声称逐步轨迹完全复现。

## 从“能加载”推进到“恢复同一个更新状态”

把一次优化窗口写成状态转移，可准确说明 checkpoint 需要保存什么。S 不仅包含模型权重，还包含所有会改变下一次更新的内部状态；B 是下一窗口数据，ξ 是所需随机量。

```math
S_{k+1}=\Phi(S_k,B_k,\xi_k).
```

若只保存 θ，Adam 动量、step、scheduler、数据次序或随机数状态不同，下一次更新就可能不同。对于在线后训练，rollout 的模型版本、在途任务与待消费轨迹还属于额外状态。应区分“恢复同一算法配置并继续训练”和“复现连续执行的下一步轨迹”。

| 固定源码路径 | 已核对的实际状态 | 不能据此直接推出的结论 |
| --- | --- | --- |
| TaH2 `simple_loop.py` | 非 `save_only_model` 路径使用 DCP 保存 FP32 主权重及 optimizer；另存 global_step、epoch、windows_per_epoch、DP/TP 信息 | 所示 trainer payload 没有包含 RNG；只导出 BF16 模型也不等于恢复主权重和动量 |
| TaH2 resume / `_RankStridedBatches` | 依据 global step、epoch 和窗口位置跳过既有批次；新建 scheduler 后推进到对应 step | 相同窗口数不保证相同样本 ID；还要求采样次序、切分拓扑和 schedule 配置兼容 |
| verl FSDP checkpoint manager | 按 contents 配置保存/加载 model、optimizer、extra；extra 包含 scheduler 与 RNG | 配置未保存或未加载某类状态时，不存在自动补齐的保证 |
| verl V1 trainer | 保存 actor、可选 critic、StatefulDataLoader；异步模式还保存 TransferQueue | pending/running prompt 重发是新生成尝试，不等于从中间 token 及原 RNG 精确续跑 |
| TaH2 resume达到正max_steps | 若仍有epoch窗口，阈值在optimizer.step之后检查，可能再更新一次 | 不能把恢复进度≥阈值等同于源码已在首个更新前退出；预算入口guard见更新页 |

对应调用链：[TaH2 保存与恢复](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L497)、[verl FSDP load/save](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/utils/checkpoint/fsdp_checkpoint_manager.py#L265)、[V1 数据与队列恢复](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py#L850)。

V1 的 `_reissue_inflight_prompts` 会清除待重发 prompt 对应的旧 trajectory keys，并重新生成；已经完成的轨迹按已保存状态恢复。因此异步恢复应把“重发次数、生成版本、旧轨迹是否删除”纳入记录，不能把 replay buffer 恢复与模型内部 KV 恢复混为一谈。[重发逻辑](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py#L910)

## 算法 T7 两轮同步与恢复状态核验

**过程** VerifyTwoUpdatesAndResume(learner, rollout_backend, checkpoint_config, fixed_protocol)

**输入** 学习端完整状态、rollout 后端、实际 checkpoint contents 配置、固定的评分/采样/数据协议及数值容差。

**输出** 两轮版本流、actor 与 rollout 概率差、连续与恢复分支的下一步损失/梯度/参数差，以及缺失状态清单。

**前置条件与终止** 在无梯度累积残留、无在途 optimizer 更新的边界保存；同步参考场景先清空在途请求。随机 gate 若开启，必须具有足够的动作和随机状态日志，否则只报告条件评分一致性。算法只跑两轮更新与一次恢复对照；遇到版本、支持或状态缺失时返回明确诊断，不无限重试。本页提供的是后续工程核验协议，本轮未执行模型训练。

![算法 T7 两轮同步与恢复状态核验](visualize:fde1_bGliZmlsZV9VTGxhTkNEMUlIRkJXVS1DQUNObFZR_FileDrive_45288dfe06608191af94334ba8699511)

**函数来源与实现状态** `trainer_sync` 在初始化及 step 结束调用权重同步；`engine_workers.update_weights` 负责学习端到生成端的参数传输和生成资源恢复；checkpoint 分别沿上述 manager 与 trainer 实现。TaH2 的 updater/decider 及执行配置是否被实际传输仍是待完成的适配项，不能由 verl 原生模型路径推断已经支持。[同步钩子](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_sync.py#L31)、[参数同步入口](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/engine_workers.py#L789)。

# 能力评测与机制结论

[eval_online.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py) 的 `run_online_eval`、`_online_runtime_metrics` 用于定位生成、评分和吞吐记录。Bundled MiniSGL 的响应提供 `usage.iter_counts`，DUO 另有 `usage.prompt_iter_counts`；字段范围须对照服务与客户端一起核实。[引擎说明](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md)

比较先采用同初始化、同源数据的单遍、固定循环与自适应循环；按训练预算与推理预算分别控制，保留原始逐题输出。准确率、文字长度、prefill／decode 计算和端到端时间一起报告。后续少文字方案再与合理调优的文字推理对照。

## Teacher forcing 验证：实际执行集合与损失分母

验证损失本来要回答一个简单问题：实际预测过的这些目标，平均损失是多少？分子需要加上它们的损失，分母也应数同一批目标。固定 `train_model_tp` 的两步处理却可能让范围错开：它先在整个 processed eval 数据上计算 `eval_num_items` 和 `eval_gbs`，再把 DataLoader 批次数向下截到实际 DP 数的整数倍。

沿用下表的控制例：952 条处理后样本，每条恰有一个移位后的有效目标，且每个目标损失固定为 2。batch size 为 2、DP 为 8 时，先形成 476 个批次，实际只执行其中 472 个，也就是 944 个目标。执行损失和为 1888；若仍除以预先统计的 952，日志就是约 1.983，尽管每个实际目标的损失都是 2。这里变化的是汇总方式，预测损失本身已被固定。

源码按 batch 索引模 DP 数分配给各 rank，最后对各 rank 的 loss 做 DP SUM。下面的公式把“预统计的目标”和“实际执行的目标”分别写出，就能看到这个差值从哪里来。分析采用默认 recipe 的 `NextTokenPredLoss`；任意 CombinedLoss 的组成与归一化需要分别检查。[eval 初始化与执行](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L620-L705) · [1.7B 配方](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_1.7/sft_tah.yaml)

设 D 是进入 DataLoader 的有序样本出现记录，N=|D|；b 是 eval batch size，P 是实际 DP 数。B* 之后的 batch 不进入模型，不能把其目标 token 仍留在“已执行目标的平均损失”分母中。

```math
B=\left\lceil\frac Nb\right\rceil,\qquad
B_*=P\left\lfloor\frac BP\right\rfloor,\qquad
\mathcal E=\{0,\ldots,B_*-1\}.
```

先区分预统计的分母与实际执行的目标。源码 helper 从 D 的未移位 `labels` 中统计全部非 −100 位置，记为 U_D。对实际执行的 batch j，按 collator 输出做因果移位，得到有效目标集 I_j；记 n_j=|I_j|，s_j 为同一损失分支在这些位置上的损失和。这里数的是 token，有重复样本时按每次出现计数，不能用行数替代。下式采用实数算术，L_raw 表示格式化前的 `loss_sum`；源码 FP32 累加误差和日志格式化另外记录。[helper 的真实分母](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_runtime.py#L425-L444) · [因果移位](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L380-L388) · [token 损失](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

```math
\begin{aligned}
U_D&=\sum_{z\in D}\sum_t\mathbf1\{\ell_t(z)\ne-100\},\\
S_{\mathcal E}&=\sum_{j\in\mathcal E}s_j,\qquad
V_{\mathcal E}=\sum_{j\in\mathcal E}n_j,\\
L_{\mathrm{raw}}
&=\sum_{r=0}^{P-1}\sum_{\substack{j\in\mathcal E\\j\bmod P=r}}\frac{s_j}{U_D}
=\frac{S_{\mathcal E}}{U_D}.
\end{aligned}
```

在 U_D>0、V_E>0 时，下式是实数算术下的恒等式。U_D=V_E 足以保证 L_raw 等于执行子集的 token 均值；当 S_E≠0 时，这也是必要条件，零损失本身不能证明分母正确。对同一批 collator 输出，各行首位置均为 −100 可保证未移位与移位计数一致；仍需另外核对执行覆盖范围。这里没有再除 DP 数：最终归约是 SUM。

```math
L_{\mathcal E}=\frac{S_{\mathcal E}}{V_{\mathcal E}},
\qquad
L_{\mathrm{raw}}
=\frac{V_{\mathcal E}}{U_D}\,L_{\mathcal E}.
```

进一步固定逐目标预测与损失，并假设 D 的未移位计数等于其全部移位监督数 V_D。把未执行部分的损失和记作 S_drop，则可以比较“完整 D 的目标”与源码日志：

```math
L_D=\frac{S_{\mathcal E}+S_{\mathrm{drop}}}{U_D},
\qquad
L_D-L_{\mathrm{raw}}=\frac{S_{\mathrm{drop}}}{U_D}.
```

归一化预测分布的非负 NLL 情形下，遗漏目标会使这一日志偏低。恒等式本身不要求损失非负；本课程已经单独说明 stop_prob_mix 的数值下限和三个损失消费分支，因此不能对所有实现值无条件宣称严格向下偏差。只修分母为 V_E 得到的是执行子集均值，仍不能恢复未执行部分的预测或完整 D 的均值。

### 手算反例与默认配置的边界

下表固定 952 条处理后样本，每条在因果移位后仍恰有一个有效目标，冻结所有目标损失为 2。它只检验调度与分母，不代表任何模型性能实测。

| 实际 DP P | eval batch b | DataLoader 批数 B | 执行批数 B* | 执行样本数 | 理想实数复算 L_raw | 执行子集均值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 952 | 952 | 952 | 2 | 2 |
| 8 | 2 | 476 | 472 | 944 | 1.9831932773 | 2 |
| 4 | 2 | 476 | 476 | 952 | 2 | 2 |

表中的关键是：952 个样本能被 8 整除，但 batch size 为 2 时形成的 476 个批次不能。官方 1.7B recipe 设置 `per_device_eval_batch_size=1`、`data.dp=8`；实际 P=8、采用正常 drop 处理且数据非空时，不触发这里的尾部批次问题，改变 b 或实际拓扑后则需重查。再看更小的例子：N=8、b=2、P=8 时，B*=0，没有任何批次进入模型，初始 `loss_sum` 归约后仍为 0。这个 0 不表示零验证误差，因为执行子集均值未定义。另一种情况是整份数据 U_D=0：helper 返回 `None`，初始化在 `float(None)` 处先报错，应与“分母为正但没有执行批次”区分。

本次重新提取固定源码的两个计数 helper、整除函数、label 移位、NextTokenPredLoss、eval 初始化、`_run_eval` 及日志格式化函数，八组检查全部通过；附加 cosine helper 检查也通过。NumPy 张量、右侧 padding、固定逐目标 NLL 和串行 DP SUM 是明确替身；真实 tokenizer／collator、模型预测、路由、GPU、FSDP／TP 和 distributed collective 均未执行。12 份源码快照逐一核对 Git blob SHA 与 SHA-256，检查包包含源文件、可运行脚本和 JSON 结果。

源码在 SUM 后以 `.4g` 格式化 loss：上表 1.9831932773 是理想实数复算，实际显示为 1.983；FP32 原始值还可能有累加舍入差。应同时保留分子、分母、未格式化值和显示值，不能把显示精度解释为估计误差。

### 八组检查与可运行记录

| 组 | 被核对的边界 | 通过的关键结果 |
| ---: | --- | --- |
| 1 | 955 行经 q=8 drop；b=1、P=8 | N=952，全部执行，L_raw=L_E=2 |
| 2 | 同数据改 b=2 | B=476、B*=472、执行944条；理想 L_raw=1888/952，显示1.983 |
| 3 | b=2、实际 P=4 | 全476批执行，L_raw=L_E=2 |
| 4 | 有效目标数 1／2／3／4／10 | 实际保留10/20个目标，L_raw=1.5、L_E=3；行比例不能代替token比例 |
| 5 | 首原始标签未mask | U_D=8、V_E=4；L_raw=1、L_E=2 |
| 6 | 最后一批不满 | N=5、b=2：P=3完整保留；P=2遗漏末批 |
| 7 | 零执行与零目标 | 前者返回原始零而均值未定义；后者在初始化float(None)报错 |
| 8 | pad 的出现权重与小数据 | 5条补至8条保留重复权重；1条补到q=8会产生越界索引 |

[循环研究验证与源码核对：八组检查及固定源码](library-file:fde1_bGliZmlsZV9iQVNrVGNvQk1ESkR6TlNCeGppSjZB_FileDrive_45288dfe06608191af94334ba8699511)

运行检查包中的 `python3 check_eval.py`；仅需 Python 与 NumPy，无需模型权重。上述数据均为控制流小例，未把发布集的原始 955 行视为已实测的预处理结果。

### 算法 T12 验证损失的执行记录核对

**输入** 固定数据出现记录 D、tokenizer／模板／collator／mask、b、实际 DP/TP 拓扑、checkpoint、路由与数值配置，以及固定源码的 NextTokenPredLoss 路径。

**输出** 处理后与执行后的出现记录、每批目标数、损失分子、U_D、V_E、源码日志复算值、执行子集均值，以及缺失／重复记录。

![算法 T12 验证损失的执行记录核对](visualize:fde1_bGliZmlsZV9zR0RBelllLWtKajNNSFZFTEFDS2ZB_FileDrive_45288dfe06608191af94334ba8699511)

**性质** 这是针对发布实现的参考核对算法。无有效目标返回“不可评估”是参考算法的防护，不是源码已经实现的修复；保持原有各 rank 的调用次数和 collective 顺序。若另行实现完整验证集评估，需同时设计等步数通信与零权重占位，不能只删除 islice 截断。

## 多次采样的 avg 与 pass@k

多次采样按题目组织统计。设第 `i` 题有 `n_i` 次独立同分布采样，其中 `c_i` 次正确，`n_i≥k`；avg 是样本正确率，pass@k 的常用无偏估计如下。公式可对照 [HumanEval 原论文第 2.1 节与附录 A](https://arxiv.org/html/2107.03374v2)。

```math
\begin{aligned}
\widehat{\mathrm{avg}}_i&=\frac{c_i}{n_i},\\
\widehat{\mathrm{pass@}k}_i
&=1-\frac{\binom{n_i-c_i}{k}}{\binom{n_i}{k}},\\
\binom{a}{k}&=0\quad\text{若 }a<k.
\end{aligned}
```

后者估计同一题 `k` 次尝试至少一次成功的概率；独立且单次成功率为 `p_i` 时为 `1−(1−p_i)^k`。先逐题计算再平均，不把全数据的平均正确率直接代入非线性公式。同一题多采样与多种子保留聚类结构；置信分析声明是否还包含训练种子变化。调优与最终确认集继续分开。

**机制证据的解释范围**

机制分析需要在相近成本下，结合状态替换、关闭额外轮次、扰动历史 KV 或改变后续操作的干预，判断计算依赖。准确率提高、范数变稳或 probe 更易解码，单独一项都不足以确定内部保存了何种可复用知识。

当前已有固定源码和测量协议，数值一致性、两卡运行和性能结果均待实现验收。

# 交付与验收

- [ ] `parity_report.json`：同前缀、同权重、同深度下的概率与梯度差异，随后比较自主 gate。

- [ ] `checkpoint_report.json`：模型导出与训练恢复分别给出实际结果。

- [ ] `cost_profile.json`：上述阶段、显存和深度统计有明确测量边界。

- [ ] `eval_records.jsonl`：逐题原始输出、评分状态、预算、版本和题目身份完整。

- [ ] `comparison_report.md`：能力与成本曲线及机制证据，区分实测、估算和未验证解释。

# 围绕本课讨论

```codex-prompt
请从两张 A100 的资源约束出发，对照 TaH2 的 TP、FSDP2、全局 token 分母、自适应控制流和缓存重计算。随后为同一小型案例设计 parity、checkpoint、cost 和逐题评测记录；先固定深度定位数值差异，再放开 gate。所有显存与速度结论以实测为准。
```
