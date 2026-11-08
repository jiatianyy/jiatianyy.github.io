---
title: "分布式执行与性能评测"
description: "本课说明怎样可靠执行循环计算，并比较能力与成本。先用小规模参考运行核对训练状态和并行归约，再分析缓存、生成延迟与最终评测。各阶段使用同一套版本记录，便于定位收益和开销的来源。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c"
sourcePageId: "page_d887d8f1a41881918ecff055e7613c9c"
sourceSha256: 3acf78646d8289cb6bfa794bb52e120b8b6b7ae813041b54d16ae5097210406d
series: loop-reasoning
references: []
---

<span id="source-block-303421fc-4014-4b38-8702-191b347d98fb"></span>

本课说明怎样可靠执行循环计算，并比较能力与成本。先用小规模参考运行核对训练状态和并行归约，再分析缓存、生成延迟与最终评测。各阶段使用同一套版本记录，便于定位收益和开销的来源。

<span id="source-block-92cb032f-2580-4451-9e14-2cd0dd1b04cb"></span>

[返回后训练研究工作台](/notes/loop-reasoning/engineering/)

<span id="source-block-4b3aa6ac-3e77-458a-a958-6ccfb4a417a8"></span>

> [!CALLOUT]
> **阅读目标：**分别定义显存、计算量、耗时和评测的统计口径。先沿参考算法测量，再用逐题记录连接能力结果与实际成本。

<span id="source-block-dcc9c7d7-a278-4093-bf1a-24cc98c159ab"></span>

## 两张 A100 的验证顺序

<span id="source-block-3b73f8ed-3d09-4ae0-b2b8-4b0f08ca66a6"></span>

验证按三步展开：固定短前缀、短序列和小批次，建立单路径数值参考；比较两卡归约；最后扩大上下文和生成并发。模型规模由显存实测决定，历史 4B 配方不是必要前提。第一轮建议将最大深度设为 2，再逐步增加深度与长度。以下是待运行的验证设计。

<span id="source-block-2b12b91f-e8e5-4326-8aeb-ffbfe404ee5a"></span>

官方 4B 示例中的 `data.dp=8` 与多卡启动规模相配。若单节点两卡使用 `TP=1`，应令 `DP=2`；若 `TP=2`，则 `DP=1`。还需相应调整全局样本数、token budget 和数据分配，不能只改启动卡数。[训练入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py) · [4B 示例](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml)

<span id="source-block-21fb203c-69a8-4b59-bfb1-dda8eff23441"></span>

## 并行时哪些量求和

<span id="source-block-836b91f7-be88-4ebd-be58-a85afec30295"></span>

| 部分 | 源码入口 | 核对重点 |
| --- | --- | --- |
| 模型切分与主权重 | [tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 `apply_tensor_parallel`、`apply_fsdp` | TP 切分一份模型计算；FSDP2 沿 DP 网格分片；计算和归约精度分别配置 |
| 全局 token 分母 | [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) 的 `train_model_tp` | 在 DP 组汇总，TP 组内不重复计数 |
| 复制组件梯度 | [tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 `sync_replicated_grads` | 对本应相同的 TP 复制梯度求均值，减少副本数值漂移；词表分片梯度的 SUM 由词表并行算子的自动微分规则完成 |
| 自适应控制流 | [iter_label.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py) 与 [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) | 某卡没有活动 token 时仍保持约定的 collective 调用顺序 |
| 重计算与缓存 | [causal_cache.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py) 的 `_TaHCacheReadOnly` | backward 重算不应把 KV 再追加一次 |

<span id="source-block-a218a3de-4ee8-462f-97db-990759f563b9"></span>

该版本 `apply_fsdp` 保留 FP32 主参数，以 BF16 计算、FP32 归约。全局 token 分母和 DP 平均梯度的补偿推导见[一次参数更新](/notes/loop-reasoning/parameter-update/)；不要再次除以累积次数。[PyTorch 2.9 FSDP2](https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html)用于补充底层机制。

<span id="source-block-06f28ec4-3717-4234-90e1-07f04cf72529"></span>

先确定每项 collective 所属的通信组，再检查调用顺序。正 CE 改善量的 coverage 候选在 DP 组汇总；路由和 label occupancy 使用 `tah_sync_group`，固定版本将其限定为 TP 组，TP=1 时就是单 rank。即使某个 rank 没有活动 token，也必须遵循所属组约定的 collective 顺序。 [通信组](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py#L36) · [标签汇总](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L110) · [词表伴随规则](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L1)

<span id="source-block-7216d639-260c-4bb5-a280-5d921c45b1a3"></span>

### 为什么三种 TP 归约需要不同的反向规则

<span id="source-block-8e78a7c2-64a1-4b6e-93fd-c203044a4c4a"></span>

词表并行中的 collective 不能只看前向输出是否相同，还要明确下游代表一个逻辑损失还是多份不同的分片贡献。固定源码把这三种情形分别实现为独立 autograd 函数：[vocab_parallel.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L26)。

<span id="source-block-95db2fe8-89fe-4209-ae24-54192c3ffb82"></span>

| 逻辑用途 | 前向 | 反向 | 固定源码 |
| --- | --- | --- | --- |
| 共享输入被不同词表分片使用 | I | SUM | `_CopyToVocabParallel` |
| 各分片贡献合成一个被复制的逻辑输出 | SUM | I | `_ReduceFromVocabParallel` |
| 合成输出被各分片不同的损失项使用 | SUM | SUM | `_SumAcrossVocabParallel` |
| 拼接词表分片并查询对应分片梯度 | all-gather concat | local slice | `_GatherFromVocabParallel` |

<span id="source-block-3eeb9f1a-12c1-4dde-b9ed-931758429998"></span>

以共享标量 u 为例，不同词表分片产生不同损失项，因此复制输入的反向必须收集全部贡献：

<span id="source-block-8ba57855-8c76-40bb-a8d8-ca4ae3a8ab1b"></span>

```math
L(u)=\sum_{r=1}^{P}\ell_r(u)
\quad\Longrightarrow\quad
\frac{dL}{du}=\sum_{r=1}^{P}\ell_r'(u).
```

<span id="source-block-0dcccaa6-8a37-4e97-a368-1230451b51b3"></span>

若 S 是各分片贡献之和，一个逻辑损失 ℓ(S) 的反向只需把同一导数送回每个贡献；但当下游是各分片不同的损失项之和时，还需要对这些导数求和。

<span id="source-block-170ecf8f-4e44-4767-a93e-61dab530098e"></span>

```math
S=\sum_r s_r,\qquad
\frac{\partial\ell(S)}{\partial s_j}=\ell'(S),\qquad
\frac{\partial\sum_r\ell_r(S)}{\partial s_j}
=\sum_r\ell_r'(S).
```

<span id="source-block-e95c0b22-77c3-4699-ba80-33008d1a8a88"></span>

表中的 I/SUM 建立在“一个逻辑值在多卡复制”的损失计数约定上，不能直接推广为任意多输出张量映射的伴随。具体到实现，`logsumexp_lean` 对应复制的 normalizer，而 `log_softmax` 的分片消费者需要另一条反向归约。全词表混合路径中的 `copy_to_shards` 用于共享 log-weight，负责聚合各词表分片对 gate 权重的贡献。 [normalizer 路径](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/vocab_parallel.py#L111)、[混合权重复制](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L600)

<span id="source-block-b86d7e0d-79a6-4f45-aa11-03964eca7e14"></span>

最后的 `sync_replicated_grads` 对副本梯度取 MEAN，不能替代前面按词表贡献所需的 SUM。例如两分片局部导数为 1 与 2，逻辑梯度为 3；直接把局部导数平均会错成 1.5。若前面的伴随已使各副本都持有 3，再取平均仍是 3。这是代数核对，不是已执行的多 GPU 一致性测试。[TP 梯度同步](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py)

<span id="source-block-6366cbfc-c998-4d9b-b356-fa07522db750"></span>

**检查 TP 转换后的实际配置。** runtime 拒绝同时启用 `vocab_parallel_ops` 与模型级 `memory_lean_fp32_reductions=True`。正常调用 `apply_tensor_parallel` 时，词表适配路径会在 `_apply_vocab_parallel` 中把这个模型级 flag 清为 `False`，所以配方原始值不足以判断 TP 能否执行。绕过适配器、手动组合这两个状态时，保护仍会触发。词表工具内部的 `logsumexp_lean` 属于另一层接口，需要与模型级开关分别记录。 [TP 配置转换](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py#L133)、[runtime 保护](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L392)。

<span id="source-block-de338bd6-40d6-454c-be8d-ac22cd0abee5"></span>

## 显存与计算成本

<span id="source-block-16313d5c-a3c1-42c2-bd45-a13852a5efcb"></span>

### 静态训练状态

<span id="source-block-25b2a68f-1a35-4ba5-b8d6-74d801bea39b"></span>

训练显存分别记录参数、梯度、优化器、活动值、KV、读出和通信工作区。以 FP32 参数、梯度与两个 FP32 Adam 状态作简化估算，分片前静态总量为：

<span id="source-block-08e8de55-d508-4fc9-9ddf-528b993c69e9"></span>

```math
M_{\rm static}\approx16n_{\rm trainable}\ {\rm bytes}.
```

<span id="source-block-3a010fb1-3a1c-41a6-bb83-442b4f158785"></span>

按上述简化估算，4B 个可训练参数对应的静态训练状态约占 64 GB（十进制），还未计入 BF16 工作副本、活动值和临时缓冲。计算每卡占用时，应区分各状态实际采用的 shard／replicate 维度，不能直接除以总 GPU 数。

<span id="source-block-89430c54-f73f-4301-a455-4913e39ad6ba"></span>

### KV 与混合累加器

<span id="source-block-cc15715c-a659-49e9-a620-ec210ed1ffe9"></span>

```math
M_{\rm KV,live}\approx
\sum_{\ell}2b_\ell H_{{\rm KV},\ell}^{\rm local}d_{h,\ell}N_{{\rm live},\ell}.
```

<span id="source-block-56bd3a4b-48e4-4253-9d33-8b978a2e9736"></span>

`b` 为每元素字节数，`H_KV^local` 为每卡实际保存的 KV head 数，`d_h` 为每头维度；`N_live` 计实际位置与深度条目，循环下不等于文字长度。GQA 使用 KV head 数。已分配池由容量决定，减少逻辑活 KV 不一定降低 allocated 显存。

<span id="source-block-2bcd0a03-3b70-4cf6-bc0e-0654dcf3a76f"></span>

miniSGL 还为请求槽分配 hidden、remaining 及 mix 模式下完整词表累加器。若最大运行请求数为 `R`、hidden 维为 `H`、词表为 `V`，仅混合累加器就约为 `b_mix(R+1)V` 字节，精度受 `TAH_MIX_FP32` 影响，需与 KV 另列。[状态池](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L260)

<span id="source-block-9d815797-6c03-4148-a716-c079c026028f"></span>

### 有效计算量与内核开销

<span id="source-block-b49cb562-44dc-4ff2-835e-7a96475fd6bd"></span>

先按阶段拆分成本：训练包括主前向、无梯度 lookahead、反向、重计算、教师评分与同步，生成包括 prefill 和 decode。下面只估算 attention 中 QK 与 AV 的算法 FLOPs：一次乘加计两次 FLOPs，并只计算有效、可见的 query–key 配对。内核实际执行还可能包含 padding 或被遮蔽 tile 的额外工作，因此应与这个估算分别记录：

<span id="source-block-8d276f1b-9774-4255-8a15-ceb02bc6e5fc"></span>

```math
F_{\rm attn}\approx4H_qd_h\sum_{q\in\mathcal A}|\mathcal V(q)|.
```

<span id="source-block-2ab69028-df37-4ad6-ae66-6deaf77bb0e5"></span>

`H_q` 为 query head 数，`A` 是单层实际执行查询集合，`V(q)` 是其可见 KV 集合；跨层、轮次求和后，另加投影、MLP、softmax 与词表读出。稀疏执行还包含索引和调度成本。平均深度没有刻画可见 KV 长度、探测和利用率，因此不能直接换算成 FLOPs 或延迟。

<span id="source-block-9c7547b6-8fab-49b9-a816-856794a1cb18"></span>

**分别登记逻辑次数与后端实际工作。** 论文用 updater 的 m−1 次、gate 的 min(m,M−1) 次描述逻辑依赖。当前 MiniSGL fused decode 图则先在全批计算 updater，再以 `where(iterN_mask, updated, base)` 选择输入，并无条件执行 decider。因此，首轮未采用的 updater 结果和末轮后丢弃的 gate 结果，也可能消耗矩阵乘法与内核工作；估计后端成本时需要把这些执行计入。 [fused graph 源码](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/graph.py#L226) · [生成页的后端核对](/notes/loop-reasoning/generation/)。训练端同样应分别记录真实活动集合、已存状态集合、missing 触发的探测，以及为 capture/side-next 额外执行的 updater；集合并集不等于某层真实继续集合，详见[内部循环的辅助探测](/notes/loop-reasoning/internal-loop/)。

<span id="source-block-6a57f512-a0cb-4667-9505-f7efdec51dd4"></span>

## 从参考路径到优化路径

<span id="source-block-410b6564-1bd7-4539-93b1-ec95fb1d04f5"></span>

先对照 SDPA 参考路径与 Triton 路径的可见性、概率和梯度，再检查缓存与后端优化。先固定执行深度比较数值，随后放开 gate 比较阈值边缘的路由变化。一个内核更快，只有在相同数学目标和硬件测量条件下才是可归因的系统收益。

<span id="source-block-740d4f10-22c8-4dcb-a9a8-bf80b728e4d0"></span>

### 概率接近不保证阈值路由和最终输出接近

<span id="source-block-dff55dd9-537c-4710-9d90-625cf6195b64"></span>

固定同一个前驱状态，设两个后端 gate logits 之差不超过 ε，阈值 τ 严格在 0 与 1 之间。由 sigmoid 单调性，以下 margin 条件保证本次继续/停止决策一致：

<span id="source-block-630dfd91-3183-4694-9117-abb4e502b9b6"></span>

```math
|a-b|\le\varepsilon,\qquad
\left|a-\log\frac{\tau}{1-\tau}\right|>\varepsilon
\quad\Longrightarrow\quad
\mathbf1[\sigma(a)>\tau]=\mathbf1[\sigma(b)>\tau].
```

<span id="source-block-6e2d81a6-3293-48b3-941d-e90e6f8c27e8"></span>

只有此前路由仍相同，才能沿执行顺序继续应用此条件；路径一旦分叉，后续 KV 和隐状态也可能不同，不能继续假定比较的是同一状态。

<span id="source-block-5e589340-b7cc-456e-9a19-db41f004d068"></span>

实际停止轮吸收剩余混合质量，进一步带来阈值两侧的输出不连续。取两轮分布 q₁=(0.9,0.1)、q₂=(0.1,0.9)，阈值 0.5：若继续概率 c 小于阈值，D=1，输出就是 q₁；若 c 大于阈值，D=2，输出为 (1−c)q₁+cq₂。c 从 0.499999 变成 0.500001 时，变化仅 2×10⁻⁶，但两个输出的总变差距离约为 0.4000008。[阈值决策](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L252)、[实际停止吸收尾质量](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L562)

<span id="source-block-52785360-2203-4f86-b75a-ac3775fcd306"></span>

因此后端一致性应先固定路线比较条件概率，再放开实际 gate，记录每一步 margin、决定和输出差异。随机 gate 在同一个状态下若使用同一个均匀随机数 U，可证明两次 Bernoulli 决策不同的概率是两个继续概率之差的绝对值；但全局 seed 相同不保证不同批调度下每个请求消耗同一个 U。

<span id="source-block-b817a788-6772-48bc-af99-69a666b61345"></span>

“策略版本”至少包含骨干、updater、decider、tokenizer/模板、mask 与位置约定、mixture_mode、threshold/max_iter、warmup/随机选项/实际 eval decider、温度/截断，以及数值精度与后端。仅权重 hash 相同不足以证明策略相同。特别是 `even_mix` 在不同后端的深度约定需逐项核验，不能因模式名字相同就省略执行语义。

<span id="source-block-6f99fd8a-b97a-4b43-8be5-26ee239fc55f"></span>

### 服务负载也是自适应策略的条件

<span id="source-block-7d169f57-809d-4fa9-99ae-495a07805d41"></span>

固定 MiniSGL 的 DUO 页预算保护按 continue_indices 的批次顺序保留前 keep_n 行，超出者使用当前轮已经采样的候选。因此在相同权重、同一问题和相同 gate 规则下，其他请求占用的页数与本请求在批内的位置也能改变它是否继续。这是明确的控制流依赖，不是已经测得的负载下性能结论。[页预算保护](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L600-L625)

<span id="source-block-394760f3-6d52-432b-badf-6dbe9a10c96d"></span>

令 s 为模型状态，b 为服务状态，u 为已声明的随机输入；实际路由应表示为二级决策：

<span id="source-block-a2cca4ed-0139-4332-909d-ec1295c48f2d"></span>

```math
\begin{aligned}
c^{\rm raw}&=\operatorname{Decide}_{\theta}(s,u),\\
c^{\rm exec}&=\operatorname{Admit}(c^{\rm raw},b),\\
(s',b')&=\operatorname{Step}_{\theta}(s,b,c^{\rm exec},u).
\end{aligned}
```

<span id="source-block-de12eb0e-f7b9-484a-8807-84ef008a22b6"></span>

只有在不触发资源覆盖且批次算子/RNG 影响得到控制时，才可以省略 b，声称在测量同一个仅依赖模型状态的策略。扩容带来的吞吐变化和强停带来的生成分布变化需要分别记录。将强停当作自主 gate 的停止，会同时污染路由校准、平均深度和成本—能力归因。

<span id="source-block-3911ce76-99d8-458c-9db4-4459480bca8c"></span>

| 记录对象 | 至少保留的字段 | 用来排除的混淆 |
| --- | --- | --- |
| 模型决策 | gate logit／概率、阈值或随机事件、深度上限、warmup／override | 未继续究竟来自模型还是配置 |
| 服务覆盖 | 批次顺序、请求 ID、可用循环页槽、raw 与 exec 掩码、覆盖原因 | 资源不足被误记为自主早退 |
| 读出与随机性 | remaining、停止分支、温度／截断、逻辑请求与随机抽样对应 | 同 seed 被误当成同一随机实验 |
| 后续状态 | 各深度 KV 条目、位置与有效性、请求结束和释放事件 | 同文字前缀被误当成同一模型状态 |

<span id="source-block-9194c56d-db72-4365-84f4-bac58af36283"></span>

后续运行按三组配对进行：先在资源充足且固定路由时验证后端数值；再放开模型 gate、维持无资源强停；最后固定请求集合改变并发和排序，单列资源覆盖率与逐题能力—成本。前两组提供模型与内核的对照，第三组测量服务策略本身。当前这里只把测量协议与字段落入工程文档，尚未执行 GPU 服务压力测试。[当前 token 的强停界及序列条件](/notes/loop-reasoning/generation/)

<span id="source-block-b760512e-8356-4ae9-a696-3cd594dcfa14"></span>

### 计时与吞吐口径

<span id="source-block-7e902ccf-8a06-4712-80d9-61f3fc7dd9a8"></span>

[torch.profiler 2.9](https://docs.pytorch.org/docs/2.9/profiler.html)用于定位 attention、MLP、词表读出、通信和 CPU 调度。

<span id="source-block-45d41c48-3f8c-40e1-8fa6-41cbe4d18071"></span>

测量固定硬件、dtype、内核、长度、并发和路由；

<span id="source-block-718adb51-aded-4558-bb10-e86f61f72c14"></span>

加载、首次编译与 warmup 单列。

<span id="source-block-14f27b13-64f2-4252-97f8-61d8414387d1"></span>

端到端时间采用 GPU 同步边界并取多卡最大耗时；CUDA events 用于另外标明的设备区间。下式 `D` 为 DP 副本数，全局监督数只跨 DP 汇总。

<span id="source-block-fb84b5a5-ef90-4f62-8238-7ac7201873c1"></span>

```math
\begin{aligned}
T_{\rm update}&=\max_r\Delta t_r,\\
N_{\rm shifted}^{\rm global}&=\sum_{r=1}^{D}N_{r,\rm shifted},\\
{\rm supervised\ tokens/s}&=\frac{N_{\rm shifted}^{\rm global}}{T_{\rm update}}.
\end{aligned}
```

<span id="source-block-a4c86a4d-7b7b-4229-832c-455f2027f619"></span>

### 算法 9 更新窗口的性能测量

<span id="source-block-213355cb-ee34-4ea8-a3a2-5730039cead4"></span>

**过程** MeasureWindow(protocol)

<span id="source-block-3cc88e3f-2466-47f0-986c-ec13711cd02c"></span>

**输入** 包含固定样本、版本、生效配置、预热规则及测量起止点的测量协议。

<span id="source-block-65ec2838-ea85-4a2c-9386-11e2db3740a5"></span>

**输出** 各窗口耗时、各卡显存峰值、监督计数、深度、KV 与阶段耗时的原始记录和统计。

<span id="source-block-741a7f1c-17f5-4dd4-9abf-c8589859fa92"></span>

<figure class="research-embed"><iframe src="/research-assets/ff73f4c3a15575fa7164127f2625d8f09a68f70e4074f4dd3ee80fa6b86296f3.html" title="算法 9 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 9 操作与数学公式 · <a href="/research-assets/ff73f4c3a15575fa7164127f2625d8f09a68f70e4074f4dd3ee80fa6b86296f3.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-25f7bfdb-3526-4add-90bf-99860780048f"></span>

源码现有 `peak_mem_gib` 为 rank 0 的 `max_memory_allocated/2³⁰`，训练循环内未逐更新 reset；它是自最后一次重置以来的峰值，不能直接当作本次更新峰值。上述算法是拟补测量协议。[日志位置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L808)

<span id="source-block-5d4c0f23-dc77-48d8-b2ee-02fc5d9515b6"></span>

| 测量对象 | 拟记录的量 |
| --- | --- |
| 一次训练更新 | 监督 token/s、更新时长、各阶段时间、峰值 allocated／reserved 显存、GPU·h |
| 单请求生成 | TTFT、decode ms/token、端到端时间及 p50／p95 |
| 服务吞吐 | 固定并发下 requests/s 与 output tokens/s |
| 循环与缓存 | prefill／decode 各深度计数、KV 条目、主路径／探测成本 |
| 版本切换 | actor 导出、传输、rollout 重载时间与概率差异 |

<span id="source-block-fc552c1e-d6b6-47ba-94d4-e90c16fe11c9"></span>

## 保存模型与恢复训练分开验收

<span id="source-block-d54bc379-3e52-49bd-a7bb-98085c60126e"></span>

[simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) 的 `save_model_tp` 导出 BF16 模型和循环组件；当 `save_only_model=false` 时，DCP 另存主权重和优化器状态。可推理的导出与可接续的训练状态分别检查；想复现后续轨迹，还要核对 scheduler、随机数、采样器、数据游标与配置是否完整恢复。

<span id="source-block-ef37e2bc-3a2c-4a84-999b-35537c8612ed"></span>

使用同一批次比较“连续执行”和“保存后重载继续”的下一次损失、参数变化与数据身份，声明比较精度和容差。没有这些记录时，只报告模型可加载或训练可继续，不声称逐步轨迹完全复现。

<span id="source-block-798b807e-0f47-4fcf-81b2-a0cddccac367"></span>

### 从“能加载”推进到“恢复同一个更新状态”

<span id="source-block-71d8a3f2-cf10-44fd-9f77-5751e32e74cd"></span>

把一次优化窗口写成状态转移，可准确说明 checkpoint 需要保存什么。S 不仅包含模型权重，还包含所有会改变下一次更新的内部状态；B 是下一窗口数据，ξ 是所需随机量。

<span id="source-block-05b3ba72-2d9d-480f-81fc-e82b551afdd7"></span>

```math
S_{k+1}=\Phi(S_k,B_k,\xi_k).
```

<span id="source-block-e601378f-11c3-4170-8859-5ba3931b57b2"></span>

若只保存 θ，Adam 动量、step、scheduler、数据次序或随机数状态不同，下一次更新就可能不同。对于在线后训练，rollout 的模型版本、在途任务与待消费轨迹还属于额外状态。应区分“恢复同一算法配置并继续训练”和“复现连续执行的下一步轨迹”。

<span id="source-block-39c503f8-9506-4a1b-9909-50d60cc12e5a"></span>

| 固定源码路径 | 已核对的实际状态 | 不能据此直接推出的结论 |
| --- | --- | --- |
| TaH2 `simple_loop.py` | 非 `save_only_model` 路径使用 DCP 保存 FP32 主权重及 optimizer；另存 global_step、epoch、windows_per_epoch、DP/TP 信息 | 所示 trainer payload 没有包含 RNG；只导出 BF16 模型也不等于恢复主权重和动量 |
| TaH2 resume / `_RankStridedBatches` | 依据 global step、epoch 和窗口位置跳过既有批次；新建 scheduler 后推进到对应 step | 相同窗口数不保证相同样本 ID；还要求采样次序、切分拓扑和 schedule 配置兼容 |
| verl FSDP checkpoint manager | 按 contents 配置保存/加载 model、optimizer、extra；extra 包含 scheduler 与 RNG | 配置未保存或未加载某类状态时，不存在自动补齐的保证 |
| verl V1 trainer | 保存 actor、可选 critic、StatefulDataLoader；异步模式还保存 TransferQueue | pending/running prompt 重发是新生成尝试，不等于从中间 token 及原 RNG 精确续跑 |
| TaH2 resume达到正max_steps | 若仍有epoch窗口，阈值在optimizer.step之后检查，可能再更新一次 | 不能把恢复进度≥阈值等同于源码已在首个更新前退出；预算入口guard见更新页 |

<span id="source-block-ddd876ff-d385-4153-a027-b5d790096501"></span>

对应调用链：[TaH2 保存与恢复](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L497)、[verl FSDP load/save](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/utils/checkpoint/fsdp_checkpoint_manager.py#L265)、[V1 数据与队列恢复](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py#L850)。

<span id="source-block-0b2e8eda-7a85-46f4-9cce-6148317f3598"></span>

V1 的 `_reissue_inflight_prompts` 会清除待重发 prompt 对应的旧 trajectory keys，并重新生成；已经完成的轨迹按已保存状态恢复。因此异步恢复应把“重发次数、生成版本、旧轨迹是否删除”纳入记录，不能把 replay buffer 恢复与模型内部 KV 恢复混为一谈。[重发逻辑](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py#L910)

<span id="source-block-a9a644a4-5c65-46a7-8fc7-2b18b2068aa8"></span>

### 算法 T7 两轮同步与恢复状态核验

<span id="source-block-564d9564-8023-49e1-a69b-6cfaf731d0a1"></span>

**过程** VerifyTwoUpdatesAndResume(learner, rollout_backend, checkpoint_config, fixed_protocol)

<span id="source-block-15980146-7057-427e-88f1-4d5772449e0b"></span>

**输入** 学习端完整状态、rollout 后端、实际 checkpoint contents 配置、固定的评分/采样/数据协议及数值容差。

<span id="source-block-4014fa11-e9ea-40a0-adfd-fc3bf4e78880"></span>

**输出** 两轮版本流、actor 与 rollout 概率差、连续与恢复分支的下一步损失/梯度/参数差，以及缺失状态清单。

<span id="source-block-f8e152a4-d592-408b-bb3d-d06356b349f4"></span>

**前置条件与终止** 在无梯度累积残留、无在途 optimizer 更新的边界保存；同步参考场景先清空在途请求。随机 gate 若开启，必须具有足够的动作和随机状态日志，否则只报告条件评分一致性。算法只跑两轮更新与一次恢复对照；遇到版本、支持或状态缺失时返回明确诊断，不无限重试。本页提供的是后续工程核验协议，本轮未执行模型训练。

<span id="source-block-3563a945-65f0-4115-ae41-495f76330a24"></span>

<figure class="research-embed"><iframe src="/research-assets/1afffa782922050bd7a44c879eafda53aa33ed2a46e5c3143dad05c95aa248eb.html" title="算法 T7 两轮同步与恢复状态核验" loading="lazy" sandbox=""></iframe><figcaption>算法 T7 两轮同步与恢复状态核验 · <a href="/research-assets/1afffa782922050bd7a44c879eafda53aa33ed2a46e5c3143dad05c95aa248eb.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-6d0e1804-36bb-46bf-9893-f59976af5238"></span>

**函数来源与实现状态** `trainer_sync` 在初始化及 step 结束调用权重同步；`engine_workers.update_weights` 负责学习端到生成端的参数传输和生成资源恢复；checkpoint 分别沿上述 manager 与 trainer 实现。TaH2 的 updater/decider 及执行配置是否被实际传输仍是待完成的适配项，不能由 verl 原生模型路径推断已经支持。[同步钩子](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_sync.py#L31)、[参数同步入口](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/engine_workers.py#L789)。

<span id="source-block-09cea550-8670-4727-ae2f-50bde1b3d9d7"></span>

## 能力评测与机制结论

<span id="source-block-65d8a639-c927-424e-bd81-85acdd52d4fe"></span>

[eval_online.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py) 的 `run_online_eval`、`_online_runtime_metrics` 用于定位生成、评分和吞吐记录。Bundled MiniSGL 的响应提供 `usage.iter_counts`，DUO 另有 `usage.prompt_iter_counts`；字段范围须对照服务与客户端一起核实。[引擎说明](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md)

<span id="source-block-ad536e7e-8792-4004-991f-93cd739cf46c"></span>

比较先采用同初始化、同源数据的单遍、固定循环与自适应循环；按训练预算与推理预算分别控制，保留原始逐题输出。准确率、文字长度、prefill／decode 计算和端到端时间一起报告。后续少文字方案再与合理调优的文字推理对照。

<span id="source-block-2071e76a-7a13-44c3-a4e3-d721a452a3ad"></span>

### Teacher forcing 验证：实际执行集合与损失分母

<span id="source-block-6bf93261-6e47-4018-beae-6d6ad5f32436"></span>

验证损失本来要回答一个简单问题：实际预测过的这些目标，平均损失是多少？分子需要加上它们的损失，分母也应数同一批目标。固定 `train_model_tp` 的两步处理却可能让范围错开：它先在整个 processed eval 数据上计算 `eval_num_items` 和 `eval_gbs`，再把 DataLoader 批次数向下截到实际 DP 数的整数倍。

<span id="source-block-8159feea-0f9d-46f9-9f88-dea5ab9beb56"></span>

沿用下表的控制例：952 条处理后样本，每条恰有一个移位后的有效目标，且每个目标损失固定为 2。batch size 为 2、DP 为 8 时，先形成 476 个批次，实际只执行其中 472 个，也就是 944 个目标。执行损失和为 1888；若仍除以预先统计的 952，日志就是约 1.983，尽管每个实际目标的损失都是 2。这里变化的是汇总方式，预测损失本身已被固定。

<span id="source-block-003f0da3-a428-41d0-a512-e5cb21746737"></span>

源码按 batch 索引模 DP 数分配给各 rank，最后对各 rank 的 loss 做 DP SUM。下面的公式把“预统计的目标”和“实际执行的目标”分别写出，就能看到这个差值从哪里来。分析采用默认 recipe 的 `NextTokenPredLoss`；任意 CombinedLoss 的组成与归一化需要分别检查。[eval 初始化与执行](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L620-L705) · [1.7B 配方](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_1.7/sft_tah.yaml)

<span id="source-block-b644630e-4a27-4963-885f-e3fd1c2c9720"></span>

设 D 是进入 DataLoader 的有序样本出现记录，N=|D|；b 是 eval batch size，P 是实际 DP 数。B* 之后的 batch 不进入模型，不能把其目标 token 仍留在“已执行目标的平均损失”分母中。

<span id="source-block-c4b4c005-cdb5-4c1c-8a50-4b608028db07"></span>

```math
B=\left\lceil\frac Nb\right\rceil,\qquad
B_*=P\left\lfloor\frac BP\right\rfloor,\qquad
\mathcal E=\{0,\ldots,B_*-1\}.
```

<span id="source-block-59dc28fb-80ae-4e1c-8ba7-65f176bdc99e"></span>

先区分预统计的分母与实际执行的目标。源码 helper 从 D 的未移位 `labels` 中统计全部非 −100 位置，记为 U_D。对实际执行的 batch j，按 collator 输出做因果移位，得到有效目标集 I_j；记 n_j=|I_j|，s_j 为同一损失分支在这些位置上的损失和。这里数的是 token，有重复样本时按每次出现计数，不能用行数替代。下式采用实数算术，L_raw 表示格式化前的 `loss_sum`；源码 FP32 累加误差和日志格式化另外记录。[helper 的真实分母](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_runtime.py#L425-L444) · [因果移位](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L380-L388) · [token 损失](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

<span id="source-block-e0c395fe-ca36-4a7f-9dfd-9b5ec74c5584"></span>

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

<span id="source-block-7605e04b-835e-4945-930f-ce1ff2e89cab"></span>

在 U_D>0、V_E>0 时，下式是实数算术下的恒等式。U_D=V_E 足以保证 L_raw 等于执行子集的 token 均值；当 S_E≠0 时，这也是必要条件，零损失本身不能证明分母正确。对同一批 collator 输出，各行首位置均为 −100 可保证未移位与移位计数一致；仍需另外核对执行覆盖范围。这里没有再除 DP 数：最终归约是 SUM。

<span id="source-block-0c91e7a8-7264-4067-9278-fb784956e4bb"></span>

```math
L_{\mathcal E}=\frac{S_{\mathcal E}}{V_{\mathcal E}},
\qquad
L_{\mathrm{raw}}
=\frac{V_{\mathcal E}}{U_D}\,L_{\mathcal E}.
```

<span id="source-block-f3b41054-806d-4707-a985-26b55bfaecc0"></span>

进一步固定逐目标预测与损失，并假设 D 的未移位计数等于其全部移位监督数 V_D。把未执行部分的损失和记作 S_drop，则可以比较“完整 D 的目标”与源码日志：

<span id="source-block-0e28e2ab-1171-4124-b691-3f43b2371ce9"></span>

```math
L_D=\frac{S_{\mathcal E}+S_{\mathrm{drop}}}{U_D},
\qquad
L_D-L_{\mathrm{raw}}=\frac{S_{\mathrm{drop}}}{U_D}.
```

<span id="source-block-b5387c04-c672-462d-aa80-2fa629bba54e"></span>

归一化预测分布的非负 NLL 情形下，遗漏目标会使这一日志偏低。恒等式本身不要求损失非负；本课程已经单独说明 stop_prob_mix 的数值下限和三个损失消费分支，因此不能对所有实现值无条件宣称严格向下偏差。只修分母为 V_E 得到的是执行子集均值，仍不能恢复未执行部分的预测或完整 D 的均值。

<span id="source-block-99e6b1a3-e7ff-4c0c-af13-80a6b588f52d"></span>

#### 手算反例与默认配置的边界

<span id="source-block-880f5c0f-5e3c-42b9-8ae8-d0abecccb3f5"></span>

下表固定 952 条处理后样本，每条在因果移位后仍恰有一个有效目标，冻结所有目标损失为 2。它只检验调度与分母，不代表任何模型性能实测。

<span id="source-block-2fc9d5dc-4c41-46f3-aef4-186d08263231"></span>

| 实际 DP P | eval batch b | DataLoader 批数 B | 执行批数 B* | 执行样本数 | 理想实数复算 L_raw | 执行子集均值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 952 | 952 | 952 | 2 | 2 |
| 8 | 2 | 476 | 472 | 944 | 1.9831932773 | 2 |
| 4 | 2 | 476 | 476 | 952 | 2 | 2 |

<span id="source-block-6bedd4fa-2dfe-4866-843e-a318eb631f15"></span>

表中的关键是：952 个样本能被 8 整除，但 batch size 为 2 时形成的 476 个批次不能。官方 1.7B recipe 设置 `per_device_eval_batch_size=1`、`data.dp=8`；实际 P=8、采用正常 drop 处理且数据非空时，不触发这里的尾部批次问题，改变 b 或实际拓扑后则需重查。再看更小的例子：N=8、b=2、P=8 时，B*=0，没有任何批次进入模型，初始 `loss_sum` 归约后仍为 0。这个 0 不表示零验证误差，因为执行子集均值未定义。另一种情况是整份数据 U_D=0：helper 返回 `None`，初始化在 `float(None)` 处先报错，应与“分母为正但没有执行批次”区分。

<span id="source-block-dae5c084-6c08-4ede-afa6-38b695507e89"></span>

本次重新提取固定源码的两个计数 helper、整除函数、label 移位、NextTokenPredLoss、eval 初始化、`_run_eval` 及日志格式化函数，八组检查全部通过；附加 cosine helper 检查也通过。NumPy 张量、右侧 padding、固定逐目标 NLL 和串行 DP SUM 是明确替身；真实 tokenizer／collator、模型预测、路由、GPU、FSDP／TP 和 distributed collective 均未执行。12 份源码快照逐一核对 Git blob SHA 与 SHA-256，检查包包含源文件、可运行脚本和 JSON 结果。

<span id="source-block-576122fc-02b0-4908-93f3-0fcbfb6d608d"></span>

源码在 SUM 后以 `.4g` 格式化 loss：上表 1.9831932773 是理想实数复算，实际显示为 1.983；FP32 原始值还可能有累加舍入差。应同时保留分子、分母、未格式化值和显示值，不能把显示精度解释为估计误差。

<span id="source-block-a6d17b59-6e65-4f4a-bf10-c6d23a8cd231"></span>

#### 八组检查与可运行记录

<span id="source-block-398276ef-9b62-4183-b4d8-8baf2362e2bd"></span>

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

<span id="source-block-69154e18-ab29-4326-a309-e3b3abfac487"></span>

[循环研究验证与源码核对：八组检查及固定源码（原页附件）](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c#69154e18-ab29-4326-a309-e3b3abfac487)

<span id="source-block-baa30c87-7962-4ed4-a518-38279c20e223"></span>

运行检查包中的 `python3 check_eval.py`；仅需 Python 与 NumPy，无需模型权重。上述数据均为控制流小例，未把发布集的原始 955 行视为已实测的预处理结果。

<span id="source-block-b445b124-6da5-430c-896a-5b1c0faf4646"></span>

#### 算法 T12 验证损失的执行记录核对

<span id="source-block-c3c39237-e7db-4b4d-8986-a1427d7ae539"></span>

**输入** 固定数据出现记录 D、tokenizer／模板／collator／mask、b、实际 DP/TP 拓扑、checkpoint、路由与数值配置，以及固定源码的 NextTokenPredLoss 路径。

<span id="source-block-0e9ee7f6-71a4-4459-9ac3-9519aeb3d5c2"></span>

**输出** 处理后与执行后的出现记录、每批目标数、损失分子、U_D、V_E、源码日志复算值、执行子集均值，以及缺失／重复记录。

<span id="source-block-d3fe4233-ae0a-4c82-aef1-1caf275ea55d"></span>

<figure class="research-embed"><iframe src="/research-assets/b895661ebf82d49ad5943fdddb7b2afbe417029b380065cff1f86df76cf6d714.html" title="算法 T12 验证损失的执行记录核对" loading="lazy" sandbox=""></iframe><figcaption>算法 T12 验证损失的执行记录核对 · <a href="/research-assets/b895661ebf82d49ad5943fdddb7b2afbe417029b380065cff1f86df76cf6d714.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-d2dd2ec2-ffe9-4eae-a0be-b23aa9263206"></span>

**性质** 这是针对发布实现的参考核对算法。无有效目标返回“不可评估”是参考算法的防护，不是源码已经实现的修复；保持原有各 rank 的调用次数和 collective 顺序。若另行实现完整验证集评估，需同时设计等步数通信与零权重占位，不能只删除 islice 截断。

<span id="source-block-25e63294-db11-426e-8969-82dd031fe5a9"></span>

### 多次采样的 avg 与 pass@k

<span id="source-block-d28e9b45-a501-4aea-91f9-f3899546bedc"></span>

多次采样按题目组织统计。设第 `i` 题有 `n_i` 次独立同分布采样，其中 `c_i` 次正确，`n_i≥k`；avg 是样本正确率，pass@k 的常用无偏估计如下。公式可对照 [HumanEval 原论文第 2.1 节与附录 A](https://arxiv.org/html/2107.03374v2)。

<span id="source-block-6fed4c79-9da2-413a-863c-d179c73fa93b"></span>

```math
\begin{aligned}
\widehat{\mathrm{avg}}_i&=\frac{c_i}{n_i},\\
\widehat{\mathrm{pass@}k}_i
&=1-\frac{\binom{n_i-c_i}{k}}{\binom{n_i}{k}},\\
\binom{a}{k}&=0\quad\text{若 }a<k.
\end{aligned}
```

<span id="source-block-d130ab6a-9266-402c-909b-ec956d013e46"></span>

后者估计同一题 `k` 次尝试至少一次成功的概率；独立且单次成功率为 `p_i` 时为 `1−(1−p_i)^k`。先逐题计算再平均，不把全数据的平均正确率直接代入非线性公式。同一题多采样与多种子保留聚类结构；置信分析声明是否还包含训练种子变化。调优与最终确认集继续分开。

<span id="source-block-1249e184-f177-48b4-b182-6ef47f0cac0c"></span>

**机制证据的解释范围**

<span id="source-block-c1ce2564-2ab6-4d25-9c35-7428b238ebf5"></span>

机制分析需要在相近成本下，结合状态替换、关闭额外轮次、扰动历史 KV 或改变后续操作的干预，判断计算依赖。准确率提高、范数变稳或 probe 更易解码，单独一项都不足以确定内部保存了何种可复用知识。

<span id="source-block-34cbfa9a-0028-47b4-a545-858bd1ab6721"></span>

当前已有固定源码和测量协议，数值一致性、两卡运行和性能结果均待实现验收。

<span id="source-block-f82ee061-b2c3-4b01-a5e1-64f1cd13ba7b"></span>

## 交付与验收

<span id="source-block-9dc6cf8b-89bd-49f2-a7f3-121234c17b1d"></span>

- [ ] `parity_report.json`：同前缀、同权重、同深度下的概率与梯度差异，随后比较自主 gate。

<span id="source-block-c3e02057-6632-4e74-85e9-25183f190a31"></span>

- [ ] `checkpoint_report.json`：模型导出与训练恢复分别给出实际结果。

<span id="source-block-aa649136-8ff7-4b5b-8e5a-3e91a1ef6e43"></span>

- [ ] `cost_profile.json`：上述阶段、显存和深度统计有明确测量边界。

<span id="source-block-ac820cad-2905-42f0-83c5-6857e116960e"></span>

- [ ] `eval_records.jsonl`：逐题原始输出、评分状态、预算、版本和题目身份完整。

<span id="source-block-f583b09a-93e0-4bab-921b-c741e4d4312f"></span>

- [ ] `comparison_report.md`：能力与成本曲线及机制证据，区分实测、估算和未验证解释。

<span id="source-block-c4268324-0470-4856-95eb-38df90ec6f13"></span>

## 围绕本课讨论

<span id="source-block-b1db85f0-fe26-43d0-bf28-9fbd0121b418"></span>

```text
请从两张 A100 的资源约束出发，对照 TaH2 的 TP、FSDP2、全局 token 分母、自适应控制流和缓存重计算。随后为同一小型案例设计 parity、checkpoint、cost 和逐题评测记录；先固定深度定位数值差异，再放开 gate。所有显存与速度结论以实测为准。
```
