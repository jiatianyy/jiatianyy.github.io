---
title: "TaH2 论文结果与实现"
description: "TaH2 在普通 Qwen3 Base 上联合训练骨干、updater 与继续决策器，用当前模型的下一轮预测收益监督 token 级循环。阅读顺序是先确认比较对象和计算图，再分析性能、消融与成本，最后沿函数索引进入工程实现。这是一条非原生循环改造路线。[原论文](https://arxiv.org/html/2609.35748v1)为 Improving "
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb"
sourcePageId: "page_51505cd26b54819180fb3ee0632da7bb"
sourceSha256: 5636d9ecb5c8a8270cfbcadb2ca1f56fb9fa9b074fcb678d4b23955ad802a93c
series: loop-reasoning
references: []
---

<span id="source-block-d228ca94-b938-4029-ac6a-25aa95cf06c4"></span>

TaH2 在普通 Qwen3 Base 上联合训练骨干、updater 与继续决策器，用当前模型的下一轮预测收益监督 token 级循环。阅读顺序是先确认比较对象和计算图，再分析性能、消融与成本，最后沿函数索引进入工程实现。这是一条非原生循环改造路线。[原论文](https://arxiv.org/html/2609.35748v1)为 Improving Test-Time Scaling with Adaptive Looped Transformers，2026 年 9 月 28 日 v1，共 32 页；代码固定到 [TaH 78ddf9a](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c)。

<span id="source-block-1d853441-0bfd-40c0-8153-d259516d2c4f"></span>

[TaH2 与后训练工程](/notes/loop-reasoning/engineering/) · [状态与循环算法](/notes/loop-reasoning/internal-loop/) · [训练到自主生成](/notes/loop-reasoning/generation/) · [基座矩阵](/notes/loop-reasoning/model-matrix/)

<span id="source-block-aecb7f27-c8fe-465d-9cda-35aba5be3edd"></span>

## 模型结构与比较对象

<span id="source-block-d34b9a5b-48ab-4e78-934c-b7a29a26fc87"></span>

TaH2 是 Dense 普通基座的全栈循环后训练模型。论文中的 Ouro 和 Huginn 也从同一个 Qwen3 Base checkpoint 改造，并在同一模型规模内使用相同数据和配方重新训练。因此，这两个名称在下表表示架构对照；原生 Ouro、Huginn 的发布权重及其他六篇原生模型，仍按各自 checkpoint 和训练阶段登记。 [§3、§5.1、A.2.1](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS1)

<span id="source-block-329c355b-f10e-4060-aef0-30e31fa28ddd"></span>

| 本文配置 | 前向与目标 | 1.7B 新增参数 |
| --- | --- | --- |
| Standard | 单次完整骨干；普通 next token CE | 0 |
| TaH2 | 全栈自适应循环；DUO；学习 updater；继续 gate；停止权概率混合；在线 lookahead 深度监督 | 46.2M，约 2.6% |
| TaH2 fixed | 全栈固定循环；causal attention；保留 updater；无 gate；对各轮概率作均匀混合再计算 NLL | 21.0M，约 1.2% |
| Ouro fixed | 全栈固定循环；直接回传 hidden；Stage I 移除 gate，对每轮 CE 求平均 | 0 |
| Ouro adaptive | 在 M2 的 Stage I 权重上冻结骨干，再训练线性 exit gate；prefill 全轮，decode 可早退并以末次 KV 补深层 | 约 2.0K，低于 0.001% |
| Huginn | 28 层的 7 层 prelude、14 层共享中段、7 层 coda；带门控输入适配器；coda 后一次 CE | 4.2M，约 0.24% |

<span id="source-block-0282a26b-4041-4bff-bbd1-426caff1c62f"></span>

TaH2 fixed 同时改变 attention 可见性、决策方式和读出权重。因此它与 TaH2 的差额是整套配置的比较，不能全部归因于“有无 gate”。Ouro 在本文的 Stage I 删除 gate 是为处理该后训练设置下的 gate collapse；这项处理也必须与原生 Ouro 的两阶段预训练区分。Stage II 使用 k=50、γ=0.005 的损失改善软标签，学习率 10⁻⁴、一个 epoch，Q-exit 阈值 0.5。[A.2.1、Table 6](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS1)

<span id="source-block-4e97517d-4993-41e5-ae33-46ee45658854"></span>

### 同名基线的具体目标

<span id="source-block-7db1f3fb-132d-43c6-87aa-1ccc34ae7247"></span>

本文 Ouro Stage I 对每轮 CE 取平均；TaH2 fixed 先平均概率再取 CE。由 −log 的凸性，后者至多为前者，但两者的梯度分配不同，不能把 loss 数值直接当作同一个目标的优劣。以下等式针对同一个监督 token、同一组每轮分布：[A.2.1](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS1)

<span id="source-block-744fea7b-11e8-4a55-a725-a441992c45d2"></span>

```math
\begin{aligned}
\ell_{\rm fixed}
&=-\log\!\left(\frac1M\sum_{m=1}^M q^{(m)}(y)\right),\\
\ell_{\rm Ouro\,I}
&=\frac1M\sum_{m=1}^M[-\log q^{(m)}(y)],\\
\ell_{\rm fixed}&\leq\ell_{\rm Ouro\,I}.
\end{aligned}
```

<span id="source-block-efa34572-495c-4690-a152-d5148ab6290c"></span>

Ouro Stage II 的 λ 表示退出概率，TaH2 的 g 表示继续分数。本文在 M2 Stage I 权重上冻结 backbone，使用下一轮的截断 NLL 改善训练线性退出 gate；BCE 的预测参数是 1−λ：[Eq.11](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS1)

<span id="source-block-b5bdcfd7-0d30-4dbb-a230-b883d93a948e"></span>

```math
\begin{aligned}
\lambda_t^{(m)}&=\sigma(w_{\rm exit}^{\top}\operatorname{sg}[h_t^{(m)}]),\\
I_t^{(m)}&=\max(0,\operatorname{sg}[\ell_t^{(m)}-\ell_t^{(m+1)}]),\\
\widetilde c_t^{(m)}&=\sigma(50[I_t^{(m)}-0.005]),\\
\mathcal L_{\rm gate}
&=\frac1M\sum_{m=1}^{M-1}\frac1N\sum_t
\operatorname{BCE}(1-\lambda_t^{(m)},\widetilde c_t^{(m)}).
\end{aligned}
```

<span id="source-block-8bba2627-d02f-4b94-a5c1-88ed6dc54d8c"></span>

Stage II 学习率为 10⁻⁴、训练一个 epoch；推理用累计退出概率达到 0.5 的 Q-exit 规则。Prefill 计算全部轮次，decode 提前退出时将最后实际 KV 复用于更深轮次。这与直接将当前继续分数和 0.5 比较具有不同语义，必须保留在基线身份中。

<span id="source-block-5a573413-b864-49b6-8c1d-59cd2aa378f1"></span>

本文 Huginn 对照只重复第 8–21 层，每轮通过带门控的 input adapter 注入第 1–7 层一次计算得到的 z。其逐维保留系数严格在 0 与 1 之间，但该事实单独不能证明整个带 attention 的 recurrence 收缩，因为中段变换的 Jacobian 仍参与复合：[Eq.12](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS1)

<span id="source-block-394ae4bf-72ad-4863-821a-b227895fd010"></span>

```math
\begin{aligned}
u_t^{(m+1)}&=a\odot r_t^{(m)}+v\odot W_{\rm in}z_t,\\
v&=\operatorname{softplus}(b_{\rm in}),\\
a&=\exp[-v\odot\exp(b_{\rm ret})],\\
0&<a_j<1.
\end{aligned}
```

<span id="source-block-a4ee0ea8-6b75-431b-b9e2-9b98d1588d58"></span>

TaH2 每一轮都复用同一套骨干、updater 和 decider 参数。首轮直接读取原 embedding；后续轮由 updater 融合原 embedding 与末层 hidden。Decider 读取 embedding、hidden 和排序后的 top-k 概率值，1.7B 的 k=2048，并决定哪些 token 进入下一轮主计算。无梯度 lookahead 生成训练监督，推理时省略。DUO 可见性、停止概率混合与完整梯度在[一次内部循环](/notes/loop-reasoning/internal-loop/)中逐步展开。[§4、Eqs.1–10、Fig.2 与 Fig.7](https://arxiv.org/html/2609.35748v1#S4)

<span id="source-block-0ae10482-ec0d-4bc3-b404-0c4806a9c5fa"></span>

### 参数计数与有效深度

<span id="source-block-88d3d7bd-d6d2-40bd-879a-fd5a60aae7ee"></span>

| 模型尺度 | 骨干参数 M | Updater M | Decider M | 新增总量 M | 占完整模型 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.7B | 1720.57 | 20.98 | 25.18 | 46.16 | 2.61% |
| 4B | 4022.47 | 32.78 | 39.33 | 72.11 | 1.76% |
| 8B | 8190.74 | 83.90 | 100.68 | 184.59 | 2.20% |

<span id="source-block-3c4f4160-77b0-4919-b2d0-f68e408743dd"></span>

Table 7 的百分比以包含新增模块的完整模型为分母。对原文无偏置 projection 与各自独立的 RMSNorm scale，可直接推得 updater 和 decider 参数量；d 为骨干宽度，dᵤ、d_d 为两个 SwiGLU 中间宽度，k 为概率输入宽度：[Eqs.8–10、Table 7](https://arxiv.org/html/2609.35748v1#A1.SS2.SSS2)

<span id="source-block-f714f7ae-1ea6-4ae3-a0b2-5d14567cd0e7"></span>

```math
\begin{aligned}
P_{\rm upd}&=2d^2+3dd_u+4d,\\
P_{\rm dec}&=d(2d+k)+3dd_d+d+(3d+k).
\end{aligned}
```

<span id="source-block-1bc238f0-61e3-4125-bb74-ccde62a2f801"></span>

取 d=dᵤ=d_d=k=2048，上式得到 updater 20,979,712 个参数、decider 25,176,064 个参数。一般 k 公式描述论文结构；固定 Qwen3MLPIterDecider 的 norm_l 和投影 shape 还要求 k=d，三个原表尺度均满足这一条件。Huginn 对照的有效层数为 14+14M：中段循环 M3、M7 对应 56、112 次层调用，与全栈 M2、M4 对齐。参数量、层调用数和实际 FLOPs 仍分别计量。[固定实现的 shape 与梯度](/notes/loop-reasoning/internal-loop/)

<span id="source-block-bc825731-d7fb-4dc0-9390-abebdc0b7a29"></span>

## 数据配方与评测协议

<span id="source-block-b206e855-c502-4885-8040-66cb74936a2e"></span>

本文的在线监督指：在 teacher forced 的数据前缀上，用当前模型自己的已执行深度和下一轮探测结果生成继续标签。token 前缀仍来自训练数据；这不是在学生自主生成前缀上进行的在线蒸馏或 RL。原文实验限于 SFT，RL 和其他后训练目标属于后续研究方向。[§4.2、§6](https://arxiv.org/html/2609.35748v1#S4)

<span id="source-block-b4c8205b-a107-4965-bed7-9b4214d71153"></span>

论文 1.7B 主实验使用约 273K prompts，Qwen3 8B 重新生成回答，训练三个 epochs、约 3.4B tokens。4B 与 8B 使用原始 Qwen3 235B A22B 回答与工具数据，分别约 8B、16B tokens，TaH2 的 ceiling 均为 M2。不同规模还更换了任务集合，因此跨尺度平均分不能单独估计“扩大参数规模”的效果。[§5.1、A.1](https://arxiv.org/html/2609.35748v1#A1.SS1)

<span id="source-block-0c8cc4ff-888e-4f26-80e0-2e7e747f2475"></span>

| 训练量 | 原文设置 |
| --- | --- |
| 数据来源 | AM Qwen3 Distilled 的 math、code、QA；Nemotron Agentic v1 工具轨迹 |
| 全局 batch | 128 条序列 |
| 最大序列长度 | 16,384 |
| 学习率与调度 | 4×10⁻⁵；warmup 0.03；cosine；最小 LR 比 0.1 |
| 训练轮次与精度 | 三个 epochs；bf16；最大梯度范数 1.0 |
| 训练 token 预算 | TPP=2，即每个模型参数对应两个训练 tokens；§5.3 用于跨规模预算 |
| 深度目标 | 正收益覆盖率 ρ=0.99；决策系数 α_D=0.05 |
| 退出阈值 | τ_exit=0.5 |
| 验证 | 1,000 条固定留出样本；teacher forcing |
| 生成默认值 | zero shot CoT；temperature 0.6；top-p 0.95；top-k 20；最大 32K 输出 |

<span id="source-block-bc297812-789a-4f13-a690-fd56beb23309"></span>

TPP 表示训练 token 数与所采用模型参数量口径之比；论文 §5.1／A.1.2 用它确定 4B、8B 的约 8B、16B token 预算。它不指定 DP／TP；实际并行拓扑由启动参数和 world size 决定，发布 recipe 中的 `data.dp` 又是数据整除配置。原文的约数、发布数据的实际 token 量与运行计数分别保留。[原文定义与 Table 5](https://arxiv.org/html/2609.35748v1#S5.SS1)

<span id="source-block-6b77ad45-c467-4895-bf72-9b37b3a8fd66"></span>

```math
\operatorname{TPP}=\frac{D_{\mathrm{train}}}{N_{\mathrm{param}}}=2.
```

<span id="source-block-53526527-abd4-40a3-82ee-f54e45895715"></span>

### 论文统计与公开数据版本

<span id="source-block-8ae31a6d-4a81-47bd-a610-f9ecbd61a83b"></span>

论文与发布数据使用各自的统计口径。仓库 README 记载的 1.7B 验证集为 955 条，论文记载 1,000 条。将 README 的单次遍历 token 数乘三，分别得到约 3.298B、7.761B、15.578B；论文对应约数为 3.4B、8B、16B。它们应并列记录，不能直接视为同一次历史运行。 [README 数据表](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/README.md)

<span id="source-block-7168d3c0-ef20-44c0-99a5-f1f44f55a402"></span>

| Release 学生规模 | Train 样本 | 单次遍历 tokens | Eval 样本 | 三次遍历 tokens 的算术值 |
| --- | ---: | ---: | ---: | ---: |
| 1.7B | 273,195 | 1,099,413,406 | 955 | 3,298,240,218 |
| 4B | 638,817 | 2,586,857,725 | 1,000 | 7,760,573,175 |
| 8B | 1,282,124 | 5,192,783,145 | 1,000 | 15,578,349,435 |

<span id="source-block-46de58c5-f2f5-46f6-84e8-40712aff58c1"></span>

发布数据入口为 [TaH2 amteam tool](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool)。README 的 1.7B 工具数据路径使用 Qwen3 8B 生成 assistant 回答，工具结果来自 reference replay 或 Qwen3 32B simulation。4B 使用按来源分层抽取的子集，8B 使用完整池，两者共享 eval 划分。实际复现还要固定数据 revision、过滤结果、输入／监督 token 分母与最终训练步数。[数据构建课](/notes/loop-reasoning/data/)

<span id="source-block-28519863-4dce-402c-972e-529e8f414224"></span>

**验证曲线与发布实现分开归因。** 原文 C.1 的 teacher-forcing 曲线继续保留；本次固定源码审计发现，处理后 eval 的监督分母在 batch 尾部截断前计算，改变 batch size 或实际 DP 后可能与执行集合不符。该代码行为不证明历史论文曲线已受影响，也不改变原表成绩；须用实际运行的样本出现记录、分子、移位目标数与配置核验。[完整推导、反例及算法 T12](/notes/loop-reasoning/systems-evaluation/#source-block-b445b124-6da5-430c-896a-5b1c0faf4646)。

<span id="source-block-a6b865a6-66ce-4766-b8d8-3bf2b86820b3"></span>

### 各 benchmark 的统计单位

<span id="source-block-12a98e91-36fe-4919-ac57-348cd6c977ba"></span>

| Benchmark | 题数 | 指标 | 子集 |
| --- | ---: | --- | --- |
| AIME24／25／26 | 各 30 | avg@32 | 各年度 |
| AMC23 | 40 | avg@32 | — |
| MATH500 | 500 | avg@4 | — |
| OlympiadBench | 675 | avg@4 | — |
| IMO AnswerBench | 400 | avg@4 | — |
| GPQA | 198 | avg@8 | Diamond |
| SuperGPQA | 7,050 | avg@4 | Hard |
| HumanEval | 164 | avg@8 | — |
| MBPP | 378 | avg@8 | — |
| LiveCodeBench v6 | 175 | avg@4 | — |
| BFCL v3 | 800 | greedy avg@1 | Multi-turn |

<span id="source-block-8dedcbba-eed0-46e2-b6c5-ba05bca129ca"></span>

以上为 [Table 8](https://arxiv.org/html/2609.35748v1#A1.T8)。设第 i 题有 K 条采样回答，正确指示量为 Yᵢⱼ；单 benchmark 的 avg@K 是每次采样正确率的题目平均。Table 1 与 Table 3 再对各自任务等权平均，分别有十项与九项；它们不按题数加权。

<span id="source-block-c1fbd23e-c5b7-4eed-b57d-822f8145dfee"></span>

```math
\widehat A_{\mathrm{avg@}K}
=\frac1Q\sum_{i=1}^{Q}\frac1K\sum_{j=1}^{K}Y_{ij}.
```

<span id="source-block-921fd3be-a548-4527-b9c3-7face169d51f"></span>

失败请求、截断且未完成的回答和答案提取失败需要保存各自状态，统计分母应与预先规定的尝试集合对应。对同一题的重复样本、同一长回答的不同 cutoff，相关性必须保留；若估计不确定性，应按题目联合重采样全部回答和 cutoff，跨模型比较也保持题目配对。此条件下的区间描述评测样本不确定性，不能替代训练 seed 的方差。

<span id="source-block-4a9fc6e5-956d-48ae-8f07-fd2c85d34c6b"></span>

### 原文采样设置与停止误差界的衔接

<span id="source-block-6c29dfbe-8a20-4b66-959e-c23f939b08af"></span>

原文 §5.1 及教师比较采用 temperature=0.6、top-p=0.95、top-k=20；因此温度 1、未截断的停止混合性质不能直接当作这组评测的行为保证。已核固定 MiniSGL 依赖为 FlashInfer 0.6.4，组合采样器默认先 top-k 归一化、再 top-p。[原文设置](https://arxiv.org/html/2609.35748v1#S5.SS1) · [依赖](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/pyproject.toml) · [固定采样器](https://github.com/flashinfer-ai/flashinfer/blob/f1e6fdcb8f65104047697f022b5d055ef022d763/flashinfer/sampling.py)

<span id="source-block-89f7baee-1ec1-495d-af9a-80192d7a063d"></span>

[生成课](/notes/loop-reasoning/generation/)新增共同支持上的 log 差振幅界、支持改变质量项、top-p 不连续反例与 21 行 T11；[RL 工程](/notes/loop-reasoning/reinforcement-learning/)补充 processed 概率梯度和行为支持外的目标质量。这些为后续复现与后训练设计提供明确的条件分析，不改变论文原始分数，也不宣称已测出某一 TaH2 checkpoint 的采样偏差或任务损失。

<span id="source-block-950007c1-0e11-43d6-b4f1-f492a6831d09"></span>

## 测试时计算斜率与同预算比较

<span id="source-block-145daadb-ec87-4432-8169-ff62d2a1bc14"></span>

Fig.1a 的 cutoff 为 4K 至 16K、每 2K 一个点；Fig.4b 与 Fig.9 的网格为 4K、6K、8K、12K、16K、20K、24K、28K、32K。曲线通过截断已生成回答后重新评分构造。相邻 cutoff 共用轨迹，横轴是响应平均 decode FLOPs，纵轴是准确率；这里的 32 个回答用于 avg，并不是 pass@32。[§3.2、A.3、C.2](https://arxiv.org/html/2609.35748v1#S3.SS2)

<span id="source-block-29c4e733-87b6-410b-bc03-ca71830a1c43"></span>

令 x_b 为 cutoff b 的 log₂ 平均 decode FLOPs，a_b 为以百分点表示的准确率。下面是论文线性拟合的显式最小二乘形式：

<span id="source-block-011ab3ff-2a39-4968-a079-6b583e83288c"></span>

```math
\begin{aligned}
x_b&=\log_2 C_b,\\
\widehat\beta&=
\frac{\sum_b(x_b-\bar x)(a_b-\bar a)}
{\sum_b(x_b-\bar x)^2},\\
\widehat\alpha&=\bar a-\widehat\beta\bar x.
\end{aligned}
```

<span id="source-block-8877dbfd-d396-4e2e-9532-90baf12b5779"></span>

| 1.7B 曲线 | 每翻倍 compute 的准确率百分点 |
| --- | ---: |
| Standard | 1.79 |
| Ouro fixed M2 | 2.12 |
| Ouro adaptive M2 | 2.27 |
| Huginn M3 | 2.26 |
| TaH2 M2 | 2.74 |

<span id="source-block-0540612b-05af-4b9b-bd68-c62927d76502"></span>

2.74/1.79−1≈53.1%，是斜率的相对增幅。原文同时指出，前三种循环对照在重叠 compute 区间仍低于 Standard。Fig.4b 延伸至 32K 时，Standard 在约 191.7 TFLOPs/response 达到约 12.0%，TaH2 在同一 compute 约 15.4%，相差约 3.4 个百分点。A.3 报告的拟合 R² 为 Ouro 0.997、Huginn 0.994、Standard 0.955，未给 TaH2 对应数值。[Fig.1、Fig.4、§5.2](https://arxiv.org/html/2609.35748v1#S5.SS2)

<span id="source-block-db7ba5f5-c963-4d00-a386-247c1905cff4"></span>

斜率更大并不自动给出同预算优势。将两条拟合线在参照预算 C₀ 处写成截距与斜率，差值及交点满足：

<span id="source-block-6fa8595e-1ed8-4edc-af02-a0c7773fd5b0"></span>

```math
\begin{aligned}
A_a(C)-A_b(C)
&=(\alpha_a-\alpha_b)
+(\beta_a-\beta_b)\log_2(C/C_0),\\
C_\times&=C_0\,2^{(\alpha_b-\alpha_a)/(\beta_a-\beta_b)}
\quad(\beta_a>\beta_b).
\end{aligned}
```

<span id="source-block-447838ac-b006-48c9-8165-cd675e15db8b"></span>

因此应同时报告曲线位置、斜率、重叠预算区间与平台期。交点公式只适用于两条局部拟合都有效的区间；它不能把准确率线性外推到未测预算。Fig.1b 的 AIME 增益 +2.8→+3.9 点使用 16K cutoff；下方 Table 1 使用 32K，两个数值口径分别保留。

<span id="source-block-36035f39-8b4b-4dec-8201-9c05ed4b1296"></span>

## 原始性能与设计消融

<span id="source-block-728198a0-442b-4d8a-a532-f06d6c432195"></span>

### 固定循环为何可能浪费计算

<span id="source-block-af5adc0d-d88b-4223-9cd1-e45500d20e39"></span>

Fig.3 按验证 token 的首轮到末轮 NLL 改善分组。定义 Δ=ℓ₁−ℓ_last，以 ε=10⁻³ 划分负收益、近零收益和正收益。三个比例之和为 1，表中百分数的末位误差来自显示舍入。[Fig.3、§3.2](https://arxiv.org/html/2609.35748v1#S3.SS2)

<span id="source-block-112128b1-ac85-4af3-b316-a48b9eddee53"></span>

```math
\begin{aligned}
f_-&=\frac1N\sum_t\mathbf1[\Delta_t<-\varepsilon],\\
f_0&=\frac1N\sum_t\mathbf1[|\Delta_t|\le\varepsilon],\\
f_+&=\frac1N\sum_t\mathbf1[\Delta_t>\varepsilon].
\end{aligned}
```

<span id="source-block-4f731bd3-959d-4b13-aa42-8061209b45b6"></span>

| 后训练架构与比较 | 负收益 | 近零收益 | 正收益 |
| --- | ---: | ---: | ---: |
| Ouro 1→2 | 21.71% | 52.27% | 26.02% |
| Huginn 1→3 | 15.94% | 33.46% | 50.61% |

<span id="source-block-1d37ce4f-a9a1-423c-aefb-93874ad37902"></span>

先把这张表还原成一个验证位置。teacher forcing 给定数据中的文字前缀和真实的下一个 token；第一轮与后续轮是在预测同一个目标。继续计算后，模型若给这个目标更高的概率，NLL 就下降，Δ 为正；概率降低时，Δ 为负。上面的 ε 进一步把变化很小的位置归入近零收益。

<span id="source-block-f02deb18-f020-40d1-9e72-6e8c23d56209"></span>

固定循环会为这些位置继续付出计算，但收益并不一致。以 Ouro 的第一轮到第二轮为例，26.02% 的位置属于正收益，52.27% 接近零，21.71% 反而变差。选择性深度的动机正在这里：再算一次是否有用，取决于当前预测位置，不能只用统一增加轮数来回答。

<span id="source-block-a9490856-dc41-448f-90ff-00cbc9bbfe29"></span>

TaH2 用当前模型的下一轮探测结果构造训练标签，让 decider 学习从可见状态判断哪些位置值得继续；实际执行时仍由 decider 决定路由。接下来的任务表检验这种局部选择能否转化为整题收益。单个 token 的 NLL 改善和最终答案正确率之间，还隔着后续生成过程；相关反例与充分条件见[后训练课程](/notes/loop-reasoning/posttraining/)。

<span id="source-block-df3c353e-a172-430a-a282-65ad16532c61"></span>

### 1.7B 完整主结果

<span id="source-block-0e525ee6-7603-4d95-adda-e75594d567b1"></span>

下列两表合并对应原文 Table 1，准确率单位为百分比，输出上限 32K。数学六项与其他四项分开排版，但保留所有原始模型配置。[Table 1，p.7](https://arxiv.org/html/2609.35748v1#S5.T1)

<span id="source-block-1062a219-bd93-45f3-84ad-17d5616e67bf"></span>

| 模型与 ceiling | AIME24 | AIME25 | AIME26 | AMC23 | MATH500 | Olympiad |
| --- | --- | --- | --- | --- | --- | --- |
| Standard M1 | 11.0 | 13.0 | 11.9 | 45.3 | 75.2 | 40.4 |
| Huginn M3 | 13.9 | 14.2 | 11.8 | 45.8 | 74.4 | 40.6 |
| Huginn M7 | 10.6 | 14.0 | 13.3 | 49.0 | 75.2 | 40.4 |
| Ouro M2 | 10.5 | 13.9 | 11.0 | 45.9 | 75.1 | 37.9 |
| Ouro M4 | 14.5 | 13.4 | 10.3 | 45.9 | 73.5 | 37.6 |
| TaH2 fixed M2 | 15.7 | 15.0 | 13.2 | 51.1 | 76.1 | 42.1 |
| TaH2 fixed M4 | 14.3 | 15.4 | 13.8 | 52.2 | 78.5 | 44.0 |
| TaH2 M2 | 15.7 | 16.5 | 14.3 | 50.7 | 78.1 | 43.8 |
| TaH2 M4 | 16.3 | 16.9 | 14.2 | 50.9 | 78.6 | 44.1 |
| TaH2 M8 | 16.3 | 17.9 | 14.5 | 52.8 | 79.3 | 47.4 |

<span id="source-block-70483734-3983-46c8-ab36-d60c62fd3319"></span>

| 模型与 ceiling | HumanEval | MBPP | GPQA | BFCL | 十项平均 | FLOPs/token |
| --- | --- | --- | --- | --- | --- | --- |
| Standard M1 | 63.6 | 70.2 | 31.5 | 15.3 | 37.7 | 1.00× |
| Huginn M3 | 61.7 | 70.3 | 33.7 | 15.6 | 38.2 | 1.90× |
| Huginn M7 | 64.2 | 69.3 | 33.4 | 14.8 | 38.4 | 3.72× |
| Ouro M2 | 61.2 | 69.6 | 34.0 | 14.3 | 37.3 | 1.91× |
| Ouro M4 | 58.2 | 67.3 | 31.2 | 12.1 | 36.4 | 3.75× |
| TaH2 fixed M2 | 63.8 | 70.8 | 31.9 | 14.6 | 39.4 | 1.99× |
| TaH2 fixed M4 | 65.2 | 71.3 | 32.8 | 15.5 | 40.3 | 3.97× |
| TaH2 M2 | 66.8 | 71.8 | 33.1 | 15.3 | 40.6 | 1.21× |
| TaH2 M4 | 65.2 | 72.3 | 33.0 | 17.4 | 40.9 | 1.47× |
| TaH2 M8 | 72.6 | 74.4 | 32.0 | 17.9 | 42.5 | 2.37× |

<span id="source-block-218eed5e-597d-49f9-926c-98947c9a29ed"></span>

现在可以把前面的动机与主结果对照起来。TaH2 的训练 ceiling 从 M2 增加到 M4、M8，十项平均由 40.6 升到 40.9、42.5；但 GPQA 对应为 33.1、33.0、32.0，HumanEval 的 M4 也低于 M2。因此，平均分提高说明这组任务的总体收益，逐项分数则告诉我们收益出现在哪里；更深的配置并没有让每项任务都单调改善。

<span id="source-block-9975b68b-d530-49fa-8fc8-948fc458f22a"></span>

成本列回答的是另一个问题：整个评测过程中，每生成一个 token 平均用了多少 decode 计算。计算方法是先汇总跨 benchmark 的全部 decode FLOPs 和生成 token 数，取两者之比，再相对 Standard 归一化。它按生成 token 汇总，不能读成“每条回答的成本倍率再等权平均”。

<span id="source-block-5bd8ab4a-ce55-48b4-bcd9-79938d6bbae1"></span>

把能力和成本放在一起读时，还要保留回答长度：FLOPs/token 较低，并不直接给出固定输出长度下的整题成本，也不能单独恢复一条实际回答的总成本。后面的逐题轨迹算法会把执行深度、可见 KV 和输出长度重新合到同一份成本记录中。

<span id="source-block-bb772680-ee6c-46fd-8b89-cb0fc7a52f58"></span>

Fig.4a 与 Fig.8 的最终验证 loss 差值使用各自深度 ceiling 重新训练的 checkpoint。负值表示低于 Standard：[Fig.8，p.26](https://arxiv.org/html/2609.35748v1#A3.SS1)

<span id="source-block-5334e2b9-be17-478f-b356-2b3acac28de1"></span>

| 配置 | 最终验证 loss 减 Standard |
| --- | ---: |
| TaH2 M2／M4／M8 | −0.0064／−0.0078／−0.0106 |
| TaH2 fixed M2／M4 | −0.0063／−0.0096 |
| Huginn M3／M7 | +0.0003／+0.0004 |
| Ouro M2／M4 | +0.0051／+0.0146 |

<span id="source-block-d781ef76-25b4-4ee1-8a88-c1f0ae56cdf8"></span>

### 4B 与 8B

<span id="source-block-b91899a2-ecd2-4a54-b524-56f30d2d40ac"></span>

Table 3 的 TaH2 ceiling 为 M2。在各自规模和配方内，九项任务都优于 Standard；平均增益分别为 3.2 与 2.4 个百分点。它支持方法在这两种规模下的有效性，跨规模强弱还同时包含教师、训练 tokens 和任务集合变化。[Table 3、§5.3](https://arxiv.org/html/2609.35748v1#S5.SS3)

<span id="source-block-517c3b8e-280e-4313-8b4a-0f433bf60039"></span>

| 任务 | 4B Standard | 4B TaH2 | 8B Standard | 8B TaH2 |
| --- | --- | --- | --- | --- |
| AIME24 | 52.0 | 57.7 | 66.4 | 70.8 |
| AIME25 | 40.3 | 43.1 | 52.0 | 55.9 |
| AIME26 | 45.5 | 52.4 | 60.9 | 63.4 |
| AMC23 | 88.6 | 89.2 | 93.5 | 96.6 |
| OlympiadBench | 67.0 | 68.4 | 73.3 | 74.4 |
| IMO AnswerBench | 31.2 | 33.8 | 40.3 | 42.0 |
| LiveCodeBench v6 | 35.4 | 37.3 | 42.1 | 44.4 |
| SuperGPQA | 28.6 | 30.5 | 38.0 | 39.2 |
| BFCL | 29.2 | 33.6 | 38.0 | 39.4 |
| 平均 | 46.4 | 49.6 | 56.1 | 58.5 |

<span id="source-block-5bc53a7c-84c6-46e4-9dfa-0f3b3ef181c4"></span>

### 六项单因素消融

<span id="source-block-dba1ba97-5dec-42e0-ad97-489adf1953b2"></span>

Table 4 在 1.7B、M2、32K 输出设置下改变一个配方因素，结果为 AIME24–26 的 avg@32。最后一列是相对完整 TaH2 的平均百分点变化。[Table 4、§5.4](https://arxiv.org/html/2609.35748v1#S5.SS4)

<span id="source-block-ef5e6f5f-2c18-46eb-b033-c25648c0c97a"></span>

| 配置变化 | AIME24 | AIME25 | AIME26 | 平均 | 变化 |
| --- | --- | --- | --- | --- | --- |
| 完整 TaH2 | 15.7 | 16.5 | 14.3 | 15.5 | — |
| 改为 top1 mismatch 标签 | 14.1 | 16.5 | 12.6 | 14.4 | −1.1 |
| 改为等权决策 BCE | 12.9 | 11.6 | 11.3 | 11.9 | −3.6 |
| 覆盖率改为 1 | 14.0 | 13.4 | 13.5 | 13.6 | −1.9 |
| 训练决策改为随机采样 | 14.4 | 16.3 | 12.0 | 14.2 | −1.3 |
| updater 改为 top100 embedding | 13.3 | 14.1 | 12.7 | 13.4 | −2.1 |
| 输出仅取最终轮 | 15.4 | 15.4 | 12.7 | 14.5 | −1.0 |

<span id="source-block-747904f7-aefa-4c7d-a0a5-b560385cc093"></span>

令 z 为六项设计选择，A(z) 为按该配置训练后得到的评测值。单因素差异只估计完整配方附近的条件效应；若要判断两个组件是否互补，需要额外的组合消融：

<span id="source-block-9fbae50b-776f-45ab-9608-6f7b35b6e5b7"></span>

```math
\begin{aligned}
\Delta_j&=A(z_\star)-A(z_\star^{(-j)}),\\
I_{jk}&=A_{11}-A_{10}-A_{01}+A_{00}.
\end{aligned}
```

<span id="source-block-64dc48a0-1975-414c-99a6-234433319fee"></span>

Table 4 没有给出完整两因素网格或训练 seed 分布，所以不能将六个下降幅度相加作为总贡献，也不能由最大下降直接推得在其他基座或预算下的优先级。Top1 mismatch 标签仅判断当轮是否猜对，iteration gain 标签判断再计算一次的收益；二者对应不同的监督对象。

<span id="source-block-83986208-b08f-4970-9dab-a0438f791349"></span>

## 训练深度与推理深度

<span id="source-block-9cb9e1ac-47b7-45d2-8ad0-c29dc7e16568"></span>

设训练程序在 ceiling M_train 下产生参数 θ̂(M_train)，评测时另选 M_infer。扩大训练 ceiling 与固定权重改变推理 ceiling 对应两类不同的有限差分：

<span id="source-block-a5882a06-5a21-4360-b0b1-ae2300376293"></span>

```math
\begin{aligned}
\Delta_{\rm recipe}
&=A(\widehat\theta(M_2),M_2)-A(\widehat\theta(M_1),M_1),\\
\Delta_{\rm runtime}
&=A(\widehat\theta(M),M'_2)-A(\widehat\theta(M),M'_1).
\end{aligned}
```

<span id="source-block-0a250638-f1fe-4646-b78e-702ea56136d3"></span>

Fig.1b、Fig.4a 和 Table 1 主要比较第一类；Table 12 的下半部分才固定同一个 M8 checkpoint。下面保留 Table 12 的完整分组，准确率为 AIME avg@32：[B.2、Table 12](https://arxiv.org/html/2609.35748v1#A2.SS2)

<span id="source-block-c40530ce-de11-42f0-8b4e-ef49ef6c2a2c"></span>

| 训练配置 | 推理 ceiling | AIME24 | AIME25 | AIME26 | 平均 | 平均实际深度 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard M1 | 1 | 11.0 | 13.0 | 11.9 | 12.0 | 1.00 |
| TaH2 M2 | 2 | 15.7 | 16.5 | 14.3 | 15.5 | 1.23 |
| TaH2 M4 | 4 | 16.3 | 16.9 | 14.2 | 15.8 | 1.49 |
| 同一 TaH2 M8 | 4 | 13.6 | 15.1 | 11.4 | 13.4 | 1.55 |
| 同一 TaH2 M8 | 8 | 16.3 | 17.9 | 14.5 | 16.2 | 2.22 |
| 同一 TaH2 M8 | 12 | 16.9 | 17.2 | 14.5 | 16.2 | 2.72 |

<span id="source-block-7931bf68-bb4b-47bc-8d3c-8210a196a99c"></span>

推理 ceiling 8→12 时，显示的平均实际深度增加约 22.5%，平均准确率仍为 16.2%；各年度变化也不同。降低为 4 则明显损失准确率。这一实验支持该 checkpoint 对适度加深具有一定鲁棒性，但没有证明超出训练深度可以继续获得有效收益。不同模型的 M4 推理结果也不能仅按相同 ceiling 合并。

<span id="source-block-77264f16-e928-415a-90c1-34168359dea8"></span>

## 教师选择与机制证据

<span id="source-block-bd572c65-a144-4782-bfe2-f800007ec0af"></span>

### 教师大小与学生收益

<span id="source-block-481fb584-4ff3-4f56-b8ba-832dff524286"></span>

Table 11 固定 Qwen3 1.7B Standard 学生、相同 prompts 和训练 recipe，比较教师回答来源。结果为 32K、avg@32：[B.1、Table 11](https://arxiv.org/html/2609.35748v1#A2.SS1)

<span id="source-block-125efc49-828a-4aea-830e-a3977f560010"></span>

| 教师 | AIME24 | AIME25 | AIME26 | 平均 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 8B | 11.04 | 13.02 | 11.87 | 11.98 |
| Qwen3 32B | 9.69 | 11.98 | 9.38 | 10.35 |
| Qwen3 235B A22B | 10.42 | 9.90 | 10.21 | 10.18 |

<span id="source-block-53fbff83-e8a5-4c8f-90fa-5de1674660f1"></span>

该实验选择的是这套学生与训练配方下的回答来源；教师规模与学生收益没有在此呈单调关系。教师的答案质量、长度、格式、过滤后的样本组成和学生可学性共同变化，不能仅由这三行识别其中某个机制。

<span id="source-block-608969a9-6a18-4943-9c02-4046b58b8bd0"></span>

### Gate 与单步收益的对应

<span id="source-block-d67e4a42-8281-4090-90fa-f2d6f8eb3d08"></span>

Fig.6 将验证 token 按继续分数分桶，再计算第二轮平均 NLL 改善。整体上分数越高，平均收益越大；0.5 附近平均收益接近零。[§5.5、Fig.6](https://arxiv.org/html/2609.35748v1#S5.SS5)

<span id="source-block-01715688-406b-42ab-81e6-df127adf36be"></span>

```math
\widehat\mu(B)
=\frac{\sum_t\delta_t\,\mathbf1[g_t\in B]}
{\sum_t\mathbf1[g_t\in B]}.
```

<span id="source-block-d2dd1134-09aa-48d2-9e03-b744f6849599"></span>

这测量条件平均收益，而非“继续概率等于发生正收益的概率”的校准。比如 1% 的 token 收益为 100、99% 为 −0.1，平均收益仍为 0.901，但正收益概率仅 1%。点估计随分数上升也不保证每个被选 token 都受益。更严格的成本决策还需考虑代价与后续可达收益。

<span id="source-block-44f9e89a-d554-4d5b-9d6f-a53c1cdf4ff7"></span>

### 单轮监督怎样对应最终混合与联合收益

<span id="source-block-31bd86de-9ef8-4505-9c43-51563b08ff81"></span>

原文 Eq.5 比较相邻两轮各自的 CE，Eq.4 则混合各轮预测，二者衡量的改善不同。为比较它们，先固定已有 KV、较浅轮次的贡献和当前继续分数 g，只让当前位置再执行一轮。记 A 为较浅轮次已贡献给目标词的概率质量，R 为到达当前轮的剩余权重，a、b 为当前轮和下一轮赋给目标词的概率。下式中，δ 是单轮 CE 改善，Δ_mix 是最终混合 NLL 改善，ξ 汇总了两者关系中的权重：

<span id="source-block-e7a9afec-34fb-4803-810d-8af1634d8c07"></span>

```math
\delta=\log(b/a),\qquad
\xi=\frac{Rga}{A+Ra},\qquad
\Delta_{\rm mix}=\log\!\left(1+\xi(e^\delta-1)\right).
```

<span id="source-block-09a3fbf2-9eea-4630-8d9f-e656eef2eed6"></span>

在精确停止混合、正目标概率且 ξ>0 时，两种改善同号，但跨 token 的排序未必相同；原始增益覆盖 99% 因而不能直接改写成最终混合 NLL 改善覆盖 99%。这个推导不推翻 Fig.6 的分桶观察，也不证明更换标签必然更好；它明确了观察和监督所针对的预测器。[原文 Eq.4–7](https://arxiv.org/html/2609.35748v1#S4) · [单轮评分](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L312-L368) · [混合实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L579)

<span id="source-block-427dadcb-57f4-442e-ab9c-4cbe0e38de6a"></span>

续算涉及哪些位置，也会改变收益的含义。在固定输入、较浅缓存和逐位置算子等条件下，posterior mask 可以同时计算“每个停止位置分别续算一次”时的查询值。若让多个位置共同续算，较早位置新增的 KV 却可能改变较晚位置的损失。因此需要区分三种量：当前位置的混合改善、只续算一个位置后对全部有效目标汇总的改善，以及整组位置共同续算的改善。完整证明、可达的排序反例、因果 attention 反例与 22 行算法 T10 见[内部循环：收益与干预范围](/notes/loop-reasoning/internal-loop/)。这是对原文目标的数学深化与诊断方案，未新增模型性能结论。

<span id="source-block-3336e938-6985-4bc4-9b8a-f9988589a0a0"></span>

### 单步监督与最优停止的差别

<span id="source-block-b2a14b6d-8cc0-4610-982f-b4862067090a"></span>

以完整状态 s 包含 hidden、KV、已执行深度和混合读出累加器；若停止损失为 ℓ_stop(s)，继续代价为 κ(s)，则有限 horizon 的最优停止问题满足 Bellman 递推。它是分析参照，TaH2 的 online gain 标签并未求解这一动态规划：

<span id="source-block-1d693cc7-d146-418d-b161-94d5ff4c0df3"></span>

```math
\begin{aligned}
V_M(s)&=\ell_{\rm stop}(s),\\
V_m(s)&=\min\!\left\{\ell_{\rm stop}(s),
\lambda\kappa(s)+\mathbb E[V_{m+1}(s')\mid s]\right\}.
\end{aligned}
```

<span id="source-block-68cd888b-5312-4562-98b5-ff76624b90be"></span>

即使每轮代价为 1，三个候选停止损失依次为 2、2.2、0.3，且 λ=0.1，第一步的单轮改善为 −0.2，继续两步后的总目标却是 0.5。该反例说明一步标签可能遗漏先退后进的路径。实际 TaH2 又以单轮 q 的 NLL 生成标签、以停止权混合 q 训练预测，两者必须分别记录。未来把 token 局部代理接到最终答案奖励或计算预算时，需要明确改变了哪个目标。[§4.2、Appendix D、§6](https://arxiv.org/html/2609.35748v1#A4)

<span id="source-block-dd327ea6-52bc-4b98-bf10-9189341575cb"></span>

### 深度可视化与跨轮注意力

<span id="source-block-18326dac-0f1b-49c8-b069-e5c40a06ccda"></span>

Fig.11 展示三个选中的正确回答。显示片段的平均深度在 HumanEval 从 3.31 降至 1.04，OlympiadBench 从 2.46 降至 1.29，GPQA 从 3.47 到 3.69。它们是局部案例，不能推广为所有代码或数学表达式都应少算。[C.4、Fig.11](https://arxiv.org/html/2609.35748v1#A3.SS4)

<span id="source-block-b72d2f68-ca05-4c2e-a5c9-cc2fd2498223"></span>

Fig.12–13 使用 M8 模型、随机抽取的 100 条 teacher forced 验证序列，并保留 decider 的实际路由。Fig.12 对到达第八轮的 queries 先作序列内平均，再在序列间平均；误差棒是一个序列级标准差。三类代表 head 的第一轮／后续轮注意力质量为 98.9／1.1%、0.0／100.0%、50.6／49.4%。Fig.13 显示首 128 个回答位置：每个 query 位置仅对实际到达第八轮的序列平均；共用对数色标，灰色表示没有已执行 query。原上下文保持不变，图中省略 prompt keys 时不重新归一化。[C.5、Fig.12–13](https://arxiv.org/html/2609.35748v1#A3.SS5)

<span id="source-block-48ba7ae2-99c9-4dbc-98c3-1125c1b49bf7"></span>

设 Qᵢ 是第 i 条序列中到达第八轮的 query 集合，α 为已经在完整可见键集合上归一化的 attention。对应的层级统计为：

<span id="source-block-27f75ebb-74cd-4d7c-a6ec-a145154b5c13"></span>

```math
\begin{aligned}
u_{ij}&=\frac1{|Q_i|}\sum_{t\in Q_i}
\sum_s\alpha_{i,t;(s,j)},\\
\bar u_j&=\frac1{|\mathcal I|}\sum_{i\in\mathcal I}u_{ij},
\qquad\mathcal I=\{i:|Q_i|>0\}.
\end{aligned}
```

<span id="source-block-71f49286-9e3f-4869-b602-cfc91ca9f772"></span>

实现需要声明空 query 序列如何处理；将所有 query 直接合并会改变序列权重。注意力质量表示检索分配，不能单独证明某轮状态对答案具有因果贡献：若所有 value 都相同，改变归一化 attention 权重也不改变 attention 输出。研究跨轮记忆时，需进一步干预相应 KV 或 value 并保持其他执行条件一致。

<span id="source-block-d92a29e8-8619-4b37-b601-e9d6fc2b5fcd"></span>

## 并行采样与投票

<span id="source-block-cdf36da3-d343-42b3-8a01-04c1bcba390b"></span>

Appendix C.3 的 cons@n 从每题已有的 32 条完整回答中均匀、无放回地抽取 n 条，投票后判断答案，再对随机抽取求平均。TaH2 M2 的 cons@32 为 27.3%，Standard 为 21.9%；M2、M4、M8 的 TaH2 在该设置中位于 27.3–29.3%。投票答案规范化、平票和无有效答案的规则属于统计定义的一部分。[Fig.10、C.3](https://arxiv.org/html/2609.35748v1#A3.SS3)

<span id="source-block-dacf3ae8-7892-44f6-bcc9-434459d8d522"></span>

令 aᵢⱼ 为规范化回答，S 为从 K=32 个索引中抽取的大小 n 子集。条件于已有回答池，cons@n 与平均 compute 的定义为：

<span id="source-block-68cf4120-03c2-4646-a375-aa4895c1acf9"></span>

```math
\begin{aligned}
\widehat A_{\mathrm{cons@}n}
&=\frac1Q\sum_i
\mathbb E_S\,\mathbf1[\operatorname{vote}(a_{ij}:j\in S)=y_i^*],\\
\widehat C_n
&=\frac1Q\sum_i\frac nK\sum_{j=1}^K C_{ij}.
\end{aligned}
```

<span id="source-block-c109fcbe-aa89-45e3-8dc3-40d145812673"></span>

第二式由每条记录入选概率 n/K 推出，不要求各回答长度相同。它是子集成本的期望；某一次抽取仍须把所选回答的实际成本求和。n=1 时在相同评分规则下退回 avg，n=32 时使用整个记录池。跨 n 的曲线共享同一批回答，不能当成独立重复实验。

<span id="source-block-2423d584-8476-407e-aefd-31d6f8302904"></span>

## 从执行轨迹计算成本

<span id="source-block-6a07b392-2ee4-458b-81f7-38cd253b08c2"></span>

### 每次调用的矩阵乘法

<span id="source-block-868ae8f0-38ad-4fce-afc9-3ee9f81147ff"></span>

论文以一次乘加为两次 FLOPs，只计矩阵乘法；normalization、activation、softmax、top-k、停止权混合、采样、显存流量与服务开销不在这一账本中。设层数 L、宽度 d、总 KV 宽度 d_kv、FFN 宽度 d_ff、词表 V；updater 和 decider 的 FFN 宽度分别为 dᵤ、d_d。[A.4.1、Eq.13](https://arxiv.org/html/2609.35748v1#A1.SS4.SSS1)

<span id="source-block-c58f146c-69ca-400a-9a39-5e73395875b9"></span>

```math
\begin{aligned}
F_{\rm bb}&=L[4d(d+d_{\rm kv})+6dd_{\rm ff}],\\
F_{\rm head}&=2dV,\\
F_{\rm upd}&=4d^2+6dd_u,\\
F_{\rm gate}&=2d(2d+k)+6dd_d+2d,\\
F_{\rm attn}(S)&=4LdS.
\end{aligned}
```

<span id="source-block-29782644-2273-4dc7-9ade-09f17534a443"></span>

| 1.7B 每次调用 | GFLOPs |
| --- | ---: |
| 完整 backbone 的投影与 MLP | 2.819 |
| LM head | 0.622 |
| attention 每个可见 key | 0.000229 |
| Updater | 0.042 |
| Decider | 0.050 |
| 本文 Huginn 输入 adapter | 0.008 |

<span id="source-block-abbdac0d-b3f4-4ec3-96a4-9a39adf539d6"></span>

Table 9 使用 L=28、d=2048、d_kv=1024、d_ff=6144、V=151936、k=dᵤ=d_d=2048。每轮 LM head 的调用时机也影响成本：TaH2 每轮都需要预测，Ouro／Huginn 的这组推理对照只在最终轮调用 head；训练目标又可能要求更多 head 调用。

<span id="source-block-3300638d-7c78-4fa6-b6bd-8828d4f43672"></span>

### DUO 历史使 attention 成本依赖深度分布

<span id="source-block-18c171aa-fa9e-4dcc-856e-9d78c344bb78"></span>

设 prompt 长度 P，第 j 个 prompt 位置真实执行 rⱼ 轮；输出共有 K 个 token，首个输出由 prefill 产生，论文的 decode 账本处理 N=K−1 个后续输入位置。位置 t 实际执行 mₜ 轮，第 m 轮可见的 key 数量与响应总成本为：[Eqs.14–16](https://arxiv.org/html/2609.35748v1#A1.SS4.SSS2)

<span id="source-block-28d75268-0217-4a53-981c-3c8cf2fa5103"></span>

```math
\begin{aligned}
S_t^{(m)}
&=\sum_{j=1}^{P}\min(r_j,m)
+\sum_{u<t}\min(m_u,m)+m,\\
C_{\rm dec}
&=\sum_{t=1}^{N}\Big[
m_t(F_{\rm bb}+F_{\rm head})
+4Ld\sum_{m=1}^{m_t}S_t^{(m)}\\
&\qquad +(m_t-1)F_{\rm upd}
+\min(m_t,M-1)F_{\rm gate}\Big].
\end{aligned}
```

<span id="source-block-80cc7d87-a17b-450b-9ef7-2273ac725a9e"></span>

这里虽然不计 prefill 的计算，prompt 在各轮留下的 KV 仍影响 decode attention。若所有位置都执行 M 轮，则每个 query 深度 m 可见 m(P+t) 个键，DUO 的 attention 项随 M(M+1)/2 增长；骨干投影项仅随 M 增长。这是上述计数公式的推论，不能用“平均循环数乘单轮 FLOPs”替代完整账本。

<span id="source-block-5c3c3764-4191-4c71-b96e-79306fcb7134"></span>

```math
\begin{aligned}
S_t^{(m)}&=m(P+t),\\
\sum_{m=1}^{M}S_t^{(m)}
&=(P+t)\frac{M(M+1)}2,\\
\text{KV bytes}
&\approx2L\,d_{\rm kv}\,b_{\rm elem}
\left(\sum_{j=1}^{P}r_j+\sum_{t=1}^{N}m_t\right).
\end{aligned}
```

<span id="source-block-856b4d12-e337-43c4-aa80-ee56aeac624b"></span>

KV 公式只计实存 K/V 元素，未包括页粒度分配、元数据或混合累加器。TaH2 fixed 使用另一种 causal 可见性，论文给出的每轮 key 数是 P+t−1+m；独立每轮 KV 流的 Ouro／Huginn 对照是 P+t。各自成本不能套用同一个可见性假设。[Eqs.17–20](https://arxiv.org/html/2609.35748v1#A1.SS4.SSS2)

<span id="source-block-bf14df18-0fcc-4436-88f7-30b29b3e7647"></span>

**论文调用计数与部署 graph 的区别。** 上述 updater 的 mₜ−1 次、gate 的 min(mₜ,M−1) 次，按论文所需的语义调用计费。固定 MiniSGL `graph.py` 先对整批执行 `compute_iter_embeds`，再用 `torch.where` 选择首轮 base embedding 或后续轮更新值；同一 graph 中 decider 也无条件执行，深度 ceiling 在 engine 后续逻辑中应用。因此当前融合实现的首轮会产生被丢弃的 updater 结果，最后一轮也可能先计算 gate 再强制退出。实际算子 FLOPs 还需记录 graph batch 大小、填充位置与各模块真实调用，不能直接把论文账本称为已测物理计算量。[graph.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/graph.py#L226-L246) · [engine 深度上限](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L510-L514) · [生成实现页](/notes/loop-reasoning/generation/)

<span id="source-block-28b52b92-4e0e-4762-bc05-b83afdd7bcbd"></span>

### Lookahead 的训练成本

<span id="source-block-f55956db-cc72-4d5b-86ef-d82b86b28a1f"></span>

令 N_m 为实际执行第 m 轮的全部输入位置数，A_m 为这一轮主路径 query–key 对数；它们与预测 loss 的有效监督 token 分母是不同的计数。按论文把 backward 近似为对应 forward 的两倍：[Eqs.21–23](https://arxiv.org/html/2609.35748v1#A1.SS4.SSS3)

<span id="source-block-701e7834-e989-4820-b6c0-49a1e970c1a5"></span>

```math
\begin{aligned}
F_{\rm train}&\approx3F_{\rm fwd}+F_{\rm label},\\
F_{\rm fwd}
&=\sum_{m=1}^{M}\!\left[N_m(F_{\rm bb}+F_{\rm head})+4LdA_m\right]\\
&\quad+\sum_{m=2}^{M}N_mF_{\rm upd}
+\sum_{m=1}^{M-1}N_mF_{\rm gate},\\
F_{\rm label}
&=\sum_{m=1}^{M-1}\!\left[
N_m(F_{\rm upd}+F_{\rm bb}+F_{\rm head})
+4Ld\widetilde A_{m+1}\right].
\end{aligned}
```

<span id="source-block-750e5bbf-4bf1-4096-85ac-843d5d5a9934"></span>

辅助探测需要重建继续 token 在下一轮的上下文，所以论文 Eq.23 对全部 N_m 个当前位置重新前向，而不只处理 N_m−N_{m+1} 个停止位置。辅助 query 只读取其他 token 的真实执行状态，并保留自己的当前轮状态：

<span id="source-block-038a1aea-b45a-4bea-aa41-76ef8b61ed5e"></span>

```math
\widetilde A_{m+1}
=\sum_{t:m_t\ge m}
\left[\sum_{u<t}\min(m_u,m+1)+m+1\right].
```

<span id="source-block-7a05f15b-59f3-4b9c-829f-692efa96dc9b"></span>

先用 M2 看论文的成本估算。按 Eq.23，每个非末轮都计入 N_m 个探测位置；忽略 attention 与辅助模块，记第二轮执行比例为 p=N₂/N₁，主要单轮成本为 B。主路径成本是 3N₁(1+p)B，lookahead 成本是 N₁B，合计相对 Standard 约为 4/3+p。这个式子说明监督计算在论文估算中占多少。实际代码是否发起探测、探测哪些位置，还取决于下面的控制流，因此该比例不能作为每次运行的实际成本或耗时下界。

<span id="source-block-114ed98a-10fb-4f6c-8957-ca230ee119b4"></span>

#### 论文账本与固定源码的辅助调用

<span id="source-block-afd53961-e830-48f3-9ea1-47ac22f7f861"></span>

把论文的探测位置数对应到代码时，需要依次查看三个动作。第一，`capture_posterior_inputs` 在每个非末轮为实际有效位置预先计算无梯度 updater 输入，无论随后是否需要探测都会产生这项成本。第二，只有 DP 范围内仍缺少所需 CE 时，`_compute_posterior_labels_side_pass` 才调用 side forward。第三，side forward 读取缓存中的 `stored_mask`；尚未到最终 ceiling 时，还会为这个集合再次计算 updater。由于 `add_side_inputs` 将新旧 mask 取并集，side forward 的位置集合不一定等于上一轮主路径实际执行的集合。[capture 与 mask 合并](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L465-L517) · [触发判定](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L275-L287) · [side forward 与下一轮输入](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L54-L211)

<span id="source-block-42598ac6-c187-4796-87fd-75699822cab9"></span>

把第 m 轮主路径有效位置记为集合 𝒜_m，side depth r 的实际有效 forward 集合记为 ℱ_r，DP 共同的探测触发指示记为 ζ_r，实际辅助 query–key 对数记为 Q_r。按有效 token 的矩阵乘法计数，固定代码的辅助成本可用以下调用账本描述；TP 分片不重复计同一个逻辑 token，dummy、填充和通信另外记录：

<span id="source-block-bc77abaa-6700-41b8-96f2-4fd1d473e2b3"></span>

```math
\begin{aligned}
G&=F_{\rm bb}+F_{\rm head},\\
U_{\rm capture}&=\sum_{m=1}^{M-1}|\mathcal A_m|F_{\rm upd},\\
F_{\rm aux}^{\rm code}
&=U_{\rm capture}
+\sum_{r=2}^{M}\zeta_r\Big[
|\mathcal F_r|G+4LdQ_r\\
&\hspace{40mm}+\mathbf1[r<M]|\mathcal F_r|F_{\rm upd}\Big].
\end{aligned}
```

<span id="source-block-f4246270-ddc9-4ce6-bc6e-48f56008c9d7"></span>

一个 M3 两 token 例子可区分集合：a 在第一轮停止，b 在第二轮停止。第二层探测为补 a 的 CE，会对 {a,b} 前向，并为深度 3 缓存 {a,b} 的更新输入；第三层为了补 b 的 CE 触发探测时，缓存集合仍是 {a,b}，而真实第二轮集合只有 {b}。多出的 a 在注意力可见性中受到隔离，但其前向仍产生计算。第二层 side pass 还需生成下一轮 updater；第三层已到 ceiling，不再生成深度 4 输入。这个例子是固定控制流的解析推演，不是模型性能实验。

<span id="source-block-f8afdab4-50b8-473b-a762-9de02f67a8bf"></span>

在 M2 的默认路径下，触发时 ℱ₂=𝒜₁。仍忽略 attention 和小模块时，同一简化成本参照改为下式；ζ₂ 必须从实际 missing CE 判定读取，不能只由平均深度推断。若所有所需 CE 已由主路径得到，side backbone 可以跳过，但 capture updater 的成本仍然存在。

<span id="source-block-69b48449-3366-45c5-997d-e4c2ba17a679"></span>

```math
\frac{F_{\rm train}^{\rm code}}{F_{\rm Standard}}
\approx 1+p+\frac{\zeta_2}{3},
\qquad p=\frac{N_2}{N_1}.
```

<span id="source-block-751cd152-5a24-4223-a22b-b5d1dcd7f0b7"></span>

因此应同时保留两份账本：论文 Eq.21–23 用于恢复原表的比较口径；真实训练 trace 另存主路径 mask、DP missing 判定、side forward_mask、side KV 可见对数、capture 与 next-input updater 的调用数。表 10 是论文报告的估算，不能把它替换成未经 profiler 或调用轨迹核对的当前实现测量。[循环源码拆解](/notes/loop-reasoning/internal-loop/) · [分布式与性能](/notes/loop-reasoning/systems-evaluation/)

<span id="source-block-3b9180d8-202b-434e-9bbe-f207af289a11"></span>

Table 10 的单位为 10¹⁸ FLOPs，所有 1.7B 模型均为 6,402 步、global batch 128、三个 epochs。论文成本不含 activation checkpoint 重算等实现开销。原文显示数有舍入，保持原表总量。[Table 10，p.25](https://arxiv.org/html/2609.35748v1#A1.T10)

<span id="source-block-7aef7efd-6d03-4119-b052-985db2ed2d20"></span>

| 配置 | M | 主 forward 加 backward | Label | 总量 | 相对 Standard |
| --- | --- | --- | --- | --- | --- |
| Standard | 1 | 41.8 | — | 41.8 | 1.00× |
| Huginn | 3 | 77.7 | — | 77.7 | 1.86× |
| Huginn | 7 | 149.3 | — | 149.3 | 3.57× |
| Ouro | 2 | 83.6 | — | 83.6 | 2.00× |
| Ouro | 4 | 167.1 | — | 167.1 | 4.00× |
| TaH2 fixed | 2 | 84.0 | — | 84.0 | 2.01× |
| TaH2 fixed | 4 | 168.4 | — | 168.4 | 4.03× |
| TaH2 | 2 | 62.8 | 15.2 | 78.1 | 1.87× |
| TaH2 | 4 | 87.6 | 26.7 | 114.3 | 2.73× |
| TaH2 | 8 | 164.2 | 52.0 | 216.2 | 5.17× |

<span id="source-block-7aee8b02-ea4d-48ed-bca7-ac4a676cdf23"></span>

同数据、同 recipe 没有实现同训练 compute：TaH2 M2 与 M8 分别为 Standard 的 1.87 与 5.17 倍。lookahead 在表中约占总训练 FLOPs 的 20–24%。Table 1 的同推理预算收益应和这项训练投入一起阅读；进一步的等完整训练预算比较是另一项实验。

<span id="source-block-ba6bf9f5-9eda-4d50-8021-e027ab0deded"></span>

### 实测服务成本

<span id="source-block-8b2c9468-6cfb-4332-bd05-dc1157a450ab"></span>

Table 2 在单张 A800 80GB、bf16、AIME26 的 30 题、32K 上限、temperature 0.6、相同评测 seed 下测量；batch 1／4 分别使用每题 1／4 个样本。吞吐是聚合输出 tokens/s。此处 GFLOPs/token 的倍率 1.22 与 Table 1 的跨任务倍率 1.21 具有不同聚合范围。[A.3、Table 2、Fig.5](https://arxiv.org/html/2609.35748v1#A1.SS3)

<span id="source-block-01470941-277f-4082-97ab-503db854215e"></span>

| Batch | 模型 | Decode GFLOPs/token | 平均 latency 秒 | 聚合 tokens/s |
| --- | --- | ---: | ---: | ---: |
| 1 | Standard | 7.04 | 139.8 | 202.3 |
| 1 | TaH2 fixed | 14.09 | 328.7 | 89.8 |
| 1 | TaH2 | 8.59 | 187.0 | 144.3 |
| 4 | Standard | 7.04 | 220.3 | 507.6 |
| 4 | TaH2 fixed | 14.09 | 483.3 | 229.9 |
| 4 | TaH2 | 8.59 | 285.4 | 376.2 |

<span id="source-block-def7115f-7029-4063-be48-39e1eaaa79f0"></span>

TaH2 的显示延迟倍率约 1.34／1.30，吞吐倍率约 0.71／0.74。原文 Fig.5 横轴写 decode latency，图注与 A.3 则称 end-to-end latency；历史作图边界需用原始日志坐实。固定仓库的 `_post_chat_completion` 在 HTTP 请求前开始计时、收到并解析完整响应后停止，`_online_runtime_metrics` 另按整段运行 wall time 汇总吞吐。该源码说明当前可复核的计时范围，不能自动证明它就是原论文 Figure 5 的完整历史作图脚本。[eval_online.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py#L96)

<span id="source-block-501feb9b-b964-42d2-a851-e8ca14f1c56e"></span>

### 算法 T8 从逐题轨迹重建计算曲线

<span id="source-block-2ec6f193-37ab-494e-8e19-49ecc8e8f9f4"></span>

**输入** 同一模型与固定训练 ceiling 的权重版本、模型维度、输出 token IDs、prompt 与 decode 输入位置的执行深度、题目身份、采样身份、cutoff 集合、固定的答案提取与评分规则。

<span id="source-block-8816edb6-fc3f-4921-bd85-7380d33058a8"></span>

**输出** 每条轨迹各 cutoff 的 decode FLOPs、每题重复采样均值、accuracy–compute 曲线、拟合斜率，以及必要时按题目聚类的区间。

<span id="source-block-a763af74-a7e7-461b-8e0c-6800be9fc712"></span>

**前置条件** 先把记录对齐到模型输入位置：首个输出归 prefill，后续输出对应 K−1 个 decode 输入位置；prompt 深度单列。完整轨迹满足 K≥1，cutoff 满足 b≥1，二者均为整数；无输出或失败尝试保留为失败记录，不进入前缀成本数组的索引计算。不同权重、训练 ceiling、推理 ceiling 和采样设置分别成组。所有预定尝试均有成功、失败或缺失状态；只有完整记录才能给出完整评测估计。FLOPs 常数使用本节 Eq.13 的定义，拟合要求所有平均成本为正且横坐标不全相同。

<span id="source-block-8921fd07-b7ed-443b-a3b8-e5ab347717cc"></span>

<figure class="research-embed"><iframe src="/research-assets/f4b8e92e6235dac5bb859488ce9048d0e71d0bf81aa9822d7a8c05b511409ebc.html" title="算法 T8 从逐题轨迹重建计算曲线" loading="lazy" sandbox=""></iframe><figcaption>算法 T8 从逐题轨迹重建计算曲线 · <a href="/research-assets/f4b8e92e6235dac5bb859488ce9048d0e71d0bf81aa9822d7a8c05b511409ebc.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-8bc09109-9f4b-4859-9900-ab94549feaa0"></span>

历史深度可用累计计数 Hⱼ 表示：Hⱼ 是当前输入位置以前、实际至少执行 j 轮的位置数。第 m 轮只需取 H₁+⋯+H_m，再加自己的 m 个键；每处理完一个深度 r 的位置，对 H₁ 至 H_r 加一。该前缀计数将直接枚举全部历史键的统计改为 O(P+M+Σmₜ) 的账本计算，不改变模型实际 attention 计算量。

<span id="source-block-fc00d6d5-16d4-4db9-b606-83605f3d5ffb"></span>

该算法按论文 A.3、A.4 与 C.2 的协议重建；当前仓库输出对应 `samples.jsonl`、`detailed_results.csv` 与可选的迭代记录。按题目聚类的区间是明确统计单位后的分析扩展，原文没有报告这一完整 bootstrap 流程。[eval_online.py 的记录与汇总](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py#L303)

<span id="source-block-cdf54ebc-4442-4785-9306-f640e92d6af7"></span>

### 将原文结果接回具体运行分支

<span id="source-block-ad303300-b992-455b-936a-4361d7ce5226"></span>

原文的主结果、12 张表与 13 幅图继续按其模型、训练 ceiling、推理 ceiling 和统计协议保留。本次进一步核对的是：这些结果被用于设计下一项后训练实验时，哪些数学对象和当前仓库分支必须分开。

<span id="source-block-3dacbe9b-da08-489c-a55e-a6a6a92c6d49"></span>

| 对象 | 已补的严格结论 | 与原文结果的关系 |
| --- | --- | --- |
| 确定阈值与随机继续 | 固定入态下，sample 加 stop_prob_mix 的文字边缘具有平方生存系数；平均计算仍按原生存率 | 这是仓库可选分支的推导，不将其替换成论文主阈值实验的预测器 |
| 资源强停 | 共同链上 TV(P_d,P_D)=R_{d+1}TV(q_d,Q_tail)；未来 KV、温度与截断需另作分析 | 说明服务结果需要附资源覆盖记录，不能直接由论文平均深度恢复部署行为 |
| 教师条件分布 | 固定教师路线先验下，平均条件 forward KL 与边缘 forward KL 的学生梯度相同，但 loss 相差路线分歧项 | 帮助明确新增 KD 实验的教师；不把原文教师规模消融改写成已验证的随机教师方法 |
| 真实策略评分 | 条件文字 logp、路由—文字联合 logp 与文字边缘 logp各有定义 | 为后续 RL/偏好接入给出数学前提，不声称论文已经完成这类接入 |

<span id="source-block-294bffc5-f928-454a-ae6b-cdd6d226c2d9"></span>

[单 token 精确参考 T9](/notes/loop-reasoning/internal-loop/) · [强停误差与生成分支](/notes/loop-reasoning/generation/) · [随机教师蒸馏](/notes/loop-reasoning/online-distillation/) · [联合与边缘策略](/notes/loop-reasoning/reinforcement-learning/)。上述每项都给出条件、推导、固定代码入口或拟议算法身份；数值反例验证公式，模型能力仍由原文与后续真实实验分别支持。

<span id="source-block-2311f4f1-97e3-4c58-b30e-97435174a42a"></span>

## 原文与源码索引

<span id="source-block-c99ea211-f7f2-4930-9464-25e438aac65f"></span>

这份对照以 2026-09-28 的 arXiv v1 为论文版本，以 TaH 仓库 `78ddf9a2c25713fced03a736c5746161f6ee590c` 为代码版本。表中“已覆盖”表示核心内容有明确讲解、数值和协议可追溯；涉及模型性能的数字均是原文报告，数学推导与小型轨迹核对另行标明。PDF 页码按文件实际页数计。[32 页原始 PDF](https://arxiv.org/pdf/2609.35748v1) · [固定仓库树](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c)

<span id="source-block-fb11d491-003e-4e50-ba1d-9fbb78db9835"></span>

### 章节、表格与图的覆盖

<span id="source-block-578f084a-08e9-43a6-b857-9f7e0b5bb0aa"></span>

| 原文单元 | 核心内容 | 对应讲解与核对 |
| --- | --- | --- |
| §1–2；Fig.1，p.1 | 测试时扩展动机；斜率；16K 深度收益 | 本页计算斜率；明确 16K 与 32K 口径、同预算曲线位置 |
| §3.1–3.2；Fig.3，p.4 | 基线 recurrence；负收益与近零收益 token | 本页基线目标、增益分组；保留三类比例与 ε |
| §4.1；Fig.2，p.2；Eqs.1–4、8–10 | 首轮输入、updater、decider、停止权概率混合 | 本页模型身份与参数；[状态与循环算法](/notes/loop-reasoning/internal-loop/)的完整前向、Jacobian 与梯度路径 |
| §4.2；Eqs.5–7；Appendix D／Eqs.24–26 | 在线 iteration gain、覆盖阈值、类别平衡、联合 NTP 与 BCE | 本页单步收益与停止参照；[状态与循环算法](/notes/loop-reasoning/internal-loop/)的精确定义、零收益／单类边界与停止语义 |
| §5.1；A.1；Table 5，p.19 | 数据、教师、训练超参数 | 本页训练配方与 release 统计；SFT 的在线深度语义 |
| §5.2；Table 1；Fig.4，p.7 | 1.7B 十项性能、每 token 成本、验证 loss、AIME 曲线 | 所有配置与十项分数；loss 图与 decode FLOPs 分别解释 |
| §5.2；Table 2；Fig.5，p.8 | A800 单卡实测与实现开销 | 全六行 latency／throughput；计时边界和标题／caption 差异 |
| §5.3；Table 3，p.8 | 4B／8B 结果 | 九项完整分数；各规模的教师、tokens 与比较对象 |
| §5.4；Table 4，p.8 | 六项设计消融 | 全部 AIME 分数；条件效应与缺少交互网格的边界 |
| §5.5；Fig.6，p.9 | 继续分数与下一轮收益 | 条件均值估计量；概率校准和最优停止的反例 |
| A.2.1；Fig.7，p.20；Table 6，p.21；Eqs.11–12 | 主路径与探测可见性；六种架构／目标 | 基线目标完整区分；[DUO 主路径与探测算法](/notes/loop-reasoning/internal-loop/) |
| A.2.2；Table 7，p.21 | 新增参数与比例 | 三个尺度的完整参数；无偏置 projection 与各 RMSNorm 的计数 |
| A.3；Table 8，p.22 | benchmark 清单、子集和采样数 | 全部任务、题数、avg@K／greedy；均值分母与题目聚类 |
| A.4；Table 9，p.23；Eqs.13–20 | 算子成本与不同 attention 拓扑下的 decode 计费 | 每次调用、DUO 历史、prefill／首 token 边界、基线 KV 差异 |
| A.4.3；Table 10，p.25；Eqs.21–23 | 总训练计算与探测开销 | 全部模型行；有效输入位置与监督分母的区分 |
| B.1；Table 11，p.25 | 1.7B 学生的教师选择 | 全部教师分数；规模不能单独解释效果 |
| B.2；Table 12，p.26 | 训练外推理深度 | 完整六行；明确同一 M8 checkpoint 的 4／8／12 推理 |
| C.1；Fig.8，p.26 | 所有配置验证 loss 曲线 | 最终差值完整记录；不同训练 ceiling 的权重身份 |
| C.2；Fig.9，p.27 | 逐 benchmark 的准确率—计算曲线 | 九点 cutoff、共享轨迹、每题记录重建算法；不从图中臆造未报数字 |
| C.3；Fig.10，p.28 | 并行采样与共识投票 | cons@n 的抽样空间、期望 compute、完整回答池相关性 |
| C.4；Fig.11，p.28 | 三个正确回答的 token 深度分配 | 全部显示片段均值；明确例证选择条件 |
| C.5；Fig.12，p.29；Fig.13，p.30 | 跨轮 attention 分配与热图 | 100 条序列、三类 head、序列级 SD、prompt 键未重归一化 |
| §6；Appendix E，p.31–32 | 适用范围与相关工作定位 | 本页 SFT 范围、下一节六个区分轴；后训练课程承接扩展问题 |

<span id="source-block-eb30d9b4-b756-44a0-8c5e-dfb83ef022cc"></span>

### Appendix E 的定位维度

<span id="source-block-6f0dbe00-4dff-4605-ac0e-602fc9dab537"></span>

原文相关工作不是单一的“loop 模型名单”。它分别讨论可表达性、循环状态与读出、预训练模型的循环改造、计算自适应维度、深度监督和部署。用于课程选择基座时，先记录预训练身份、Dense／MoE 与拓扑，再补训练目标和执行语义；架构同名不保证这些字段相同。[Appendix E](https://arxiv.org/html/2609.35748v1#A5)

<span id="source-block-20bf1157-5524-4684-8a7e-592bedad2cd7"></span>

| 区分轴 | TaH2 在本文的具体选择 | 学习时需要保留的差别 |
| --- | --- | --- |
| 循环来自哪里 | 在普通 Qwen3 Base 上联合 SFT 改造 | 与原生循环预训练权重分别登记 |
| 状态如何进入下一轮 | 原 embedding 与末层状态经过学习 updater | 原状态直连、输入注入、中段 adapter 的导数与初始化不同 |
| 读出如何聚合 | 对已执行轮次的预测概率做停止权混合 | 混合 hidden、混合 logits、混合 probabilities 具有不同目标与梯度 |
| 计算在哪个维度变化 | token 级深度 | Dense／MoE 的专家宽度变化与循环深度分别计费 |
| 决策何时、由何信号得到 | 每轮决策；当前模型的单步预测收益标签 | 初始一次性分配、置信度、收敛规则、terminal reward 的估计对象不同 |
| 理论节省怎样进入系统 | 深度批处理引擎与实际 KV 可见性 | FLOPs、显存、延迟、吞吐分别测量，不能互相代替 |

<span id="source-block-3fce22ba-ea6a-40e1-9ec9-5198c235baeb"></span>

### 从原文对象到源码函数

<span id="source-block-5cc1df75-f6b3-44f5-8c52-60ad2e63d137"></span>

下面列出固定 commit 上可逐项定位的实现入口；更完整的 shape、dtype、mask、detach、缓存所有权和分布式顺序继续放在对应工程页。

<span id="source-block-6ff8e6a1-8f5c-4f1e-8f55-d4c5050d5226"></span>

| 数学或实验对象 | 固定源码入口 | 关键实现对象与课程 |
| --- | --- | --- |
| 数据再生成与工具轨迹 | [regenerate.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/data/regenerate.py)；[tool_rollout.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/data/tool_rollout.py) | 教师回答与已有工具结果的来源；[数据页](/notes/loop-reasoning/data/) |
| 模板与监督 token | [data_prepare.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py)；[data_collator.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/data_collator.py) | `_tokenize_messages`、`preprocess_pretokenized_batch`；raw 与预分词分支 |
| 训练入口与 ceiling 配方 | [SFT_TaH.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py)；[1.7B recipes](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_1.7) | `sft_tah.yaml`、`sft_tah_max4.yaml`、`sft_tah_max8.yaml`；配置与权重身份绑定 |
| 主前向、标签移位、位置与 mask | [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) | `TaHForCausalLM.forward`、`_shift_labels`、`_init_position_ids`、`_build_attention_mask_for_iter`；[循环页](/notes/loop-reasoning/internal-loop/) |
| Updater 与 decider | [input_updater.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py)；[iter_decider.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) | `Qwen3MLPUpdater.forward`、`Qwen3MLPIterDecider.forward`；原 embedding、top-k 与梯度 |
| 探测及混合读出 | [tah_runtime.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py) | `capture_posterior_inputs`、`_run_side_score_pass`、`accumulate_iter`、`finalize_logits` |
| 收益标签与联合 loss | [iter_label.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py)；[loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py) | `DynamicIterLabelGenerator`、`_coverage_threshold`、`IterDeciderLoss.final_loss_func` |
| KV 历史与探测快照 | [causal_cache.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py)；[sparse_ops.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py) | `TaHCache`、`snapshot_iter`、`get_cache_upto_iter`；主路径与辅助路径的所有权 |
| 优化与并行 | [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py)；[tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) | `train_model_tp`；[更新页](/notes/loop-reasoning/parameter-update/)与[系统页](/notes/loop-reasoning/systems-evaluation/) |
| 导出后的自主生成 | [engine.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py)；[服务入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/launch_minisgl_server.py) | 实际 backend 的 prefill、depth batching、decode 与停止；[生成页](/notes/loop-reasoning/generation/) |
| 请求计时、逐题记录与评分 | [eval_online.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py)；[matheval.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/evaluate/matheval.py)；[local_eval.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/evaluate/local_eval.py) | `_post_chat_completion`、`_online_runtime_metrics`、`run_online_eval` 与 grader |
| 发布数据、权重与复现说明 | [README.md](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/README.md)；[默认数据集配置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/evaluate/eval_configs/dataset_configs.json) | 发布统计保留自身口径；默认配置文件不等同于论文完整 benchmark manifest |

<span id="source-block-4bd12e3f-3557-4952-a3c4-4f636557cdb2"></span>

官方 Standard、max2、max4、max8 四个 1.7B 发布对象均固定到不可变 revision。完整版本、`tah_config`、骨干与两个侧模块的清单，以及 loader 合并配置的规则，见[加载与生成课的算法 R1](/notes/loop-reasoning/generation/)。这里固定的是下载对象和小文件内容；完整权重尚未下载、载入，也未确认与论文历史运行的对应关系。评测方面，默认 `dataset_configs.json` 的八项入口不足以恢复论文全部 benchmark manifest。数据方面，发布 revision 固定为 `e9088ae272a81e2fce720e585a005a1547964cfd`，17 份小文件与 provenance 的范围见[数据快照核对](/notes/loop-reasoning/data/)。完整复现仍缺少 Arrow 字节校验与有效监督重计、grader、评测 seed，以及模型数值和性能记录。

<span id="source-block-361a19bb-0f98-4e96-a0c8-1fb00718b1fe"></span>

### 已完成的数学核对与结果边界

<span id="source-block-9fe25aa0-4e2a-40f9-8045-ddad3436a323"></span>

已重新计算 1.7B updater／decider 参数、Eq.13 每次调用 FLOPs、53.1% 的斜率相对增幅、M8 权重在推理 8→12 时 22.5% 的平均深度增幅、训练成本倍率、release 三遍历 tokens 和表格宏平均。用 128 组确定性生成的小型深度轨迹，将逐键枚举与前缀深度账本逐 token 比较，计数和累计成本误差为零；另外核对了无放回子集期望成本与两个机制反例。

<span id="source-block-fbe8f138-9697-4994-bf0a-cf5f20fa9924"></span>

这些检查验证本页公式、统计口径和参考账本的内部一致性。真实模型训练、发布权重数值对齐、原文性能复现与 A800 延迟复测仍需要对应的权重、数据和运行环境；当前源码读取得到的计时边界也不足以反推论文历史运行使用的确切脚本。

<span id="source-block-25c3c4fa-efdf-430d-bf10-e1fb34795677"></span>

[返回 TaH2 与后训练工程](/notes/loop-reasoning/engineering/) · [继续状态与循环算法](/notes/loop-reasoning/internal-loop/) · [回到后训练课程](/notes/loop-reasoning/posttraining/)
