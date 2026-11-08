---
title: "TaH2 与后训练工程"
description: "这条学习路径把 TaH2 的论文结果、内部循环计算和后训练工程连起来。先从一份样本重建状态、监督与一次更新，再追踪导出、生成和成本；随后按研究问题接入蒸馏、偏好优化或 RL。十二个技术环节、八个主题页及既有论文式算法共同支撑这条路径；原生循环与普通基座适配分别保留各自的状态、深度、缓存和目标约定。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133"
sourcePageId: "page_90a0e38077148191b6c40007fbb15133"
sourceSha256: 44badd02b875e40a64a37f3846741e64d3adcd90af0c92fbe474c14344f649ae
series: loop-reasoning
references: []
---

<span id="source-block-7e638cc3-e8ec-4123-9d41-ba5af5c3b9fa"></span>

这条学习路径把 TaH2 的论文结果、内部循环计算和后训练工程连起来。先从一份样本重建状态、监督与一次更新，再追踪导出、生成和成本；随后按研究问题接入蒸馏、偏好优化或 RL。十二个技术环节、八个主题页及既有论文式算法共同支撑这条路径；原生循环与普通基座适配分别保留各自的状态、深度、缓存和目标约定。

<span id="source-block-f5ca4198-b367-4e99-bc13-2cb42445cbd2"></span>

[TaH2 论文结果与实现](/notes/loop-reasoning/tah2/) · [循环推理后训练](/notes/loop-reasoning/posttraining/) · [研究总览](/notes/loop-reasoning/overview/)

<span id="source-block-013284ce-e24d-4d36-9c6a-846458cb7ac3"></span>

> [!CALLOUT]
>
> **建议阅读顺序**
>
> 先读[TaH2 的模型身份、主结果与消融](/notes/loop-reasoning/tah2/)，再按[数据构建](/notes/loop-reasoning/data/) → [一次更新](/notes/loop-reasoning/parameter-update/) → [内部循环与探测](/notes/loop-reasoning/internal-loop/)重建同一批样本的计算，最后进入生成与性能测量。
>
> 每个问题依次阅读数学定义、算法步骤和源码执行。推导条件紧随公式，现有函数与待接入部分分别标明。

<span id="source-block-2ab58c84-354d-4a92-94b5-f3f9fe082150"></span>

## 按核心问题查找数学与实现

<span id="source-block-469b7494-d7f4-42b8-aa24-4122b8bbe6fc"></span>

[TaH2 论文结果与实现](/notes/loop-reasoning/tah2/)集中保留原文主结果、全部 12 张表、13 幅图及附录 A–E；下表把每项核心结果接回数学条件、真实执行与主题课。论文性能数字、固定源码行为、由此推导的性质和拟议接入算法分别标明。

<span id="source-block-10ac2207-13a6-4449-8a05-8b72c0230908"></span>

| 核心对象 | 数学与算法已经落实什么 | 工程与学习入口 |
| --- | --- | --- |
| 基座与对照身份 | 原生预训练与普通 Qwen3 后训练分开；同名 Ouro／Huginn 对照的目标及层调用数 | [论文身份与主结果](/notes/loop-reasoning/tah2/) · [七篇矩阵](/notes/loop-reasoning/model-matrix/) |
| Updater 与 decider | RMSNorm／SwiGLU 的完整 Jacobian；参数计数；top-k 概率值与全部 logits 的梯度 | [一次内部循环](/notes/loop-reasoning/internal-loop/)：shape、k=d、norm 默认值、离散索引及连续值 |
| DUO 主路径与探测 | 可见键集合；missing 触发；真实执行、候选和已存辅助输入集合；快照与并集合并 | [循环课 T1](/notes/loop-reasoning/internal-loop/)：pack、mask、detach、临时 KV 与重计算 |
| 深度标签与 BCE | 正收益覆盖阈值、等成本最小基数条件、并列及单类边界、加权 BCE 的最优校准点 | [循环课 T2](/notes/loop-reasoning/internal-loop/)：DP collective、候选级联、cost detach 与真实分母 |
| 停止混合与梯度 | 尾质量吸收；责任度；gate 中间变量偏导；floor 后总质量与 CE/NLL 差值 | [更新课 T3](/notes/loop-reasoning/parameter-update/) · [混合蒸馏梯度](/notes/loop-reasoning/offline-distillation/) |
| 训练、数据与版本 | 输入位置与监督位置分母、release 与论文统计、全局归约和配置展开 | [数据页](/notes/loop-reasoning/data/) · [一次更新](/notes/loop-reasoning/parameter-update/) · [框架](/notes/loop-reasoning/framework/) |
| 自主生成与后端 | prefill／decode、深度调度、继续与采样两条混合分支、资源强停及 EOS | [生成课 T4](/notes/loop-reasoning/generation/)：HF／MiniSGL even_mix 差异、温度与实际服务能力 |
| 主结果、消融与机制 | 所有模型行和任务分数；训练／推理 ceiling；斜率与曲线位置；单因素效应和缺失的交互证据 | [论文结果专页](/notes/loop-reasoning/tah2/)：统计单位、反例与论文结论边界 |
| 计算与性能 | DUO 历史成本；论文语义调用、当前物理执行和 wall time 三种口径；逐题配对与相关性 | [论文轨迹账本 T8](/notes/loop-reasoning/tah2/) · [系统与测量](/notes/loop-reasoning/systems-evaluation/) |
| 离线与在线蒸馏 | 责任权、CODI 真实对齐与移动目标、状态访问梯度、top-k 尾部桶信息及保护分支 | [离线蒸馏](/notes/loop-reasoning/offline-distillation/) · [在线蒸馏 T5](/notes/loop-reasoning/online-distillation/) |
| RL 与偏好 | 联合／文字边缘策略、随机 prefill 与深度历史、posterior score、importance ratio 和 KL 分解 | [强化学习 T6](/notes/loop-reasoning/reinforcement-learning/)：有限路由树参考，不把理论算法写成现有集成 |
| 分布式与恢复 | TP 伴随规则、阈值 margin、版本字段与完整优化状态、两轮同步恢复条件 | [系统课 T7](/notes/loop-reasoning/systems-evaluation/) · [可部署选择器与 oracle](/notes/loop-reasoning/research-ideas/) |
| 局部与联合收益 | 单轮到混合的精确恒等式、排序反转与覆盖界；singleton查询和联合KV交互 | [循环课T10](/notes/loop-reasoning/internal-loop/) · [诊断接口](/notes/loop-reasoning/framework/) |
| 采样变换与支持 | 固定 FlashInfer 0.6.4 的 top-k-first 顺序；温度 log 差振幅、独占支持质量与条件 TV 界 | [生成课 T11](/notes/loop-reasoning/generation/) · [processed score 与支持外目标质量](/notes/loop-reasoning/reinforcement-learning/) |

<span id="source-block-0be43889-5a11-48cd-8af5-b74579d29a40"></span>

### 需要连在一起理解的三组边界

<span id="source-block-2b78e811-238b-41eb-a0f8-c16b2fa31366"></span>

**训练信号与部署决策。** 可以从同一条训练样本问起：在这个前缀上，再算一轮能否更好地预测真实的下一个词？训练时目标词已知，用本轮对它的 NLL 减去下一轮 NLL，就得到 teacher-forced 的单步收益。这是在当前前缀上评价额外计算的信号。

<span id="source-block-53c52657-8475-4b0f-8205-9a98f52ffd97"></span>

同一条样本的最终预测还会混合多个已执行轮次。一次额外计算得到新的分布后，较浅轮次仍可按停止权重参与最终预测。混合似然的责任度描述各轮对这次目标词预测的相对贡献；它解释混合目标如何分配梯度，与“再算一轮改善多少”是两个不同的问题。

<span id="source-block-320e287c-63cd-43d8-9c33-a6698b203402"></span>

到了自主生成，真实的下一个词尚不可见，继续策略只能使用当时可见的状态和预测分布。我们让它接受训练信号，是希望学会提前作出有用的深度决定。完整任务回报则要等一段回答生成后才能评价，关心的是最终任务及其成本。研究监督或 RL 接入时，需要说清自己正在优化哪一个对象、决策能看到什么，以及梯度怎样到达决策器；局部 NLL 改善本身还不能给出最优停止策略。

<span id="source-block-f5c932ac-112c-422a-a136-7a9e2bbc6aef"></span>

**论文估算与实际执行。** 论文 Eq.23 定义了训练 FLOPs 的统计方式。当前 runtime 则先检查整个 DP 范围是否仍缺少所需 CE，再决定是否执行 side pass；深层缓存的辅助位置集合可能大于真正继续计算的位置集合。MiniSGL 融合图还可能计算首轮随后丢弃的 updater 结果和末轮 gate 结果。比较成本时，先按论文公式理解原表，再按调用 mask、实际算子与 profiler 记录实现成本；平均深度不足以恢复这些计数。

<span id="source-block-57213a73-a6f1-440a-a69e-4ffcb8577ad7"></span>

**条件概率与实际策略。** 固定路由上的小数值差不保证阈值路由一致，随机路线的一次条件 logp 也不是完整文字边缘概率。学习端、生成端和恢复后的版本必须同时对应权重、路由、读出、mask、温度、截断和缓存状态。

<span id="source-block-3e757dc4-5873-46cb-a264-008d4c1c21b8"></span>

### 论文式算法索引

<span id="source-block-8975f763-49c9-4fdc-8d8d-0df6f8b8a088"></span>

索引包含原有九段算法、T1–T12 和 D1–D3。T9–T12 分别处理随机读出、续算收益、实际采样和验证分母；D1（21 行）、D2（23 行）、D3（16 行）分别处理文件权重与打包、token 预算结转与数据恢复、工具 episode 展开与监督。每段算法的输入输出、成立条件、对应源码和替身检查范围，都在所在课页说明。

<span id="source-block-2599e14c-ae3e-4677-9f1b-03b524d48597"></span>

| 编号 | 执行问题 | 性质 |
| --- | --- | --- |
| T1 | DUO 辅助评分与缓存隔离 | 固定源码控制流 |
| T2 | 多深度标签与加权监督 | 固定源码标签与损失 |
| T3 | 读出质量与损失路径核对 | 小型数值参考；区别三条损失分支 |
| T4 | DUO 生成调度与终止 | MiniSGL 固定执行分支 |
| T5 | 选定 token 的混合分布蒸馏 | 拟接入参考；尚非既有 TaH2–verl 功能 |
| T6 | 小型路由树的精确文字评分 | 有限离散定义的精确参考；最坏指数成本 |
| T7 | 两轮同步与恢复状态核验 | 后续集成的验证协议 |
| T8 | 从逐题轨迹重建计算曲线 | 论文成本口径的统计算法；bootstrap 为分析扩展 |
| T9 | 单 token 随机路由的读出与成本 | 固定入态的精确代数参考；23 行；不替代全序列 T6 |
| T10 | 固定前缀的续算收益诊断 | 22行分析扩展；区分单轮、局部混合与整组收益，显式保存混合前质量和独立缓存 |
| T11 | 采样变换与支持审计 | 21 行离线参考；显式温度、top-k 归一化和 nucleus 支持；返回行为 logp、TV 与条件界；不替代 GPU kernel |
| T12 | 验证损失的执行记录核对 | 22 行参考核对；附八组可运行检查；区分原始分子、预统计分母、实际移位目标、重复出现和四位显示值；不替代模型／GPU验收 |
| D1 | 文件倍率、打包和来源账本 | 21行；Huginn原文96项与当前113项分开；文件抽样矩、尾部与4097→4096对齐 |
| D2 | 来源token预算结转与数据恢复 | 23行；Done Right当前loader原控制流；累计误差界与active状态恢复，历史loader差异保留 |
| D3 | 工具episode展开及监督 | 16行；TaH2完整历史输入、当前episode标签、offset边界和长度过滤 |

<span id="source-block-5b6d1a96-1476-4707-9310-c65f80e4aab6"></span>

## 学习路径与导航

<span id="source-block-4666d79e-00cf-48bd-bd00-e6c67674c7b3"></span>

选择一个主题查看计算顺序，再打开课页。先看数学与算法主线，需要实现时展开源码索引和验收清单。

<span id="source-block-6630fa70-085f-4ff5-a1b3-204200ab2364"></span>

<figure class="research-embed"><iframe src="/research-assets/8c36c3c3980d9ee55e0868df2c720dc91dbc3462dd03f777db7db0d6e858f9f3.html" title="后训练全栈学习导航" loading="lazy" sandbox=""></iframe><figcaption>后训练全栈学习导航 · <a href="/research-assets/8c36c3c3980d9ee55e0868df2c720dc91dbc3462dd03f777db7db0d6e858f9f3.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-eb4e8572-bdd3-4ee8-9f4c-04473c86aa08"></span>

## 十二项技术路线

<span id="source-block-bb9c9fa1-8691-49eb-a31a-a79b3a05cae5"></span>

| 环节 | 学习页与源码入口 | 必须解决的问题 | 后续验收产物 |
| --- | --- | --- | --- |
| 01 环境与版本 | [数据构建与版本固定](/notes/loop-reasoning/data/) · [依赖](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/pyproject.toml) | 模型、tokenizer、模板、数据和代码怎样固定到可重建版本 | manifest |
| 02 数据与配比 | [数据构建与版本固定](/notes/loop-reasoning/data/) · [preprocess](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) | 角色监督、截断、去重、问题留出；样本比例与 token 比例 | batch trace 与 data audit |
| 03 SFT 与更新 | [一次参数更新](/notes/loop-reasoning/parameter-update/) · [train_model_tp](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) | 损失分母、累积、参数组、混合精度与实际梯度 | update trace |
| 04 循环接口 | [一次内部循环](/notes/loop-reasoning/internal-loop/) · [Qwen3MLPUpdater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py) | 原始 embedding 与跨轮状态怎样接回骨干 | loop trace |
| 05 位置与缓存 | [一次内部循环](/notes/loop-reasoning/internal-loop/) · [TaHCache](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py) | DUO 可见性、RoPE、探测及重计算的副作用 | visibility 与 cache report |
| 06 深度监督 | [一次内部循环](/notes/loop-reasoning/internal-loop/) · [DynamicIterLabelGenerator](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py) | 收益覆盖 cutoff、决策 BCE 与停止概率混合 | mixture 与 grad report |
| 07 自主生成 | [从训练到自主生成](/notes/loop-reasoning/generation/) · [MiniSGL engine](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py) | 导出、prefill、decode、采样与两端概率对应 | export 与 backend parity |
| 08 离线与过程蒸馏 | [离线蒸馏与过程监督](/notes/loop-reasoning/offline-distillation/) · [CODI.forward](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py) | 教师输入、对齐位置、显式任务和学生梯度 | teacher target trace |
| 09 在线蒸馏 | [接入在线蒸馏](/notes/loop-reasoning/online-distillation/) · [distillation_loss](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py) | 学生前缀上的 KD／PG、top-k 支持、两次更新与同步 | two update timeline |
| 10 RL 后训练与偏好 | [强化学习与偏好优化](/notes/loop-reasoning/reinforcement-learning/) · [v1 trainer](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py) | 策略梯度、价值与奖励、在线数据分布，以及 rollout／更新／同步闭环 | 角色与轨迹 trace、目标重算与训练诊断 |
| 11 分布式与性能 | [分布式执行与性能评测](/notes/loop-reasoning/systems-evaluation/) · [TP 与 FSDP2](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) | 分片归约、控制流对齐、checkpoint、峰值显存与阶段耗时 | checkpoint 与 cost profile |
| 12 评测与机制 | [分布式执行与性能评测](/notes/loop-reasoning/systems-evaluation/) · [run_online_eval](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/eval/eval_online.py) | 能力与成本曲线、统计单位、状态干预与中间结果续用 | 逐题 records 与比较报告 |

<span id="source-block-2754aeed-d9f1-4904-b071-6cf2d1ae038f"></span>

## 框架与研究

<span id="source-block-a785697e-439b-440d-8f60-3b9d04b176fc"></span>

- [框架结构与接口](/notes/loop-reasoning/framework/)：完整调用链、六个模块、状态约定与实现检查。

<span id="source-block-a02e4745-4807-40cc-8156-838612e65ddf"></span>

- [研究设想与框架需求](/notes/loop-reasoning/research-ideas/)：保留自蒸馏、接口、门控、少量文字和动力学等研究问题。

<span id="source-block-28ad99dd-f6fc-4525-8600-3c5bea4b55de"></span>

- [Full CoT 基础讲解](/notes/loop-reasoning/full-cot/)：需要时补充标准监督与梯度。

<span id="source-block-ef192d34-e9bf-4824-95c4-2d5ede427de8"></span>

## 数学与算法的组织方式

<span id="source-block-e902606e-ccb2-4c96-b7c0-72e6abe0e7e0"></span>

每个核心环节按下面的顺序讲解。符号服务于当前计算问题，原始期望目标、梯度估计、代理损失与优化过程分别定义；条件、近似和工程选择放在相关公式旁。

<span id="source-block-359fc98b-7253-4116-b206-3ed64fc1a186"></span>

| 层次 | 必须给出的内容 |
| --- | --- |
| 数学对象 | 随机变量、条件分布、状态、参数与张量对应，符号的定义域和值域 |
| 目标与推导 | 优化目标、估计器、假设与等式成立条件；必要的证明或反例 |
| 参考算法 | 采用论文式算法，单列输入输出，以行号、缩进和控制词说明初始化、循环、分支、停止及状态变更；操作中的数学表达式须真实排版 |
| 工程实现 | shape、dtype、索引、mask、detach、缓存所有权、归约与同步顺序 |
| 对照验证 | 数学不变量、源码函数与可检查的数值例子；说明参考算法省略的优化 |

<span id="source-block-3bae7c36-ecd0-4a2c-897a-258fe344b7c9"></span>

这一组织方式已扩展到八个主题页：数据页定义样本与 token 配比；更新页推导全局分母与 DP 梯度；蒸馏页区分分布目标、特征对齐和状态访问梯度；生成页给出缓存等价条件；系统页定义内存、FLOPs 与测量口径。

<span id="source-block-553cc697-f967-42f0-befd-b518bef8ba15"></span>

[RL 数学模型与同步更新](/notes/loop-reasoning/reinforcement-learning/)和 [TaH2 状态与前向算法](/notes/loop-reasoning/internal-loop/)仍作为两份完整示范。各页均将数学参考、固定源码已有行为与拟补诊断入口分开标明。

<span id="source-block-f48c75e6-d359-4dda-872a-5d6e1875e182"></span>

## 怎样推进

<span id="source-block-b81b146b-963c-4164-8c27-e9bfda9f56d6"></span>

把学习和实现合并为一个逐步扩展的案例：同一份样本清单、同一套状态记录、同一条训练到生成路径。保留模型和配方选择；先把参考计算做清楚，再判断方法收益。

<span id="source-block-bc058401-8736-45f7-8e72-728ed93811a3"></span>

| 阶段 | 覆盖环节 | 形成的能力与结果 |
| --- | --- | --- |
| 先读清计算 | 01 至 06 | 从一条样本重建监督、损失、两轮状态、缓存可见性与梯度路径 |
| 再闭合执行 | 07、11、12 | 完成参考前向与更新、导出加载、prefill／decode 对照和成本记录 |
| 按问题接教师或奖励 | 08 至 10 | 选择离线过程监督、OPD 或 RLVR／DPO；在线分支覆盖连续两次更新与同步 |
| 最后形成研究比较 | 02、04 至 06、12 | 在明确预算下研究接口、门控、状态重用或数据配比，保留竞争解释和逐题证据 |

<span id="source-block-0a1a8ced-a432-4cc5-b87b-ec0fc505f59a"></span>

一个完整案例应产出数据与版本清单、一次更新和循环的 trace、训练／生成数值对照、成本与逐题评测记录。RL 的基础、在线数据分布与工程闭环纳入完整学习范围；实际采用 RLVR、DPO 或教师反馈，以及是否部署异步服务和更大规模并行，继续按研究问题与资源选择。

<span id="source-block-c9fc7ef2-0164-4233-a6f6-6b49de96ec92"></span>

> [!CALLOUT]
>
> **内容与检查范围。** TaH2 的全部表图、附录、函数索引及既有算法均保留，T12 主体、验证分母、参数组和调度已有讲解，TPP 的含义已修正。Huginn 评测版本、Ouro Base／Thinking 结果、Done Right MoE 图表与实际生成路径也已保存，并同步到三门课程。
>
> 已完成的检查包括八组 CPU 替身检查、调度函数检查、固定源码核对和保存检查。模型训练、TaH2–verl 集成、GPU 后端一致性、性能与浏览器实际渲染，仍需按原清单验证。

<span id="source-block-5b76ad65-d917-4687-8be6-6b921d367d75"></span>

### 数据配方与训练序列的核对记录

<span id="source-block-de319fc2-db51-4bfb-be60-6f37e2d6f6e2"></span>

本轮继续核对七篇材料的数据条件和实现链，实质补充Huginn、Ouro、Done Right与TaH2的原始配方及源码，并接回预训练、后训练和框架接口。Loopie现有阶段预算、SMELT的配对数据与未公开范围、LoopMoE的共同子集和packing条件继续保留。本轮更新9页；没有把当前代码清单或替身检查当作历史训练复现。

<span id="source-block-a90ffc1f-5fe9-4d0e-a97b-6d5314dbc779"></span>

| 核对项 | 本轮落实与证据 |
| --- | --- |
| 原始实验与评测 | Done Right Q1–Q3 全分项及 MoE Fig.13 九任务／七任务／四任务口径；Huginn Fig.14 与 v2／v3／非论文入口；Ouro Base／Thinking 协议与 Tables12–13 深度结果 |
| 训练输入 | xLLM packing、单次 shift 与 BOS 分段保留；TaH2 补发布行数→预处理→drop／pad出现记录→实际batch执行→移位目标的计数链 |
| 候选运行 | Ouro 固定 cache／Transformers v4.55.0 的构造与 mask-size 契约冲突；版本结论限定，未宣称其他组合均失败 |
| rollout 接口 | Loopie 实际 HTTP 字段、服务端默认、响应长度回退与 raw／processed 概率；不将通用路径当作论文专用 launcher |
| TaH2 数学 | 共同支持上的温度界、支持改变质量项及 top-p 不连续反例；processed score 的局部梯度与行为支持外目标质量 |
| 数学与算法增量 | 已有15段原生数学块与21行T11保留；本次新增3段原生数学块和22行T12；沿用正文18px、公式约22px、注释15px |
| 小型核验 | 根任务 512 组支持界与 512 组条件化界；独立审阅另核 5,000 组；最大不等式数值违约约 3.33×10⁻¹⁶，属于浮点舍入 |
| 前轮源码与保存 | 已有TaH→FlashInfer顺序审计保留；T12上传后读回源码逐字一致；本次11页原有块ID与图示／算法引用均未删除，未编辑区域hash一致 |
| 证据范围 | 原文/PDF、固定源码、CPU 小例与保存已核；未下载模型权重、启动训练或 CUDA 服务；真实后端等价、性能和浏览器实渲染仍待验收 |
| 实际更新与调度 | 参数前缀分组、运行时 norm 排除、真实组 LR；完整 epoch horizon 与提前停止分开；更新使用 LR 与推进后日志 LR 分开 |
| TPP 纠正 | 配方表改为训练 token 预算／参数量；与 TP、实际 DP 及 data.dp 分开；原论文和 release token 数保留各自口径 |
| 验证目标与 T12 | 在实数算术下 L_raw=(V_E/U_D)L_E；22行主体和12处MathML已保存；原始值、FP32累加与四位显示分开 |
| 前轮验证检查 | 八组 eval 边界和一个调度函数检查通过；12份源码快照核对Git blob SHA与SHA-256；源文件、脚本、JSON和算法附于检查包 |
| Huginn数据 | Fig.4全部11类比例；Tables7–8完整96项；当前113项的17项差集和1处地址变化；文件倍率、实际保留量与D1 |
| Ouro数据 | Table3全部13来源的池规模与Used；Stage1／2完整比例；Stage4新监督与回放的30%／10%／60%，各统计口径分开 |
| Done Right数据 | 当前42项权重合计1；token结余额、望远镜恒等式和有限长度条件下的累计误差界；D2与真实恢复字段 |
| TaH2工具数据 | 原tool_rows只监督当前episode；历史前缀重复处理的输入量公式、offset跨界拒绝及filter／cutoff分支；D3 |
| 本轮数学与算法 | 新增6段原生数学块；D1／D2／D3共60行、7处MathML；沿用正文18px、公式22px和注释15px |
| 本轮检查与工件 | D1–D8八组数据检查通过；14份固定源码核对Git blob SHA和SHA-256；96／113来源映射、42权重、源码与JSON结果已附数据页 |

<span id="source-block-78d6b2ca-a7f7-40f6-b7f6-21c2cab01c1a"></span>

[数据页与本轮检查包](/notes/loop-reasoning/data/)提供完整源清单和D1–D8；[预训练课](/notes/loop-reasoning/pretraining/)给出七篇材料的数据条件对照，[框架接口](/notes/loop-reasoning/framework/)新增SourceRecipe／DataResumeState并显式区分labels对齐。前轮[T12](/notes/loop-reasoning/systems-evaluation/#source-block-b445b124-6da5-430c-896a-5b1c0faf4646)、[一次更新](/notes/loop-reasoning/parameter-update/)、[生成T11](/notes/loop-reasoning/generation/)及[RL](/notes/loop-reasoning/reinforcement-learning/)的条件、反例、算法与验证范围继续保留。

<span id="source-block-642b8b90-97ea-4dad-b2d2-022ce95298c3"></span>

### 前次继续核对记录：监督目标与实现契约

<span id="source-block-f8d9a701-06c3-490f-8325-995d026efaeb"></span>

本轮继续沿原文和固定实现收紧结论条件。TaH2重点推进[单轮／混合／联合收益与T10](/notes/loop-reasoning/internal-loop/)、[对应论文结果的解释](/notes/loop-reasoning/tah2/)和[框架记录契约](/notes/loop-reasoning/framework/)；[RL工程](/notes/loop-reasoning/reinforcement-learning/)同时补入Loopie公开GRPO路径到论文GSPO／动态补样的具体迁移条件。

<span id="source-block-9a4236a8-8b7c-4649-b10b-04bd2a673820"></span>

| 核对项 | 本轮落实与验证 |
| --- | --- |
| 页面范围 | 七篇专页、三课程及相关工程／索引；全24页回读，实质更新16页 |
| 原始结果 | LoopMoE全部9B任务与27行BBH；SMELT图9逐规模／类别与ICL筛选；Ouro原文mask条件补全 |
| 理论与算法 | 新增27段原生数学块；T10为22行；精确条件、反例、概率对象与干预范围就近说明 |
| TaH2数学核对 | 256组混合恒等式，最大误差4.44×10⁻¹⁶；64个因果小图、162个singleton查询与单位置重跑一致；含完整较浅KV的交互反例 |
| 固定源码核对 | Huginn参数组AST／最小替身；Ouro依赖loss的NumPy小张量；Done裁剪与日志分母；Loopie真实过滤guard及token／序列梯度 |
| 内容保留 | 原有block ID与相对顺序、全部38个原算法／图示引用保留；T10仅保留当前审定版本的单一引用 |
| 检查范围 | 表内原值、数学与源码语义、保存读回已核；没有权重载入、模型训练或GPU计时；浏览器实渲染仍待验收 |

<span id="source-block-534e408d-938c-4057-9f88-58874cf5dd8d"></span>

独立审阅进一步修正了教学例子的候选集可达性，并把T10收紧到明确的目标概率记录、混合累加前快照、KV截深和hard活动集合干预。每项结论区分原文观察、固定代码、数学推导和拟议诊断；Huginn的契约模拟不冒称历史训练复现。前轮记录继续保留如下。

<span id="source-block-d33eebb0-ed8d-4b5e-a8f2-275483f94157"></span>

### 前次继续核对记录：T9与有序放大探针

<span id="source-block-6efacaca-f03c-4750-94a4-125b14676752"></span>

本次没有替换原论文的性能结论，而是把用于后续研究的执行条件进一步落实。TaH2 的主要新增内容集中在[随机深度与 T9](/notes/loop-reasoning/internal-loop/)、[资源强停与当前／序列分布](/notes/loop-reasoning/generation/)、[联合与边缘策略梯度](/notes/loop-reasoning/reinforcement-learning/)、[随机教师蒸馏](/notes/loop-reasoning/online-distillation/)和[服务负载](/notes/loop-reasoning/systems-evaluation/)。

<span id="source-block-a98078e1-fe2c-473d-be5c-50f2853286a2"></span>

| 核对项 | 本次落实与验证 |
| --- | --- |
| 页面范围 | 回读全部 24 页；更新 19 页，包含六模型、TaH2 论文、三课程、工作台及相关工程主题 |
| 理论增量 | 六模型各补 3 组，共 18 组；全套新增 49 段原生数学块，写明局部／全局、条件／边缘、精确／数值的成立条件 |
| 算法增量 | T9 为 23 行，SMELT 有序局部放大探针为 25 行；输入输出、终止、退化处理与来源分别列明 |
| 原有内容 | 原有 block ID 与顺序保留，36 项原图示／算法引用保留；两项新增引用各出现一次 |
| TaH2 概率与成本 | 256 组有限分布案例：停止枚举与封闭式最大差 2.23×10⁻¹⁶；归一化 3.34×10⁻¹⁶；真实访问成本 3.56×10⁻¹⁵；gate NLL 中心差分误差 2.97×10⁻¹² |
| 随机教师目标 | 128 组 forward-KL 分解，最大恒等式误差 4.45×10⁻¹⁶，学生 logit 梯度误差 1.12×10⁻¹⁶；学生侧 Jensen 条件另行标明 |
| 六模型核对 | 非保守场与锁步方差；采样与缓存反例；相反 gate 梯度；初始化／AdamW；两种归约；有序 Jacobian、RMSNorm、500 组 attention 界；104 个 Jaccard 均值的锐界 |
| 算法保存 | 两段新增算法逐字回读一致；行号、54 处数学片段、字形引用与外部资源检查通过；正文 18px、公式约 22px、注释 15px |
| 验证范围 | 数学、源码控制流、形状与保存核对已完成；未加载模型、未运行训练或 GPU 服务压力／性能实验；浏览器实渲染仍待验收 |

<span id="source-block-6faddeb2-0c51-4693-90d8-79603701c3eb"></span>

理论与源码核对已进入收尾。TaH2 的 Standard／max2／max4／max8 发布 revision、23 份小文件与骨干／侧组件清单已落实；算法 R1 说明完整配置继承、深度覆盖及载入报告。未公开的官方实现、模型级数值等价、真实任务收益和部署成本仍各自登记；[生成课](/notes/loop-reasoning/generation/)保存本轮可重跑检查。

<span id="source-block-0b6a5c6b-c269-4be0-b3d2-e0afcd4eb1fb"></span>

### 前一轮核对记录

<span id="source-block-e7414f05-d864-414c-ae5c-3ba6e39f2797"></span>

核对同时保留原文证据、理论条件和执行语义。以下计数对应本轮保存后的版本；公式检查与模型实验分别记录。

<span id="source-block-a725d591-f0d4-44bf-9f93-667c2a101e2a"></span>

| 核对项 | 已完成的结果 |
| --- | --- |
| 页面范围 | 更新原有 23 页，新增 TaH2 论文结果页，共 24 页相互连接 |
| 原文覆盖 | 六篇原有 83 项核对保留；TaH2 新增 23 项章节／图表证据索引，覆盖 Tables 1–12、Figures 1–13 及 Appendix A–E |
| 数学与算法 | 本轮新增 76 段原生数学块；新增 T1–T8 共 164 行；每段列明输入输出、前提、来源与实现状态 |
| 内容保留 | 原有页面块均在；原有算法和图示引用均保留；修正过宽表述时同时写明条件与源码 |
| 保存回读 | 八段新增算法的保存源码和逐行内容与审定稿一致，每段仅嵌入一次；新增跨页链接均指向已核页面 |
| 数学核对 | 参数和 FLOPs 重算；128 组 DUO 轨迹逐键计数与前缀账本完全一致；updater、top-k、混合／BCE／KD 梯度的教学有限差分误差均在所示小例约 3×10⁻¹⁰ 内 |
| 后续运行范围 | 模型训练、GPU 后端对照、TaH2–verl 完整接入、性能复测与浏览器实渲染另行验收 |

<span id="source-block-7b7b4f20-0496-43f3-8670-e4bfc7a5dc47"></span>

```text
请按本工作台的十二项路线推进教学。先读取对应主题页的数学定义和参考算法，每次围绕一个真实计算边界展开：明确目标、假设与推导，再对应源码、张量和可验证记录。算法采用论文式排版：编号标题、输入输出、逐行编号、循环与分支缩进；不使用伪代码文本框。算法正文以 18px、主公式约 22px、注释 15px 呈现，长表达式换行而不整体缩小。操作公式须真实渲染上下标、分式、集合和赋值箭头，避免把 LaTeX 源码或转义文本直接放进普通表格。重点解释索引、mask、detach、缓存、梯度归约与同步。区分原始目标、估计器、代理损失、数值护栏和工程实现；保持符号一致，发现源码与参考公式有差异时就地说明。教学数值核对与真实模型运行结果分别报告。
```

<span id="source-block-8f51ed0a-7e0b-4de3-bf73-a90e61b259b0"></span>

研究扩展沿现有接口推进：数据采样器连接配比，循环状态与 KV 连接记忆和续算，教师接口连接过程教学，门控连接按需计算。模型、数据与配方继续按研究问题选择，相关设想保留在研究页。

<span id="source-block-c0943308-0782-4827-8a3c-5af72f6c0c3a"></span>

## 源码版本

<span id="source-block-c9b00d50-5e2a-4bb8-9c8d-1e4ea435ef7d"></span>

[TaH 78ddf9a](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c/) · [verl 7cea539](https://github.com/verl-project/verl/tree/7cea5392165ac26d5eadf879701337c85e4a2b2f/) · [CODI 2c23146](https://github.com/zhenyi4/codi/tree/2c2314662c63e9f482ebc46614ffe9af17a241e5/) · [Marin f5e8a9c](https://github.com/marin-community/marin/tree/f5e8a9cfe9d9d50b79fd03e4ed0ed0087edf627d/) · [TRL 1acd8ad](https://github.com/huggingface/trl/tree/1acd8adceffa285dcabd1f54b08e0830bea0279a/)。各课定位具体文件和函数，PyTorch 2.9 文档用于解释固定 TaH2 依赖附近的底层机制。源码阅读版本已固定，组合环境的兼容性在集成时验收。 本轮采样依赖单列：[FlashInfer 0.6.4 / f1e6fdc](https://github.com/flashinfer-ai/flashinfer/tree/f1e6fdcb8f65104047697f022b5d055ef022d763)，核对其 Python 分派、内核及上游测试中的并列／舍入条件。

<span id="source-block-0b5766cd-cc9d-4413-b8d9-548c0602f74f"></span>

[循环推理研究总览](/notes/loop-reasoning/overview/) · [返回后训练主线](/notes/loop-reasoning/posttraining/)
