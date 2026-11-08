循环学生需要学会利用教师信息继续计算。本课先区分三类信号：教师文字轨迹、输出概率分布和内部特征；再分别说明它们监督什么、梯度流向哪里，以及如何检验学生是否形成可复用状态。

[返回后训练研究工作台](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
>
> **阅读顺序** 先用三类信号表确定监督对象，再推导固定前缀上的 KL 与混合梯度；随后定义特征对齐，最后用 CODI 的源码检验位置、尺度和教师梯度。算法 5 连接分布目标与一次更新。

# 三类教师信号

| 方式 | 教师与学生看到什么 | 学生获得的训练信号 |
| --- | --- | --- |
| 轨迹 SFT | 学生以教师文字过程为真实前缀 | 下一 token 的硬标签，沿显式轨迹学习 |
| 分布 KD | 在指定的相同前缀上查询教师与学生 | 教师分布或选定词表支持上的概率目标 |
| 特征或过程监督 | 教师可能看完整 CoT，学生走自身循环路径 | 明确位置、层与语义角色之间的对齐或续算目标 |

前两种方式首先教输出行为；特征监督增加内部约束。是否形成可复用中间结果，需要后续操作的证据。教师已看见答案之后的状态若被用于学生答案之前的目标，应明确其训练信息范围和潜在捷径。

# 分布蒸馏的概率与梯度

先看一个已经固定的预测位置：学生读到前缀 `s=(x,y_<t)`，下一步该怎样分配词表上的概率？硬标签只指定一个词，分布蒸馏则把教师对其他候选词的判断也交给学生。为使两边比较的是同一件事，前缀、词表 ID 和有效预测位置都要对齐。令教师 logits 为 `zᵀ`，学生 logits 为 `zˢθ`；两边先按温度 τ 得到概率，教师概率在本次损失中停止梯度。下面的 KL 便是在这个共同前缀上，衡量学生还差多少才能匹配教师分布。

一条样本会贡献多个这样的预测位置，batch 中也会有 padding 或不参与监督的位置。因此，先用 m 选出有效位置，再把这些位置的 KL 相加，最后除以有效位置数 N。这样，每个计分位置对单进程参考目标具有相同权重。多卡执行仍需按“一次参数更新”中的全局分母与梯度归约方式缩放，不能把各卡局部平均直接当成同一个全局平均。

```math
\begin{aligned}
q^\tau(\cdot\mid s)&=\operatorname{softmax}(z^T(s)/\tau),\\
p_\theta^\tau(\cdot\mid s)&=\operatorname{softmax}(z^S_\theta(s)/\tau),
\qquad \tau>0.
\end{aligned}
```

```math
\begin{aligned}
L_{\rm KD}&=\frac{\tau^2}{N}\sum_{b,t}m_{bt}
D_{\rm KL}\!\left(\operatorname{sg}(q^\tau_{bt})\Vert p^\tau_{\theta,bt}\right),\\
N&=\sum_{b,t}m_{bt}>0.
\end{aligned}
```

## 温度与学生梯度

为了看清学生究竟收到什么信号，把 KL 展开为教师熵与教师对学生的交叉熵。教师和前缀固定后，教师熵不随学生参数变化，更新学生时只剩交叉熵这一部分。继续对一个位置的学生 logits 求导，就得到下式的“学生概率减教师概率”：教师认为更可能、而学生给得不足的词，其 logit 会沿下降损失的方向被提高；给得过多的词则相反。这里先不乘位置 mask，也不除 batch 分母，以便看清单个位置的局部更新。

```math
\nabla_{z^S}\!\left[\tau^2D_{\rm KL}(q^\tau\Vert p^\tau)\right]
=\tau(p^\tau-q^\tau).
```

式中的 `τ²` 用来调节梯度尺度；若不乘它，梯度就是 `(pτ−qτ)/τ`。因此，改变温度既会改变教师希望学生学习的分布形状，也会影响更新尺度，配方应写清温度施加在哪一侧。实现时还需要足够的教师分布信息：只有温度 1 下的 top-k 概率，一般无法重建另一个温度下、经全词表归一化的概率。

## 混合分布中的温度位置

循环模型还须声明温度施加的位置。若最终分布由各深度混合得到，下式两种操作一般不相等；上面的单一 softmax logits 梯度式不能直接套到各深度 logits 或 decider 参数，须按完整计算图继续求导。

```math
\begin{aligned}
p&=\sum_d\alpha_dp_d,\\
T_\tau[p]_v&=\frac{p_v^{1/\tau}}{\sum_up_u^{1/\tau}},\\
T_\tau\!\left[\sum_d\alpha_dp_d\right]
&\ne\sum_d\alpha_dT_\tau[p_d]\quad\text{一般情形}.
\end{aligned}
```

## 混合读出如何向各轮与门控传递梯度

循环学生让这个问题多了一层：同一个前缀经过 D 轮计算，会得到 D 个词表分布，最后被判分的却是它们的混合。我们需要把混合输出的误差继续分到各轮和门控上。本节用 q_d 表示第 d 轮学生分布，用 α 表示混合权重，用 t 表示固定教师分布。先把各轮 logits 与 gate logits 当作独立中间变量，求出局部偏导，再合并它们对共享参数的依赖。下面分析温度为 1 的归一化混合；教师支持上的学生概率须为正，硬阈值执行分支不变，权重下界保护未触发。

```math
q_d(v)=\operatorname{softmax}(z_d)_v,\qquad
p(v)=\sum_{d=1}^{D}\alpha_dq_d(v),\qquad
r_d(v)=\frac{\alpha_dq_d(v)}{p(v)}.
```

```math
\mathcal L=-\sum_v t(v)\log p(v),\qquad
\bar r_d=\sum_v t(v)r_d(v),\qquad
\frac{\partial\mathcal L}{\partial z_{d,u}}
=\bar r_dq_d(u)-t(u)r_d(u).
```

先用金标准词 y 的 NLL 来理解责任权。第 d 轮给最终正确词概率贡献了 α_dq_d(y)，除以混合后的 p(y)，就是这轮对 y 的责任权 r_d(y)。于是，各轮收到的局部 CE 梯度按这份贡献缩放，而不是平均分配。分布蒸馏没有唯一的正确词：要对教师分布中的每个词分别计算上述责任权，再按教师概率 t(v) 加权，得到 r̄_d。前式中的 r̄_dq_d(u) 来自本轮 softmax 的归一化，t(u)r_d(u) 则是教师对词 u 分给本轮的监督贡献。计算上，先对混合概率求导，再代入 softmax 的 Jacobian，就能得到这些表达式。责任权在训练时使用了监督目标，部署时的深度选择器另需可用信息。

```math
t(v)=\mathbf 1[v=y]
\quad\Longrightarrow\quad
\nabla_{z_d}\mathcal L
=r_d(y)\bigl(q_d-e_y\bigr).
```

这些责任权也会影响门控。继续概率 c 决定最终分布能分配多少权重给后面的轮次：前 D−1 个 gate 逐步分出停止概率，实际最后一轮接收全部剩余概率。把 α 对 c 的依赖代回同一个混合损失，就得到下面的 gate 梯度。它调整的是各轮如何共同解释教师目标，而不是要求每一轮独立完成同一任务；后者对应先计算每轮 CE、再按先验 α 加权求和，是另一种训练目标。

```math
c_j=\sigma(a_j),\qquad
\alpha_d=(1-c_d)\prod_{j<d}c_j\quad(d<D),\qquad
\alpha_D=\prod_{j<D}c_j.
```

```math
\frac{\partial\mathcal L}{\partial a_j}
=c_j\bar r_j-(1-c_j)\sum_{d>j}\bar r_d
\quad(j<D).
```

式中的 gate 偏导只描述经过本 token 混合权重的那条路径。最后一轮已经接收全部剩余概率，最后一次实际停止的 gate 不再独立分配停止质量，所以这条路径给它的梯度为零；独立 gate 辅助损失或其他共享路径仍可更新它。真实 TaH2 的 gate 读取隐状态和 top-k 概率，后续轮也继续使用此前状态。回到共享骨干参数时，需要把这些依赖一并用链式法则合并，才得到完整更新。

源码对应 `accumulate_iter` 的停止尾质量吸收及 `finalize` 的混合读出；实际实现还使用 `log(max(w, 1e-10))`，不同输出路径的归一化应按源码单独核验。上式不跨越权重 floor 或离散路由边界。固定提交：[混合运行时](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520)、[gate 输入与决策](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L228)、[NLL 与 CE 分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L45)。

小数值核验：用三轮、三个词的有限例子对 logits 与 hazard logits 做中心差分，最大绝对误差分别为 9.2×10⁻¹¹ 与 1.1×10⁻¹¹。此处验证公式，不代表已训练 TaH2 或已接入蒸馏框架。

## 算法 5 固定前缀的分布蒸馏

**过程** DistributionKD(prefixes, target\_positions, mask, teacher, student, τ)

**输入** 共同前缀、目标位置、监督 mask、教师与学生、温度 τ&gt;0，以及全词表或已声明的稀疏目标模式。

**输出** 蒸馏损失与学生累积梯度；教师目标无梯度，N=0 时返回无有效损失。

**前置条件** 算法 5 的第 3–4 行按普通 logits 定义温度化分布。TaH2 采用混合读出时，应先声明温度放在分量内还是最终分布上，再按该目标产生 `logp`；不能直接把原始 logits 路径套在 target-only 输出上。分母 N 是逻辑有效位置数；DP 多卡平均梯度时，需要按全局分母及实际归约方式缩放。

![算法 5 操作与数学公式](visualize:fde1_bGliZmlsZV91VWhWdlVQdXJEdE40ZTFONVd5V2pB_FileDrive_45288dfe06608191af94334ba8699511)

即使只查询选定 token，学生概率仍需使用全词表的归一化常数。若只对选中的 logits 做 softmax，优化的就变成“给定 token 落在选中集合内”的条件分布目标。top-k 与尾部桶的严格定义及数值保护见[在线蒸馏页](https://chatgpt.com/space/page_e434e46c533c819192afedba5f10a140)；这些目标也适用于相同的离线前缀。

# 特征监督的目标与对齐

输出分布监督告诉学生“这个前缀后应该预测什么”，但我们的研究还关心“循环中的状态能否支持后续计算”。特征监督把训练要求推进到一个选定的中间表示：让学生自身循环路径上的表示，去匹配教师完成相应显式计算后提供的表示。下面的候选目标把这项要求与学生答案损失、显式任务损失和分布蒸馏放在一起；各个 λ 决定它们对更新的相对影响。是否采用某一项、如何对齐及怎样取权重，仍由研究问题决定，这不是 TaH2 或 CODI 的通用配方。

```math
\begin{aligned}
\mathcal L&=L_{\rm student\,answer}
+\lambda_TL_{\rm explicit\,task}
+\lambda_DL_{\rm KD}+\lambda_HL_H,\\
L_H&=\frac1{|\mathcal A|}\sum_{(u,v)\in\mathcal A}
d\!\left(P_\phi h_u^S,\operatorname{sg}(h_v^T)\right).
\end{aligned}
```

公式中最先需要选择的其实不是距离 d，而是映射集 `A`：学生哪一轮、哪一层、哪一位置的表示，应该对应教师的哪个位置和语义角色？只有这个对应关系明确，缩小特征距离才有确定含义。`Pφ` 用于把学生表示适配到可比较的空间，其维度、尺度与可更新参数范围也需写清。每个 (u,v) 都按该映射取出一对向量，再平均它们的距离；因此 A 必须非空。若不使用特征监督，直接去掉这一项即可。

这也解释了为什么显式 CoT 的第 i 个词不必对应学生第 i 轮：教师和学生走的是两条不同计算路径，位置编号本身没有给出功能对应。CODI 的可选 `prj` 用于 latent 反馈，承担的是连续输入变换，也不能直接替代这里设计的特征适配器。接下来选择距离和归一化时，应一直围绕已确定的对应关系；训练后既要看表示是否接近，也要检查它能否被后续操作继续使用。同前缀 token KL 比较输出分布，特征监督比较路径上的表示，两者提供的信号由此区分开来。

框架保留两种学生初始化：从原 Base 开始，或从显式 CoT 检查点继续。也保留冻结骨干、训练部分骨干与联合训练；每种选择都记录参数组及梯度，不把小 LoRA 或逐步撤掉 CoT 预设为唯一路线。

# CODI 的特征监督实现

[CODI 原论文](https://aclanthology.org/2025.emnlp-main.36/)把显式教师任务与隐式学生任务连接起来。固定阅读提交为 `2c2314662c63e9f482ebc46614ffe9af17a241e5`，依次读 [训练脚本](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/scripts/train_gpt2_gsm8k-aug.sh)、[train.py](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/train.py) 的 `CustomTrainer.compute_loss`、[src/model.py](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py) 的 `CODI.forward`。

在该前向中，同一模型先处理问题，再用末层状态及可选投影构造连续输入。最后一次 latent 计算后解码学生答案；在指定答案边界，逐层提取学生与显式教师的 hidden states 计算特征损失。监督并非逐一对齐每个文字步骤和每轮 latent。

教师特征来自无梯度前向并作为 detached 目标；另一次带梯度的显式分支计算教师任务的 CE。

总目标同时包含学生 CE、特征损失和显式 CE，因此“特征目标 detach”不等于“整个教师分支不训练”。投影、归一化和各损失系数均应沿实际配置核对。[forward 实现](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L262)

## 对齐位置 损失尺度与共享教师

在这份提交中，学生完成最后一次 latent 迭代后进入答案 decoder。代码随后遍历 `outputs.hidden_states`，按学生与教师各自的答案边界位置 gather 向量。因此，对齐对象是答案边界处的逐层表示，并非第 i 个 latent 与第 i 个显式推理词。

Qwen/Llama 分支先对答案位置加 1，随后统一减 1。复用这些索引前，必须同时核对 tokenizer 与位置约定。[位置处理与逐层 gather](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L304)

对齐位置确定后，还要说明各层损失怎样合并。设实际返回 M 组 hidden states，S 与 T 分别是各层 gather 后的学生、教师向量。启用 `distill_loss_div_std` 时，源码将 `l2` 损失除以教师标准差的平方，将 SmoothL1 损失除以教师标准差；未启用时，两种损失都不除标准差。下式写出这一分支的尺度。

```math
\mathcal L_H=\frac1M\sum_{j=0}^{M-1}
\frac{D_j(S_j,\operatorname{sg}(T_j))}{\sigma_j^{a}},
\qquad
\sigma_j=\operatorname{std}(T_j),\quad
a=\begin{cases}2,&\mathrm{MSE},\\1,&\mathrm{SmoothL1}.\end{cases}
```

`std()` 没有指定维度，作用于整块 gather 后张量；所示代码没有给除数加 ε。公式要求标准差有限且非零，迁移实现应显式处理退化张量。SmoothL1 不是二次齐次函数，因此“先算 loss 再除 σ”和“先把两个向量除 σ 再算 loss”一般不等价。[损失及标准差处理](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L355)

CODI 的特征教师与学生使用同一个 `self.codi`。教师表示在 `torch.no_grad()` 中重新计算，并在当前特征损失中停止梯度；另一条开启梯度的显式任务前向则通过 CE 更新共享参数。因此，教师目标会随训练移动，并非来自永远冻结的独立模型。令学生和教师表示都依赖 θ，J_S、J_T 分别为两者对参数的 Jacobian；下式对比实际特征梯度与两侧都求导时的梯度。

```math
g_{\mathrm{feature}}
=J_S^\top\nabla_1d(S,T),\qquad
\nabla_\theta d(S_\theta,T_\theta)
=J_S^\top\nabla_1d+J_T^\top\nabla_2d.
```

两种梯度的方向也可能相反。对半平方误差、S=θ、T=2θ，在 θ=1 时，停止教师梯度得到 −1；若把教师也视作 θ 的可导函数，则得到 +1。这个反例区分的是算法实际采用的局部更新规则，并不说明停止梯度无效。

显式任务 CE 仍会训练共享模型。`no_grad()` 本身也不把模型切成 eval，因此两次教师前向仍需核对训练态随机性。[两次显式任务前向](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L295) · [显式 CE](https://github.com/zhenyi4/codi/blob/2c2314662c63e9f482ebc46614ffe9af17a241e5/src/model.py#L383)

# 教师接口

| 拟定接口内容 | 必须明确的含义 |
| --- | --- |
| `TeacherRequest` | 教师所见前缀、目标位置、请求 token IDs 或特征位置、教师与 tokenizer 版本 |
| `TeacherTargets` | log-probability、词表支持与尾部质量，或 hidden states 的层、位置和归一化方式 |
| `AlignmentMap` | 学生位置与轮次到教师目标的显式映射 |
| `GradientPolicy` | 教师目标是否 detach、教师任务是否另有梯度、学生哪些模块可更新 |
| `LossReport` | 每个分量的分子、分母、系数和有效目标数量 |

TaH2 的 target-only 混合 log-probability 足以计算一个硬标签 NLL，却不足以直接计算任意教师 top-k 的分布 KD。扩展时需要查询相同词表 ID 上的学生概率，或提供全分布参考路径。不同 tokenizer 的概率不能按位置直接相减。

# 与循环机制有关的研究检查

把检查合并到同一个案例中：先固定输入、读出和后续操作，比较完整状态与受干预状态。干预可包括替换为不匹配样本的状态、去掉一轮、改变接口输出或改变后续查询；同时控制额外计算与分布破坏。

单次干预降低准确率，只说明计算依赖受到影响，尚不足以证明保存了某种语义变量。

若研究过程监督，优先要求中间结果支持预先定义的多个后续操作，并与直接训练最终答案的对照比较。增加更多 loss 之前，先检查教师目标是否对学生可达、各分量是否冲突、最终答案梯度是否到达 updater。

# 交付与验收

- [ ] `teacher_target_trace.json`：同一条样本上，列清教师输入、学生输入、目标位置与特征对齐。

- [ ] `loss_and_grad_report.json`：分别展示学生 CE、显式 CE、KD 或特征项的值及到各模块的梯度。

- [ ] `continuation_protocol.md`：写清中间状态支持什么后续操作、干预和成本对照。

- [ ] 解释 CODI 的实际监督边界，并能说明哪些接口可以移植、哪些只是新设计。

这些是后续实现和研究的验收产物。CODI 的代码案例不自动保证迁移到当前基座、循环位置或训练配方后有效。

# 围绕本课讨论

```codex-prompt
请沿固定版本 CODI.forward 解释教师与学生各自看到什么、哪些前向保留梯度、特征损失对齐在哪里。随后把其设计拆成教师接口、AlignmentMap 和损失分量，讨论如何接到 TaH2；最后用中间结果的后续操作来定义验证，保留不同初始化与更新范围。
```

[继续学习 接入在线蒸馏](https://chatgpt.com/space/page_e434e46c533c819192afedba5f10a140)
