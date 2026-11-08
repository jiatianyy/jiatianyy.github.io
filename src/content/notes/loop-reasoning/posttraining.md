---
title: "循环推理后训练"
description: "后训练从能力差距出发：已有基座能做什么，部署任务还要求什么，哪些差距能通过训练或接口调整来改善。本课分别讨论原生循环基座的后训练，以及普通基座增加循环的改造；两者共用数据、监督、蒸馏和 RL 技术，但需要各自定义状态、目标与证据。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb"
sourcePageId: "page_860625e31d1c8191a35d6ab14da2abbb"
sourceSha256: 7aaabc0b94a6bd3433c02a83d8e6c9bbcf0a8b9ddfbcd015134ee27c593293f3
series: loop-reasoning
references: []
---

<span id="source-block-2d5be02e-e0e6-4390-8f71-faa679710dd0"></span>

后训练从能力差距出发：已有基座能做什么，部署任务还要求什么，哪些差距能通过训练或接口调整来改善。本课分别讨论原生循环基座的后训练，以及普通基座增加循环的改造；两者共用数据、监督、蒸馏和 RL 技术，但需要各自定义状态、目标与证据。

<span id="source-block-8102b5e7-d458-470e-a6ba-7292d5c94298"></span>

[研究总览](/notes/loop-reasoning/overview/) · [原生预训练](/notes/loop-reasoning/pretraining/) · [TaH2 工程](/notes/loop-reasoning/engineering/)

<span id="source-block-827a2608-75e7-467b-8a9d-4b3ed0ae51ea"></span>

> [!CALLOUT]
>
> **阅读顺序：目标与梯度 → 两类基座的问题 → 执行条件 → 交接接口 → 采样与更新。** 先用 H1 区分“更多深度有效”和“门控会选择深度”，再用 H2 检查动作、概率与重放。训练技术由诊断结果决定。

<span id="source-block-3d32d923-9537-44af-9788-9274a17620ab"></span>

## 后训练的数学对象与目标

<span id="source-block-fb62bd59-5a25-4ee2-b75a-7c0cb102cc8e"></span>

全部可训练参数记为 η=(θ,φ)：θ 包含原模型参数，φ 包含新增适配或决策参数。x 是部署时可见的输入，y 是文字输出，a 是实际采样的内部动作，ω=(y,a) 是完整轨迹。

<span id="source-block-fefe73ba-c734-4681-926a-1d567103adf4"></span>

状态 S 包括表示 H、KV 缓存、已有文字、循环轮次和剩余预算，只能由当时可获得的信息构造。监督答案和未来动作不能成为部署状态。

<span id="source-block-efad4107-f3f7-4c62-9048-fc794b440bba"></span>

```math
\begin{aligned}
\max_{\eta}\quad &J_R(\eta)
 =\mathbb E_{x\sim\mathcal D_x,\,\omega\sim\pi_\eta(\cdot\mid x)}[R(x,\omega)] ,\\
\text{subject to}\quad &J_C(\eta)
 =\mathbb E[C(x,\omega)]\le C_{\max} .
\end{aligned}
```

<span id="source-block-fc673adc-67cc-4fec-b9b9-38e283d798b1"></span>

R 可以是答案正确性或预先规定的任务效用；C 必须指定为实际测量的 FLOPs、延迟或另一种成本，C_max 是对应预算。输出 token 数与循环次数是成本的解释变量，不能自动代替成本本身。若同时约束显存、时延或显式能力，应分别增加约束。

<span id="source-block-e8105a34-c8d2-44e5-81d0-7ce055ffbecb"></span>

```math
\mathscr L(\eta,\lambda)
 =\mathbb E[R-\lambda C]+\lambda C_{\max},
\qquad \lambda\ge0 .
```

<span id="source-block-c617a48b-7b78-4a86-9dfb-dfdc93f987cd"></span>

λ 把计算成本放进与任务收益相同的比较中：一次额外循环带来的收益，要扣去 λ 倍的额外成本。于是，即使继续计算有一点帮助，也还要问这点帮助是否值得付出代价。调节 λ 可以研究不同的质量与成本折中，候选策略的实际预算再由独立评测确认。

<span id="source-block-f06f4f03-b1c6-4b48-b3ec-d560cc42e05d"></span>

这个目标先规定我们要比较什么；它不保证非凸神经策略满足强对偶，也没有预先决定应采用 SFT、蒸馏还是 RL。具体训练信号还要根据下面的能力差距来选择。

<span id="source-block-a2be4ac5-acc1-4bce-8e29-2323e2531a49"></span>

## 传统后训练提供什么参照

<span id="source-block-9ea0849a-8134-4242-936e-56eecae2c67d"></span>

传统后训练将语言建模目标转向更具体的使用要求。InstructGPT 用示范和人类偏好改善指令行为；DPO 从偏好数据构造直接优化目标；DeepSeekMath 展示数学数据、监督训练与 RL 的组合。它们提供目标与工程参照，迁移到 loop 时还需重新明确动作、状态和概率语义。[InstructGPT](https://arxiv.org/abs/2203.02155) · [DPO](https://arxiv.org/abs/2305.18290) · [DeepSeekMath](https://arxiv.org/abs/2402.03300)

<span id="source-block-8c032a94-50ea-47c3-9fc1-a9a0457a33c7"></span>

| 使用需求 | 对应技术 | 在循环模型上新增的问题 |
| --- | --- | --- |
| 按指令完成任务 | SFT、任务格式与数据构建 | 哪些深度参与训练，哪个出口承担监督 |
| 生成更符合偏好或更正确的结果 | 偏好优化、奖励与验证器、RL | 行为概率包含哪些随机决策，内部路由如何获得学习信号 |
| 利用教师已有能力 | 轨迹监督、分布蒸馏、过程或特征监督 | 教师与学生的状态／时间步如何对齐，是否形成真正可续用的状态 |
| 在自身生成分布上学习 | 在线蒸馏、在线 RL | 轨迹版本、采样状态、概率重算与权重同步 |
| 保持通用与显式能力 | 混合数据、约束与恢复分支 | 循环专门化后的能力变化，恢复量及额外训练成本 |

<span id="source-block-d7752379-54a0-49c0-b246-f746f5c12eb4"></span>

### 监督和蒸馏究竟对什么求导

<span id="source-block-b9b0f19c-ca6d-44fd-ad87-c0f5662b57f9"></span>

先固定一组监督样本及执行约定 κ，包括循环深度、出口和概率读出。第 i 个样本的第 t 个回答 token 只能使用提示词和此前回答组成的前缀；m 表示该位置是否接受监督，padding 和不监督的提示词取 0。N 是整个更新窗口内有效监督 token 的总数。

<span id="source-block-be6c0492-08e8-4b9d-89c0-65910cf22bcf"></span>

```math
\begin{aligned}
c_{it}&=(x_i,y_{i,<t}),\qquad m_{it}\in\{0,1\},\\
N&=\sum_{i,t}m_{it}>0,\\
L_{\rm SFT}(\eta)
&=-\frac1N\sum_{i,t}m_{it}
  \log p_\eta(y_{it}\mid c_{it};\kappa).
\end{aligned}
```

<span id="source-block-50263362-e495-41ed-baca-a52840317b7e"></span>

这一定义区分了“把答案作为被预测的标签”和“把已出现的回答 token 作为后续位置的输入”。实现必须匹配 logits 与下一 token 的移位；循环中的任何可见性规则都不能把尚未出现的标签传给对应预测。κ 若含随机路由，需说明条件化、边缘化或额外的路由梯度，不能把三者混为一个 CE。

<span id="source-block-acaf1ea2-bace-4bb0-abd5-5cf768fc09a1"></span>

对固定前缀的分布蒸馏，再假设师生使用同一词表及同一条件信息。zᵀ 与 zˢ 是师生 logits，τ>0 是温度，教师目标停止梯度。这里采用乘 τ² 的损失约定；改用不乘温度平方的实现时，梯度尺度也随之改变。

<span id="source-block-87c33139-f702-4779-9f96-f423caf861ad"></span>

```math
\begin{aligned}
q^{(\tau)}&=\operatorname{softmax}(z^{\rm T}/\tau),\\
p_\eta^{(\tau)}&=\operatorname{softmax}(z^{\rm S}/\tau),\\
L_{\rm KD}
&=\frac{\tau^2}{N}\sum_{i,t}m_{it}
 \operatorname{KL}\!\left(q_{it}^{(\tau)}\Vert p_{\eta,it}^{(\tau)}\right).
\end{aligned}
```

<span id="source-block-6dc04fec-c087-462d-8b14-59ea1aa8643b"></span>

在固定教师和前缀下，KL 中依赖学生的部分是温度交叉熵。对一个位置展开，再对学生的第 v 个 logit 求导，得到：

<span id="source-block-2816ddbc-9663-463d-920a-1b8355c3b8a9"></span>

```math
\begin{aligned}
\operatorname{KL}(q\Vert p)
 &=\operatorname{const}(q)
  -\sum_v q_v\frac{z_v}{\tau}
  +\log\sum_v e^{z_v/\tau},\\
\frac{\partial L_{\rm KD}}{\partial z^{\rm S}_{it,v}}
 &=\frac{m_{it}\tau}{N}
   \left(p_{\eta,it,v}^{(\tau)}-q_{it,v}^{(\tau)}\right).
\end{aligned}
```

<span id="source-block-e5c6fb53-1546-4b28-b0a4-f257d76db4eb"></span>

对有效监督位置（m_it=1），概率差 p−q 给出了蒸馏信号的直接含义：在同一前缀上，教师给某个词的概率比学生更高，该维 logit 的梯度就是负的，梯度下降会推动它上升；学生给得过高的词则受到相反方向的推动。τ 与监督分母决定这份信号的尺度。随后，这些输出端的误差信号沿实际循环计算图回到骨干、适配层与可微读出。

<span id="source-block-fc2ed76c-ae2d-41d8-b640-8bfa05ea0e0a"></span>

因此，分布蒸馏首先要求模型在给定前缀上作出相近预测。学生内部是否形成了可续用的中间计算，还需要另外的功能诊断；仅看输出对齐无法回答这个问题。上述比较假设师生词表和条件信息已对齐，否则须先定义对齐关系。[温度蒸馏原论文第 2 节](https://arxiv.org/pdf/1503.02531)；[离线蒸馏与过程监督](/notes/loop-reasoning/offline-distillation/)。

<span id="source-block-981ed29a-c925-4a45-a9b4-9a74adf6ef72"></span>

在线蒸馏还要区分前缀分布。若固定行为模型生成的数据，再对这些前缀优化，求的是固定数据分布下的条件梯度。若声称优化“当前学生自身前缀分布上的总体期望”，则对分布的导数也在目标内：

<span id="source-block-9cffcdc9-a487-4826-a20f-8900d302ef23"></span>

```math
\begin{aligned}
\nabla_\eta\mathbb E_{c\sim\mu_\eta}[\ell_\eta(c)]
&=\mathbb E_{\mu_\eta}[\nabla_\eta\ell_\eta(c)]\\
&\quad+\mathbb E_{\mu_\eta}
 [\ell_\eta(c)\nabla_\eta\log\mu_\eta(c)].
\end{aligned}
```

<span id="source-block-3c70527a-a864-4989-8ec6-ac299c550122"></span>

这里假设支持集固定且允许交换求导与期望。仅在采样前缀上反传 KL 对应第一项；它与包含状态访问项的完整梯度应分别说明。[在线蒸馏的详细推导](/notes/loop-reasoning/online-distillation/)。

<span id="source-block-33f4043a-5b02-48a2-a5fc-af2f559492f6"></span>

## 原生循环基座的后训练

<span id="source-block-a8ce3d03-9f78-4eff-8fe6-383ed3f2472b"></span>

先保留并理解基座的循环单元、状态、训练深度分布、出口与缓存定义，再选择要改变的任务行为。直接更换深度、缓存或读出方式也会改变部署模型，应单独记录。

<span id="source-block-cfd149f1-9ce3-4d42-bc74-c06aabca87c3"></span>

| 待诊断问题 | 候选干预 | 成功判据 |
| --- | --- | --- |
| 预训练能力能否被任务指令稳定调用 | 指令监督、模板和输出格式 | 任务能力与指令遵循改善，检查能力保持 |
| 增加循环是否提供有效增量 | 任务目标、深度课程、教师或过程信号 | 可比预算下改善准确率与成本关系 |
| 训练与推理深度改变时表现如何 | 深度覆盖、读出适配、受约束的退出学习 | 在预先声明的深度范围内检验泛化和退化 |
| 能否按问题分配计算 | 固定策略、简单规则、可学习门控 | 部署选择器优于调优后的固定策略、随机和简单规则 |
| 内部计算能否与短记录和 CoT 协作 | 监督、蒸馏、短记录或恢复分支 | 准确率、真实成本和显式能力共同评估 |
| 中间结果能否被后续计算利用 | 续算目标、状态干预与任务留出 | 结果内容可被读取、更新并支持后续任务 |

<span id="source-block-88efe667-af11-4a54-8adc-11604a315604"></span>

Loopies 的 PT→高质量数据阶段→Base→SPT→数学／代码 RL→Thinking 已在 [完整专页](/notes/loop-reasoning/loopie/)按目标、样本、规模和实现拆开。Ouro 的退出 gate 校准、Reasoning SFT 与未奏效的 RL 尝试也分别列入 [Ouro 专页](/notes/loop-reasoning/ouro/)。课程据这些具体交接点判断后训练额外塑造了哪些行为，避免把所有“Stage II”或“annealing”视为同一种训练。

<span id="source-block-441fbba5-d846-4d86-8627-c2bf6b36ff87"></span>

### 继续计算的收益与部署门控

<span id="source-block-ee0d9563-b68d-46af-b21a-6ee8575a20a8"></span>

先把门控要作的决定具体化。到达状态 S 后，模型可以现在输出，也可以支付一次循环成本，更新状态后再继续决策。比较这两条路时，已经花掉的计算相同，不必重复计入；真正有关的是此后还能获得多少任务收益，以及还要付出多少成本。

<span id="source-block-059986f7-d2b4-4d42-979f-fb92238438e9"></span>

先固定后续采用的策略 π，用 V 表示从后续状态开始的预期剩余净收益。这样，继续一次循环的价值就是当前循环成本与后续 V 的合计，而现在输出有它自己的剩余收益。二者之差 Δ 衡量在这个状态继续是否值得：

<span id="source-block-eb9b7d31-3453-4342-b638-46c246dd47c8"></span>

```math
\begin{aligned}
Q_\lambda^\pi(S,\mathrm{emit})
 &=\mathbb E_\pi[R-\lambda C_{\rm rem}\mid S,\mathrm{emit}],\\
Q_\lambda^\pi(S,\mathrm{loop})
 &=\mathbb E[-\lambda c_{\rm loop}+V_\lambda^\pi(S')
             \mid S,\mathrm{loop}],\\
\Delta_\lambda^\pi(S)
 &=Q_\lambda^\pi(S,\mathrm{loop})
   -Q_\lambda^\pi(S,\mathrm{emit}).
\end{aligned}
```

<span id="source-block-37c0513e-ca6b-4acf-8a3d-47e9c841cd58"></span>

在这两个条件决策及相同后续策略下，Δ>0 表示继续更好。若要得到整个有限深度问题的最优停止策略，需要用相应最优价值解 Bellman 递推，并在最大深度处强制输出；仅估计一次局部收益不等于求得全局最优策略。门控只能从 S 预测收益；事后使用正确答案选出每题最佳深度，只能给出一个信息更充分的参照。

<span id="source-block-db9e4825-27e7-4c75-b8fa-7cb272db32e9"></span>

实际研究可以把两个问题分开回答。先固定检查点和两种深度规则，对同一批题观察完整回答的收益与成本，确认额外计算有没有可利用的空间。然后再训练部署选择器：它只能看当时的状态，需要自己判断哪些题值得继续。在相同成本约束下，选择器应与经调优的固定深度、随机策略和简单可观测规则比较。

<span id="source-block-7df011b4-f6a4-42ef-88af-647e1631049f"></span>

这也解释了两类诊断为何不同。整段回答的配对差异可以由下面的 H1 观察；单个状态的续算价值，则需要从同一合法状态分叉，并固定后续策略。前者提供总体深度收益的证据，后者才直接针对当前这个决策点。

<span id="source-block-7515daa8-c979-411a-9088-c50c894b9efb"></span>

#### 单步交叉熵改善与任务收益的区别

<span id="source-block-2d94056a-806c-4640-920d-e2a22cbd6eaf"></span>

在同一合法前缀上，固定金标准下一 token y。前后两轮的交叉熵差，恰好是该 token 的概率比的对数；它衡量这一监督位置的似然改善。

<span id="source-block-0b73610b-75dd-4420-99d6-fca7bc453bbc"></span>

```math
g_r(y)
=\ell_{r-1}(y)-\ell_r(y)
=\log\frac{p_r(y)}{p_{r-1}(y)}.
```

<span id="source-block-5f1cc415-fa2e-4b26-9f3f-41ae67d0e1cc"></span>

它不等于下一 token 的贪心正确率变化。一个三类反例取 y=1：

<span id="source-block-1ff6fadd-eff9-45b7-a6e8-763b792c4192"></span>

```math
\begin{aligned}
p&=(0.40,0.35,0.25),& \operatorname*{argmax}_v p_v&=1,\\
q&=(0.45,0.50,0.05),& \operatorname*{argmax}_v q_v&=2,\\
\log(q_y/p_y)&=\log(1.125)\approx0.1178>0.
\end{aligned}
```

<span id="source-block-4dbdc7c0-3a5b-4159-99fe-f5c4d80432b9"></span>

金标准概率升高，但贪心输出由正确变成错误；完整回答奖励还依赖后续 token、状态、缓存与成本。因此 Ouro Stage II 的相邻轮次损失信号应作为可训练代理理解，不能直接等同上面的继续计算价值 Δ。论文目标、gate 梯度及实际部署选轮的区别见 [Ouro 的两阶段推导](/notes/loop-reasoning/ouro/)。这个反例是教学构造，不是对 Ouro 检查点的一次评测。

<span id="source-block-a7a9085b-57fe-4a7e-8d50-89a3f54b16a1"></span>

#### KL 退出信号能保证什么

<span id="source-block-84edb537-bda3-475d-a3c4-ae1be8c13270"></span>

先只考察同一前缀上的两份下一 token 分布 p、q，采用自然对数。总变差 TV 是两个分布在任一事件上的最大概率差。Pinsker 不等式给出：

<span id="source-block-6a4404fa-53a4-45e1-b8da-bb14c67d4ef6"></span>

```math
\begin{aligned}
\operatorname{TV}(p,q)
&=\frac12\sum_v|p_v-q_v|,\\
D_{\rm KL}(p\Vert q)\le\varepsilon
&\Longrightarrow
\operatorname{TV}(p,q)\le\sqrt{\varepsilon/2}.
\end{aligned}
```

<span id="source-block-de6c04d1-f367-4249-a27e-b4159ec49fa1"></span>

一个简短证明是取 A={v:pᵥ≥qᵥ}，令 a=p(A)、b=q(A)。log-sum 不等式把词表 KL 下界为二项 KL；二项 KL 关于 a 的二阶导数至少为 4，且在 a=b 时函数值和一阶导数为零。因此：

<span id="source-block-2342f4d6-8097-458d-a96a-a5d92da93f7f"></span>

```math
\begin{aligned}
D_{\rm KL}(p\Vert q)&\ge d_{\rm Ber}(a\Vert b)
\ge2(a-b)^2,\\
\frac{\partial^2}{\partial a^2}d_{\rm Ber}(a\Vert b)
&=\frac1{a(1-a)}\ge4,\\
a-b&=\operatorname{TV}(p,q).
\end{aligned}
```

<span id="source-block-7a7256c7-3984-4eee-852d-a3fb159bbb71"></span>

对零概率或边界值取连续延拓；若 KL 为无穷，前提不成立。若 p 的唯一最大概率 token 为 v⋆，其与次大概率的间隔为 δ，则有如下充分条件：

<span id="source-block-9f990736-75f4-4ec7-9944-a5ddfa447973"></span>

```math
\begin{aligned}
\delta&=p_{v^\star}-\max_{v\ne v^\star}p_v,\\
q_{v^\star}-q_v
&\ge\delta-2\operatorname{TV}(p,q),\\
\delta>\sqrt{2\varepsilon}
&\Longrightarrow
\operatorname*{argmax}_v q_v=v^\star.
\end{aligned}
```

<span id="source-block-daf39f74-c9f7-49f9-8725-b14e56b2a4e2"></span>

Huginn 论文实验的 KL 阈值 0.0005 对应 TV 上界约 0.01581；上述 argmax 保持的充分间隔是 0.03162。这是数学上界，不是测得的精度保证。固定 HF revision 的 `KLExitEvaluator.check` 实际计算 D_KL(p_prev∥p_current)，`auto` 阈值为 0.001，首轮参考为均匀分布；论文实验阈值与发布默认值应分别记录。[Huginn 专页与固定实现](/notes/loop-reasoning/huginn/) · [固定版本 `raven_modeling_minimal.py`](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)

<span id="source-block-bb4be594-4f62-45c7-93ae-d54aad66d5c1"></span>

将局部界扩展到整段生成，需要更强的前提。设 P、Q 是同一最大长度 T、EOS 后吸收的两条自回归生成分布；必须在 P 可达的每个共同前缀上约束条件 KL。链式分解才给出：

<span id="source-block-87b48a16-acac-47cd-b7a7-3236d02a9841"></span>

```math
\begin{aligned}
D_{\rm KL}(P\Vert Q)
&=\sum_{t=1}^{T}\mathbb E_{y_{<t}\sim P}
 D_{\rm KL}\bigl(p_t(\cdot\mid y_{<t})\Vert q_t(\cdot\mid y_{<t})\bigr),\\
D_{\rm KL}(p_t\Vert q_t)&\le\varepsilon_t
\quad\text{在上述前缀上成立},\\
\operatorname{TV}(P,Q)
&\le\sqrt{\tfrac12\sum_{t=1}^{T}\varepsilon_t}.
\end{aligned}
```

<span id="source-block-cb0058bb-42ad-445a-b84b-48a01ee3afce"></span>

早退如果改变后续缓存与状态，当前一次相邻轮 KL 检查并不提供所有未来前缀上的条件 KL 界。工程上须分别记录参考分布、KL 方向、检查轮次、被返回 logits 的轮次和后续使用的 KV；这也是自适应退出算法的状态必须完整的原因。

<span id="source-block-411c2abb-2566-47ba-afe1-75c91e79bf4b"></span>

#### 算法 H1 固定检查点的深度收益配对诊断

<span id="source-block-3d6b31bd-3830-4e3a-b218-72c30d95713b"></span>

**输入** 独立题目集、冻结检查点、两个预先指定的深度规则 K₁ 与 K₂、同一解码规则和成本测量协议。

<span id="source-block-b37affe5-32ab-4447-ab2e-00e776c6f7f1"></span>

**输出** 逐题输出、奖励与成本，配对差值及其不确定性；供比较准确率与实际成本使用。

<span id="source-block-b90f6b46-c811-4e94-9fac-d1857a76ea5c"></span>

**执行约定 H1** 这是诊断协议，不更新参数。两种深度未必等成本；最终按实测成本比较。若采用随机文字采样，记录初态随机量，并使用按题目和文字位置索引的公共随机数，避免增加循环次数改变其他随机数的消耗顺序。

<span id="source-block-ca45a914-a582-442a-99fa-cbb26e43bc3e"></span>

<figure class="research-embed"><iframe src="/research-assets/239b88f040edc6add052356d47facac2cd96f7869f9b1bb80a07dc9191cec260.html" title="算法 H1 固定检查点的深度收益配对诊断" loading="lazy" sandbox=""></iframe><figcaption>算法 H1 固定检查点的深度收益配对诊断 · <a href="/research-assets/239b88f040edc6add052356d47facac2cd96f7869f9b1bb80a07dc9191cec260.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-c99c369f-7943-4130-865d-3ee80b3beb8f"></span>

## 普通基座的循环推理后训练

<span id="source-block-145b9141-4ec3-4e03-b174-d7019db3eefc"></span>

这里多了一项核心任务：为原本没有按这种循环方式训练的模型建立可用的重复计算接口。层块重入、逐层重复、连续状态反馈等形式先按完整计算图分类，再分析它们需要哪些初始化、适配和训练信号。

<span id="source-block-f456b6b8-bd6e-4b20-951e-cea765a799bf"></span>

保留“建立显式能力—塑造内部计算—按需恢复显式能力”的研究框架，同时比较直接从 Base 开始与从显式能力检查点继续的分支。初始化、监督、循环位置、可训练参数、门控和恢复安排分别接受检验；Full-CoT 不是所有分支必须经过的固定步骤。

<span id="source-block-fb38dd72-8c97-4af9-ae46-1ba6ef451864"></span>

| 分析顺序 | 必须说清的问题 |
| --- | --- |
| 接口与初始化 | 新输入是否符合原模型表示尺度；循环何处进入，参数如何共享 |
| 状态与 KV | 当前表示、原始输入、位置、缓存如何共同确定下一轮 |
| 训练信号 | 最终答案、教师分布与中间功能目标分别监督什么 |
| 梯度与更新范围 | 骨干、适配层、读出和门控分别怎样获得梯度 |
| 机制与有效性 | 计算增量是否改善结果，中间状态是否真的被后续步骤使用 |
| 显式恢复与部署 | 是否需要短记录或 CoT 恢复，其额外训练和推理成本是多少 |

<span id="source-block-496cf043-bb40-451e-beec-ef7b1fa35696"></span>

TaH2 保持为这一分支的核心工程案例，用来读清输入更新、DUO 可见性、动态深度标签、停止混合、训练和自主生成。原生循环模型的具体状态语义需要另行映射，不能直接套用 TaH2 的缓存和退出定义。[进入完整工程课程](/notes/loop-reasoning/engineering/)

<span id="source-block-f83cea0c-8ec6-4c8d-9c32-3db54118fa64"></span>

### TaH2 的连续混合梯度与离散执行

<span id="source-block-2e7532ca-5fa5-4195-9598-6a3a87eddec1"></span>

TaH2 论文从同一 Qwen3 Base 出发改造 Standard、Ouro、Huginn 与 TaH2；它提供同源后训练比较，不能替代六种原生预训练模型的横向排名。论文的“在线”指当前模型在 teacher-forced 数据上生成继续标签；自主采样、在线蒸馏和 RL 各有额外的数据分布与概率接口。[论文结果与实现](/notes/loop-reasoning/tah2/)

<span id="source-block-9ad2ee15-abb5-4cd8-885d-cca83b697a57"></span>

下面先只分析固定执行路线上的 gate logit 偏导。记 D 为实际执行深度，gⱼ 为继续概率，qᵣ 为第 r 轮词表分布，aⱼ 为 gate logit；求这项偏导时，把各轮 qᵣ 视为固定值。

<span id="source-block-758fcbe7-2f9f-4c4f-bd97-cacf8106b576"></span>

为看清连续混合本身的作用，先采用温度 1、精确归一化且未触发数值下限的理想条件。停止权重及混合预测定义为：

<span id="source-block-1be304fe-d75e-4bea-b1f6-a43459c6a665"></span>

```math
\begin{aligned}
g_j&=\sigma(a_j),\qquad
R_1=1,\quad R_{r+1}=R_rg_r,\\
w_r&=R_r(1-g_r)\quad(r<D),\qquad w_D=R_D,\\
P(y)&=\sum_{r=1}^{D}w_rq_r(y),\qquad
\gamma_r=\frac{w_rq_r(y)}{P(y)},\qquad
L=-\log P(y).
\end{aligned}
```

<span id="source-block-f4a9bce9-b1f9-4fdf-8987-51f242ae8334"></span>

最后实际执行轮吸收剩余质量，故权重和为 1。对 j<D，本轮 log 权重的导数是 −gⱼ，后续轮的是 1−gⱼ；代入对混合似然的求导得到：

<span id="source-block-da59d416-1cdc-4fd8-8192-8d934a598e0c"></span>

```math
\begin{aligned}
\frac{\partial L}{\partial a_j}
&=g_j\gamma_j-(1-g_j)\sum_{r>j}\gamma_r,\qquad j<D,\\
\left.\frac{\partial L}{\partial a_D}\right|_{\text{直接混合权重路径}}
&=0.
\end{aligned}
```

<span id="source-block-7559468a-946f-477f-8806-f919bd6c4751"></span>

这项偏导说明，NLL 可以通过连续混合权重训练 gate。它没有对硬阈值选择的离散路线求导，因此不是策略梯度。末轮的结论也只针对当前 token 的直接混合权重路径；辅助 BCE、共享 hidden、其他 token 和后续计算仍可能贡献梯度。

<span id="source-block-45ef58a1-3abe-43bc-a52f-562c09e7133b"></span>

若要得到模型参数的完整梯度，还需把 qᵣ、gate 输入和共享骨干的链式依赖加回来。权重 clamp、dtype 舍入和分支变化则按实际源码处理。[混合梯度与教师目标](/notes/loop-reasoning/offline-distillation/) · [三条实际损失路径](/notes/loop-reasoning/parameter-update/)

<span id="source-block-b231a90d-3ead-413e-a839-86c341b551cf"></span>

| 对象 | 依赖的信息 | 可以解释什么 | 不能直接替代什么 |
| --- | --- | --- | --- |
| 责任度 γ | 当前目标 token、已执行各轮概率、停止混合权重 | 此 token 的混合 NLL 如何分配梯度 | 最优循环深度、真实 wall-clock 收益 |
| 动态深度标签 | 训练目标与下一轮无梯度 lookahead；全批正收益覆盖 cutoff | 当前数据前缀上的单步 NLL 改善监督 | 部署时可直接访问的 oracle、长期任务成功率 |
| 实际路由 | gate、阈值或可选随机采样、warmup、上限与 override | 哪些 token 真正继续计算及留下何种 KV | 所有潜在路线的边缘概率 |
| 后训练奖励 | 完整回答、验证器或偏好、已声明成本 | 研究所关心的任务收益及计算约束 | 自带正确缓存、概率重放或权重同步 |

<span id="source-block-bef286a4-393f-42ad-903f-4de62fc56dd6"></span>

调整温度、混合方式、奖励、任务或数据分布后，应重新核对标签与部署决策的对应。一次探测的正收益、事后最佳深度与可部署 gate 是不同统计对象；在序列任务中，当前步骤的额外计算还会改变后续 hidden 和 KV。

<span id="source-block-ac74dc94-76d6-4ace-948c-cc5baae0c209"></span>

#### 随机深度历史如何进入后训练目标

<span id="source-block-f69402db-0727-449d-a18e-91b004695f84"></span>

默认确定性阈值只产生一条路线。固定源码另有 stochastic_sampling；关闭训练模式并不会自动清除已经打开的该选项。若使用内部随机路由，记录 a 为整条深度动作历史，联合策略与文字边缘满足：

<span id="source-block-e4f864e1-1559-40d3-a926-82f71285dea3"></span>

```math
\begin{aligned}
\pi_\theta(y,a_{0:T}\mid x)
&=q_\theta(a_0\mid x)
\prod_{t=1}^{T}
q_\theta(a_t\mid y_{<t},a_{0:t-1},x)\\
&\qquad\qquad\cdot
p_\theta(y_t\mid y_{<t},a_{0:t},x),\\
p_\theta(y\mid x)
&=\sum_{a_{0:T}}\pi_\theta(y,a_{0:T}\mid x).
\end{aligned}
```

<span id="source-block-8300d12f-0999-4d1a-83c3-0bbf9768946f"></span>

式中的 a₀ 包含 prompt／prefill 中影响后续状态的随机路由；若 prefill 确定，它的分布就是概率为 1 的单点。aₜ 则包含当前文字位置的一串继续／停止决策。对这些动作历史求和，才得到只保留文字的边缘似然。

<span id="source-block-a4484ff9-72ee-4150-ac78-2620df55c863"></span>

早先的深度动作会改变后续缓存，因此不能把各 token 的深度独立平均，也不能仅凭文字加一条路线上的条件 logp 还原文字边缘似然。联合动作训练需要记录真实采样事件和行为概率；文字偏好目标则需要明确如何边缘化，或声明所用近似。[联合与边缘 KL、策略梯度及有限路由评分算法](/notes/loop-reasoning/reinforcement-learning/)

<span id="source-block-2174d94a-656e-42bc-82ef-d94a41eb7fb2"></span>

#### 分别比较训练深度与推理深度的收益

<span id="source-block-7b4119f2-4fc3-40f7-b699-fe489482ba0b"></span>

每项比较同时固定或单列初始 checkpoint、训练数据来源、训练成本、文字 cutoff、训练与推理深度上限、每题采样数、评分规则及执行后端。TaH2 论文报告的训练深度扩展、固定检查点推理深度扩展、并行投票及单条轨迹延长分别回答不同问题。平均深度只能描述执行的一部分；DUO 历史、LM head、lookahead、prefill 与 decode 的不同成本都应进入相应账本。[论文全部主结果和消融](/notes/loop-reasoning/tah2/) · [性能与恢复状态](/notes/loop-reasoning/systems-evaluation/)

<span id="source-block-773031d4-dd19-42f6-a672-faca36ef1d67"></span>

## 后训练比较需要固定的执行条件

<span id="source-block-a4628d6d-feb1-4a7f-8f0e-96e706899018"></span>

选训练方法之前，先确定要改善哪一个预测器、哪一种更新过程。下表比较各模型的具体差异，随后从标签、采样、验证和数据四个环节说明它们怎样影响训练设计。

<span id="source-block-7c5aef46-d94b-47ec-a58c-969944304bef"></span>

| 需要改善的问题 | 本次核到的具体边界 | 后训练设计先落实什么 |
| --- | --- | --- |
| Ouro 的出口学习 | HF labels 的加权 logits CE 与论文逐轮目标，可对同一 gate 产生相反梯度；默认 threshold 又可抢先于 weighted exit | 明确损失定义、训练分支、推理分支及实际短路规则，再比较 SFT/KD/RL |
| Huginn 的浅深协作生成 | 发布 speculative 路径使用 argmax／相对概率阈值验收，一般不保持目标抽样分布 | 若研究无损采样，先实现相应接受校正与逐行 KV 回滚；若保留启发式，按实际行为分布评测和评分 |
| Loopie 的 SPT 与后续更新 | 响应 mask 保留 prompt→response 的间接梯度；归约 flag 改变有效 token 的权重 | 固定有效监督集合与分母，区分改配方的收益和改样本权重的收益 |
| SMELT／LoopMoE 的跨轮机制 | 有序局部放大、RMS gain、router overlap 与代数 gate 吸收各有明确条件 | 用状态或专家干预检验机制；相似度、Jaccard 不能替代干预证据；联合改动的消融不能分离各组件贡献 |
| TaH2 的按需循环 | 路由概率、条件停止混合、文字边缘和真实访问成本不相同；资源覆盖还能改变路由 | 先定联合／边缘训练对象，保存实际执行与服务覆盖，分别评价局部分布和完整任务收益 |

<span id="source-block-d9f7e2f4-7e1c-4199-ab4a-dd14ea15b472"></span>

[Huginn](/notes/loop-reasoning/huginn/) · [Ouro](/notes/loop-reasoning/ouro/) · [Loopies](/notes/loop-reasoning/loopie/) · [SMELT](/notes/loop-reasoning/smelt/) · [LoopMoE](/notes/loop-reasoning/loopmoe/)。论文结果保留，代码差异不替代原文成绩；尚无官方代码可核的部分继续明确登记证据缺口。

<span id="source-block-63106da6-8d33-4743-ae0e-4a59fdbfa042"></span>

TaH2 的二轮随机分支能具体说明“执行概率”和“读出权重”的差别。沿 T9 中同一条确定的全继续计算链，记继续率为 g。先取温度 1，忽略权重下限与舍入，不使用 top-k／top-p 采样截断，也不触发资源强制停止、warmup 或 override。在这些条件下，固定执行两轮的读出、随机深度下的文字边缘分布以及平均深度分别为：

<span id="source-block-fb7e4ffb-4b52-4bef-87e0-3fde3c9d444c"></span>

```math
P_{\rm fixed}=(1-g)q_1+gq_2,\qquad
P_{\rm marginal}=(1-g^2)q_1+g^2q_2,
\qquad \mathbb E[D]=1+g.
```

<span id="source-block-44c89a36-36fa-406f-851b-472fb1e56cde"></span>

差异来自两次操作的组合：先随机选择执行深度，再由停止轮吸收剩余混合质量。因此，式中的平方不能直接用作节省计算的证据。一般 M 的推导和 T9 见[内部循环页](/notes/loop-reasoning/internal-loop/)；联合分数、文字边缘分数和 gate 梯度见 [RL 页](/notes/loop-reasoning/reinforcement-learning/)。这一例子分析代码中的可选随机分支，论文主配方仍采用确定阈值。

<span id="source-block-5a9be7ff-de0b-4aac-b8b0-312002559dd7"></span>

资源不足导致的强制停止也需要单独分析。[生成页](/notes/loop-reasoning/generation/)给出从共同内部状态出发时的局部 TV 界，并说明贪心输出、NLL 和整段序列分布还需要哪些条件。若门控目标是降低延迟或显存，策略描述还应包含服务资源状态；仅凭平均深度下降，不能判断模型是否更合理地分配了计算。

<span id="source-block-aa3176de-9ba5-479a-8f1c-4e7f63ed5326"></span>

### 标签对齐与训练目标的作用范围

<span id="source-block-06cb02e1-9baf-4c81-a501-d365e90b44c8"></span>

统一训练器不能假定各基座的同名参数具有相同语义。Huginn HF 的 labels 需要预先 next-token 对齐；Ouro 默认依赖的 Transformers loss 会在内部再移位，并且 logits_to_keep 不会同步选择 labels。接入时先构造全序列 next-token target，再按同一位置选择所需 logits 与 target；归约分母、mask 和损失项的作用范围一起固定。[Huginn](/notes/loop-reasoning/huginn/) · [Ouro](/notes/loop-reasoning/ouro/)

<span id="source-block-27ade001-947b-44a7-9b30-c5010b1b46a8"></span>

Loopie 论文的序列级 GSPO 和动态补样，需要在具体实现上分别落实。固定公开 Megatron 通用路径使用逐 token ratio 与 clip；过滤同奖励组的开关被校验器拒绝，源码说明当前没有丢组后的再生成。因而“已有 RL 入口”尚不足以闭合论文配方。应先保留 response ID 和有效响应长度，完成序列 log-ratio 归约，再裁剪；补样还需要明确组计数、终止和样本耗用。[Loopie 完整调用链与迁移契约](/notes/loop-reasoning/loopie/) · [RL 工程课](/notes/loop-reasoning/reinforcement-learning/)

<span id="source-block-4f088633-09f9-45ad-850e-8fdc2e8d963f"></span>

TaH2 则把训练目标的定位进一步拆成三个可测对象：单轮分量的 CE 改善、同一位置最终混合预测的改善、改变续算集合后对全部有效目标的改善。前两个在固定上下文和精确停止混合下同号，但排序可反转；第三个还包含 KV 对后续位置的影响。算法 T10 用独立快照分别执行 singleton 与 pair 干预，并显式报告交互项。选择监督、蒸馏或 RL 前，应先说明希望优化哪个对象，再判断监督信号是否足够。[TaH2 推导、反例与 T10](/notes/loop-reasoning/internal-loop/)

<span id="source-block-740c197b-ef68-4dc0-9c4a-93e03d5be6d3"></span>

### 从请求配置到实际采样与评分

<span id="source-block-06f53512-8fda-4051-92e5-c3cfa521aef2"></span>

一次 rollout 要依次核对请求配置、服务端执行、文字行为分布和学习端评分。它们之间的转换会改变响应预算与概率，不能仅看同名字段。具体案例见 [Loopie 参数与评分路径](/notes/loop-reasoning/loopie/)、[TaH2 采样变换与 T11](/notes/loop-reasoning/generation/)及 [RL 支持质量推导](/notes/loop-reasoning/reinforcement-learning/)。

<span id="source-block-893ccb2a-99ed-48f3-bf82-7981ed1a1042"></span>

| 对象 | 核对的实际记录 | 已发现的具体边界 |
| --- | --- | --- |
| 配置意图 | 上游请求中的温度、top-k、top-p、响应上限 | Loopie 通用 MegatronLocal 路径不发送 top-k 与响应上限；显式零温度经 `or 1.0` 变为 1 |
| 有效执行 | HTTP 字段、服务端默认、总上下文与响应预算、kernel 版本 | 缺少响应上限可退回剩余上下文容量；TaH2 固定 FlashInfer 默认先 top-k 归一化再 top-p |
| 文字行为分布 | 实际温度与最终支持上的归一化概率 | 原始混合误差小，不保证 nucleus 后误差小；需要共同支持误差和独占支持质量 |
| 学习端评分 | raw、温度后或 processed logp；目标策略及重要性分母 | 原始模型 logp 不自动等于行为 logp；目标在行为支持外的质量无法由已有样本重加权补回 |

<span id="source-block-356c02ef-360e-4912-ad97-de47e64b471a"></span>

确定这些字段后，先选择要优化原始文字策略还是采样变换后的 processed 策略，再决定重评分、KL 方向和数据收集方式。不同后端的变换顺序也要分别确认：Loopie 的固定 Torch 后端禁止 top_k>0 与 top_p>0 同时启用，不能套用 TaH2 的组合采样顺序。[固定 TorchSampling](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/sampling/torch_sampling.py)。

<span id="source-block-4d2dc3dc-602e-4eac-9566-f42bbe4d3887"></span>

概率定义之外，rollout 还依赖运行兼容性。Ouro 的固定 cache／依赖组合存在接口冲突，理论机制已经说明，仍不代表该组合能直接采样；具体条件见[矩阵中的运行边界](/notes/loop-reasoning/model-matrix/)。这些结论限定于已核查的仓库分支，不用于推断作者未公开的论文专用 launcher。

<span id="source-block-23d19613-9c51-4aae-b2f9-5ba099e571ba"></span>

### 用同一有效验证目标比较检查点

<span id="source-block-70c11ff1-9c56-484b-a2ac-3aa37cf2ab0f"></span>

TaH2 的固定 eval 路径说明，处理后行数能被数据配置 q 整除，并不保证 DataLoader 的 batch 数能被实际 DP 数 P 整除。改变 batch size、拓扑或过滤规则后，应重新对齐样本出现记录、监督 mask、损失分支与实际分子，再比较验证曲线。[数据计数链](/notes/loop-reasoning/data/#source-block-23d1af89-77e4-4c85-a0d0-d3abf5b46601) · [公式、八组检查与 T12](/notes/loop-reasoning/systems-evaluation/#source-block-b445b124-6da5-430c-896a-5b1c0faf4646)

<span id="source-block-9abad154-4a44-427d-bec4-fa6fc1124189"></span>

同一执行集和同一正分母下，统一正比例缩放不改变固定候选的 NLL 排序；这不意味着遗漏部分的均值已经被恢复。跨配置改变执行集或 mask 后，原缩放和目标都可能变化。零执行批次产生的原始零值必须标记为不可评估；pad 引入的重复出现也不是新的独立题目。

<span id="source-block-abb8135c-9c67-44ca-9613-3e8876ed3739"></span>

实际优化器组的 LR、scheduler horizon、本次更新使用的 LR 和更新后的日志 LR 同样分开保存。[一次更新课](/notes/loop-reasoning/parameter-update/#source-block-673e822e-b486-411f-9aee-6fe5de3c2e6c)给出固定前缀分组、warmup／cosine 公式及提前退出条件。训练目标、teacher-forcing NLL 与自主生成任务分数分别验收，再据公平预算选择 SFT、KD 或 RL。

<span id="source-block-ccb56edc-8d72-4147-9c08-abec51c1c24a"></span>

### 同源教师材料怎样形成实际训练序列

<span id="source-block-30ac03bb-3748-4722-a48e-63c9bd139405"></span>

同一条工具轨迹可以展开为多条完整历史前缀，当前段监督而历史段仅作上下文；同一份语料也可以按文件倍率或token quota进入训练。故同源教师材料与同样本条数，均不足以推出相同输入量、监督量或训练FLOPs。[TaH2 episode展开与D3](/notes/loop-reasoning/data/#source-block-9f477f57-3017-463b-858f-093b6c18790b)给出重复输入的闭式计数；[Done Right D2](/notes/loop-reasoning/done-right-i/#source-block-ba6de3ee-0ee9-4b39-8050-3c0ea2cab73f)的quota保证只到packer前，不自动保证最终CE比例。

<span id="source-block-1d422d27-4e78-40a0-a60c-28190edef995"></span>

比较 SFT、蒸馏或 RL 时，先固定问题／题族划分与教师版本，再记录模板、episode 划分、截断与过滤、输入位置数、有效目标数和来源权重单位。prompt／历史的 loss mask 只移除直接 CE，其 K/V 仍可影响当前回答并接收梯度；detach 则是另一种干预。这些是研究协议与接口要求，TaH2–verl 的具体集成另需实现。

<span id="source-block-d010369f-771d-455e-9bc6-f07ab54b0642"></span>

## 预训练与后训练的接口

<span id="source-block-3550b92c-7aea-48fa-adc1-2b803ff12ce5"></span>

为每个具体 checkpoint 填写下面的交接表。它把论文中的结构描述变成后训练能够依赖的条件；先核到实现，再据实际载入与数值对照确认。

<span id="source-block-ff2725dd-e438-4937-8be2-8a03d15117b5"></span>

| 接口项 | 必须继承或明确改变的内容 |
| --- | --- |
| 模型结构 | 循环单元、共享参数、输入注入、状态初始化、归一化和输出头 |
| 深度与出口 | 训练深度分布、推理范围、退出规则、各出口读出方式 |
| 位置与缓存 | token／层／轮次索引、可见性、保留、清除和复用规则 |
| 梯度 | 展开范围、截断与 detach、重计算、监督位置和可训练参数 |
| 输入与输出 | tokenizer、控制 token、模板、监督 mask、概率支持与答案提取 |
| 数值与版本 | 权重 revision、配置、实现 commit、精度及依赖 |
| 在线执行 | 随机动作、行为概率、旧／新权重版本、同步和恢复状态 |
| 成本与评测 | prefill／decode 计算、循环次数、输出长度、延迟和显存口径 |

<span id="source-block-f24c6247-7f50-4f14-a271-31f4bf41ff1c"></span>

## 六类原生基座的具体接入条件

<span id="source-block-0a720a33-341c-4d20-bc7f-3b99f2fb3e76"></span>

| 基座／材料 | 本轮核到的实际条件 | 后训练首先落实的接口 |
| --- | --- | --- |
| [Huginn](/notes/loop-reasoning/huginn/) | HF 标量 num_steps 令循环段无梯度；labels 需预先对齐；mask／KV 策略有专有语义 | 显式指定有梯度轮次，检查 core／prelude 梯度与监督移位，再研究 SFT、退出和连续 latent |
| [Ouro](/notes/loop-reasoning/ouro/) | HF labels 对加权 logits 求 CE；退出在全部轮次计算后选输出 | 决定并实现训练目标；对齐每轮读出、行为概率和更新分布；将输出选择与真实短路执行分别验收 |
| [Done Right I](/notes/loop-reasoning/done-right-i/) | native 无 KV、末轮读出；BF16 权重不含旧优化／数据状态 | 以固定 recipe 开启新的后训练；实现可核对的生成／logp／缓存，再接高吞吐 rollout |
| [Loopies](/notes/loop-reasoning/loopie/) | preview 阶段未明；两个规模配置不同；HF／vLLM 缓存路径不同 | 固定阶段与配置，验收逐 visit KV；先保持两次循环研究响应行为，再定义动态深度 |
| [SMELT](/notes/loop-reasoning/smelt/) | 完整训练栈不能公开，未取得可下载官方 Base | 复用三预算控制、消融和评测范式；具备明确权重后再进入模型实验 |
| [LoopMoE](/notes/loop-reasoning/loopmoe/) | 官方实现／权重仍未核实；论文没有现成 halting 或后训练方案 | 先完整实现 IterAdaLN 与配平规则；再研究固定 K 对齐、读出／停止校准及调制开销 |

<span id="source-block-6f75ceda-7007-4305-a9cf-eac0d151c9bd"></span>

### Ouro 已报告的 RL 尝试

<span id="source-block-d3a420c4-26ed-4ce0-9a45-e07a094f8991"></span>

[Ouro 原文 §4.5](/notes/loop-reasoning/ouro/)记录了 DAPO／GRPO 没有超过最终 SFT checkpoint 的尝试。其中固定四轮 rollout 后模拟较早退出、用较早轮分布更新的分支出现行为分布与更新分布失配；全部固定四轮的另一分支可训练但没有显著收益。作者对原因的解释仍是假说。这使“生成在哪个深度发生、更新重算哪个分布、退出与 KV 历史如何进入样本”成为本课需要优先写清的工程问题，也对应下文 H2 的轨迹重放要求。

<span id="source-block-805926cc-81fc-41e3-a50d-aad7fb74aa9f"></span>

后训练收益同时登记任务质量与真实成本。名为 early_exit 的 API 可能只是选择已算好的输出；每样本逻辑轮数减少也可能仍受 batch 内最深样本支配。因此成本应从实际模块调用、prefill／decode、路由和同步中测量，不能仅由退出编号推出。各专页已把论文机制、代码行为和未完成的运行验收分开。

<span id="source-block-ba645932-b7d2-40f3-8745-e3baf8fc59aa"></span>

## 内部动作的概率与策略梯度

<span id="source-block-f3173f05-46b5-4817-adfa-38f6593c2dd0"></span>

如果内部深度也由采样决定，一条生成轨迹就同时包含“继续还是输出”的内部动作，以及实际输出的文字。两类事件共同决定后来会到达什么状态。写轨迹概率时，需要沿发生顺序，把每个真实采样动作在当时状态下的概率乘起来；取对数后，它们就成为下面的两组和。

<span id="source-block-69436e24-ea13-46c1-b2c9-47844c0c037c"></span>

这里先定义一种确实采样内部动作的参考策略。事件 j 的历史 ℋⱼ 只含此前文字、动作与已记录的外生随机量 ξ；ξ 与参数 η 无关，外层期望也对它平均。当前状态 Sⱼ(η) 由参数 η 沿这段历史重建。G、T 分别记录门控和文字采样事件；确定性的内部向量更新不另外贡献一个动作概率。

<span id="source-block-81b6422d-965c-4b49-a206-60782a3f314d"></span>

```math
\begin{aligned}
\pi_\eta(\omega\mid x,\xi)
&=\prod_{j\in\mathcal G}q_\eta(a_j\mid S_j(\eta))\\
&\quad\cdot\prod_{j\in\mathcal T}
 p_\eta(y_{t(j)}\mid S_j(\eta)),\\
\log\pi_\eta(\omega\mid x,\xi)
&=\sum_{j\in\mathcal G}\log q_\eta(a_j\mid S_j(\eta))\\
&\quad+\sum_{j\in\mathcal T}
 \log p_\eta(y_{t(j)}\mid S_j(\eta)).
\end{aligned}
```

<span id="source-block-767f9185-e28a-4080-a5d7-ce77a08df9f8"></span>

固定同一段文字和离散动作历史后，改变参数仍会改变沿途的 hidden 与 KV。因此，用当前策略给旧动作重新评分时，要用当前权重重建这些状态；只在旧模型的 hidden 上换一个输出头，并不等于重新计算了当前策略的概率。完整求导既包括输出头或门控头，也包括先前循环怎样影响 Sⱼ。

<span id="source-block-aa2be567-3a0c-4d7b-8241-bd4828d8b9f7"></span>

事件记录还应分清采样与强制执行：到达深度上限后必须输出，这一步没有额外的可学习门控 logp；真正采样到 EOS token 时，则仍有正常的文字概率。

<span id="source-block-8cd3f86a-323d-4fc0-8d8e-3394eeb8ecc8"></span>

令 Fη 为一条轨迹的净收益。对参数无关的离散轨迹集合，在支持集和可微条件满足时，由乘积法则得到：

<span id="source-block-833b44b1-76d8-4180-adef-463eca903986"></span>

```math
\begin{aligned}
\nabla_\eta J
 &=\nabla_\eta\sum_\omega\pi_\eta(\omega)F_\eta(\omega)\\
 &=\mathbb E_{\pi_\eta}\!\left[
  F_\eta\nabla_\eta\log\pi_\eta(\omega)
  +\nabla_\eta F_\eta(\omega)\right].
\end{aligned}
```

<span id="source-block-2596b673-3961-412c-a7ff-5fef0d285053"></span>

当正确性奖励和按已执行操作计数的成本在给定轨迹后不再显式依赖 η，第二项为零；若另加依赖隐藏状态的可微代价，则要保留它。每个事件可以减去仅依赖动作前历史的 baseline，并使用剩余回报。在策略代理目标中，baseline 与优势作为停止梯度的权重；价值函数若训练，采用单独的价值损失。对条件概率归一化求导可得：

<span id="source-block-ddc0d4c6-0663-4be1-bf4b-2be3e563152c"></span>

```math
\mathbb E_{d_j\sim\pi_\eta(\cdot\mid\mathcal H_j)}
 \left[b(\mathcal H_j)\nabla_\eta
       \log\pi_\eta(d_j\mid\mathcal H_j)\right]=0 .
```

<span id="source-block-65bc9a92-6cb4-49bf-a95f-aed4d1d2f9be"></span>

TaH2 的默认阈值路由、连续混合和辅助 BCE 有各自的梯度来源；固定源码也提供可选 Bernoulli 采样。是否采样、warmup、强制上限和 override 均需记录。采用采样选项本身并不会自动建立完整的联合轨迹概率、行为记录与策略更新；下述参考目标仍需相应的评分、重放与同步接口。[TaH2 内部循环与门控](/notes/loop-reasoning/internal-loop/)。

<span id="source-block-203f2751-94fd-4bb4-ac52-bb320e26ad5f"></span>

同一句回答可能由多条内部路线产生：有的地方只算一轮，有的地方多算几轮，最后文字却相同。若偏好标签只评价这句文字，模型给它的概率就应包含所有能够生成它的路线，所以需要对内部路线求和。

<span id="source-block-e6761886-c478-499a-be70-e7a6f7fbae97"></span>

一条采样路线的联合概率只记录其中一次生成方式。把它直接放进文字偏好目标，还会同时评价这条内部路线本身，从而改变目标。下面的对数求和写出了两者的区别：

<span id="source-block-4126f4d0-8c90-437d-8256-d60374de8531"></span>

```math
p_\eta(y\mid x)
 =\mathbb E_{\xi}\!\left[\sum_a\pi_\eta(y,a\mid x,\xi)\right].
```

<span id="source-block-823be8f7-3988-42a3-ba93-f11af63be826"></span>

采用固定可重放路线，或显式优化带路线的数据，均可形成另外的目标，但需先写清其含义。[DPO 原论文第 4 节](https://arxiv.org/pdf/2305.18290)。

<span id="source-block-51445168-5060-41f1-8f64-b419f2fa478a"></span>

## 从概率重放到一次同步更新

<span id="source-block-b46f4307-e9fb-4a20-a030-13f855cf995d"></span>

现在把概率写成一次可以执行的更新。先用行为快照 η_old 生成轨迹，保存每个真实采样事件的旧 logp。训练端沿相同的文字和动作历史重放，用当前参数重新评分。旧分数回答“当时多大概率作出这个选择”，新分数回答“当前模型对同一选择赋多大概率”；两者相除，就得到该事件的概率比率 ρ。

<span id="source-block-b0821ab3-153b-4d60-ab3f-add501cf8cea"></span>

下面用 PPO 的逐事件裁剪作参照，将优势估计作为本轮停止梯度的权重。这样就能把轨迹收到的反馈与每个已采样事件的新旧概率连接起来：

<span id="source-block-46fecc9f-b1d5-408a-965c-5972ed6b3f8e"></span>

```math
\begin{aligned}
\rho_{ij}(\eta)
 &=\exp\!\left(\ell_{ij}(\eta)-\ell_{ij}^{\rm old}\right),\\
L_{\rm clip}(\eta)
 &=\frac1n\sum_{i=1}^{n}\sum_{j\in\mathcal E_i}
  \min\!\left(\rho_{ij}\widehat A_{ij},\,
  \operatorname{clip}(\rho_{ij},1-\epsilon,1+\epsilon)
     \widehat A_{ij}\right).
\end{aligned}
```

<span id="source-block-19a640e6-204a-4f48-a6c3-f64b1a45e1de"></span>

这里选用“轨迹平均、轨迹内事件求和”，对应每道题的回报及其轨迹得分和。改成按全批总事件数归一化，会改变批次梯度尺度；先在每条轨迹内平均事件再汇总，则改变轨迹长度权重。裁剪目标是代理目标，不能把重复使用旧样本的每一步更新都叫作当前策略真实收益的无偏梯度。单事件比率也不是整条轨迹的重要性权重。[PPO 原论文第 3、5 节](https://arxiv.org/pdf/1707.06347)。

<span id="source-block-412a85bf-28a9-4fb5-8a5e-37a68aee7d13"></span>

参考算法采用固定正温度的完整支持 softmax，关闭 dropout，并记录参数无关的随机初态。若引入 top-k／top-p、不同温度、可学习随机初态或其他随机路由，应重新核对行为分布、重归一化、支持集和评分定义。概率重放一致、梯度定义一致、权重版本一致是三项不同约定。

<span id="source-block-d6ce7d62-1cb6-4da9-8b59-df8400187db4"></span>

| 数学对象 | 工程记录 | 必须保持的语义 |
| --- | --- | --- |
| 离散事件与历史 | `event_kind`、token／route 值、文字位置、轮次、动作 mask | 当前动作只依赖此前历史；强制事件与采样事件区分 |
| 完整状态 S 与缓存 | 原始 prompt、初始化随机量、重建顺序、可见性规则 | learner 用当前权重重建，行为 KV 不作为当前状态 |
| 行为概率 | `behavior_logp`、版本、温度及支持变换 | 分母对应真正执行的采样分布，更新期间固定 |
| 优势和成本 | 题目 ID、奖励、分段成本、baseline、有效事件 mask | baseline 只看动作前信息；成本按声明口径累计 |
| 同步更新 | actor／learner 版本与发布回执 | 本轮采样结束后更新；新一轮采样前全部 actor 切换完成 |

<span id="source-block-94eb5230-8fab-4b60-8599-0ebffa88dc21"></span>

这套算法给出概率与状态接口的语义参照。TaH2 与 verl 的具体组合仍需按各自源码实现并核验；完整回报、GAE、偏好优化、KL 与分布式细节继续在[强化学习与偏好优化](/notes/loop-reasoning/reinforcement-learning/)及[分布式执行与性能评测](/notes/loop-reasoning/systems-evaluation/)展开。

<span id="source-block-556c90e0-ac92-49ed-bc3a-2a7a4ec55899"></span>

### 算法 H2 文字与内部随机动作的同步采样和更新

<span id="source-block-31b0baee-26f2-4d05-a06e-f5c42bd55168"></span>

**输入** 参数 η、输入批次、有限文字与循环上限、固定温度、奖励和成本函数、裁剪范围及更新次数。

<span id="source-block-c06d546f-7783-43d2-a65a-40f56a70f226"></span>

**输出** 新参数版本、逐事件行为记录、概率重放核对及收益与成本统计。

<span id="source-block-a34cad14-8667-46af-b8b4-6381817c8cbb"></span>

**H2 的动作。** 策略真正采样 loop／emit。TaH2 默认硬阈值执行不直接适用；其可选 stochastic_sampling 也需要补齐行为记录与联合评分。

<span id="source-block-00fe52f6-c306-4bc3-863a-da404d574f27"></span>

**H2 的状态。** InitState 使用已记录的参数无关随机初态和确定性 prefill。可学习的随机 prefill 路由应展开为门控事件，加入概率与重放。StepLoop 执行一次循环，AfterEmit 进入下一个 token；三者都显式更新表示、位置及 KV，并包含必执行计算。

<span id="source-block-046644c5-f9a8-4f6c-ba46-23b2dca41d8d"></span>

**H2 的梯度。** 从 prompt 开始重放完整历史，保留已定义的参数依赖；采用截断反传时，需要另列这一近似。

<span id="source-block-4148ce2d-65e8-4f98-bb53-647e7a257da7"></span>

<figure class="research-embed"><iframe src="/research-assets/293ae9ee3361dd5294bc15ef18df058275a9a24cc059ceea2c33eda41f8752b2.html" title="算法 H2 文字与内部随机动作的同步采样和更新" loading="lazy" sandbox=""></iframe><figcaption>算法 H2 文字与内部随机动作的同步采样和更新 · <a href="/research-assets/293ae9ee3361dd5294bc15ef18df058275a9a24cc059ceea2c33eda41f8752b2.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-52e26bc8-00c8-4707-941b-2d90ef7f0ba3"></span>

## 复用既有工程材料

<span id="source-block-fdde03b7-b299-4e9a-89dc-643ad28aeb65"></span>

| 需要学习的能力 | 已有入口 |
| --- | --- |
| 样本、监督和更新 | [数据构建与版本固定](/notes/loop-reasoning/data/) · [一次参数更新](/notes/loop-reasoning/parameter-update/) |
| 内部循环和自主生成 | [一次内部循环](/notes/loop-reasoning/internal-loop/) · [从训练到自主生成](/notes/loop-reasoning/generation/) |
| 教师与蒸馏 | [离线蒸馏与过程监督](/notes/loop-reasoning/offline-distillation/) · [接入在线蒸馏](/notes/loop-reasoning/online-distillation/) |
| 完整 RL 与系统 | [强化学习与偏好优化](/notes/loop-reasoning/reinforcement-learning/) · [分布式执行与性能评测](/notes/loop-reasoning/systems-evaluation/) |

<span id="source-block-9c2303ba-6adc-4512-a835-3894d91ef431"></span>

TaH2 与 verl 的资料和接口设计已经保留；二者集成、真实模型数值对照及性能仍待实现验收。通用 RL 课程继续完整学习，不把它缩减为“调用一个奖励函数”。

<span id="source-block-982ae2bf-84ed-4357-8935-1b443d783c12"></span>

## 第一份研究产物

<span id="source-block-a63d5c8c-d2a4-4b68-be58-9edace82235e"></span>

先从一个可观察的能力差距写出问题定位卡：具体基座与任务、状态与条件信息、目标和成本约束、梯度路径、候选干预及其否定证据。结合算法 H1 区分深度收益与部署门控，结合算法 H2 说明动作与概率重放。随后分别为一个原生候选和 TaH2 案例填写交接表，比较共同接口与专有规则。

<span id="source-block-c623dedf-8cee-4303-9ec3-155bf611f55a"></span>

```text
请从具体基座的推理问题出发，先写清状态、条件信息、任务收益和计算约束，再选择监督、蒸馏、偏好优化或 RL。推导固定前缀 KD 梯度，区分当前学生前缀分布的额外导数；明确内部随机动作与文字动作的联合概率、硬阈值路由和强制停止的差别。按算法 H1/H2 解释缓存、轨迹重放、旧新参数、归约与同步顺序。原生与非原生两条路线分别定位，TaH2 专有行为按其源码核对，不预设唯一配方。
```
