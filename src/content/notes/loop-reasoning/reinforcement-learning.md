---
title: "强化学习与偏好优化"
description: "大模型 RL 后训练需要把奖励、概率与一次真实参数更新对应起来。本课从决策建模和策略梯度出发，比较 GAE、GRPO 与 RLOO 的优势估计，再连接 PPO、rollout、权重同步和评测。循环模型的路由概率在后半部分单独展开；DPO 与 OPD 分别提供偏好目标和教师反馈的对照。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_a04bfe2924d48191b95083d418d93368"
sourcePageId: "page_a04bfe2924d48191b95083d418d93368"
sourceSha256: 540080d5fb68391c41ca36ea3206774daf9eb6608fd7f39849d9958039ecbcd0
series: loop-reasoning
references: []
---

<span id="source-block-8e95766e-64b3-4e15-bfca-76cafa091ee6"></span>

大模型 RL 后训练需要把奖励、概率与一次真实参数更新对应起来。本课从决策建模和策略梯度出发，比较 GAE、GRPO 与 RLOO 的优势估计，再连接 PPO、rollout、权重同步和评测。循环模型的路由概率在后半部分单独展开；DPO 与 OPD 分别提供偏好目标和教师反馈的对照。

<span id="source-block-924c3835-7c26-44e5-95d1-077fb2018ba6"></span>

[返回后训练研究工作台](/notes/loop-reasoning/engineering/)

<span id="source-block-a218c42c-37e6-4051-b0e2-91e9f7e05441"></span>

> [!CALLOUT]
>
> **阅读顺序** 先读期望奖励、策略梯度与优势估计，再读四个策略分布和 PPO 目标；用算法 7 与算法 8 追踪一次更新。采样截断、随机路由和多轮执行可作为接入专题细读，最后比较 DPO 和研究选型。

<span id="source-block-8317e6e7-567c-4269-97a3-43fde73fbff2"></span>

## RL 全链路学习主线

<span id="source-block-5e247397-f3f6-4218-b961-f7844dbf2414"></span>

学习时同时追踪三条关系：奖励怎样影响动作概率，rollout 怎样决定训练数据分布，参数更新怎样返回生成后端。TaH2 提供内部计算案例，RL 提供自主行为的更新方法；具体实验再按问题选择算法与系统。

<span id="source-block-07c655b5-ec4f-498b-94e3-ac54dcaf807e"></span>

| 学习层次 | 核心内容 | 需要形成的判断 |
| --- | --- | --- |
| 决策建模 | 文字与工具动作、状态与观察、轨迹、终止和截断；内部路由是否也是动作 | 清楚定义优化的是哪一个策略，哪些状态和动作参与目标 |
| 策略梯度 | REINFORCE、return-to-go、baseline、梯度估计的期望与方差 | 能解释奖励怎样改变一个实际生成动作的概率，以及 baseline 的条件 |
| 价值与稳定更新 | critic、TD、GAE、bootstrap、PPO ratio／clip／value loss | 能比较偏差与方差；区分 PPO 近端更新与 reference KL 约束 |
| 组内优势估计 | GRPO、RLOO、组 ID、标准差归一化和无 critic 路径 | 用同一批回答比较 advantage 的定义、尺度及退化情形 |
| 反馈与奖励 | 偏好数据、reward model、RLHF、可验证奖励、outcome／process reward | 说明判分对象、误差、覆盖率与奖励投机；区分奖励模型训练和策略训练 |
| 在线数据分布 | rollout 行为策略、old anchor、当前 actor、reference；重复更新与重要性修正 | online 采样不自动意味着每次更新严格 on-policy；概率分母必须与实际数据来源对应 |
| 训练系统 | actor／critic／reference／reward 角色、rollout 引擎、队列、资源池、并行与权重同步 | 读通一次采样到下一次采样，定位每个张量、版本和状态的所有者 |
| 评测与扩展 | 熵、KL、clip fraction、长度、梯度、任务成功率和成本；多轮工具与异步陈旧度 | 判断奖励上涨是否带来目标能力提升，并能定位算法、数据或系统问题 |

<span id="source-block-04f53f62-8b32-4528-820f-0d1158f912d4"></span>

算法阅读顺序为 [REINFORCE 与 RLOO](https://arxiv.org/html/2402.14740v2) → [GAE](https://arxiv.org/abs/1506.02438) → [PPO](https://arxiv.org/abs/1707.06347) → [DeepSeekMath 的 GRPO](https://arxiv.org/abs/2402.03300)。GAE、GRPO、RLOO 放在“如何估计 advantage”的共同问题下比较；PPO 的 clipping 不能直接解释为严格保证 KL 上限。

<span id="source-block-8a336f4d-ba59-48fb-bde9-db098e41851a"></span>

反馈路线用 [InstructGPT](https://arxiv.org/abs/2203.02155)理解偏好收集、奖励学习与策略优化的连接，用 [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050)理解过程与结果监督的区别。过程评分模型可以用于选择或训练；读到评分效果改善时，先核对它用于哪一种流程。DPO 保留作偏好目标的对照，OPD 保留作教师反馈的交叉分支。

<span id="source-block-8ba07a35-4363-4493-88d7-b1738eb75906"></span>

## 从期望目标到梯度估计

<span id="source-block-77ff9212-cb01-4284-baae-a7df85b8acc0"></span>

### 自回归策略与期望奖励

<span id="source-block-dc311e78-5a31-46f6-9654-ce854a0c5f62"></span>

设模型已经生成一条回答，验证器只在结尾给出奖励 R。我们没有每个位置的正确标签，却仍希望提高更常得到高奖励的回答的生成概率。先把任务限制为普通自回归文本策略：词表有限，问题分布固定，EOS 也是动作，回答在 EOS 或最大时域 H 处结束，超时判分规则预先确定。终态奖励不直接依赖模型参数，策略可微，并满足交换微分与求和的支持条件。下面 t 表示从 1 开始的回答动作，先推导这个基本问题，再处理工具环境、参数相关奖励、随机内部路由以及 KL、critic 和 clipping。

<span id="source-block-b2afebce-9c64-45f3-a724-5ba003c2d870"></span>

```math
\begin{aligned}
s_t&=(x,y_{<t}),&
p_\theta(y\mid x)&=\prod_{t=1}^{T(y)}\pi_\theta(y_t\mid s_t),\\
J(\theta)&=\mathbb E_{x\sim\mathcal D,\ y\sim p_\theta(\cdot\mid x)}
[R(x,y)].
\end{aligned}
```

<span id="source-block-9d350944-99fc-488f-9462-7a7f9284b4ab"></span>

### 策略梯度的推导

<span id="source-block-af24e822-cde4-43be-8fb8-7ba534097858"></span>

一条完整回答的概率是各位置条件概率的乘积，取 log 后便成为各位置 log-probability 的和。虽然采到哪个离散 token 不是普通可微算子，我们仍可对每一条可能回答的概率求导：先展开期望，再用 ∇p=p∇logp，就得到下式。它给出的采样更新很具体——生成回答、取得终态奖励，再用这个奖励加权整条回答的动作 score。这里分配的是一份轨迹级估计，并不声称成功回答中的每个词都单独造成了成功。

<span id="source-block-1f9ab586-7989-4069-8416-e360df45b828"></span>

```math
\begin{aligned}
\nabla_\theta J
&=\mathbb E_x\sum_y R(x,y)\nabla_\theta p_\theta(y\mid x)\\
&=\mathbb E_{x,y}\!\left[
R(x,y)\sum_{t=1}^{T(y)}
\nabla_\theta\log\pi_\theta(y_t\mid s_t)\right].
\end{aligned}
```

<span id="source-block-00687e96-86de-4157-bfba-1f9e45191788"></span>

### baseline 的零期望恒等式

<span id="source-block-eca672bf-d8d0-4022-a5be-e18b2d5e0242"></span>

仅用原始奖励作系数，会把“这个前缀通常就容易得到的奖励”也带进动作更新。baseline 给出一个比较基准：这次回答的结果，比处在当前状态时通常预期的结果好多少？只要 b(s) 不按当前采到的动作取条件依赖，就可以从奖励中减去它而保持期望梯度不变。原因就在下式：给定状态后，所有动作概率之和为 1，所以 baseline 对动作 score 的期望贡献为零。

<span id="source-block-342d8d07-c4db-4ccb-9ade-8ca20a0e623b"></span>

```math
\begin{aligned}
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]
&=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)\\
&=b(s)\nabla_\theta 1=0.
\end{aligned}
```

<span id="source-block-1bf1a8e3-e088-4eeb-8c6d-b9e0bd580bb8"></span>

减去 baseline 后，正的 R−b 会沿梯度上升方向强化本条轨迹的动作，负的 R−b 则作相反更新；baseline 改变的是有限样本估计的波动，而不改变上面期望梯度。落实为 actor loss 时，把奖励、baseline 和估计出的 advantage 当作已经计算好的系数，停止它们的梯度。即使 baseline 与 actor 共用参数，这个 actor 更新也不顺着系数回传；训练 baseline 自身时，再使用另行定义的价值目标。

<span id="source-block-72907e57-018f-49cc-8efb-b46e361b4858"></span>

```math
\nabla_\theta J
=\mathbb E_{x,y}\sum_{t=1}^{T(y)}
\bigl(R(x,y)-b(s_t)\bigr)
\nabla_\theta\log\pi_\theta(y_t\mid s_t).
```

<span id="source-block-f1fb0b25-8a2e-4b55-b91e-65da2e861481"></span>

[REINFORCE 与 RLOO 原文](https://arxiv.org/html/2402.14740v2)用于对照轨迹回报和 baseline；下文将这个期望梯度、其有限样本估计与 PPO 代理损失分别记述。

<span id="source-block-73995e4f-79cb-4541-8c17-f84c4fbcf42b"></span>

### 价值估计与 GAE 的参考定义

<span id="source-block-5c23ca3f-3975-4513-b5dd-8057df7f57db"></span>

存在逐步奖励时，先定义价值函数和 advantage。令 T_star 为所定义任务的终止时刻，K 为本次采集片段的末端，K≤T_star。真正的任务终止取 bootstrap 为零；若只是采集片段截断且任务仍继续，则保留下一状态价值。若 H 本身被定义为有限任务的终点，在 H 处应按终端零值处理；有限时域价值还需包含剩余时域信息。

<span id="source-block-702eb875-3182-4998-87ed-05e39f38beeb"></span>

```math
\begin{aligned}
G_t&=\sum_{k=t}^{T_\star}\gamma^{k-t}r_k,\\
V^\pi(s)&=\mathbb E_\pi[G_t\mid s_t=s],\\
Q^\pi(s,a)&=\mathbb E_\pi[G_t\mid s_t=s,a_t=a],\\
A^\pi(s,a)&=Q^\pi(s,a)-V^\pi(s).
\end{aligned}
```

<span id="source-block-1e1bb32d-42f0-4153-ad3b-ea03c786ea1f"></span>

critic 用参数模型近似价值函数；V_old 表示本批次估计时冻结的价值预测。前面的无折扣终态目标对应 gamma=1；改折扣或奖励定义时，相应期望目标也需重新声明。

<span id="source-block-46d07ce9-ba05-454d-ba9e-58fcaf7cdda1"></span>

```math
\begin{aligned}
\delta_t&=r_t+\gamma(1-d_t)V_{\rm old}(s_{t+1})
          -V_{\rm old}(s_t),\\
\widehat A_{K+1}^{\rm GAE}&=0,\\
\widehat A_t^{\rm GAE}
&=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1}^{\rm GAE},
\quad t=K,\ldots,1,\\
\widehat U_t&=\operatorname{sg}
\bigl(V_{\rm old}(s_t)+\widehat A_t^{\rm GAE}\bigr),\\
L_V(\phi)&=\tfrac12\operatorname{Agg}_t
\bigl(V_\phi(s_t)-\widehat U_t\bigr)^2.
\end{aligned}
```

<span id="source-block-a7572c59-87c0-4e27-8c43-237161055989"></span>

sg 表示停止梯度。最后一行是价值回归的参考目标；具体配方可以另有 value clipping。GAE 的偏差与方差取决于价值估计、折扣与 lambda。源码中的额外 whitening、mask 递推和损失聚合需逐项核对，不能省略后仍声称与实现完全相同。

<span id="source-block-badb0598-50c4-47b9-90a3-79999acd4908"></span>

## 组内优势的数学定义

<span id="source-block-3c920ba1-3125-40ad-a364-456a120e7237"></span>

```math
\begin{aligned}
\bar R&=\frac1G\sum_{j=1}^{G}R_j,&
s_R^2&=\frac1{G-1}\sum_{j=1}^{G}(R_j-\bar R)^2,\\
\widehat A_i^{\rm GRPO}
&=\frac{R_i-\bar R}{s_R+\varepsilon},\\
\widehat A_i^{\rm RLOO}
&=R_i-\frac1{G-1}\sum_{j\ne i}R_j.
\end{aligned}
```

<span id="source-block-13d9fa26-f5b8-4bb4-896d-b6217d27c9c2"></span>

R_i 是同一问题第 i 条回答的终态奖励，G>1；这些系数广播到本回答的有效动作位置。RLOO 的留一 baseline 在同题条件独立采样等条件下，可用于无偏轨迹 score-function 估计；GRPO 的组均值与标准差含自身奖励，不能直接套用该无偏结论。固定实现的 torch.std 使用样本标准差。奖励 [1,0,1,0] 时，忽略 epsilon，GRPO 为正负约 0.866，RLOO 为正负 2/3；同组奖励全相同时，此 outcome advantage 为零，其他损失项仍可能更新参数。

<span id="source-block-1abd8306-0562-4cbb-bd77-9d8fef3cd1b1"></span>

## 四个策略分布与 PPO 代理目标

<span id="source-block-bc2e7532-af81-4847-85e3-74709b985ba0"></span>

| 数学对象 | 算法角色 | 工程记录 |
| --- | --- | --- |
| `mu` | 实际生成轨迹的行为策略，含采样变换 | rollout 版本、采样设置、行为概率 |
| `pi_old` | 当前批次更新期间冻结的近端基准 | detached old log-probability 与计算版本 |
| `pi_theta` | 正在更新的 actor | 每次前向重新计算的 log-probability |
| `pi_ref` | KL 等约束所用的参照策略 | reference 版本与概率定义 |

<span id="source-block-24a7dd9e-d195-41d8-8a4a-abf1674b43e9"></span>

### PPO 的固定样本代理目标

<span id="source-block-515223a2-96c6-4a18-ad39-c75e579d26b7"></span>

实际训练通常会保存一批回答，对它做多轮 minibatch 更新。第一轮之后，当前 actor 已经变化，而这些动作仍由旧策略生成；ratio 就用来比较“当前策略给这次动作的概率”和“生成基准给它的概率”。先看最小同步情形：行为策略等于冻结的 old 基准，以温度 1、无 top-k／top-p 截断的完整分布采样。此后固定动作、状态、old 概率、reference 和 advantage，只重新计算当前 actor 的 log-probability，便得到下式的 ρ。

<span id="source-block-56d32ab7-c7b7-4ef6-b8f2-04095cf67c58"></span>

```math
\begin{aligned}
\rho_{it}(\theta)
&=\exp\!\left(\ell^\theta_{it}-\operatorname{sg}(\ell^{\rm old}_{it})\right),\\
u_{it}(\theta)&=\rho_{it}\operatorname{sg}(\widehat A_{it}),\\
v_{it}(\theta)&=
\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)
\operatorname{sg}(\widehat A_{it}),\\
f_{it}(\theta)&=\min\{u_{it}(\theta),v_{it}(\theta)\},\\
L_{\rm clip}(\theta)&=-\operatorname{Agg}_{i,t}f_{it}(\theta).
\end{aligned}
```

<span id="source-block-554ea53e-f15a-4d03-b1c2-7c1aa6fd9c02"></span>

PPO 在这批固定样本上限制继续追逐同一优势信号：对正 advantage，把 ratio 推到 1+ε 以上不再增加这一项收益；对负 advantage，把 ratio 压到 1−ε 以下也不再获得额外收益。公式用原项与裁剪项的较小值实现这种单侧限制，最后取负号作为最小化损失。这是为反复利用样本而构造的代理目标，不是原始 J 的恒等改写，也没有给出严格 KL 上界。更新开始时，若当前策略与 old 基准及其概率计算规则一致，有效位置上的 ratio 应接近 1；这也提供了一个直接的接入检查。

<span id="source-block-865974c8-8ab2-48dc-9046-43dc42643250"></span>

### 聚合分母是算法的一部分

<span id="source-block-947638a2-edb9-4b4b-83f1-daf57ee7b6a4"></span>

令 m 为模型动作 mask，B 为回答数，T_i 为每条回答的有效动作数，N 为整批有效动作数，且所有 T_i 均大于零。下面三种聚合分别定义为：

<span id="source-block-70299705-ea3a-4e82-857d-51fda5677ce6"></span>

```math
\begin{aligned}
\operatorname{Agg}_{\rm seqsum}(f)
&=\frac1B\sum_{i,t}m_{it}f_{it},\\
\operatorname{Agg}_{\rm token}(f)
&=\frac1N\sum_{i,t}m_{it}f_{it},
\qquad N=\sum_iT_i,\\
\operatorname{Agg}_{\rm seqmean}(f)
&=\frac1B\sum_i\frac1{T_i}\sum_t m_{it}f_{it}.
\end{aligned}
```

<span id="source-block-6f17047d-4b38-451e-bf15-e044f7ef2c27"></span>

最后的 Agg 决定一条回答究竟以多大权重进入更新。seqsum 先把回答内的动作项相加，再按回答数平均；token 平均则把所有有效动作放在一起除以 N。在同一固定 batch 上，后者只是前者的 B/N 倍，但跨 batch 的回答长度变化也会改变这个尺度。seqmean 先除每条回答自己的长度 T_i，再按回答平均，因而还改变长短回答之间的相对权重。未裁剪且精确 on-policy 的轨迹 score 梯度，首先对应回答内动作梯度之和；工程上选择其他归约时，应明确是在改整体尺度，还是连样本权重也一起改了。

<span id="source-block-5c3254d0-41e6-458e-8026-b0c8391562ed"></span>

### 行为策略不同的情形

<span id="source-block-c27d1e6d-2c30-4eaa-95bb-e4213d237e53"></span>

```math
J(\theta)=
\mathbb E_{x\sim\mathcal D,\ y\sim\mu(\cdot\mid x)}
\left[
R(x,y)\prod_{t=1}^{T(y)}
\frac{\pi_\theta(y_t\mid s_t)}{\mu(y_t\mid s_t)}
\right].
```

<span id="source-block-933808ba-76c5-4b5b-9906-4ca084270f26"></span>

这个整轨迹重要性恒等式要求目标轨迹分布对行为分布绝对连续，并使用真实的行为条件概率。单个 token ratio 不会校正整个前缀分布。top-k 截断可能破坏支持条件；重新评分 old 概率不会改变既有轨迹来自谁。

<span id="source-block-dfea4eaa-6afd-4437-8d7d-365bed72cdf6"></span>

[固定 v1 的 old 概率实现](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py#L1645)有 bypass 与 decoupled 两条路径：前者复用 rollout 概率，后者另算近端基准。实现推导时先选定路径，再讨论陈旧度与修正。

<span id="source-block-834d0c5e-d88b-4f91-b0b0-eb381d9c98a6"></span>

#### 采样截断后的支持与行为概率

<span id="source-block-caa21a4f-ea39-4b1c-850c-9e446fb3b215"></span>

“行为 logp”还必须包括支持归一化。令 zθ 为当前路线的最终 sampler 输入，T>0 固定，Sθ 为温度、top-k、top-p 等实际变换产生的支持。对 y∈Sθ，采样策略及局部导数如下；导数只在路由、排序和支持保持不变的参数邻域内成立。若 y∉Sθ，行为 logp 为负无穷，不能以原模型的有限 logp 代替。[TaH 的采样变换、固定依赖与 T11](/notes/loop-reasoning/generation/)

<span id="source-block-0a7bc467-77b7-4349-a9c1-49a4daa0f44d"></span>

```math
\begin{aligned}
\log b_\theta(y)
&=\frac{z_{\theta,y}}T-\log\sum_{v\in S_\theta}e^{z_{\theta,v}/T},\\
\nabla_\theta\log b_\theta(y)
&=\frac1T\left[\nabla_\theta z_{\theta,y}
-\sum_{v\in S_\theta}b_\theta(v)\nabla_\theta z_{\theta,v}\right].
\end{aligned}
```

<span id="source-block-6588a8e8-cb75-4043-bb31-1700f102e1fe"></span>

这给出具体的接口检查：`raw` 模型 logp、温度缩放但未截断的 logp、最终 processed 行为 logp 是三个不同字段；最终归一化项一般会随参数改变。Loopie 固定 Megatron 路径还存在请求参数未完整透传、零温度被 `or 1.0` 改写以及默认返回 raw logp 的条件，详见[Loopie 实现页](/notes/loop-reasoning/loopie/)。不能仅凭请求 JSON 或字段名 `logprobs` 判定分母。

<span id="source-block-c220073e-bb9d-4670-a083-c4835c06ec39"></span>

正确保存行为概率，还不足以保证重要性加权能够恢复目标期望：目标策略可能给被采样器截掉的词分配正概率。固定一个共同条件历史，令实际采样分布为 b₀、支持为 S₀，目标策略为 πθ，并用 ℓ 表示目标落在 S₀ 外的概率质量。即使 b₀(y) 记录正确，重要性加权也只能覆盖目标在 S₀ 内的部分：

<span id="source-block-0a2fc8ca-a000-4dd8-abf6-9a0a83fb4ff4"></span>

```math
\begin{aligned}
\ell&=\sum_{v\notin S_0}\pi_\theta(v),\\
\mathbb E_{y\sim b_0}\!\left[\frac{\pi_\theta(y)}{b_0(y)}\right]
&=1-\ell,\\
\mathbb E_{y\sim b_0}\!\left[\frac{\pi_\theta(y)}{b_0(y)}f(y)\right]
&=\sum_{v\in S_0}\pi_\theta(v)f(v),\\
\left|\mathbb E_{\pi_\theta}f-
\mathbb E_{b_0}\!\left[\frac{\pi_\theta}{b_0}f\right]\right|
&\le M\ell\qquad (|f|\le M).
\end{aligned}
```

<span id="source-block-bf08e59e-06cf-4fac-bab8-e03059e177f1"></span>

当 ℓ>0 时，KL(πθ∥b₀)=∞；反方向 KL(b₀∥πθ) 在词表有限、且 πθ 在 S₀ 上为正时可以有限，因此正则项必须写清 KL 的方向。自归一化权重也无法补回缺失支持：样本量趋于无穷时，得到的是目标在 S₀ 上的条件期望。这些结论固定了共同历史；完整的 off-policy 序列更新还需处理前缀分布、内部路线与实际行为概率，不能只靠单 token 的比率。

<span id="source-block-2d52ee24-5757-4289-9493-6f5ba7480212"></span>

**接入时记录什么** rollout 需要保存解析后的采样参数、实际支持语义、返回概率类型与后端 revision；重评分则要声明目标是原始文字策略还是 processed 策略。

<span id="source-block-7b988d0d-f90f-420c-a105-03e8031e58b6"></span>

若固定前缀具有完整词表信息，可以计算 ℓ 和 T11 的支持变化质量。只有被采样词的 logp 时，无法重建 ℓ；裁剪 ratio、加 epsilon 或刷新 old logp 也不会恢复缺失支持。这里给出接入所需的接口和数学条件，TaH2 的 RL 接入仍需实现。

<span id="source-block-5cb32580-6be7-484c-b983-7a5f564af32a"></span>

### KL 估计值与反传梯度

<span id="source-block-9564e6e7-5fab-4ff5-a836-4ce77505c9b0"></span>

为什么 KL 的采样估计还要单独检查梯度？固定一个状态和一个概率为正的参照分布 q，下面第一行是 reverse KL 的真实梯度；第二行则先把采样动作固定，再对 log-ratio 直接求导，最后求期望。它们一般不同。

<span id="source-block-978a65e6-b7e5-460d-91b2-cecbfd415679"></span>

```math
\begin{aligned}
\nabla_\theta{\rm KL}(\pi_\theta\Vert q)
&=\mathbb E_{a\sim\pi_\theta}
\left[
\log\frac{\pi_\theta(a)}{q(a)}
\nabla_\theta\log\pi_\theta(a)\right],\\
\mathbb E_{a\sim\pi_\theta}
\left[\nabla_\theta\log
\frac{\pi_\theta(a)}{q(a)}\right]&=0.
\end{aligned}
```

<span id="source-block-7f1b637f-5777-44a1-bc53-8456b6c819c7"></span>

因此“采样估计的数值在期望下等于 KL”不足以保证自动微分给出 KL 梯度。KL 放入 detached reward，或作为可微 loss 直接反传，需要各自声明估计器和梯度路径。参考算法先关闭这些附加项，之后逐项加入。

<span id="source-block-fd7b90f3-94a8-4f7e-99dc-8a9a0257bae5"></span>

二动作例中，若当前概率为 0.7／0.3、reference 为 0.4／0.6，以第一个动作的 logit 为参数，真实 KL 梯度约为 0.26308，而固定动作 log-ratio 的直接微分在当前策略下的期望为 0。这个例子检验的是梯度路径，不是估计器的数值是否接近 KL。

<span id="source-block-47b88663-2e25-42a8-aea4-ea61870d7379"></span>

## 一轮 RL 的执行流程

<span id="source-block-1c510166-43d3-41ea-b2ca-5825c55a3ac8"></span>

| 顺序 | 数据或操作 | 固定源码入口 |
| --- | --- | --- |
| 1 | 初始化角色、资源和轨迹队列 | [main_ppo.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/main_ppo.py) 的 `TaskRunnerV1.run` 与 [trainer_base.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_base.py) 的初始化 |
| 2 | 学生生成回答或多轮工具轨迹，放入队列 | [agent_loop_tq.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/agent_loop_tq.py) 的 `generate_sequences`、`_run_prompt`、`_agent_loop_postprocess` |
| 3 | 取训练批次、取得奖励并平衡 DP 负载 | `PPOTrainer.fit → step → _step_once`；奖励可能已在 agent loop 算好，或由 `_compute_reward_colocate` 补算 |
| 4 | 取得 old／reference 概率及可选 value | `_compute_old_log_prob`、`_compute_ref_log_prob`、`_compute_values` |
| 5 | 优势估计、可选 critic 更新与 actor 更新 | `_compute_advantage → _update_critic → _update_actor`；具体公式见 [core_algos.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py) |
| 6 | 同步参数并进入下一次采样 | [trainer_sync.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_sync.py) 的 `on_step_end` 调用 checkpoint manager 的 `update_weights`，再追到对应 worker 与 rollout loader |
| 7 | 验证与恢复 | `trainer_base` 的 `_validate`、`_save_checkpoint`、`_load_checkpoint`、`_reissue_inflight_prompts` |

<span id="source-block-fba343ab-a9a8-4c2c-be3d-c1aa2e819a8c"></span>

固定 verl 提交为 `7cea5392165ac26d5eadf879701337c85e4a2b2f`。[配置](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/config/ppo_trainer.yaml)默认使用 v1 与同步模式；当前主入口是 `TaskRunnerV1.run`，旧 `RayPPOTrainer.fit` 保留作同步控制流对照。具体采用哪种 advantage、critic、KL 和 loss 聚合，按展开后的配置核对。[DeepSeekMath](https://arxiv.org/abs/2402.03300)提供 GRPO 的原始算法入口。

<span id="source-block-402e778c-313f-43e4-81be-48c0bca6bf22"></span>

### Loopie 的 token 目标与序列目标

<span id="source-block-2d8aaf5f-6c58-43ee-ad7f-4a83f71c7ae6"></span>

论文 §4.2 使用 GSPO；固定 Megatron 公开入口自称 GRPO 示例，`calculate_grpo_loss` 对每个 token 的 exp(current−old) 分别 clip。它还通过 MegatronLocal 的独立 inference model 生成，不能从存在作者 vLLM fork 推断此入口已经调用了该 fork。[论文](https://arxiv.org/html/2607.16051v3) · [公开 RL 示例](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/examples/rl/README.md) · [实际 RL 调用链](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)

<span id="source-block-08ac9372-cccb-417b-acbc-9b3de08f2539"></span>

若实现论文的序列级目标，令 m 为响应位置 mask、nᵢ 为有效响应长度、Aᵢ 为 detached 的组优势；应先汇总同一响应的 log-ratio，再指数、裁剪，最后按响应平均。下面 J 为要最大化的 surrogate，训练 loss 取 −J。

<span id="source-block-015d9513-8fd0-4159-87a9-f8e0f6a11d87"></span>

```math
\begin{aligned}
n_i&=\sum_tm_{i,t}>0,\\
s_i&=\exp\!\left[\frac1{n_i}\sum_tm_{i,t}
\bigl(\log\pi_\theta(y_{i,t})-\log\pi_{\rm old}(y_{i,t})\bigr)\right],\\
J_{\rm seq}&=\frac1B\sum_i\min\!\left(
s_iA_i,\operatorname{clip}(s_i,1-\epsilon_l,1+\epsilon_h)A_i\right).
\end{aligned}
```

<span id="source-block-ec19910c-f5d0-4445-8feb-15a5011c5f1a"></span>

packed layout 中完整序列长度包含 prompt，不能用它代替 nᵢ；next-token shift 后的 mask、response ID、跨序列边界和 CP 切分后的可微分子归约都要保持一致。把同一个序列 loss 复制到每个 token 再直接取 token 均值，会重新增加长响应的权重。例子 ratio=(2,1/2)、A=1、clip 区间 [0.8,1.2] 给出 token surrogate 0.85、序列 surrogate 1.0，其 log-ratio 梯度分别为 (0,0.25) 与 (0.5,0.5)。完整推导、逐函数 shape 和 mask 见[Loopie 专页](/notes/loop-reasoning/loopie/)。

<span id="source-block-724e2cb7-ec50-4439-b83a-607614476017"></span>

动态组筛选还有实际运行边界：固定 `_GranularityConfig._validate` 拒绝 `filter_groups_with_same_reward=True`，因为当前丢组不再生成，可能无法交付调用方要求的完整 batch。论文式 DAPO 补样需要另建“采样→计算组奖励→筛选→不足则补样”的完整循环，记录生成组数与保留组数；不能只打开现有 flag。这个结论限定于已核公开路径，不用于判断作者未公开的历史 launcher。[真实校验器](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/api.py)

<span id="source-block-8f70e5be-d5a3-4e83-852f-7c0f7e4db981"></span>

## 算法 7 一次同步策略更新

<span id="source-block-b820d163-b9f4-45cb-a01a-f8448516c605"></span>

下面限定终态奖励、组内 advantage、无 critic、无 KL 的参考分支，使用上文定义的 clipped surrogate。每题独立采样 G 个回答，G 至少为 2；若有 P 个问题，张量中的回答行数 B=P×G。普通非循环模型先提供概率参考；TaH2 接入时另核对训练与部署的路由和读出语义。伪代码是语义参考，不是完整 verl 启动脚本。

<span id="source-block-87aacb9f-bb54-44f0-b061-d4daee4b82fe"></span>

**输入** 当前参数 theta，问题批次 X，组大小 G，最大时域 H；epoch 数 E，clip 参数 epsilon，明确指定的 Agg

<span id="source-block-6477b123-1d30-4af5-a07e-e4720928624b"></span>

**输出** 新参数 theta_new、冻结轨迹 D、目标与版本记录

<span id="source-block-794edc68-1d92-4892-a452-8fceca723824"></span>

**前置条件** 算法 7 的文字策略评分采用温度 1、无截断采样和已声明的有限时域任务：EOS 是有效停止动作，超时按该任务的终止与奖励定义处理。旧策略的评分模式、tokenizer、mask 与生成版本必须一致。随机路由若影响后续状态，需改用完整联合动作日志或正确边缘概率评分，不能只在 `Score` 外加一个未校正的 gate 概率。

<span id="source-block-be6fc7ac-37da-41f4-8b80-6d1f579e980d"></span>

<figure class="research-embed"><iframe src="/research-assets/f177924f9c8092609e54738c2db7391c2196b03c093a90c211062044694148da.html" title="算法 7 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 7 操作与数学公式 · <a href="/research-assets/f177924f9c8092609e54738c2db7391c2196b03c093a90c211062044694148da.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-54262d34-2f3b-4cb9-aff2-bc65d4a53926"></span>

若 DP 后端把各卡梯度取平均，而各卡损失已除以整个 DP 组的全局分母，则本地 backward 前需补偿 DP size；若后端已经采用求和语义，则不再补偿。TP 卡共享同一份数据，不能把同一批 token 在分母中重复计数。具体归约顺序按后端验证。

<span id="source-block-964e5fc5-ed65-47de-a8f1-046f3e47fa67"></span>

### 算法 8 动作对数概率评分

<span id="source-block-4d7a217a-3197-4b3c-a21e-14424bfc882f"></span>

**过程** Score(theta, D)

<span id="source-block-33f61874-d98a-4bd0-88b2-455b842ebeca"></span>

**输入** 参数 theta、实际轨迹 D，以及对应的 attention mask、问题边界与动作 mask。

<span id="source-block-05d56897-69ab-4a15-a4ef-62047a78fdf7"></span>

**输出** 回答动作对应的对数概率 lp，形状为 \[B,T\]；梯度模式由调用方决定。

<span id="source-block-68188522-552e-4a7c-843b-fe7bac609884"></span>

**前置条件** 算法 8 仅规定原始词表 logits 的确定性评分路径。若随机路由开启，需明确条件联合评分还是文字边缘评分；重算一次随机路线不等于重放原路线。先核对实际 decider 的随机选项与 warmup，再设置梯度模式；`no_grad` 和 `eval` 各自都不是完整的概率语义说明。

<span id="source-block-8b671561-a3c0-4aa6-b3df-707b92998d96"></span>

<figure class="research-embed"><iframe src="/research-assets/767b020a456cf16c97c39d05451d0baa489293a0b91144fe8a4f683771c8e900.html" title="算法 8 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 8 操作与数学公式 · <a href="/research-assets/767b020a456cf16c97c39d05451d0baa489293a0b91144fe8a4f683771c8e900.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-22d2ea55-2ff7-480a-aa34-fd90cb304d38"></span>

第一个回答 token 由问题最后位置的输出预测，mask 与抽取索引必须随这次因果移位一起对齐。以上 Score 假设模型输出原始 logits；若输出已经是归一化混合 log-probability，可直接查询相应 token。若输出是总质量 Z 不等于 1 的原始 log-mass，PPO/DPO 评分还需匹配实际 sampler 的归一化，保留 log Z。target-only 的最后一维为 1，不能当成完整词表再做 log-softmax。

<span id="source-block-b47b2d5c-3d60-4c33-95c1-38eb790ec781"></span>

Score 本身不采样，但仍需固定概率计算模式。禁用 dropout 或路由 warmup 与禁用梯度是不同操作，`no_grad` 不能代替模式核对。[混合累计与输出](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520)

<span id="source-block-df09646a-b26f-4c12-8f15-6c91d8214cc4"></span>

| 量 | 形状与类型 | actor 更新中的梯度 |
| --- | --- | --- |
| token IDs 与 group ID | 整数；`[B,T]` 与 `[B]` | 无 |
| 动作 mask | bool `[B,T]`；prompt／padding 不计分 | 无 |
| reward 与 advantage | 浮点 `[B]` 或 `[B,T]` | detach 后作为系数 |
| old 与 reference logp | 浮点 `[B,T]`，版本固定 | 无 |
| current logp | 浮点 `[B,T]`，每次更新重算 | 有 |
| ratio 与局部 loss | 有效动作位置；归约尽量 FP32 | 传到 current actor |

<span id="source-block-0173d9b9-714e-4688-83eb-679f855bd347"></span>

[PPO 原论文](https://arxiv.org/pdf/1707.06347)给出代理目标和多轮 minibatch 更新；[固定 core_algos.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py)落实 advantage、clipping 与聚合。真实配置中的额外校正、dual clip、数值 clamp、critic、熵和 KL 都是需要单独写入算法的扩展。

<span id="source-block-a9467514-0bb6-4d00-b143-ec035e9a78ca"></span>

## 奖励与概率的接口

<span id="source-block-5a69e25f-bbb9-44d7-8339-80cdd93ade10"></span>

验证器至少返回分数、是否成功解析、是否完成验证以及错误类型。数学答案判定、代码执行和格式奖励分开记录。验证器失败应计入覆盖率；仅在协议明确规定时，才将特定失败纳入任务奖励。对任务正确率、格式合规率和验证器覆盖率分别看结果，防止模型只学到格式。

<span id="source-block-3a518e20-cf7c-4988-9ebf-8546961ee780"></span>

每条 rollout 记录 prompt ID、group ID、response token、response mask、EOS 或截断、生成权重版本、采样配置和所用 log-probability 的定义。温度、top-k、top-p 会改变采样分布；“参数同步”本身不足以说明采样概率与训练端概率完全相同。

<span id="source-block-7c70b88e-d842-48b3-8d7b-9784d92d3af6"></span>

TaH2 的策略概率还取决于内部循环、混合读出和执行模式。论文主配方采用确定阈值路由，仓库的 `Qwen3MLPIterDecider` 另有 `stochastic_sampling` 选项。其执行顺序是 max_iter 强停、训练态 warmup、随机采样；随机分支本身没有 `self.training` 限制，因此启用该选项后，仅调用 eval 仍不能保证确定性。使用随机 gate 时，需要明确优化文字边缘策略还是路由—文字联合策略，并保存对应动作与概率。下文给出这种接入所需的数学定义；完整联合策略梯度仍属于待扩展功能。[实际分支顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L199)

<span id="source-block-d799f869-b088-4092-89b2-91823e38ad33"></span>

### 随机路由历史与文字边缘概率

<span id="source-block-4c62e674-b245-46b2-9191-11fb039a9ec3"></span>

循环模型还多了一种选择：同样输出文字 y，两次运行可能经过不同的内部路由。若奖励和目标关注的是文字输出，我们需要的是“所有能够生成 y 的路线加起来有多大概率”；若连内部动作一起优化，则要保留本次实际走过的路线。令 a 包含全部内部随机路由，π 为路由与文字的联合概率，p 为对路由求和后的文字策略，就得到下式。这里假定时域与离散路由空间有限，隐藏表示及 KV 按对应参数和完整历史重建；max_iter 强停是概率 1 的事件，不再额外分配一个自由 Bernoulli 停止概率。

<span id="source-block-23a023df-c7e4-45fa-bdfc-d60ac80aaf38"></span>

```math
p_\theta(y)=\sum_a\pi_\theta(y,a),\qquad
\nabla_\theta\log p_\theta(y)
=\mathbb E_{a\sim\pi_\theta(\cdot\mid y)}
\left[\nabla_\theta\log\pi_\theta(y,a)\right].
```

<span id="source-block-d7dafc97-a965-482e-a7b1-a058488cd9b5"></span>

对 p(y) 求导时，每条路线并不是按执行前的先验概率参与更新，而是按“已经看到 y 后，这条路线对生成 y 有多大贡献”来加权。把有限和的导数整理为 log-probability 的导数，就得到这个路由后验责任权。它与离线混合读出中的责任权是同一类求导关系，但这里路线可以影响整条文字轨迹。一次运行返回的条件 token log-probability 只描述那条路线；随机 gate 的概率和其他路线的求和还需在相应目标中处理。若改用连续随机量，则以积分替代求和，并检查微分与积分交换的条件。

<span id="source-block-f273eedd-9877-4497-ae34-bdd30992c793"></span>

缓存会把历史路由影响带到后续 token，不能每个 token 都重置为独立路由先验。考虑一个整条轨迹共享的隐藏变量 Z，两个值先验各为 1/2；给定 Z 后，每个位置输出 token 1 的概率分别为 0.9 和 0.1。于是：

<span id="source-block-6d6b8fa4-5f78-448f-9e46-adbd1ea095ab"></span>

```math
P(11)=\tfrac12(0.9)^2+\tfrac12(0.1)^2=0.41,
\qquad
P(1_2\mid1_1)=0.82.
```

<span id="source-block-387d90ca-b029-400f-bdb2-f4d473444e1d"></span>

若每个位置重新使用 0.5/0.5 先验，乘积会变成 0.25。TaH2 中是否存在这种跨 token 的路由影响，应按实际保留的 KV 与回放语义判断；该例说明需要核对什么，不声称源码使用此二状态模型。

<span id="source-block-a8ea7d90-cbeb-4404-a25a-f0c3d78e2eff"></span>

### TaH2 随机分支给出一个可手算的策略评分例子

<span id="source-block-17a9ddc5-e457-452e-a9a2-693d25553fd6"></span>

[单 token 条件模型](/notes/loop-reasoning/internal-loop/)可以把这件事算清。最大深度为 2，继续率为 g：以 1−g 的概率停在第一轮，读出 q₁；以 g 的概率执行第二轮，读出却仍是 P₂=(1−g)q₁+gq₂。后一条路线中的 g 既决定是否执行，也参与混合读出。所以，先用路线概率乘条件读出，再把两条路线相加，最终边缘中 q₂ 的权重会出现 g²。这个平方来自概率组合，不代表第二轮执行了两次。

<span id="source-block-7cfc237f-a429-407a-b298-e453427aae51"></span>

```math
\begin{aligned}
\pi(D=1,v)&=(1-g)q_1(v),\\
\pi(D=2,v)&=g\bigl((1-g)q_1(v)+gq_2(v)\bigr),\\
\bar P(v)&=(1-g^2)q_1(v)+g^2q_2(v).
\end{aligned}
```

<span id="source-block-efb6cc36-84cd-4846-8222-c64166dc8ae5"></span>

现在只问 gate 本身应怎样改变。暂时固定 q₁、q₂，令 a 为 gate logit、g=sigmoid(a)，并要求观察词 y 的边缘概率 P̄(y)>0。把前式对 a 求偏导，便得到下面的局部文字 score；它明确显示，门控收到的信号来自两轮对同一词 y 的预测差异。

<span id="source-block-1d2f4ae5-7a23-441e-8c8c-5a2c11c854e9"></span>

```math
\frac{\partial\log\bar P(y)}{\partial a}
=\frac{2g^2(1-g)\,[q_2(y)-q_1(y)]}{\bar P(y)}.
```

<span id="source-block-ad2d3fd4-7a30-48c5-8290-5023646522e9"></span>

当 q₂(y)>q₁(y) 时，这个偏导为正：沿这条局部路径提高 gate logit，会提高观察词 y 的边缘概率；两轮的预测相同时，这条读出路径便没有理由偏向更多循环。若计算联合 score，还要包含 log Pr(D)；若计算文字边缘 score，则按观察到 y 后的路由后验加权。回到骨干参数的更新，还须合并 q 与 g 对共享状态的依赖。此处应用 Fisher 恒等式分析的是可选随机分支中的采样与混合读出，论文主配方仍使用确定阈值路由。[固定 sample 与混合分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

<span id="source-block-7ab5c247-31f0-4f1b-85e7-ce6d2e826332"></span>

计算成本则按实际执行另算。第二轮只以概率 g 发生一次，所以平均轮数是 1+g；若每多一轮成本为 c，期望成本对 a 的导数就是 c·g(1−g)。读出中的 g² 和成本中的 g 因而回答两个不同问题：前者描述 q₂ 对文字分布的贡献，后者描述额外计算发生的频率。做成本惩罚的 RL 或蒸馏时，应先确定联合动作目标、文字边缘目标还是固定路线条件目标，再分别实现对应奖励、成本和概率评分。

<span id="source-block-fa7345eb-0a1a-4661-99e1-210e3930a599"></span>

### 同一个 PPO ratio 或 KL 名称可能对应不同目标

<span id="source-block-13ca3b84-3d4b-4ee6-a041-9f36b00b489a"></span>

重要性比也必须与优化对象一致。令行为联合分布为 μ，并要求目标联合分布位于行为分布的支持内。若比较的是文字边缘概率，就要将联合比按“行为策略在看到 y 后的路由后验”取平均：

<span id="source-block-752d8037-6b20-481a-a311-7e17e802a8b4"></span>

```math
\frac{p_\theta(y)}{p_\mu(y)}
=\mathbb E_{a\sim\pi_\mu(\cdot\mid y)}
\left[\frac{\pi_\theta(y,a)}{\pi_\mu(y,a)}\right].
```

<span id="source-block-f98136f5-c933-4f2c-af63-7ac108141446"></span>

例如旧路由概率为 (0.5,0.5)，新路由为 (0.9,0.1)，两分支对某个 y 的条件概率仍为 (0.9,0.1)。单看条件 decoder 概率，比值为 1；联合比为 1.8 或 0.2；正确文字边缘比是 1.64。三者不能互换，PPO 的裁剪又是非线性操作，先对比值取平均与逐路由裁剪也一般不交换。

<span id="source-block-a82ffb29-dd10-4403-9612-9e124836e70e"></span>

```math
D_{\mathrm{KL}}(\pi_\theta(y,a)\Vert\pi_{\mathrm{ref}}(y,a))
=D_{\mathrm{KL}}(p_\theta(y)\Vert p_{\mathrm{ref}}(y))
+\mathbb E_{y\sim p_\theta}
D_{\mathrm{KL}}(\pi_\theta(a\mid y)\Vert\pi_{\mathrm{ref}}(a\mid y)).
```

<span id="source-block-21a55b77-dc5f-4deb-ac2c-5736c8d90814"></span>

这条 KL 链式分解把两种约束分开：文字 KL 约束输出文字，额外的条件 KL 约束同一文字由哪些内部路由产生。因此，只有文字偏好标签时，DPO 若改用某一条联合轨迹的 log-probability，就改变了训练目标。确定阈值路由还有支持变化的问题：参数变化可能改变可发生的联合路径，不能直接套用要求共同支持的 Bernoulli 重要性采样公式；其文字概率则可按当前参数重新执行前缀计算。

<span id="source-block-5a71b4fa-4ff6-4008-8860-1d0f016ea983"></span>

数值核验使用上述二分支离散分布：联合 KL 为 0.368064，文字 KL 为 0.221754，条件后验 KL 期望为 0.146311；链式分解误差约 5.6×10⁻¹⁷。边缘 score 恒等式的有限差分误差小于 1.3×10⁻¹⁰。这些是公式检查，未运行策略优化或 TaH2–verl 集成。

<span id="source-block-9fd96431-5654-468c-9b80-7d7f29524287"></span>

### 算法 T6 小型路由树的精确文字评分

<span id="source-block-ff77cd3b-6c20-487d-b2c7-94156abd2367"></span>

**过程** ExactRouteMarginalScore(observed_tokens, model, routing_semantics)

<span id="source-block-528564ab-757f-4ce2-96af-b41f451c5229"></span>

**输入** 一条已观察文字轨迹、当前模型参数、有限离散路由及其概率、最大文字长度与最大内部轮数；起始缓存和采样语义已声明。

<span id="source-block-86fa7bf0-36a9-4316-b18b-947bb76be670"></span>

**输出** 文字边缘 log-probability、每步路由历史后验及支持诊断；在可导分支内保留完整计算图。

<span id="source-block-702701d7-f418-4bbc-95f8-e5c05a4e3a6f"></span>

**前置条件与终止** 路由空间有限，条件读出为归一化概率，分支状态包含所有影响未来的缓存；观察到 EOS 后，按声明的吸收语义处理。文字前缀相同，并不足以合并内部状态。

<span id="source-block-d4717731-6bde-4509-8bf0-d82ad2a37446"></span>

算法最多处理 H 个文字位置，每个位置枚举有限内部路径，因此可以终止；最坏成本随时域指数增长，只适合小型语义参考。起始前沿必须包含所有先验随机状态：若 prompt/prefill 也执行随机路由，须纳入相应路径及概率，不能用任取的一个生成前 KV 代替其分布。

<span id="source-block-ff7ecbac-1bb0-4aaa-b047-fd40ead8c9f8"></span>

<figure class="research-embed"><iframe src="/research-assets/953491db5930189988cb02c14e42104cfbd8648ccb00469adce14c9282f0a2e9.html" title="算法 T6 小型路由树的精确文字评分" loading="lazy" sandbox=""></iframe><figcaption>算法 T6 小型路由树的精确文字评分 · <a href="/research-assets/953491db5930189988cb02c14e42104cfbd8648ccb00469adce14c9282f0a2e9.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-47fc4364-1b09-40be-a484-be097bd20ad8"></span>

**函数来源与实现状态** 本算法依据边缘化定义自行推导，不是现有 TaH2 的服务实现。实际分支概率与强制停止须沿 `Qwen3MLPIterDecider.forward` 核对，分量读出沿 `TaHRuntime.accumulate_iter / finalize` 核对；生产随机路由的近似推断与日志协议需要独立实现。[decider](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L190)、[runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520)。

<span id="source-block-2cff48aa-88ac-47d4-b894-4e4a0dc85002"></span>

## 在线分布与多轮执行

<span id="source-block-b07b8fdf-c15b-4752-88fc-7161f1095342"></span>

行为策略是生成数据时实际使用的策略；old policy 是某次更新的近端基准；当前 actor 正在被更新；reference 为 KL 等约束提供参照。它们的版本可能不同。采样温度、截断支持、引擎数值差异和重复更新都会影响概率对应；是否需要重加权取决于具体目标与采样协议。

<span id="source-block-173e4662-1628-4a07-b51e-1c2a63629419"></span>

在 [agent_loop_tq.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/agent_loop_tq.py) 的 `_agent_loop_postprocess` 中，`loss_mask` 接收 `response_mask`，轨迹还记录生成版本的最小值与最大值。多轮学习需从 agent loop 核对模型动作、工具观测和 padding 的边界，并按终止原因处理截断与 bootstrap；不能把全部有效上下文都当作模型动作计分。

<span id="source-block-ee57a7ca-a235-4b23-901f-b35724b2b0ab"></span>

先读同步的 [trainer_sync.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_sync.py)，再读 [trainer_separate_async.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/v1/trainer_separate_async.py) 的分离执行。异步学习增加队列供需、轨迹陈旧度、同步频率和在途任务恢复问题。队列里的 ReplayBuffer 与模型内部历史 KV 的 replay 是不同层次的数据与状态。

<span id="source-block-4450c20a-d3f1-4b06-bb73-c006a4bc9a0e"></span>

## 与循环记忆和数据配比的研究连接

<span id="source-block-aa497d4a-23fa-47f2-8fd7-29f9c461f6b2"></span>

以下是可检验的研究切口，学习这些工具后再决定实验，不预设 RL 优于监督或蒸馏。

<span id="source-block-fa2a54a1-0fb7-41ec-9bff-a82edaee1e02"></span>

| 研究问题 | RL 提供的建模工具 | 仍需检验的内容 |
| --- | --- | --- |
| 内部循环怎样获得任务信号 | 在学生真实生成路径上按最终任务收益更新策略 | 收益是否来自内部计算，还是文字长度、采样或其他行为变化 |
| 门控何时继续计算 | 用任务收益与实际计算成本定义目标，比较停止与继续的价值 | gate 的梯度、成本测量、可部署策略及公平预算；局部 CE 改善与最终任务收益分别评价 |
| 中间状态与记忆怎样续用 | 用后续多个操作的完成情况评价保存、读取和更新行为 | 状态是否可复用；通过替换、读取干预及匹配成本排除捷径 |
| 数据配比怎样影响学习 | 分析各来源的采样量、有效 advantage、长度和梯度贡献 | 样本配比与有效训练权重的差异；是否需要 RL，还是固定采样或 bandit 已足够 |
| 少量文字与 latent 怎样协同 | 将文字动作与内部计算的目标、信用分配分别定义 | 能力与成本是否改善；不能由总奖励提高直接证明 latent 的语义机制 |

<span id="source-block-c1bcc284-3359-4e1e-af8a-076f840dc91d"></span>

门控的候选目标可写为期望任务收益减去计算成本，但先要定义成本单位、动作粒度和梯度估计。硬阈值路由、连续混合权重与随机深度策略分别处理；若优化随机路由，必须给出与文字动作一致的联合策略或合法边缘化方案。[PonderNet](https://arxiv.org/abs/2107.05407)提供自适应计算的对照设计，其目标与标准语言策略 RL 的关系仍需单独分析。

<span id="source-block-1ffb526b-ec65-44ce-90e5-b5fc4d2231e6"></span>

## DPO 作为另一条独立分支

<span id="source-block-aa129a0c-715f-4a67-8ecc-e8955ea298f7"></span>

DPO 从同一问题的 chosen 与 rejected 回答构造相对目标。[原论文](https://arxiv.org/abs/2305.18290)及 [TRL DPOTrainer](https://github.com/huggingface/trl/blob/1acd8adceffa285dcabd1f54b08e0830bea0279a/trl/trainer/dpo_trainer.py) 作为参考；TRL 固定提交为 `1acd8adceffa285dcabd1f54b08e0830bea0279a`。先读 `compute_ref_log_probs`，再读 `_compute_loss` 的标准 `reverse_kl + sigmoid` 分支。

<span id="source-block-5aad9686-800c-4f29-a54a-17dc12689959"></span>

```math
\begin{aligned}
\Delta_\theta&=
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\rm ref}(y^+\mid x)}
-\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\rm ref}(y^-\mid x)},\\
\mathcal L_{\rm DPO}&=-\log\sigma(\beta\Delta_\theta).
\end{aligned}
```

<span id="source-block-0986981e-9802-4fb5-a8ba-02e900076e95"></span>

上述标准分支使用 completion token 的 log-probability 之和。更改为长度平均会改变目标，不能只当作数值缩放。偏好来自人、规则或教师，也应固定来源与版本。DPO 能训练相对偏好；是否改善内部续算能力仍需独立证据。

<span id="source-block-0efac162-1267-464b-9966-49216eb18a9b"></span>

## 按研究需要选择

<span id="source-block-30c3123e-ca9b-4f59-908b-ed9be580866c"></span>

| 当前问题 | 优先使用的路径 |
| --- | --- |
| 先理解与修改循环模型 | 先完成 SFT 与自主生成闭环 |
| 教师能在学生错误前缀给密集反馈 | 接入 OPD，核对直接梯度与 PG 分支 |
| 结果可可靠验证，希望改进自主行为 | RLVR，先固定 verifier 和 rollout 协议 |
| 已有稳定的成对偏好 | DPO，核对 reference 与长度处理 |
| 希望门控节约计算 | 明确深度监督或成本目标，另测收益与计算开销 |

<span id="source-block-60df336b-dbe5-4feb-9a25-5bffff084bdb"></span>

[Marin Quick RL loop](https://github.com/marin-community/marin/blob/f5e8a9cfe9d9d50b79fd03e4ed0ed0087edf627d/docs/recipes/quick-rl-loop.md)补充固定任务选择、初始化、训练和前后评测的工程组织。第一版沿同步流程学习；异步队列和大规模服务留待出现实际吞吐需求后展开。

<span id="source-block-9a01fee0-f5fa-46dc-9042-8e66d522a4d0"></span>

## 交付与验收

<span id="source-block-2a7d651b-52a4-4c8c-9a4d-e9639211073a"></span>

围绕同一套小型可验证任务完成一个工程案例：先固定 SFT 起点、问题分组和采样协议，保存 rollout 后离线重算目标，再闭合连续两次同步更新。接着选择一个研究变量，如 advantage 估计或计算成本目标。两张 A100 的模型规模、角色驻留与上下文长度按显存测量选择；critic 和奖励模型分支另做资源核算。

<span id="source-block-66450991-b85a-49bf-9349-36f37604f0f7"></span>

- [ ] `rl_role_and_batch_trace.json`：actor、行为版本、old anchor、reference、可选 critic／reward 的职责、输入与输出完整。

<span id="source-block-270a2fb8-64fb-4dad-bba6-09438582312a"></span>

- [ ] `rl_learning_diagnostics.json`：奖励、成功率、熵、KL、clip fraction、长度、梯度与实际成本可联合检查，保存所用聚合定义。

<span id="source-block-6e148f9b-9cb8-42d8-b463-f2692f1eb08b"></span>

- [ ] `rollout_trace.jsonl`：一次采样到下一次采样的版本、group、mask 和概率定义完整。

<span id="source-block-4b260992-015e-4949-ab64-69b0a5dc1797"></span>

- [ ] `reward_audit.json`：正确、错误、解析失败与执行失败分别统计。

<span id="source-block-52a19780-558b-4eef-a658-8d303e8e2509"></span>

- [ ] `objective_trace.json`：从一组回答重算奖励、advantage、ratio 和各损失分量。

<span id="source-block-b090e28f-e135-4169-bcf4-7f4ad1b2342b"></span>

- [ ] 连续两次 optimizer step 后检查权重重载和 actor／rollout 概率；在目标精度与后端误差范围内解释差异。

<span id="source-block-48679b46-98a3-4108-afa6-ab0ce962a2ad"></span>

- [ ] 若选择 DPO，可用固定一对回答重算标准损失，并核对 reference 和 completion 边界。

<span id="source-block-ba43e964-05d6-4818-a3d5-f8a74f6647ba"></span>

学习范围覆盖完整 RL 原理和工程闭环；实际训练按研究问题选择算法与反馈，不以跑遍所有配方作为学习完成标准。当前已补齐路线与源码入口，上述实现和运行验收待逐项完成。

<span id="source-block-d21831ac-7518-46ed-abe8-fcf98144134f"></span>

## 围绕本课讨论

<span id="source-block-30e1135a-97d8-4e4c-88c1-d33c6c636262"></span>

```text
请按数学定义、假设与推导、梯度估计、训练代理目标、参考算法、源码与张量对应的顺序教学。先从自回归轨迹的期望奖励推出策略梯度和baseline恒等式，再区分GAE、GRPO/RLOO与PPO代理损失。用保存的一批token逐步说明Score、动作mask、detach、全局分母和参数同步。每个等式注明成立条件，每段伪代码注明与固定verl实现的差异；最后再讨论TaH2的循环与路由。
```
