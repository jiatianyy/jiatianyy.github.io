---
title: "接入在线蒸馏"
description: "在线蒸馏让学生在自己生成的前缀上接受教师反馈。本课以冻结教师为起点，依次说明轨迹怎样产生、教师怎样评分、损失怎样形成梯度，以及更新后的参数怎样返回生成后端；随后讨论循环模型的接入条件。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_e434e46c533c819192afedba5f10a140"
sourcePageId: "page_e434e46c533c819192afedba5f10a140"
sourceSha256: ba6d9c83fe569e1ff74aab24be9cb3b954b513e679f1959b333888a2a458af28
series: loop-reasoning
references: []
---

<span id="source-block-6ffe3a2a-1b14-4e30-8b47-c13f769b6b46"></span>

在线蒸馏让学生在自己生成的前缀上接受教师反馈。本课以冻结教师为起点，依次说明轨迹怎样产生、教师怎样评分、损失怎样形成梯度，以及更新后的参数怎样返回生成后端；随后讨论循环模型的接入条件。

<span id="source-block-0e19602d-45ca-46ba-9ca3-1391d86a562e"></span>

[返回学习地图](/notes/loop-reasoning/engineering/)

<span id="source-block-a4c0b1bb-402b-4776-8c73-16c3bc9569a5"></span>

> [!CALLOUT]
>
> **阅读顺序** 先沿一轮采样、评分、更新与同步理解执行流程，再区分固定前缀梯度和状态访问梯度；之后阅读 top-k 与尾部桶，最后核对 TaH2 和 verl 的接入与验收。算法 6 给出同步主线，T5 处理混合分布的选定 token 目标。

<span id="source-block-3730c442-9956-4cee-afd9-ff972dbacc77"></span>

## 一轮在线蒸馏

<span id="source-block-331e1cc1-6054-4903-8d61-8498580375c5"></span>

1. **学生产生轨迹。** 用当前生成策略产生回答，记录 token、mask、采样条件和生成参数版本。

<span id="source-block-62247122-f868-4bac-8283-94133e966e21"></span>

2. **教师给出条件分布。** 在学生实际到达的前缀和目标位置上查询教师，取得目标 token 或 top-k token 的 log-probability，核对 tokenizer、词表与位置。

<span id="source-block-93e3a556-dd5f-4e6f-b014-ee73f8315aea"></span>

3. **学生计算目标并更新。** 根据选择的蒸馏算法构造损失与梯度。

<span id="source-block-0089a3cd-c4ed-48d7-8c34-94bfcd375f65"></span>

4. **同步后再次生成。** 把更新后的参数交给 rollout 后端，检查下一轮使用哪一版权重。

<span id="source-block-6d9e82ed-0e90-41bd-a63f-c566e07391d6"></span>

### 固定轨迹分布的训练目标

<span id="source-block-91d28cd9-ca57-4839-85d2-e84ae9c5d204"></span>

下面先定义固定行为策略下的训练目标；直接 KD 的逐状态损失可以取全词表 KL。

<span id="source-block-f3f9c47f-c750-4f73-aed0-40c7d41fe0d2"></span>

```math
\begin{aligned}
F_k(\theta)&=\mathbb E_{x\sim\mathcal D,\ y\sim\mu_k(\cdot\mid x)}
[f_\theta(x,y)],\\
f_\theta(x,y)&=\sum_{t=1}^H m_t\,\ell_\theta(s_t),\\
s_t&=(x,y_{<t}).
\end{aligned}
```

<span id="source-block-96ebf6fe-8932-4299-9d21-c767b63d36c6"></span>

本页以有限最大长度 `H`、终止后 mask 为零、轨迹损失求和作数学参考；实际聚合方式另记入配置。

<span id="source-block-5c6fb8a0-87b8-4506-884f-b66311b42ea4"></span>

第 `k` 轮的行为策略 `μ_k` 和教师固定，更新时采样轨迹作为固定输入；direct KD 可取 `ellθ(s)=KL(q(·|s)||pθ(·|s))`。

<span id="source-block-cf2a3098-1317-48dc-910c-576115c2edd4"></span>

## 固定前缀更新与完整期望梯度

<span id="source-block-ae600066-9f36-4e0d-ad8d-4973ff5c3918"></span>

### 轨迹分布是否参与求导

<span id="source-block-8b887128-d03f-4c92-9c8b-9855933e8933"></span>

现在让前缀来自学生自己。学生先生成一条回答，教师随后在这条回答经过的各个前缀上给出纠正信号。若本轮更新只把这批前缀当作固定输入，优化的就是 `F_k`：改善学生在已经到达的位置上的预测。这正是 direct KD 的梯度。另一种目标 `J` 还关心学生将来会选择哪些词、进入哪些前缀，因为这些选择会改变后续蒸馏损失的分布。在有限支持、概率为正且可微等条件下，对两个目标求导，就分别得到下面两式。

<span id="source-block-1a840022-64a4-4be5-95b3-4ef027411b19"></span>

```math
\begin{aligned}
\nabla F_k(\theta)&=\mathbb E_{\mu_k}[\nabla_\theta f_\theta],\\
J(\theta)&=\mathbb E_{x\sim\mathcal D,\ y\sim p_\theta(\cdot\mid x)}[f_\theta(x,y)],\\
\nabla J(\theta)&=\mathbb E_{p_\theta}\!\left[
\nabla_\theta f_\theta+
f_\theta\nabla_\theta\log p_\theta(y\mid x)\right].
\end{aligned}
```

<span id="source-block-4e2776c3-c8d6-48b1-9f67-e84261079abf"></span>

两式相差的不是一个数值修正系数，而是一种学习信号。`∇J` 的第一项让学生在当前前缀上更接近教师；第二项把后续损失与导致这些前缀的生成动作联系起来，使更新能够影响“今后会遇到什么前缀”。如果选择优化 `F_k`，固定这批输入就已经定义了完整目标；如果选择 `J`，则要显式构造第二项。下一轮重新 rollout 会提供新的输入，但不会替本轮 backward 计算这部分梯度。

<span id="source-block-9c72dc66-2df8-4d21-adf6-04515cacd44b"></span>

### 把状态访问项写成可执行的因果梯度

<span id="source-block-92c28a3d-63fb-4a90-8e0e-6320c5e351f5"></span>

沿一条回答轨迹看，某个动作只会改变它之后的前缀。因而，把整条轨迹的损失都乘到这个动作的 score 上虽然能写出期望梯度，却会掺入在动作发生前就已确定的损失；这些量的条件期望贡献为零，可以去掉。为把这个时间关系写清，假定时域 H 有限，EOS 后进入零损失吸收状态，每步 ℓ 只依赖当前动作发生前的状态。状态以可观察历史标识，隐藏计算的参数依赖保留在损失和策略的显式导数中。下式第二项因此只累加当前动作之后的损失。

<span id="source-block-74bc803f-ab01-4c2c-837f-bd53505b514f"></span>

```math
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=1}^{H}\ell_\theta(s_t)\right],
```

<span id="source-block-3d877681-0b1d-4328-b8bd-c488d91e3195"></span>

```math
\nabla J
=\mathbb E\left[
\sum_{t=1}^{H}\nabla_\theta\ell_\theta(s_t)
+\sum_{t=1}^{H}
\left(\sum_{u>t}\ell_\theta(s_u)\right)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
```

<span id="source-block-5fd91564-d137-433f-8293-f085eff9c52e"></span>

证明用到 E[b∇logπ(a|s)｜s]=0：在当前动作发生前已经确定的 b，无法通过这个动作改变其期望。于是，一个动作要承担的是它之后到达的状态的蒸馏损失，而不是过去的损失。若每步损失还直接依赖所采的当前动作，当前损失也属于它的回报，求和便从 u≥t 开始。与此同时，公式第一项始终保留 ℓ 自身对 θ 的显式导数。工程上的两个反传来源，正是“当前预测如何变准”和“生成选择怎样改变未来前缀”。

<span id="source-block-8118c95a-ffed-4353-a723-357d072291e0"></span>

用一个两步例子看这份缺失的信号。第一步按 Bernoulli(σ(θ)) 选择分支，第二步的损失在两个分支上分别为 0 和 1，并在分支给定后不再显式依赖 θ。拿到这条轨迹后，direct KD 看到的是一个固定的 0 或 1，局部反传当然为零；但改变 θ 会改变以后进入损失为 1 的分支的概率，所以总体期望仍有导数 σ(θ)(1−σ(θ))，θ=0 时为 0.25。这里缺少的信号来自第一步的选择，而不是第二步预测函数的显式导数。

<span id="source-block-9daa667b-295d-497f-9330-99171b114975"></span>

据此选择实现时，可以先把问题说成一句具体的话：这次更新只纠正学生已经生成的前缀，还是还要训练它选择更合适的后续前缀？前者可以采用固定轨迹的半梯度规则，后者需要把未来损失归给相应动作。两种规则都可以成为研究对象；记录实际使用的目标和梯度路径，比把它们统称为“on-policy 蒸馏”更能说明训练做了什么。

<span id="source-block-0a079ed1-97f2-422f-a7ec-cc83d7dc1da2"></span>

verl 的 direct 与 PG 分支按配置实现不同更新。把即时分布损失 detach 后作为单一 token 的 advantage，也不自动等于上式完整的未来损失回报；需逐项确认回报定义、基线、ratio 与聚合方式。[蒸馏损失入口](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L306)、[GKD 原论文](https://arxiv.org/abs/2306.13649v3)。

<span id="source-block-95ffed13-55ce-482f-b22c-b840c879ed69"></span>

### 估计值相同不代表梯度相同

<span id="source-block-92461af1-fac6-4121-8eb8-f6d8e2ec5af3"></span>

即使两个采样估计器的期望数值相同，直接对样本反传也可能得到不同梯度。下面在一个固定状态上比较 k1 与 k3：假设 `a~pθ`，教师 `q` 固定，概率为正且没有裁剪，并令 `r=q(a)/pθ(a)`。

<span id="source-block-1912f457-eec1-407f-b6a6-5134eb9f0793"></span>

```math
\begin{aligned}
k_1&=-\log r,\qquad k_3=r-\log r-1,\\
\mathbb E_{a\sim p_\theta}k_1
&=\mathbb E_{a\sim p_\theta}k_3
=D_{\rm KL}(p_\theta\Vert q).
\end{aligned}
```

<span id="source-block-215b7337-0cd1-407c-a25e-bea0153a108f"></span>

```math
\begin{aligned}
\mathbb E_{a\sim p_\theta}[\nabla_\theta k_3(a)]
&=\mathbb E_{p_\theta}[(1-r)\nabla_\theta\log p_\theta(a)]\\
&=-\sum_aq(a)\nabla_\theta\log p_\theta(a)\\
&=\nabla_\theta D_{\rm KL}(q\Vert p_\theta).
\end{aligned}
```

<span id="source-block-a1508794-5dbb-4eb3-bd7c-da0c52d28694"></span>

在 `∇k3(a)` 中，动作 a 已经采好，求导不会穿过采样分布。因此，k3 的数值期望等于 reverse KL，但其直接反传梯度的期望对应 forward KL。估计器的名称或期望数值本身，不能决定实际更新方向。直接反传 `k1` 时，单样本梯度不含教师概率，在上述理想条件下期望为零。

<span id="source-block-afb635f4-e6d9-4cdd-8f6a-20f5569224ee"></span>

### 直接蒸馏与策略梯度

<span id="source-block-54c936d6-e59b-4e74-b153-7d33476aa9a4"></span>

verl 的 direct 分支聚合损失后反向；PG 分支将负损失 detach 为优势再交给 policy loss。[蒸馏分支](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L310)

<span id="source-block-c4843be7-8eb9-45b0-a8ab-ee1f08352471"></span>

**理想条件：**同状态 on-policy 且无 clipping 时，以 `A=−sg(k1)` 作 PG 信号可恢复 reverse KL 梯度；这仍不自动包含整条轨迹的状态访问梯度。

<span id="source-block-c407a3a6-6fee-487d-adf9-104134ba1bbe"></span>

**固定实现：**实际 `k3` 有 log-ratio 与损失裁剪，`vanilla` policy loss 另含 PPO clipping、负优势 dual clipping 和可选 rollout 修正，均须与上述理想等式区分。[k3](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py#L2239) · [policy loss](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py#L1336)

<span id="source-block-1ff116bf-a655-47cc-9cf6-7ce3a6e62db6"></span>

### 算法 6 同步在线蒸馏

<span id="source-block-6c4decf2-18c6-4fb9-9034-5ba8823ee377"></span>

**过程** SyncOPD(resolved\_config)

<span id="source-block-2a7a6c2d-d876-4512-9d7c-14b34c0a136b"></span>

**输入** 解析后的配置、教师、当前学生及优化器、带版本的 rollout 后端。

<span id="source-block-0f4af71d-35d0-4bc8-a0fe-f10bab76205f"></span>

**输出** 更新并同步后的策略状态，以及行为概率、版本与同前缀概率核验记录。

<span id="source-block-191bbcf3-837a-44d6-be4d-621c4cf9b477"></span>

**前置条件** 算法 6 是按已声明目标更新的参考流程。direct 分支使用固定本批状态分布的显式梯度；PG 分支的 `A = -stopgrad(per_token_loss)` 也不能自动视为完整 on-policy KD 目标的因果回报。采样动作估计器还需满足实际行为分布与所声明目标的支持、ratio 和截断约定。

<span id="source-block-775536ea-91d9-4747-bd98-2cffbc1f71f7"></span>

<figure class="research-embed"><iframe src="/research-assets/479c72e038f83053f4090e86900695e54031a51bbff9a255bdb233c05b0e1cf0.html" title="算法 6 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 6 操作与数学公式 · <a href="/research-assets/479c72e038f83053f4090e86900695e54031a51bbff9a255bdb233c05b0e1cf0.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-ddb2c883-9d6f-4c4e-9c6b-edbc7dbe7372"></span>

PG 将整个 top-k 分布损失 detach 为单个采样动作的优势，通常不能保留 direct KD 中多个教师偏好 token 的梯度。

<span id="source-block-6bc089ff-4628-455d-ba1f-81ceed885c72"></span>

行为策略、old-policy anchor、当前学生与教师版本分别记录；重算 old log-probability 不改变数据的实际来源。

<span id="source-block-3d8ccae0-50ea-41be-86ce-4b92e03bffb5"></span>

### 任务奖励与有效配置

<span id="source-block-36ae217f-c1da-4961-8f3c-6884912a14ea"></span>

任务奖励与数值护栏均检查最终展开配置。

<span id="source-block-b66ce90d-e48f-4db8-a082-f45fc23d258e"></span>

- **任务奖励：**固定 YAML 的 `use_task_rewards=true`。[YAML](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/config/distillation/distillation.yaml)

<span id="source-block-a22cf44e-5ea0-4d98-ac87-77bd4e19d506"></span>

- **默认配置：**YAML 与 dataclass 在 top-k、direct／PG 和 clamp 默认值上存在差异，不能笼统称为“verl 默认算法”。[dataclass](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/config/distillation.py#L70)

<span id="source-block-65b9f5bf-8b44-4cf9-b84e-36c6f6129d3c"></span>

- **教师温度：**教师评分请求在该版本被设为温度 1，改变 rollout temperature 不会实现其他温度的 KD。[教师评分](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/experimental/teacher_loop/teacher_manager.py#L40)

<span id="source-block-c64c413d-164c-4c34-9773-b480e9fb39c7"></span>

TaH2 的在线深度监督与学生自产文字轨迹分别涉及内部计算深度和训练前缀来源。阅读时先明确 on-policy 指向哪一个选择过程。

<span id="source-block-e5093916-d24d-45d3-96dd-e368a0507940"></span>

## 固定版本的源码调用链

<span id="source-block-5273e124-ebc3-45ec-81b5-ecac68c178d8"></span>

固定版本：verl `7cea5392165ac26d5eadf879701337c85e4a2b2f`。

<span id="source-block-69554f4e-3eea-4519-b2d8-3054b37d1875"></span>

- [OPD 文档](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/algo/opd.md)：先读学生轨迹、教师评分与更新的数据流。

<span id="source-block-84180d07-a8a9-4143-adfc-2b9c2a6c1520"></span>

- [distillation/losses.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py)：对照 `distillation_ppo_loss` 与 `distillation_loss`。

<span id="source-block-fd1dad1a-68ca-4182-b1a9-6c3f3b5662f5"></span>

- [distillation.yaml](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/config/distillation/distillation.yaml)：检查目标、任务奖励和教师设置。

<span id="source-block-0f95dc9e-a525-4042-b3ec-c34abe913c8e"></span>

- [FSDP 接入指南](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)：理解训练支持、生成支持和权重同步的边界。

<span id="source-block-8b03f2c3-46c0-41ac-920c-b353c6514c6f"></span>

### 按需对照 CODI

<span id="source-block-a0c29271-bd0d-461e-847a-cdec02b222ea"></span>

[CODI 原论文](https://aclanthology.org/2025.emnlp-main.36/)与[官方仓库](https://github.com/zhenyi4/codi)放在显式教师到隐式学生的监督设计中阅读。对照教师与学生各自看见什么、状态如何对齐、损失放在哪里，再判断哪些部分适合我们的内部循环。

<span id="source-block-28114935-1d9a-450d-b146-3305e5c3ef21"></span>

## top k 监督与尾部桶

<span id="source-block-21179b0c-18d6-49b3-bee6-64b0b5de881d"></span>

完整教师分布可能很大，查询接口通常希望只返回 top-k。这样做首先遇到一个信息问题：我们知道若干词各自的概率，却不知道其余词之间怎样分配概率。令 `K(s)` 为教师选出的 top-k IDs，师生概率 q、p 都仍按全词表归一化。只对 K 内的词累加 KL 项，会漏掉其余词的贡献，所得截断和甚至可能为负。尾部桶保留了还可获得的一条信息——其余词的总概率：将它们合成一个桶，用 q_⊥ 与 p_⊥ 比较总量，就得到下式目标。

<span id="source-block-a56544ff-b344-42ab-9b5e-7baae69d9c39"></span>

```math
\begin{aligned}
L_K&=\sum_{v\in K}q_v\log\frac{q_v}{p_v},\\
q_\bot&=1-\sum_{v\in K}q_v,\qquad
p_\bot=1-\sum_{v\in K}p_v,\\
L_{K+\bot}&=L_K+q_\bot\log\frac{q_\bot}{p_\bot}.
\end{aligned}
```

<span id="source-block-1d7dc3cf-a8b0-4440-849c-dc734aa6572f"></span>

### 尾部桶的 KL 分解

<span id="source-block-c904d168-3c8a-4548-a113-a4b1c0d36045"></span>

精确概率运算且尾部质量为正时，将完整 KL 按 top-k 与尾部分解，可得：

<span id="source-block-29b70cd6-8a27-406f-8276-dbef89035345"></span>

```math
\begin{aligned}
D_{\rm KL}(q\Vert p)
&=L_{K+\bot}
+q_\bot D_{\rm KL}\!\left(q(\cdot\mid\bar K)\Vert p(\cdot\mid\bar K)\right),\\
0&\le L_{K+\bot}\le D_{\rm KL}(q\Vert p).
\end{aligned}
```

<span id="source-block-758aa01a-dd3c-4b93-8688-ed98923be908"></span>

证明只需将尾部每项的概率分解为“尾部质量×条件概率”后展开对数；零质量边界用连续延拓。

<span id="source-block-5ec581e6-f39a-4338-be93-55fb9e450c96"></span>

KL 分解说明尾部桶究竟教会学生什么。它要求选中词逐一匹配教师，并要求未选词合起来的总概率也匹配；省下的是在尾部内部继续分辨各个词的要求。右端额外的条件 KL 正是在衡量这部分被省略的分布差异。因此，桶目标已经很小，只说明保留下来的信息已经接近；若研究问题需要教师在尾部内部的偏好，就要保留更多词或查询完整分布。分别把师生 top-k 重新归一化，则是在只比较选中集合内部的条件概率，又是另一种目标。

<span id="source-block-b67f1355-0d43-4dd6-90ee-7cdeb9fdb4bd"></span>

### 尾部桶到底给每个 logit 什么梯度

<span id="source-block-bbba3e24-6b0f-419a-99ee-a616af568150"></span>

这些目标的差别会体现在哪些学生 logits 上？先固定教师 q 和教师选出的集合 K，令学生 p 为普通 softmax，所有概率均在全词表归一化，并假设尾部保护未触发。对 top-k 加尾部桶的 KL 求导，得到：

<span id="source-block-58815d25-4089-40f3-96ac-d893de81d433"></span>

```math
\frac{\partial L_{K+\bot}}{\partial z_v}
=\begin{cases}
p_v-q_v,&v\in K,\\[2pt]
p_v\left(1-\dfrac{q_\bot}{p_\bot}\right),&v\notin K.
\end{cases}
```

<span id="source-block-0cb6cc60-29dd-4e77-afb0-530a43adbe06"></span>

梯度把同一信息限制表现得很直接。对 v∈K，损失能读取教师给出的 q_v，所以局部 logit 梯度仍是全 KL 的 p_v−q_v。对 v∉K，接口没有每个词的 q_v，梯度只能通过统一的总质量比例 q_⊥/p_⊥ 来调整尾部，无法指出其中哪个词应更受偏好。这个结论把 K 当作教师提供的固定索引；若改用学生自身 top-k，还需要处理索引随参数变化的离散选择。

<span id="source-block-b9923303-302b-45f7-aedf-efdde15d2ed3"></span>

例如已有的三词分布 q=(0.6,0.3,0.1)、p=(0.6,0.1,0.3)，只查询第一个词。学生把第一个词和尾部总量都给对了，因此桶 KL 与梯度都是零；但它在尾部把两个词的偏好反了过来，全 KL 仍为 0.2 ln 3，约 0.219722。若训练日志只记桶损失，这个状态看起来已经匹配；是否值得增大 k，应取决于这部分缺失的偏好对任务是否重要，而不只看桶损失是否继续下降。

<span id="source-block-bb836479-a044-433e-884b-57b25443e710"></span>

```math
p_\bot<\varepsilon
\quad\Longrightarrow\quad
\frac{\partial L_{\mathrm{guarded}}}{\partial z_v}
=Q_Kp_v-q_v\mathbf1[v\in K],
\qquad
Q_K=\sum_{v\in K}q_v.
```

<span id="source-block-c6c3eaaa-616c-4014-b48f-6561d2ed2c79"></span>

当学生尾部质量严格小于保护阈值时，上式中的 clamp 会阻断尾部项对学生的直接梯度，剩下的是未重新归一化的 top-k 贡献。恰好等于阈值时，应使用实际算子的边界梯度约定，不能套用光滑区间公式。固定提交中的 `compute_tail_bucket_kl` 计算数值目标，`compute_tail_aware_logit_gradient` 给出融合路径的解析梯度。[tail_kl.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L18)

<span id="source-block-d7507487-a92d-4502-81ef-2d78befd8d32"></span>

TaH2 的输出是多轮概率混合；上式不能直接分别套到各轮 logits。应先对最终混合概率求梯度，再沿各分量与 gate 的依赖回传，具体责任权公式见[离线蒸馏页](/notes/loop-reasoning/offline-distillation/)。在三个词的未保护例子中，桶梯度的中心差分最大绝对误差为 2.6×10⁻¹¹；这只是有限维公式核验。

<span id="source-block-1b9c2809-6c55-43fc-b7b2-cee25a811375"></span>

### 源码中的数值护栏

<span id="source-block-33b93f47-d0a3-42de-9024-45ab4481289a"></span>

未启用额外 log-probability clamp 时，固定源码 `forward_kl_topk` 计算裸截断项后再 `clamp_min(0)`，负区间梯度被截断；额外 log-probability clamp 也会改变裸截断目标。[截断保护](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L436)

<span id="source-block-986b4bf0-fc2a-4bbf-952d-8f0ed877ebc3"></span>

`forward_kl_topk_tail` 不作上述负值截断，但会保护尾部质量。保护触发时已是修改后的数值目标，不能不加条件地套用精确 KL 分解或非负结论；学生尾部质量必须保持可导。[尾部数值实现](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L20)

<span id="source-block-bb0e6005-eabb-4831-b06e-b3ec6750b9c7"></span>

先在教学小词表上保存完整教师和学生分布，再同时计算全 KL、截断贡献和尾部桶目标，检查各自优化的量。生产模型按教师 token IDs 查询学生概率时，需要扩展 TaH2 仅收集硬标签概率的路径，并核对全局词表分片。

<span id="source-block-21eb6de5-8941-43f2-88d9-b067f64743eb"></span>

### 算法 T5 选定 token 的混合分布蒸馏

<span id="source-block-dfa046f4-97a8-4b16-b90f-da3f6e40d717"></span>

**过程** SelectedTokenMixtureKD(prefixes, teacher_ids, teacher_probabilities, mask, policy_semantics)

<span id="source-block-06367016-084b-4a4c-bc81-cfae51ea1ba5"></span>

**输入** 实际前缀、教师给出的唯一全局 token IDs 及其全词表概率、监督 mask、学生参数和明确的路由与混合语义；本参考算法固定温度为 1。

<span id="source-block-fad72039-8667-458b-bd3e-0e1cef3a4157"></span>

**输出** top-k 加尾部桶损失、学生梯度、各位置混合质量与尾质量诊断；教师无梯度。

<span id="source-block-32b23078-b795-4020-9f9d-bb993f1a5c1e"></span>

**前置条件与终止** 教师和学生词表及位置语义一致；教师 IDs 不重复，教师尾质量合法；参考目标采用理想归一化 stop mixture，实际执行轮数有有限上界。生产代码若使用权重 floor，必须先声明沿用未归一化质量还是构造归一化概率目标，并与参考结果比较。空监督窗口按全局并行协议一致结束；所有有效位置和有限轮数处理完毕后返回。

<span id="source-block-09e73152-b796-45b7-9617-4620afbfb9b1"></span>

<figure class="research-embed"><iframe src="/research-assets/18470e7192a95e000efa94d6517ba99f50e598d729e8f308affb5c5b743446ff.html" title="算法 T5 选定 token 的混合分布蒸馏" loading="lazy" sandbox=""></iframe><figcaption>算法 T5 选定 token 的混合分布蒸馏 · <a href="/research-assets/18470e7192a95e000efa94d6517ba99f50e598d729e8f308affb5c5b743446ff.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-554ed073-2b3f-4621-9a34-79b68fcd0601"></span>

**实现范围** T5 是待实现的接入参考：它把 TaH2 `accumulate_iter / finalize` 的分量累计、全词表 normalizer 与 verl `compute_tail_bucket_kl` 组合起来。仓库当前的 target-only 路径面向硬标签，尚不构成任意教师 K 个 IDs 的完整蒸馏接口。[TaH2 runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520) · [尾部桶函数](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L18)

<span id="source-block-212f9910-8a34-45b0-bbe3-faeeda5b6d5a"></span>

### 随机教师的条件读出与边缘蒸馏

<span id="source-block-011aaf31-086b-4a3e-9553-b13e166d10c9"></span>

随机路由还会使同一次教师查询产生不同的条件读出。以 TaH2 的可选随机分支为例，固定教师开始计算时的内部状态后，停止于深度 D 得到 P_D；将所有随机深度按其概率平均，才得到文字边缘分布 P̄=E_D[P_D]。它一般不等于固定最大深度时的混合读出。因此，查询接口需要先明确要返回哪一种教师预测。[随机深度的精确读出与成本 T9](/notes/loop-reasoning/internal-loop/) · [固定引擎分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

<span id="source-block-7ab44d62-39d0-46e8-bda3-30c9672adb47"></span>

固定教师参数、路由分布和入态，令学生在同一条件信息下给出 S_theta，且在教师支持上为正。教师目标全部停止梯度。前向 KL 满足精确分解：

<span id="source-block-bbba92fe-9826-42fb-9f22-5197bd1ac0e3"></span>

```math
\mathbb E_D D_{\rm KL}(P_D\Vert S_\theta)
=D_{\rm KL}(\bar P\Vert S_\theta)
+\mathbb E_DD_{\rm KL}(P_D\Vert\bar P).
```

<span id="source-block-e625cb0e-1c36-4964-b21c-eab70dcca10e"></span>

证明时，把 log(P_D/S) 拆成 log(P_D/P̄)+log(P̄/S)，再对 D 求期望，并使用 E[P_D]=P̄。右端第二项只比较教师路线，与学生参数无关。因此，在上述条件下，抽取一个教师深度、用完整 P_D 计算 forward-KL 梯度，可得到边缘教师目标梯度的无偏估计。两种 loss 数值仍相差教师路线之间的分布差异，一次条件 log-probability 也不等于边缘 log-probability。若只保留 top-k、改变尾部桶或采用其他非线性变换，就需要针对实际目标重新分析。

<span id="source-block-cecfe184-360b-4406-ba87-f2e7806199b0"></span>

随机性在学生端时，不能照搬这条梯度结论。给定固定教师 T、固定路由权重，按 −log 的凸性，有：

<span id="source-block-617c7eb5-6a2b-4322-9b0b-bcdc494544bf"></span>

```math
\mathbb E_D\operatorname{CE}(T,S_D)
\ge\operatorname{CE}\!\left(T,\mathbb E_D[S_D]\right).
```

<span id="source-block-721274c2-696d-4daa-9b35-07369a6359b8"></span>

左式要求每条具有正权重的学生路线分别拟合教师，右式只要求这些路线平均后的预测拟合教师，因而是两个不同目标。若学生路由权重还依赖 theta，左式的完整梯度还包含路由分布的 score 项；只反传抽到路线的 CE，既没有补齐左式梯度，也不能据此声称优化右式。即使补入先验路由 score，得到的仍是左式梯度；右式的边缘 CE 需要使用观察目标词后的路由后验责任权。[状态访问与策略梯度](/notes/loop-reasoning/reinforcement-learning/)

<span id="source-block-2ca3544c-8771-4df1-8c04-5ea01cf94d52"></span>

上述比较都固定了开始计算时的内部状态。在线蒸馏中，若师生各自保留随机 KV 历史，就应保存这些真实历史，或对它们做边缘化；不能把来自不同教师历史的输出拼成同一个状态下的多深度分量。T5 的选定 token gather 只负责求出既定读出下的概率，联合目标、条件目标还是边缘目标仍由调用方明确选择。

<span id="source-block-6add9871-7175-4ae4-92df-2f97c839c363"></span>

## 循环模型接入训练与生成的条件

<span id="source-block-36e40c20-5c7b-49c3-9cd1-e0e52ddb6731"></span>

| 边界 | 需要补齐的计算与记录 |
| --- | --- |
| 训练模型 | 正确构造循环模型与配置；沿学生真实路径重算 token 概率，声明 full／target-only／selected-token 支持 |
| rollout 后端 | 自主执行相同位置、缓存、路由和混合规则，返回 token、mask、深度和版本 |
| 教师评分 | 与学生实际前缀、词表 ID 和目标位置对应，教师权重与梯度策略明确 |
| 权重转换与同步 | 骨干、updater、decider 的命名、形状、分片和 dtype 都有转换依据 |
| 训练目标 | 显式选择直接 KD 或 PG、KL 模式、任务奖励、reference KL 与 loss 分母 |
| 再次生成 | 连续两次 optimizer step 后都检查同步，记录第二轮轨迹实际使用的参数版本 |

<span id="source-block-6a8b2e1c-0541-48f3-bbbe-856d5438a94b"></span>

[FSDP 接入指南](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)说明训练端加载、rollout 支持和参数流转换的边界；沿 [engine_workers.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/engine_workers.py) 的 `update_weights` 阅读实际同步路径。将 TaH2 包装成可加载模型只完成其中一项。

<span id="source-block-63987191-3310-47e9-a217-7332d25b3dfd"></span>

第一版接入方案采用同步控制流：学生生成一批轨迹，教师评分，学生更新，然后同步并重新生成。

<span id="source-block-a3e93b1f-0392-48e5-841c-eb936d01d4bb"></span>

两张 A100 下的教师与学生驻留、分时加载或缓存评分需按显存测量选择；

<span id="source-block-0193f576-3b70-4dd4-95fe-46e521cd5704"></span>

评分缓存仅对完全相同的前缀、目标位置与教师版本复用。

<span id="source-block-29356cc9-8d61-4215-970b-edf3e22bd723"></span>

## 接入验收

<span id="source-block-96e04b91-9757-4f09-92d4-086d9e5cf75d"></span>

- [ ] `opd_config_resolved.yaml`：目标分支、任务奖励、reference KL、top-k 与聚合配置全部展开。

<span id="source-block-b642a51c-988d-4bb3-84f9-6a3fa3297b18"></span>

- [ ] `teacher_student_scores.json`：同前缀、同位置、同词表 ID 的教师与学生概率可对照。

<span id="source-block-5d3724dd-a49f-4809-9ce4-19974f44bd31"></span>

- [ ] `two_update_timeline.json`：采样、评分、反向、同步和下一轮版本有实际时间顺序。

<span id="source-block-94cbc366-d29f-4593-8842-a8c80dbfcb97"></span>

- [ ] `weight_sync_parity.json`：两次更新后的 actor 与 rollout 概率差异可解释，并覆盖循环组件。

<span id="source-block-b0f43d39-3071-4801-b979-3c12cda6e42e"></span>

[离线蒸馏与过程监督](/notes/loop-reasoning/offline-distillation/)补教师目标与状态对齐；OPD 的学生前缀反馈与内部状态续算仍分别验证。

<span id="source-block-c530370e-0c9b-4392-a99f-4ecdf1083a89"></span>

## 动手检查

<span id="source-block-9273c3bb-28f6-497a-820f-2b8751f59606"></span>

画出连续两轮在线更新的时序，给每条轨迹标注生成版本。若优化器已更新学生，而 rollout 后端仍保留旧权重，下一批轨迹与当前学生是什么关系？应记录哪些信息才能发现问题？

<span id="source-block-90d95677-129b-4a74-891e-f926d1f772ff"></span>

### 参考思路

<span id="source-block-ddd850cf-b520-48bb-8dbd-3cf0ffb5d208"></span>

这些轨迹来自旧的行为策略；是否可用于当前目标，取决于算法如何处理策略差异。至少记录生成参数版本、更新步数、采样设置，以及所需的行为概率或重算条件。不能仅以训练 loss 下降判断同步正确。

<span id="source-block-4fbd808c-d172-423d-ac33-4199cc9a2790"></span>

官方接入指南建议至少检查两次在线 optimizer step，使第二次权重重载也被覆盖，并比较同步后的 actor 与 rollout token log-probability。这可以作为后续实际对接的检查任务。[接入检查](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)

<span id="source-block-667ec38a-ad4b-49b8-b8e2-b5e189fd4696"></span>

## 与我们的研究连接

<span id="source-block-5e6a0b08-44f6-4a30-ab3e-21d52a105dda"></span>

在线蒸馏能够提供能力迁移的训练信号，但不会自动证明循环状态更有用，也不保证恢复原有 CoT 能力。仍需明确更新哪些参数、梯度是否到达接口、部署时学生是否依赖帮助。TaH2 与 verl 的实际对接是后续实现工作。

<span id="source-block-6b51dc94-23bb-49dd-be60-5af781dcdb1a"></span>

## 围绕本课讨论

<span id="source-block-31454bfb-979e-467a-847c-8b7f821b559c"></span>

```text
请带我画连续两轮在线蒸馏的时序，逐项标注学生轨迹、教师所见前缀、损失、梯度和参数版本。随后对照固定版本 verl 的直接蒸馏与策略梯度分支，最后讨论接入 TaH2 时需要补哪些接口。
```

<span id="source-block-4e31ef24-5dce-4526-ac1b-b4924bf925e3"></span>

## 继续学习

<span id="source-block-fd172c34-bf65-44e8-8ab9-4239f726d17d"></span>

[上一课：从训练到自主生成](/notes/loop-reasoning/generation/) · [学习地图](/notes/loop-reasoning/engineering/) · [回到框架设计](/notes/loop-reasoning/framework/)
