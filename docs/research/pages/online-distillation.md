在线蒸馏让学生在自己生成的前缀上接受教师反馈。本课以冻结教师为起点，依次说明轨迹怎样产生、教师怎样评分、损失怎样形成梯度，以及更新后的参数怎样返回生成后端；随后讨论循环模型的接入条件。

[返回学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
>
> **阅读顺序** 先沿一轮采样、评分、更新与同步理解执行流程，再区分固定前缀梯度和状态访问梯度；之后阅读 top-k 与尾部桶，最后核对 TaH2 和 verl 的接入与验收。算法 6 给出同步主线，T5 处理混合分布的选定 token 目标。

# 一轮在线蒸馏

1. **学生产生轨迹。** 用当前生成策略产生回答，记录 token、mask、采样条件和生成参数版本。

2. **教师给出条件分布。** 在学生实际到达的前缀和目标位置上查询教师，取得目标 token 或 top-k token 的 log-probability，核对 tokenizer、词表与位置。

3. **学生计算目标并更新。** 根据选择的蒸馏算法构造损失与梯度。

4. **同步后再次生成。** 把更新后的参数交给 rollout 后端，检查下一轮使用哪一版权重。

## 固定轨迹分布的训练目标

下面先定义固定行为策略下的训练目标；直接 KD 的逐状态损失可以取全词表 KL。

```math
\begin{aligned}
F_k(\theta)&=\mathbb E_{x\sim\mathcal D,\ y\sim\mu_k(\cdot\mid x)}
[f_\theta(x,y)],\\
f_\theta(x,y)&=\sum_{t=1}^H m_t\,\ell_\theta(s_t),\\
s_t&=(x,y_{<t}).
\end{aligned}
```

本页以有限最大长度 `H`、终止后 mask 为零、轨迹损失求和作数学参考；实际聚合方式另记入配置。

第 `k` 轮的行为策略 `μ_k` 和教师固定，更新时采样轨迹作为固定输入；direct KD 可取 `ellθ(s)=KL(q(·|s)||pθ(·|s))`。

# 固定前缀更新与完整期望梯度

## 轨迹分布是否参与求导

现在让前缀来自学生自己。学生先生成一条回答，教师随后在这条回答经过的各个前缀上给出纠正信号。若本轮更新只把这批前缀当作固定输入，优化的就是 `F_k`：改善学生在已经到达的位置上的预测。这正是 direct KD 的梯度。另一种目标 `J` 还关心学生将来会选择哪些词、进入哪些前缀，因为这些选择会改变后续蒸馏损失的分布。在有限支持、概率为正且可微等条件下，对两个目标求导，就分别得到下面两式。

```math
\begin{aligned}
\nabla F_k(\theta)&=\mathbb E_{\mu_k}[\nabla_\theta f_\theta],\\
J(\theta)&=\mathbb E_{x\sim\mathcal D,\ y\sim p_\theta(\cdot\mid x)}[f_\theta(x,y)],\\
\nabla J(\theta)&=\mathbb E_{p_\theta}\!\left[
\nabla_\theta f_\theta+
f_\theta\nabla_\theta\log p_\theta(y\mid x)\right].
\end{aligned}
```

两式相差的不是一个数值修正系数，而是一种学习信号。`∇J` 的第一项让学生在当前前缀上更接近教师；第二项把后续损失与导致这些前缀的生成动作联系起来，使更新能够影响“今后会遇到什么前缀”。如果选择优化 `F_k`，固定这批输入就已经定义了完整目标；如果选择 `J`，则要显式构造第二项。下一轮重新 rollout 会提供新的输入，但不会替本轮 backward 计算这部分梯度。

## 把状态访问项写成可执行的因果梯度

沿一条回答轨迹看，某个动作只会改变它之后的前缀。因而，把整条轨迹的损失都乘到这个动作的 score 上虽然能写出期望梯度，却会掺入在动作发生前就已确定的损失；这些量的条件期望贡献为零，可以去掉。为把这个时间关系写清，假定时域 H 有限，EOS 后进入零损失吸收状态，每步 ℓ 只依赖当前动作发生前的状态。状态以可观察历史标识，隐藏计算的参数依赖保留在损失和策略的显式导数中。下式第二项因此只累加当前动作之后的损失。

```math
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}
\left[\sum_{t=1}^{H}\ell_\theta(s_t)\right],
```

```math
\nabla J
=\mathbb E\left[
\sum_{t=1}^{H}\nabla_\theta\ell_\theta(s_t)
+\sum_{t=1}^{H}
\left(\sum_{u>t}\ell_\theta(s_u)\right)
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
```

证明用到 E[b∇logπ(a|s)｜s]=0：在当前动作发生前已经确定的 b，无法通过这个动作改变其期望。于是，一个动作要承担的是它之后到达的状态的蒸馏损失，而不是过去的损失。若每步损失还直接依赖所采的当前动作，当前损失也属于它的回报，求和便从 u≥t 开始。与此同时，公式第一项始终保留 ℓ 自身对 θ 的显式导数。工程上的两个反传来源，正是“当前预测如何变准”和“生成选择怎样改变未来前缀”。

用一个两步例子看这份缺失的信号。第一步按 Bernoulli(σ(θ)) 选择分支，第二步的损失在两个分支上分别为 0 和 1，并在分支给定后不再显式依赖 θ。拿到这条轨迹后，direct KD 看到的是一个固定的 0 或 1，局部反传当然为零；但改变 θ 会改变以后进入损失为 1 的分支的概率，所以总体期望仍有导数 σ(θ)(1−σ(θ))，θ=0 时为 0.25。这里缺少的信号来自第一步的选择，而不是第二步预测函数的显式导数。

据此选择实现时，可以先把问题说成一句具体的话：这次更新只纠正学生已经生成的前缀，还是还要训练它选择更合适的后续前缀？前者可以采用固定轨迹的半梯度规则，后者需要把未来损失归给相应动作。两种规则都可以成为研究对象；记录实际使用的目标和梯度路径，比把它们统称为“on-policy 蒸馏”更能说明训练做了什么。

verl 的 direct 与 PG 分支按配置实现不同更新。把即时分布损失 detach 后作为单一 token 的 advantage，也不自动等于上式完整的未来损失回报；需逐项确认回报定义、基线、ratio 与聚合方式。[蒸馏损失入口](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L306)、[GKD 原论文](https://arxiv.org/abs/2306.13649v3)。

## 估计值相同不代表梯度相同

即使两个采样估计器的期望数值相同，直接对样本反传也可能得到不同梯度。下面在一个固定状态上比较 k1 与 k3：假设 `a~pθ`，教师 `q` 固定，概率为正且没有裁剪，并令 `r=q(a)/pθ(a)`。

```math
\begin{aligned}
k_1&=-\log r,\qquad k_3=r-\log r-1,\\
\mathbb E_{a\sim p_\theta}k_1
&=\mathbb E_{a\sim p_\theta}k_3
=D_{\rm KL}(p_\theta\Vert q).
\end{aligned}
```

```math
\begin{aligned}
\mathbb E_{a\sim p_\theta}[\nabla_\theta k_3(a)]
&=\mathbb E_{p_\theta}[(1-r)\nabla_\theta\log p_\theta(a)]\\
&=-\sum_aq(a)\nabla_\theta\log p_\theta(a)\\
&=\nabla_\theta D_{\rm KL}(q\Vert p_\theta).
\end{aligned}
```

在 `∇k3(a)` 中，动作 a 已经采好，求导不会穿过采样分布。因此，k3 的数值期望等于 reverse KL，但其直接反传梯度的期望对应 forward KL。估计器的名称或期望数值本身，不能决定实际更新方向。直接反传 `k1` 时，单样本梯度不含教师概率，在上述理想条件下期望为零。

## 直接蒸馏与策略梯度

verl 的 direct 分支聚合损失后反向；PG 分支将负损失 detach 为优势再交给 policy loss。[蒸馏分支](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L310)

**理想条件：**同状态 on-policy 且无 clipping 时，以 `A=−sg(k1)` 作 PG 信号可恢复 reverse KL 梯度；这仍不自动包含整条轨迹的状态访问梯度。

**固定实现：**实际 `k3` 有 log-ratio 与损失裁剪，`vanilla` policy loss 另含 PPO clipping、负优势 dual clipping 和可选 rollout 修正，均须与上述理想等式区分。[k3](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py#L2239) · [policy loss](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/ppo/core_algos.py#L1336)

## 算法 6 同步在线蒸馏

**过程** SyncOPD(resolved\_config)

**输入** 解析后的配置、教师、当前学生及优化器、带版本的 rollout 后端。

**输出** 更新并同步后的策略状态，以及行为概率、版本与同前缀概率核验记录。

**前置条件** 算法 6 是按已声明目标更新的参考流程。direct 分支使用固定本批状态分布的显式梯度；PG 分支的 `A = -stopgrad(per_token_loss)` 也不能自动视为完整 on-policy KD 目标的因果回报。采样动作估计器还需满足实际行为分布与所声明目标的支持、ratio 和截断约定。

![算法 6 操作与数学公式](visualize:fde1_bGliZmlsZV9JS1RCMXptMzNRcGFzVXlOYzg1Y0x3_FileDrive_45288dfe06608191af94334ba8699511)

PG 将整个 top-k 分布损失 detach 为单个采样动作的优势，通常不能保留 direct KD 中多个教师偏好 token 的梯度。

行为策略、old-policy anchor、当前学生与教师版本分别记录；重算 old log-probability 不改变数据的实际来源。

## 任务奖励与有效配置

任务奖励与数值护栏均检查最终展开配置。

- **任务奖励：**固定 YAML 的 `use_task_rewards=true`。[YAML](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/config/distillation/distillation.yaml)

- **默认配置：**YAML 与 dataclass 在 top-k、direct／PG 和 clamp 默认值上存在差异，不能笼统称为“verl 默认算法”。[dataclass](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/config/distillation.py#L70)

- **教师温度：**教师评分请求在该版本被设为温度 1，改变 rollout temperature 不会实现其他温度的 KD。[教师评分](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/experimental/teacher_loop/teacher_manager.py#L40)

TaH2 的在线深度监督与学生自产文字轨迹分别涉及内部计算深度和训练前缀来源。阅读时先明确 on-policy 指向哪一个选择过程。

# 固定版本的源码调用链

固定版本：verl `7cea5392165ac26d5eadf879701337c85e4a2b2f`。

- [OPD 文档](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/algo/opd.md)：先读学生轨迹、教师评分与更新的数据流。

- [distillation/losses.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py)：对照 `distillation_ppo_loss` 与 `distillation_loss`。

- [distillation.yaml](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/config/distillation/distillation.yaml)：检查目标、任务奖励和教师设置。

- [FSDP 接入指南](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)：理解训练支持、生成支持和权重同步的边界。

## 按需对照 CODI

[CODI 原论文](https://aclanthology.org/2025.emnlp-main.36/)与[官方仓库](https://github.com/zhenyi4/codi)放在显式教师到隐式学生的监督设计中阅读。对照教师与学生各自看见什么、状态如何对齐、损失放在哪里，再判断哪些部分适合我们的内部循环。

# top k 监督与尾部桶

完整教师分布可能很大，查询接口通常希望只返回 top-k。这样做首先遇到一个信息问题：我们知道若干词各自的概率，却不知道其余词之间怎样分配概率。令 `K(s)` 为教师选出的 top-k IDs，师生概率 q、p 都仍按全词表归一化。只对 K 内的词累加 KL 项，会漏掉其余词的贡献，所得截断和甚至可能为负。尾部桶保留了还可获得的一条信息——其余词的总概率：将它们合成一个桶，用 q_⊥ 与 p_⊥ 比较总量，就得到下式目标。

```math
\begin{aligned}
L_K&=\sum_{v\in K}q_v\log\frac{q_v}{p_v},\\
q_\bot&=1-\sum_{v\in K}q_v,\qquad
p_\bot=1-\sum_{v\in K}p_v,\\
L_{K+\bot}&=L_K+q_\bot\log\frac{q_\bot}{p_\bot}.
\end{aligned}
```

## 尾部桶的 KL 分解

精确概率运算且尾部质量为正时，将完整 KL 按 top-k 与尾部分解，可得：

```math
\begin{aligned}
D_{\rm KL}(q\Vert p)
&=L_{K+\bot}
+q_\bot D_{\rm KL}\!\left(q(\cdot\mid\bar K)\Vert p(\cdot\mid\bar K)\right),\\
0&\le L_{K+\bot}\le D_{\rm KL}(q\Vert p).
\end{aligned}
```

证明只需将尾部每项的概率分解为“尾部质量×条件概率”后展开对数；零质量边界用连续延拓。

KL 分解说明尾部桶究竟教会学生什么。它要求选中词逐一匹配教师，并要求未选词合起来的总概率也匹配；省下的是在尾部内部继续分辨各个词的要求。右端额外的条件 KL 正是在衡量这部分被省略的分布差异。因此，桶目标已经很小，只说明保留下来的信息已经接近；若研究问题需要教师在尾部内部的偏好，就要保留更多词或查询完整分布。分别把师生 top-k 重新归一化，则是在只比较选中集合内部的条件概率，又是另一种目标。

## 尾部桶到底给每个 logit 什么梯度

这些目标的差别会体现在哪些学生 logits 上？先固定教师 q 和教师选出的集合 K，令学生 p 为普通 softmax，所有概率均在全词表归一化，并假设尾部保护未触发。对 top-k 加尾部桶的 KL 求导，得到：

```math
\frac{\partial L_{K+\bot}}{\partial z_v}
=\begin{cases}
p_v-q_v,&v\in K,\\[2pt]
p_v\left(1-\dfrac{q_\bot}{p_\bot}\right),&v\notin K.
\end{cases}
```

梯度把同一信息限制表现得很直接。对 v∈K，损失能读取教师给出的 q_v，所以局部 logit 梯度仍是全 KL 的 p_v−q_v。对 v∉K，接口没有每个词的 q_v，梯度只能通过统一的总质量比例 q_⊥/p_⊥ 来调整尾部，无法指出其中哪个词应更受偏好。这个结论把 K 当作教师提供的固定索引；若改用学生自身 top-k，还需要处理索引随参数变化的离散选择。

例如已有的三词分布 q=(0.6,0.3,0.1)、p=(0.6,0.1,0.3)，只查询第一个词。学生把第一个词和尾部总量都给对了，因此桶 KL 与梯度都是零；但它在尾部把两个词的偏好反了过来，全 KL 仍为 0.2 ln 3，约 0.219722。若训练日志只记桶损失，这个状态看起来已经匹配；是否值得增大 k，应取决于这部分缺失的偏好对任务是否重要，而不只看桶损失是否继续下降。

```math
p_\bot<\varepsilon
\quad\Longrightarrow\quad
\frac{\partial L_{\mathrm{guarded}}}{\partial z_v}
=Q_Kp_v-q_v\mathbf1[v\in K],
\qquad
Q_K=\sum_{v\in K}q_v.
```

当学生尾部质量严格小于保护阈值时，上式中的 clamp 会阻断尾部项对学生的直接梯度，剩下的是未重新归一化的 top-k 贡献。恰好等于阈值时，应使用实际算子的边界梯度约定，不能套用光滑区间公式。固定提交中的 `compute_tail_bucket_kl` 计算数值目标，`compute_tail_aware_logit_gradient` 给出融合路径的解析梯度。[tail_kl.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L18)

TaH2 的输出是多轮概率混合；上式不能直接分别套到各轮 logits。应先对最终混合概率求梯度，再沿各分量与 gate 的依赖回传，具体责任权公式见[离线蒸馏页](https://chatgpt.com/space/page_a1b4546efc988191a19aeb2193c29651)。在三个词的未保护例子中，桶梯度的中心差分最大绝对误差为 2.6×10⁻¹¹；这只是有限维公式核验。

## 源码中的数值护栏

未启用额外 log-probability clamp 时，固定源码 `forward_kl_topk` 计算裸截断项后再 `clamp_min(0)`，负区间梯度被截断；额外 log-probability clamp 也会改变裸截断目标。[截断保护](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/losses.py#L436)

`forward_kl_topk_tail` 不作上述负值截断，但会保护尾部质量。保护触发时已是修改后的数值目标，不能不加条件地套用精确 KL 分解或非负结论；学生尾部质量必须保持可导。[尾部数值实现](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L20)

先在教学小词表上保存完整教师和学生分布，再同时计算全 KL、截断贡献和尾部桶目标，检查各自优化的量。生产模型按教师 token IDs 查询学生概率时，需要扩展 TaH2 仅收集硬标签概率的路径，并核对全局词表分片。

## 算法 T5 选定 token 的混合分布蒸馏

**过程** SelectedTokenMixtureKD(prefixes, teacher_ids, teacher_probabilities, mask, policy_semantics)

**输入** 实际前缀、教师给出的唯一全局 token IDs 及其全词表概率、监督 mask、学生参数和明确的路由与混合语义；本参考算法固定温度为 1。

**输出** top-k 加尾部桶损失、学生梯度、各位置混合质量与尾质量诊断；教师无梯度。

**前置条件与终止** 教师和学生词表及位置语义一致；教师 IDs 不重复，教师尾质量合法；参考目标采用理想归一化 stop mixture，实际执行轮数有有限上界。生产代码若使用权重 floor，必须先声明沿用未归一化质量还是构造归一化概率目标，并与参考结果比较。空监督窗口按全局并行协议一致结束；所有有效位置和有限轮数处理完毕后返回。

![算法 T5 选定 token 的混合分布蒸馏](visualize:fde1_bGliZmlsZV9WTFQwc19rVDh1dUNEMFMzNFJBaFpR_FileDrive_45288dfe06608191af94334ba8699511)

**实现范围** T5 是待实现的接入参考：它把 TaH2 `accumulate_iter / finalize` 的分量累计、全词表 normalizer 与 verl `compute_tail_bucket_kl` 组合起来。仓库当前的 target-only 路径面向硬标签，尚不构成任意教师 K 个 IDs 的完整蒸馏接口。[TaH2 runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520) · [尾部桶函数](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/trainer/distillation/tail_kl.py#L18)

## 随机教师的条件读出与边缘蒸馏

随机路由还会使同一次教师查询产生不同的条件读出。以 TaH2 的可选随机分支为例，固定教师开始计算时的内部状态后，停止于深度 D 得到 P_D；将所有随机深度按其概率平均，才得到文字边缘分布 P̄=E_D[P_D]。它一般不等于固定最大深度时的混合读出。因此，查询接口需要先明确要返回哪一种教师预测。[随机深度的精确读出与成本 T9](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811) · [固定引擎分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

固定教师参数、路由分布和入态，令学生在同一条件信息下给出 S_theta，且在教师支持上为正。教师目标全部停止梯度。前向 KL 满足精确分解：

```math
\mathbb E_D D_{\rm KL}(P_D\Vert S_\theta)
=D_{\rm KL}(\bar P\Vert S_\theta)
+\mathbb E_DD_{\rm KL}(P_D\Vert\bar P).
```

证明时，把 log(P_D/S) 拆成 log(P_D/P̄)+log(P̄/S)，再对 D 求期望，并使用 E[P_D]=P̄。右端第二项只比较教师路线，与学生参数无关。因此，在上述条件下，抽取一个教师深度、用完整 P_D 计算 forward-KL 梯度，可得到边缘教师目标梯度的无偏估计。两种 loss 数值仍相差教师路线之间的分布差异，一次条件 log-probability 也不等于边缘 log-probability。若只保留 top-k、改变尾部桶或采用其他非线性变换，就需要针对实际目标重新分析。

随机性在学生端时，不能照搬这条梯度结论。给定固定教师 T、固定路由权重，按 −log 的凸性，有：

```math
\mathbb E_D\operatorname{CE}(T,S_D)
\ge\operatorname{CE}\!\left(T,\mathbb E_D[S_D]\right).
```

左式要求每条具有正权重的学生路线分别拟合教师，右式只要求这些路线平均后的预测拟合教师，因而是两个不同目标。若学生路由权重还依赖 theta，左式的完整梯度还包含路由分布的 score 项；只反传抽到路线的 CE，既没有补齐左式梯度，也不能据此声称优化右式。即使补入先验路由 score，得到的仍是左式梯度；右式的边缘 CE 需要使用观察目标词后的路由后验责任权。[状态访问与策略梯度](https://chatgpt.com/space/page_a04bfe2924d48191b95083d418d93368)

上述比较都固定了开始计算时的内部状态。在线蒸馏中，若师生各自保留随机 KV 历史，就应保存这些真实历史，或对它们做边缘化；不能把来自不同教师历史的输出拼成同一个状态下的多深度分量。T5 的选定 token gather 只负责求出既定读出下的概率，联合目标、条件目标还是边缘目标仍由调用方明确选择。

# 循环模型接入训练与生成的条件

| 边界 | 需要补齐的计算与记录 |
| --- | --- |
| 训练模型 | 正确构造循环模型与配置；沿学生真实路径重算 token 概率，声明 full／target-only／selected-token 支持 |
| rollout 后端 | 自主执行相同位置、缓存、路由和混合规则，返回 token、mask、深度和版本 |
| 教师评分 | 与学生实际前缀、词表 ID 和目标位置对应，教师权重与梯度策略明确 |
| 权重转换与同步 | 骨干、updater、decider 的命名、形状、分片和 dtype 都有转换依据 |
| 训练目标 | 显式选择直接 KD 或 PG、KL 模式、任务奖励、reference KL 与 loss 分母 |
| 再次生成 | 连续两次 optimizer step 后都检查同步，记录第二轮轨迹实际使用的参数版本 |

[FSDP 接入指南](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)说明训练端加载、rollout 支持和参数流转换的边界；沿 [engine_workers.py](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/verl/workers/engine_workers.py) 的 `update_weights` 阅读实际同步路径。将 TaH2 包装成可加载模型只完成其中一项。

第一版接入方案采用同步控制流：学生生成一批轨迹，教师评分，学生更新，然后同步并重新生成。

两张 A100 下的教师与学生驻留、分时加载或缓存评分需按显存测量选择；

评分缓存仅对完全相同的前缀、目标位置与教师版本复用。

# 接入验收

- [ ] `opd_config_resolved.yaml`：目标分支、任务奖励、reference KL、top-k 与聚合配置全部展开。

- [ ] `teacher_student_scores.json`：同前缀、同位置、同词表 ID 的教师与学生概率可对照。

- [ ] `two_update_timeline.json`：采样、评分、反向、同步和下一轮版本有实际时间顺序。

- [ ] `weight_sync_parity.json`：两次更新后的 actor 与 rollout 概率差异可解释，并覆盖循环组件。

[离线蒸馏与过程监督](https://chatgpt.com/space/page_a1b4546efc988191a19aeb2193c29651)补教师目标与状态对齐；OPD 的学生前缀反馈与内部状态续算仍分别验证。

# 动手检查

画出连续两轮在线更新的时序，给每条轨迹标注生成版本。若优化器已更新学生，而 rollout 后端仍保留旧权重，下一批轨迹与当前学生是什么关系？应记录哪些信息才能发现问题？

## 参考思路

这些轨迹来自旧的行为策略；是否可用于当前目标，取决于算法如何处理策略差异。至少记录生成参数版本、更新步数、采样设置，以及所需的行为概率或重算条件。不能仅以训练 loss 下降判断同步正确。

官方接入指南建议至少检查两次在线 optimizer step，使第二次权重重载也被覆盖，并比较同步后的 actor 与 rollout token log-probability。这可以作为后续实际对接的检查任务。[接入检查](https://github.com/verl-project/verl/blob/7cea5392165ac26d5eadf879701337c85e4a2b2f/docs/advance/fsdp_extension.rst)

# 与我们的研究连接

在线蒸馏能够提供能力迁移的训练信号，但不会自动证明循环状态更有用，也不保证恢复原有 CoT 能力。仍需明确更新哪些参数、梯度是否到达接口、部署时学生是否依赖帮助。TaH2 与 verl 的实际对接是后续实现工作。

# 围绕本课讨论

```codex-prompt
请带我画连续两轮在线蒸馏的时序，逐项标注学生轨迹、教师所见前缀、损失、梯度和参数版本。随后对照固定版本 verl 的直接蒸馏与策略梯度分支，最后讨论接入 TaH2 时需要补哪些接口。
```

# 继续学习

[上一课：从训练到自主生成](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9) · [学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133) · [回到框架设计](https://chatgpt.com/space/page_972238e454f88191a1c09dfe60350931)
