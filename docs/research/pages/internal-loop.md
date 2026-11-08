一次内部循环把当前隐状态变成下一轮输入，同时更新活动位置、KV 缓存和读出。本课先定义这些状态与继续规则，再推导 updater、decider 和停止混合，最后沿主前向与辅助探测算法说明梯度和缓存如何流动。

[返回学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
>
> **阅读主线**：状态与继续规则 → updater 和 decider → KV 可见性 → 停止混合 → 深度标签与联合损失。用数学定义理解依赖，用逐行算法跟踪控制流，再到源码检查 shape、梯度和缓存副作用；混合权重的归一化证明位于读出部分。

# 状态与更新的数学定义

## 状态与活动集合

以预测位置 t 为索引，并在有效的移位监督位置集合 I 上定义下一 token 目标；文本末位没有目标，按 ignore 处理。这与模型内部移位后的 labels 对齐。下文省略 batch 索引，KV 始终限于同一条样本。深度从1开始，参考分支采用确定性 gate，不含深度 override。

```math
\begin{aligned}
E&\in\mathbb R^{B\times T\times d},\\
A_m&\subseteq\{(b,t):\text{位置有效}\},\\
S_m&=(E,H_m,p,m,A_m,\mathcal C_m,\nu_m),\\
y_t^*&=\texttt{input\_ids}_{t+1}.
\end{aligned}
```

E 保存原始 token embedding；H_m 仅在本轮实际执行的位置有定义；p 是逻辑位置；C 是按 layer、深度与位置组织的 KV；nu 是有效性信息。工程可以压紧活动位置，也可以用 dense 缓冲加 mask 表示这些部分定义的量。

```math
\begin{aligned}
X_{1,t}&=e_t,\\
(H_m,\Delta\mathcal C_m)
&=\operatorname{Backbone}_\theta
(X_m,\mathcal C_{<m};\mathcal V_m),\\
q_{m,t}&=\operatorname{softmax}(W_{\rm out}h_{m,t}),\\
X_{m+1,t}&=U_\phi(e_t,h_{m,t})
\quad (t\in A_{m+1}).
\end{aligned}
```

Backbone 不只是对一个向量应用函数：它同时接受活动 token、缓存与可见性规则，并返回状态和 KV 增量。updater 的输入是原始 embedding 与当前末层状态；跨轮梯度是否保留属于计算图定义。

## 继续概率与实际路由

```math
\begin{aligned}
g_{m,t}&=\sigma\!\left(
D_\psi(e_t,h_{m,t},\operatorname{TopK}(q_{m,t}))\right),\\
A_{m+1}&=\{t\in A_m:g_{m,t}>\tau\}\quad(m<M),\\
d_t&=\min\{m:m=M\ \text{或}\ g_{m,t}\le\tau\}.
\end{aligned}
```

部署决策由当前状态、预测分布和阈值决定。所选 Qwen3MLPIterDecider 不读取真目标标签；标签用于训练期的后验监督。其他 oracle decider 或随机路由是不同算法，需要另行定义。

## 实际路由的分支顺序

实际执行先保存 gate logits，再依次应用深度 override、causal 监督位置规则和可选的图对齐规则，由此确定 finished mask。连续混合仍读取原 gate logits：强制某个执行深度，并不会自动把混合中的 gate 概率改成0或1。上面的阈值公式描述其中常用的确定性分支。[调用顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L694-L775)

| 分支 | 生效条件与行为 |
| --- | --- |
| 深度上限 | decider 达到 max_iter 后返回停止；名为 MINUS_INFINITY_LOGITS 的哨兵实际为 −10，尾部质量由 runtime 的 finished 分支吸收 |
| 训练 warmup | 仅训练且仍在 warmup 时，阈值从 0 线性升至配置 threshold |
| 可选 Bernoulli | stochastic_sampling=True 且没有进入 warmup 分支时采样；该 elif 没检查 self.training，单独调用 eval() 不会关闭它 |
| 显式深度重放 | 正的 iter_count 用 count>当前深度覆盖布尔决策；count=0 保留原决策；不替换已保存的 gate logits |
| causal 监督规则 | 有移位 labels 时，无监督预测位置被要求首轮后停止；纯推理无 labels 时不进入此分支 |
| 可选 FSDP 图对齐 | 训练、开启 layer_wrap、无显式 iter_count、尚未到上限且全体停止时，选 gate logit 最大的一个有效活动位置继续 |

默认循环训练的 layer_wrap=False；单轮模型默认 True，可由配置显式覆盖。最后一行是可选工程分支，不能当成默认 TaH2 的路由规则；它在 causal 规则之后执行，trace 应记录最终实际路由。[decider](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L196-L267) · [override](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L458-L480) · [训练入口默认值](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py#L125-L135)

updater 的 prev_inputs 在主调用中接收 active_first_iter_embeds，含义是原始嵌入。阅读递推式时应沿这条调用关系理解输入，不能按参数名称将它误读成上一轮输入。[updater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py) · [调用现场](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)

## Updater 与 decider 的精确映射

以下把固定源码的 Qwen3 MLP 组件写成矩阵运算。令隐藏宽度为 d、中间宽度为 i；RMSNorm 的可学习尺度为向量 s。默认激活是 SiLU，线性层无 bias。定义：

```math
\begin{aligned}
N(x;s)&=\operatorname{diag}(s)x/r(x),&
r(x)&=\sqrt{\|x\|_2^2/d+\epsilon},\\
F(x)&=W_{\rm down}
\bigl[\operatorname{SiLU}(W_{\rm gate}x)
\odot(W_{\rm up}x)\bigr].
\end{aligned}
```

Updater 对原始嵌入 e 与末层状态 h 分别归一化，拼接后投影，再执行 MLP。输出直接成为下一轮输入，源码没有额外加上 e 或 h 的 residual carrier：

```math
\begin{aligned}
v&=W_eN_e(e)+W_hN_h(h),\\
x&=N_{\rm pre}(v),\\
U_\phi(e,h)&=N_{\rm out}(F(x)).
\end{aligned}
```

关闭某个可选 norm 时，把对应 N 换成恒等映射。构造函数中 `enable_norm_pre_mlp` 与 `enable_norm_out` 均默认为 True；该文件较早的说明文字把后者写成默认关闭，复现应以构造参数与 checkpoint 的 `init_args` 为准。[Qwen3MLPUpdater 初始化与 forward](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py#L66-L105)

这些归一化调整了状态尺度，但不能单独证明循环收敛。固定可微分支时，RMSNorm 与 updater 的局部 Jacobian 为：

```math
\begin{aligned}
J_N(x)&=\operatorname{diag}(s)
\left[\frac{I}{r}-\frac{xx^\top}{d\,r^3}\right],\\
J_F(x)&=W_{\rm down}\Bigl[
\operatorname{diag}(W_{\rm up}x)
\operatorname{diag}(\operatorname{SiLU}'(W_{\rm gate}x))W_{\rm gate}\\
&\hspace{27mm}+
\operatorname{diag}(\operatorname{SiLU}(W_{\rm gate}x))W_{\rm up}
\Bigr],\\
D_hU&=J_{N_{\rm out}}J_FJ_{N_{\rm pre}}W_hJ_{N_h},\\
D_eU&=J_{N_{\rm out}}J_FJ_{N_{\rm pre}}W_eJ_{N_e}.
\end{aligned}
```

各 Jacobian 在上一式对应的中间量处求值。只有进一步约束这些矩阵的谱范数及 backbone、KV 依赖，才可能得到完整循环的收缩条件。原始嵌入在每轮重复进入 updater，因此它也形成多条参数梯度路径；不能把 updater 近似成只依赖前一隐藏状态的一元变换。

Decider 的第三路输入是词表 softmax 后最大的 K 个概率值，不含这些值的 token ID。固定实现将该向量直接送入宽度 d 的 RMSNorm，再与两个 d 维状态拼接；因此必须满足 K=d，词表不足 K 时先补零。其 MLP 前没有 updater 的那层可选 pre norm：

```math
\begin{aligned}
q&=\operatorname{softmax}(z),\qquad
u=\operatorname{pad}_{K}(\operatorname{TopKValues}(q)),\\
v_D&=W_D[N^D_e(e);N^D_h(h);N^D_q(u)],\\
s_D&=w_D^\top N^D_{\rm out}(F_D(v_D)),\qquad g=\sigma(s_D).
\end{aligned}
```

该概率特征没有 detach。若局部 top K 索引集合 S 不变，选择矩阵 P_S 给出下面的导数；未被选中的词表 logit 仍可通过 softmax 分母收到梯度。并列排序边界需按所执行的选择分支理解。[decider 张量路径与路由](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L152-L267)

```math
\frac{\partial q_S}{\partial z}
=P_S\bigl(\operatorname{diag}(q)-qq^\top\bigr),
\qquad
\frac{\partial q_i}{\partial z_j}=-q_iq_j
\quad(i\in S,\ j\notin S).

```

两层可选 norm 都启用时，updater 参数量是 `2d²+3di_U+4d`，decider 是 `3d²+3di_D+5d`。当两者中间宽度均等于 d，合计为 `11d²+9d`；4B 配方的 d=2560 对应 **72,112,640 个侧组件参数**，不计骨干。这是按矩阵形状推导的计数。[组件定义](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py#L66-L105) · [decider 定义](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L131-L180) · [4B 配方](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml#L18-L27)

## 可选初始化：uniform_init 没有生成均匀 gate 概率

可选 uniform_init 以正态权重控制 score 的初始化尺度。Qwen3MLPIterDecider.__init__ 在该标志为 True 时调用 normal_(std=π/√(3d))，为 False 时使用 std=0.01；它没有直接采样 Logistic logit。1.7B／4B／8B 的默认 sft_tah.yaml 均未设置该标志，类默认 False。以下条件分布说明这一可选分支，不能据此解释论文默认训练的效果。[原构造函数](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) · [配方到组件参数](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_model.py)

先在理想的实数正态采样下分析这一初始化。固定输入与其他参数，记 a 为 score 前的 MLP 输出，q=‖a‖²/d>0，并令归一化的 gain 初始为 1。归一化后的特征为 ξ=a/√(q+ε)。若最终 score 权重 w 与 ξ 独立，且 w∼N(0,σ_w²I)，就有下面的条件分布：

```math
\begin{aligned}
s_D\mid\xi&\sim\mathcal N(0,v),\qquad v=\sigma_w^2\|\xi\|^2,\\
\sigma_w^2=\frac{\pi^2}{3d}&\quad\Longrightarrow\quad
v=\frac{\pi^2}{3}\frac{q}{q+\epsilon}.
\end{aligned}
```

令Φ是标准正态CDF。对0<t<1，继续分数g=σ(s_D)的条件分布为logistic-normal，而非Uniform(0,1)：

```math
\Pr(g\le t\mid\xi)=\Phi\!\left(\frac{\log(t/(1-t))}{\sqrt v}\right),\qquad
\mathbb E[g\mid\xi]=\tfrac12,\qquad \Pr(g>\tfrac12\mid\xi)=\tfrac12.
```

当q远大于ε时，在t=0.1和0.25处CDF约为0.11287和0.27236；若真为均匀分布，则应分别为0.1和0.25。让logit具有Logistic(0,1)分布才会经sigmoid得到精确均匀分布；只匹配其方差不够。q=0时g退化为0.5，上述除以√v的CDF式不适用。

上式的 Pr(g>1/2|ξ)=1/2 是对随机初始化求得的概率。同一个模型在不同 token 和轮次中共享 score 权重，所以这个结论不代表初始化完成后恰有一半 token 继续，也不保证各次继续决策是独立的 Bernoulli 变量。对两份都与 w 独立的固定特征，其 logit 协方差为：

```math
\operatorname{Cov}_{\rm init}(s_i,s_j\mid\xi_i,\xi_j)=\sigma_w^2\xi_i^\top\xi_j.
```

例如ξ_i=ξ_j时两项logit完全相同。若后续特征已经受前一轮gate选择影响，就不能继续假定ξ与w独立。以上概率是对随机score初始化的分布，不是单次checkpoint上的token频率、训练后校准结果或实际停止深度分布。

实际构造默认建立 BF16 score，再调用 to(dtype)；训练入口随后将整体参数转为 FP32 主权重，已发生的舍入不会被恢复。[入口与 dtype 顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py)。因此，上式是理想连续正态采样的参考律，真实张量分布还需按设备和 dtype 测量。原构造函数的替身检查只验证分支、std 与 cast 调用，没有验证 BF16 抽样精确满足该分布。

工程核对顺序是：保存flag与seed→记录score生成dtype和后续cast→对固定首轮输入分别统计logit/gate分布与共享相关性→将参数初始化重复与单模型token统计分开→单独检查阈值、warmup和实际执行深度；不因flag名而假设初始化已校准。

# 梯度与缓存分别看

主路径调用 updater 时没有自动 `detach`，因此已执行轮次之间可以传递梯度；布尔路由选择本身不是可微的连续运算。决策器怎样学习，还要追踪监督损失和连续混合权重。

辅助探测同时使用 `torch.no_grad()` 与主缓存的 snapshot。前者控制计算图，后者处理缓存读写边界。关闭梯度不会自动阻止函数修改缓存。[辅助探测](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)

`finalize_logits` 返回按深度累计的混合 log 值；在理想未 clamp、无舍入时，它是归一化的 log-probability。源码对权重设下限，target-only 与全词表路径还采用不同的缓冲精度，严格归一化应另行核对。训练采用目标 token 累计路径时不一定构造完整词表混合结果。[概率质量与损失分支](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647)

归一化可以调整数值尺度，但尺度合适不等于下一轮已经会使用信息。接口是否形成可继续使用的中间状态，仍然需要后续训练与干预来判断。

# 对照源码

固定版本：TaH `78ddf9a2c25713fced03a736c5746161f6ee590c`。

- [input_updater.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py)：先读 `Qwen3MLPUpdater.forward`，画出张量变换。

- [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)：再找 `TaHForCausalLM.forward` 中调用 updater 和更新 `next_iter_mask` 的位置。

- [tah_runtime.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)：对照 `capture_posterior_inputs`、`_run_side_score_pass`、`accumulate_iter` 和 `finalize_logits`。

- [loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py)：分清预测损失和决策损失怎样进入同一参数更新。

# 位置与深度的注意力可见性

## 主路径可见性

DUO 主路径的可见性集合如下；只纳入已经实际执行、有效且同样本的 KV。position 与 depth 是两个独立条件。

```math
\mathcal V(t,m)=
\{(s,j):s\le t,\ j\le m,\ (s,j)\text{已执行且有效}\}.
```

## 辅助探测可见性

lookahead 的假想深度使用专门规则：它可以读真实的先前深度；在当前假想深度，只能读真正继续的位置或查询自身，且仍受文字因果条件约束。它不能读取其他停止位置的假想状态。

```math
\begin{aligned}
\mathcal V_{\rm probe}(t,m)
=\{(s,j):s\le t,\ \text{有效},\
[j<m\ \lor\
(j=m\land(c_s=1\lor s=t))]\}.
\end{aligned}
```

键域同时要求：j<m 的键来自 snapshot 中真实主路径已执行的条目，j=m 的键来自当前 side pass。这里 c_s 表示该位置是否在主路径真正继续到当前深度，并非后验监督标签。源码采用零起始缓存深度；核对实现时先统一到这里的一起始记号。[posterior mask](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L216) 明确实现了这一例外。 数学架构对照[TaH2 原论文第4节](https://arxiv.org/html/2609.35748v1)。

## 辅助探测怎样保持上下文

DUO 的 side pass 同时重算已存输入中的停止位置和真实继续位置；继续位置提供当前深度的合法 KV 上下文。只重算停止位置会缺失这部分上下文，而让所有假想停止位置互相可见又会增加不存在的依赖。评分写回覆盖全部 forward 位置，因此继续位置的 CE 也会被 side pass 结果替换。[完整前向集合与评分写回](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L54-L211)

这里要依次回答“是否调用探测”和“调用哪些位置”。先在 DP 组检查是否存在缺失 CE；没有缺失时，跳过这一轮 side backbone。若进入 DUO side pass，前向集合 F_m 包含全部已存 side inputs，范围不限于缺失评分的位置。当 m<M 时，side pass 末尾还会为所有有效前向位置计算 updater：下一深度缓冲按位置覆盖输入值，mask 则与已存 mask 取并集：

```math
\mathcal F_{m+1}
\leftarrow \mathcal F_{m+1}\cup\mathcal F_m
\qquad\text{当深度 }m\text{ 的 side pass 执行且 }m<M.

```

因此深层探测集合可能大于上一轮真实执行集合。例如 M=3，a 在主路径停于第一轮、b 停于第二轮；为补 a 的第二轮 CE，DUO side pass 转发 {a,b}，并把两者第三轮输入都存下。若 b 仍是第三轮的缺失评分候选，下一次 side pass 的实际转发集合仍可为 {a,b}，尽管主路径第二轮只执行了 b。a 的假想当前轮 KV 不对 b 开放，额外计算与真实可见性是两个不同问题。

这也要求成本按实际 F_m 和调用指示计数。主前向的 capture_posterior_inputs 为下一轮准备输入，与 side pass 末尾的下一层 updater 是两处独立调用；最后探测深度 m=M 不再准备下一层。论文 Eq.23 的 N_m 账本与固定实现的具体执行范围分别说明，不能直接把前者当作后者的 kernel 次数。[调用触发](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L214-L314) · [side updater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L191-L211) · [输入合并与主路径准备](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L465-L517) · [论文成本与代码边界](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb)

以下过程对应一次指定深度的辅助评分。深度 m 使用一起始数学记号，函数内部转为零起始 `iter_depth=m-1`。探测缓存包含主路径深度小于 m 的条目；本轮新产生的条目只写入临时容器。

### 算法 T1 DUO 辅助评分与缓存隔离

**输入** 主路径缓存、深度 m、已存 side inputs、实际继续掩码、缺失分数的目标位置、移位标签与位置。

**输出** 深度 m 的 detached CE 表，以及需要时供下一探测深度使用的 detached updater 输入；主路径缓存内容保持原值。

**前置条件** m≥2；调用方已确认 DP 全局存在缺失 CE；保存的输入始终锚定同一原始 embedding；同一 DP 组按既定边界顺序进入探测与对齐路径。实际转发集合取已存 side inputs，包含按 OR 合并保留的位置。

![算法 T1 DUO 辅助评分与缓存隔离](visualize:fde1_bGliZmlsZV9zOU1qd19zOUc2TF9qRjVSMVBJTFhn_FileDrive_45288dfe06608191af94334ba8699511)

**源码** [`_run_side_score_pass`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L54-L211) · [`snapshot_iter`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L402-L422) · [posterior attention mask](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L216-L308)

缓存隔离与 backward 重计算是两个不同机制。side pass 在 no_grad 中使用 detach 后的 snapshot；gradient checkpoint 的重计算则需要保留梯度，并临时启用只读缓存。该上下文还恢复前向时捕获的深度，避免 backward 恰好使用主前向结束时的深度。

| 路径 | 梯度规则 | 缓存行为 |
| --- | --- | --- |
| 主前向 | 保留已执行计算图 | 追加或替换缓存容器中的张量引用 |
| 后验 side pass | no_grad，旧 KV detach | snapshot 共享旧存储，新 KV 写入临时容器 |
| checkpoint 重计算 | 重新建立所需计算图 | 返回较浅缓存加当前重算 KV，不向主缓存再次追加 |

主缓存正常追加使用 torch.cat 后替换引用，因此 snapshot 共享旧张量的约定能够成立；若改成原地覆盖旧存储，就必须同时改变 snapshot 的所有权设计。当前只读重算分支读取的是严格较浅深度的缓存，不应未经分析就推广到携带同深度历史前缀的分块训练。[缓存 update 与只读上下文](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L11-L137) · [checkpoint 调用](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L45-L66)

## 两位置示例与缓存边界

两位置、两深度均已执行时，下面是主路径的教学可见性表。实际 padding、停止位置与探测状态还需增加相应限制。

| 查询与键 | t1 深度1 | t2 深度1 | t1 深度2 | t2 深度2 |
| --- | --- | --- | --- | --- |
| t1 深度1 | 可见 | 不可见 | 不可见 | 不可见 |
| t2 深度1 | 可见 | 可见 | 不可见 | 不可见 |
| t1 深度2 | 可见 | 不可见 | 可见 | 不可见 |
| t2 深度2 | 可见 | 可见 | 可见 | 可见 |

沿 [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) 的 `_init_position_ids`、`_build_attention_mask_for_iter` 与 [causal_cache.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py) 的 `update`、`get_cache_upto_iter` 核对实际存储。每个条目至少追踪逻辑位置、深度、layer、有效 mask、KV 是否已应用位置编码；不能把循环一次自动等同于文字位置前进一次。

lookahead 是额外监督路径：停止位置的探测不能彼此建立原本不允许的依赖。`snapshot_iter` 创建新的缓存容器，但共享并 detach 已有 KV 张量；它依赖后续追加、不原地修改旧条目的约定。未来修改缓存写法时，要同时检查别名和副作用。[缓存 snapshot](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L402)

# 停止决策与概率读出

## 停止权重与混合分布

一旦某个位置决定继续，最终预测怎样使用已经算过的各轮分布？TaH2 保留这些分布，并让它们共同组成读出。这里需要分清两个动作：[Qwen3MLPIterDecider.forward](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) 用继续分数和阈值决定下一轮是否实际执行；[runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py) 再为已经执行的轮次分配混合权重。

下式的 u_m,t 表示尚未分配给较浅轮次的混合质量，首轮从 1 开始。若继续分数为 g_m,t，本轮保留其中的 1−g_m,t，剩余部分传给下一轮；到了实际停止深度 d_t，就把全部剩余质量交给当前轮。u 是这个混合过程的累加状态，实际执行哪些轮次仍由路由规则决定。

```math
\begin{aligned}
u_{1,t}&=1,\qquad u_{m+1,t}=u_{m,t}g_{m,t},\\
\omega_{m,t}&=u_{m,t}(1-g_{m,t})\quad(m<d_t),\\
\omega_{d_t,t}&=u_{d_t,t},\\
q_t&=\sum_{m=1}^{d_t}\omega_{m,t}q_{m,t}.
\end{aligned}
```

## 权重归一化的证明

归一化可以直接沿这个分配过程看出来：每个非末轮拿走的质量，正好是 u_m,t−u_m+1,t；实际末轮再拿走 u_d,t。把这些项相加，中间的剩余量逐项抵消，只留下首轮的 u_1,t=1。每项非负，而各轮 q_m,t 本身是归一化分布，所以最终概率混合也归一化。

关键是“实际末轮”吸收尾部质量。如果某个位置在最大深度之前停止，未执行的轮次没有预测可供混合，剩余质量就在当前轮结清；证明并不要求它算满最大深度。这里使用未 clamp、无舍入的理想定义，源码权重下限和有限精度的影响见“一次参数更新”的归一化分支分析。

```math
\begin{aligned}
\sum_{m=1}^{d_t}\omega_{m,t}
&=\sum_{m<d_t}(u_{m,t}-u_{m+1,t})+u_{d_t,t}=1,\\
\log q_t(y_t^*)
&=\operatorname{logsumexp}_{m\le d_t}
\bigl(\log\omega_{m,t}+\log q_{m,t}(y_t^*)\bigr).
\end{aligned}
```

训练一个已知目标词 y 时，我们只需混合分布在 y 上的概率。上式第二行因此可以直接在各轮的目标 logp 上做 logsumexp；target-only 路径只保存这个累计结果，节省完整词表的混合缓冲。每一轮得到正确的 q_m,t(y) 仍依赖词表归一化，decider 也仍需要门控输入，词表读出没有因此消失。

这也说明 gate 可以从哪里学习：硬阈值决定是否执行，其布尔选择本身不可微；已执行路径上的连续混合权重却仍在计算图中，辅助监督也沿另一条损失路径训练决策器。跟踪梯度时，要分别沿这两条连续路径回看，不能仅凭硬路由不可微就判断 gate 没有训练信号。

用一个两轮教学示意看完整过程：最大深度为 2，确定性阈值为 0.5，温度为 1，采用归一化停止混合且没有资源覆盖。先固定此前 token 已经形成的状态，再沿当前预测位置追踪；当前位置仍按 DUO 规则产生和读取各轮 KV。

首轮从原始 embedding 得到 h₁ 和词表分布 q₁。若 decider 给出 g₁=0.7，这个位置便继续；updater 把原始 embedding 与 h₁ 合成第二轮输入，骨干据此得到 q₂。此时达到最大深度，最终读出为 0.3q₁+0.7q₂。同一个 0.7 参与两步：先决定第二轮是否发生，再决定已执行的两轮怎样共同贡献预测。

若 g₁=0.4，这个位置在第一轮停止，第二轮不执行，读出直接是 q₁。要先确定实际得到哪些分布，才能讨论它们怎样混合。这里混合的是概率；先线性组合 hidden 再做 softmax，通常会得到不同结果。

## 随机路由与停止混合叠加后的真实边缘分布

固定 MiniSGL 的 sample 分支先按 gate 抽样“是否继续”，到实际停止轮后又按 stop_prob_mix 读出。两次使用 gate 对应两个不同随机／加权对象。下面给出这一可选分支的单 token 条件结论；论文主配方的确定阈值不能直接替换为此随机模型。[Bernoulli 决策与混合读出](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

**条件。** 先固定当前 token 的入态，包括此前路由产生的全部 KV。给定这个入态，设始终继续至 M 的前缀链唯一，链上的每轮分布和 gate 均确定。若骨干还含随机性，先固定与路由随机抽样独立的那部分随机性，再分析这条条件链。以下令每轮条件继续率确为 g，温度为 1，最终从归一化概率混合中抽取文字；不加入 top-k／top-p、资源强停、warmup、override、权重下限或数值舍入。M 为正整数，末轮强制停止。

```math
\begin{aligned}
R_1&=1,\qquad R_{j+1}=R_jg_j\quad(j<M),\\
\Pr(D=d)&=\begin{cases}R_d(1-g_d),&d<M,\\R_M,&d=M,\end{cases}\\
P_d(v)&=\sum_{j<d}R_j(1-g_j)q_j(v)+R_dq_d(v).
\end{aligned}
```

D 是真正执行的轮数，P_d 是在 d 停止时的文字分布。对同一固定入态边缘化 D，可得到一个封闭形式：

```math
\begin{aligned}
\bar P(v)&=\sum_{d=1}^{M}\Pr(D=d)P_d(v)\\
&=\sum_{j<M}R_j^2(1-g_j^2)q_j(v)+R_M^2q_M(v),\\
R_j^2(1-g_j^2)&=R_j^2-R_{j+1}^2.
\end{aligned}
```

**证明。** 对 j<M，第 j 个分量在 D=j 时吸收 R_j，在 D>j 时贡献 R_j(1−g_j)。其边缘系数因此是 R_j·Pr(D=j)+R_j(1−g_j)·Pr(D>j)=R_j²(1−g_j²)。末轮系数为 R_M·Pr(D=M)=R_M²；望远镜求和为 1。这是由固定源码两条规则推导出的性质，不是原文额外报告的实验结果。

二轮例子取 g₁=0.5、q₁=(0.8,0.2)、q₂=(0.2,0.8)。固定执行两轮的混合为 (0.5,0.5)；随机 D 再作停止混合的文字边缘为 (0.65,0.35)。二者使用同一 gate 数值，输出分布仍不同。不能把 log P_D、log Pr(D) 或 log P̄ 当作同一个评分字段。

```math
\mathbb E[D]=\sum_{j=1}^{M}R_j,\qquad
\mathbb E[C_D]=\sum_{j=1}^{M}R_jc_j,
\qquad C_d=\sum_{j\le d}c_j.
```

c_j 是在这一固定入态和前缀链上执行第 j 轮的确定增量成本。平方出现在边缘读出的分量权重；真实访问概率仍为 R_j，计算成本不能用 R_j² 记账。二轮例子的平均实际深度为 1.5。服务批次中的填充、额外采样和资源压力需另按执行 trace 计费。

温度或截断变换作用于每个 P_d 时，应先变换，再以 Pr(D=d) 求和；通常不等于先求 P̄ 再变换。不同 D 还会留下不同 KV，所以跨 token 的文字边缘必须维护历史路由后验，不能把上式在某一条缓存上逐 token 相乘。[完整路由评分 T6](https://chatgpt.com/space/page_a04bfe2924d48191b95083d418d93368) · [生成语义与系统强停](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)

### 算法 T9 单 token 随机路由的读出与成本

**输入** 同一固定入态、同一全继续前缀链上的 q₁…q_M、g₁…g_{M−1}，以及非负增量成本 c₁…c_M。各 q 为相同词表上的归一化概率；不可拼接不同 KV 历史的中间输出。

**输出** 各停止深度的概率与条件读出、真实文字边缘、固定 M 读出、平均实际深度与条件成本，以及枚举式和封闭式的一致性残差。

**前置条件与终止** 使用上文理想采样条件。最多处理 M 轮；给定输入后时间为 O(MV)、工作缓冲 O(V)。若保留全部条件分布作审计，另需 O(MV) 存储；获取各轮 q 与 g 的骨干前向成本不包括在这个代数算法中。

![算法 T9 单 token 随机路由的读出与成本](visualize:fde1_bGliZmlsZV9RUmtSZXdXYnVZUkxvZDhFN0w4V0Zn_FileDrive_45288dfe06608191af94334ba8699511)

**性质与来源** 这是从固定引擎的 Bernoulli continuation、继续累计及终止尾质量吸收推导的数值参考，不是新增的生产采样器，也不意味着 TaH2–verl 已完成接入。[固定 dispatch](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

**教学核对。** 256 组含 g=0／1 和 M=1 边界的小词表案例，将停止深度枚举、平方生存系数、分布归一化和真实访问成本分别比较；边缘分布最大误差 2.23×10⁻¹⁶，归一化误差 3.34×10⁻¹⁶，访问成本误差 3.56×10⁻¹⁵。这不构成已加载模型的行为或性能测量。

# 在线深度标签与损失的数学定义

I 是有效监督位置集合，A_m 是本轮实际执行位置，z 的前一边界标签决定下一边界的候选资格，初值 z_0=1。停止位置缺失的下一轮预测由隔离的无梯度 lookahead 补齐。

```math
\begin{aligned}
\ell_{m,t}&=-\log q_{m,t}(y_t^*),\\
\delta_{m,t}&=\ell_{m,t}-\ell_{m+1,t},\\
C_m&=I\cap A_m\cap\{t:z_{m-1,t}=1\}.
\end{aligned}
```

在整个 DP 组的候选中收集有限的正改善，降序写为 delta_(1) 至 delta_(n)。覆盖系数取 0 到 1 之间的正数；cutoff 定义为达到指定正改善总量的最小前缀末值：

```math
\begin{aligned}
k_m^*&=\min\left\{k:
\sum_{r=1}^{k}\delta_{(r)}
\ge\rho\sum_{r=1}^{n}\delta_{(r)}\right\},\\
\kappa_m&=\delta_{(k_m^*)},\\
z_{m,t}&=\mathbf1\{t\in C_m,\ \delta_{m,t}\ge\kappa_m\}.
\end{aligned}
```

没有正改善时，全设为停止。cutoff 处若并列，按阈值纳入的全部位置可能使覆盖量超过 rho；rho 约束改善量，不是 token 数比例。[标签实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py)中的候选掩码、DP 汇集与边界选择应分别追踪。

## 覆盖选择的最优性范围

对有限且为正的改善量，降序前缀在“每个位置的额外计算成本相同”的前提下，给出满足覆盖目标所需的最小位置数。证明很直接：任意 k 个位置的改善总量，都不超过最大的 k 个改善之和；若前 k−1 个仍不足，其余任何 k−1 个也不足。源码以 cutoff 比较而不是显式取前 k 个，所以并列值会扩大选集。例如改善为 0.50、0.25、0.25，覆盖 75% 的最小前缀只有两个位置，而 cutoff=0.25 会选中全部三个。

该结论不等价于最小 FLOPs 或最短延迟。假设两个位置的改善分别为 0.6、0.4，成本分别为 100、1，覆盖 40% 时按改善排序先选成本 100 的位置，而只选第二个位置已经满足覆盖。因此在动态 KV 长度、不同 kernel 批形状下，原规则应解释为正改善覆盖启发式，不能自动称为最优计算分配。排序、FP64 汇集与阈值比较的对应位置见[coverage 实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L325-L401)。

源码的 cost_sensitive 权重由候选改善到 cutoff 的距离给出，而不是由 GPU 时间或 FLOPs 给出。候选外的位置填最小权重；这些位置只要实际执行且有监督，仍可能进入 BCE。

```math
\begin{aligned}
a_{m,t}&=
\begin{cases}
\operatorname{clip}(|\delta_{m,t}-\kappa_m|,10^{-6},1),
&t\in C_m,\\
10^{-6},&t\notin C_m,
\end{cases}\\
z_{m,t}&=\mathbf 1\{d_t^{\rm label}>m\}.
\end{aligned}
```

cutoff 附近的标签权重较小，远离边界的标签权重较大；权重随后 detach。收益是当前条件上下文中的一步 CE 差，不能直接解释成多轮未来收益或最终答案奖励。[cost 与深度标签](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L284-L309)

联合目标分开写成混合预测 NLL 和决策 BCE。a 为停止梯度的代价权重，beta 为可选的正类权重；BCE 的监督范围是所有已执行且有效监督位置，不仅是正候选集合。

```math
\begin{aligned}
{\rm BCE}_{\beta}(g,z)
&=-\beta z\log g-(1-z)\log(1-g),\\
L_{\rm NLL}
&=-\frac1N\sum_{t\in I}\log q_t(y_t^*),\\
L_D&=\frac1N\sum_{m<M}\sum_{t\in I\cap A_m}
a_{m,t}\,{\rm BCE}_{\beta_m}(g_{m,t},z_{m,t}),\\
L&=L_{\rm NLL}+\alpha_D L_D.
\end{aligned}
```

标签、cutoff 和监督权重不反向传播；已保存的 gate logits、连续混合权重和已执行 updater 路径可以承接损失梯度。监督窗口要求 N>0，并采用训练器传入的归一化约定；当前 forward 内的标签统计和动态正类权重仍可能随微批次划分改变。[损失实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py)

## 加权 BCE 的梯度与概率含义

令 s 为保存的 gate logit，z 为 detached 二元标签，a 为代价权重，beta 为当次 forward 的正类权重。可选 margin b>0 时，源码先把正类 logit 减 b、负类加 b，再计算 BCE；b=0 是普通分支。对一个决策的导数为：

```math
\begin{aligned}
s_{\rm eff}&=s-b(2z-1),\qquad p_{\rm eff}=\sigma(s_{\rm eff}),\\
\ell_D&=a\left[-\beta z\log p_{\rm eff}-(1-z)\log(1-p_{\rm eff})\right],\\
\frac{\partial\ell_D}{\partial s}
&=a\left[(1-z)p_{\rm eff}+\beta z(p_{\rm eff}-1)\right].
\end{aligned}
```

总代价还乘辅助损失系数，并按传入的监督 token 分母 N 归一化；分母不是决策次数。因此执行更多边界会增加辅助项的决策数，其尺度不能由单个 BCE 的大小推断。[BCE 与归约](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L463-L570)

在 b=0、固定 beta、给定 gate 特征 x 的条件分布且可逐点自由拟合时，最优 sigmoid 输出满足：

```math
\sigma(s^*(x))=
\frac{\beta\,\mathbb E[az\mid x]}
{\beta\,\mathbb E[az\mid x]+\mathbb E[a(1-z)\mid x]}.

```

只有再满足相应的权重条件，它才等于未经加权的标签条件概率。若 a 在给定 x 后与标签无关，记 eta=P(z=1|x)，则最优输出为 beta·eta/(beta·eta+1−eta)；阈值 0.5 对应 eta>1/(1+beta)。例如 beta=4、eta=0.2 时输出为 0.5。这一推导解释了类别平衡会改变概率刻度；实际联合 NLL 训练、共享参数、有限容量和动态 beta 又增加了耦合，不能把该点态最优式当成整个 TaH2 模型的已证最优解。

源码日志的决策准确率与 TP/FP/FN 以 sigmoid(s)>0.5 计数，并未复用配置 threshold、训练 warmup 阈值或随机路由结果。修改部署阈值后，这组日志仍是 0.5 分类指标，应另外统计实际路由。[指标边界](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L188-L305)

### 算法 T2 多深度标签与加权监督

**输入** 每深度 CE 表、实际执行集合、有效监督位置、逐深度覆盖率、保存的可微 gate logits、DP 组与正类权重配置。

**输出** detached 深度标签及代价表、按 token 分母归一化的决策损失。

**前置条件** 用于覆盖论证的 CE 有限；所有 DP rank 进入相同的边界 collective 序列；N>0；深度候选已满足真实执行上限。

![算法 T2 多深度标签与加权监督](visualize:fde1_bGliZmlsZV9zN2FZRkRMR0g1YVlKS1dTVE9GeDBn_FileDrive_45288dfe06608191af94334ba8699511)

**源码** [`finalize_posterior_labels`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L230-L310) · [`_coverage_threshold`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L370-L401) · [`IterDeciderLoss.final_loss_func`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L307-L570)

## 收益覆盖的数值例子

教学例中，三个候选的正改善为 `0.80、0.19、0.01`，总量为 1。按精确实数计算，覆盖 99% 只需前两个，cutoff 为 0.19。随后 [IterDeciderLoss](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py) 使用标签及代价权重训练决策器；这与用最终答案奖励做 RL 是不同训练信号。

## 单轮 CE 与混合读出改善的精确关系

原文 Eq.5 和 `compute_posterior_score` 使用单轮预测的 CE 差；Eq.4 和 `accumulate_iter` 使用已执行深度的停止权混合。以下关系由两者联立推导，说明这个监督信号如何对应实际读出；它不是原论文另报的实验结果。[原文 §4.1–4.2](https://arxiv.org/html/2609.35748v1#S4) · [原始 logits 的 CE](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L312-L368) · [末轮吸收剩余质量](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L579)

固定一个预测位置、移位目标 y、进入深度 m 的全部上下文，以及该位置的继续链。a、b 为相邻两轮赋给 y 的概率，A 为较浅轮次已累积的目标概率质量，R 为进入 m 的剩余权重，g 为 m 的继续概率。比较“立即停于 m”与“只再算一轮并停于 m+1”；较浅状态、gate、上下文和其他决策固定。采用精确算术，关闭权重下限、温度、截断采样和资源强停，且 a、b>0。

```math
\begin{aligned}
a&=q^{(m)}(y),\qquad b=q^{(m+1)}(y),\\
R&=\prod_{j<m}g^{(j)},\qquad
A=\sum_{j<m}R_j(1-g^{(j)})q^{(j)}(y),\\
P_m(y)&=A+Ra,\\
P_{m+1}(y)&=A+R(1-g)a+Rgb,\\
\delta&=\log\frac ba,\qquad
\xi=\frac{Rga}{A+Ra}\in[0,1],\\
\Delta_{\rm mix}
&=\log\frac{P_{m+1}(y)}{P_m(y)}
=\log\!\left(1+\xi(e^{\delta}-1)\right).
\end{aligned}
```

证明只需从后一种读出中减去前一种：变化为 Rg(b−a)，再除以原目标概率。ξ 是“当前轮对立即停止读出的目标概率贡献”再乘继续率；因此同一个单轮增益对已有混合预测的影响可以差很多。当 ξ>0 时，两种改善同号；ξ=0 时，混合改善恒为零。由单调性和 log 的凹性得到以下界：

```math
\begin{aligned}
\min(0,\delta)&\leq\Delta_{\rm mix}\leq\max(0,\delta),\\
\Delta_{\rm mix}&\geq\xi\delta,\\
\frac{\partial\Delta_{\rm mix}}{\partial\delta}
&=\frac{\xi e^{\delta}}{1-\xi+\xi e^{\delta}}.
\end{aligned}
```

同号不保证跨 token 排序相同。以下构造取两个独立样本的首个监督位置，m=1、R=1、A=0；两个位置都必定执行首轮并进入初始标签候选集，没有更早边界筛选。继续率分别为 0.01、0.5；第二轮概率由实际执行或 no-grad 探测取得。其余词的概率取补数即可得到合法分布。

| 位置 | a | b | g | ξ | 单轮改善 δ | 混合改善 Δmix |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.001 | 0.004 | 0.01 | 0.01 | 1.386294 | 0.029559 |
| B | 0.2 | 0.4 | 0.5 | 0.5 | 0.693147 | 0.405465 |

若教学覆盖率取 65%，原始正改善排序会只选 A：它覆盖原始改善的 66.67%，却只覆盖这两个位置混合改善的 6.79%。这不是对论文 ρ=0.99 的实测复现，而是说明两种“覆盖”不是同一个量。若正候选集合 C₊ 的总改善非零，且统一满足 ξ≥ξmin>0，选集 S 的原始改善覆盖 ρ，则有更紧的局部覆盖下界：

```math
\frac{\sum_{t\in S}\Delta_{{\rm mix},t}}
{\sum_{t\in C_+}\Delta_{{\rm mix},t}}
\geq\frac{\xi_{\min}\rho}{\xi_{\min}\rho+1-\rho}.
```

证明用到：选中部分的混合改善至少为 ξmin 乘其原始改善，未选部分的混合改善至多为其原始改善，再取二者的比例下界。没有正的 ξ 下界，不能沿用原覆盖百分比。这仍只比较固定上下文下的局部收益表，不包含联合续算产生的 KV 交互。

若将额外一步的成本惩罚写为 λc≥0，在上述局部比较和 ξ>0 条件下，可精确写出对应原始增益的阈值。它是诊断参照，不是源码使用的新标签规则：

```math
\Delta_{\rm mix}>\lambda c
\quad\Longleftrightarrow\quad
\delta>\log\!\left(1+\frac{e^{\lambda c}-1}{\xi}\right).
```

工程上应同时记录 `raw_ce_gain`、`mixed_nll_gain`、`target_mass_before`、`remaining_mass` 和 `continue_prob`。替换成混合增益标签未必更好：当前 gate 较小会压低 ξ，可能进一步减少弱 gate 的训练机会；原始增益则衡量下一轮预测分量的潜在改善。两者代表不同监督目标，需要配对实验判断。真实代码还 clamp 混合权重并可能以模型 dtype 累积；复核恒等式时先用 FP64 参考读出。计算 Δmix 可用 logaddexp(log(1−ξ),log ξ+δ)，并单独处理 ξ=0/1，避免先计算 exp δ 溢出；实际 dtype 与权重下限误差另记。

## DUO 探测对应哪一种干预

固定一个深度边界，把主路径实际续算的位置记为 C，把同轮停止但具有已存输入的位置记为 T。以下仅分析一次 backbone 调用：进入该轮的 updater 输入 X 和较浅 KV 快照 K 都固定。令 Fₜ(B;X,K) 表示本轮只让位置集合 B 参加、使用普通 DUO 因果可见性时，位置 t 的单轮预测分布。

posterior mask 只允许 query 读取当前轮真实继续位置的 KV 和自己的 KV，其他假想停止位置互不可见。要将停止位置 t 的 side pass 输出解释为单位置干预，还需明确计算条件：采用精确算术，归一化与 MLP 按位置独立计算，并且不存在跨样本或跨位置的容量竞争。随机性也必须一致：关闭 dropout，或按原位置和层显式耦合相同随机变量；不同 compact 形状下仅设置同一个 seed 不够。

```math
q_{t,\rm side}^{(m+1)}
=F_t(C\cup\{t\};X,K),\qquad t\in T.
```

这个结论可逐 backbone 层归纳。位置 t 之前的真实继续位置不能因 t 的加入而改变；t 只看到这些合法前驱及自己；更晚位置即使在两种运行中不同，也不能反向影响 t。side pass 的并行转发因而能同时计算多个单位置查询值。这里只等价于冻结 X、K 的本轮图；并未重新生成整段文字，也不等价于重新执行全部历史深度。若把该设计迁移到跨 token 竞争容量的 MoE，应重新检查等价条件。[posterior mask 的 `continue_kv | own_position`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L269-L308) · [side 输入与快照](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L79-L166)

若把多个停止位置组成 S 并同时续算，目标位置 t 的实际新图允许较早的 S 内位置提供 KV，故一般不能继续使用同一个 singleton 值：

```math
F_t(C\cup S;X,K)
\ne F_t(C\cup\{t\};X,K),\qquad t\in S.
```

把同一 teacher-forced 前缀上的停止混合 NLL 记为 L(S)，其中只改变本轮活动集合 C∪S，并统一在 m+1 停止；所有目标、已有质量及更浅状态固定。定义整组目标改善 G(S)，以及两个位置的交互项：

```math
\begin{aligned}
G(S)&=L(\varnothing)-L(S),\qquad G(\varnothing)=0,\\
\mathcal I_{ij}
&=G(\{i,j\})-G(\{i\})-G(\{j\}),\qquad i<j.
\end{aligned}
```

即使已把原始 CE 增益转换成该位置的混合 NLL 增益，G({t}) 仍可能不同：新增 KV 还会影响较晚的真实继续位置。进一步，交互项也未必为零。因果性限定影响方向，并不保证收益可加，更不保证交互一定正或一定负。

一个可手算的反例：位置 0 是固定前驱，位置 1、2 是停止候选，三个位置的首轮 KV value 均为 0；第二轮两个候选的 value 为 0.2、4；所有 attention logits=0，读出为 sigmoid(attention 输出)。候选的首轮目标概率都为 0.5、gate 都为 0.25，因此默认阈值 0.5 下都停止。位置 1 第二轮看到两条较浅 KV 与自己，输出 sigmoid(0.2/3)；位置 2 单独续算看到三条较浅 KV 与自己，输出 sigmoid(4/4)；联合续算则输出 sigmoid(4.2/5)。这个构造保留 DUO 的自身与前驱缓存条目，但使用简化的标量 attention，并非 Qwen 或论文评测数值。

| 续算集合 S | 最终 P₁(y₁) | 最终 P₂(y₂) | 整组混合 NLL 改善 G(S) |
| --- | --- | --- | --- |
| ∅ | 0.500000 | 0.500000 | 0 |
| {1} | 0.504165 | 0.500000 | 0.008296 |
| {2} | 0.500000 | 0.557765 | 0.109329 |
| {1,2} | 0.504165 | 0.549616 | 0.102908 |

交互项为 −0.014717 nats；把两个 singleton 收益相加会高估联合收益。由此，原论文的正增益覆盖仍是给定当前轨迹的局部监督规则。把它解释成“全序列最优计算预算分配”还需要联合干预、后续损失和实际成本的额外证据；本页前面的等成本最小位置数证明只针对固定的数值收益表。

### 算法 T10 固定前缀的续算收益诊断

**输入** 固定检查点与语义配置、有效移位目标集合 I、边界快照、下一轮输入 X、真实继续集合 C、候选停止集合 T、待检查的 singleton 与 pair 集合。快照保存：第 m 轮混合累加之前的目标质量 A、进入 m 的 R、当前单轮目标概率 a、gate g，以及更早已停止位置的最终目标概率。当前 runtime 中已吸收停止尾质量的 buffer 不能直接当成 A。KV 只保留真实执行到 m 的条目及 position／depth／valid 元数据；当前轮同一样本的原位置必须唯一。

**输出** 每候选位置的原始 CE 改善、局部混合改善、整组 singleton 改善；每 pair 的联合改善与交互项；每次参考运行的实际调用范围和数值误差。算法中的 Pᴮ 是 I 上的目标概率记录，qᴮ 是本轮 B∩I 上的单轮目标概率记录，不声称恢复完整词表分布；完整分布版本还需保存全词表累加量。

**前置条件** 所有候选在 m 已实际执行且 m<M；固定输入文字和目标，关闭模型随机性；模型参数不更新；本诊断将当前边界统一限制到 m+1。干预只改变 hard 活动集合，保持原 soft gate g，不把 g 改成 1；新末轮吸收剩余 Rg。参考分支采用精确停止混合，不加入采样、权重下限或资源强停。每次运行使用独立可追加缓存；分布式实现必须让所有 rank 依次参与相同调用。下面给出诊断算法，并未改动 TaH2 原有标签生成器。

![算法 T10 固定前缀的续算收益诊断](visualize:fde1_bGliZmlsZV9VcXIzUEZyYXB2QnNwb2JmQnBIaFJn_FileDrive_45288dfe06608191af94334ba8699511)

**实现落点** 在 `accumulate_iter` 写入第 m 轮之前记录 A、R，另外保存本轮 a、g；用 `capture_posterior_inputs` 与 `side_pass_continue_mask` 获取输入和真实 C。对一起始探测深度 m+1，源码零起始 `iter_depth=m`、`snapshot_iter(m-1)`，必须排除本次真实 C 已生成的 m+1 KV，避免重复或泄漏。用 posterior mask 计算 singleton 查询，用普通 DUO mask 重算 C∪S 的联合图，不能直接拼接探测 logits。损失只在固定 I 上归约，记录 `counterfactual_scope=frozen_prefix_one_boundary`。[runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L483-L579) · [快照与探测](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L117-L165) · [可见性实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py)

# 算法 3 TaH2 前向与辅助监督

以下使用 dense 逻辑位置描述语义，实际代码会压紧与散射活动位置。当前轮是否继续必须先由部署 gate 决定；监督标签在主路径之后生成，不能反过来替代这一决定。

**输入** token batch、有效 mask、可选监督标签、最大深度 M 与阈值 tau

**输出** 混合 logp、主缓存、实际深度、可选 NLL 和决策损失

**约定与范围** 按原位置 u=(batch,position) 索引，pack/scatter 显式转换，TopK 只取概率值。该语义算法采用 DUO、stop_prob_mix、空初始缓存及确定性阈值路由，关闭 warmup、随机路由、深度 override 和强制续算；权重未 clamp，按精确算术书写。第 9 行末层 g=0 是等价的终止记号，源码使用有限哨兵并由 finished 分支吸收尾部质量。算法中的“有监督”还要求配置后验标签生成器并启用该路径：训练或 return_eval_metadata；仅给普通 eval 传 labels 不保证运行辅助探测。

![算法 3 操作与数学公式](visualize:fde1_bGliZmlsZV80X09HTHpXa09fQTRTSExPWkNBcXBR_FileDrive_45288dfe06608191af94334ba8699511)

这是语义算法，省略了 TP、dummy collective、全局归约、数值 clamp、override 和 checkpoint 重算。原缓存张量可由 snapshot 共享，但后续不能原地改写旧条目。零起始源码深度与此处的一起始数学记号也需显式转换。

| 数学量 | 工程表示 | 关键不变量 |
| --- | --- | --- |
| 活动集合 A_m | dense bool `[B,T]` 与压紧索引 | 稀疏行始终能映射回原位置 |
| E、X、H | `[B,T,d]` 逻辑缓冲或活动布局 | 原始 E 与当前状态不混用 |
| 继续概率与剩余质量 | 浮点 `[B,T]` | 最后执行轮吸收全部剩余质量 |
| C 与可见性 | 分层分深度 KV、位置和 valid mask | 保留文字因果性及深度约束，探测不污染主状态 |
| 混合目标 logp | 全词表路径或 target-only `[B,T,1]` | 最后一维1表示目标概率，不是1个类别 |
| 标签与 gate logits | detached 标签与保留计算图的 logits | 真标签不进入部署路由，BCE 范围为已执行监督位置 |

# 交付与验收

## 实现验收清单

- [ ] `loop_trace.json`：每轮活动位置、原始 embedding、updater 输入输出、继续概率与实际深度。

- [ ] `visibility_table.json`：主路径和 lookahead 分开记录 query 到 key 的允许关系。

- [ ] `cache_side_effect_report.json`：探测与 backward 重算后，主路径旧 KV 的数量、内容和别名约定保持正确。

- [ ] `mixture_and_grad_report.json`：手算混合与 runtime 对齐，token NLL 和决策 BCE 的梯度路径分别说明。

## 循环记忆的研究扩展

历史 KV replay、跨轮记忆和其他循环位置可在此状态接口上扩展；每次改变可见性，都要重新给出参考计算与缓存生命周期。更深不自动代表更有效的推理。

# 动手检查

画一个位置执行两轮的计算图，用不同箭头分别标记梯度依赖与缓存读写。然后回答：如果只把主路径 updater 的输出 `detach`，保持其他代码不变，第二轮答案损失还能通过这条回填边训练什么？

## 参考思路

第二轮仍可训练自身计算中参与的可训练参数，但这条回填边不再把梯度传回 updater 及生成其输入的前轮计算。其他损失、共享参数和缓存依赖仍可能产生梯度，因此不能笼统说第一轮没有梯度。

对我们的研究而言，画清这张图以后，才有依据决定教师信号放在哪里，以及哪些接口能被最终任务训练到。

# 围绕本课讨论

```codex-prompt
请从预测位置与移位目标的定义开始，用数学状态、DUO可见性、停止决策和概率混合刻画TaH2，再按前向算法追踪每轮的输入、KV增量、活动集合和剩余质量。证明混合权重和为1；单独解释lookahead的专用mask、标签生成与BCE梯度。最后逐项核对固定源码的张量布局、索引、detach、缓存别名和多卡collective，明确参考算法省略的优化。
```

# 继续学习

[上一课：一次参数更新](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647) · [学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133) · [下一课：从训练到自主生成](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)
