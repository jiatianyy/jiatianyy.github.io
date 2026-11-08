---
title: "一次内部循环"
description: "一次内部循环把当前隐状态变成下一轮输入，同时更新活动位置、KV 缓存和读出。本课先定义这些状态与继续规则，再推导 updater、decider 和停止混合，最后沿主前向与辅助探测算法说明梯度和缓存如何流动。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811"
sourcePageId: "page_31a843d260588191a252ae63fa65f811"
sourceSha256: 0ec32ab505c72766e7393c73e0715a1726e00e537e4c2d95030e395e71c811f9
series: loop-reasoning
references: []
---

<span id="source-block-204fb5d1-4057-48aa-92ac-0ce07ca4ffa7"></span>

一次内部循环把当前隐状态变成下一轮输入，同时更新活动位置、KV 缓存和读出。本课先定义这些状态与继续规则，再推导 updater、decider 和停止混合，最后沿主前向与辅助探测算法说明梯度和缓存如何流动。

<span id="source-block-29f806b1-a645-4695-b9e2-9184adb169da"></span>

[返回学习地图](/notes/loop-reasoning/engineering/)

<span id="source-block-d14db057-9411-49c2-bd9a-dd9038238a9a"></span>

> [!CALLOUT]
>
> **阅读主线**：状态与继续规则 → updater 和 decider → KV 可见性 → 停止混合 → 深度标签与联合损失。用数学定义理解依赖，用逐行算法跟踪控制流，再到源码检查 shape、梯度和缓存副作用；混合权重的归一化证明位于读出部分。

<span id="source-block-7ea8da3a-e953-456a-ae80-5d840be570fb"></span>

## 状态与更新的数学定义

<span id="source-block-ea97b8a8-b76b-4145-b9f8-d9abbf53ad74"></span>

### 状态与活动集合

<span id="source-block-3aacb9b6-815c-45c5-8b06-81ed704ac8d3"></span>

以预测位置 t 为索引，并在有效的移位监督位置集合 I 上定义下一 token 目标；文本末位没有目标，按 ignore 处理。这与模型内部移位后的 labels 对齐。下文省略 batch 索引，KV 始终限于同一条样本。深度从1开始，参考分支采用确定性 gate，不含深度 override。

<span id="source-block-6ce1ab31-a1f7-42c7-a53e-73acbef07a1c"></span>

```math
\begin{aligned}
E&\in\mathbb R^{B\times T\times d},\\
A_m&\subseteq\{(b,t):\text{位置有效}\},\\
S_m&=(E,H_m,p,m,A_m,\mathcal C_m,\nu_m),\\
y_t^*&=\texttt{input\_ids}_{t+1}.
\end{aligned}
```

<span id="source-block-ce2a7b8b-219d-4ad0-87f9-05d0fceeb49c"></span>

E 保存原始 token embedding；H_m 仅在本轮实际执行的位置有定义；p 是逻辑位置；C 是按 layer、深度与位置组织的 KV；nu 是有效性信息。工程可以压紧活动位置，也可以用 dense 缓冲加 mask 表示这些部分定义的量。

<span id="source-block-334c8cb6-9d38-4252-b158-9bc897c2bcb0"></span>

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

<span id="source-block-547ac6f9-a8f6-4dd3-97f0-59b68e7222fc"></span>

Backbone 不只是对一个向量应用函数：它同时接受活动 token、缓存与可见性规则，并返回状态和 KV 增量。updater 的输入是原始 embedding 与当前末层状态；跨轮梯度是否保留属于计算图定义。

<span id="source-block-925f7b17-71f3-49f2-afc0-ca7d1d02b823"></span>

### 继续概率与实际路由

<span id="source-block-2df70ac7-9376-4274-87d8-068ade9a3ce7"></span>

```math
\begin{aligned}
g_{m,t}&=\sigma\!\left(
D_\psi(e_t,h_{m,t},\operatorname{TopK}(q_{m,t}))\right),\\
A_{m+1}&=\{t\in A_m:g_{m,t}>\tau\}\quad(m<M),\\
d_t&=\min\{m:m=M\ \text{或}\ g_{m,t}\le\tau\}.
\end{aligned}
```

<span id="source-block-da286087-42d3-4465-8454-b8ab2c7d20dc"></span>

部署决策由当前状态、预测分布和阈值决定。所选 Qwen3MLPIterDecider 不读取真目标标签；标签用于训练期的后验监督。其他 oracle decider 或随机路由是不同算法，需要另行定义。

<span id="source-block-530c07f3-f43b-4791-b558-7a9fdbb26596"></span>

### 实际路由的分支顺序

<span id="source-block-fdc72673-9053-4688-9a3a-b36d77625abf"></span>

实际执行先保存 gate logits，再依次应用深度 override、causal 监督位置规则和可选的图对齐规则，由此确定 finished mask。连续混合仍读取原 gate logits：强制某个执行深度，并不会自动把混合中的 gate 概率改成0或1。上面的阈值公式描述其中常用的确定性分支。[调用顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L694-L775)

<span id="source-block-1b5514f7-bfc8-43d6-876f-dc42fe001f83"></span>

| 分支 | 生效条件与行为 |
| --- | --- |
| 深度上限 | decider 达到 max_iter 后返回停止；名为 MINUS_INFINITY_LOGITS 的哨兵实际为 −10，尾部质量由 runtime 的 finished 分支吸收 |
| 训练 warmup | 仅训练且仍在 warmup 时，阈值从 0 线性升至配置 threshold |
| 可选 Bernoulli | stochastic_sampling=True 且没有进入 warmup 分支时采样；该 elif 没检查 self.training，单独调用 eval() 不会关闭它 |
| 显式深度重放 | 正的 iter_count 用 count>当前深度覆盖布尔决策；count=0 保留原决策；不替换已保存的 gate logits |
| causal 监督规则 | 有移位 labels 时，无监督预测位置被要求首轮后停止；纯推理无 labels 时不进入此分支 |
| 可选 FSDP 图对齐 | 训练、开启 layer_wrap、无显式 iter_count、尚未到上限且全体停止时，选 gate logit 最大的一个有效活动位置继续 |

<span id="source-block-2d0342b7-554b-47bb-bb3f-b920d178b5e3"></span>

默认循环训练的 layer_wrap=False；单轮模型默认 True，可由配置显式覆盖。最后一行是可选工程分支，不能当成默认 TaH2 的路由规则；它在 causal 规则之后执行，trace 应记录最终实际路由。[decider](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L196-L267) · [override](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L458-L480) · [训练入口默认值](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py#L125-L135)

<span id="source-block-23b3e3bc-875c-4d54-ab69-43f81e74624a"></span>

updater 的 prev_inputs 在主调用中接收 active_first_iter_embeds，含义是原始嵌入。阅读递推式时应沿这条调用关系理解输入，不能按参数名称将它误读成上一轮输入。[updater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py) · [调用现场](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)

<span id="source-block-9937cde9-a608-427d-b65c-5372c04a4984"></span>

### Updater 与 decider 的精确映射

<span id="source-block-bdb486e4-9b3f-4edf-8879-dfa8adbca601"></span>

以下把固定源码的 Qwen3 MLP 组件写成矩阵运算。令隐藏宽度为 d、中间宽度为 i；RMSNorm 的可学习尺度为向量 s。默认激活是 SiLU，线性层无 bias。定义：

<span id="source-block-c56cf15e-4cc4-4e85-9cdb-cc8a69207931"></span>

```math
\begin{aligned}
N(x;s)&=\operatorname{diag}(s)x/r(x),&
r(x)&=\sqrt{\|x\|_2^2/d+\epsilon},\\
F(x)&=W_{\rm down}
\bigl[\operatorname{SiLU}(W_{\rm gate}x)
\odot(W_{\rm up}x)\bigr].
\end{aligned}
```

<span id="source-block-daf19f9f-4860-4e10-ab4a-892ecf4b0a4c"></span>

Updater 对原始嵌入 e 与末层状态 h 分别归一化，拼接后投影，再执行 MLP。输出直接成为下一轮输入，源码没有额外加上 e 或 h 的 residual carrier：

<span id="source-block-6936e390-f0dc-4d2e-ac8e-b6c2b2546ff7"></span>

```math
\begin{aligned}
v&=W_eN_e(e)+W_hN_h(h),\\
x&=N_{\rm pre}(v),\\
U_\phi(e,h)&=N_{\rm out}(F(x)).
\end{aligned}
```

<span id="source-block-550fc2f6-d767-4f64-8094-bcd40529bee6"></span>

关闭某个可选 norm 时，把对应 N 换成恒等映射。构造函数中 `enable_norm_pre_mlp` 与 `enable_norm_out` 均默认为 True；该文件较早的说明文字把后者写成默认关闭，复现应以构造参数与 checkpoint 的 `init_args` 为准。[Qwen3MLPUpdater 初始化与 forward](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py#L66-L105)

<span id="source-block-210d9ac0-10be-4ccb-af67-d720bf480891"></span>

这些归一化调整了状态尺度，但不能单独证明循环收敛。固定可微分支时，RMSNorm 与 updater 的局部 Jacobian 为：

<span id="source-block-f4b67697-b769-478c-8a51-2f4365583575"></span>

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

<span id="source-block-4ec0ac48-0396-4364-94a2-6f7ac5fea82e"></span>

各 Jacobian 在上一式对应的中间量处求值。只有进一步约束这些矩阵的谱范数及 backbone、KV 依赖，才可能得到完整循环的收缩条件。原始嵌入在每轮重复进入 updater，因此它也形成多条参数梯度路径；不能把 updater 近似成只依赖前一隐藏状态的一元变换。

<span id="source-block-90b88e16-95e9-46e6-b1a2-9644705bf810"></span>

Decider 的第三路输入是词表 softmax 后最大的 K 个概率值，不含这些值的 token ID。固定实现将该向量直接送入宽度 d 的 RMSNorm，再与两个 d 维状态拼接；因此必须满足 K=d，词表不足 K 时先补零。其 MLP 前没有 updater 的那层可选 pre norm：

<span id="source-block-3eb0d2ec-91a4-4eba-bdce-f89b3785f889"></span>

```math
\begin{aligned}
q&=\operatorname{softmax}(z),\qquad
u=\operatorname{pad}_{K}(\operatorname{TopKValues}(q)),\\
v_D&=W_D[N^D_e(e);N^D_h(h);N^D_q(u)],\\
s_D&=w_D^\top N^D_{\rm out}(F_D(v_D)),\qquad g=\sigma(s_D).
\end{aligned}
```

<span id="source-block-bf834fae-f444-47e4-88b8-fe46d6fb004e"></span>

该概率特征没有 detach。若局部 top K 索引集合 S 不变，选择矩阵 P_S 给出下面的导数；未被选中的词表 logit 仍可通过 softmax 分母收到梯度。并列排序边界需按所执行的选择分支理解。[decider 张量路径与路由](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L152-L267)

<span id="source-block-11112dbf-02e1-4f19-8a38-485552e8cea7"></span>

```math
\frac{\partial q_S}{\partial z}
=P_S\bigl(\operatorname{diag}(q)-qq^\top\bigr),
\qquad
\frac{\partial q_i}{\partial z_j}=-q_iq_j
\quad(i\in S,\ j\notin S).

```

<span id="source-block-322679b7-b79b-4b60-b4ee-a60ce916aaa5"></span>

两层可选 norm 都启用时，updater 参数量是 `2d²+3di_U+4d`，decider 是 `3d²+3di_D+5d`。当两者中间宽度均等于 d，合计为 `11d²+9d`；4B 配方的 d=2560 对应 **72,112,640 个侧组件参数**，不计骨干。这是按矩阵形状推导的计数。[组件定义](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py#L66-L105) · [decider 定义](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py#L131-L180) · [4B 配方](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml#L18-L27)

<span id="source-block-17f0dd41-4723-4243-b7a7-787da86097dc"></span>

### 可选初始化：uniform_init 没有生成均匀 gate 概率

<span id="source-block-8bdda57e-e312-4149-874c-4563509c9940"></span>

可选 uniform_init 以正态权重控制 score 的初始化尺度。Qwen3MLPIterDecider.__init__ 在该标志为 True 时调用 normal_(std=π/√(3d))，为 False 时使用 std=0.01；它没有直接采样 Logistic logit。1.7B／4B／8B 的默认 sft_tah.yaml 均未设置该标志，类默认 False。以下条件分布说明这一可选分支，不能据此解释论文默认训练的效果。[原构造函数](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) · [配方到组件参数](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_model.py)

<span id="source-block-8e67bf13-d32a-4e8e-9bbe-a63118141793"></span>

先在理想的实数正态采样下分析这一初始化。固定输入与其他参数，记 a 为 score 前的 MLP 输出，q=‖a‖²/d>0，并令归一化的 gain 初始为 1。归一化后的特征为 ξ=a/√(q+ε)。若最终 score 权重 w 与 ξ 独立，且 w∼N(0,σ_w²I)，就有下面的条件分布：

<span id="source-block-7322d207-12eb-4d95-a63a-fbec2916d0f0"></span>

```math
\begin{aligned}
s_D\mid\xi&\sim\mathcal N(0,v),\qquad v=\sigma_w^2\|\xi\|^2,\\
\sigma_w^2=\frac{\pi^2}{3d}&\quad\Longrightarrow\quad
v=\frac{\pi^2}{3}\frac{q}{q+\epsilon}.
\end{aligned}
```

<span id="source-block-3aabe9f2-54bc-4b93-a99d-31d65c327e35"></span>

令Φ是标准正态CDF。对0<t<1，继续分数g=σ(s_D)的条件分布为logistic-normal，而非Uniform(0,1)：

<span id="source-block-b5f7b70a-e58f-42de-bc4b-2155dc8aef92"></span>

```math
\Pr(g\le t\mid\xi)=\Phi\!\left(\frac{\log(t/(1-t))}{\sqrt v}\right),\qquad
\mathbb E[g\mid\xi]=\tfrac12,\qquad \Pr(g>\tfrac12\mid\xi)=\tfrac12.
```

<span id="source-block-e0d4bf11-0815-4c03-b09e-dc878e9c05d7"></span>

当q远大于ε时，在t=0.1和0.25处CDF约为0.11287和0.27236；若真为均匀分布，则应分别为0.1和0.25。让logit具有Logistic(0,1)分布才会经sigmoid得到精确均匀分布；只匹配其方差不够。q=0时g退化为0.5，上述除以√v的CDF式不适用。

<span id="source-block-e589e2ae-477d-492e-a9e1-66b882a574fb"></span>

上式的 Pr(g>1/2|ξ)=1/2 是对随机初始化求得的概率。同一个模型在不同 token 和轮次中共享 score 权重，所以这个结论不代表初始化完成后恰有一半 token 继续，也不保证各次继续决策是独立的 Bernoulli 变量。对两份都与 w 独立的固定特征，其 logit 协方差为：

<span id="source-block-2aca8773-33d9-4e2d-8f02-7268f2cd8850"></span>

```math
\operatorname{Cov}_{\rm init}(s_i,s_j\mid\xi_i,\xi_j)=\sigma_w^2\xi_i^\top\xi_j.
```

<span id="source-block-6546bbe6-2ee8-4515-8099-25efa4280e77"></span>

例如ξ_i=ξ_j时两项logit完全相同。若后续特征已经受前一轮gate选择影响，就不能继续假定ξ与w独立。以上概率是对随机score初始化的分布，不是单次checkpoint上的token频率、训练后校准结果或实际停止深度分布。

<span id="source-block-9dfc2b3f-d07b-4837-bde3-83b40092e08e"></span>

实际构造默认建立 BF16 score，再调用 to(dtype)；训练入口随后将整体参数转为 FP32 主权重，已发生的舍入不会被恢复。[入口与 dtype 顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py)。因此，上式是理想连续正态采样的参考律，真实张量分布还需按设备和 dtype 测量。原构造函数的替身检查只验证分支、std 与 cast 调用，没有验证 BF16 抽样精确满足该分布。

<span id="source-block-91fa752d-7d62-49c5-b40e-5b598631d157"></span>

工程核对顺序是：保存flag与seed→记录score生成dtype和后续cast→对固定首轮输入分别统计logit/gate分布与共享相关性→将参数初始化重复与单模型token统计分开→单独检查阈值、warmup和实际执行深度；不因flag名而假设初始化已校准。

<span id="source-block-b8236598-e51d-4192-b2c8-d08b8809750d"></span>

## 梯度与缓存分别看

<span id="source-block-da9c5c06-3ad6-40cc-8e29-f101bcc5ef55"></span>

主路径调用 updater 时没有自动 `detach`，因此已执行轮次之间可以传递梯度；布尔路由选择本身不是可微的连续运算。决策器怎样学习，还要追踪监督损失和连续混合权重。

<span id="source-block-a9c7a564-ebf5-4f75-8c86-4829556d75b9"></span>

辅助探测同时使用 `torch.no_grad()` 与主缓存的 snapshot。前者控制计算图，后者处理缓存读写边界。关闭梯度不会自动阻止函数修改缓存。[辅助探测](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)

<span id="source-block-28083252-c393-4b2b-b48f-3444edac04d8"></span>

`finalize_logits` 返回按深度累计的混合 log 值；在理想未 clamp、无舍入时，它是归一化的 log-probability。源码对权重设下限，target-only 与全词表路径还采用不同的缓冲精度，严格归一化应另行核对。训练采用目标 token 累计路径时不一定构造完整词表混合结果。[概率质量与损失分支](/notes/loop-reasoning/parameter-update/)

<span id="source-block-02ad95bf-0c5d-40bf-8ebf-781d9ca0f5dc"></span>

归一化可以调整数值尺度，但尺度合适不等于下一轮已经会使用信息。接口是否形成可继续使用的中间状态，仍然需要后续训练与干预来判断。

<span id="source-block-67ec17f6-0693-482c-85cb-eeb3ee4763b7"></span>

## 对照源码

<span id="source-block-c6a42693-4540-4956-a650-623dc3fb064a"></span>

固定版本：TaH `78ddf9a2c25713fced03a736c5746161f6ee590c`。

<span id="source-block-8d8961e2-a0cb-4c7b-8b19-b092876bc28c"></span>

- [input_updater.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py)：先读 `Qwen3MLPUpdater.forward`，画出张量变换。

<span id="source-block-6fbd6eee-52be-496e-96a6-47186f7ea2eb"></span>

- [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)：再找 `TaHForCausalLM.forward` 中调用 updater 和更新 `next_iter_mask` 的位置。

<span id="source-block-b04fcc16-0ad4-4bdd-a0ad-77bb2ae0c38a"></span>

- [tah_runtime.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)：对照 `capture_posterior_inputs`、`_run_side_score_pass`、`accumulate_iter` 和 `finalize_logits`。

<span id="source-block-ac140e19-9bfc-43bf-86da-24ac3963d755"></span>

- [loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py)：分清预测损失和决策损失怎样进入同一参数更新。

<span id="source-block-5ef0d88e-4b3f-4975-a1db-4e08ed85080a"></span>

## 位置与深度的注意力可见性

<span id="source-block-d2eea9b9-d301-4a24-9e8e-8d2fabdb559c"></span>

### 主路径可见性

<span id="source-block-87b10cf0-eed5-4ef1-8934-7ea05fca7e06"></span>

DUO 主路径的可见性集合如下；只纳入已经实际执行、有效且同样本的 KV。position 与 depth 是两个独立条件。

<span id="source-block-94205c44-0182-4532-9b0c-3fb4e9593c9c"></span>

```math
\mathcal V(t,m)=
\{(s,j):s\le t,\ j\le m,\ (s,j)\text{已执行且有效}\}.
```

<span id="source-block-af9fe7fd-7644-4233-adcb-7fe1913ecbc0"></span>

### 辅助探测可见性

<span id="source-block-b2f6421c-1d08-4cc2-8c66-12542d3cfb71"></span>

lookahead 的假想深度使用专门规则：它可以读真实的先前深度；在当前假想深度，只能读真正继续的位置或查询自身，且仍受文字因果条件约束。它不能读取其他停止位置的假想状态。

<span id="source-block-ded1e55f-a034-4888-ad2b-5b0334c53838"></span>

```math
\begin{aligned}
\mathcal V_{\rm probe}(t,m)
=\{(s,j):s\le t,\ \text{有效},\
[j<m\ \lor\
(j=m\land(c_s=1\lor s=t))]\}.
\end{aligned}
```

<span id="source-block-a556fa3b-7b03-4b33-833e-2532df71d235"></span>

键域同时要求：j<m 的键来自 snapshot 中真实主路径已执行的条目，j=m 的键来自当前 side pass。这里 c_s 表示该位置是否在主路径真正继续到当前深度，并非后验监督标签。源码采用零起始缓存深度；核对实现时先统一到这里的一起始记号。[posterior mask](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L216) 明确实现了这一例外。 数学架构对照[TaH2 原论文第4节](https://arxiv.org/html/2609.35748v1)。

<span id="source-block-5cf41de8-1892-4041-820a-cf97c5d87367"></span>

### 辅助探测怎样保持上下文

<span id="source-block-33d3d8d1-f066-4329-bfbb-72e1dbb4bb3a"></span>

DUO 的 side pass 同时重算已存输入中的停止位置和真实继续位置；继续位置提供当前深度的合法 KV 上下文。只重算停止位置会缺失这部分上下文，而让所有假想停止位置互相可见又会增加不存在的依赖。评分写回覆盖全部 forward 位置，因此继续位置的 CE 也会被 side pass 结果替换。[完整前向集合与评分写回](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L54-L211)

<span id="source-block-462630cb-7c58-4bcc-bceb-c3d0ac21b4cb"></span>

这里要依次回答“是否调用探测”和“调用哪些位置”。先在 DP 组检查是否存在缺失 CE；没有缺失时，跳过这一轮 side backbone。若进入 DUO side pass，前向集合 F_m 包含全部已存 side inputs，范围不限于缺失评分的位置。当 m<M 时，side pass 末尾还会为所有有效前向位置计算 updater：下一深度缓冲按位置覆盖输入值，mask 则与已存 mask 取并集：

<span id="source-block-03e90650-4d34-4def-a482-656c55d39ea9"></span>

```math
\mathcal F_{m+1}
\leftarrow \mathcal F_{m+1}\cup\mathcal F_m
\qquad\text{当深度 }m\text{ 的 side pass 执行且 }m<M.

```

<span id="source-block-e6d10ac8-c412-4e6c-9dcb-7c591a21e763"></span>

因此深层探测集合可能大于上一轮真实执行集合。例如 M=3，a 在主路径停于第一轮、b 停于第二轮；为补 a 的第二轮 CE，DUO side pass 转发 {a,b}，并把两者第三轮输入都存下。若 b 仍是第三轮的缺失评分候选，下一次 side pass 的实际转发集合仍可为 {a,b}，尽管主路径第二轮只执行了 b。a 的假想当前轮 KV 不对 b 开放，额外计算与真实可见性是两个不同问题。

<span id="source-block-33811572-6978-4014-ae34-167fd9708d6e"></span>

这也要求成本按实际 F_m 和调用指示计数。主前向的 capture_posterior_inputs 为下一轮准备输入，与 side pass 末尾的下一层 updater 是两处独立调用；最后探测深度 m=M 不再准备下一层。论文 Eq.23 的 N_m 账本与固定实现的具体执行范围分别说明，不能直接把前者当作后者的 kernel 次数。[调用触发](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L214-L314) · [side updater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L191-L211) · [输入合并与主路径准备](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L465-L517) · [论文成本与代码边界](/notes/loop-reasoning/tah2/)

<span id="source-block-84ecba13-8669-48c5-8782-d1e559031bfb"></span>

以下过程对应一次指定深度的辅助评分。深度 m 使用一起始数学记号，函数内部转为零起始 `iter_depth=m-1`。探测缓存包含主路径深度小于 m 的条目；本轮新产生的条目只写入临时容器。

<span id="source-block-2378b5ea-cbc4-4fa1-bda4-fa6ec0b3a248"></span>

#### 算法 T1 DUO 辅助评分与缓存隔离

<span id="source-block-7efddd9c-49c4-4932-a9a1-a1ab0519a44e"></span>

**输入** 主路径缓存、深度 m、已存 side inputs、实际继续掩码、缺失分数的目标位置、移位标签与位置。

<span id="source-block-84f53b7c-6cc8-4745-9999-6ba49d8335ef"></span>

**输出** 深度 m 的 detached CE 表，以及需要时供下一探测深度使用的 detached updater 输入；主路径缓存内容保持原值。

<span id="source-block-b5c212d6-257f-42ac-b094-71e94607ce21"></span>

**前置条件** m≥2；调用方已确认 DP 全局存在缺失 CE；保存的输入始终锚定同一原始 embedding；同一 DP 组按既定边界顺序进入探测与对齐路径。实际转发集合取已存 side inputs，包含按 OR 合并保留的位置。

<span id="source-block-8fad9d34-222b-4af6-9c50-93a28a6d62ca"></span>

<figure class="research-embed"><iframe src="/research-assets/05531940ed67bb7217369efe1339f0def93d0977623cf2501f7a8e915b77bfbf.html" title="算法 T1 DUO 辅助评分与缓存隔离" loading="lazy" sandbox=""></iframe><figcaption>算法 T1 DUO 辅助评分与缓存隔离 · <a href="/research-assets/05531940ed67bb7217369efe1339f0def93d0977623cf2501f7a8e915b77bfbf.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-cba6d2f4-9ca5-4cdd-b7f6-c77acb920c88"></span>

**源码** [`_run_side_score_pass`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L54-L211) · [`snapshot_iter`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L402-L422) · [posterior attention mask](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L216-L308)

<span id="source-block-298eed15-b7f6-4def-b3c0-5d4a779d3269"></span>

缓存隔离与 backward 重计算是两个不同机制。side pass 在 no_grad 中使用 detach 后的 snapshot；gradient checkpoint 的重计算则需要保留梯度，并临时启用只读缓存。该上下文还恢复前向时捕获的深度，避免 backward 恰好使用主前向结束时的深度。

<span id="source-block-e690ea9d-c1e3-4306-9400-90ffdbc21bad"></span>

| 路径 | 梯度规则 | 缓存行为 |
| --- | --- | --- |
| 主前向 | 保留已执行计算图 | 追加或替换缓存容器中的张量引用 |
| 后验 side pass | no_grad，旧 KV detach | snapshot 共享旧存储，新 KV 写入临时容器 |
| checkpoint 重计算 | 重新建立所需计算图 | 返回较浅缓存加当前重算 KV，不向主缓存再次追加 |

<span id="source-block-3b96c75b-a9e2-4041-a5a9-4186eb22c48a"></span>

主缓存正常追加使用 torch.cat 后替换引用，因此 snapshot 共享旧张量的约定能够成立；若改成原地覆盖旧存储，就必须同时改变 snapshot 的所有权设计。当前只读重算分支读取的是严格较浅深度的缓存，不应未经分析就推广到携带同深度历史前缀的分块训练。[缓存 update 与只读上下文](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L11-L137) · [checkpoint 调用](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L45-L66)

<span id="source-block-1cb72cf2-c255-49ea-9286-3d556307f4ed"></span>

### 两位置示例与缓存边界

<span id="source-block-9ede51d2-456b-4ccf-9f0d-62815a11e04d"></span>

两位置、两深度均已执行时，下面是主路径的教学可见性表。实际 padding、停止位置与探测状态还需增加相应限制。

<span id="source-block-5e143a03-4496-4e2c-b18f-bd4b33d22427"></span>

| 查询与键 | t1 深度1 | t2 深度1 | t1 深度2 | t2 深度2 |
| --- | --- | --- | --- | --- |
| t1 深度1 | 可见 | 不可见 | 不可见 | 不可见 |
| t2 深度1 | 可见 | 可见 | 不可见 | 不可见 |
| t1 深度2 | 可见 | 不可见 | 可见 | 不可见 |
| t2 深度2 | 可见 | 可见 | 可见 | 可见 |

<span id="source-block-22a300ee-4519-45df-b131-d62b56e9a2e5"></span>

沿 [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) 的 `_init_position_ids`、`_build_attention_mask_for_iter` 与 [causal_cache.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py) 的 `update`、`get_cache_upto_iter` 核对实际存储。每个条目至少追踪逻辑位置、深度、layer、有效 mask、KV 是否已应用位置编码；不能把循环一次自动等同于文字位置前进一次。

<span id="source-block-23cfc8a0-11b9-4043-9abb-b284c1b7bfc4"></span>

lookahead 是额外监督路径：停止位置的探测不能彼此建立原本不允许的依赖。`snapshot_iter` 创建新的缓存容器，但共享并 detach 已有 KV 张量；它依赖后续追加、不原地修改旧条目的约定。未来修改缓存写法时，要同时检查别名和副作用。[缓存 snapshot](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/causal_cache.py#L402)

<span id="source-block-ef4edd38-43c9-40be-9f26-46a4139078c6"></span>

## 停止决策与概率读出

<span id="source-block-8f25fb02-f0b1-4d9e-ba01-2ba0df5583bd"></span>

### 停止权重与混合分布

<span id="source-block-8a8d86e6-28a2-4766-9be9-4872447f9337"></span>

一旦某个位置决定继续，最终预测怎样使用已经算过的各轮分布？TaH2 保留这些分布，并让它们共同组成读出。这里需要分清两个动作：[Qwen3MLPIterDecider.forward](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) 用继续分数和阈值决定下一轮是否实际执行；[runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py) 再为已经执行的轮次分配混合权重。

<span id="source-block-0a22bab5-34f7-4415-a2de-8d763fb2022d"></span>

下式的 u_m,t 表示尚未分配给较浅轮次的混合质量，首轮从 1 开始。若继续分数为 g_m,t，本轮保留其中的 1−g_m,t，剩余部分传给下一轮；到了实际停止深度 d_t，就把全部剩余质量交给当前轮。u 是这个混合过程的累加状态，实际执行哪些轮次仍由路由规则决定。

<span id="source-block-0abe48d5-1f70-4b8c-97be-8fbd2b56b890"></span>

```math
\begin{aligned}
u_{1,t}&=1,\qquad u_{m+1,t}=u_{m,t}g_{m,t},\\
\omega_{m,t}&=u_{m,t}(1-g_{m,t})\quad(m<d_t),\\
\omega_{d_t,t}&=u_{d_t,t},\\
q_t&=\sum_{m=1}^{d_t}\omega_{m,t}q_{m,t}.
\end{aligned}
```

<span id="source-block-ccfed9ab-fdeb-4658-9193-03e5c336bfd9"></span>

### 权重归一化的证明

<span id="source-block-db51700d-098c-4b52-b0d3-e1b2eba9ef7c"></span>

归一化可以直接沿这个分配过程看出来：每个非末轮拿走的质量，正好是 u_m,t−u_m+1,t；实际末轮再拿走 u_d,t。把这些项相加，中间的剩余量逐项抵消，只留下首轮的 u_1,t=1。每项非负，而各轮 q_m,t 本身是归一化分布，所以最终概率混合也归一化。

<span id="source-block-afa6d464-6788-4e48-9362-8f7825422e24"></span>

关键是“实际末轮”吸收尾部质量。如果某个位置在最大深度之前停止，未执行的轮次没有预测可供混合，剩余质量就在当前轮结清；证明并不要求它算满最大深度。这里使用未 clamp、无舍入的理想定义，源码权重下限和有限精度的影响见“一次参数更新”的归一化分支分析。

<span id="source-block-62c19e45-97dd-48ac-9f10-713fe26517b4"></span>

```math
\begin{aligned}
\sum_{m=1}^{d_t}\omega_{m,t}
&=\sum_{m<d_t}(u_{m,t}-u_{m+1,t})+u_{d_t,t}=1,\\
\log q_t(y_t^*)
&=\operatorname{logsumexp}_{m\le d_t}
\bigl(\log\omega_{m,t}+\log q_{m,t}(y_t^*)\bigr).
\end{aligned}
```

<span id="source-block-45176e4d-5464-498f-b295-c5005024403c"></span>

训练一个已知目标词 y 时，我们只需混合分布在 y 上的概率。上式第二行因此可以直接在各轮的目标 logp 上做 logsumexp；target-only 路径只保存这个累计结果，节省完整词表的混合缓冲。每一轮得到正确的 q_m,t(y) 仍依赖词表归一化，decider 也仍需要门控输入，词表读出没有因此消失。

<span id="source-block-79a4951f-9d16-442c-9ebb-28130f6f32c5"></span>

这也说明 gate 可以从哪里学习：硬阈值决定是否执行，其布尔选择本身不可微；已执行路径上的连续混合权重却仍在计算图中，辅助监督也沿另一条损失路径训练决策器。跟踪梯度时，要分别沿这两条连续路径回看，不能仅凭硬路由不可微就判断 gate 没有训练信号。

<span id="source-block-ef416d93-8ff6-4568-a5a2-d64aa264b9aa"></span>

用一个两轮教学示意看完整过程：最大深度为 2，确定性阈值为 0.5，温度为 1，采用归一化停止混合且没有资源覆盖。先固定此前 token 已经形成的状态，再沿当前预测位置追踪；当前位置仍按 DUO 规则产生和读取各轮 KV。

<span id="source-block-6ff2aea1-528d-4ce0-96c0-c0fb7da26614"></span>

首轮从原始 embedding 得到 h₁ 和词表分布 q₁。若 decider 给出 g₁=0.7，这个位置便继续；updater 把原始 embedding 与 h₁ 合成第二轮输入，骨干据此得到 q₂。此时达到最大深度，最终读出为 0.3q₁+0.7q₂。同一个 0.7 参与两步：先决定第二轮是否发生，再决定已执行的两轮怎样共同贡献预测。

<span id="source-block-3baaf387-c2e1-4460-8778-18a9ce40b813"></span>

若 g₁=0.4，这个位置在第一轮停止，第二轮不执行，读出直接是 q₁。要先确定实际得到哪些分布，才能讨论它们怎样混合。这里混合的是概率；先线性组合 hidden 再做 softmax，通常会得到不同结果。

<span id="source-block-944101dc-f669-41da-a9b9-e3b3792720ee"></span>

### 随机路由与停止混合叠加后的真实边缘分布

<span id="source-block-c7461187-6817-42c7-9f64-b089c4c4eded"></span>

固定 MiniSGL 的 sample 分支先按 gate 抽样“是否继续”，到实际停止轮后又按 stop_prob_mix 读出。两次使用 gate 对应两个不同随机／加权对象。下面给出这一可选分支的单 token 条件结论；论文主配方的确定阈值不能直接替换为此随机模型。[Bernoulli 决策与混合读出](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

<span id="source-block-640f02c4-467a-4d39-a49a-92e14c91f633"></span>

**条件。** 先固定当前 token 的入态，包括此前路由产生的全部 KV。给定这个入态，设始终继续至 M 的前缀链唯一，链上的每轮分布和 gate 均确定。若骨干还含随机性，先固定与路由随机抽样独立的那部分随机性，再分析这条条件链。以下令每轮条件继续率确为 g，温度为 1，最终从归一化概率混合中抽取文字；不加入 top-k／top-p、资源强停、warmup、override、权重下限或数值舍入。M 为正整数，末轮强制停止。

<span id="source-block-250de612-b377-44c4-b248-2d8a0642c689"></span>

```math
\begin{aligned}
R_1&=1,\qquad R_{j+1}=R_jg_j\quad(j<M),\\
\Pr(D=d)&=\begin{cases}R_d(1-g_d),&d<M,\\R_M,&d=M,\end{cases}\\
P_d(v)&=\sum_{j<d}R_j(1-g_j)q_j(v)+R_dq_d(v).
\end{aligned}
```

<span id="source-block-3636849b-9309-4490-b4a8-6a41d3a2f4af"></span>

D 是真正执行的轮数，P_d 是在 d 停止时的文字分布。对同一固定入态边缘化 D，可得到一个封闭形式：

<span id="source-block-74636ad1-897d-4507-87d2-c4228901e4b4"></span>

```math
\begin{aligned}
\bar P(v)&=\sum_{d=1}^{M}\Pr(D=d)P_d(v)\\
&=\sum_{j<M}R_j^2(1-g_j^2)q_j(v)+R_M^2q_M(v),\\
R_j^2(1-g_j^2)&=R_j^2-R_{j+1}^2.
\end{aligned}
```

<span id="source-block-fd28b54e-d486-4878-ab60-bbd7542488e8"></span>

**证明。** 对 j<M，第 j 个分量在 D=j 时吸收 R_j，在 D>j 时贡献 R_j(1−g_j)。其边缘系数因此是 R_j·Pr(D=j)+R_j(1−g_j)·Pr(D>j)=R_j²(1−g_j²)。末轮系数为 R_M·Pr(D=M)=R_M²；望远镜求和为 1。这是由固定源码两条规则推导出的性质，不是原文额外报告的实验结果。

<span id="source-block-e12afca4-063b-43f9-8aa9-084c304e48ed"></span>

二轮例子取 g₁=0.5、q₁=(0.8,0.2)、q₂=(0.2,0.8)。固定执行两轮的混合为 (0.5,0.5)；随机 D 再作停止混合的文字边缘为 (0.65,0.35)。二者使用同一 gate 数值，输出分布仍不同。不能把 log P_D、log Pr(D) 或 log P̄ 当作同一个评分字段。

<span id="source-block-017ee850-86d6-496c-bda1-72de0fc44b69"></span>

```math
\mathbb E[D]=\sum_{j=1}^{M}R_j,\qquad
\mathbb E[C_D]=\sum_{j=1}^{M}R_jc_j,
\qquad C_d=\sum_{j\le d}c_j.
```

<span id="source-block-728cb73a-be4a-4b84-bf34-33979c1950e6"></span>

c_j 是在这一固定入态和前缀链上执行第 j 轮的确定增量成本。平方出现在边缘读出的分量权重；真实访问概率仍为 R_j，计算成本不能用 R_j² 记账。二轮例子的平均实际深度为 1.5。服务批次中的填充、额外采样和资源压力需另按执行 trace 计费。

<span id="source-block-8b6f6c95-5012-463f-ab74-5c0e5265cb5c"></span>

温度或截断变换作用于每个 P_d 时，应先变换，再以 Pr(D=d) 求和；通常不等于先求 P̄ 再变换。不同 D 还会留下不同 KV，所以跨 token 的文字边缘必须维护历史路由后验，不能把上式在某一条缓存上逐 token 相乘。[完整路由评分 T6](/notes/loop-reasoning/reinforcement-learning/) · [生成语义与系统强停](/notes/loop-reasoning/generation/)

<span id="source-block-886099f9-28e2-49d1-9ec3-7a7cb7e3e63d"></span>

#### 算法 T9 单 token 随机路由的读出与成本

<span id="source-block-1b7879f7-a9a4-4433-b914-7e82f0315ad5"></span>

**输入** 同一固定入态、同一全继续前缀链上的 q₁…q_M、g₁…g_{M−1}，以及非负增量成本 c₁…c_M。各 q 为相同词表上的归一化概率；不可拼接不同 KV 历史的中间输出。

<span id="source-block-3ec6bc36-dc3b-4070-886e-8e1838d68c10"></span>

**输出** 各停止深度的概率与条件读出、真实文字边缘、固定 M 读出、平均实际深度与条件成本，以及枚举式和封闭式的一致性残差。

<span id="source-block-c9193944-c544-4d95-bc71-f26856a3f6cf"></span>

**前置条件与终止** 使用上文理想采样条件。最多处理 M 轮；给定输入后时间为 O(MV)、工作缓冲 O(V)。若保留全部条件分布作审计，另需 O(MV) 存储；获取各轮 q 与 g 的骨干前向成本不包括在这个代数算法中。

<span id="source-block-b3f55b03-5dff-41bc-9ed3-bf8592678e57"></span>

<figure class="research-embed"><iframe src="/research-assets/0bc82dec4d1690a0288812280a2d365d0400993c88919a005ffdcf23925b7905.html" title="算法 T9 单 token 随机路由的读出与成本" loading="lazy" sandbox=""></iframe><figcaption>算法 T9 单 token 随机路由的读出与成本 · <a href="/research-assets/0bc82dec4d1690a0288812280a2d365d0400993c88919a005ffdcf23925b7905.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-cfb372b3-e5ca-4b11-9033-e045ac83a672"></span>

**性质与来源** 这是从固定引擎的 Bernoulli continuation、继续累计及终止尾质量吸收推导的数值参考，不是新增的生产采样器，也不意味着 TaH2–verl 已完成接入。[固定 dispatch](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L573)

<span id="source-block-898e5730-9a3d-4d08-8939-25054f67b0e4"></span>

**教学核对。** 256 组含 g=0／1 和 M=1 边界的小词表案例，将停止深度枚举、平方生存系数、分布归一化和真实访问成本分别比较；边缘分布最大误差 2.23×10⁻¹⁶，归一化误差 3.34×10⁻¹⁶，访问成本误差 3.56×10⁻¹⁵。这不构成已加载模型的行为或性能测量。

<span id="source-block-a2fd97e1-2939-41bb-8a25-31c73b096642"></span>

## 在线深度标签与损失的数学定义

<span id="source-block-9e83defc-3202-418c-a8e0-7fd2236135b7"></span>

I 是有效监督位置集合，A_m 是本轮实际执行位置，z 的前一边界标签决定下一边界的候选资格，初值 z_0=1。停止位置缺失的下一轮预测由隔离的无梯度 lookahead 补齐。

<span id="source-block-3a6d4b93-5181-40c6-a042-b671c9fa814f"></span>

```math
\begin{aligned}
\ell_{m,t}&=-\log q_{m,t}(y_t^*),\\
\delta_{m,t}&=\ell_{m,t}-\ell_{m+1,t},\\
C_m&=I\cap A_m\cap\{t:z_{m-1,t}=1\}.
\end{aligned}
```

<span id="source-block-4ce8312a-78a1-4b7d-be5d-aa97bd311e7f"></span>

在整个 DP 组的候选中收集有限的正改善，降序写为 delta_(1) 至 delta_(n)。覆盖系数取 0 到 1 之间的正数；cutoff 定义为达到指定正改善总量的最小前缀末值：

<span id="source-block-e6258b26-fd71-4189-b6af-0f70adcd7427"></span>

```math
\begin{aligned}
k_m^*&=\min\left\{k:
\sum_{r=1}^{k}\delta_{(r)}
\ge\rho\sum_{r=1}^{n}\delta_{(r)}\right\},\\
\kappa_m&=\delta_{(k_m^*)},\\
z_{m,t}&=\mathbf1\{t\in C_m,\ \delta_{m,t}\ge\kappa_m\}.
\end{aligned}
```

<span id="source-block-fc51b4c9-6c81-4ee0-b35d-28fd9e0d34e1"></span>

没有正改善时，全设为停止。cutoff 处若并列，按阈值纳入的全部位置可能使覆盖量超过 rho；rho 约束改善量，不是 token 数比例。[标签实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py)中的候选掩码、DP 汇集与边界选择应分别追踪。

<span id="source-block-24d0d4ed-4aeb-45a2-bb3e-50c727132d7d"></span>

### 覆盖选择的最优性范围

<span id="source-block-13341b61-6721-462d-972f-3e003abb06bf"></span>

对有限且为正的改善量，降序前缀在“每个位置的额外计算成本相同”的前提下，给出满足覆盖目标所需的最小位置数。证明很直接：任意 k 个位置的改善总量，都不超过最大的 k 个改善之和；若前 k−1 个仍不足，其余任何 k−1 个也不足。源码以 cutoff 比较而不是显式取前 k 个，所以并列值会扩大选集。例如改善为 0.50、0.25、0.25，覆盖 75% 的最小前缀只有两个位置，而 cutoff=0.25 会选中全部三个。

<span id="source-block-ac252eb6-94d9-4b9c-a367-3a0ad7172f13"></span>

该结论不等价于最小 FLOPs 或最短延迟。假设两个位置的改善分别为 0.6、0.4，成本分别为 100、1，覆盖 40% 时按改善排序先选成本 100 的位置，而只选第二个位置已经满足覆盖。因此在动态 KV 长度、不同 kernel 批形状下，原规则应解释为正改善覆盖启发式，不能自动称为最优计算分配。排序、FP64 汇集与阈值比较的对应位置见[coverage 实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L325-L401)。

<span id="source-block-abfade1a-8796-4646-bb17-ff5ad870a561"></span>

源码的 cost_sensitive 权重由候选改善到 cutoff 的距离给出，而不是由 GPU 时间或 FLOPs 给出。候选外的位置填最小权重；这些位置只要实际执行且有监督，仍可能进入 BCE。

<span id="source-block-aa818989-5a7e-4461-af5e-e55647455a1f"></span>

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

<span id="source-block-a2b11aaa-40ba-4867-99cc-8c9c449cd6d3"></span>

cutoff 附近的标签权重较小，远离边界的标签权重较大；权重随后 detach。收益是当前条件上下文中的一步 CE 差，不能直接解释成多轮未来收益或最终答案奖励。[cost 与深度标签](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L284-L309)

<span id="source-block-7e08811b-aaac-47e1-be58-d7114dc075fd"></span>

联合目标分开写成混合预测 NLL 和决策 BCE。a 为停止梯度的代价权重，beta 为可选的正类权重；BCE 的监督范围是所有已执行且有效监督位置，不仅是正候选集合。

<span id="source-block-34483d57-f892-4ad6-b26e-18a0c54b7b62"></span>

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

<span id="source-block-c3aecd11-89cb-4ea1-aaed-5ec10717e8e6"></span>

标签、cutoff 和监督权重不反向传播；已保存的 gate logits、连续混合权重和已执行 updater 路径可以承接损失梯度。监督窗口要求 N>0，并采用训练器传入的归一化约定；当前 forward 内的标签统计和动态正类权重仍可能随微批次划分改变。[损失实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py)

<span id="source-block-ae438bf8-b779-4c20-b145-7d0aeccf8310"></span>

### 加权 BCE 的梯度与概率含义

<span id="source-block-3e981a9c-0fe9-4259-a261-461a9608751f"></span>

令 s 为保存的 gate logit，z 为 detached 二元标签，a 为代价权重，beta 为当次 forward 的正类权重。可选 margin b>0 时，源码先把正类 logit 减 b、负类加 b，再计算 BCE；b=0 是普通分支。对一个决策的导数为：

<span id="source-block-837ce60f-2b93-427d-9a8d-c1dac96bb9aa"></span>

```math
\begin{aligned}
s_{\rm eff}&=s-b(2z-1),\qquad p_{\rm eff}=\sigma(s_{\rm eff}),\\
\ell_D&=a\left[-\beta z\log p_{\rm eff}-(1-z)\log(1-p_{\rm eff})\right],\\
\frac{\partial\ell_D}{\partial s}
&=a\left[(1-z)p_{\rm eff}+\beta z(p_{\rm eff}-1)\right].
\end{aligned}
```

<span id="source-block-ea00f53b-8406-4daa-b692-9b18318a4d99"></span>

总代价还乘辅助损失系数，并按传入的监督 token 分母 N 归一化；分母不是决策次数。因此执行更多边界会增加辅助项的决策数，其尺度不能由单个 BCE 的大小推断。[BCE 与归约](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L463-L570)

<span id="source-block-2cadaf90-32b7-41ec-af6d-2413d68919e2"></span>

在 b=0、固定 beta、给定 gate 特征 x 的条件分布且可逐点自由拟合时，最优 sigmoid 输出满足：

<span id="source-block-795a9689-c2d8-41b7-a7b7-acfeed32da53"></span>

```math
\sigma(s^*(x))=
\frac{\beta\,\mathbb E[az\mid x]}
{\beta\,\mathbb E[az\mid x]+\mathbb E[a(1-z)\mid x]}.

```

<span id="source-block-7f5ca2b8-c0e0-4fbf-9b8f-b35b2253ebf5"></span>

只有再满足相应的权重条件，它才等于未经加权的标签条件概率。若 a 在给定 x 后与标签无关，记 eta=P(z=1|x)，则最优输出为 beta·eta/(beta·eta+1−eta)；阈值 0.5 对应 eta>1/(1+beta)。例如 beta=4、eta=0.2 时输出为 0.5。这一推导解释了类别平衡会改变概率刻度；实际联合 NLL 训练、共享参数、有限容量和动态 beta 又增加了耦合，不能把该点态最优式当成整个 TaH2 模型的已证最优解。

<span id="source-block-4ecd2c4e-5680-41bf-a094-26925598fdde"></span>

源码日志的决策准确率与 TP/FP/FN 以 sigmoid(s)>0.5 计数，并未复用配置 threshold、训练 warmup 阈值或随机路由结果。修改部署阈值后，这组日志仍是 0.5 分类指标，应另外统计实际路由。[指标边界](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L188-L305)

<span id="source-block-7d618fc3-d0cc-4d59-8dc7-8d0bfc70de3d"></span>

#### 算法 T2 多深度标签与加权监督

<span id="source-block-e68edad6-98ef-45a0-9745-b36dc2fe9c9e"></span>

**输入** 每深度 CE 表、实际执行集合、有效监督位置、逐深度覆盖率、保存的可微 gate logits、DP 组与正类权重配置。

<span id="source-block-bf05a712-a574-44b9-8eb8-cf3e54f8ba0d"></span>

**输出** detached 深度标签及代价表、按 token 分母归一化的决策损失。

<span id="source-block-31bdae59-3370-4cc4-8bd5-22dc7cab99d6"></span>

**前置条件** 用于覆盖论证的 CE 有限；所有 DP rank 进入相同的边界 collective 序列；N>0；深度候选已满足真实执行上限。

<span id="source-block-7e060381-81d0-433a-ae36-caea08b863ac"></span>

<figure class="research-embed"><iframe src="/research-assets/e0b85cb518efa55eb32ee831b381a0bca47bf54693810b3cc1ed437ae3430167.html" title="算法 T2 多深度标签与加权监督" loading="lazy" sandbox=""></iframe><figcaption>算法 T2 多深度标签与加权监督 · <a href="/research-assets/e0b85cb518efa55eb32ee831b381a0bca47bf54693810b3cc1ed437ae3430167.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-d0993b8c-bfd0-4961-9075-2073287e2551"></span>

**源码** [`finalize_posterior_labels`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L230-L310) · [`_coverage_threshold`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L370-L401) · [`IterDeciderLoss.final_loss_func`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L307-L570)

<span id="source-block-796cd0a9-ea54-4f99-b21e-bb3f953609d9"></span>

### 收益覆盖的数值例子

<span id="source-block-d2c0f9bb-e9bf-455d-9476-b08753a43c1e"></span>

教学例中，三个候选的正改善为 `0.80、0.19、0.01`，总量为 1。按精确实数计算，覆盖 99% 只需前两个，cutoff 为 0.19。随后 [IterDeciderLoss](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py) 使用标签及代价权重训练决策器；这与用最终答案奖励做 RL 是不同训练信号。

<span id="source-block-8d918da6-2f29-4c1d-9d24-0260ae295ad8"></span>

### 单轮 CE 与混合读出改善的精确关系

<span id="source-block-2b45d556-0f32-4d42-a3ad-0a2111866833"></span>

原文 Eq.5 和 `compute_posterior_score` 使用单轮预测的 CE 差；Eq.4 和 `accumulate_iter` 使用已执行深度的停止权混合。以下关系由两者联立推导，说明这个监督信号如何对应实际读出；它不是原论文另报的实验结果。[原文 §4.1–4.2](https://arxiv.org/html/2609.35748v1#S4) · [原始 logits 的 CE](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L312-L368) · [末轮吸收剩余质量](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L579)

<span id="source-block-e0f4dcd5-2a99-4c32-b2cb-33ee214ce851"></span>

固定一个预测位置、移位目标 y、进入深度 m 的全部上下文，以及该位置的继续链。a、b 为相邻两轮赋给 y 的概率，A 为较浅轮次已累积的目标概率质量，R 为进入 m 的剩余权重，g 为 m 的继续概率。比较“立即停于 m”与“只再算一轮并停于 m+1”；较浅状态、gate、上下文和其他决策固定。采用精确算术，关闭权重下限、温度、截断采样和资源强停，且 a、b>0。

<span id="source-block-4d82e495-8010-40ea-90cd-f65b8383d225"></span>

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

<span id="source-block-7060f0fc-4f30-485a-af13-0bcab90c8a91"></span>

证明只需从后一种读出中减去前一种：变化为 Rg(b−a)，再除以原目标概率。ξ 是“当前轮对立即停止读出的目标概率贡献”再乘继续率；因此同一个单轮增益对已有混合预测的影响可以差很多。当 ξ>0 时，两种改善同号；ξ=0 时，混合改善恒为零。由单调性和 log 的凹性得到以下界：

<span id="source-block-dec55047-0b0d-4540-9c6e-718879b682c5"></span>

```math
\begin{aligned}
\min(0,\delta)&\leq\Delta_{\rm mix}\leq\max(0,\delta),\\
\Delta_{\rm mix}&\geq\xi\delta,\\
\frac{\partial\Delta_{\rm mix}}{\partial\delta}
&=\frac{\xi e^{\delta}}{1-\xi+\xi e^{\delta}}.
\end{aligned}
```

<span id="source-block-060394fd-beeb-4aa8-9fed-b05e859eb39a"></span>

同号不保证跨 token 排序相同。以下构造取两个独立样本的首个监督位置，m=1、R=1、A=0；两个位置都必定执行首轮并进入初始标签候选集，没有更早边界筛选。继续率分别为 0.01、0.5；第二轮概率由实际执行或 no-grad 探测取得。其余词的概率取补数即可得到合法分布。

<span id="source-block-06a6634b-2c7f-4255-8062-657cfcf1caf0"></span>

| 位置 | a | b | g | ξ | 单轮改善 δ | 混合改善 Δmix |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.001 | 0.004 | 0.01 | 0.01 | 1.386294 | 0.029559 |
| B | 0.2 | 0.4 | 0.5 | 0.5 | 0.693147 | 0.405465 |

<span id="source-block-67c14970-1fef-4b8d-b4c0-2dfb89eabc50"></span>

若教学覆盖率取 65%，原始正改善排序会只选 A：它覆盖原始改善的 66.67%，却只覆盖这两个位置混合改善的 6.79%。这不是对论文 ρ=0.99 的实测复现，而是说明两种“覆盖”不是同一个量。若正候选集合 C₊ 的总改善非零，且统一满足 ξ≥ξmin>0，选集 S 的原始改善覆盖 ρ，则有更紧的局部覆盖下界：

<span id="source-block-d7a134c7-6644-47a2-bd71-a3419ce41486"></span>

```math
\frac{\sum_{t\in S}\Delta_{{\rm mix},t}}
{\sum_{t\in C_+}\Delta_{{\rm mix},t}}
\geq\frac{\xi_{\min}\rho}{\xi_{\min}\rho+1-\rho}.
```

<span id="source-block-3d10e37b-f9c4-4c13-91fe-dae5cfe97eb2"></span>

证明用到：选中部分的混合改善至少为 ξmin 乘其原始改善，未选部分的混合改善至多为其原始改善，再取二者的比例下界。没有正的 ξ 下界，不能沿用原覆盖百分比。这仍只比较固定上下文下的局部收益表，不包含联合续算产生的 KV 交互。

<span id="source-block-8c10a516-3fb5-49b0-b5a5-e26c1202a2df"></span>

若将额外一步的成本惩罚写为 λc≥0，在上述局部比较和 ξ>0 条件下，可精确写出对应原始增益的阈值。它是诊断参照，不是源码使用的新标签规则：

<span id="source-block-9d886b64-acf5-4947-b1b5-0bde59abc45d"></span>

```math
\Delta_{\rm mix}>\lambda c
\quad\Longleftrightarrow\quad
\delta>\log\!\left(1+\frac{e^{\lambda c}-1}{\xi}\right).
```

<span id="source-block-09c51817-c930-408b-9ebc-c8c584207310"></span>

工程上应同时记录 `raw_ce_gain`、`mixed_nll_gain`、`target_mass_before`、`remaining_mass` 和 `continue_prob`。替换成混合增益标签未必更好：当前 gate 较小会压低 ξ，可能进一步减少弱 gate 的训练机会；原始增益则衡量下一轮预测分量的潜在改善。两者代表不同监督目标，需要配对实验判断。真实代码还 clamp 混合权重并可能以模型 dtype 累积；复核恒等式时先用 FP64 参考读出。计算 Δmix 可用 logaddexp(log(1−ξ),log ξ+δ)，并单独处理 ξ=0/1，避免先计算 exp δ 溢出；实际 dtype 与权重下限误差另记。

<span id="source-block-eb4512d4-c39f-4c8d-ab79-0db6504aa61f"></span>

### DUO 探测对应哪一种干预

<span id="source-block-c8aecc55-c31a-4778-9bd4-3a95fe4a2701"></span>

固定一个深度边界，把主路径实际续算的位置记为 C，把同轮停止但具有已存输入的位置记为 T。以下仅分析一次 backbone 调用：进入该轮的 updater 输入 X 和较浅 KV 快照 K 都固定。令 Fₜ(B;X,K) 表示本轮只让位置集合 B 参加、使用普通 DUO 因果可见性时，位置 t 的单轮预测分布。

<span id="source-block-eb4d07ec-3591-4f72-a352-4b84e6f4a7d9"></span>

posterior mask 只允许 query 读取当前轮真实继续位置的 KV 和自己的 KV，其他假想停止位置互不可见。要将停止位置 t 的 side pass 输出解释为单位置干预，还需明确计算条件：采用精确算术，归一化与 MLP 按位置独立计算，并且不存在跨样本或跨位置的容量竞争。随机性也必须一致：关闭 dropout，或按原位置和层显式耦合相同随机变量；不同 compact 形状下仅设置同一个 seed 不够。

<span id="source-block-d7ee06ba-2240-4f4d-aa93-3696752777dc"></span>

```math
q_{t,\rm side}^{(m+1)}
=F_t(C\cup\{t\};X,K),\qquad t\in T.
```

<span id="source-block-ceaa7dcf-54ae-4351-b9f2-e4367b9cabbf"></span>

这个结论可逐 backbone 层归纳。位置 t 之前的真实继续位置不能因 t 的加入而改变；t 只看到这些合法前驱及自己；更晚位置即使在两种运行中不同，也不能反向影响 t。side pass 的并行转发因而能同时计算多个单位置查询值。这里只等价于冻结 X、K 的本轮图；并未重新生成整段文字，也不等价于重新执行全部历史深度。若把该设计迁移到跨 token 竞争容量的 MoE，应重新检查等价条件。[posterior mask 的 `continue_kv | own_position`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py#L269-L308) · [side 输入与快照](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L79-L166)

<span id="source-block-e4f224d4-e82f-49e9-8d17-b0680ce2fcca"></span>

若把多个停止位置组成 S 并同时续算，目标位置 t 的实际新图允许较早的 S 内位置提供 KV，故一般不能继续使用同一个 singleton 值：

<span id="source-block-bbf8e7a7-e625-49c2-8ff1-bc3acdb7bad6"></span>

```math
F_t(C\cup S;X,K)
\ne F_t(C\cup\{t\};X,K),\qquad t\in S.
```

<span id="source-block-71884b63-63d4-421d-b1d7-44d99f85bb44"></span>

把同一 teacher-forced 前缀上的停止混合 NLL 记为 L(S)，其中只改变本轮活动集合 C∪S，并统一在 m+1 停止；所有目标、已有质量及更浅状态固定。定义整组目标改善 G(S)，以及两个位置的交互项：

<span id="source-block-0297a48e-023d-4799-9464-f77cc8d6699e"></span>

```math
\begin{aligned}
G(S)&=L(\varnothing)-L(S),\qquad G(\varnothing)=0,\\
\mathcal I_{ij}
&=G(\{i,j\})-G(\{i\})-G(\{j\}),\qquad i<j.
\end{aligned}
```

<span id="source-block-4fd8974f-512b-4b52-b5df-77f539b60aac"></span>

即使已把原始 CE 增益转换成该位置的混合 NLL 增益，G({t}) 仍可能不同：新增 KV 还会影响较晚的真实继续位置。进一步，交互项也未必为零。因果性限定影响方向，并不保证收益可加，更不保证交互一定正或一定负。

<span id="source-block-bdf80ac8-b3f3-4756-acf3-c3acc5c3ed06"></span>

一个可手算的反例：位置 0 是固定前驱，位置 1、2 是停止候选，三个位置的首轮 KV value 均为 0；第二轮两个候选的 value 为 0.2、4；所有 attention logits=0，读出为 sigmoid(attention 输出)。候选的首轮目标概率都为 0.5、gate 都为 0.25，因此默认阈值 0.5 下都停止。位置 1 第二轮看到两条较浅 KV 与自己，输出 sigmoid(0.2/3)；位置 2 单独续算看到三条较浅 KV 与自己，输出 sigmoid(4/4)；联合续算则输出 sigmoid(4.2/5)。这个构造保留 DUO 的自身与前驱缓存条目，但使用简化的标量 attention，并非 Qwen 或论文评测数值。

<span id="source-block-7fb6224b-f19f-4819-8ea2-a62d0269aeb9"></span>

| 续算集合 S | 最终 P₁(y₁) | 最终 P₂(y₂) | 整组混合 NLL 改善 G(S) |
| --- | --- | --- | --- |
| ∅ | 0.500000 | 0.500000 | 0 |
| {1} | 0.504165 | 0.500000 | 0.008296 |
| {2} | 0.500000 | 0.557765 | 0.109329 |
| {1,2} | 0.504165 | 0.549616 | 0.102908 |

<span id="source-block-92179eb4-6242-43ec-b0c9-6add61a92213"></span>

交互项为 −0.014717 nats；把两个 singleton 收益相加会高估联合收益。由此，原论文的正增益覆盖仍是给定当前轨迹的局部监督规则。把它解释成“全序列最优计算预算分配”还需要联合干预、后续损失和实际成本的额外证据；本页前面的等成本最小位置数证明只针对固定的数值收益表。

<span id="source-block-6ae476ab-2c15-4365-bb6e-babf0237fb55"></span>

#### 算法 T10 固定前缀的续算收益诊断

<span id="source-block-32a099ee-e57a-4ec1-8532-624fc28976e1"></span>

**输入** 固定检查点与语义配置、有效移位目标集合 I、边界快照、下一轮输入 X、真实继续集合 C、候选停止集合 T、待检查的 singleton 与 pair 集合。快照保存：第 m 轮混合累加之前的目标质量 A、进入 m 的 R、当前单轮目标概率 a、gate g，以及更早已停止位置的最终目标概率。当前 runtime 中已吸收停止尾质量的 buffer 不能直接当成 A。KV 只保留真实执行到 m 的条目及 position／depth／valid 元数据；当前轮同一样本的原位置必须唯一。

<span id="source-block-3634ae04-44fd-4cfb-8f2a-7914f3db6b83"></span>

**输出** 每候选位置的原始 CE 改善、局部混合改善、整组 singleton 改善；每 pair 的联合改善与交互项；每次参考运行的实际调用范围和数值误差。算法中的 Pᴮ 是 I 上的目标概率记录，qᴮ 是本轮 B∩I 上的单轮目标概率记录，不声称恢复完整词表分布；完整分布版本还需保存全词表累加量。

<span id="source-block-9f7921ec-5252-4e6c-a4b7-b9a9dbcab39b"></span>

**前置条件** 所有候选在 m 已实际执行且 m<M；固定输入文字和目标，关闭模型随机性；模型参数不更新；本诊断将当前边界统一限制到 m+1。干预只改变 hard 活动集合，保持原 soft gate g，不把 g 改成 1；新末轮吸收剩余 Rg。参考分支采用精确停止混合，不加入采样、权重下限或资源强停。每次运行使用独立可追加缓存；分布式实现必须让所有 rank 依次参与相同调用。下面给出诊断算法，并未改动 TaH2 原有标签生成器。

<span id="source-block-df6ca167-bae8-4094-b313-216be911e0a0"></span>

<figure class="research-embed"><iframe src="/research-assets/3cdd17bab9adbf5786808a589daa39d1c9076909052cc970d47c94f71caaac31.html" title="算法 T10 固定前缀的续算收益诊断" loading="lazy" sandbox=""></iframe><figcaption>算法 T10 固定前缀的续算收益诊断 · <a href="/research-assets/3cdd17bab9adbf5786808a589daa39d1c9076909052cc970d47c94f71caaac31.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-93bb01be-657a-41b6-b93f-91212af7e593"></span>

**实现落点** 在 `accumulate_iter` 写入第 m 轮之前记录 A、R，另外保存本轮 a、g；用 `capture_posterior_inputs` 与 `side_pass_continue_mask` 获取输入和真实 C。对一起始探测深度 m+1，源码零起始 `iter_depth=m`、`snapshot_iter(m-1)`，必须排除本次真实 C 已生成的 m+1 KV，避免重复或泄漏。用 posterior mask 计算 singleton 查询，用普通 DUO mask 重算 C∪S 的联合图，不能直接拼接探测 logits。损失只在固定 I 上归约，记录 `counterfactual_scope=frozen_prefix_one_boundary`。[runtime](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L483-L579) · [快照与探测](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L117-L165) · [可见性实现](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/sparse_ops.py)

<span id="source-block-c8539f3d-6d75-4643-8a8c-553f59eed460"></span>

## 算法 3 TaH2 前向与辅助监督

<span id="source-block-c527c3dc-1e88-48b1-bca4-590837ba6621"></span>

以下使用 dense 逻辑位置描述语义，实际代码会压紧与散射活动位置。当前轮是否继续必须先由部署 gate 决定；监督标签在主路径之后生成，不能反过来替代这一决定。

<span id="source-block-25b3857a-2ac4-43b4-851c-226b58f9605f"></span>

**输入** token batch、有效 mask、可选监督标签、最大深度 M 与阈值 tau

<span id="source-block-79b0e712-ed58-46b4-b89f-66d48bcf804e"></span>

**输出** 混合 logp、主缓存、实际深度、可选 NLL 和决策损失

<span id="source-block-dfdfd222-6b3e-40a5-94b1-b8635fd3c773"></span>

**约定与范围** 按原位置 u=(batch,position) 索引，pack/scatter 显式转换，TopK 只取概率值。该语义算法采用 DUO、stop_prob_mix、空初始缓存及确定性阈值路由，关闭 warmup、随机路由、深度 override 和强制续算；权重未 clamp，按精确算术书写。第 9 行末层 g=0 是等价的终止记号，源码使用有限哨兵并由 finished 分支吸收尾部质量。算法中的“有监督”还要求配置后验标签生成器并启用该路径：训练或 return_eval_metadata；仅给普通 eval 传 labels 不保证运行辅助探测。

<span id="source-block-6bb2705a-e440-4dd8-a736-5923abcd48ad"></span>

<figure class="research-embed"><iframe src="/research-assets/6ad05533e7efc3078fa11d8f90f37aad3841f4d4934356e4092cf53950e47e9b.html" title="算法 3 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 3 操作与数学公式 · <a href="/research-assets/6ad05533e7efc3078fa11d8f90f37aad3841f4d4934356e4092cf53950e47e9b.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-ef8eec31-e7bd-4686-adb8-317ee941d105"></span>

这是语义算法，省略了 TP、dummy collective、全局归约、数值 clamp、override 和 checkpoint 重算。原缓存张量可由 snapshot 共享，但后续不能原地改写旧条目。零起始源码深度与此处的一起始数学记号也需显式转换。

<span id="source-block-30e16df0-461d-4aa6-9cef-ca2b51a781fd"></span>

| 数学量 | 工程表示 | 关键不变量 |
| --- | --- | --- |
| 活动集合 A_m | dense bool `[B,T]` 与压紧索引 | 稀疏行始终能映射回原位置 |
| E、X、H | `[B,T,d]` 逻辑缓冲或活动布局 | 原始 E 与当前状态不混用 |
| 继续概率与剩余质量 | 浮点 `[B,T]` | 最后执行轮吸收全部剩余质量 |
| C 与可见性 | 分层分深度 KV、位置和 valid mask | 保留文字因果性及深度约束，探测不污染主状态 |
| 混合目标 logp | 全词表路径或 target-only `[B,T,1]` | 最后一维1表示目标概率，不是1个类别 |
| 标签与 gate logits | detached 标签与保留计算图的 logits | 真标签不进入部署路由，BCE 范围为已执行监督位置 |

<span id="source-block-85ca3386-e564-46aa-97a1-b718bb9dc368"></span>

## 交付与验收

<span id="source-block-8923234c-e43b-4322-820c-3bdf33ada01f"></span>

### 实现验收清单

<span id="source-block-c001a6f2-b11f-4fe3-ba8c-6de3221c5d03"></span>

- [ ] `loop_trace.json`：每轮活动位置、原始 embedding、updater 输入输出、继续概率与实际深度。

<span id="source-block-f6c5debe-e3fd-416f-bfa8-9b5664859b64"></span>

- [ ] `visibility_table.json`：主路径和 lookahead 分开记录 query 到 key 的允许关系。

<span id="source-block-573aec8b-9a43-4102-8c93-160b58f382c8"></span>

- [ ] `cache_side_effect_report.json`：探测与 backward 重算后，主路径旧 KV 的数量、内容和别名约定保持正确。

<span id="source-block-4143e74e-3426-4637-ad3d-6019d616300f"></span>

- [ ] `mixture_and_grad_report.json`：手算混合与 runtime 对齐，token NLL 和决策 BCE 的梯度路径分别说明。

<span id="source-block-dda5dfcb-db01-4bec-9001-a7566b2ffc58"></span>

### 循环记忆的研究扩展

<span id="source-block-ebb716ea-4d7a-4578-9d6c-6afc6a0d7fb1"></span>

历史 KV replay、跨轮记忆和其他循环位置可在此状态接口上扩展；每次改变可见性，都要重新给出参考计算与缓存生命周期。更深不自动代表更有效的推理。

<span id="source-block-a8891084-c55e-4deb-bfb7-facd23264bc3"></span>

## 动手检查

<span id="source-block-19349fe6-f397-4796-89c3-feb660491404"></span>

画一个位置执行两轮的计算图，用不同箭头分别标记梯度依赖与缓存读写。然后回答：如果只把主路径 updater 的输出 `detach`，保持其他代码不变，第二轮答案损失还能通过这条回填边训练什么？

<span id="source-block-3f75bcdf-e5ad-48fe-96e4-b3c0e3c7ad7d"></span>

### 参考思路

<span id="source-block-f0dcceca-c549-4e29-a419-1a76c1bacd71"></span>

第二轮仍可训练自身计算中参与的可训练参数，但这条回填边不再把梯度传回 updater 及生成其输入的前轮计算。其他损失、共享参数和缓存依赖仍可能产生梯度，因此不能笼统说第一轮没有梯度。

<span id="source-block-96848178-2559-4b6c-a9a4-ac60ee612a00"></span>

对我们的研究而言，画清这张图以后，才有依据决定教师信号放在哪里，以及哪些接口能被最终任务训练到。

<span id="source-block-9dd198e6-0c96-4378-af94-208169b406f3"></span>

## 围绕本课讨论

<span id="source-block-ecf3183b-2726-4132-b774-f3aabc1e0e0b"></span>

```text
请从预测位置与移位目标的定义开始，用数学状态、DUO可见性、停止决策和概率混合刻画TaH2，再按前向算法追踪每轮的输入、KV增量、活动集合和剩余质量。证明混合权重和为1；单独解释lookahead的专用mask、标签生成与BCE梯度。最后逐项核对固定源码的张量布局、索引、detach、缓存别名和多卡collective，明确参考算法省略的优化。
```

<span id="source-block-9b0a392f-fa7b-41e3-939c-73661c7c5af2"></span>

## 继续学习

<span id="source-block-745e163d-cecb-4d84-81fe-b7c4a9d78d9e"></span>

[上一课：一次参数更新](/notes/loop-reasoning/parameter-update/) · [学习地图](/notes/loop-reasoning/engineering/) · [下一课：从训练到自主生成](/notes/loop-reasoning/generation/)
