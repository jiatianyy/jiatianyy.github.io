[基座矩阵与七篇论文](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [预训练](https://chatgpt.com/space/page_4f0508753a6081919a82b0c89213fd6c) · [后训练](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

# 研究问题和阅读路线

LoopMoE是从头预训练的原生loop MoE，采用prelude–shared body–coda的sandwich结构。它去掉每轮的prelude再注入，用token状态和迭代编号共同产生调制信号。阅读时先看递推结构与IterAdaLN，再看容量配平、组件消融和路由动力学，最后把这些机制接到推理成本与后训练接口。

原论文为Chen、Li、Huang、Yin、Shang、Qin的《LoopMoE: Unifying Iterative Computation with Mixture-of-Experts for Language Modeling》。上传英文版是EMNLP 2026 / 2026.emnlp-main.2160（16页），对应arXiv:2606.04438v2；其循环顺序与Ouro整栈循环、Loopie逐层立即复用分别登记。[上传原论文](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal)；[ACL 正式入口](https://aclanthology.org/2026.emnlp-main.2160/)；[arXiv v2](https://arxiv.org/abs/2606.04438v2)

## 1 设计目标与比较基准

论文比较共享权重的重复计算与新增独立参数的作用。重复dense stack时，resident参数与每token FLOPs通常难以同时对齐非循环基线；MoE则可以分别调整expert pool大小、top-k，以及attention与FFN的容量分配。LoopMoE据此提出两项设计：IterAdaLN为共享计算加入显式迭代条件，capacity balancing调整active容量的分配。[§1、§3](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=1)

| 配置 | 3B LoopMoE | 9B LoopMoE | 对照 |
| --- | --- | --- | --- |
| Prelude / body / suffix 物理层 | 2 / 2 / 2 | 3 / 3 / 3 | Vanilla MoE 各层独立 |
| body 重复次数 | 4 | 4 | Vanilla 无循环 |
| 物理层 / 逻辑层 | 6 / 12 | 9 / 18 | 12 / 12；18 / 18 |
| 原文 total 参数 | 3B | 9B | 对齐同尺度 Vanilla |
| 原文 physical active | 0.6B | 1.3B | Vanilla 为 0.8B / 1.9B |
| 表中 FLOPs 对齐的 active 标记 | 0.8B | 1.9B | 该数不能与 physical active 混用 |
| routed / shared experts；top-k | 136 / 2；6 | 136 / 2；6 | Vanilla 为 64 / 2；6 |
| 预训练 tokens | 200B | 300B | 同尺度使用相同预算 |

注意 0.8B / 1.9B 在主表脚注被用作计算量对齐标记，不能据此宣称 LoopMoE 每 token 访问了同样数量的独立物理参数。总参数和 active 数均按原文四舍五入尺度报告；缺少实际配置与权重时，不能将这些数当成精确张量计数。[Tables 1、2、7、9，pp.6、13–14](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=6)

### 带迭代条件的共享算子

将第k轮完整body记为Fθ,k。各轮复用参数θ，但接收不同的iteration条件，因此应按执行顺序复合这一组算子。只有移除迭代条件，或证明该条件不影响映射时，才能把它写成同一个自治算子的K次幂。

```math
\begin{aligned}
G_{\theta,K}&=C_\theta\circ F_{\theta,K}\circ\cdots
\circ F_{\theta,1}\circ P_\theta,\\
D_{\rm exec}&=L_p+KL_b+L_s,\\
D_{\rm physical}&=L_p+L_b+L_s.
\end{aligned}
```

3B 配方是 2+4×2+2=12 次执行、6 个物理层；9B 是 3+4×3+3=18 次执行、9 个物理层。把相同 θ 用于 k=1 和 k=4 不意味着这两个映射相同，也不要求它们共享固定点。因而后训练中“运行前三轮”“跳过第一轮再运行后面三轮”“把 2,3,4 重新编号为 1,2,3”是三个不同的函数。

主比较的条件向量至少包括 total resident 参数、每 token 计算口径、训练 token 数与顺序、有效深度、评测协议。两组 3B/9B 分数只能描述这些给定点的模型差异；两点正增益并不确定关于规模的幂律指数，也不证明任意更大模型都有更大收益。

依据：[§3.2–3.4、Tables 1–2、7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=3)。算子族记法及三种删轮操作的区分由论文计算图直接导出。

## 2 递推结构与IterAdaLN

共享 body 每轮使用同一组权重，但它接到的状态承接了上一轮计算。LoopMoE 进一步把轮次 k 也作为输入，使同一组权重能根据“当前状态是什么”和“现在执行到第几轮”共同产生更新。IterAdaLN 改变的是归一化后的缩放、平移及 attention 输出门值；它不需要为每轮复制一份 body 参数。

令 prelude 输出为 H_pre。状态只在开始时从它初始化，第 k 轮顺序执行 body 内的全部层，再进入下一轮，最后交给独立 suffix 和语言模型读出。下面 F 的第二个输入 k 表示显式轮次条件；即使去掉 k，状态本身的变化也可能让各轮输出不同。这样，状态本身决定本轮接收什么内容，k 则让共享计算能够针对不同轮次学习不同调制。

```math
\begin{aligned}
H_0&=H_{\rm pre}=P_\theta(E_\theta(x)),\\
H_k&=F_\theta(H_{k-1};k),\quad k=1,\ldots,4,\\
Y&=C_\theta(H_4).
\end{aligned}
```

每层由MLA attention和fine-grained MoE组成，embedding后的第一层使用dense FFN。每条分支都在输入处归一化，并在输出加入residual之前再作RMSNorm，这就是这里的sandwich normalization。共享body的两处输入归一化由IterAdaLN替代。

下面用h表示当前loop layer的输入，m表示attention更新后的中间状态。A和M已经包含各自输出处的post RMSNorm；这样，Eq.8–9的简写就与§3.1的归一化顺序一致。[§3.1–3.3、Fig.1、Eqs.1–9，pp.3–4](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=3)

```math
\begin{aligned}
v_k&=\operatorname{MLP}_{\rm iter}(\operatorname{PE}(k)),\\
c^{\rm attn}_{k,t}&=\operatorname{Linear}_{\rm attn}(h_t)+v_k.
\end{aligned}
```

```math
\begin{aligned}
(\gamma^{\rm attn}_{k,t},\beta^{\rm attn}_{k,t},\alpha_{k,t})
&=\operatorname{MLP}^{\rm attn}_0(c^{\rm attn}_{k,t}),\\
\operatorname{I}(h;\gamma,\beta)
&=(1+\gamma)\odot
\frac{h}{\sqrt{\operatorname{mean}(h^2)+\epsilon}}+\beta.
\end{aligned}
```

```math
\begin{aligned}
m_t&=h_t+\alpha_{k,t}
A\!\left(\operatorname{I}
(h;\gamma^{\rm attn}_{k},\beta^{\rm attn}_{k})\right)_t,\\
c^{\rm moe}_{k,t}&=\operatorname{Linear}_{\rm moe}(m_t)+v_k.
\end{aligned}
```

```math
\begin{aligned}
(\gamma^{\rm moe}_{k,t},\beta^{\rm moe}_{k,t})
&=\operatorname{MLP}^{\rm moe}_0(c^{\rm moe}_{k,t}),\\
h'_t&=m_t+
M\!\left(\operatorname{I}
(m;\gamma^{\rm moe}_{k},\beta^{\rm moe}_{k})\right)_t.
\end{aligned}
```

这些式子可以沿数据流来读。PE 先把 k 变成固定 sinusoidal 编码，再经过可学习 MLP，形成轮次信息。attention 条件把这份信息与当前 h 的线性投影相加，生成该 token 的 γ、β 和 α；因而同一轮的不同 token 也可以接受不同调制。底层 RMSNorm 没有静态可学习 scale，归一化后的通道改由动态 γ 调节，再加上动态 β。

attention 完成更新后得到 m，MoE 的条件网络才读取 m。它有自己的线性投影和输出 MLP，反映的是 attention 已经写入的内容。若提前用旧 h 生成 MoE 条件，两条分支之间的这个先后关系就被改变了。

最后要区分两种权重：α 乘在 attention 分支输出上；MoE 没有对应的额外输出 gate，其 residual gate 固定为 1，但 routed experts 内部仍按 router 权重聚合。论文未给出调制输出的完整张量尺寸，所以保留原文 αA 记法；α 是标量还是逐通道、如何广播，仍需实际配置确定。

路由层可在算子层记为 shared experts 的输出与 top-6 routed experts 的加权输出之和；具体 score function、重归一化、expert-bias 更新率须由作者代码补证，不能照搬别篇的 sigmoid/top-2 设置。

```math
\begin{aligned}
M_{\rm raw}(u)
&=\sum_{j=1}^{2}E_j^{\rm shared}(u)
+\sum_{e\in S(u)}w_e(u)E_e^{\rm routed}(u),\\
S(u)&=\operatorname{Top6}(r(u)).
\end{aligned}
```

这是依据 §4.2 shared-expert strategy 给出的结构化算子表达，不是已验证的作者函数签名。Appendix A.3 将 MoE-branch gate 移入加权和，解释为何选择 attention-only gating；严格说这不自动证明任意逐通道 gate 都能被传统标量且归一化的 router 等价吸收，也不替代独立的 gate 消融。

### Eq.10 的分配律成立，但完整 MoE gate 的冗余需要额外条件

Appendix A.3 把分支 gate 乘入 selected expert 权重。对固定输入、固定专家输出，以及没有中间非线性归一化的 raw weighted sum，标量乘法确实满足下式。这里“代数上能写为新系数”与“原 router 函数族能产生这些系数”是两个命题。

```math
\alpha\sum_{e\in S}w_eE_e
=\sum_{e\in S}(\alpha w_e)E_e.
```

若 router 强制非负且入选权重和为 1，其输出属于所选 expert 输出的凸包；乘 α 后一般已离开该凸包。最小例子取两个 expert 输出 (1,0)、(0,1)，原权重各 1/2，α=2；目标变为 (1,1)，任何和为 1 的非负权重都不能产生它。把权重乘 α 再归一化，也只会在 α>0 时得到原权重，不能保留输出幅度。这个反例是条件性的：LoopMoE 实际是否归一化仍须官方实现确认，不能替作者指定 router。

```math
w\in\Delta^{k-1},\ \alpha>0
\quad\Longrightarrow\quad
\frac{\alpha w_e}{\sum_j\alpha w_j}=w_e,
\qquad
\sum_e\alpha w_e=\alpha.
```

还需核对另外三个实现条件。第一，若 α 是逐通道向量，修改标量 router 权重未必能实现它：所有 expert 都输出 (1,1) 时，任何标量线性组合仍在同一条直线上，逐通道 gate (2,1) 的目标却离开该直线。第二，主模型还有两个 shared experts；只改 routed 权重无法同时缩放 shared 分支。第三，§3.1 的 sandwich normalization 在分支输出后放置 RMSNorm，完整残差 gate 若位于这个归一化之后，就不能径直移动到归一化之前的 router 权重上。[§3.1、§4.2、Appendix A.3、Eq.10](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=12)

令 n_ε 是不含静态 affine 的 RMSNorm，q=‖z‖²/d；即使只考虑标量 gate，也有下列不交换关系。固定的逐通道 RMSNorm scale 不消除这一问题。

```math
n_\epsilon(\alpha z)
=\frac{\alpha z}{\sqrt{\alpha^2q+\epsilon}},
\qquad
\alpha n_\epsilon(z)
=\frac{\alpha z}{\sqrt{q+\epsilon}}.
```

当 z≠0、ε=0、α>0 时，前者等于 n₀(z)，后者仍保留 α。取 z=(1,0)、α=2、ε=10⁻⁶，两种次序的最大坐标差约为 1.4142。这是对 gate 移位的反例，不是说论文实际已经实现了一个错误的 MoE gate；本文实际选择的是不设置该 gate。Eq.10 本身没有证明包含 shared experts 与 post-RMSNorm 的完整模块具有同一个可表达函数集合。

“干扰负载控制”也要按计算图分层说明：在一次固定 forward 中，若 gate 只乘在选路完成后的分支输出上，那么 logits、top-k 身份和由身份计数得到的负载不会被该乘法直接改变。训练时它可以改变 task gradient，进而影响后续 router 参数或控制器轨迹；若 controller 读取 gate 后的量，才可能有额外直接依赖。Appendix A.3 未给出足以核验这些依赖的 router/bias-update 代码，Table 3 也没有单独开关 MoE gate 的消融。因此目前保留 attention-only gate 作为作者的设计与实验配置，不把其必要性写成已经证明的普遍定理。

工程核验应分别记录 raw routed sum、shared sum、post-RMSNorm 输出、gate 广播形状，以及 controller 读取的 tensor；对每一种拟议 gate 吸收只比较同输入下的完整分支输出和梯度。以上反例均为小型代数/数值检验；未借用 Done Right 的 router 或同名 Foil 仓库替代作者实现。

### 动态归一化的导数包含条件网络

先看单 token 的 affine-free RMSNorm。以下 ε>0，范数使用欧氏范数；完整序列的 RMSNorm Jacobian 是各 token 矩阵组成的分块对角矩阵。

```math
\begin{aligned}
n(h)&=h/s,\qquad s=\sqrt{\|h\|^2/d+\epsilon},\\
J_n(h)&=\frac1s I-\frac{hh^\top}{d s^3}.
\end{aligned}
```

对与 h 正交的方向，特征值为 1/s；沿 h 的方向为 ε/s³。RMSNorm 并没有与输入尺度无关的单位 Lipschitz 常数。更关键的是，γ 与 β 本身依赖当前 hidden，故 IterAdaLN 的完整导数还有两项：

```math
\begin{aligned}
u(h)&=(1+\gamma(h))\odot n(h)+\beta(h),\\
J_u&=\operatorname{Diag}(1+\gamma)J_n
+\operatorname{Diag}(n(h))J_\gamma+J_\beta.
\end{aligned}
```

固定当前 k，设 B 为将 α 按实际实现规则广播到 hidden 形状的线性映射；α 的精确张量宽度尚未由作者代码核验，此记法同时容纳每 token 标量与逐通道 gate。A、M 均含分支末尾的 RMSNorm，u_A、u_M 分别为两条条件归一化路径。一次完整层更新的 Jacobian 为：

```math
\begin{aligned}
m&=h+(B\alpha(h))\odot A(u_A(h)),\\
J_m&=I+\operatorname{Diag}(B\alpha)J_AJ_{u_A}
+\operatorname{Diag}(A(u_A))B J_\alpha,\\
J_{h'}&=(I+J_MJ_{u_M})J_m.
\end{aligned}
```

这里 Jᵤᴹ 必须在 m 处求导，包含 `Linear_moe(m)` 和其调制 MLP 的路径。把 MoE 条件提前用 h 计算，会删改上式依赖而产生另一种架构。对于 MoE，导数是在 top-k 集合不变、分数无并列的区域内理解。

这也解释了为什么平均 γ 由负变正不能作为稳定性证明：Jγ、Jβ、Jα 与 attention/MoE 本身的导数同样影响增益。完整 K 轮扰动传播是各实际调用 Jacobian 的有序乘积；单次平均门值不足以确定其谱范数或谱半径。

依据：[Eqs.3–9、§3.1–3.3、Appendix A.3](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)。以上为链式法则展开，并非论文报告的实测 Jacobian。

## 3 更新顺序与训练动力学

按论文语义，一次 body 层执行应顺序落实为：

1. 保存本层旧 residual h；以 h 和 iteration k 生成 attention 条件及 γ、β、α。

2. 对 h 作 affine-free RMSNorm 后动态缩放、平移；运行因果 MLA，作分支输出 RMSNorm。

3. 用 attention gate 调制这一分支输出，再与旧 h 相加得到 m。

4. 以 m 和同一个 iteration k 重新生成 MoE 条件；不能在 attention 更新前预计算并复用。

5. 对 m 作动态归一化；计算 router score、top-6 索引、权重，执行 2 个 shared expert 和入选 routed experts，聚合并作分支输出 RMSNorm。

6. 将 MoE 输出直接加到 m，得到当前物理层输出；按 body 层顺序推进，完整 body 结束后才把 iteration 加一。

7. 第 4 轮之后进入 suffix / LM head。全过程没有重复注入 H_pre，也没有已训练停止分支。

上述是原文计算图的算法拆解；尚无已验证作者源码可把每步绑定到真实类名、张量布局或融合 kernel。[§3、Fig.1](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=3)

零初始化决定条件网络刚开始训练时能改变什么。MLP₀ 的输出 projection 为零，使初始 γ=β=α=0；attention 输出乘上 α 后暂不写入 residual。MoE 则没有这个 α gate，γ=β=0 只是让其输入回到普通 affine-free RMSNorm，MoE 分支仍然计算并写入一般非零的输出。

因此，原文的“near-identity”表达了逐步引入调制的训练动机；仅靠零调制，整个 block 并不精确等于恒等映射。要分析状态是否收缩、梯度是否有界，还需观察分支权重初始化、Jacobian、更新范数、梯度和 router 负载轨迹；正文对这些实现量的报告仍不充分。后文的首步梯度展开据此分别分析 attention 与 MoE 哪些路径最先开始学习。[Eq.5–9，p.4](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)

训练数据是 Dolma3Mix；3B 使用完全相同的 200B-token subset、document order 与 packing，9B 为 300B。优化器为 AdamW、β=(0.9,0.95)、cosine、2000-step warmup；3B / 9B 峰值 LR 分别 8.6×10⁻⁴ / 5.9×10⁻⁴。Table 7 最后一行原名为 “Min. LR ratio”，数值 7e-6 / 5e-6；本页按原标签保留，不擅自把它改成“绝对最小学习率”。报告结果均为同 seed 的单次运行，没有跨 seed 误差条。[§4、Appendix A.1、Table 7，pp.5、12–13](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=5)

自回归预训练的 token prediction 项可形式化为下面的标准目标；但原文没有给出可核验的完整训练 loss 配置。Appendix A.3 提到 load-balancing loss 与 expert-bias update，并未报告系数/更新率。下式仅标记待落实接口，不能当作已经复现的作者损失：

```math
\begin{aligned}
L_{\rm next}(\theta)
&=-\frac{1}{N_{\rm target}}\sum_t
\log p_\theta(x_{t+1}\mid x_{\le t};K=4),\\
L_{\rm train}
&=L_{\rm next}+\lambda_{\rm bal}L_{\rm bal}
+\text{其他实现项（若有）}.
\end{aligned}
```

训练所需的 global batch、context length、精确 d_model / heads、调制 MLP 宽度、完整初始化、clip/weight decay、精度和并行拓扑没有在 Table 7 完整列出；无法从“following DeepSeek-V3”补出确定数值。循环是否 full-BPTT、是否 activation recompute、是否 detach 也没有已验证实现入口。原文没有提出截断反向算法，应将实际梯度实现标为未核验，而不是默认复用 Huginn 的 truncated backprop。

### 零初始化在两个分支上产生不同的梯度开关

假设论文所述两个 MLP₀ 的输出 projection 和偏置都初始化为零。此时调制量为零，而且它们对条件输入的导数也为零。沿当前层的计算路径，前向与状态 Jacobian 因而化为：

```math
\begin{aligned}
m&=h,\\
h'&=h+M(n(h)),\\
J_{h'}&=I+J_MJ_n.
\end{aligned}
```

所以 attention 分支关闭，MoE 分支仍然执行，整个层不必接近恒等。令 gₘ 是传回 attention 更新后状态的完整伴随量，gᵤᴹ 是传回 MoE 动态归一化输出的伴随量；它们可以包含后续层与负载辅助项的贡献。初始化时：

```math
\begin{aligned}
\nabla_{\theta_A}L&=0,\qquad
\nabla_{\gamma_A}L=\nabla_{\beta_A}L=0,\\
\nabla_\alpha L&=B^\top\bigl(g_m\odot A(n(h))\bigr),\\
\nabla_{\gamma_M}L&=n(h)\odot g_{u_M},\qquad
\nabla_{\beta_M}L=g_{u_M}.
\end{aligned}
```

这些等式只描述该共享层路径传回的损失梯度；不包括额外参数正则项，也不声称 optimizer 的 weight decay 一定不移动参数。gate 输出行、MoE 调制输出行和已选专家的梯度可以非零，但不保证在任意 batch 上都非零。

若调制输出写作 o=W₀f(c)+b₀，则第一次反向满足下面的分离。输出 projection 可先学习；位于零 projection 之前的 token projection、隐藏 MLP 与 iteration MLP 沿此路径先收到零梯度。之后 gate 或输出 projection 打开，原先关闭的梯度路径才可能出现。

```math
\begin{aligned}
\nabla_{W_0}L&=g_o f(c)^\top,\qquad
\nabla_{b_0}L=g_o,\\
\nabla_cL&=J_f(c)^\top W_0^\top g_o=0
\quad\text{at initialization}.
\end{aligned}
```

共享参数在全部 body 调用中的贡献应求和。若以完整反向作为参考实现，状态敏感度递推为下式，不能仅因参数共享而只保留最后一次调用。作者实际是否 checkpoint 或 detach 仍须源码核验；该递推定义的是完整图的数学目标。

```math
D_k=J_kD_{k-1}+B_k,\qquad
D_k=\frac{\partial H_k}{\partial\theta},\quad
B_k=\left.\frac{\partial F_{\theta,k}(H)}{\partial\theta}\right|_{H=H_{k-1}}.
```

依据：[§3.3 的零初始化说明、Eq.8–9 与 Table 7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)。首步梯度结论是对这些明确假设的推导，不作为已运行作者训练器的证据。

## 4 容量配平与参数预算

重复计算增加了执行次数，却未必按同样比例增加可接触的独立权重。共享 attention 每轮仍使用同一组投影；MoE 每轮则可能选中专家池中的不同 experts。因此，判断 attention 与 FFN 的容量比例时，要先区分“访问过多少不同权重”与“把这些权重执行了多少次”。

设同一物理 MoE 层第 k 轮选择的集合为 S_k，每个 expert 大小为 P_E。下面第一式取各轮集合的并集：重复访问同一个专家只计一次，表示这条轨迹接触的独立 active FFN 容量。第二式逐轮求和：同一个专家每执行一次都计入，表示参数应用量。共享专家每轮都会执行，所以它们在第一式只计一份，在第二式计 K 份；attention 的两式也遵循同一原则。

```math
\begin{aligned}
A_{\rm FFN,unique}
&=A_{\rm shared}+P_E\left|\bigcup_{k=1}^{K}S_k\right|,\\
A_{\rm FFN,applications}
&=K A_{\rm shared}+P_E\sum_{k=1}^{K}|S_k|.
\end{aligned}
```

```math
\begin{aligned}
A_{\rm attn,unique}&=P_A,\\
A_{\rm attn,applications}&=K P_A,\\
\rho&=\frac{A_{\rm attn,unique}}
{\mathbb E[A_{\rm FFN,unique}]}.
\end{aligned}
```

这是为澄清物理容量与调用次数而作的数学展开；固定架构下，集合并集随追加轮次不会缩小，但新增专家数因重合通常小于 top-k。它不是从几张 router 热图即可精确算出的全模型 FLOPs。全模型还须加 prelude/suffix、dense 首层、调制器和具体 attention 的序列长度成本。

这种区别说明了 capacity balancing 想调整什么。随着循环推进，MoE 可能接触新的专家，attention 却始终复用同一组物理权重；两类子层接触的独立容量比例因而可能偏离匹配的 Vanilla MoE。配平方向是提高 MLA 低秩投影的 rank，并缩小各 expert 的 FFN，把 active 容量分配更多地给 attention、较少地给单个专家，使 attention／FFN 的比例靠近基线。

单个专家变小后，再增加 expert pool，把总参数补回目标容量，同时保持 top-k=6。Table 7 应沿这两步来读：rank 与 FFN width 改变子层之间的分配，expert pool 改变驻留容量。它们是联合调整，效果由后文 Loop Base、仅 Balancing 与完整模型的组件对照判断，而非由参数表本身保证。

| 尺度与配置 | Q LoRA rank | KV LoRA rank | Routed experts | Expert FFN hidden |
| --- | ---: | ---: | ---: | ---: |
| 3B Vanilla | 384 | 512 | 64 | 896 |
| 3B Loop Base | 384 | 512 | 128 | 896 |
| 3B LoopMoE | 512 | 640 | 136 | 864 |
| 9B Vanilla | 768 | 512 | 64 | 1280 |
| 9B Loop Base | 768 | 512 | 128 | 1280 |
| 9B LoopMoE | 896 | 640 | 136 | 1216 |

这里的 LoRA rank 是 MLA 的低秩投影结构超参数，并非已经采用参数高效后训练 LoRA adapter。论文声称完成 total params、per-token FLOPs 与 active sublayer ratio 对齐；公开表格提供配置级证据，缺少实现和精确 profiler 结果时不能自行认证逐操作 FLOPs 完全相等。[§3.4、Table 7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=5)

Fig.3 的曲线与文字说明存在一处未解决的矛盾。Appendix A.4 的正文和图注称 attention 物理 active 恒定、FFN 随 K 增长；PDF p.13 中的两条参数曲线却都随 1→2→4 下降，原图核查也排除了文字提取列错位的解释。因此，这里的机制分析依据集合并集的定义与 Table 7，不把该图的纵轴读数作为上述单调性的数值证据；具体作图量和控制条件仍待作者澄清。

### 唯一专家容量的期望必须保留跨轮依赖

同一个 token 在同一物理层第 k 轮选择集合 Sₖ。对 E 个 routed experts，唯一访问数的期望始终满足第一式；只有额外假设各轮选择独立时才能分解联合概率。

```math
\begin{aligned}
\mathbb E\left|\bigcup_{k=1}^{K}S_k\right|
&=\sum_{e=1}^{E}\left[1-
\Pr(e\notin S_1,\ldots,e\notin S_K)\right],\\
&=\sum_{e=1}^{E}\left[1-\prod_{k=1}^{K}(1-q_{k,e})\right]
\quad\text{if rounds are independent}.
\end{aligned}
```

以 E=136、top-6、K=4 为例，任何轨迹都满足 6≤唯一访问数≤24。若仅作独立均匀抽样的思想实验，期望为约 22.458；若四轮完全重复同一集合，则只有 6。论文模型的 hidden 连续递推，不能默认采用前一种独立假设。

```math
\mathbb E U_K=136\left[1-\left(1-\frac6{136}\right)^K\right]
\quad\text{under independent uniform rounds}.
```

对任意固定轨迹，并集随 K 单调不减；其每轮新增数却不必单调递减，例如前两轮相同、第三轮换成不交集合。独立均匀模型的凹增长性质不能直接移植为 learned routing 的定理。

### 增大 expert pool 也有自己的计算成本

若每个 routed expert 的参数数为 P_E，两个 shared experts 合计 P_S，则仅就该物理层的专家权重记账，resident 与 K 轮参数应用数分别为：

```math
\begin{aligned}
P_{\rm FFN,resident}&=P_S+E P_E,\\
P_{\rm FFN,applications}&=K(P_S+6P_E),\\
P_{\rm FFN,unique}&=P_S+P_E U_K.
\end{aligned}
```

这解释了“保持 top-k、增加专家池可主要增加 resident 容量”的依据，但不能把其改写为“专家池免费增大”：router score 计算、参数驻留、dispatch/communication 与负载尾部仍会变化。MLA rank、FFN width 和 pool 的联调应逐一核对真实 tensor numel、实际调用 FLOPs 与 token 级并集统计，而非用 active 参数的一个数替代三者。

把 §3.4 写成工程验收，可定义相对匹配误差，而不假定已经求得某个唯一最优配置：

```math
\epsilon_N=\frac{N-N^\star}{N^\star},\quad
\epsilon_F=\frac{F-F^\star}{F^\star},\quad
\epsilon_\rho=\frac{\rho-\rho^\star}{\rho^\star}.
```

N 必须对共享张量身份去重；F 必须给出 context、token 阶段和操作计数约定；ρ 必须说明是全模型容量之比还是各层比值的平均。缺少实际配置与 trace 时，Table 7 的超参数改动支持设计方向，不能补出上述误差的精确数值。

依据：[§3.4、Appendix A.4、Table 7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=5)。并集恒等式、概率思想实验及验收误差为本页推导；Fig.3 的原图歧义仍保留。

#### 算法 唯一容量与路由动力学的共同记账

**输入** 能返回完整 trace 的待验证实现、参数身份清单、固定样本与有效 token mask、对齐基线。

**输出** resident 参数、参数应用数、唯一访问容量、FLOPs 口径、路由相似度/间隔与更新幅度。

**执行前提** 诊断参考算法；作者工件未核验。只在所需 tensor/trace 可获得时计算相应量，缺失量保持缺失。

![算法 唯一容量与路由动力学的共同记账](visualize:fde1_bGliZmlsZV9QYVVOdlZJWS13d2FxVVNCMG01Z21B_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 容量定义来自 §3.4，观测指标来自 §6.2、Appendix A.7–A.8；参数去重、score margin 与更新幅度是本页补充的验收方法，不是已运行实验。

## 5. 主结果与完整组件消融

Table 1 比较 3B / 200B tokens 下同数据、同预算的 Vanilla 与 LoopMoE；Table 2 和 Table 9 对应 9B / 300B tokens。3B 九任务平均从 34.84→35.89，八项提高，MMLU 则从 27.86→26.81；9B 平均从 41.22→44.03，九项均提高，其中 MMLU 为 36.62→45.33、GSM8K 为 14.53→20.94。这说明较大设置中的差距更大，但两个尺度和单 seed 结果不足以建立 scaling law。

完整组件的贡献还需相对 Loop Base 判断。9B 的 Loop Base 九任务平均为 42.92，已高于 Vanilla 的 41.22；完整模型继续增至 44.03。不过，MATH 从 Loop Base 的 8.32 降到完整模型的 7.12，虽仍高于 Vanilla 的 6.61，却说明整体增益可以伴随单任务损失。[Tables 1、2、9](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=6)

| 3B / 200B tokens，Table 1 | Vanilla MoE | LoopMoE |
| --- | ---: | ---: |
| MMLU | 27.86 | 26.81 |
| HellaSwag | 58.68 | 60.98 |
| PIQA | 74.12 | 75.22 |
| WinoGrande | 49.92 | 51.42 |
| TriviaQA | 36.19 | 37.63 |
| RACE | 35.00 | 35.77 |
| BBH | 26.12 | 27.94 |
| GSM8K | 4.12 | 5.26 |
| MATH | 1.53 | 1.88 |
| 九任务平均 | 34.84 | 35.89 |

Table 3 的组件消融同为 3B / 200B，使用五任务子集；平均值与上面九任务均值不是同一指标。下面完整保留原表：

| 结构 | ARC-E | ARC-C | HellaSwag | BBH | GSM8K | 五任务平均 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla MoE | 75.04 | 45.80 | 58.68 | 26.12 | 4.12 | 41.95 |
| Loop Base | 74.24 | 45.50 | 58.87 | 27.33 | 5.34 | 42.26 |
| Loop Base + Balancing | 75.04 | 44.86 | 61.32 | 27.58 | 5.49 | 42.86 |
| Loop Base + iteration-only AdaLN | 73.41 | 43.77 | 58.74 | 26.21 | 5.23 | 41.47 |
| Loop Base + IterAdaLN | 74.75 | 45.38 | 59.19 | 27.26 | 5.34 | 42.38 |
| LoopMoE = IterAdaLN + Balancing | 75.21 | 45.72 | 60.98 | 27.94 | 5.26 | 43.02 |

这组消融先比较循环本身，再比较调制条件。仅加 loop 已改善 BBH/GSM8K，但部分知识任务退步；只用 iteration 的统一 modulation，五项均低于 Loop Base；再加入 token 状态形成 IterAdaLN 后，五项又均优于 iteration-only。在这组设置中，显式迭代条件的效果取决于如何与当前状态结合。

Balancing 与 IterAdaLN 组合获得最高平均值，却并非所有单项的最佳配置：GSM8K 最好的是仅 Balancing 的 5.49，完整模型为 5.26；Vanilla 的 ARC-C 45.80 也高于完整模型 45.72。因此，这些结果支持给定对照中的收益，不能概括为每个组件对每项任务都全面有益。[§5.3、Table 3，p.7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=7)

BBH 子任务进一步显示收益异质性（Table 4 / Table 10，3B / 200B）：hyperbaton 为 Vanilla 31.25、Loop Base 50.83、LoopMoE 51.25；navigate 为 40.83、52.50、57.08。反例是 salient translation error detection 22.50→16.67→15.00、logical deduction five objects 22.50→17.50→16.25。作者把增益联系到可迭代组合推理，把下降联系到每次调用独立容量变窄；这是对分任务现象的解释，未通过特定容量/步骤干预单独识别因果机制。[§6.1、Tables 4、10](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=8)

评测采用 lm-evaluation-harness：MMLU 5-shot / HellaSwag 10-shot / PIQA、WinoGrande、RACE 0-shot accuracy；TriviaQA 5-shot、BBH 3-shot、GSM8K 5-shot、MATH 4-shot exact match；ARC-E / ARC-C 为 25-shot accuracy。此处 MATH 不是 Done Right 的 MATH500，GSM8K shots 也与 Done Right 的 8-shot 不同，不能把两篇裸分拼成统一排行榜。Table 1 外部 Ouro/Qwen 等更长 token 预算只是背景，不是严格配平架构对照。[Table 8，p.13](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=13)

### 用四个消融角点区分主效应与交互

令 Y₀₀ 为 Loop Base，Y₁₀ 为加 balancing，Y₀₁ 为加 IterAdaLN，Y₁₁ 为完整模型；先在同一任务和同一报告预算下定义差分，再汇总。iteration-only AdaLN 是另一个结构，不占用这四个角点。

```math
\begin{aligned}
\Delta_B^{(I=0)}&=Y_{10}-Y_{00},\quad
\Delta_B^{(I=1)}=Y_{11}-Y_{01},\\
\Delta_I^{(B=0)}&=Y_{01}-Y_{00},\quad
\Delta_I^{(B=1)}=Y_{11}-Y_{10},\\
\Gamma&=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\end{aligned}
```

用 Table 3 已四舍五入的五任务均值，balancing 的两种条件增益分别为 +0.60 与 +0.64 个百分点，IterAdaLN 为 +0.12 与 +0.16，交互约 +0.04。按五个任务原表数值先算交互再平均得到 +0.036，差别来自均值显示精度，不能把这点差距解释为新的实验信号。

各任务交互方向并不相同：ARC-C 为 +0.98，HellaSwag 为 −0.66，GSM8K 为 −0.23 个百分点。因此“两个组件都有效”的精确含义是给定消融设置下若干聚合对照为正，不是每个任务上独立正增益，更不是所有规模上的超加性定理。

同理，BBH 的总均值可能由少数子任务带动；保留 hyperbaton、navigate 的正差与 salient translation、logical deduction 的反例，才能区分“任务分布中的平均改善”与“所有推理模式都改善”。这些是单次训练产物的描述性对照，没有跨 seed 标准误，不能从 +0.04 推导统计显著的协同作用。

来源：[§5.3、Table 3、Tables 4 与 10](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=7)。上述效应与交互是对原表的直接算术分解。

### 九十亿参数结果与跨规模比较的识别边界

Table 9 的完整九任务结果如下。它同时保留 Vanilla、Loop Base 与完整 LoopMoE，因而可区分“引入循环后的变化”与“完整组件相对 Loop Base 的变化”；不能仅用两个最终模型的均值代替这两个比较。[Appendix A.5、Table 9，p.14](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=14)

| 9B 与 300B tokens | Vanilla MoE | Loop Base | LoopMoE |
| --- | ---: | ---: | ---: |
| MMLU | 36.62 | 45.28 | 45.33 |
| HellaSwag | 68.79 | 69.77 | 70.24 |
| PIQA | 76.95 | 76.92 | 77.51 |
| WinoGrande | 50.09 | 50.32 | 51.91 |
| TriviaQA | 50.03 | 50.81 | 52.90 |
| RACE | 35.35 | 35.64 | 37.40 |
| BBH | 32.01 | 32.17 | 32.89 |
| GSM8K | 14.53 | 17.07 | 20.94 |
| MATH | 6.61 | 8.32 | 7.12 |
| 原表九任务平均 | 41.22 | 42.92 | 44.03 |

在这个尺度，Loop Base 的 MMLU 已比 Vanilla 高 8.66 个百分点，完整组件再增加 0.05；GSM8K 的两段增量则为 2.54 与 3.87。MATH 的对应增量为 +1.71 与 −1.20。完整 LoopMoE 相对 Vanilla 九项均高，仍不等于“增加 IterAdaLN 与 balancing 后九项均高于 Loop Base”。这些是原表的条件差分，不把复合组件增量单独归因给其中一个组件。

设 Δ(N,T) 是给定规模 N、训练 token 预算 T 及该规模配方下，LoopMoE 相对 Vanilla 的九任务平均差。原文两个已观察点给出 Δ(3B,200B)=1.05、Δ(9B,300B)=2.81；扩大设置后的差距增加 1.76 个百分点。要把这 1.76 单独解释为固定 token 预算下的规模效应，还缺少交叉设置，例如 Δ(3B,300B)：

```math
\begin{aligned}
&\Delta(9\mathrm B,300\mathrm B)-\Delta(3\mathrm B,200\mathrm B)\\
&=\bigl[\Delta(9\mathrm B,300\mathrm B)-\Delta(3\mathrm B,300\mathrm B)\bigr]\\
&\quad+\bigl[\Delta(3\mathrm B,300\mathrm B)-\Delta(3\mathrm B,200\mathrm B)\bigr].
\end{aligned}
```

右侧两个差分尚不能由原表各自求值；两尺度的峰值学习率及几何配置也不同。因而已支持的结果是“在两组各自配平的规模与训练预算设置中，较大设置的优势更大”，而非已识别纯参数规模效应或确定幂律。每个设置内仍保留论文的同 token 预算控制；这个跨设置限制不否定其各自的对照。[§4.1–4.2、§5.2、Tables 2、7、9](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=5)

### BBH 全部子任务与聚合权重

Table 10 的 27 个子任务如下，数值沿用原表两位小数。末行保留论文报告的 Average；它的聚合权重未在 Table 8 或 A.6 中列出，不能自行把它指定为 27 项的简单平均。[Table 10，p.14](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=14)

| BBH 子任务 | Vanilla MoE | Loop Base | LoopMoE |
| --- | ---: | ---: | ---: |
| boolean_expressions | 49.58 | 52.08 | 54.58 |
| causal_judgement | 47.16 | 48.86 | 50.57 |
| date_understanding | 17.92 | 15.00 | 12.08 |
| disambiguation_qa | 35.00 | 31.25 | 40.00 |
| dyck_languages | 0.00 | 0.00 | 0.83 |
| formal_fallacies | 41.25 | 53.33 | 48.33 |
| geometric_shapes | 20.00 | 19.17 | 27.08 |
| hyperbaton | 31.25 | 50.83 | 51.25 |
| logical_deduction_five_objects | 22.50 | 17.50 | 16.25 |
| logical_deduction_seven_objects | 14.17 | 13.33 | 13.33 |
| logical_deduction_three_objects | 34.17 | 26.25 | 32.50 |
| movie_recommendation | 24.17 | 28.75 | 19.58 |
| multistep_arithmetic_two | 0.42 | 0.00 | 0.83 |
| navigate | 40.83 | 52.50 | 57.08 |
| object_counting | 17.08 | 28.33 | 22.92 |
| penguins_in_a_table | 19.44 | 18.75 | 18.06 |
| reasoning_about_colored_objects | 12.08 | 15.83 | 9.58 |
| ruin_names | 22.50 | 23.33 | 17.08 |
| salient_translation_error_detection | 22.50 | 16.67 | 15.00 |
| snarks | 44.89 | 55.68 | 53.98 |
| sports_understanding | 51.25 | 50.00 | 53.33 |
| temporal_sequences | 25.00 | 26.67 | 25.42 |
| tracking_shuffled_objects_five_objects | 21.67 | 15.00 | 20.83 |
| tracking_shuffled_objects_seven_objects | 15.00 | 1.25 | 13.33 |
| tracking_shuffled_objects_three_objects | 32.50 | 32.92 | 33.75 |
| web_of_lies | 49.58 | 54.17 | 54.17 |
| word_sorting | 1.25 | 0.42 | 1.67 |
| 原表 Average | 26.12 | 27.33 | 27.94 |

若用 a_j 表示任务 j 的百分制分数，q_j 为汇总权重，则下面三个对象必须分别命名。按样本加权的表达只在各任务有效样本数 n_j 已确定、且同模型间使用相同评测样本时成立。

```math
\begin{aligned}
S_q&=\sum_{j=1}^{27}q_j a_j,
\quad q_j\ge0,\quad \sum_jq_j=1,\\
S_{\rm task}&=\frac1{27}\sum_{j=1}^{27}a_j,\\
S_{\rm sample}&=\frac{\sum_j n_j a_j}{\sum_jn_j}.
\end{aligned}
```

从这 27 行逐列计算的 S_task 为 26.4133、27.6989、28.2744，与论文 Average 的差依次为约 0.2933、0.3689、0.3344 个百分点。每个子任务四舍五入到两位小数只会使其等权平均偏移至多 0.005；即使再计入 Average 自己的舍入误差，仍不能解释这些差距。不同样本权重或另一汇总契约可能产生不同值，但缺少 n_j、task-version 与 aggregation 配置，现有证据不能选定其中一种，也不能据此宣布模型结果错误。

在明确采用上述 S_task 的描述性重算下，LoopMoE 相对 Vanilla 为 16 项上升、11 项下降；总差分之和为 50.25，等权平均增量约 1.8611。hyperbaton 与 navigate 的差分之和为 36.25，约占这个未加权净增量的 72.14%；移除这两项后，剩余 25 项的等权平均增量仍为 +0.56。这个分解说明聚合提升的集中程度；它既不是原文 27.94−26.12 的精确分解，也没有重新估计统计显著性。

评测实现的最小核验记录应包含每个 task 的 version、有效样本数、correct 数、单项 metric、group aggregation 权重与被剔除样本。先由 correct/有效样本数产生原始单项分数，再按配置聚合，最后统一舍入。lm-evaluation-harness 的名称和 3-shot 设置不足以恢复这一契约；在作者工件可得之前，论文 Average 与本节显式定义的重算量并列保留。[A.2、A.6、Tables 8、10](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=13)

## 6. Routing dynamics 与 IterAdaLN 随轮变化

接下来考察共享 body 每轮的状态和专家选择怎样变化。原文分别测 router 输入的 cosine similarity 和所选 expert 集合的 Jaccard similarity。前者比较连续向量的方向，后者比较真正入选的离散集合。

这两个量可能给出不同信息。两个 hidden 方向接近时，只要 router 中靠近 top-k 截止位置的分数发生小幅交叉，入选集合仍会改变。所以下面的定义需要同时保留：既看状态表示多相似，也看哪些专家实际继续参与。

```math
\begin{aligned}
\operatorname{cos}(u,v)
&=\frac{u^\top v}{\|u\|\|v\|},\\
J(S_r,S_s)&=\frac{|S_r\cap S_s|}{|S_r\cup S_s|}.
\end{aligned}
```

在 BBH、3B 四轮模型的 Fig.4 中（轮次编号 0–3），第一个共享层的相邻 cosine 为 0.651、0.808、0.816，对应 Jaccard 为 0.482、0.551、0.409；第二个共享层的相邻 cosine 为 0.744、0.848、0.787，Jaccard 为 0.542、0.508、0.271。重点是后一个层的最后一对：表示方向仍较相似，专家集合重合却比中间一对更低。表示方向保持较高相似度，并不意味着下一轮还会交给相同专家处理。

作者把不同阶段概括为“入口适配—中间递推—出口重编码”。热图支持这些阶段有不同统计特征；要进一步说出口重编码改善了预测，就需要像 Done Right 冻结路由那样，在明确干预后重新比较任务结果。这里尚未做这种因果验证，而有限轮次的相似度也不等于固定点收敛。[§6.2、Fig.2、Appendix A.7、Fig.4，pp.8、14–15](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=15)

Appendix A.8 / Table 11 则直接记录调制参数在 BBH 上的统计量：

| 分支 | γ 均值：第 0→3 轮 | attention α 均值：第 0→3 轮 | α 的 token 标准差：第 0→3 轮 |
| --- | --- | --- | --- |
| 第 1 个共享层 attention | −0.159→0.503 | 0.707→1.355 | 0.181→0.486 |
| 第 2 个共享层 attention | 0.034→0.500 | 0.601→1.613 | 0.147→0.922 |
| 第 1 个共享层 FFN | −0.339→0.008 | 无可学习输出 α | 不适用 |
| 第 2 个共享层 FFN | −0.252→−0.075 | 无可学习输出 α | 不适用 |

Table 11 提供了另一条线索：第 2 层 attention α 的 token range 从 0.761 扩到 3.432，shift β 的均值/绝对均值则很小。晚轮 attention gate 更大，而且不同 token 之间的差异更强；作者据此观察到调制强度向后移。

门值描述的是分支输出的乘数，实际写入 residual 的量还取决于分支输出本身。因此，平均 gate 与逐 token 更新范数不是同一统计量，平均 γ 也不是函数 Lipschitz 常数。若要据此设计早退，应接着测量各轮实际更新及经同一 suffix/head 后的预测变化，判断哪些计算可以省去。现有门值统计尚未给出每个 token 的收敛性，也未证明早轮可删除或增大 α 就能提高准确率。[Table 11，p.16](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=16)

原文叙述有两处需要按表限定：p.15 称早轮各分支均抑制、FFN 从不越过零，但 Table 11 的第 2 层 attention 初始 γ=+0.034，第 1 层 FFN 末轮 γ=+0.008；p.16 才把后者称为 marginal exception。因此本页保留具体数据，而不重复“一律抑制”的概括。α 后移也不是现成的提前停止策略：直接提前退出会丢掉晚轮 attention 作用；作者自己明确指出这一现象没有直接给出 prefix-truncation 的推理加速。

### 表示相似并不保证 top-k 身份稳定

令 router 的完整 score 向量为 r(u)，第 k 大与第 k+1 大分数之间的严格间隔为 m。若两次 score 的无穷范数差小于 m/2，则所有入选专家仍高于所有未入选专家，top-k 集合不变。这是一个充分条件，不要求指定 score 使用 sigmoid 还是 softmax。

```math
m=r_{(k)}(u)-r_{(k+1)}(u)>0,\qquad
\|r(v)-r(u)\|_\infty<\frac m2
\ \Longrightarrow\ S(v)=S(u).
```

相反，cosine 接近 1 不能单独验证这个条件：还缺 hidden 范数、router 对扰动的敏感度以及 top-k margin。

```math
\|u-v\|^2=(\|u\|-\|v\|)^2
+2\|u\|\|v\|\bigl(1-\cos(u,v)\bigr).
```

对每次 top-6 决策，若两集合交集大小为 m，则 Jaccard=m/(12−m)。这个非线性关系应逐 token 使用；不能把 Fig.4 中已经平均的 Jaccard 直接反解成平均重合专家数。

### 平均 Jaccard 不能反解平均重合数，但可以给出严格界

固定 top-6 的两个集合，其交集大小 M 只能取 0,…,6，逐 token Jaccard 为 J=M/(12−M)。若热图每格确实是对这些逐 token 决策按同一组非负权重求均值，下面的界适用；如果实现先合并所有 token 的专家集合再算 Jaccard，则不适用。作者未提供 probe 脚本，归约顺序必须列入后续核验。

```math
M=g(J)=\frac{12J}{1+J},\qquad
g''(J)=-\frac{24}{(1+J)^3}<0,
\qquad
6\,\mathbb E[J]\le\mathbb E[M]
\le\frac{12\mathbb E[J]}{1+\mathbb E[J]}.
```

右界来自凹函数的 Jensen 不等式；左界来自 0≤J≤1 上 g(J)≥6J。利用 M 为整数还能收紧右界：记 j_m=m/(12−m)，当 j_m≤J̄≤j_{m+1}，把相邻两点 (j_m,m)、(j_{m+1},m+1) 线性连接即可。上下界均可由相应两点分布达到，因此在只知道该均值与 top-6 时不能进一步唯一恢复平均重合数。

```math
\mathbb E[M]\le
m+\frac{\overline J-j_m}{j_{m+1}-j_m},
\qquad j_m\le\overline J\le j_{m+1}.
```

把 Fig.4 第二个共享层的三个相邻轮次均值代入，得到以下条件范围。下界按显示均值计算，上界向上取三位小数；原图均值自身的舍入误差另在下文处理。

| 第二个共享层的轮次对 | 图中平均 Jaccard | 平均重合专家数的可行范围 |
| --- | ---: | ---: |
| 0→1 | 0.542 | 3.252 至 4.196 |
| 1→2 | 0.508 | 3.048 至 4.038 |
| 2→3 | 0.271 | 1.626 至 2.533 |

在上述归约前提下，最后一对的上界已低于中间一对的下界；即便给每个热图值保留 ±0.0005 的舍入误差，这个分离仍成立。于是可以严格支持“平均集合重合下降”，却仍不能推出确切重合数、每个 token 都换专家，或这些切换对准确率有正的因果作用。它把原文出口路由变化的观测推进为有条件、可核查的数量结论。[Fig.4、Appendix A.7](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=15)

### 门值的平均不是残差更新的大小

对每个 token 实际记录 attention 更新与 MoE 更新，再计算相对幅度，才能检验“更新主要发生在晚轮”。下式中的 Bα 继续表示实际 gate 广播。

```math
\begin{aligned}
\Delta^A_{k,t}&=(B\alpha_{k,t})\odot A(u^A_k)_t,\\
\Delta^M_{k,t}&=M(u^M_k)_t,\\
a^A_{k,t}&=\frac{\|\Delta^A_{k,t}\|}{\|h_{k,t}\|+\varepsilon},\qquad
a^M_{k,t}=\frac{\|\Delta^M_{k,t}\|}{\|m_{k,t}\|+\varepsilon}.
\end{aligned}
```

即使 α 的均值增加，分支输出变小或方向改变仍可使实际残差更新减小；逐通道时还取决于 gate 与输出各坐标的相关性。Table 11 的 α/γ 统计支持调制策略随轮变化，但没有直接给出上述更新范数，更没有提供提前删除末轮后的准确率。核对三阶段解释时，应把 cosine、Jaccard、score margin、更新幅度和经过同一 suffix/head 的预测变化放在同一 token trace 上。

原观测：[§6.2、Fig.4、Table 11](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=15)。margin 条件、范数恒等式与建议 trace 指标是数学推导和诊断设计；不冒充作者已经完成的因果消融。

## 7 推理成本与后训练接口

相同 FLOPs 不保证相同 wall time。Table 5 报告训练吞吐 TFLOP/s/GPU：Vanilla 66.0、Loop Base 86.2、LoopMoE 81.8。完整 LoopMoE 比 Vanilla 的报告吞吐高约 24%，但比简单 Loop Base 低。Table 6 的 batch=1 自回归 decoding 结果如下；数据按论文的相同软硬件设置比较，文中没有提供足够实现材料供本次复测。[§6.3、Tables 5–6，p.9](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=9)

| 模型，ms/token | Context 256 | Context 1024 | Context 2048 |
| --- | ---: | ---: | ---: |
| Vanilla MoE | 66.495 | 65.759 | 64.919 |
| Loop Base | 64.321 | 64.173 | 64.012 |
| + Balancing | 63.809 | 64.026 | 64.223 |
| + IterAdaLN | 74.308 | 73.615 | 72.278 |
| LoopMoE | 72.461 | 72.292 | 71.042 |

所以 IterAdaLN 的质量收益有可见的解码开销；不能从更高训练 throughput 宣布最终生成更快。MLA 可以压缩 attention 状态，但本论文未交代可核验的 KV cache class、逻辑轮次索引、跨轮复用、prefill/decode 分离或 cache 更新顺序。相同权重在不同轮次接收不同 hidden，产生的 K/V 或 MLA latent cache 一般不同。依计算图设计增量执行时，至少要标明物理层、逻辑轮次、token position 与 mask；这是实现要求，不是作者已发布缓存策略的陈述。

这些质量与成本结果引出三类后训练问题。首先，可保持固定 K=4 的预训练计算图，进行 SFT / 偏好优化，并测量它是否破坏已学到的迭代条件和负载分配。

若希望降低执行深度，则需为可停止的中间态训练可读出性和停止校准，同时保留其接入 suffix 的分布。“省略早轮”与“删掉晚轮”是不同干预，α 小本身不足以说明该轮可删除。

另一方向是针对 IterAdaLN 的调制开销进行融合或蒸馏，在同质量、同序列长度下比较延迟。若加入 LoRA/adapter，还需区分 MLA 原有低秩投影与新增 adapter，并明确 adapter 是否跨轮共享。

这些方向是依据模型结构和观测提出的待研究接口。论文没有完成动态 halting、SFT/RL、非原生模型 retrofit 或高效 KV serving 的公开端到端方案。将其移植到普通 Transformer 还需处理参数合并/重复调用后的分布变化，不能把已预训练的 LoopMoE 性能当成直接支持任意 retrofit 的证据。

### 吞吐增益与解码时间必须使用各自分母

Table 5 的训练吞吐增幅为 81.8/66.0−1≈23.94%；Table 6 在 context=1024 的完整模型解码时间增幅为 72.292/65.759−1≈9.93%。一个是报告 FLOPs 每秒的比值，一个是毫秒每 token 的比值，方向不同并不矛盾。若相同 FLOPs 口径成立，时间比是吞吐比的倒数；不能把“吞吐提高 24%”写成“时间减少 24%”。

### IterAdaLN 的张量依赖决定哪些计算能预计算或融合

公式中的 token condition 不是一个可无条件折叠进静态权重的常数。令 batch、序列和 hidden 维度为 B、T、d，条件宽度为 c，attention gate 的原始输出宽度为 a。γ、β 经实际广播规则作用于 hidden 通道；以下把广播后的张量记作 [B,T,d]，并不宣称调制网络一定直接输出 d 个独立通道。论文未公开 c、MLP 中间宽度与所有调制输出的完整尺寸，因此不擅自取 a=d 或 a=1。

| 阶段 | 张量与依赖 | 可确定的执行顺序 |
| --- | --- | --- |
| iteration 编码 | PE(k) → v_k，形状 [c] | 同一已固定条件模块内按 token 广播；参数更新后必须重算 |
| attention 条件 | h:[B,T,d] → Linear_A(h)+v_k:[B,T,c] | 在 attention 之前，以当前 h 为输入 |
| attention modulation | 广播后的 γ_A、β_A:[B,T,d]，原始 α:[B,T,a] | RMSNorm 与 affine 调制产生 attention 输入；α 的广播按实际配置 |
| attention 更新 | m=h+α⊙A(u_A)，m:[B,T,d] | 必须等待 attention 分支输出及 gate |
| MoE 条件与执行 | Linear_M(m)+v_k → γ_M、β_M → u_M → router/expert | 条件依赖更新后的 m；不能复用基于旧 h 的条件 |

对某个直接读取 IterAdaLN 输出的线性投影 W，精确代数式如下。若 γ、β 仅依赖 k，则在固定模型的推理阶段，可以为每个允许 k 预先形成 W_k 与 b_k；原线性层即使无 bias，这次折叠也一般会产生 b_k。

```math
\begin{aligned}
y&=W\bigl[\operatorname{Diag}(1+\gamma_{k,t})n(h_t)+\beta_{k,t}\bigr],\\
W_k&=W\operatorname{Diag}(1+\gamma_k),\qquad
b_k=W\beta_k
\quad\text{if modulation depends only on }k.
\end{aligned}
```

LoopMoE 的 γ、β 还依赖 h，所以一般不存在对所有 token 通用的固定 W_k、b_k。一个明确反例取 d=1、ε=0、h>0、W=1、γ(h)=h、β(h)=0：因为 n(h)=1，动态输出为 1+h；任何只作用于 n(h) 的固定 affine 投影却恒定。固定 ε>0 时同样不能全局等价：h→∞ 时 n(h) 有界，而 (1+h)n(h) 无界。特殊参数可以偶然等价，但不能仅凭“是 affine modulation”就宣布一般可静态合并。这个反例说明表达能力与运行依赖，不是假定作者的实际调制 MLP 等于该标量函数。

因此可审查的推理优化有明确边界：固定权重且确定性 inference 时，可缓存每个实际 iteration 条件模块的 v_k；可把 RMSNorm、乘 (1+γ) 与加 β 融合到同一数值算子以减少中间读写。但仍需生成每个 token 的 γ、β；attention 与 MoE 的两次条件计算存在 m 的先后依赖。提前用 h 生成 MoE 条件、跨轮缓存 token modulation，或把四轮都替换为同一个静态 affine，都会改变模型函数。训练时跨 optimizer update 复用 v_k 还会留下过期参数值，或者切断 iteration MLP 的梯度。

一次 dense token projection 至少包含形状 [BT,d]×[d,c] 的矩阵乘法；在乘加计作 2 FLOPs 的约定下为约 2BTdc，另外还要付调制 MLP 和逐元素操作。仅 v_k 与 token 无关，不能把整个 IterAdaLN 开销按每轮一个向量来估计。论文的解码延迟表说明完整模块有实际开销，但未提供这些融合方案的速度数据。[Eqs.3–9、§3.3](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)；[Table 6](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=9)

上述张量契约和等价变换条件均由原公式导出；表中的阶段表示计算依赖，不对应已核验的官方函数名。实现融合时，应先以原始执行顺序为参照，比对输出与需要保留的梯度，再测量实际 latency。

### 早退必须支付观察它所需的计算

考虑一个 token/前缀状态上的深度政策，τ 为实际执行轮数，qᵣ=Pr(τ≥r)，C_P、C_B、C_R 分别为 prelude、一次 body 和 suffix/head 的费用。若 gate 只看 hidden 且最后读出一次，期望费用为第一式；若 gate 必须在每轮经过 suffix/head 才能判断，且最后一次读出复用，则是第二式。C_G,r 包含 gate 本身；若固定硬上限处不再检查或采样停止，则令 C_G,K=0。

```math
\begin{aligned}
\mathbb E C_{\rm hidden\ gate}
&=C_P+C_R+\sum_{r=1}^{K}q_r(C_B+C_{G,r}),\\
\mathbb E C_{\rm logit\ gate}
&=C_P+\sum_{r=1}^{K}q_r(C_B+C_R+C_{G,r}).
\end{aligned}
```

这两式是研究成本模型，实际时间还取决于 batch 中不同退出深度能否压缩、MoE dispatch、kernel launch 与缓存。训练停止策略时应测量真实调用或 wall time；仅记录一个较小的预测退出编号不保证节省计算。

精确增量缓存还需把物理层、iteration 与 token 位置一起作为逻辑地址。即便 MLA 投影权重相同，不同 k 的 hidden 与 IterAdaLN 条件也可不同；只有 K/V、位置处理、mask 与所需依赖都等价时才能复用。对 K=4 模型删轮后，须重新核验缓存路径与完整前缀参考路径，不能沿用原四轮缓存并把新 logits 当作同一政策。

这些条件把后训练任务分成两个具体目标：先让允许的中间态经过既定 suffix/head 可读，再学习具有实际计算收益的停止政策。原论文给出的晚轮强门值不能代替这两项训练与验收。

原始效率数据：[§6.3、Tables 5–6](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=9)；结构依据：[Eqs.3–9](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)。成本公式与停止接口为本页推导。

## 8 公开资源与实现约定

截至2026 年 11 月 8 日，对 ACL 正式页、arXiv v2 与第一作者 Wenkai Chen 个人主页的检查，尚未定位到可由这些作者入口确认归属的官方 GitHub 实现或正式模型权重。作者主页的 LoopMoE 条目只链接论文。因此，文件路径、类名、commit、训练启动命令和权重加载方式仍待官方工件补证。[作者主页](https://wk-chen.github.io/)；[ACL](https://aclanthology.org/2026.emnlp-main.2160/)；[arXiv](https://arxiv.org/abs/2606.04438v2)

检索中出现的 SR-A-W/how-to-loop-moe 是 Foil 项目，其 README 明确写明内部 LoopMoE 命名与 Chen 等人的 arXiv:2606.04438 无关；ShourenWSR 的 loopmoe-U4_100B 也来自另一条实现谱系。两者均不作为本文实现证据或可选官方权重。[排除同名仓库的原始说明](https://github.com/SR-A-W/how-to-loop-moe)

| 待落实项 | 已核实的信息 | 尚不能声称完成的部分 |
| --- | --- | --- |
| 模型主图 | 2/2/2 与 3/3/3，K=4；MLA + MoE；移除反复注入 | 精确 tensor config、模块对象共享范围、权重名 |
| IterAdaLN | Eq.3–9；branch-specific token condition；sinusoidal iteration；输出 projection zero-init | MLP 宽度/激活、α 形状、kernel 与数值精度 |
| MoE router | routed/shared expert 数量、top-6；Appendix A.3 提及 balancing 与 expert bias | score function、归一化、loss 系数、bias-update rate、dispatch kernel |
| 预训练 | Dolma3Mix、tokens、optimizer、LR 与 warmup | batch/context、完整数据 revision/packing、初始化、梯度实现及精确恢复 |
| 容量配平 | Table 7 的 ranks / FFN width / expert pool | 精确参数计数与 FLOPs profiler；Fig.3 作图量矛盾 |
| Routing probes | cosine/Jaccard 热图与 Table 11 的调制统计 | 作者 probe 脚本、样本处理细节、可运行 checkpoint |
| 推理 | batch=1 三个 context 的延迟表 | KV/latent cache 生命周期、硬件型号与完整执行链的可复核配置 |
| 评测 | Tables 1–4、8–11、shots 与 metrics | harness commit、prompt/hash、完整结果日志和多 seed 不确定性 |

LoopMoE的论文算法和实验支持架构学习，但官方可运行工件尚未核验，当前不能作为已经可直接后训练的现成权重。基座矩阵因此保留“原生预训练MoE／sandwich family／官方工件未核验”的状态；这一有限范围的资料结论不排除作者后续发布代码。

### 可复核实现需要显式给出尚未确定的算子契约

下面的前向算法只实现论文已经确定的顺序：同一物理 body 层在四轮复用参数，attention 条件读旧 hidden，MoE 条件读更新后的 m，post RMSNorm 位于各分支输出与 residual addition 之间。具体 router score、top-6 权重归一化、α 广播和模块尺寸作为输入契约；没有该配置时不能静默借用另一篇论文的默认值。

一个可运行工件应同时提供以下四类内容。它们是验收对象，不是已存在的作者文件名。

| 契约 | 需要验证的对象 |
| --- | --- |
| 参数与结构 | 每个独立 tensor 的形状与身份；四轮对应同一 body 参数；MLA 与调制器完整配置 |
| 前向与状态 | α 广播、affine-free pre RMSNorm、分支 post RMSNorm、router score 与权重、mask 与位置 |
| 训练与恢复 | labels shift 与 mask、完整 auxiliary 目标、梯度截断或重算策略、optimizer/data/RNG 状态 |
| 效率与评测 | 同一 FLOPs 口径、真实 cache/dispatch、固定输入输出预算、原始任务分数与 seed |

容量和路由诊断算法要求在同一次实际 trace 上记录唯一权重身份、每次调用、专家集合和调制更新。这样得到的“唯一参数”“参数应用数”“FLOPs”“运行时间”可以分别核对；任何一个量缺失时保持缺失，不能由另一个量推测补齐。

在官方源码与权重尚未核验的当前状态下，这两份算法是论文语义参考与未来接入契约。它们使设计可以审查和移植，但不构成对作者训练运行或基准分数的复现声明。

依据：[ACL 正式论文与作者归属](https://aclanthology.org/2026.emnlp-main.2160/)；[Eqs.3–9、Table 7 与评测附录](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=4)。

#### 算法 LoopMoE 论文语义的完整前向

**输入** tokens、因果与文档 mask、对齐目标；P、共享 body、C；完整 MLA/router/modulation 配置；K=4。

**输出** 末轮 logits、masked next-token loss、每层每轮的路由与调制 trace。

**执行前提** 论文语义参考。score、权重归一化和 α 广播由实现契约显式给出；缺项则报告缺项，不能从别篇借用。

![算法 LoopMoE 论文语义的完整前向](visualize:fde1_bGliZmlsZV9oOC1jZWdWQWwzbjlQa3ZhOUNsTGN3_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** LoopMoE Eqs.3–9、§3.1–3.3。此算法不虚构作者函数名、kernel、缓存类、辅助系数或实际反向策略；n 为 affine-free RMSNorm。

## 动态深度下的逐轮负载与汇总负载

若在后训练中引入 token-dependent depth，第 r 轮就只处理尚未退出的活动集合 A_r。这时，仅看跨轮汇总可能掩盖某一轮的集中负载，因此需同时记录逐轮条件负载与汇总负载。下面以归一化 router 概率和每 token top-k 定义诊断量；这些定义并不补全 LoopMoE 尚未公开的真实辅助目标。

```math
\begin{aligned}
f_{e,r}&=\frac{1}{k|\mathcal A_r|}\sum_{t\in\mathcal A_r}\mathbf1\{e\in S_{t,r}\},\\
\bar p_{e,r}&=\frac{1}{|\mathcal A_r|}\sum_{t\in\mathcal A_r}p_{e,t,r},\\
w_r&=\frac{|\mathcal A_r|}{\sum_j|\mathcal A_j|},\qquad
\bar f=\sum_rw_rf_r,\quad\bar p=\sum_rw_r\bar p_r.
\end{aligned}
```

空活动轮次不进入归约。令逐轮代理量为 E⟨f_r,p̄_r⟩，其加权平均与先汇总再计算之差为：

```math
\sum_rw_r E\langle f_r,\bar p_r\rangle-E\langle\bar f,\bar p\rangle
=E\sum_rw_r\langle f_r-\bar f,\bar p_r-\bar p\rangle.
```

展开内积并使用 Σw_r=1 即得该式。两专家两轮的极限例：第一轮全部选专家 1，第二轮全部选专家 2，条件概率集中在对应专家，等权逐轮代理为 2，汇总代理为 1；有限 softmax 可任意接近此例。总体均匀不能排除单轮拥塞。

门控与 MoE 联合研究因此记录活动 token 数、每轮专家负载、容量溢出及 dispatch 时间。某 token 的边际耗时受同批其他 token 和专家分片影响；单 token CE 改善标签不会自动优化整批 wall-clock。参数容量、调用量和真实延迟继续分别记账。

来源：[LoopMoE 路由与效率分析](https://aclanthology.org/2026.emnlp-main.2160.pdf#page=7)；[TaH2 活动深度](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811)。恒等式属于显式诊断定义的推导。

# 逐项覆盖核对

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 身份与拓扑 | §3.1–3.2、Fig.1、Table7，pp.3、13 | 无已核验作者仓库；按论文2/2/2和3/3/3、K4重建 | 论文已核对，sandwich原生MoE，不是Ouro整栈或Loopie layerwise |
| 取消prelude reinjection | Eq1–2、§3.2，p.3 | 算法H0=Hpre，只在初始读取；Fbody4轮共享 | 已落实原文计算图；未把explicit iteration conditioning说成状态变化的唯一来源 |
| IterAdaLN完整顺序 | Eqs3–9、Fig1、§3.3，pp.3–4 | token+sinusoidal iteration条件；两分支不同MLP；MoE条件使用attention后m | 公式图像核验；未伪造函数名或MLP维度 |
| 归一化与非对称gate | §3.1、Eq5–9、AppendixA.3，pp.3–4、12 | affine-free pre RMSNorm动态scale；post RMSNorm；attention-only gate | 已落实；零init不推出整个block恒等；gate冗余结论按适用假设限定 |
| MoE router/expert组成 | §4.2、Eq10、Table7，pp.5、12–13 | top6、2 shared、136 routed；score/normalization/bias/loss系数未公开核验 | 论文结构已核对，工程细节缺口明确 |
| 容量配平 | §3.4、AppendixA.4、Table7、Fig3，pp.4–5、12–13 | Q/KV ranks↑、FFN width↓、pool↑；物理并集与参数调用数公式区分 | 表7图像核验；Fig3线图与图注单调性冲突保留，不伪造数值 |
| 训练数据/优化器 | §4、AppendixA.1、Table7，pp.5、12–13 | Dolma3Mix，200B/300B，同order/packing；AdamW beta(.9,.95)，warmup2000，峰值LR | 论文已核对；batch/context/init/精度/完整loss与梯度实现未核验 |
| 参数与active口径 | Table1脚注、Tables2/7/9，pp.6、13–14 | 3B物理active0.6B与FLOPs标记0.8B；9B1.3B与1.9B | 原表脚注核验，不能将FLOPs口径当独立active参数 |
| 主结果3B与9B | Tables1/2/9，pp.6、14 | 无官方checkpoint复跑；完整3B九任务和9B代表性结果保存 | 图像核验原分数；两尺度单seed不等于scaling law |
| 组件消融 | §5.3、Table3，p.7 | Vanilla、LoopBase、Balancing、iteration-only、IterAdaLN、完整模型 | 全表图像核验且完整落地；五任务均值与九任务均值分开 |
| BBH子任务异质性 | §6.1、Tables4/10，pp.8、14 | hyperbaton/navigate gains与salient translation/logical deduction regressions | 数值已核对；作者容量/迭代解释与因果结论区分 |
| 迭代路由动力学 | §6.2、Fig2、AppendixA.7、Fig4，pp.8、14–15 | cosine与Jaccard按层轮分别描述；无可运行probe | Fig4原图具体数值核验；不把相似度当固定点证明 |
| 逐轮调制统计 | AppendixA.8、Table11，pp.15–16 | 两层attention γ/α/tokstd与FFN γ；没有动态halt | Table11原图核验；FFN末轮+0.008、L2 attention初轮+0.034例外保留 |
| 实际效率与推理缓存 | §6.3、Tables5/6，p.9 | 原文training throughput与batch1 latency；KV生命周期未给代码 | 数字核对；完整模型更慢decode与更快training分开 |
| 官方repo与同名排除 | ACL页面、arXiv v2、第一作者主页；同名仓库README | 尚未验证作者official repo或weights；Foil/how-to-loop-moe明确不同作品 | 有限检索结论，未声明不存在也未借用同名代码 |
| 后训练接口 | 模型图与AppendixA.8导出的研究问题 | 固定K后训练、可读出/停止校准、调制蒸馏或融合、router负载监控 | 推导方案，未标注已有作者实现或效果 |
| MoE gate 吸收的成立条件 | Eq.10、§3.1/§4.2、Appendix A.3，pp.3、5、12 | 区分 raw sum、归一化 router、shared experts、channel gate 与 post-RMSNorm；实际 router 未公开核验 | 凸包与归一化次序反例已验证；attention-only 配置保留，必要性不升格为定理 |
| IterAdaLN 的融合边界 | Eqs.3–9、Table 6，pp.4、9 | iteration 向量可按固定条件模块缓存；token modulation 与更新后 m 保留依赖；广播前尺寸待核 | 静态 affine 折叠已作代数检验；动态条件一般不能静态合并，无虚构作者 kernel 或加速数据 |
| Jaccard 均值的可识别范围 | Fig.4、Appendix A.7，pp.14–15 | 逐 token top-6 Jaccard 均值前提下，平均重合数有锐界；probe 的实际归约仍待源码核验 | 104 个均值的两点分布极值已核；末轮重合下降为有条件结论，非切换收益的因果证明 |
| 完整 9B 结果与跨设置归因 | §5.2、Table 9，p.14；Table 7 | 官方实现仍未核验；原表三模型九任务全部保留 | 3B/200B 与 9B/300B 差距增加 1.76；缺少交叉设置，不能只归因参数规模 |
| BBH 全部子任务与汇总契约 | Tables 8、10，pp.13–14 | 需作者 task version、有效样本数与 group aggregation 配置 | 27 行原值逐项核验；原 Average 保留；等权重算不一致且舍入不足以解释，聚合权重待补证 |
