[基座矩阵与七篇论文](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [预训练](https://chatgpt.com/space/page_4f0508753a6081919a82b0c89213fd6c) · [后训练](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

# 模型定位与公开检查点

[英文原文：Loop the Loopies!，2607.16051v3](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)。Loopie是从随机初始化开始训练的原生循环MoE，采用Qwen3-MoE架构模板和tokenizer，并非改造已预训练的Qwen3权重。它按物理层推进：一个完整Attention＋MoE层连续执行两次，再进入下一层。理解这种顺序后，再与Ouro的整栈循环和Huginn拓扑比较。（§2.1、§3，pp.4、13）

建议先读逐层计算图，再看硬件与训练配方，随后读结果，最后核对三套实现及缓存边界。论文的贡献包括：逐层局部权重共享、把激活显存余量转化为 microbatch 效率的硬件实测配方、从头预训练及高质量数据阶段、SPT 与数学／代码 RL。核心问题是固定实际预训练成本下循环能否胜过增加普通层／宽度；论文没有系统验证任意增加推理循环数都有效。（§2、§6，pp.4–13、24–25）

公开有 [6B preview](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview/blob/5fd7647161dec494e37abe08b698cb13434b2c34/README.md) 和 [20B preview](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview/blob/652d78bb761afc4ac83e454fe71a02beb3f974cf/README.md)，模型卡称主要面向数学和科学推理，列出推理指标与 fork 安装入口；没有明确把这两个具体 checkpoint 标为纯 Base、SPT 中间态或某一轮 RL。因此，选择后训练起点时要单独确认具体checkpoint阶段；训练流程中的Base节点，并不能直接确定这两个preview的身份。它们仍需作者明确标注阶段。

# 计算图：权重共享在哪一层发生

Loopie增加深度的办法，是让一个完整层先处理两遍当前状态，再把结果交给下一层。沿第ℓ层看，第一次Attention＋MoE从hℓ,₀得到hℓ,₁；第二次用同一组参数处理更新后的hℓ,₁，得到hℓ,₂。它们承接的是先后两次计算，只有第二次结束后才进入ℓ+1层。

下面把§2.1与实际forward展开。h是batch×token×hidden张量，初态来自embedding；ℓ表示物理层，r∈{1,2}表示该层的访问次数。每次都包括完整注意力、MoE及两条残差更新：

```math
\begin{aligned}
h_{0,2}&=\operatorname{Embedding}(x),\\
h_{\ell,0}&=h_{\ell-1,2},\\
a_{\ell,r}&=h_{\ell,r-1}+\operatorname{Attn}_{\ell}\!\left(\operatorname{RMSNorm}^{A}_{\ell}(h_{\ell,r-1})\right),\\
h_{\ell,r}&=a_{\ell,r}+\operatorname{MoE}_{\ell}\!\left(\operatorname{RMSNorm}^{M}_{\ell}(a_{\ell,r})\right).
\end{aligned}
```

```math
\begin{aligned}
\operatorname{MoE}_{\ell}(z)&=\sum_{e\in\operatorname{Top8}(g_{\ell}(z))}\rho_e(z)\,E_{\ell,e}(z),\\
p(x_{t+1}\mid x_{\le t})&=\operatorname{softmax}\!\left(W_U\operatorname{RMSNorm}(h_{L,2,t})\right).
\end{aligned}
```

两次访问共享本层的Attention、router、experts和RMSNorm，但第二次接收的状态已经改变，因此注意力与路由也重新计算。共享的是变换参数，不是把第一次输出或第一次选中的专家直接重复使用；不同物理层仍各有参数。两次处理对应同一段文本，RoPE位置不随访问次数增加。当前公开forward也没有SMELT式的1/r残差缩放。

| 配置 | 6B-A0.6B | 20B-A2B | 证据 |
|---|---:|---:|---|
| 物理层／执行层 | 18／36 | 27／54 | 附录 A 表4 |
| hidden／expert intermediate | 1536／576 | 2304／832 | 表4、preview config |
| Q heads／KV heads | 48／24 | 72／36 | 表4、preview config |
| 显式 head_dim | 128 | 64 | preview config，不可用 hidden÷heads 猜32 |
| experts／top-k | 128／8 | 128／8 | 表4、preview config |
| top-k 概率重归一化 | true | false | preview config：norm_topk_prob |
| embedding 与 lm_head 共享 | false | true | preview config：tie_word_embeddings |

两份 config 的 top-k 概率定义不同：先求全专家 softmax 后，6B 再除以所选专家的概率和，20B 保留所选原概率；工程移植不能把这个字段忽略。两者均有 QK norm、SwiGLU、RoPE，preview 的 YaRN factor=16、原上下文8192、max_position_embeddings=131072。配置是当前发布权重契约；论文表4没有完整写出 head_dim 和这些推理字段。

源码／配置：[6B config](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview/blob/5fd7647161dec494e37abe08b698cb13434b2c34/config.json)、[20B config](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview/blob/652d78bb761afc4ac83e454fe71a02beb3f974cf/config.json)、[HF 模型实现](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。

## 两次共享访问的导数与路由归一化

把完整Attention＋MoE层记为Fℓ，就能看清逐层循环与整栈循环的差别。Loopie的状态先连续经过F₁两次，再经过F₂两次；整栈循环则先从F₁走到F_L，再从头走一遍。即使每组权重都调用两次，后一次调用接收到的状态也不同。按§2.1的计算图，两种顺序分别为：

```math
\begin{aligned}
H_{\rm layer}&=F_L^{\circ2}\circ\cdots\circ F_1^{\circ2}(E),\\
H_{\rm stack}&=(F_L\circ\cdots\circ F_1)^{\circ2}(E).
\end{aligned}
```

两层线性情形把这种顺序差异直接暴露出来：逐层复用给出B²A²，整栈复用给出BABA，差为B(BA−AB)A作用于输入。只有相应交换关系成立，或差异恰好在这个输入上抵消，输出才相同。相邻复用确实缩短了同一参数与状态两次相遇的间隔；这种顺序是否让训练梯度更一致，则要继续看两次访问各自传回什么。

对一个物理层，从不依赖本层参数的输入x₀开始，第一次调用生成x₁，第二次生成x₂。改变共享参数时，x₂会受到两条影响：参数直接改变第二次变换；参数先改变x₁，再由第二次变换把这份变化传下去。下面的B₁、B₂分别表示两次调用对参数的直接导数，J₂表示第二次调用对输入的导数，v₂来自后续网络。公式在top-k排序不切换的可微分支上成立：

```math
\begin{aligned}
x_1&=F_\theta(x_0),\qquad x_2=F_\theta(x_1),\\
J_2&=\partial_xF_\theta(x_1),\\
B_1&=\partial_\theta F_\theta(x_0)\big|_{x_0},\quad
B_2=\partial_\theta F_\theta(x_1)\big|_{x_1},\\
\partial_\theta x_2&=J_2B_1+B_2,\\
\nabla_\theta\mathcal L
&=\underbrace{B_1^\top J_2^\top v_2}_{g_1}
+\underbrace{B_2^\top v_2}_{g_2}.
\end{aligned}
```

反传于是把两次访问的贡献加到同一组参数上：g₁先穿过第二次调用的J₂，g₂从第二次调用直接进入参数。这就是共享权重如何得到梯度，通常不等于把某一次梯度乘二。工程上应完成两次前向及其反传，再执行一次optimizer step，保持这条计算图中的参数一致。

两份梯度还可能相互强化或抵消。作者关于局部共享更一致的动机，可以通过同一批数据上的夹角检验；下面的恒等式显示了这项测量与合成梯度大小的关系：

```math
\|g_1+g_2\|^2
=\|g_1\|^2+\|g_2\|^2
+2\|g_1\|\|g_2\|\cos(g_1,g_2).
```

当夹角余弦为正时，交叉项增大合成梯度的平方范数；为负时则减小。它说明应怎样检验“更一致”，但梯度大小与泛化收益仍是不同问题。论文没有报告这一夹角指标，因此局部共享的性能结果与这份可能的梯度解释应分别理解。（[原文 §2.1–2.2](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[TransformerLayer.forward](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_layer.py)）

两份 preview 的 norm_topk_prob 还改变了上述导数。令 π 为全专家 softmax，S 为选中的八个专家，Z 为其概率质量，eⱼ 为专家输出；在相同输入和相同专家权重下：

```math
\begin{aligned}
Z&=\sum_{j\in S}\pi_j,\\
y_{\rm renorm}&=\sum_{j\in S}\frac{\pi_j}{Z}\,e_j,\\
y_{\rm raw}&=\sum_{j\in S}\pi_j e_j
=Zy_{\rm renorm}.
\end{aligned}
```

在 S 固定且专家输出固定时，对未选中专家的 router logit zⱼ，重归一化路径的直接导数为零，保留全 softmax 质量的路径则保留分母耦合：

```math
\begin{aligned}
\partial_{z_j}y_{\rm renorm}&=0,\qquad j\notin S,\\
\partial_{z_j}y_{\rm raw}&=-\pi_j y_{\rm raw},\qquad j\notin S.
\end{aligned}
```

这里的零导数仅指 router logit 到该次 MoE 输出的直接路径；共享隐状态和其他损失仍可能产生梯度。6B 与 20B 实际权重、维度不同，上式用于解释配置语义，不能据此把两个成品模型的输出直接相除。实现位置是 [Qwen3MoeTopKRouter.forward](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)。

# 训练成本与硬件配方

两次执行仍要完成两次计算，Loopie的训练效率收益来自另一处：权重和激活的存储方式给更大的microbatch留下了空间，而硬件可能更高效地处理它。论文§2.4（pp.6–9）的比较因此固定硬件、序列长、每次optimizer step的token数、checkpointing策略、优化器与数据，再匹配实测step时间；compute-matched在这里指实际训练时间，而非理论FLOPs相等。

用P表示数据并行副本数、b表示每副本microbatch、g表示累积次数、s表示序列长。把b翻倍、g减半，就能保持一次参数更新看到的token数不变：

```math
\begin{aligned}
B_{\rm global}&=Pbg,\\
T_{\rm update}&=Pbg\,s,\\
b_1&=2b_0,\\
g_1&=g_0/2.
\end{aligned}
```

```math
\begin{aligned}
\widehat C&=\frac{LRD^2}{48\cdot2048^2},\\
\widehat M_{\rm act}&=\frac{LD}{48\cdot2048},\\
t_{\rm step}(2304,27,2;2b_0,g_0/2)
&\approx t_{\rm step}(2048,48,1;b_0,g_0).
\end{aligned}
```

具体配方先从Qwen3-like 30B-A3B的48层出发：减为24个物理层，每层算两遍，总执行层数仍为48。论文把同一物理层的全部访问放进一个checkpoint单元，跨单元保存的边界因此随物理层数减少；相同microbatch下，层边界激活proxy先降为一半。

这份空间余量随后用于扩大microbatch、减少累积次数，再把实测效率收益投入更宽、更深的模型。候选width／depth为2176／25、2304／27和2432／28，最终选中2304／27。省下的是特定checkpoint划分下的存储，不是第二次访问的计算。

扩宽加深以后，所选配置的块计算proxy达到baseline的1.424×，相同microbatch下的激活proxy约为0.633×。这两项一起说明配方的取舍：执行更多块计算，同时仍为batch调整留出存储余量。完整峰值还包括优化器、通信和临时buffer，能否运行、每步多久最终由实测决定。

图3中EP=8时，Loopie采用MBS=2、吞吐261.53 TFLOP/s，baseline采用MBS=1、189.65 TFLOP/s。这是各自网格搜索后的最好配置；数值描述该训练设置的硬件利用，不能直接转成所有GPU或推理服务的固定加速比。

算法顺序：固定数据／global batch／硬件 → 枚举宽深与 TP/PP/EP/MBS → 测可运行峰值显存与完整 optimizer-step 时间 → 保持 token/update 不变 → 选最接近 baseline 时间的候选 → 用相同 token 序列训练。论文主张的效率来自双倍 microbatch，不是跳过第二次计算。

## 从激活边界到账本与计时目标

checkpoint会保存单元入口，反向时从这里重算内部过程。把两次循环访问放在同一单元，便无须把它们之间的状态作为跨单元边界长期保存；但重算时内部张量仍要存在，专家通信与优化器状态也照常占用空间。因此峰值显存要分成下面几项，减少边界只改变其中一项：

```math
\begin{aligned}
M_{\rm peak}
&=M_{\rm weights}+M_{\rm optimizer}+M_{\rm grads}\\
&\quad+M_{\rm boundary}+M_{\rm live}+M_{\rm communication},\\
M_{\rm boundary}&\simeq\kappa\,b sD\,n_{\rm checkpoints}.
\end{aligned}
```

若每个物理层就是一个checkpoint，保存边界的数量随物理层数L变化；若把多层合为一个chunk，则按实际边界数计算。这解释了为何R不再直接乘在边界项上。完整前向仍执行R次访问，反向重算也必须恢复同样的访问顺序，存活的中间张量与MoE工作区仍在峰值账目中。[checkpointed_forward 的 custom_forward](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/recompute.py) 按物理层调用，而每层的forward包含所有访问。

所选 2304×27×2 模型的 0.633 是“相同 microbatch”的边界激活比。若 microbatch 翻倍，该同一 proxy 为：

```math
\frac{M_{{\rm boundary},1}}{M_{{\rm boundary},0}}
\simeq\frac{2\cdot2304\cdot27}{2048\cdot48}
=1.265625.
```

代入翻倍后的microbatch，边界proxy已经变为baseline的1.265625倍。原先的余量用于承载更大的batch和模型，而非保留到最终配置仍只占63.3%显存。0.633与1.265625分别回答同microbatch和新microbatch下的边界存储问题；完整峰值仍需测量。（[原文 §2.4、表1](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)）

模型、并行划分、microbatch和累积次数需要一起选择，因为任何一项都会改变可行显存与step时间。把这一组选择记为候选a，下面的约束要求它既装得进设备，又保持每次更新的token数，再从中寻找最接近baseline时间的配置。这是论文实验选择过程的数学表达，未假定作者公布了对应搜索程序：

```math
\begin{aligned}
a^*&\in\arg\min_{a\in\mathcal A_{\rm feasible}}
\left|\frac{\widehat t_{\rm step}(a)}{\widehat t_{\rm step}(a_0)}-1\right|,\\
T_{\rm update}(a)&=P(a)b(a)g(a)s,\\
\mathcal A_{\rm feasible}&=\{a\in\mathcal A:M_{\rm peak}(a)\le M_{\rm device}\}\\
&\quad\cap\{a:T_{\rm update}(a)=T_{\rm target}\}.
\end{aligned}
```

执行顺序是：固定全局 token batch 与全局数据次序；枚举候选并计算整数累积次数；预热至稳定；在所有 rank 完成 GPU 工作及 optimizer step 后记录时间和峰值；报告重复测量的统计量；按上式选配置。GPU 工作异步时，仅测 CPU 发射耗时不满足这一计时定义。并行网格改变后应重新取得实际数据并行副本数，不能沿用旧的 P。

同一次测量中，token 吞吐与所报 FLOP 吞吐分别为：

```math
\begin{aligned}
\operatorname{tokens/s}&=T_{\rm update}/t_{\rm step},\\
\operatorname{FLOPs/s/GPU}
&=F_{\rm train/token}\,T_{\rm update}/(n_{\rm GPU}t_{\rm step}).
\end{aligned}
```

FLOP吞吐的分子还乘了每token训练FLOPs，所以更大的模型即使每步时间相近，也可以报告更高的TFLOP/s。图3的吞吐比约1.379，反映单位时间完成的所报浮点运算量；固定token预算的训练速度要看tokens/s或step时间，单请求decode则需要另一项测量。（[原文 §2.4–2.5、图3](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)）

# 原生预训练：初始化、数据、优化与损失

§3、附录 B（pp.13–15、36–37）规定：embedding 与 lm_head 的初始化方差为1/D，输入 embedding 乘 √D；block 投影采用 SmallInit 方差2/(5D)。这是初始化与尺度设计，不是循环收敛定理。

```math
\begin{aligned}
E_{ij},(W_U)_{ij}&\sim\mathcal N(0,D^{-1}),\\
W^{\rm block}_{ij}&\sim\mathcal N\!\left(0,\frac{2}{5D}\right),\\
\mathcal L_{\rm PT}&=\mathcal L_{\rm token\ CE}+0.01\,\mathcal L_{\rm router}.
\end{aligned}
```

最后一式汇总论文给出的 CE 目标与 MoE auxiliary coefficient；具体跨 token／并行组归约以训练实现为准。AdamW β=(0.9,0.95)、ε=10⁻¹⁵、weight decay=0.1、global grad clip=1；global batch=1024、context=8192。6B／20B 的 LR 分别5×10⁻⁴／3×10⁻⁴，warmup2000／6000 steps，之后恒定；Stage2仍恒定 LR。这里“高质量 annealing”主要指数据阶段，不代表余弦降学习率。

| 阶段 | 原文数据与规模 | 需要落实的工程契约 |
|---|---|---|
| Stage1 | Nemotron-CC-v2-HQ，约570B unique tokens，4 epochs≈2.28T | 多 epoch 采样、shuffle、tokenizer与loss mask |
| Stage2 | 约1263B：SFT-style351B，specialized277B，code262B，synthetic web197B，math126B，web25B，synthetic25B | 比例27.8／21.9／20.7／15.6／10／2／2%；code采60%，synthetic采16%，math质量≥4 |
| Base | 完成前两阶段 | 保存拓扑、参数、tokenizer、位置配置和数据版本 |

原文存在预算口径未统一：p.13称 Stage1=3T，p.14给570B×4≈2.28T，p.20表3写总预训练3.5T。按详细阶段相加约3.54T与表3接近，但不能据此抹去p.13的3T。页面同时保留各说法，精确复现仍需作者训练日志／勘误。

## 初始化矩与有效监督量

在独立零均值初始化的假设下，embedding 缩放的直接作用可由二阶矩说明：一个未缩放 token 向量的期望平方范数为1；乘以宽度平方根后变为 D。对于输出维度 m 的 SmallInit 线性映射，条件于固定输入 h：

```math
\begin{aligned}
\mathbb E\|E(x)\|^2&=1,\\
\mathbb E\|\sqrt D\,E(x)\|^2&=D,\\
\mathbb E_W\|Wh\|^2&=\frac{2m}{5D}\|h\|^2.
\end{aligned}
```

最后一式来自逐行独立权重的方差相加。它解释初始激活量级，但不把注意力 softmax、SwiGLU、共享访问之间的相关性或训练后的权重纳入保证。特别是 20B 采用 embedding 与读出权重绑定，不能把它的输入和读出当作彼此独立的随机矩阵来证明全网稳定。（[原文 §3、附录B](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；本节二阶矩是条件化推导。）

论文将 PT 与 SPT 的 CE 写成有效标签上的全局平均。以下先明确该数学目标；工程上是否与之相等，还须沿实际 schedule 与分布式归约核对。令 d 为数据并行副本，j 为一次 update 内的 microbatch：

```math
\begin{aligned}
S_{d,j}&=\sum_t w_{d,j,t}\,\ell_{d,j,t},\\
N_{d,j}&=\sum_t w_{d,j,t},\\
\mathcal L_{\rm CE}
&=\frac{\sum_{d,j}S_{d,j}}{\sum_{d,j}N_{d,j}}.
\end{aligned}
```

该目标一般不等于各 microbatch 平均损失的等权平均。SPT 的 prompt 长度、response 长度和 padding 量不同，使这个区别成为真实的训练权重差异。若某执行器将各副本梯度取平均、将 microbatch 梯度相加，则每个局部损失应乘 P/N；若执行器还对 microbatch 取平均，需要补回相应因子。这里给的是目标所要求的归约关系，Megatron 的 TP、CP、DP 和辅助项须按其实际外层实现对齐，不能机械再除一次 global batch。

```math
\frac1P\sum_d\sum_j
\nabla_\theta\!\left(\frac{P S_{d,j}}N\right)
=\nabla_\theta\frac{\sum_{d,j}S_{d,j}}N,
\qquad N=\sum_{d,j}N_{d,j}>0.
```

[pretrain_gpt.loss_func](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/pretrain_gpt.py) 的普通 CE 路径实际返回 masked loss 总和、loss_mask 之和及二者的报告值；PT 中后者通常是非 padding token 数，SPT 中则须是有效响应标签数。SPT 名义 batch 容量为134,217,728个位置；只有 response loss mask 为1的位置进入上述 N。为了比较学习率和数据效率，应同时记录 nominal positions、有效监督 token 数以及 router 辅助项自己的归约分母，而不把三者合成一个 token 数。（[原文 §4.1、附录D表8](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[TopKRouter._apply_aux_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py)）

## 从损失返回值到真正的更新目标

同一个 loss_func 返回分子与计数，还不足以证明最终梯度是全局 token 平均。已固定 Megatron revision 中，TransformerConfig.calculate_per_token_loss 默认 false；forward_step_calc_loss 的三返回值分支会按该开关执行两种不同归约。论文没有公布 SPT 的实际启动参数，因此下面识别的是可核代码分支，不能据默认值判定作者采用了哪一种。

先在 TP=CP=EP=1、P 个数据并行副本、每副本 J 个 microbatch 的情形写清数学语义。各 microbatch 的监督 token 数 Nᵈʲ>0，分子仍记 Sᵈʲ：

```math
\begin{aligned}
\mathcal L_{\rm micro}
&=\frac1{PJ}\sum_{d,j}\frac{S_{d,j}}{N_{d,j}},\\
\mathcal L_{\rm token}
&=\frac{\sum_{d,j}S_{d,j}}{\sum_{d,j}N_{d,j}}.
\end{aligned}
```

| 开关及代码阶段 | 实际执行 | 该简化并行情形的结果 |
|---|---|---|
| false：forward_step_calc_loss | 局部 loss 先除 clamp(num_tokens,1)，再除 num_microbatches；DDP 完成副本平均 | microbatch 平均损失的等权平均 |
| true：forward_step_calc_loss 与 DDP | 保留局部 loss 总和；DDP 设置梯度预缩放为1，并禁止 average_in_collective | 先累加分子梯度 |
| true：finalize_model_grads | 完成梯度同步；token 数从最后 pipeline stage 广播，再在 DP×CP 组求和；统一 scale_gradients | 除全局有效标签数，得到 token 平均 |

当各 Nᵈʲ 相等时，两目标一致；否则连更新方向也可能不同。取两个 microbatch，监督数为1、9，各自平均损失对一个参数的导数为2、−1，则 micro 平均梯度为0.5，token 平均为−0.7。这个反例说明：把 microbatch 加倍、累积次数减半，只有同时保持全局样本与正确归约，才能把它解释为相同训练目标的调度变化。零标签 microbatch 的 clamp 防止除零，却没有创造监督；全局零标签 batch 应作为无 CE 监督事件单独登记。

MoE 辅助项还存在另一套计数。在 TP=CP=DP=1、标准 aux_loss、每次访问均重新路由的情形，令 Aʲν 已包含 auxiliary coefficient，Mʲν 是该访问传给 router 的有效路由 token 数。当前 attach_and_log_load_balancing_loss 在 per-token 分支先乘 Mʲν；最终统一梯度归一化产生：

```math
\begin{aligned}
g_{\rm aux,token}
&=\frac{\sum_{j,\nu}M_{j,\nu}\nabla_\theta A_{j,\nu}}
{\sum_jN_j},\\
g_{\rm aux,micro}
&=\frac1J\sum_{j,\nu}\nabla_\theta A_{j,\nu}.
\end{aligned}
```

若 Mʲν=M 恒定而 Nʲ=qM，前者相对后者多出1/q；因此监督比例改变会改变 CE 与辅助梯度的相对权重。此式限定于所列分支；TP／CP 的额外因子由源码中的 tp_cp_group.size() 与梯度归约处理。loss_mask 指定哪些标签计 CE，padding_mask 指定哪些激活属于 padding；prompt 虽不计直接 CE，仍是有效上下文，不能为了凑分母而把 prompt 标成 padding。

工程核对顺序是：保存实际配置开关及并行组 → 记录每个 microbatch 的 S、N 和每个 visit 的 M → 跟踪 schedule 是否已做局部除法 → 核对 DDP 的 sum／average → 在 finalize_model_grads 后检查全局分母 → 在整次 update 上做裁剪和一次 optimizer.step。router 的系数、次数、分母应与 CE 分开记录。源码支持这些分支，但没有补出论文未公开的 SPT packing／启动脚本。

来源：[TransformerConfig](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_config.py)、[forward_step_calc_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/pipeline_parallel/schedules.py)、[DistributedDataParallel](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/distributed/distributed_data_parallel.py)、[finalize_model_grads](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/distributed/finalize_model_grads.py)、[attach_and_log_load_balancing_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py)。目标等价条件和方向反例为本节推导。

# SPT 与 RL：原论文后训练链路

图8给出完整阶段链：PT → 高质量阶段 → Base → SPT → RL → Thinking。理解SPT时，要把“哪些位置计loss”和“以多大规模优化”分开：它沿用SFT的监督位置与损失形式，改变的是优化规模。原文§4.1（pp.16–18）用下式统一描述这些目标：

```math
\begin{aligned}
\mathcal L_{\rm CE}(\theta;w)
&=-\frac{\sum_i\sum_t w_{i,t}\log p_\theta(z_{i,t}\mid z_{i,<t})}
{\sum_i\sum_t w_{i,t}},\\
w_{i,t}&=\mathbf1\{z_{i,t}\text{ 属于监督响应且非 padding}\}.
\end{aligned}
```

PT 对所有非 padding 位置计损失；SPT／SFT 只对 target 计损失。SPT 用1024条×131072位置／batch，LR=10⁻⁵、warmup500后恒定，约2T token、10 epochs，aux coefficient仍0.01（附录D表8）。名义 token positions 与实际受监督 token 数必须分别记录，不能把128M名义位置全当有效标签。图9–11报告训练中损失平滑、知识和推理指标同时提高；大 batch 防遗忘的解释是作者假说，论文自述尚无充分 SPT 因子消融，不能当普适定理。

## 原文 Table 2 中的监督位置与优化规模

下表完整保留原文代表性配置与观察；箭头相对各自的起始checkpoint，稳定性只描述作者考察过的训练范围。SFT列主要引用OLMo3、Nemotron3、Step-3.5-Flash等配方，不是固定同一模型／数据的单因素SFT对照。[§4.1、Table2，p.16](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)

| 原文维度 | SPT | SFT | PT |
| --- | --- | --- | --- |
| loss function | Cross-entropy | Cross-entropy | Cross-entropy |
| 计loss的位置 | 仅监督目标token | 仅监督目标token | 全部非padding token |
| 预训练指标 | ↑ | ↓ | ↑ |
| 推理指标 | ↑ | ↑ | ≈ |
| 多epoch过拟合 | 未观察到 | 观察到 | 未观察到 |
| global batch size | ≥1024 | 32–128 | ≥1024 |
| sequence length | ≥128K | 8–32K | 4–8K |
| 每batch名义token位置 | ≥128M | 128K–1M | 8–32M |
| 总训练token预算 | ≥2T | 10–100B | ≥4T |

读这张表时，先区分配置范围和具体训练设置。各列范围来自多套配方，不能任意组合；实际名义位置仍由B×T计算。PT列的≥4T是Table2的代表性范围，不替代§3／Table3中Loopie自身的阶段预算。原文的3T、2.28T与3.5T记载仍有差异，需要单独保留。

再看表中的性能与稳定性观察。它们并不意味着SPT一定不遗忘，或SFT必定过拟合。Fig.9只绘6B训练loss；原文称验证曲线与之近乎重叠，十个epoch边界也未出现loss cliff。Figs.10–11则跟踪推理与通用指标。要区分batch、context、学习率和重复次数各自的作用，仍需受控消融。计算loss时，还须按下文区分目标mask、有效监督量和归约方式。

## 响应损失掩码保留 prompt 的间接梯度

SPT 把 prompt 的直接 CE 权重设为零，但响应预测仍以 prompt 为条件。设 zₜ 是预测目标 yₜ 的 logits，pₜ 为其 softmax，hⱼ 是较早 prompt 位置的一个隐状态节点，𝒯 为计损失的响应位置集合。固定 token 与 mask，在可微分支上：

```math
\frac{\partial\mathcal L_{\rm CE}}{\partial h_j}
=\frac1N\sum_{\substack{t\in\mathcal T\\t>j}}
\left(\frac{\partial z_t}{\partial h_j}\right)^\top
(p_t-e_{y_t}).
```

右侧可通过 attention 的 K/V 路径非零，与 prompt 的直接标签权重为零并不矛盾。最小例子是响应仅读取一个 prompt value，令该 value 为 θ、响应 logit 为 aθ，则二分类 CE 的梯度为 a[σ(aθ)−y]。即使 prompt 没有自己的 CE 标签，这个梯度一般仍不为零。

因此，按论文的完整 SPT 目标训练时，应保留 prompt 与 response 的联合因果计算图，然后对响应标签乘 loss_mask。若先用 no_grad 计算 prompt，或把其 K/V detach 后再训练响应，就删去了上式的一部分参数梯度；那是另一种训练近似，不能由“只监督响应”直接推出。把跨文档 attention 切断则由 document／causal mask 控制，两种 mask 的作用不同。

从公式追到代码，先看输入计算，再看loss加权：[pretrain_gpt.forward_step 与 loss_func](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/pretrain_gpt.py) 先将完整 tokens、position_ids、attention_mask 送入模型，再对逐位置 loss 使用 loss_mask；[GPTModel.forward](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/models/gpt/gpt_model.py) 的完整 decoder 路径与该语义相容。实际 SPT 数据适配器仍须提供正确的目标边界、packing 与 mask。（[原文 §4.1](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)）

## token 效率应连同监督预算和更新次数阅读

PT、SPT 和最终 Thinking 的数据账本必须保留阶段。名义位置数 Dₙₒₘ 统计执行 batch 的容量，Dₛₑₑₙ 统计非 padding token 曝光，Dₛᵤₚ 统计 loss_mask 为1的标签；Dᵤₙᵢq 按文档 ID 与 token offset 去重，计数的是未重复的语料位置，并非词表中的不同 token。通常有：

```math
\begin{aligned}
D_{\rm sup}&\le D_{\rm seen}\le D_{\rm nom},\\
D_{\rm uniq}&\le D_{\rm seen},\\
q&=D_{\rm sup}/D_{\rm nom}.
\end{aligned}
```

原文 Stage1 的570B unique×4明确区分语料量与重复曝光；SPT 的2T、约10 epochs不能在未说明 token 口径时直接改成2T有效监督标签，也不能与无重复预训练数据等价。只有口径一致、完整重复同一固定语料等条件成立时，除以10才给出约200B的单轮对应 token 量。

更新次数是另一项独立预算。若每步 batch 填满、序列长与 batch 固定、没有跳过更新，且文中 token 指名义位置，则 U≈Dₙₒₘ/(B·s)。附录 B／D 的 batch 容量给出下列条件化回算；这不是作者训练日志：

| 数量 | PT | SPT |
|---|---:|---:|
| 每步名义位置 B·s | 8,388,608 | 134,217,728 |
| 按表3的3.5T与SPT的2T条件回算 update | 约417,233 | 约14,901 |
| SPT 500步 warmup 对应名义位置 | — | 67,108,864,000 |

所以“大 batch＋十轮数据”同时改变每步监督聚合、总更新次数与学习率时间尺度。仅把小 batch SFT 多跑几轮，不能等价实现这组条件；大 batch 防遗忘仍是原文提出的机制假说，现有实验没有隔离 batch、context、学习率、数据与更新次数的各自因果作用。工程上还应记录 successful_updates：当前 train_step 只在 optimizer.step 成功时推进 opt_param_scheduler，且增量按全局样本数计，并非每出现一个有效标签就推进一次。

原文表3的“3.5T vs 25T”等比较只报告预训练列；最终 Thinking 还经历约2T SPT和数学／代码RL，且开放模型的数据混合与后训练不相同。因此表3可保留为该阶段口径的效率与最终能力对照，不能推出“完整 Thinking 训练总成本小于七分之一”或循环拓扑的单独因果倍率。（[原文 §3、§4.1、§4.3、附录B/D](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[train_step 与 scheduler 更新](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/training/training.py)）

## RL 的序列目标与阶段安排

RL阶段的反馈给完整回答：数学答案由等价verifier判断，代码由沙箱测试判断。论文§4.2（pp.18–19）据此采用GSPO，先比较同一prompt下各回答的奖励，得到组内优势，再把一个回答内的token概率变化聚合为单个序列权重。训练先数学后代码，使用Guru-RL子集并清洗数据，结合DAPO的不对称裁剪与动态过滤。下面两式分别对应“这份回答相对同组如何”和“当前策略相对旧策略怎样改变了它的概率”：

```math
\begin{aligned}
\widehat A_i&=\frac{R_i-\operatorname{mean}_jR_j}{\operatorname{std}_jR_j+\varepsilon},\\
s_i(\theta)&=\exp\!\left[\frac{1}{|o_i|}\sum_t
\log\frac{\pi_\theta(o_{i,t}\mid q,o_{i,<t})}{\pi_{\rm old}(o_{i,t}\mid q,o_{i,<t})}\right].
\end{aligned}
```

```math
\begin{aligned}
J(\theta)&=\mathbb E\frac1G\sum_i\min\!\bigl[\\
&\qquad s_i\widehat A_i,
\operatorname{clip}(s_i,1-\varepsilon_{\rm low},1+\varepsilon_{\rm high})\widehat A_i\bigr],\\
\varepsilon_{\rm high}&>\varepsilon_{\rm low}.
\end{aligned}
```

更新时，一个回答先得到一个sᵢ，再作为整体接受裁剪；不是把每个token各自裁剪。采样也以prompt组为单位：只保留同组既有成功又有失败的情况，丢组后持续补样，直到有效batch满。这样的组才能提供成功与失败的相对信号。

长度安排与学习目标分开控制：先用32K最大响应，截断比例超过10%时转为64K；训练监控准确率、响应长、熵和截断率，选择持续验证退化前的checkpoint。这里调整的是显式回答／推理行为，原文没有另行训练每个token的循环停止策略。

## GSPO 的梯度与动态筛选后的分布

沿一个已采到的回答看，真实序列概率比ρᵢ把各token的概率比相乘；取log后就是求和。GSPO先对这份log ratio除以有效长度nᵢ，再取exp，所以sᵢ是这些概率比的几何平均，也是ρᵢ的nᵢ次方根。更新时回答、奖励与长度都视为旧策略采到的固定数据，仅对当前策略概率求导：

```math
\begin{aligned}
\rho_i(\theta)
&=\frac{\pi_\theta(o_i\mid q)}{\pi_{\rm old}(o_i\mid q)},\\
s_i(\theta)&=\rho_i(\theta)^{1/n_i},\\
\nabla_\theta s_i
&=\frac{s_i}{n_i}\sum_{t=1}^{n_i}
\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
\end{aligned}
```

这一步长度归一化把裁剪施加在“平均每token的概率变化”上，并使每个token对∇sᵢ的贡献带有1/nᵢ。sᵢ由此成为论文选用的优化surrogate权重，通常不再具备真实密度比将旧策略期望精确换成新策略期望的性质。一个固定长度n、概率归一化的离散响应空间已足以显示差别：Hölder不等式给出其旧策略期望至多为1，而非密度比所需的恒等于1：

```math
\mathbb E_{o\sim\pi_{\rm old}}[s(o)]
=\sum_o\pi_{\rm old}(o)^{1-1/n}\pi_\theta(o)^{1/n}
\le1.
```

该固定长度特例用于辨别概率比的含义；真实可变长度响应不能直接套用这一统一指数的不等式。响应必须包含实际采样的终止事件；到达长度上限是截断，不应伪造一个采样 EOS 的概率。

裁剪控制的是继续强化某种变化的梯度。正优势回答希望提高sᵢ，超过上界后才停止进一步奖励增大；负优势回答希望降低sᵢ，低于下界后才停止进一步推动减小。因此“落在区间外”本身还不足以判定梯度为零，要同时看优势的符号。除裁剪边界外，这个选择和前面的长度归一化共同给出：

```math
\begin{aligned}
\chi_i&=\mathbf1\{\widehat A_i>0,\ s_i<1+\varepsilon_{\rm high}\}\\
&\quad+\mathbf1\{\widehat A_i<0,\ s_i>1-\varepsilon_{\rm low}\},\\
\nabla_\theta J
&=\frac1G\sum_i
\chi_i\widehat A_i\frac{s_i}{n_i}
\sum_t\nabla_\theta\log\pi_\theta(o_{i,t}\mid q,o_{i,<t}).
\end{aligned}
```

动态筛选处理的是另一种浪费：同一prompt的G个回答若全成功或全失败，组内奖励相同，前述优势都为零。保留有成有败的组，可以把有效batch放在当前策略还能区分回答的prompt上，但也会改变prompt出现的频率。在给定prompt的G次完成独立采样、单次成功概率为p(q)时，“至少一成一败”的概率为：

```math
\begin{aligned}
a(q)&=1-p(q)^G-[1-p(q)]^G,\\
\mathcal D_{\rm kept}(q)
&=\frac{\mathcal D(q)a(q)}{\mathbb E_{q\sim\mathcal D}a(q)}.
\end{aligned}
```

p(q)接近0或1时，很难采到混合结果，保留权重a(q)较小；中间难度的prompt更容易留下。持续补样得到的因此是按a(q)重加权的prompt分布，而且a(q)随旧策略能力变化。组内完成在经过“有成有败”条件筛选后，也不再是无条件独立样本。

这解释了动态筛选如何把训练集中到当前有区分度的问题，同时说明它不是对原始prompt分布保持不变的无偏估计。上述关系由[Loopie §4.2 原始目标](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)推导，不额外假定论文公布了RL训练程序。

下方算法按“采组—筛组—补足—更新”的顺序实现论文目标。进入更新后，当前参数要重新处理这些回答，才能取得可求导的当前log probability；旧参数生成的hidden或KV承接的是旧计算，不能代替当前前向。rollout与重放同时使用两次逐层循环、相同位置配置和概率定义，才使两份log probability对应同一事件。

精确学习率、采样温度、裁剪阈值和并行配置仍由实际训练配方提供；算法只展开论文已经描述的目标和筛选规则。

### 算法 动态筛选与序列级 GSPO 更新

**输入** 旧策略、prompt 分布、每组样本数、二值 verifier、有效 batch 大小、最大候选组数与裁剪阈值。

**输出** 一次 GSPO surrogate 更新及采样、截断和筛选统计。

**执行前提** 固定两次逐层循环；rollout 与重放使用相同位置、tokenizer 和概率定义。；使用二值成功奖励，G至少为2；旧 log probability、奖励、优势与实际响应长度均作为常量。

![算法 动态筛选与序列级 GSPO 更新](visualize:fde1_bGliZmlsZV8yMzFrZWc4MFRWZDg4RUxpYmlsbFpR_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 按 Loopie §4.2 的二值 verifier 情形重写；最大候选组数和凑不满时不更新是此参考算法显式加入的工程终止规则。当前未取得作者 RL 脚本，具体超参及执行器归约须作为输入。

## 公开 RL 调用链与论文 GSPO 配方的差异

论文 §4.2 明确规定序列级 GSPO 与持续补齐有效组的 DAPO 动态过滤。固定 Megatron revision 的 examples/rl/README.md 则自称通用 GRPO 示例，示例对象为 Qwen2.5-32B；不能把它的超参或算法直接标为 Loopie 的实际训练配方。这里沿真实调用追踪这条公开路径，并不据此判定作者的论文训练采用了错误算法。（[论文 §4.2](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[公开示例](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/examples/rl/README.md)）

| 调用位置 | 输入与状态变化 | 与论文复现直接相关的事实 |
|---|---|---|
| training.train → get_grpo_data_iterator | 首次或重采样边界收集rollout，其余update复用buffer | 原始公式用 grpo_iterations×⌊prompts×group_size/global_batch_size⌋ 决定重采样间隔；须保证正整数和预期整除关系 |
| get_environment_rollouts → get_rollout_generator | 独立inference_model先swap_model_weights；否则复用训练模型；传入temperature/top_p/top_k | get_inference_interface实际调用MegatronLocal.launch；这条路径没有自动调用作者vLLM fork |
| prepare_data_for_update | 先按prompt组计算奖励均值/标准差及优势，再展平、分配DP、构造mask与packing | 组归一化必须发生在打散之前；多轮通用实现会把trajectory优势复制到各turn，不自动等于论文单响应样本 |
| compute_logprobs_batch | 更新前在no_grad中计算old logprobs；暂载reference权重计算ref，再恢复当前权重 | 旧概率与ref概率是数据；train_rl.forward_step另以no_grad=false重算current logprobs |
| calculate_grpo_loss → train_rl.loss_func | ratio和clip保持逐token形状，优势广播至token；最后loss_mask求和 | 这一已读分支是token GRPO，未实现论文“先按响应平均log ratio，再exp与clip”的操作 |

对应源文件：[训练调度](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/training/training.py)、[rl_utils.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)、[train_rl.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/train_rl.py)。旧策略重放与实际rollout仍须使用一致的权重版本、循环拓扑、位置、采样概率语义和缓存；refit只复制参数，不替代这项一致性检验。partial rollouts还须检查policy_epoch与kv_cache_epoch，不能默认所有保留轨迹都来自同一无过期旧策略。

动态过滤有一个明确的入口阻断。get_rollout_generator将grpo_filter_groups_with_same_reward传给WeightedMultiTask，后者继续传给子agent。示例DAPOAgent→MathAgent→RewardOnlyAgent继承GroupedRolloutGenerator；其pipeline构造调用_GranularityConfig._validate，并显式断言该过滤开关为false。源码解释：丢弃的组没有重新生成，非streaming调用会少交付，按batch顺序消费时会等不齐。stage_assemble虽保留std(rewards)>10⁻⁶条件，注释明确称其为当前不可达的过滤代码。

上面的校验会拒绝该flag，因此这条调用链不能只靠打开开关，就完成论文的“筛掉全对/全错并补样至满batch”。要实现这一过程，还需补齐丢组后的生成、计数和消费调度，并在提交有效组时保留完整G个响应及其共同prompt身份。验证器超时、数值异常和合法的失败奖励也应分别记录。这个判断仅针对上述固定revision与示例调用链；尚未取得可核的论文专用GSPO入口，不能由此判断作者其他未公开实现。（[WeightedMultiTask.get_grouped_rollouts](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/weighted_multi_task.py)；[GroupedRolloutGenerator与_GranularityConfig](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/api.py)；[RewardOnlyAgent](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/reward_only_agent.py)）

## 从 token 损失接口落实序列级目标

公开calculate_grpo_advantages对同prompt组的reward使用NumPy总体标准差（ddof=0）和10⁻⁴；在单turn情形返回每响应一个常数优势。calculate_grpo_loss先令rᵢₜ=exp(current_logprobs−old_logprobs)，逐token裁剪，再广播Aᵢ；packed情形按seq_starts/seq_lengths把优势填回相应范围。相同的组优势和不对称clip参数并不使这一路径变成GSPO。

```math
\begin{aligned}
d_{i,t}&=\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})
-\log\pi_{\rm old}(o_{i,t}\mid q,o_{i,<t}),\\
\phi(r,A)&=\min\{rA,\operatorname{clip}(r,1-\epsilon_l,1+\epsilon_h)A\},\\
J_{{\rm token},i}&=\frac1{n_i}\sum_t m_{i,t}\phi(e^{d_{i,t}},A_i),\\
J_{{\rm seq},i}&=\phi\!\left(
\exp\!\left[\frac{\sum_t m_{i,t}d_{i,t}}{n_i}\right],A_i\right),
\qquad n_i=\sum_t m_{i,t}>0 .
\end{aligned}
```

取一份只有两个有效token的回答，概率比分别为2与1/2，优势A=1。使用示例对称阈值0.2时，token目标先把2截成1.2、保留1/2，再平均，得到0.85；序列目标则先把两者的log ratio平均，得到s=1，裁剪不触发，目标为1。

更新也随顺序改变：对两个log ratio的梯度分别是token目标的(0,0.25)和序列目标的(0.5,0.5)。前者只推动第二个token，后者让整个回答的平均概率变化一起参与。这一例子关闭额外正则与推理校正，只隔离归约和裁剪顺序；阈值不是论文未公开的超参。

| 张量或索引 | 已读实现的形状与对齐 | GSPO迁移所需契约 |
|---|---|---|
| 未packing tokens / logits | [B,T] / [B,T,V]；get_logprobs使用logits[:,:−1]预测tokens[:,1:] | old/current/reference都为[B,T−1]；生成mask同样右移一位 |
| packing bin | tokens为[1,K]，logprobs为[1,K−1]；seq_lengths记录prompt＋response完整长度 | nᵢ须取该响应有效mask之和，不能取seq_lengths−1或整个bin长度 |
| 序列边界与位置 | 每序列position_ids从0重新开始；PackedSeqParams的cu_seqlens划分独立attention段 | 保留原response ID；分别segment-sum log ratio和mask，跨序列的下一token边界预测不得计入 |
| 空bin与零有效响应 | 调度可能填空bin；loss_func把零token计数替成1以避免除零 | 不得把空bin计成一条有效响应；nᵢ=0不能进入序列ratio |
| 旧概率与优势 | prepare_data_for_update在无梯度区域生成，更新时复用 | 对old/ref/inference logprobs、奖励、优势和mask停止求导，仅current路径参与更新 |

这些形状来自[get_logprobs与calculate_grpo_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)、[load_packed_data_by_index、SequencePacker.pack_sequences与create_packed_seq_params_for_bin](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/sequence_packing_utils.py)。cu_seqlens隔离attention文档，generation/loss mask选择受优化的响应事件；两者需要同时正确。

若复用逐token输出接口表达论文的“每响应等权”目标，必须把同一响应的序列损失除以其有效响应长度后再散回token。下式是一种接口恒等式；Nseq统计有效响应数，不能由packed bin数或token数代替。

```math
\begin{aligned}
\widetilde\ell_{i,t}
&=-\frac{m_{i,t}}{n_i}\phi(s_i,A_i),\\
\mathcal L_{\rm GSPO}
&=\frac1{N_{\rm seq}}\sum_i\sum_t\widetilde\ell_{i,t}
=-\frac1{N_{\rm seq}}\sum_i\phi(s_i,A_i).
\end{aligned}
```

这要求外层执行器按有效响应数完成全局归约，或提供与其严格等价的缩放；直接把sᵢ广播后沿用token平均，会让长响应权重更大。若响应被分到不同CP分片，序列log-ratio分子必须通过可微的跨分片归约得到，分母是同一响应的全局有效数，然后再做exp和clip。bin或microbatch重排应保持同一全局目标，而不改变每响应权重。 参考实现还应在有效位置检查有限logprob，并以足够精度求差与归约；无效位置先安全置零，不能用0×NaN充当mask。

```math
\omega_{i,t}
=\exp(\log\pi_{\rm old}^{\rm train}
-\log\pi_{\rm rollout}^{\rm recorded}),\qquad
\ell^{\rm public}_{i,t}
=-\omega_{i,t}\phi(r_{i,t},A_i)+\beta\,k_{i,t}-\lambda\,h_{i,t}.
```

最后一式对应公开calculate_grpo_loss的可选项：inference_logprobs存在时乘token级推理校正权重，可再截上界；k使用reference差的exp−difference−1，h只计算所选token的−p log p项，不能直接称为完整词表熵。论文§4.2所写GSPO目标没有这些附加项；是否启用、记录的rollout logprob是否包含temperature/top-p变换、正则如何聚合，都必须由真实训练配置说明。它们不是只因同在通用文件中就自动成为Loopie论文配方。（[calculate_grpo_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)）

## 采样参数是否真正到达执行器

要落实采样配置，还须继续追踪字段如何到达引擎。公开GRPO路径中，get_rollout_generator先将四个采样字段装入GroupedRolloutRequest，MegatronLocal.base_generate随后经OpenAI chat接口发送请求；中间转换可能改变或遗漏字段。下表逐项列出固定revision的实际转换，范围限于该公开调用链，不代替尚未公开的论文GSPO launcher。（[rl_utils.get_rollout_generator](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)；[MegatronLocal.base_generate](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/inference/megatron.py)）

| 上游请求字段 | adapter 实际发送 | 服务端与执行含义 |
|---|---|---|
| temperature | request.generation_args.temperature or 1.0 | None 与显式0都变为1；非零值原样传入。服务端原有temperature=0→top_k=1、top_p=0的greedy分支，无法由这个零值请求触发 |
| top_p | request.generation_args.top_p or 0.0 | None／0变为0；在已核Torch路径中0表示不做nucleus筛选 |
| top_k | 未发送，extra_body也没有 | chat服务端取默认0；上游指定17等数值不在此路径生效 |
| max_tokens | 未发送，也未发送max_completion_tokens | 服务端构造num_tokens_to_generate=None；该请求的响应预算转由引擎默认总长度约束决定 |

chat_completions 的参数解析明确将缺失的top_k取0，将缺失的两种max-token字段取None。对合法新请求、未另设num_tokens_total的分支，DynamicInferenceEngine._add_request随后使用剩余上下文容量；EOS仍可提前结束，显存／请求容量检查也仍可能拒绝请求。这里不能把None解释为无限生成。

```math
n_{\rm gen}^{\max}=T_{\rm context}^{\max}-n_{\rm prompt}
\quad\text{当服务端未收到单独的响应长度上限时}.
```

用一个具体请求看这一区别：上游写max_tokens=7，引擎总长度为2048，tokenize后的prompt长100；若响应上限字段未传到服务端，上述默认分支计算出的上限仍是1948，并没有变成7。论文§4.2的32K→64K指最大响应长度，因此既要让该上限真正到达服务端，也要为prompt留出上下文空间。只修改GroupedRolloutRequest，还不能确定实际执行了这一阶段的长度预算。这个例子核对的是字段和默认值，没有运行模型生成。（[论文§4.2](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[chat_completions参数解析](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py)；[DynamicInferenceEngine._add_request](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/engines/dynamic_engine.py)）

## 原始模型概率与采样行为概率

返回的logprob有两种含义，比较前须固定同一合法前缀，以及一致的权重和缓存版本。InferenceConfig与训练侧配置默认raw_logprobs，DynamicInferenceContext._processed_log_probs在此分支直接对未变换logits做log_softmax；processed_logprobs则调用采样后端。

普通非推测、非greedy的Torch采样路径通过同一filter_logits先温度缩放、再做top-k或top-p筛选，并重新归一化。训练侧get_logprobs仍对原始模型logits求所选token的log_softmax，未在这里应用temperature/top-p。（[配置](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/config.py)；[训练侧配置](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/training/config/inference_config.py)；[DynamicInferenceContext._processed_log_probs](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/contexts/dynamic_context.py)；[TorchSampling](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/sampling/torch_sampling.py)）

对固定合法前缀h、有效词表及温度τ>0，令A是按实际采样参数在温度缩放后保留的token集合；在没有额外logits处理器时，这两个单步分布分别是：

```math
\begin{aligned}
p_\theta(v\mid h)
&=\frac{e^{z_{\theta,v}(h)}}{\sum_u e^{z_{\theta,u}(h)}},\\
q_{\theta,\tau,A}(v\mid h)
&=\frac{\mathbf1\{v\in A\}\,e^{z_{\theta,v}(h)/\tau}}
{\sum_{u\in A}e^{z_{\theta,u}(h)/\tau}}.
\end{aligned}
```

raw记录对应p，processed记录在上述路径与条件下对应q。已有公开推理校正项exp(old_logprobs−inference_logprobs)因而有两个不同分母：

```math
\begin{aligned}
\omega^{\rm raw}(y\mid h)
&=\frac{p_{\rm old}^{\rm train}(y\mid h)}
{p_{\rm old}^{\rm infer}(y\mid h)},\\
\omega^{\rm processed}(y\mid h)
&=\frac{p_{\rm old}^{\rm train}(y\mid h)}
{q_{\rm rollout}(y\mid h)}.
\end{aligned}
```

第一项比较同一原始策略在训练／推理后端的概率，不能自动校正温度改变的采样分布。例如两后端原始p均为(0.8,0.2)，τ=2且不截断时q=(2/3,1/3)；raw记录给出权重(1,1)，行为概率作分母则为(1.2,0.6)。使用processed也不能凭单token比值恢复被top-p排除的支持，或自动把逐token校正变成完整响应的换测度。论文实际是否启用校正、何种logprob模式及采样参数，仍不能由这些通用默认值反推。（[get_logprobs与calculate_grpo_loss](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py)）

实施时应依次保存上游参数、实际HTTP字段、服务端SamplingParams、有效总长度／响应上限、logprob模式及策略／缓存版本，再比较同一前缀的原始p与实际行为q。不能只验收最终生成文本。配置还显式拒绝processed_logprobs与num_speculative_tokens>0的组合，不能把普通解码的评分契约直接迁移到推测解码。本轮已用真实方法的孤立mock核对参数运输和长度默认值，用源码AST与两logit纯Python算例核对概率定义；没有启动推理服务或运行checkpoint。（[配置兼容性检查](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/config.py)；[后训练课程](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)）

# 主结果、消融与不能混用的比较

| 问题／条件 | 原文结果 | 解释边界 |
|---|---|---|
| 6B逐层 vs 整栈顺序，图2 | 逐层约1.2T tokens后反超 | 训练早期领先不等于最终领先；图无精确端点表，不制造小数 |
| 20B vs 同数据重训30B-A3B，图3 | baseline训练800B，约600B后Loopie反超；261.53 vs189.65 TFLOP/s | 匹配step时间，不是FLOPs或开放模型最终排行榜 |
| 四档同wall-clock配对，图4、表7 | 八项平均分增益+1.1、+0.6、+1.7、+2.2百分点 | baseline active0.15／0.25／0.51／1.00B；训练150／250／500／500B，末档受算力限额 |
| 250B-token loop-count实验，图6 | 两次的相对收益最好；三／四次回报下降 | 2×stored-layer小模型并非精确成本匹配，作者注明其实际成本明显更高 |
| 最终20B Thinking，表3 | AIME24=92.09、AIME25=83.75、AMC=94.21；Qwen3-30B-A3B为90.10／83.75／96.73 | AIME是avg@8；其余pass@1；不能声称全指标领先 |
| 最终6B，p.21表 | AIME24=80.42、AIME25=70.83、MATH-500=93.80 | 与不同数据／规模／后训练的公开模型比较，只说明最终能力 |

预训练八项平均由 ARC-C/E、BoolQ、CommonsenseQA、HellaSwag、MMLU、OpenBookQA、WinoGrande构成，用 LM Evaluation Harness；Thinking比较用 EvalScope，IFEval采用inst_level_loose。主结果须保留训练阶段和评测协议。

§2.7／图5标“2.14× speedup”，但图注说移除layer-loop，正文又说保持looped computation budget并隔离执行顺序；原文没有把该对照的计算图交代得足够一致。先记录论文报告，不将2.14×改写为已证实的GPU训练吞吐或无循环baseline因果效应。

## 各项性能结果到底估计什么

设 a 同时包含架构和训练配置，ξ 表示训练随机性，V 表示固定评测协议。对 wall-clock 配对，观测的是完整配方的差异：

```math
\begin{aligned}
\widehat\theta_a&=\operatorname{Train}(a,D,\xi),\\
\Delta_{\rm recipe}
&=V(\widehat\theta_{a_1})-V(\widehat\theta_{a_0}),\\
t_{\rm train}(a_1)&\approx t_{\rm train}(a_0).
\end{aligned}
```

图2主要改变循环顺序；图4的配方对照同时改变存储深度、宽度与 microbatch；图6对不同循环数重新训练并调整模型；最终 Thinking 表还含后训练差异。这四类结果分别回答拓扑、预算分配、训练架构选择和最终能力问题。图6的函数是“训练后随配置变化的性能”，不是“固定一个 checkpoint 后随推理轮数变化的性能”：

```math
\begin{aligned}
V_{\rm architecture}(r)&=V\!\left(\operatorname{Train}(a(r),D,\xi),r\right),\\
V_{\rm inference}(r)&=V(\widehat\theta_{r=2},r).
\end{aligned}
```

预训练八任务的百分点差是任务级等权宏平均，而不是把所有题混成一个 accuracy：

```math
\Delta_{\rm macro,pp}
=\frac{100}{8}\sum_{j=1}^{8}
\bigl(\widehat A_{1,j}-\widehat A_{0,j}\bigr).
```

同样，AIME 的 avg@8 对每题八次采样的命中率取平均，与“八次只要一次成功”的 pass@8 不同。用二值正确性 cᵢⱼ 表示：

```math
\begin{aligned}
\operatorname{avg@8}&=\frac1{8N}\sum_{i=1}^N\sum_{j=1}^8c_{ij},\\
\operatorname{pass@8}&=\frac1N\sum_i\mathbf1\!\left\{\sum_jc_{ij}>0\right\}.
\end{aligned}
```

多次采样能减少对固定 checkpoint 采样表现的测量噪声，却不产生独立的训练重复。现有单次训练曲线、八任务平均和 AIME avg@8 都不能直接给出“换一个预训练 seed 后仍领先”的置信区间。（[原文图2、图4、图6，§2.9、§4.3、表3与表7](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)）

# 工程入口：三套实现逐项核验

下面按固定revision列出三套公开实现的文件和调用职责。模型卡未指定论文专用branch；所见Transformers远端只有main，vLLM的其他分支用于依赖更新。因此表中说明的是这些公开版本，不能据此确定作者训练时的全部私有配置或脚本。

| 系统 | revision | 真实文件／函数与职责 |
|---|---|---|
| [Megatron-LM](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_config.py) | cb2daa267d3c5458e75751b3f73a17672fe7a55a | TransformerConfig：layer_loop_times、model_loop_times默认1；layer_loop_reuse_expert默认false |
| [训练模型](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/models/gpt/gpt_model.py) | 同上 | pretrain_gpt.forward_step → GPTModel.forward → TransformerBlock.forward → TransformerLayer.forward；外循环在GPTModel，内循环在TransformerLayer |
| [训练层](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_layer.py) | 同上 | forward每轮更新hidden/context，通知router.set_layer_loop_state，然后_forward_attention、_forward_mlp |
| [checkpoint](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/recompute.py) | 同上 | checkpointed_forward的custom_forward按物理layer调用；完整layer内部含循环，重算边界包住全部访问 |
| [路由](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py) | 同上 | TopKRouter.forward默认重新gating/routing；只有显式reuse配置才复用第一次路由；新出现的UELB选项不能反推论文已用 |
| [Transformers](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | 90ce33bdfdcdd921c9536f4506f4a28cff840db1 | Qwen3MoeModel.forward逐物理层；Qwen3MoeDecoderLayer.forward硬编码range(2)，当前不从layer_loop_times读取 |
| [vLLM](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py) | 9a9703d9918115cac2c6abfdb84aca411468c16c | Qwen3MoeDecoderLayer读取layer_loop_times；_swift_forward_with_attn共享投影但每visit选择独立Attention对象 |

Megatron的pretrain_gpt.loss_func先计算masked loss总和与有效token数，供外层归约；跨microbatch／数据并行的最终归一化不可在适配时重复除法。论文的0.01 MoE辅助项应与router真实求导路径一起检查，不能只看终端CE。

## HF 后训练中的路由辅助目标

论文附录 B、D 将 MoE 辅助项系数列为 0.01，但这不能单独确定导出模型的后训练目标。固定的 [6B config](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview/blob/5fd7647161dec494e37abe08b698cb13434b2c34/config.json) 与 [20B config](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview/blob/652d78bb761afc4ac83e454fe71a02beb3f974cf/config.json) 同时设定 `router_aux_loss_coef=0.01`、`output_router_logits=false`。在已固定的作者 Transformers revision 中，`Qwen3MoeForCausalLM.forward` 先解析调用参数对 config 的覆写，再以该标志决定是否计算 aux；只有同时提供 `labels`，才将 aux 加入返回的 `loss`。把原 `self.loss_function(...)` 记为 `L_labels`，直接调用该入口的契约如下。[原论文](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[forward 实现](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)

| config 与调用 | labels | returned loss | returned aux_loss |
| --- | --- | --- | --- |
| preview 默认 false；调用未覆写 | 有 | L_labels | None |
| 调用显式 output_router_logits=True | 有 | L_labels + 0.01 × aux | 已计算的 aux |
| config 为 true，但调用显式设 false | 有 | L_labels | None |
| 有效标志为 true | 无 | None | 已计算的 aux |

这个判定针对上述直接 `forward` 返回值。自定义 trainer 可能覆写标志、替换 labels loss 或在外部另加辅助项，须沿其调用链判断；这里没有取得论文 SPT／GSPO 专用 launcher，不能反推作者实际漏用该项。HF 的此分支本身没有 `self.training` 检查；返回辅助标量也不表示它已被用于反向传播。

**捕获单位是执行 visit。** `_can_record_outputs` 将 `router_logits` 绑定到 `OutputRecorder(Qwen3MoeTopKRouter,index=0)`。钩子只安装一次，但每次 gate 调用都会向当前前向的 collector 追加输出，结束后转为 tuple，并不按物理模块去重。两份 preview 均为全 MoE 层；在普通完整前向下，逐层 `range(2)` 因而产生 J=2L 个张量，6B／20B 分别为 36／54 个。每个张量对应一个执行位置的全部 token。这个计数限定于已核的普通前向，不声称覆盖 gradient checkpointing 重算或编译替换。来源：[捕获与 collector](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/utils/output_capturing.py)；[router、DecoderLayer 与注册定义](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)

**拼接后求乘积，会改变优化目标。** 令执行位置 ν=1,…,J，每个位置有同一组 n>0 个有效 token、E 个专家、每 token 选 k 个专家。p 是原始 router logits 的完整 softmax，S 是 top-k 集合；f 按全部 kn 次选择归一化，P 按 n 个 token 归一化。此处有效性由辅助函数的二维 `attention_mask` 决定。下面是从固定源码推得的数学表达，不是论文新增定理：

```math
\begin{aligned}
p_{\nu te}&=\operatorname{softmax}(z_{\nu t})_e,
& S_{\nu t}&=\operatorname{TopK}(p_{\nu t},k),\\
f_{\nu e}&=\frac{1}{kn}\sum_{t=1}^{n}\mathbf{1}\{e\in S_{\nu t}\},
& P_{\nu e}&=\frac{1}{n}\sum_{t=1}^{n}p_{\nu te},\\
\bar f_e&=\frac1J\sum_\nu f_{\nu e},
& \bar P_e&=\frac1J\sum_\nu P_{\nu e}.
\end{aligned}
```

`load_balancing_loss_func` 先沿 token 轴 `cat` 所有捕获张量，再分别求选择频率和 softmax 概率均值、相乘求和。它对 k 个选择槽求和，没有再除 k。因此其未乘系数的标量是下面的 pooled 目标；对照项 local 则是保持同一 HF 归一化后，逐执行位置计算再平均。二者之差为位置间协方差，其中 Cov 明确定义为分母 J 的经验协方差：

```math
\begin{aligned}
\ell_{\mathrm{pool}}^{\mathrm{HF}}
&=kE\sum_e\bar f_e\bar P_e
=\frac{kE}{J^2}\sum_{\nu,\mu,e}f_{\nu e}P_{\mu e},\\
\ell_{\mathrm{local}}^{\mathrm{HF}}
&=\frac{kE}{J}\sum_{\nu,e}f_{\nu e}P_{\nu e},\\
\ell_{\mathrm{local}}^{\mathrm{HF}}-\ell_{\mathrm{pool}}^{\mathrm{HF}}
&=kE\sum_e\operatorname{Cov}_{\nu}(f_{\nu e},P_{\nu e}),\\
\operatorname{Cov}_{\nu}(f_{\nu e},P_{\nu e})
&:=\frac1J\sum_{\nu}(f_{\nu e}-\bar f_e)(P_{\nu e}-\bar P_e).
\end{aligned}
```

跨物理层的相同 expert 下标对应不同参数；跨 visit 即使共享参数，路由状态也可不同。因此 pooled 均衡不推出每个执行位置均衡，也不能靠修改一个系数就消掉上式的交叉位置项。两个目标没有普遍大小关系，差值取决于协方差。此处辅助概率由完整 logits 重新 softmax，和 dispatch 时 `norm_topk_prob` 是否把已选专家概率再归一化是两个不同步骤。[HF 辅助函数与 gate](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)

在 top-k 排序不变的邻域内，离散选择不回传导数，pooled 目标对单个 router logit 的偏导为：

```math
\frac{\partial\ell_{\mathrm{pool}}^{\mathrm{HF}}}{\partial z_{\nu tj}}
=\frac{kE}{Jn}\,p_{\nu tj}
\left(\bar f_j-\sum_e\bar f_e p_{\nu te}\right).
```

取 E=2、k=1、J=2、n=1，两位置概率分别为 (0.9,0.1) 与 (0.1,0.9)。pooled=1，而 local=1.8；混合后的 f̄=(0.5,0.5) 使 pooled 对四个 logits 的偏导均为 0，local 的对应偏导为 (0.09,−0.09) 与 (−0.09,0.09)。这解释了整体统计如何掩盖位置差异；它是 logits 的偏导反例，不保证共享权重的各路径梯度一定不相消。

**与 Megatron 对接还需核对 k 的归一化。** 固定 Megatron 默认不复用路由时的普通 softmax `aux_loss` 路径在训练、梯度开启且相应系数启用时，经 `compute_routing_scores_for_aux_loss → _apply_aux_loss → attach_and_log_load_balancing_loss` 按当前 router 调用附着辅助梯度。非 fused 标准函数的分母是 `topk × total_num_tokens²`；在同一位置、相同 token 集与概率、暂不计系数和执行器缩放时，其标量为 E∑fP，HF 同归一化的逐位置标量为 kE∑fP。k=8 且概率均匀时，两者分别为 1 与 8。除系数以 k 只能校准这种标量归一化，不能把 HF 的跨位置 pooled 目标变成逐位置目标；也不能据此声称实际完整训练梯度恰差 8 倍，后者还取决于 visit 聚合、分布式归约及前文的辅助梯度缩放。来源：[router 调用及启用条件](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py)；[完整辅助概率与标准分母](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/moe_utils.py)

适配后训练入口时，依次记录有效标志与 labels 是否存在、collector 的实际 visit 数及映射、每位置 f/P 和 pooled 标量，再决定保留哪个目标及其归一化；trainer 的外加 aux 必须与模型返回项合并核对，防止重复相加。上述启用分支、捕获顺序已用固定源码的原函数体作轻量 mock，协方差恒等式和反例偏导已作纯 Python 数值核验；没有运行 Torch 模型、checkpointing、分布式反向或论文 launcher。

### 四维可见性 mask 与二维辅助 mask 不能复用同一接口

固定HF fork的forward在有效output_router_logits=True时计算aux_loss，labels只决定是否把它加进loss。若捕获到非空router tuple，aux函数随即按“batch_size, sequence_length = attention_mask.shape”解包。故传入[B,1,T,T]四维custom mask时，即使它已被所选attention预处理接受，aux路径仍会形状解包失败；去掉labels不能绕开该分支。两份preview配置默认关闭output_router_logits，不默认触发这一辅助路径。

| 有效output_router_logits | labels | 传入4D mask后的aux分支 |
| --- | --- | --- |
| False | 无／有 | 不调用aux函数 |
| True | 无 | 仍计算aux；shape解包失败 |
| True | 有 | 同样失败；尚未执行加进loss |
| True，改用2D mask作隔离对照 | 无／有 | 可通过该shape解包，后续张量运算仍需完整验证 |

这里有三个不同对象：attention可见性mask决定文档／因果边；二维aux有效token mask决定router统计的分母；response loss mask决定直接CE的监督位置。原aux mask不自动等于response mask，prompt位置不能未经说明就从路由统计中删去。参考适配应在接线中分离可见性和辅助统计，可新增独立aux mask或用明确的hook／外部归约；这些需要实现，不是当前forward已经支持的新参数。不能为通过解包而把4D文档mask降成2D并丢失隔离规则。

检查执行原aux函数体及forward中的aux-if AST，张量／Torch算子均为替身。4D分支在labels有无两种情况下均复现Python解包错误；2D对照只确认已越过该语句，不表示完整前向成功。[固定forward与辅助函数](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) · [同提交mask预处理](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/masking_utils.py)

## 参数共享与执行状态的分离

训练中每个物理层的 Parameter 对象只有一份；每次调用都把梯度贡献累加到它。正常更新的顺序是清零梯度、完成整个全局 batch 的前向与反向、完成必要归约、全局梯度裁剪、执行一次 optimizer step。循环轮次是同一个计算图的内部执行位置，不是额外的 optimizer step。当前 [TransformerLayer.forward](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_layer.py) 中没有 Huginn 式在前若干轮使用 no_grad 的调度。

每次访问的最小追踪键应包含物理层和访问号；路由与激活统计再带上 batch、token 和文档标识。checkpoint 重算会再次调用 forward，收集统计时要把“同一逻辑访问的重算”与“第二次逻辑访问”分开，避免把重算次数当成路由次数。RNG 和其他影响前向的状态必须能在重算时恢复；这是重算产生正确共享梯度的条件，不是保存了 checkpoint 边界便自动满足的性质。

Megatron 的 [router.set_layer_loop_state 与 TopKRouter.forward](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py) 逐访问更新状态；默认 reuse_expert=false，因此第二次重新路由。一次访问选中了八个专家，并不意味着两次访问只执行八次专家计算；若交集为 O，则唯一访问专家数为16−O，而调用次数仍为16。应分别记录这两个数量，以及每轮 load-balancing 辅助项的求导与归约，不能用“共享权重”替代计算账本。

# 逐次访问的缓存与位置

精确增量推理应为每个物理层与访问分别维护历史K/V，因为两次访问的输入隐状态不同。以下是实现解释与应验证的不变量：

```math
\begin{aligned}
\nu(\ell,r)&=R\ell+r\quad(\ell,r\text{ 从0起}),\\
C_{\ell,r}^{(t)}&=\{K_{\ell,r,1:t},V_{\ell,r,1:t}\},\\
|C_{\ell,r}^{(t)}|&=t.
\end{aligned}
```

vLLM 的当前实现正按虚拟索引ν重写Attention prefix，首轮用self_attn.attn，其余用_swift_loop_extra_attns；同一组QKV与o_proj重复计算，但cache对象分离。Attention类按独立prefix注册static_forward_context并绑定KV，重复名字会报错。这是代码可见的设计，不是已经完成端到端正确性实测。

Transformers 的已核 revision／默认 DynamicCache 路径存在具体静态问题：DecoderLayer两次调用同一self_attn；Attention.forward都以相同self.layer_idx调用past_key_values.update；Cache.update按该索引取同一槽，DynamicLayer.update沿序列轴cat。于是一次T-token prefill在该槽会追加两组T，而非建立两份长度T的visit缓存；下次默认position_ids又取该cache长度。此问题仅针对上述revision和路径，不外推所有作者实现，更不能用vLLM的分槽逻辑替它辩护。尚未运行模型，页面记录的是静态证据与待验证项。

后训练入口应先固定6B／20B的config与上述代码revision，用短序列比较“每步重算全前缀”与“带cache逐token”的logits／位置／cache长度；满足不变量后再做SFT或RL。若要学自适应循环，必须同时设计逐层visit调度、mask、router统计与cache分配；不能只把range(2)改成动态数。改变循环数后的性能与停机策略属于新研究任务，原文没有给出保证。

静态证据：[HF Attention／DecoderLayer](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)、[Cache.update／DynamicLayer.update](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/cache_utils.py)、[vLLM双visit构造](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py)、[Attention注册](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/layers/attention/attention.py)。

## 增量缓存正确性的归纳不变量

用 ν 表示展开后的执行位置，用 t 表示文本位置。精确缓存的含义是：槽 ν 中的历史 K/V，必须等于完整前缀计算在同一执行位置得到的 K/V；其长度不应因同一文本 token 的第二次访问而再增加一次。

```math
\begin{aligned}
\mathcal I_t:\quad
C_\nu^{(t)}
&=\{(K^{\rm full}_{\nu,j},V^{\rm full}_{\nu,j})\}_{j\le t},\\
|C_\nu^{(t)}|&=t\quad\text{对无 padding 的单文档序列}.
\end{aligned}
```

归纳证明有两层：假设前 t−1 个 token 的所有槽都正确；再按当前 token 的执行位置顺序，假设上一个位置的隐状态等于完整前缀结果。相同的物理权重、文本位置和归一化便产生相同当前 Q/K/V；写入正确槽后，在相同 causal／document mask 下读取相同历史，得到相同 Attention 输出；逐 token 路由与专家计算也相同。因此当前隐状态正确，依次推进至最终 logits，并建立下一 token 所需的不变量。

该论证要求参数与位置变换固定，评测中的随机性受控，router 不因其他 batch token 触发容量丢弃，且各执行位置的历史缓存已经就绪。在这些条件下证明的是精确算术语义；不同 kernel 的浮点加和顺序仍需按容差比较，不能要求必然逐 bit 相同。多文档 packing 应把 token 序号、位置和 segment mask 分别保存，不能从一个槽的物理长度猜所有 position_ids。

当前 vLLM 的双 visit Attention 槽满足这种地址分离的设计意图；当前 HF 默认 DynamicCache 路径的问题则是在地址上把 ν 压回物理层 ℓ。下面给出 vLLM 计算语义的去融合算法，展示共享投影与分离缓存如何同时成立；它不是已完成的端到端运行验证。（[Qwen3MoeAttention._swift_forward_with_attn 与 DecoderLayer.forward](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py)；[HF Cache.update](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/cache_utils.py)）

### 算法 共享两次逐层访问的增量前向

**输入** 当前 token、固定文本位置、已验证的前缀缓存、物理层权重与 preview 配置。

**输出** 下一 token logits，以及各虚拟执行位置更新后的缓存。

**执行前提** 算法展示单条无 padding 文档；所有槽起始长度为 t−1。；固定参数、R=2、确定的 top-k tie 规则及相同 causal mask。

![算法 共享两次逐层访问的增量前向](visualize:fde1_bGliZmlsZV84czBmWEM3S21sVmVMZmRNZmJKU1pn_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 据 vLLM commit 9a9703d9918115cac2c6abfdb84aca411468c16c 的 Qwen3MoeDecoderLayer.forward、_swift_forward_with_attn 去融合重写；保留共享权重但分开 Attention/cache 对象。

# 逐层深度选择需要保留后缀计算和层间交互

若逐层访问次数写成向量 r=(r₁,…,r_L)，前向是按层有序复合的函数。判断某层第二次访问的最终预测收益，必须经过后续层得到读出；直接将该层 hidden 当作 TaH2 的末层输出，会改变标签含义。

固定权重、样本、位置及其余层访问策略，以完整后缀重算定义收益。下面的 R 包括最终读出与 softmax，F 是单层访问的概念算子；增量实现仍保留每个虚拟访问的缓存状态：

```math
\begin{aligned}
q_{\boldsymbol r}(x)&=R_\theta\circ F_L^{r_L}\circ\cdots\circ F_1^{r_1}(E_\theta(x)),\\
\mathcal L(\boldsymbol r)&=-\log q_{\boldsymbol r}(x)_y,\\
\Delta_\ell(\boldsymbol r_{-\ell})
&=\mathcal L(r_\ell=1,\boldsymbol r_{-\ell})-\mathcal L(r_\ell=2,\boldsymbol r_{-\ell}).
\end{aligned}
```

两个层的交互由四个受控角点刻画：

```math
I_{\ell j}=\mathcal L(1,1)-\mathcal L(2,1)-\mathcal L(1,2)+\mathcal L(2,2).
```

在 j=1 与 j=2 条件下，第二次访问 ℓ 的收益之差恰为 I_(ℓj)。有交互时，固定路径上测得的单层收益排序不能推广到所有路径。即使额外假设收益可加且成本可加，离散预算选择通常仍是 0–1 背包；按收益取前若干项还需等成本等条件。

可用少量预先指定的层对做四角诊断，四条路径各自重建后缀与缓存，并记录实际访问量。此处是后训练诊断设计，现有 Loopie 的变深度执行仍需单独实现。

来源：前文固定 Megatron／Transformers／vLLM 调用链及 [TaH2 末层深度监督](https://arxiv.org/html/2609.35748v1#S4.SS2)；后缀重算遵循 L_CACHE 的缓存不变量。

# 接入前需要补齐的条件

原文、模型卡、配置和源码已逐项交叉阅读；关键结果表已核渲染图。完成的是文献与静态实现审计，尚未加载checkpoint、运行训练或测量真实解码延迟。

- 发布身份：请以作者明确的checkpoint阶段／训练日志确认preview是否适合从Base起做后训练；仅凭名称和指标不能确定。

- 原文口径：Stage1与总token预算口径，以及图5对照的完整执行图仍需作者解释。

- 运行正确性：对已固定HF默认cache路径补逐visit分槽后，须检验全前缀与增量logits一致；也须独立验收vLLM路径，源码结构合理不等于运行已经通过。

- 配方复现：SPT的loss mask／packing、RL实际超参、硬件网格与step计时日志尚未取得。通用库里有功能不等于论文训练配置已公开。

课程优先顺序是先把逐层共享计算图与两个preview配置讲透，再用固定两次循环完成可运行参考，之后研究SFT／RL能否改善回答行为；动态循环或提前停止须另设预算与缓存验收标准。

# 固定版本的文件与函数索引

来源：[github.com/zitian-gao/Megatron-LM](https://github.com/zitian-gao/Megatron-LM)；固定 revision：`cb2daa267d3c5458e75751b3f73a17672fe7a55a`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [pretrain_gpt.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/pretrain_gpt.py) | forward_step; loss_func | 训练入口和masked loss；取tokens/labels/loss_mask，调用model；loss_func返回loss sum与num_tokens供外层归约 |
| [megatron/core/transformer/transformer_config.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_config.py) | TransformerConfig | 循环配置；layer_loop_times=1，model_loop_times=1，layer_loop_reuse_expert=False；后续存在UELB选项但不是论文证据 |
| [megatron/core/models/gpt/gpt_model.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/models/gpt/gpt_model.py) | GPTModel.forward; set_decoder_model_loop_state | 整模型循环；循环调用self.decoder，设置router model-loop state；Loopie preview model_loop_times=1 |
| [megatron/core/transformer/transformer_block.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_block.py) | TransformerBlock.forward | 物理层遍历和checkpoint入口；full recompute训练路径调用checkpointed_forward；普通路径逐self.layers调用 |
| [megatron/core/recompute.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/recompute.py) | checkpointed_forward; custom_forward; chunk_runner | 激活重计算边界；按物理layer调用，整个TransformerLayer.forward含所有visit；uniform/block决定checkpoint chunk |
| [megatron/core/transformer/transformer_layer.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/transformer_layer.py) | TransformerLayer.forward; _forward_attention; _forward_mlp | 逐层循环；range(layer_loop_times)；每次设置router状态、更新hidden/context、再注意力及MoE |
| [megatron/core/transformer/moe/router.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/router.py) | Router.set_layer_loop_state; TopKRouter.forward; _apply_aux_loss | 跨visit路由与aux loss；默认每visit重新gating/routing；仅显式reuse_expert时读首visit的cache；aux coefficient由配置驱动 |
| [megatron/core/pipeline_parallel/schedules.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/pipeline_parallel/schedules.py) | forward_step_calc_loss | calculate_per_token_loss=false时按局部标签数及microbatch数归一化；true保留分子；设置MoE辅助梯度缩放 |
| [megatron/core/distributed/distributed_data_parallel.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/distributed/distributed_data_parallel.py) | DistributedDataParallel | per-token路径梯度预缩放为1，禁止collective平均；普通路径补足副本平均 |
| [megatron/core/distributed/finalize_model_grads.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/distributed/finalize_model_grads.py) | finalize_model_grads | finish_grad_sync及相关归约后，广播并求和token count，再统一缩放梯度 |
| [megatron/training/training.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/training/training.py) | train_step | optimizer.step成功才推进opt_param_scheduler；increment按全局sample数计 |
| [train_rl.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/train_rl.py) | forward_step; loss_func | current logprobs重算；逐token GRPO调用；shifted生成mask与masked sum；不是论文GSPO入口证明 |
| [megatron/rl/rl_utils.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/rl_utils.py) | get_grpo_data_iterator; get_environment_rollouts; prepare_data_for_update; get_logprobs; calculate_grpo_advantages; calculate_grpo_loss | rollout/refit/buffer调度、old/ref无梯度重放、总体std优势、token ratio与clip及可选IS/KL/所选token熵项 |
| [megatron/rl/sequence_packing_utils.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/sequence_packing_utils.py) | load_packed_data_by_index; SequencePacker.pack_sequences; create_packed_seq_params_for_bin | tokens[1,K]与logprobs[1,K−1]；mask右移；完整sequence长度、位置重置与cu_seqlens隔离 |
| [megatron/rl/agent/weighted_multi_task.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/weighted_multi_task.py) | WeightedMultiTask.get_grouped_rollouts | 将过滤flag转发给子agent；未替代子pipeline的补样职责 |
| [megatron/rl/agent/api.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/agent/api.py) | GroupedRolloutGenerator; _RolloutPipeline; _GranularityConfig._validate | 示例RewardOnlyAgent继承的pipeline拒绝same-reward过滤；过滤后补样未实现，std过滤标记为不可达 |
| [megatron/rl/inference/megatron.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/rl/inference/megatron.py) | MegatronLocal.base_generate | OpenAI请求仅转发temperature/top_p；temperature=0被or 1.0替换；未发送top_k及max_tokens |
| [dynamic_text_gen_server/endpoints/chat_completions.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/text_generation_server/dynamic_text_gen_server/endpoints/chat_completions.py) | chat-completions参数解析 | 缺失top_k默认0；两种max-token字段均缺失则响应数None；直接temperature=0可转greedy，但上述adapter已将0改写 |
| [megatron/core/inference/engines/dynamic_engine.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/engines/dynamic_engine.py) | DynamicInferenceEngine._add_request | 未设num_tokens_total且响应数None时，回退至max_sequence_length−prompt length；另做EOS与容量检查 |
| [megatron/core/inference/config.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/config.py)；[training/config/inference_config.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/training/config/inference_config.py) | logprobs_mode；inference_dynamic_batching_logprobs_mode | 默认raw_logprobs；processed与num_speculative_tokens>0组合被明确拒绝 |
| [megatron/core/inference/contexts/dynamic_context.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/contexts/dynamic_context.py) | _processed_log_probs；calculate_log_probs | raw对原始logits归一化；processed按请求映射temperature/top-k/top-p并调用采样后端 |
| [megatron/core/inference/sampling/torch_sampling.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/inference/sampling/torch_sampling.py) | filter_logits；sample_from_logits；log_probs_kernel | 普通非greedy路径共享温度→筛选定义，再softmax或log_softmax；实现分支不等于作者论文启用配置 |
| [megatron/core/transformer/moe/moe_utils.py](https://github.com/zitian-gao/Megatron-LM/blob/cb2daa267d3c5458e75751b3f73a17672fe7a55a/megatron/core/transformer/moe/moe_utils.py) | compute_routing_scores_for_aux_loss; switch_load_balancing_loss_func | 普通 softmax 辅助概率为全专家 softmax；非 fused 标准分母含 topk×total_num_tokens²；与 HF pooled 目标和归一化分别核对 |

来源：[github.com/zitian-gao/transformers](https://github.com/zitian-gao/transformers)；固定 revision：`90ce33bdfdcdd921c9536f4506f4a28cff840db1`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [src/transformers/models/qwen3_moe/modeling_qwen3_moe.py](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | Qwen3MoeModel.forward; Qwen3MoeDecoderLayer.forward | 推理计算图；物理ModuleList；DecoderLayer硬编码range(2)；同RMSNorm/Attention/MoE重复使用 |
| [src/transformers/models/qwen3_moe/modeling_qwen3_moe.py](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | Qwen3MoeAttention.forward | QKV/RoPE/cache写入；显式head_dim；QK norm；past_key_values.update使用相同self.layer_idx，没有visit字段 |
| [src/transformers/cache_utils.py](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/cache_utils.py) | Cache.update; DynamicLayer.update; DynamicLayer.get_seq_length | 静态cache审计；Cache按layer_idx取槽；DynamicLayer沿-2轴torch.cat；在上述同索引两次调用的默认路径混合visit并扩大序列长度。未做运行复现。 |
| [src/transformers/models/qwen3_moe/modeling_qwen3_moe.py](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py) | Qwen3MoeForCausalLM.forward; load_balancing_loss_func | 有效 output_router_logits 控制 aux 计算，labels 控制是否加进 loss；先 cat 全 visit 再乘 pooled 均值，未除 top-k |
| [src/transformers/utils/output_capturing.py](https://github.com/zitian-gao/transformers/blob/90ce33bdfdcdd921c9536f4506f4a28cff840db1/src/transformers/utils/output_capturing.py) | OutputRecorder; install_output_capuring_hook; capture_outputs | index=0 捕获 router logits；逐次 gate 调用 append；普通全 MoE 前向记录 2L visits；每次前向独立 collector |

来源：[github.com/zitian-gao/vllm](https://github.com/zitian-gao/vllm)；固定 revision：`9a9703d9918115cac2c6abfdb84aca411468c16c`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [vllm/model_executor/models/qwen3_moe.py](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py) | _swift_replace_layer_index; Qwen3MoeDecoderLayer.__init__; Qwen3MoeDecoderLayer.forward | 每visit独立cache与共享权重；virtual_idx=layer_idx*layer_loop_times+loop_idx；首轮attn，其余_swift_loop_extra_attns；prefix不同 |
| [vllm/model_executor/models/qwen3_moe.py](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/models/qwen3_moe.py) | Qwen3MoeAttention._swift_forward_with_attn | 共享投影/位置，切换attention cache；每visit重新QKV、QK norm、RoPE，传入选定Attention模块，输出共享o_proj |
| [vllm/model_executor/layers/attention/attention.py](https://github.com/zitian-gao/vllm/blob/9a9703d9918115cac2c6abfdb84aca411468c16c/vllm/model_executor/layers/attention/attention.py) | Attention.__init__ | cache注册；layer_name=prefix；在static_forward_context注册，不允许duplicate；kv_cache后续绑定 |

来源：[huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview)；固定 revision：`5fd7647161dec494e37abe08b698cb13434b2c34`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [config.json](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview/blob/5fd7647161dec494e37abe08b698cb13434b2c34/config.json) | 模型配置 | 发布checkpoint契约；layer_loop_times2；model_loop_times1；hidden1536；layers18；head_dim128；norm_topk_prob true；tie_word_embeddings false；YaRN16；output_router_logits false；router_aux_loss_coef 0.01 |
| [README.md](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview/blob/5fd7647161dec494e37abe08b698cb13434b2c34/README.md) | 模型卡 | 发布用途/安装/指标；数学科学推理preview；指向author transformers/vllm，无专门branch，无明确Base/SPT/RL阶段标签 |

来源：[huggingface.co/zitian-gao/Loopie-20B-A2B-preview](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview)；固定 revision：`652d78bb761afc4ac83e454fe71a02beb3f974cf`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [config.json](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview/blob/652d78bb761afc4ac83e454fe71a02beb3f974cf/config.json) | 模型配置 | 发布checkpoint契约；layer_loop_times2；model_loop_times1；hidden2304；layers27；head_dim64；norm_topk_prob false；tie_word_embeddings true；YaRN16；output_router_logits false；router_aux_loss_coef 0.01 |
| [README.md](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview/blob/652d78bb761afc4ac83e454fe71a02beb3f974cf/README.md) | 模型卡 | 发布用途/安装/指标；同系列preview，未明确具体训练阶段 |

# 逐项覆盖核对

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 原生预训练与逐层拓扑 | §2.1 p.4；§3 p.13；附录A表4 p.35 | Megatron TransformerLayer.forward；HF Qwen3MoeDecoderLayer.forward；vLLM对应类 | 已核原文与相关实现 |
| 物理参数、head geometry、MoE概率与上下文 | 附录A表4；preview config | 两份HF config已固定SHA；explicit head_dim128/64、norm_topk_prob true/false、YaRN16、tie embeddings差异 | 已核原文与相关实现 |
| 硬件预算与checkpoint | §2.4–2.5 pp.6–10；表1、图3 | TransformerBlock.forward→core.recompute.checkpointed_forward包住含loop的物理层；数值为论文报告，未重复GPU计时 | 已核原文与相关实现 |
| 初始化与优化超参 | §3.2–3.3 p.13；附录B表5–6 pp.36–37 | 论文recipe已落实；未取得作者精确启动脚本，当前通用训练入口已核 | 原文已核；完整复现实验配置待补 |
| 数据比例与token预算 | §3.4–3.5 pp.14–15；表3 p.20 | 3T vs570B×4≈2.28T vs总3.5T并列登记 | 原文口径冲突待作者澄清 |
| SPT loss mask、batch与token accounting | §4.1 pp.16–18；附录D表8 p.38 | pretrain_gpt.loss_func给出masked sum与token count；SPT具体数据管道/脚本未核 | 目标/超参已核；特定SPT脚本未取得 |
| GSPO/DAPO与领域/长度阶段 | §4.2 pp.18–19 | 写明序列ratio、组advantage、非退化prompt过滤、数学→代码、32K→64K | 原文已核；论文RL特定脚本未取得 |
| 结果、顺序/深度消融与评测协议 | 图2–6、表3、p.21表、§3.1/4.3 | 关键表格已对照同version PDF渲染图；§2.7歧义保留 | 原文已核；图5计算图定义未明 |
| 训练router及可选路径 | §2.1；附录B/D auxiliary coefficient | router.TopKRouter.forward重新gating/routing；reuse_expert默认为false；UELB不反推论文使用 | 已核原文与相关实现 |
| 增量KV语义 | §6声明推理计算未系统研究 | HF默认DynamicCache同槽双追加风险；vLLM不同virtual prefix分cache，均限已核revision | 静态源码已核；未运行端到端模型 |
| Base/Thinking阶段与preview资格 | 图8 p.15；模型卡README | 模型卡未明确preview具体训练阶段 | 阶段身份待作者标注 |
| loss 归约与辅助梯度 | §4.1 全局 CE；固定 Megatron revision | forward_step_calc_loss 的 false/true 分支、DDP 缩放、finalize_model_grads、router 有效数均已核；0.5 对−0.7反例 | 代码分支与代数已核；论文实际开关未公开 |
| 响应 mask 与 prompt 梯度 | §4.1；GPTModel.forward／loss_func | loss_mask 删除直接 CE，不删除条件 K/V 的反向路径；标明 detach 的目标变化 | 链式法则与标量有限差分已核 |
| 全阶段 token 与 update 预算 | §3–4、表3、附录B/D | nominal／seen／supervised／unique 分开；约14,901 SPT更新只在指定口径下成立 | 原文与条件算术已核；完整训练日志待补 |
| 论文GSPO与公开通用GRPO调用链 | §4.2；examples/rl/README；固定Megatron revision | training→rollout/refit→old/ref重放→current→calculate_grpo_loss；token ratio/优势广播/附加项逐项区分 | 真实路径与小算例已核；未取得论文专用GSPO launcher |
| 动态过滤与序列目标的迁移条件 | §4.2；agent/api.py；sequence_packing_utils.py | _validate拒绝过滤flag；补样职责、response ID、shifted mask、CP序列归约与每响应分母明确 | 孤立源码guard及packed toy核验通过；未运行训练/模型 |
| rollout参数运输与响应长度 | §4.2最大响应32K→64K；固定Megatron revision | GroupedRolloutRequest→MegatronLocal→chat-completions→_add_request；0温度改写、top_k/max_tokens未发送及剩余总长度默认值 | 孤立真实方法mock通过；没有运行服务器或checkpoint |
| raw与processed推理logprob | 固定Megatron配置、DynamicInferenceContext、TorchSampling、rl_utils | 默认raw；非推测非greedy路径p与q区分；训练侧原始logprob及IS两种分母；processed推测解码禁配 | AST与两logit纯Python算例通过；没有执行Torch sampler或模型 |
| HF 后训练辅助目标的启用与归约 | 论文附录B/D系数0.01；固定 preview configs／Transformers／Megatron | 默认 output_router_logits=false；显式开启且 labels 非空才加进返回 loss；2L visit 捕获、pooled/local 协方差差值及 top-k 归一化 | 原函数体轻量 mock、代数与有限差分通过；无 Torch／模型／论文 launcher 运行 |
| SPT／SFT／PT完整原表与曲线 | Table2、Figs.9–11，pp.16–18 | 完整loss／监督位置／规模／观察记录；引用型SFT列与单因素比较分开 | 原文各列已核；没有补造SPT因子消融 |
