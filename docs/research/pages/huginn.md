[基座矩阵与七篇论文](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [预训练](https://chatgpt.com/space/page_4f0508753a6081919a82b0c89213fd6c) · [后训练](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

# Huginn：把共享循环核、训练动力学与推理缓存连成一条链

**Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach，2502.05171v2，37页**研究从预训练开始采用循环深度的 Dense 基座：Prelude–共享 Core–Coda，约3.5B独立参数、800B训练tokens。建议先读结构，再看随机深度与截断反传，随后比较结果，最后核对推理缓存和实现边界。沿这条顺序，始终区分独立参数与循环调用次数，以及经验上的路径独立与数学上的收缩保证；论文、发布代码和推导各自标明依据。

**原文**：[上传英文PDF](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal)。**官方训练仓库**：[recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e)。**发布模型与推理代码**：[huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c)。训练与推理实现分别采用以上两个revision；权重载入和大模型运行尚未验收。

## 结构：固定输入、随机初态、共享迭代算子

令输入为长度n的token序列，隐藏宽度h=5280。Prelude和Coda各2层，Core有4层；55个注意力头，每头96维；SwiGLU中间宽度17920；词表65536；RoPE base=50000；RMSNorm epsilon=10⁻⁶。注意力为MHA，Q、K有可学习bias，其他主要线性层无bias。输入embedding和输出词表投影共享权重。论文的宏观图是：

```math
\begin{aligned}
e&=P_\theta(\sqrt h\,E(x)),\\
s_{0,j}&\overset{\rm iid}{\sim}\operatorname{TN}(0,\sigma_s^2;[-3\sigma_s,3\sigma_s]),\\
s_r&=R_\theta(e,s_{r-1}),\\
p_\theta(\cdot\mid x;R)&=\operatorname{softmax}(C_\theta(s_R)).
\end{aligned}
```

沿一次前向看，Prelude先把token序列变成固定表示e；Core随后接收随机初态s₀和e，产生s₁。下一轮送回Core的是s₁，同时再次送入同一个e。变化的是正在迭代的状态，始终保留的是输入条件；轮与轮之间没有采样token，也没有把词表logits重新送入embedding。

源码把这两份信息沿隐藏维拼接，再经无bias的2h→h线性adapter压回h维。每轮复用同一组4层Core，因而R=32时展开深度为2+4R+2=132层，物理参数仍只有首尾各2层、Core 4层及adapter等模块。这正是共享循环增加计算深度的方式。[原文§3.1–3.2，pp.3–4；发布config.json]

Huginn的SandwichBlock明确把**残差和**归一化：

```math
\begin{aligned}
u&=N_2\!\left(z+\operatorname{Attn}(N_1(z))\right),\\
B(z)&=N_4\!\left(u+\operatorname{MLP}(N_3(u))\right).
\end{aligned}
```

归一化的位置决定了计算图，因此比较不同模型时要展开具体顺序。官方训练与HF推理实现都采用上式。论文§3.2另称Core末有额外的n_c；发布的RecurrentGPT.core_block_forward和RavenForCausalLM.core_block_forward则以每个SandwichBlock的norm_4结束本轮，不在每轮末调用独立ln_f。**整段循环完成后**，iterate_forward调用ln_f，Coda后再调用同一个ln_f。复现以这些调用位置为准，不能仅依据论文文字在所有循环末插入额外norm。[训练代码 recpre/model_dynamic.py；HF raven_modeling_minimal.py]

这种结构面对一个直接的问题：迭代应当减少对偶然初态的依赖，却不能连输入问题一起忘掉。Huginn把两者分开处理。随机初态使模型在训练时遇到不同起点；每轮重新注入e，使输入条件不必仅靠最初状态经过许多轮传递。Prelude和Coda分别承担输入表示与输出读出，也就不要求共享Core同时完成这两种不同功能。

因此，把Core理解为受输入e约束的迭代算子是有用的设计解释。它尚不等于一个已证明收敛的求解算法：论文没有证明真实网络在最小化凸能量，也没有给出全局小于1的Lipschitz常数。脚注针对特定算子和优化类比的稳定性讨论，也不涵盖所有无输入注入的循环模型。

### 推导 输入注入与初态遗忘可以同时成立

要看清“忘记起点”与“保留输入”怎样同时发生，可以比较两种微小扰动：一次只改变s₀，另一次只改变e。把完整因果前缀的所有位置展平成一个状态向量，并保持位置与mask不变。adapter的状态部分Aₛ与输入部分Aₑ便对应这两条入口；下面的Jᵣ、Kᵣ分别衡量一轮计算对旧状态和输入的敏感度。虽然参数共享，这些导数在每轮实际状态处求值，数值可以不同：

```math
\begin{aligned}
A[s;e]&=A_s s+A_e e,\\
F_\theta(s;e)&=(B_4\circ B_3\circ B_2\circ B_1)(A_s s+A_e e),\\
J_r&=\partial_s F_\theta(s_{r-1};e)
=J_{B_4,r}J_{B_3,r}J_{B_2,r}J_{B_1,r}A_s,\\
K_r&=\partial_e F_\theta(s_{r-1};e)
=J_{B_4,r}J_{B_3,r}J_{B_2,r}J_{B_1,r}A_e.
\end{aligned}
```

只扰动初态时，扰动进入第一轮后，只能沿状态链继续传播，所以经过R轮会乘上全部J。改变输入时则不同：第一轮注入的影响会继续传播，第二轮又有新的影响进入，直到第R轮。因此输入导数是各次注入贡献之和，而不是只有一条穿过全部轮次的长乘积。初态与输入独立、空矩阵乘积取恒等映射时，这个区别写成：

```math
\begin{aligned}
\frac{\partial s_R}{\partial s_0}&=J_RJ_{R-1}\cdots J_1,\\
\frac{\partial s_R}{\partial e}
&=\sum_{j=1}^{R}(J_R\cdots J_{j+1})K_j.
\end{aligned}
```

若另行验证：某个闭合、完备且对迭代不变的状态域内，固定输入时算子一致收缩，常数为 ρ<1；那么不同初态会以 ρ 的幂次接近唯一固定点。若该固定点可微，其输入敏感度仍可非零：

```math
\begin{aligned}
\|s_R^{(a)}-s_R^{(b)}\|&\le \rho^R\|s_0^{(a)}-s_0^{(b)}\|,\\
\frac{\partial s_\ast}{\partial e}
&=(I-J_\ast)^{-1}K_\ast
=\sum_{j=0}^{\infty}J_\ast^jK_\ast,\\
\left\|\frac{\partial s_\ast}{\partial e}\right\|
&\le \frac{\|K_\ast\|}{1-\rho}.
\end{aligned}
```

上式给出了一种可能的工作方式：状态链逐渐压低起点差异，但新的输入作用不断补入，固定点仍可随问题改变。这一解释依赖前述不变域和一致收缩条件；论文中的path independence是实验观察，没有验证这些条件。论文还观察到轨道和sliders，因此真实状态演化也可能不以固定点结束。实现中可直接找到这两条入口：[原文 §3、§7；以上为条件性推导。源码对应 `core_block_forward` 的拼接与共享 Core，见固定版本 [model_dynamic.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)。]

## 初始化与尺度：稳定性来自联合配方

论文使用在±3σ截断的正态初始化。基准方差为2/(5h)，输出投影方差为1/(5hL_eff)，embedding乘√h；初态按同一embedding尺度生成，从而在截断前具有2/5的目标方差。这里L_eff=132，是按配置r̄=32得到的参照展开深度，不是物理层数8，也不是含+1采样分布的精确期望；E[R]=33时E[2+4R+2]=136。[原文§4.1，p.5，已核页面图像]

```math
\begin{aligned}
\sigma_{\rm base}^2&=\frac{2}{5h},&
\sigma_{\rm out}^2&=\frac{1}{5hL_{\rm eff}},\\
\gamma_{\rm emb}&=\sqrt h,&
\sigma_{s_0,\rm nominal}^2&=\frac25.
\end{aligned}
```

工程实现要继续往下读：HF的_init_func给出基准尺度；_init_weights按模块名派发初始化；init_glu对合并的两半MLP输入权重分别用基准尺度和较小的out_proj尺度，不能只用一条“所有矩阵同分布”替代代码。initialize_state先采截断正态再乘embedding scale；init_scale=0分支得到零初态，不代表主训练使用零初始化。

论文§4.3的失败运行很关键：参数自由RMSNorm、无embedding scale、加法注入与较大学习率的组合，虽然早期有进展，却在约50B tokens后出现不利用深循环的退化；最终改为可学习norm、缩放embedding、拼接adapter并降低学习率后继续稳定训练。它支持“这套联合修复在该规模有效”，不能把同时变化的多因素强行解释成一个干净的单变量消融。[原文p.6，Fig.5–6]

### 归一化与迭代雅可比

对一个 token 的 d 维向量，令 RMSNorm 的可学习缩放为 γ，epsilon 为 ε。去掉不影响导数的可选常数平移，定义 a(x) 为 RMS 分母；精确雅可比为

```math
\begin{aligned}
a(x)&=\sqrt{d^{-1}\|x\|_2^2+\varepsilon},\\
N_\gamma(x)&=\operatorname{Diag}(\gamma)\frac{x}{a(x)},\\
J_N(x)&=\operatorname{Diag}(\gamma)
\left[\frac{I}{a(x)}-\frac{xx^\top}{d\,a(x)^3}\right],\\
\|J_N(x)\|_2&\le \frac{\|\gamma\|_\infty}{a(x)}.
\end{aligned}
```

在 γ 全为 1 时，切向方向的特征值是 1/a，径向方向是 ε/a³。归一化可以压低径向变化，却可在输入 RMS 很小时放大切向扰动；可学习 γ 又会改变尺度。令 A、M 分别为注意力与 MLP，则 Huginn 一个 block 的链式法则是

```math
J_B
=J_{N_4}(I+J_MJ_{N_3})J_{N_2}(I+J_AJ_{N_1}).
```

所有因子需在各自中间激活处求值，不能把四个 RMSNorm 都替换为一个固定常数。一个训练配置中保持隐藏 RMS 平稳，不足以推出多轮雅可比乘积有界；norm 位置、adapter、激活尺度和学习率需要联合分析。[`SandwichBlock.forward`，固定版本 [训练代码](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)；原文 §4.3 的联合修复与 Fig.5–6。]

初始化还应区分“正态分布的参数方差”与“截断后实际方差”。前面 TN 记号表示每个坐标在给定区间截断的正态，标称 σ_s²=2/5。代码对每个坐标做 ±3σ 截断，而非裁剪已有样本；记 φ、Φ 为标准正态密度和分布函数，则

```math
\begin{aligned}
\operatorname{Var}(Z\mid |Z|\le3)
&=1-\frac{6\varphi(3)}{2\Phi(3)-1}
\approx0.9733369,\\
\operatorname{Var}(s_{0,j})
&=\frac25\left[1-\frac{6\varphi(3)}{2\Phi(3)-1}\right]
\approx0.3893348.
\end{aligned}
```

0.4 是截断前的标称初态方差；上式给出忽略有限精度的实际分布方差。输出投影按参照展开深度 132 缩放，是初始化尺度选择，不是后来随机展开深度的精确均值，也不能据此消去注意力和 MLP 更新之间的协方差。[原文 §4.1；`initialize_state` 与 [HF 初始化路径](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)。]

## 训练目标：随机前向深度与截断梯度分别定义

训练时，模型不会总在同一个轮次接受监督：每个样本先抽一个前向深度R，再在第R轮经Coda预测下一个token。这样，目标对数据、随机起点和计算深度共同取期望。与此分开，训练还要决定为多少轮保存反向图；源码把这个数限制为k=min(8,R)。采样分布与任务目标分别写为：

```math
\begin{aligned}
Z&\sim\mathcal N(0,1),\qquad \sigma=0.5,\\
\lambda&=\exp\!\left(\log32-\frac{\sigma^2}{2}+\sigma Z\right),\\
R&=1+\operatorname{Poisson}(\lambda),\\
k&=\min(8,R),\qquad n=R-k,\\
\mathcal L(\theta)&=
\mathbb E_{x,R,s_0}
\left[-\sum_i\log p_\theta(x_{i+1}\mid x_{\le i};R,s_0)\right].
\end{aligned}
```

原文写mean_recurrence=32，但分布有“+1”，因此这个实现的数学期望是33；Fig.3同时显示均值33、中位数29、众数24。不能把配置字段32直接说成精确采样均值。[原文§3.3，p.4；randomized_iteration_sampler]

例如这次抽到R=32，模型会先算24轮，再算末8轮，最后只在末轮读出一个LM loss。前24轮确实参与了状态计算，但在no_grad中运行；到第24轮时，所得状态被当作本次反传的固定边界。末8轮才保存反向图。因此，训练看见的是完整32轮的前向结果，反向信用分配只穿过最后8轮。这里没有每轮各附一个辅助CE。用stop-gradient标出这个边界，实际求导过程是：

```math
\begin{aligned}
\bar s_n&=\operatorname{sg}\!\left(R_\theta^{\,n}(e,s_0)\right),\\
\widehat{\nabla_\theta\mathcal L}
&=\nabla_\theta\ell
\left(C_\theta\!\left(R_\theta^{\,k}(e,\bar s_n)\right),y\right).
\end{aligned}
```

这解释了随机深度目标与实际更新之间的区别：期望loss评价完整R轮预测，更新却忽略了边界状态怎样随参数变化，因而一般是该目标梯度的有偏近似。Prelude并未完全失去训练信号，因为同一个e仍注入末k轮，这些注入路径留在反向图中。

这项节省发生在反传激活的保存上：循环部分主要随k增长。前向仍须完成R轮，R增加时FLOPs和时间也随之增加。[论文§3.3；我们对截断计算图的表述]

### 推导 随机深度的计算预算与截断梯度的遗漏项

令 m=32、σ=0.5、R=1+Pois(λ)，λ 为前述 lognormal 随机变量。由全期望和全方差公式，

```math
\begin{aligned}
\mathbb E[\lambda]&=m,&
\operatorname{Var}(\lambda)&=m^2(e^{\sigma^2}-1),\\
\mathbb E[R]&=1+m=33,&
\operatorname{Var}(R)&=m+m^2(e^{\sigma^2}-1)\approx322.8420,\\
\operatorname{sd}(R)&\approx17.9678,&
\mathbb E[2+4R+2]&=136.
\end{aligned}
```

因此该分布不仅“平均约 32 轮”，还具有显著深度波动。保存反传图的深度为 k=min(8,R)，精确均值可写成尾概率之和；在每轮前向和反向代价近似固定时，预算应分别记账：

```math
\begin{aligned}
\mathbb E[k]&=\sum_{j=1}^{8}\Pr(R\ge j)\le8,\\
\mathbb E[C_{\rm core}]&\approx33c_{\rm fwd}+\mathbb E[k]c_{\rm bwd}.
\end{aligned}
```

这里未计 checkpoint 重算；重算会增加前向项。lognormal 混合的正向矩母函数发散，不能不加论证地套用普通 Poisson 的指数尾界。若只需保守的一侧预算界，对任意 a>0，Cantelli 不等式给出 Pr(R−33≥a)≤322.8420/(322.8420+a²)。这是数学上界，并非训练观察到的尾频率。[采样器 `poisson-lognormal-filling`、原文 §3.3/Fig.3；矩与成本是由实现推导。]

截断具体删掉了哪条依赖，可以沿同一次前向逐轮求导。样本、R和随机初态确定后，Dⱼ记录第j轮状态对全部参数的导数；一轮计算既传递旧导数，又加入本轮直接使用参数以及注入e产生的导数，后两者合记为Gⱼ。完整反传把两部分一直带到末轮，截断反传则在n轮边界把已积累的Dₙ置零。以下针对n>0；n=0时没有这个截断边界，初态导数照常保留：

```math
\begin{aligned}
D_j&=J_jD_{j-1}+G_j,\\
G_j&=\partial_\theta F_\theta(s_{j-1};e)
+K_j\frac{\partial e}{\partial\theta},\\
\widetilde D_n&=0,\qquad
\widetilde D_j=J_j\widetilde D_{j-1}+G_j\quad(j>n),\\
D_R-\widetilde D_R&=(J_R\cdots J_{n+1})D_n.
\end{aligned}
```

G 中的第一项是在状态与 e 固定时求偏导，以免把 Prelude 梯度重复计数。若损失写作 ell(s_R,θ)，末端读出对 θ 的直接导数在两种计算图中相同；其梯度差为

```math
\begin{aligned}
g_{\rm full}-g_{\rm trunc}
&=D_n^\top(J_R\cdots J_{n+1})^\top v_R,\\
v_R&=\nabla_{s_R}\ell(s_R,\theta).
\end{aligned}
```

上式中的Dₙ正是早期访问留下的参数依赖；末k轮还会访问同一组权重，却不会重新生成这项历史依赖。它被删去后，对最终loss的影响还要经过末k轮的状态雅可比。如果这段传播足够收缩，遗漏项可能衰减；若另外有统一界∥Jⱼ∥≤ρ<1、∥Gⱼ∥≤B、∥v_R∥≤V，就得到下面的充分界：

```math
\|g_{\rm full}-g_{\rm trunc}\|
\le V\rho^k
\left[\rho^n\|D_0\|+
B\frac{1-\rho^n}{1-\rho}\right].
```

这是解释何时尾部反传可能足够的充分条件，论文没有给出真实 Huginn 的这些常数。主配方初态抽样不依赖可训练参数，故 D₀=0；R≤8 时没有被丢弃的 Core 前段。若更换成可学习初态，n=0 时必须保留初态计算图。Prelude 在末 k 轮通过 G_j 持续得到梯度，正是输入注入在训练动力学中的另一作用。[`iterate_forward`，固定版本 [model_dynamic.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)；本节为精确链式法则与条件误差界。]

### 目标语义 截断反传是随边界重新定义的局部损失

前面的遗漏项公式刻画了完整 BPTT 与尾部 BPTT 的差别。还需区分另一件事：每次 backward 都是某个冻结边界损失的合法导数，并不意味着跨参数更新的截断梯度场一定来自同一个固定标量目标。固定样本、深度与随机初态，在当前参数 θ₀ 处计算无梯度前段；只对后段的 θ 求导，可明确写成

```math
\begin{aligned}
\widetilde L_{\theta_0}(\theta)
&=\ell_\theta\!\left(F_\theta^{\,k}(s_n(\theta_0);e_\theta),y\right),\\
g_{\rm trunc}(\theta_0)
&=\left.\nabla_\theta\widetilde L_{\theta_0}(\theta)\right|_{\theta=\theta_0}.
\end{aligned}
```

这里 sₙ(θ₀) 在本次导数中固定，但下一次 forward 会重新计算。以下两参数反例说明为什么不应把这种更新自动解释为某个全局 surrogate 的精确梯度。取共享递推 Fₐ,ᵦ(s)=as+b、s₀=0、R=2、k=1，末端损失为 ½(s₂−y)²。无梯度边界为 s₁=b，真实前向仍给 s₂=(a+1)b；令 e=(a+1)b−y，则

```math
\begin{aligned}
g_{\rm trunc}(a,b)&=(eb,\ e),\\
\frac{\partial g_a}{\partial b}
&=2(a+1)b-y,\\
\frac{\partial g_b}{\partial a}&=b,\\
\frac{\partial g_a}{\partial b}-\frac{\partial g_b}{\partial a}
&=(2a+1)b-y.
\end{aligned}
```

在 (a,b,y)=(0,1,0) 处，两项混合偏导分别为 2 与 1。因此该点附近不存在以此更新场为梯度的二阶连续可微标量势函数。这是一般截断反传的反例，不声称真实 Huginn 恰好呈现这组数值，也不否定局部冻结损失的定义。它限制的是优化理论的表述：不能把完整随机深度 CE 的下降保证、固定 surrogate 的驻点条件或隐式固定点梯度直接套到尾部反传。

**执行对应。** `RecurrentGPT.iterate_forward` 先 `initialize_state`，随后在 `torch.no_grad()` 内调用前 n 次 `core_block_forward`，再调用 k 次有梯度的共享核；尾部仍注入同一个带计算图的 `input_embeds`。下一 microbatch 重跑前段，边界不会作为永久常量保存。activation checkpoint 只重算已选尾部的图，不恢复被截断的前段。依据：[原文 §3.3](https://arxiv.org/abs/2502.05171v2)；固定 [`iterate_forward/core_block_forward`](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py)。局部损失表述与反例为本页推导。

### 锁步随机深度与公共梯度噪声

源码在同一 microbatch 的所有样本与 worker 上共享深度；这有利于执行长度对齐，但与每样本独立抽深度具有不同的梯度方差。为分离这一影响，固定参数，令 g(X,R) 为实际截断梯度，假设深度独立于数据，给定 R 后 N 个样本及其初态独立同分布，并且二阶矩有限。记 μ(R)=E[g|R]、Σ(R)=Cov(g|R)，则全协方差公式给出

```math
\begin{aligned}
\operatorname{Cov}(\bar g_{\rm shared})
&=\frac{\mathbb E[\Sigma(R)]}{N}
+\operatorname{Cov}(\mu(R)),\\
\operatorname{Cov}(\bar g_{\rm independent})
&=\frac{\mathbb E[\Sigma(R)]+\operatorname{Cov}(\mu(R))}{N},\\
\operatorname{Cov}(\bar g_{\rm shared})
-\operatorname{Cov}(\bar g_{\rm independent})
&=\left(1-\frac1N\right)\operatorname{Cov}(\mu(R))\succeq0.
\end{aligned}
```

两者的期望更新场相同，但增加同一深度下的样本数不会消除第二项。若一个更新累积 A 个相互独立的锁步 microbatch，上式共享深度的协方差再整体除以 A；相邻随机数、相关语料、动态 batch 或其他相关性会破坏这一简化条件。它不是对真实训练噪声的测量。一个可完全枚举的标量例子是 g=X+R、X 与 R 独立取 ±1：N=4 时，共享 R 的方差为 1.25，逐样本独立 R 的方差为 0.5。

**执行对应。** `train_step` 先递增 `microbatch_step` 并写入 `model.step`；`randomized_iteration_sampler` 用 `514229+step` 和 `317811+step` 初始化 CPU generator，关闭 lockstep 时才混入 rank。公开配置同时开启 `lockstep_n` 与 `lockstep_k`。深度随 microbatch 改变，而非整个梯度累积窗口只抽一次；重算同一 step 保持同一深度。实际随机数序列由种子确定，数学方差模型是解释采样设计的理想化比较，不把确定种子自动当成独立性证明。依据：[原文 §3.3、Appendix A.2](https://arxiv.org/abs/2502.05171v2)；固定 [`train_step`](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/train.py)、[`randomized_iteration_sampler`](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py) 与[主运行配置](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/launch_configs/frontier_nebel_512_nodes.yaml)。

## 从数据到一次更新的真实执行链

1. 按公开混合数据读取packed样本；训练长度4096。BPE词表在instruction split上训练，文档前加begin_text；除部分长数学文档外，对打包边界做避免无上下文文档尾部的处理。指令数据已混入预训练，不能把它称为纯web无监督语料。

2. train.py::train_step把microbatch_step写入model.step；随机采样器用与step相关的CPU generator种子。lockstep_n和lockstep_k使各worker在同一microbatch取相同n、k，避免不同深度造成等待，并令activation checkpoint重算可复现。

3. RecurrentGPT.forward运行embedding与Prelude；initialize_state生成初态；iterate_forward将n轮和k轮明确分开；每个core_block_forward执行concat→adapter→四个共享SandwichBlock。

4. 每轮或每block的activation checkpoint由activation_checkpoint_impl决定。它通过重算节省激活，与截断BPTT改变梯度路径是两种不同机制。

5. 循环结束ln_f→两层Coda→ln_f→共享LM head→CE；labels在进入模型前预先对齐为 labels_t=x_{t+1}。训练器做梯度累积、全局norm clip、优化器更新和归零；不要再在模型内部重复shift。

6. 官方512节点配方使用ELLISAdam、lr=4×10⁻⁵、betas=(0.9,0.95)、update_clipping=True、atan_adam=True、decouple_wd=True、weight_decay=4×10⁻⁵。_single_tensor_modded_adamw使用atan2的一阶/二阶矩更新，且weight decay乘子由lr/init_lr控制；不能用普通AdamW的同名数值直接替代。代码仍在update clipping中使用epsilon下界，因此“移除epsilon”应限定为主要Adam更新分母机制。

学习率在论文中有两处不同记载：§4.1写η=5×10⁻⁴，§4.3的最终成功运行写4×10⁻⁵，公开frontier_nebel_512_nodes.yaml也采用后者。因此最终配方对应4×10⁻⁵，5×10⁻⁴作为原文内部差异保留。

训练使用bf16、Frontier AMD设备、DDP与优化器状态分片。公开配置中的max_tokens=4T只是上限占位，文件明确说明真实运行未达到该上限，实际训练量仍按论文的800B记录。接下来再看语料构成：原文Fig.4、Appendix C及scripts/sources.yaml、instructionsets.yaml分别给出论文与发布配置中的来源。[原文§4，pp.5–6；固定commit下 train.py、recpre/optim.py、launch_configs/frontier_nebel_512_nodes.yaml]

## 数据配方补核：原始 96 项与仓库 113 项分开保留

原文 Fig.4、Appendix C Tables7–8 共列出 96 个来源（标准来源46项、Instruction Data50项）。固定 sources.yaml 有113项，包含上述96个同名项，另有17项；共同项的 W 和类别均一致，但 everyday-conversations 的地址从 HuggingFaceTB/everyday-conversations 改为带 llama3.1-2k 后缀的版本。检查包保留完整96项、17项差集、原始地址／W／MG／类别和固定源码。使用这份配方时，要把论文的来源清单与发布配置分开；113项发布配置不能直接作为800B历史运行的精确清单。[原文 Fig.4、Tables7–8](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal) · [固定 sources.yaml](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/scripts/sources.yaml)

| 原文类别 | 附录来源项数 | Fig.4 报告比例 |
| --- | ---: | ---: |
| generic-text | 7 | 28.71% |
| code | 10 | 25.36% |
| scientific-text | 3 | 18.73% |
| synthetic-text | 11 | 8.14% |
| longform-text | 10 | 7.50% |
| math | 14 | 6.14% |
| generic-instruct | 14 | 2.09% |
| Q&A-text | 3 | 1.58% |
| math-instruct | 13 | 1.51% |
| writing-instruct | 8 | 0.12% |
| misc-reasoning | 3 | 0.11% |

图中比例因显示舍入合计99.99%，按原值保留；来源项数不代表训练 token 比例。作者明确未对这份偏重代码、科学与数学的混合做充分消融，不能由此单独识别数据配方或循环结构的贡献。

### 文件复用倍率如何变成 token 配比

固定 weighted_file_selection 对 n 个文件使用整数次复制，再无放回抽取分数部分。设 w≥0、m=floor(w)、r=round((w−m)n)，文件 i 含 x_i 个待计量 token；下面只在给定文件列表、均匀子集抽样的条件下成立。r 使用 Python ties-to-even 舍入；例如 n=2、w=0.25 时 r=0，来源会选中零文件。

```math
\begin{aligned}
N_{\rm files}&=mn+r,\\
M&=m\sum_{i=1}^{n}x_i+\sum_{i\in\mathcal S_r}x_i,\\
\mathbb E[M]&=\left(m+\frac rn\right)\sum_i x_i,\\
\operatorname{Var}(M)&=\frac{r(n-r)}{n-1}\sigma_x^2,\qquad n>1,\\
\sigma_x^2&=\frac1n\sum_i(x_i-\bar x)^2.
\end{aligned}
```

所以 w 既不是归一化的来源概率，也不保证恰好 w 倍 token；文件粒度、长度差异和舍入均有影响。继续把实际选中 token 量记为 M_d、打包后保留量记为 R_d，可定义最终经验来源比率；它必须由真实保留量计算，不能用 W 列直接归一化，也不能把随机比率的期望擅自换成期望之比。 以下按同一token计数口径，M_d=0时应有R_d=0；仅对正输入质量来源定义保留率，并要求总保留量Z>0。若全部来源为空或全部丢弃，经验配比未定义，应报告空数据，不能返回零向量冒充概率分布。

```math
\begin{aligned}
\mathcal D_+&=\{d:M_d>0\},\qquad
\rho_d=R_d/M_d\quad(d\in\mathcal D_+),\\
Z&=\sum_{j\in\mathcal D_+}R_j
=\sum_{j\in\mathcal D_+}\rho_jM_j>0,\\
\widehat\pi_d&=R_d/Z
=\begin{cases}\rho_dM_d/Z,&d\in\mathcal D_+,\\0,&d\notin\mathcal D_+.\end{cases}
\end{aligned}
```

### 打包函数的尾部与历史运行的身份

已读 parquet_to_parquet_tokenizer.py 的 map_to_block_operation 在追加下一文档之前检查旧 buffer；普通分支输出一个 S+1 块并丢弃余尾，keep_all 分支输出所有完整块。函数结尾没有 flush，因此 keep_all 不能无条件解释为每个 token 都保留。以 S=4、单个10-token文档为例，若它是 map batch 的最后一项，函数输出为空；再追加下一文档作触发，keep_all=true 输出两个5-token块，false只输出一个。小例直接执行原函数体，不等于原语料丢失率测量。

从发布脚本到800B历史训练，还缺完整的数据准备链。脚本读取 scripts/cooldown_set_v0.yaml，但固定 tree 的 scripts 目录未列此文件，输出名称也属于 cooldown；512节点训练配方读取的则是已完成的 v02／reshuffled_1 parquet。因此，已明确的是发布函数的行为。要重建历史训练，仍需连接该脚本、96项论文配方与实际语料产物的 launcher／manifest。[tokenizer 与切块](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/scripts/parquet_to_parquet_tokenizer.py#L227-L390) · [训练数据入口](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/launch_configs/frontier_nebel_512_nodes.yaml)

当前 ParquetStreamPure 按文件划分 rank，shift_inputs_and_labels 将4097个存储 token 转为4096个输入和4096个目标；存储 token、模型处理位置与有效 CE 位置分别记录。打包及块级划分也不能替代原始文档／问题身份的留出审计。[读取与状态](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/huggingface_dataset.py#L124-L257) · [标签对齐](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/data_loading_utils.py#L102-L116)

#### 算法 D1 文件权重、打包与来源账本

**输入** 固定来源、可读文件及顺序、w、tokenizer、文档 keep_all 标志、S，以及 map batch／分片设置。

**输出** 文件出现记录、输出块、来源保留量、丢弃量和监督位置账本。

![算法 D1 文件权重、打包与来源账本](visualize:fde1_bGliZmlsZV9vUkFmVFU4VEllUzB4MVBnUmFGU213_FileDrive_45288dfe06608191af94334ba8699511)

**性质** 按发布函数重写的数据准备参考；内层切块小例直接执行原函数，外层历史配方缺口保持明确。完整源码与八组新数据检查见[数据与版本页](https://chatgpt.com/space/page_b39954c20a348191b95d45a766f24ef2)。

## 大规模实现：算得多而通信少仍需具体系统配方

Appendix A.2报告跨节点DDP、节点内ZeRO-1优化器分片、每轮activation checkpoint；512节点/4096个GPU逻辑设备、每设备batch=1，使全局每步约16M tokens。PyTorch编译与hidden size 5280的矩阵形状选择带来其环境下单节点108.75 TFLOP/s；57% MFU使用名义峰值，87% AFU使用实测可达峰值125 TFLOP/s，两个分母不能混用。作者报告在当时AMP+截断BPTT组合下只有编译执行能得到正确梯度，并绕开PyTorch SDPA的AMD问题使用ROCm FlashAttention分支；这是特定软件版本的工程观察，不是今天所有PyTorch配置的一般结论。[原文p.24]

RCCL互连在大节点数上出现hang，作者以固定大小分块all-reduce修复。官方recpre/utils.py::_allreduce_chunk_stream将梯度按64×1024²字节的fp32 buffer分块，末块仍传完整大小buffer，只有有效前缀被写回，SimpleFabric.backward调用它；这对应论文所谓64MB packets。它解释了为何仅列“DDP/ZeRO”不足以复现该运行。训练还监控每层有效步长、优化器RMS、参数/梯度norm、相邻循环状态相对变化与[1,4,8,16,32,64]轮验证困惑度，以区分loss下降和模型是否真在利用循环。[Appendix A.2 pp.24–25；recpre/utils.py；recpre/model_dynamic.py::monitor_module]

### 实现推导 一次参数更新必须包括优化器状态

对 Frontier 配方使用的 ELLISAdam 分支，以下公式以一个参数张量为单位。g_t 是梯度同步和全局 norm clipping 之后的梯度；η_t 是实际本地更新核读取的参数组学习率（分片 wrapper 的传递条件见后文），η₀ 是该优化器组保存的初始学习率。此处采用配置中的 `update_clipping=True`、`atan_adam=True`、`decouple_wd=True`，以及默认关闭的 Nesterov/Kahan 等额外分支：

```math
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\max\{\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},\varepsilon^2\},\\
c_t&=\max\left\{1,\sqrt{\operatorname{mean}(g_t^{\odot2}/v_t)}\right\},\\
\eta_t'&=\frac{\eta_t}{c_t(1-\beta_1^t)},\\
\theta_t&=\left(1-\frac{\eta_t}{\eta_0}w\right)\theta_{t-1}
-\eta_t'\operatorname{atan2}\left(m_t,\sqrt{\frac{v_t}{1-\beta_2^t}}\right).
\end{aligned}
```

max、除法与 atan2 逐坐标计算，mean 在当前参数张量内聚合。源码对 v 做原地 clamp，所以这也改变保留下来的二阶矩状态。这里的 weight decay 缩放是 η_t/η₀，而不是通常 AdamW 的 η_t；norm/bias 参数组的衰减还由分组配置决定。[`_single_tensor_modded_adamw` 与 `get_param_groups`，固定版本 [optim.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/optim.py)、[Frontier 配方](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/launch_configs/frontier_nebel_512_nodes.yaml)。]

配方还打开 `running_init=True`：某张量第一次拥有梯度时，以该梯度初始化 m、以其平方初始化 v，再进入上述更新；代码仍计算 bias correction，不能擅自改写为零初始化动量的普通 AdamW。另一个真实执行细节在 `train_step`：`optimizer_step=0` 的第一个累积边界跳过 `optimizer.step()`，随后归零并递增计数；因此“到达第一个 update 边界”不等于“权重已经更新一次”。后续非有限梯度按配置跳过或触发终止。[`ELLISAdam._init_group`；[train.py::train_step](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/train.py)。]

对系统成本，设独立可训练参数数为 P，保存的 Core 尾部轮数为 k。共享权重使参数、优化器状态、一次梯度归约的数据量依赖 P，而 Core 前向应用次数依赖 R；不能将这两类量合并成一个“有效参数量”：

```math
\begin{aligned}
M_{\rm parameters+optimizer}&=O(P),\\
V_{\rm reduced\ gradients}&=O(P),\\
C_{\rm core\ forward}&=O(R),\\
M_{\rm retained\ core\ activations}&=O(k)
\quad\text{在固定 batch、长度、宽度与保存策略下}.
\end{aligned}
```

全量 FP32 梯度的内容体积约为 4P 字节；64 MiB 分块归约改变传输粒度，末块有效前缀之外仍可能被传输，不能按有效参数字节直接推断网络包数或端到端时间。lockstep 随机深度控制同一步 worker 的执行长度，checkpoint 控制激活保存，ZeRO-1 控制优化器状态分片，它们分别处理不同资源项。[原文 Appendix A.2；源码 `SimpleFabric.backward/_allreduce_chunk_stream`。]

#### 算法 Huginn 随机深度与截断更新

**输入** 已经 next-token 对齐的 G 个 microbatch；参数 θ、优化器状态、microbatch 计数 j 与 update 边界计数 u。

**输出** 按固定训练代码更新的参数、动量、计数和终止标志。

**执行前提** 采用 poisson-lognormal-filling、末八轮反传、lockstep 随机深度；一个累积窗口内 G 固定，数据/mask 走原训练器。

![算法 Huginn 随机深度与截断更新](visualize:fde1_bGliZmlsZV9KN08wT1djeGRKMG8ybXZyb05pZi1B_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 下方算法采用固定训练分支的 ELLISAdam 更新，不能按通用 AdamW 替换。依据 [train.py::train_step](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/train.py)、[RecurrentGPT](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py) 与 [ELLISAdam](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/optim.py)。checkpoint 重算必须保持所需随机状态和原 mask；原文 §3.3、Appendix A.2。 初始化与采样按实际 helper 顺序重放：Prelude → initialize_state → randomized_iteration_sampler；initialize_state 中先 randn 再 trunc_normal_ 覆盖的随机数消耗也要保留。 图中“设置当前学习率”对应 train_step 对外层参数组赋值；分片 wrapper 真正更新时读取本地组，二者的对象关系见下节。

### 调度契约 日志学习率必须传到实际更新的参数组

前述 ELLISAdam 公式中的 η_t 必须取实际执行更新的本地参数组学习率。固定 Frontier 配方打开 `fabric.optim_sharding=True`，`get_optimizer` 在 distributed 已初始化时选择 `SimpleZeroRedundancyOptimizer`；`SimpleFabric.setup_optimizers` 原样返回该对象。于是训练器持有外层参数组，真正的 ELLISAdam 更新发生在 `local_optim_group`。这一层对象关系会影响 warmup 是否到达更新算子。[固定主配方、`get_optimizer`、`SimpleZeroRedundancyOptimizer` 与 `SimpleFabric.setup_optimizers`。]

| 新建优化器时的分支 | 本地组与外层组的对象关系 | train_step 重新赋 lr 后的结果 |
| --- | --- | --- |
| local_rank=0 的 embedding/head 组 | 直接传入 `params[1]`，共享组字典 | 本地 ELLISAdam 读到新的调度值 |
| local_rank=1,…,d−1 的主 weight 分片 | `params[0].copy()` 后，将参数列表改为 `[rank−1::d−1]`；组字典独立 | 外层 lr 改变，本地 lr 保留构造值，直到本地组被另外修改或恢复 |
| 各 rank 的 norm/bias 组 | `local_param_groups.extend(params[2:])`，共享组字典 | 本地组跟随调度；主配方对这些组设 weight_decay=0 |
| 未启用分片、直接使用 ELLISAdam | 训练器操作的就是执行更新的组 | 不存在上述外层与本地组的分离 |

这里 d 是每节点参与分片的本地设备数，主运行按 8 处理。共享的是 Parameter 对象，并不意味着超参数字典也共享。`train_step` 使用 `param_group['lr'] = torch.as_tensor(current_step_lr * base_lr)` 重新绑定字典值，没有对旧 lr tensor 原地 `copy_`；wrapper 的 `step` 只调用本地优化器再广播参数，`sync_parameters` 不同步学习率。因此，即使曾共享一个 lr tensor，重新绑定也不能更新另一个独立字典；当前 weight 分片本身已经在组字典这一层分离。[固定 `optim.py` 与 `train.py`；PyTorch `Optimizer.add_param_group` 保留传入的组字典。]

设同一次更新前的参数、梯度、动量和二阶矩完全相同，d_t 为前述去掉学习率的 ELLISAdam 更新方向。对任意两个实际组学习率 η_a、η_b，有

```math
\begin{aligned}
d_t&=\frac{\operatorname{atan2}\!\left(m_t,\sqrt{v_t/(1-\beta_2^t)}\right)}{c_t(1-\beta_1^t)},\\
U(\eta)&=\left(1-\frac{\eta}{\eta_0}w\right)\theta-\eta d_t,\\
U(\eta_a)-U(\eta_b)
&=-(\eta_a-\eta_b)\left(\frac{w}{\eta_0}\theta+d_t\right).
\end{aligned}
```

调度值是否传到本地参数组，会同时影响参数衰减和沿优化方向移动的步长。以主配方η₀=4×10⁻⁵、调度值2×10⁻⁵为例，新建的主weight分片仍把4×10⁻⁵传给更新核，而embedding与norm/bias传入2×10⁻⁵。若只比较这一次更新，并固定前述参数、梯度、动量和二阶矩，两项系数都相差两倍；后续梯度会随训练轨迹改变，因此不能据此推断训练质量差多少。平台期的调度值等于构造值时，这个数值差异消失。直接使用ELLISAdam，或显式更新本地参数组的路径，也不满足上述触发条件。

恢复训练还需重新检查组身份。PyTorch v2.6.0 的 `Optimizer.load_state_dict` 会 deepcopy 保存的组并重建 `param_groups`；wrapper 只恢复 `local_optim_group`。在该版本语义下，原先共享字典的 embedding/norm 也不能再依赖旧别名传播外层赋值。历史运行使用的 PyTorch pre-release、实际 checkpoint 和恢复路径仍须逐一核对，不能从当前发布源码反推 800B 运行始终采用了何种有效学习率。复现时应在 fresh 与 resume 两条路径分别记录每个实际本地组的 lr、init_lr、weight_decay 和计数，再对照调度器；仅记录 `current_step_lr` 不足以验证调度生效。

上述对象关系已通过固定 wrapper/ELLISAdam 类与 PyTorch v2.6.0 参数组方法的 AST 执行检查覆盖 8 个 local_rank；Tensor、通信与数值更新核使用最小替身，仅验证字典身份、赋值和传参。它是源码契约核验，不是 GPU 训练或历史运行复现。依据：[固定 optim.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/optim.py)、[train.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/train.py)、[utils.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/utils.py)、[Frontier 配方](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/launch_configs/frontier_nebel_512_nodes.yaml)、[PyTorch v2.6.0 Optimizer](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/optimizer.py)。

## 推理：prefill、逐token循环、KV复用与退出

生成一个新token时，Huginn仍要经过Prelude、若干轮Core和Coda。此时注意力不仅需要当前状态，还要读取历史token在相同展开深度产生的K/V。常规prefill先处理整个提示，为这些深度分别留下缓存；后续decode才逐token追加。

为何共享权重仍需多份KV？同一Core层第一次和第二次接收的隐藏状态不同，投影出的Q/K/V也会不同。HF的HuginnDynamicCache因此按展开层／循环步和token位置寻址：Prelude通常用0、1，Core继续递增，Coda用负索引。Coda的编号由此不会随R变化而与Core槽位冲突。

既然每轮都可以经Coda读出预测，便可检查继续计算是否还明显改变输出。论文用相邻迭代输出分布的KL决定零样本退出，实验阈值为5×10⁻⁴。HF的generate_with_adaptive_compute在循环中真正检查条件并停止，还提供KLExitEvaluator、LatentDiffExitEvaluator和EntropyDiffExitEvaluator等判据。

单样本退出后，本token可以结束迭代；批量执行则还要等待其他未完成样本。代码能记录各行的首次退出轮次，却仍运行到所有行满足条件，因此记录的深度减少不直接等于同比例GPU加速。_prefill_with_varied_exit_steps采用逐token处理且assert batch=1，源码将其定位为科学实验接口。[原文§6.1；HF相应函数]

退出还会留下一个跨token的问题：前一个token若只算到较浅轮次，下一个token算得更深时，就找不到它在那些深度的历史KV。latest-matching等策略为这个缺口提供替代值，从同一物理Core层的可用深度取缓存；部分实现用mod 4定位该物理层，修改Core层数时要同步修改寻址。

这种做法保留了可继续计算的历史，却改变了注意力实际读取的状态，因而属于近似复用，需要与full-cache比较。循环缓存压缩进一步按预算覆盖槽位。另一项独立选择是连续计算：用上一生成token的最终latent作为下一token的s₀，它改变的是起点，而不是增加可见的CoT token。[原文§6.1–6.3；HF HuginnDynamicCache.update]

自推测解码用小R生成draft、大R验证。论文§6.4提出这项原生能力；固定HF generate_speculative采用argmax一致性或概率阈值验收，并包含cache回滚。它不是一般的分布保持抽样实现，下文给出精确反例与批量状态边界。工程评估仍须记录草稿接受率、验证代价、cache回滚及与目标分布的一致性。[原文§6.4；固定HF generate_speculative]

### 实现推导 退出判据 缓存策略与真实执行量

KL检查比较的是词表上的整个预测分布：本轮相对于上一轮还改了多少。固定HF的`KLExitEvaluator.check`调用`kl_div(current_log_probs, previous_log_probs, log_target=True)`，计算方向为**上一分布到当前分布**。第一次检查以前没有上一轮读出，代码便用均匀分布作为比较起点。

下面把该单样本退出规则写出。源码`auto`阈值为0.001，论文Fig.10/Table 6实验为0.0005；复现原表时应显式传入阈值，不能把`auto`当作论文设置。

```math
\begin{aligned}
p_0(v)&=1/V,\\
d_r&=D_{\rm KL}(p_{r-1}\|p_r),\\
R_{\rm exit}&=\min\!\left(\{1\le r\le R_{\max}:d_r<\varepsilon\}\cup\{R_{\max}\}\right).
\end{aligned}
```

上式描述每轮检查、min_steps=0的单样本路径。若隔若干轮才检查，模型就只能在这些检查点退出。检查本身也不是免费的：为取得pᵣ，每次都要执行Coda和LM head。因而一枚token的计算量既取决于Core跑了几轮，也取决于读出了几次分布，下面把这两项分开：

```math
C_{\rm token}
=C_{\rm Prelude}+R_{\rm exit}C_{\rm Core}
+N_{\rm checks}(C_{\rm Coda}+C_{\rm head})
+C_{\rm cache}+C_{\rm sampling}.
```

批量情形还有更深一层影响。`generate_with_adaptive_compute`在所有行结束前继续处理整个batch；已记录退出的行仍可能写入更深KV。后续token读取了这些额外KV，就不再承接单独batch=1运行时的同一段历史。因此差别可能进入生成轨迹，而不只是运行时间。要比较论文型自适应结果，可先用batch=1得到对应轨迹，再评价能真正停止或压缩已退出行的服务实现。[固定版本 [HF 代码](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)：`KLExitEvaluator`、`generate_with_adaptive_compute`。]

例如缺失的是Core展开地址s上的历史token j，`latest-m4`会往前寻找与s相差4的整数倍、且确实存在的最近地址。相差4表示回到同一个物理Core层的更早访问。把Core地址记为s≥2，查找和替代过程就是：

```math
\begin{aligned}
u(s,j)&=\max\{u\le s:
 u\ge2,\ u\equiv s\pmod4,\ (u,j)\text{已存在}\},\\
(\widetilde K_{s,j},\widetilde V_{s,j})
&=(K_{u(s,j),j},V_{u(s,j),j}).
\end{aligned}
```

这个查找式明确了近似发生在哪里：目标是深度s的KV，实际读到的却可能是较早深度u的KV。二者不相等时，keys、values以及由它们决定的softmax权重都可能变化；权重共享并不消除状态差异。作为比较基准，固定R、保留完整缓存，并使用相同随机初态和位置语义时，MHA的KV内容量为：

```math
M_{\rm KV}=2bBnh(2+4R+2),
```

其中 b 为每元素字节数，B 为 batch，n 为已缓存 token 数，h=5280；该式不含字典、分配器和临时工作区。压缩缓存与连续 warm start 分别改变历史注意力状态和下一 token 的初态，须各自保留为实验变量。

**边界分支的静态发现。** 固定 revision 中若 `max_steps=4`、`check_criterion_every_n_steps=2`、`min_steps=0` 且没有早退，会在第 1、3 轮产生 `outputs`，第 4 轮不检查。循环自然结束时，只在 `outputs is None` 才重新读出，因此可复用第 3 轮 logits，却记录深度 4。默认每轮检查不触发这一情况。接入稀疏检查时需要让“保存的 logits 对应轮次”与“实际执行轮次”分别可追踪，并在封顶轮保证新读出；这是依据控制流发现的边界风险，并非模型运行测量。[同文件 `generate_with_adaptive_compute` 的 for–else 分支。]

自推测解码的分布保持也有可核验条件：若 draft 概率为 q、目标概率为 p，标准抽样校正的接受率是 min(1,p(y)/q(y))，拒绝时从正部 (p−q) 归一化后的残差分布取样。仅有 cache rollback 不能替代这个概率条件；贪心验证则需要另一套一致性检查。这里只给验收条件，不将它当成官方所有分支均已满足的结论。[原文 §6.4；`generate_speculative`。]

### 自推测解码 实际验收规则与状态回滚的反例

论文 §6.4 提出利用较浅循环草拟、较深循环验证的原生能力，并讨论复用已计算状态。固定 HF `generate_speculative` 的具体算法更窄：它不保存 draft 分布 q，也没有执行上述 p/q 接受率与残差采样。源码按以下顺序运行。

| 执行位置 | 固定版本的行为 | 对实验含义的影响 |
| --- | --- | --- |
| prefill 与 draft 循环 | 先以 `num_steps` prefill 并生成首 token；每次 draft 调用完整 forward、用 `draft_steps` 和 `_sample_next_token` 追加 token | 草拟深度与验证深度分别记录；采样设置属于实际分布 |
| 验证前回滚 | `clear_last_k_entries(L)` 删除当前 L 次 draft 写入的 KV；`cache_position` 回到待验证窗口 | 保留旧 prefix 的 KV，但本窗口会以目标深度重新计算；不能据此声称已复用本窗口浅层 latent 来续算 |
| `verification_threshold >= 1` | 将 draft token 与目标 logits 的 argmax 逐个比较 | 是 argmax 一致性判据，不是抽样概率校正 |
| `verification_threshold < 1` | 用目标原始 softmax 检查 p(y)≥τ maxᵥp(v) | 阈值使用的分布不经过 `_sample_next_token` 的 temperature/top-k/top-p 变换 |
| 首次不匹配 | 接受此前前缀，再由目标 logits 调用 `_sample_next_token` 生成一个替代 token | 替代来自目标分布本身，而不是正部 (p−q) 的残差 |
| 全部匹配 | 接受全部 L 个 draft token | 这是该控制流的实际分支；没有额外的目标 bonus token |

即使先排除所有缓存问题，这种采样也一般不保持目标分布。固定同一入态与目标分布 p，令单个 draft token Y∼q；温度为 1、不做 top-k/top-p，接受集合为 A，失败时独立从 p 重采样。输出分布恰为

```math
\widetilde p(v)=q(v)\mathbf1\{v\in A\}+q(A^c)p(v).
```

取 p=(0.6,0.4)、q=(0.5,0.5)，默认 argmax 验证的 A 仅含第一个 token，便得到 p̃=(0.8,0.2)。甚至 q=p 时也得到 (0.84,0.16)，仍不等于目标。这是一 token 的解析反例，已经足以排除该分支的一般无损抽样保证；它没有测量真实模型的偏差大小。贪心生成是另一情形：如果验证分布、完整 cache、位置与初态确实匹配目标执行，argmax 验证可以保持该贪心选择；这些前提仍需逐项满足。

批量回滚还存在可由控制流构造的反例。设当前 `input_ids` 长度 C，完整 cache 长度 C−1，L=2；第一行在首个 draft token 不匹配，第二行两个全匹配。各行实际追加数为 1 与 2，但全匹配行的 `first_mismatch` 也是 all-false argmax 返回的 0。源码用跨行最小值回滚，保留的公共 cache 长度为 C，再把输入补齐到 C+2。下一次仅设置 `cache_position=[C]`，与第二行下一个尚未处理的 token 位置 C+1 不一致。因此异质接受长度不能仅凭公共回滚加 pad 就视为独立逐行推测轨迹；应以单样本或相同接受长度为基线，并核验逐行位置、有效长度和 KV。

`continuous_compute=True` 也需要独立审查：每次 draft 后保存的 `input_states` 形状为 B×1×h，而 L>1 的整窗 verification 输入是 B×L×h；固定 `core_block_forward` 直接在最后一维 `torch.cat([x,input_embeds])`，不广播序列维。该组合在正常正循环数下具有明确形状不相容。连续计算的逐 token 有效性不能直接转成整窗并行验证的有效性。以上是静态执行和小型计数反例，未运行模型；默认关闭连续计算不触发这个形状分支。

依据：[原文 §6.4](https://arxiv.org/abs/2502.05171v2)；固定 [`generate_speculative`、`_sample_next_token`、`prepare_inputs_for_generation`、`iterate_forward/core_block_forward` 与 `HuginnDynamicCache.clear_last_k_entries`](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)。概率式、批量计数和形状条件为本页核验；不把论文的机制提案、当前 HF 的启发式实现和满足概率校正的推测采样混为同一结果。

#### 算法 Huginn 单样本逐轮 KL 退出

**输入** 单条序列、历史缓存、显式 KV lookup 策略 π、最大轮数 Rmax、阈值 ε、随机或连续初态策略。

**输出** 采样的新 token、真实退出轮次、对应 logits、更新后的 cache 与 warm-start 状态。

**执行前提** batch=1、min_steps=0、每轮检查；π 对每个缺失历史地址都有定义。缓存近似属于该策略的一部分。

![算法 Huginn 单样本逐轮 KL 退出](visualize:fde1_bGliZmlsZV9PSXZMWWU1eEpfamlXZl9xRWRrbll3_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 转写 [HF generate_with_adaptive_compute/KLExitEvaluator](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) 的单样本每轮检查路径；拆开了 `SandwichBlock` 内部顺序。显式 ε=0.0005 对应论文实验，auto=0.001 是源码默认。此算法未把近似 KV lookup 宣称为 full-cache 等价。

## 结果与消融：对齐数据、checkpoint、prompt与读出协议

| 比较与原始定位 | 结果 | 能支持的结论 |
|---|---|---|
| 同数据/设置、180B tokens，Table 4，p.9 | 固定深度baseline ARC-C=26.96；循环模型R=32为29.18，R=1为23.72；HellaSwag对应37.34→48.80 | 同一训练环境下，收益依赖实际执行循环；非仅数据混合不同 |
| Huginn 3.5B、800B tokens、R=32，Table 3，p.8 | MBPP pass@1=24.80，HumanEval pass@1=23.17 | 数学/代码具备额外计算收益，但不能由此推出胜过任何现代大模型 |
| OpenBookQA closed/open设置，Table 5，p.9 | R=32从38.2升至49.2，增加11.0点 | 提供相关事实可补足记忆短板，与操作上下文知识的解释相容 |
| 一轮MT-Bench，Table 6，p.24 | 标准R=32为5.662±0.388；KL早退5.562±0.389；缓存预算4为5.856±0.395 | 小规模对话评估中差异落在较大不确定性范围，不能据分数排序宣布必然提升 |

### 完整 Table 6 与配对误差的条件

原文 Table 6（p.24）全部六行如下。s是cache-compression预算，KL实验阈值是5×10⁻⁴；这里沿用原表条件，不把HF的auto阈值或其他生成变体混入。[英文原文 Table 6](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal)

| 原表推理方案 | MT-Bench一轮分数 | 标准误 |
| --- | ---: | ---: |
| cache compression，s=4 | 5.856 | 0.395 |
| baseline，64 iterations | 5.693 | 0.386 |
| cache compression，s=16 | 5.687 | 0.402 |
| baseline，32 iterations | 5.662 | 0.388 |
| cache compression，s=8 | 5.631 | 0.384 |
| KL exit，t=5×10⁻⁴ | 5.562 | 0.389 |

原文报告相对baseline差异不显著。现有边际标准误不足以独立重建配对检验：若X、Y表示同一题目采样设计下两个均值估计量，其标准误为e_X、e_Y，相关系数为ρ，则：

```math
\operatorname{SE}(X-Y)=\sqrt{e_X^2+e_Y^2-2\rho e_Xe_Y},\qquad
|e_X-e_Y|\le\operatorname{SE}(X-Y)\le e_X+e_Y.
```

以s=4减R=32为例，分差0.194；只凭所报两项SE，差值SE的可能范围是0.007–0.783。这个范围不是置信区间，也不是新的显著性结论；不能按“误差条重叠”或默认独立来代替原检验。复核需要逐题配对分数、judge与重复评分方案；若报告配对bootstrap，应共同重采样题目索引并保留其他随机层级。

上述数值按PDF表格核对；Table 2的GSM8K双提取指标、系统prompt与Table 4/EMA结果不能混成一行。论文还报告EMA使用最后75个checkpoint、间隔7、β=.9，在R=64时GSM8K CoT flexible=47.23%、strict=38.59%；这是另一个权重平均模型，应单独标识。[原文§5.4，pp.9–10]

Fig.7–9显示任务与上下文决定饱和深度：HellaSwag较早饱和，数学可使用更多循环；ARC-C中更多few-shot示例使饱和点后移。Table 4的固定深度baseline只训练到180B，不能把它当800B等训练量的完整对照。模型由单个主大规模运行支撑，很多设计选择只有小规模实验或联合修复证据，因果结论必须保持这个范围。

### 评测入口的身份 论文版本与任务配置须同时固定

固定仓库中的评测文件并不都对应上传的 v2 论文。按文件自己的声明逐项区分后，原表分数、后续方便运行的命令、实验性任务变体才有可追踪的关系；目录中存在某个同名 task，不能证明原文用了它。

| 固定文件 | 源码或说明中的实际用途 | 对 v2 结果复核的含义 |
| --- | --- | --- |
| [README.md](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/README.md) | 常规基准直接调用 lm-eval；论文代码任务使用 bigcode；另提供后来的 HumanEval 运行方式 | 分开记录原论文管线与后续替代入口 |
| [local_lm_eval.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/local_lm_eval.py) | 顶部明确只用于测试本地模型实现的改动；普通评测应直接使用 lm-eval | 这是本地包装入口，不是原文全部结果的统一复现脚本 |
| [misc_benchmark_variants/README.md](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/misc_benchmark_variants/README.md) | 明确声明该目录的 benchmark variants 未用于论文 | 不能把目录内 YAML／自定义过滤器当作原表默认任务 |
| [saturation_eval_dist.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/saturation_eval_dist.py) | 文件说明明确面向 v3 的饱和图；任务、few-shot、chat 均可由调用方改变 | 上传材料是 v2，不能仅凭“saturation”文件名认定其设置对应 v2 Figs.7–9 |

两处具体差异影响结果解释。v2 Fig.8 明确 HellaSwag、HumanEval 为 zero-shot 且不使用 chat template，GSM8K CoT 使用 chat template 与 8-way few-shot 多轮格式。固定 README 的后续 HumanEval 示例则使用 `humaneval_instruct`、chat template、`do_sample=True`、temperature=0.2、top_p=0.95；它不能无注释地回填成 v2 Fig.8 那条 HumanEval 曲线的原执行协议。README 的 GSM8K 示例使用数学助手 system prompt，而 `saturation_eval_dist.py` 在 `use_chat=True` 时使用普通 helpful-assistant prompt；二者的输入也并不相同。[v2 §5.1–5.3、Fig.8](https://arxiv.org/abs/2502.05171v2)；上述固定 README 与 saturation 源码。

`saturation_eval` 的执行顺序是创建 HFLM、显式赋值 `model._model.config.mean_recurrence=num_steps`，再调用 `evaluator.simple_evaluate`。固定 HF `iterate_forward/randomized_iteration_sampler` 在模型处于 eval、没有显式 `num_steps` 覆盖且不走 meta 统计分支时，才从当前配置返回 n=mean_recurrence、k=0；因此复核深度应检查真正模型对象的配置和本次传参，不能只保留命令行上写过的 R。源码关于 model_args 的注释只证明作者在其评测环境中采用了这次显式赋值，不代表所有 lm-eval／Transformers 版本都会忽略该参数。[固定 HF 采样入口](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)。

具体复核一行结果时，记录论文版本与表／图、checkpoint revision、harness 与 task 配置版本、实际循环深度、完整格式化 prompt、few-shot、生成设置、dtype 和结果过滤器。GSM8K 的 strict／flexible 结果分别保留；代码任务同时记录测试集与执行器。这里完成的是协议及源码身份核对，未运行原评测，也没有用新的入口声称重现旧表。

### 受控算术补核 原文的命中指标与失败边界

Appendix A.1/Fig.14 另做了未专门算术微调的多操作数加法实验：随机生成 m 个各有 d 位的十进制数，放入数学助手 system prompt 与聊天模板；每个设置运行 50 次。热图固定 R=32，折线扫描 R∈{1,2,4,8,16,24,32,48,64}。原文报告较深循环可改善部分设置，但操作数增加后仍明显失败，且两位、三位数实验的深度排序不一致，不能把它概括成已经学到可任意扩展的加法算法。[原文 Appendix A.1/Fig.14，p.23](https://arxiv.org/abs/2502.05171v2)。

p.23 图像核对后，热图中“两操作数、d=1…6”的标注依次为 1.0、1.0、0.8、0.7、0.6、0.5；“四／五操作数、d=1”标注为 0.3／0.1。**这些是一位小数的图中标注，不是精确的正确题数**，尤其不能把显示 1.0 或 0.0 当作已确认 50/50 或 0/50。同一图还有待核的跨面板差异：R=32、五个两位数的热图为 0.0，而左下对应 R=32 曲线约为 0.58。原文没有给出可消解该差异的逐题记录，因而保留两处观察，不合成一份虚构的一致精确数据。

更关键的是，原文判分只检查“正确和的字符串是否出现在整个输出中”。令 o_{j,r} 是第 j 题在深度 r 的完整输出，a_{jk} 为操作数，则其报告量为

```math
\widehat A_{\rm hit}(m,d,r)
=\frac1{50}\sum_{j=1}^{50}
\mathbf1\!\left\{
\operatorname{Contains}\!\left(o_{j,r},
\operatorname{str}\!\left(\sum_{k=1}^{m}a_{jk}\right)\right)
\right\}.
```

它不等于提取最终答案后的整数精确匹配。例如真实和为 12、模型输出仅为“112”，子串命中为 1，整数答案却错误。这个反例只说明判分器的逻辑差异，不估计原实验误判率。50 次试验使未舍入分数以 2 个百分点为步长；只有在固定协议下独立同分布 Bernoulli 试验的条件下，标准误才满足

```math
\operatorname{SE}(\widehat A_{\rm hit})
=\sqrt{\frac{A_{\rm hit}(1-A_{\rm hit})}{50}}
\le \frac{1}{2\sqrt{50}}\approx 0.07071.
```

这里上界约为 7.07 个百分点；它不包含训练种子方差、判分偏差或跨深度复用题目的相关性。复现时应同时保存题目、真实整数和、原始输出、原文子串命中和最终答案精确匹配；跨深度比较还要标明是否使用同一组题。已核固定 [GitHub tree 1ea7220](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e) 与现有评测入口，尚未定位该图的完整逐题实验脚本及日志。上述执行规格来自论文图注，不能冒充已取得的原实验代码。

### 结果解释 区分循环收益 拓扑收益与训练运行不确定性

对同一 checkpoint、相同 prompt、读出与采样协议，令 c_i(R) 是第 i 题的正确性指标。改变深度的配对收益有一个直接分解：

```math
\begin{aligned}
\widehat\Delta_{b,a}
&=\frac1N\sum_{i=1}^{N}[c_i(b)-c_i(a)]\\
&=\widehat\Pr(0\!\to\!1)-\widehat\Pr(1\!\to\!0).
\end{aligned}
```

因此平均分随深度提高，仍允许部分题目被额外循环改错。应保留两个方向的转换频率，并把成本与深度同时记录；一条平均准确率曲线不能给出逐样本单调性。

同一训练条件下，180B Table 4 可进一步区分两个差值：

```math
\begin{aligned}
\Delta_{\rm use}
&=\operatorname{Score}(\theta_{\rm loop},R=32)
-\operatorname{Score}(\theta_{\rm loop},R=1),\\
\Delta_{\rm design}
&=\operatorname{Score}(\theta_{\rm loop},R=32)
-\operatorname{Score}(\theta_{\rm fixed},R_{\rm fixed}).
\end{aligned}
```

前者回答同一循环模型是否利用额外计算；后者还包含参数组织、有效深度、训练优化等设计差异。两者不能互相替代。800B 主模型、180B 对照、EMA 权重和不同 GSM8K 答案提取协议均应拥有不同记录键。[原文 Tables 2–4、§5.4、Figs.7–9。]

若原始逐题数据可得，可用配对 bootstrap 或配对差值的标准误描述测试题抽样不确定性；它们仍不包含重新预训练的随机种子方差。MT-Bench 的评分波动也不等价于多次训练方差。对联合修复实验，用干预向量 v=(norm, embedding scale, injection, learning rate) 记结果：一次 v₀→v₁ 只估计整套配置的变化，不能分配成四个单组件因果效应。[原文 §4.3、Table 6；上述估计量是对实验设计的严格解释。]

## 状态演化：路径独立不是固定点保证

论文§7与Fig.11–12、Appendix Fig.22观察到三类latent轨迹：近固定点收敛、轨道/旋转、单方向漂移“sliders”。以128轮状态作参照时，距离不保证逐轮下降；多次随机初始化仍出现相似轨迹形态，被称为经验path independence。可用以下诊断量区分“输出稳定”和“状态不动”：

```math
\begin{aligned}
d_r&=\|s_r-s_{r-1}\|,\\
q_r&=D_{\rm KL}(p_{r-1}\|p_r),\\
a_r&=\|\nabla_sR_\theta(e,s_{r-1})\|_2.
\end{aligned}
```

前两个量可直接观测；第三个是我们建议的局部敏感度分析，不是论文已证明的收缩常数。轨道中的状态可能一直移动而输出已稳定；反之，状态变化很小也不保证答案正确。论文不使用DEQ求根/隐式反传，不应把它改写为必然收敛的深度平衡模型。[原文§7，pp.11–13]

### 推导 轨迹稳定 输出稳定与答案正确是三个命题

固定输入和权重，从不同初态 ξ 产生轨迹 s_r(ξ)。除逐步残差外，可以定义同一步的初始化间距离和候选周期残差：

```math
\begin{aligned}
D_r^{\rm init}&=\mathbb E_{\xi,\xi'}
\|s_r(\xi)-s_r(\xi')\|_2^2,\\
\delta_r^{(m)}&=\|s_r-s_{r-m}\|_2
\quad(m\ge1).
\end{aligned}
```

这两个量观察的是不同现象。δ^(1) 变小支持“局部接近不动”；若某个 m>1 的周期残差变小而 δ^(1) 不小，则更符合周期或近周期运动。不同轨道若有相位差，D_init 也可能不减小，因此只有明确允许相位对齐时，才能改为比较整条轨道的形态。即使轨道形态相似，也不表示逐轮状态相等。PCA只是正交投影，投影距离不大于原空间距离；二维图靠近仍不足以证明完整隐藏状态靠近。[原文 §7/Figs.11–12、Appendix Fig.22；这些量是可执行的诊断定义。]

对 softmax 读出的 logit 函数 z(s)，若在所比较线段上其 Lipschitz 常数不超过 L_z，log-sum-exp 的 Hessian 谱范数不超过 1/2，可得

```math
D_{\rm KL}(p(s)\|p(s'))
\le\frac14\|z(s)-z(s')\|_2^2
\le\frac{L_z^2}{4}\|s-s'\|_2^2.
```

这只给“状态接近 ⇒ 输出接近”的条件上界；反向推断不成立，因为不同隐藏状态可能被读出映射到同一分布。这里需把其他上下文与 cache 固定，或将它们并入完整状态。一次相邻输出变化很小，只说明这一轮的局部稳定；它不约束尚未执行的后续轮次，也不等于答案正确。退出风险的 TV、argmax margin 与序列分布条件界见[后训练课程](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)。专页评测继续保留停止深度、相对固定深度质量损失、实际检查代价和缓存策略四个独立量。

## 后训练接入与证据缺口

Huginn适合研究固定/随机R下的SFT、保持随深度提高的能力、输出稳定性退出、缓存近似与连续latent warm start之间的相互作用。接口必须核清：

- HF forward的scalar num_steps被解释成“无梯度轮数、梯度轮数0”。若训练时为了固定R直接传scalar，Core的计算路径会全部在no_grad中；应显式传(n,k)并确认预期模块有梯度。

- labels必须已对齐next-token；HF实现直接flatten logits和labels做CE，不像部分通用HF模型自动在内部shift。

- HF默认forward把prepared_attn_mask置None，compile_mask调用被注释；padding/文档隔离训练应使用已审查的训练路径，不能假设传attention_mask就会按期望生效。

- 任意R推理、KL退出和KV复用是可用接口/研究机制；“输出质量单调增长”“无限深度稳定”“动态batch获得线性加速”均未被证明。

本页核对公开源码，不声称已重现800B训练、测量新硬件吞吐或验证所有采样分支。后续对具体后训练实验应先做小张量gradient-flow、全前缀与cache解码一致性检查，并将checkpoint、depth、cache policy、随机初态、prompt及评测脚本固定。HF `RavenForCausalLM.forward / iterate_forward`。参见[后训练课程](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)。

### 接口契约 先确定优化的是哪一条计算图

针对固定 HF revision，`num_steps` 的三种用法具有不同梯度可达性。下表讨论梯度路径是否存在，非零数值还取决于样本、权重和损失；所有路径都要求在外层开启 autograd。

| 传参 | Core 前向 | Core 参数与 Prelude 参数的梯度路径 | 适用含义 |
| --- | --- | --- | --- |
| scalar R | R 轮全部 no_grad | 经 Core 的路径被切断；循环后 ln_f、Coda 与 head 仍可能有梯度 | 推理接口语义，不能直接当固定深度全模型 SFT |
| (0,R) | R 轮全部建图 | 保留所有轮次贡献 | 固定深度完整 BPTT |
| (R−k,k) | 前 R−k 轮 no_grad，末 k 轮建图 | 丢失边界以前的状态导数；保留尾部直接参数与输入注入导数 | 与所选截断目标一致的后训练 |

由于输入 embedding 与 LM head 绑权，即使 scalar 路径切断了输入侧梯度，共享 embedding 权重仍可通过输出 head 得到梯度；仅检查该权重 `.grad` 非空会漏掉 Core 未训练的问题。应按 adapter、Core、Prelude、Coda、共享 embedding/head 分组检查计算图与梯度来源。[`RavenForCausalLM.forward/iterate_forward`，固定版本 [HF 源码](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)。]

把训练样本的有效预测位置记作集合 I，则预先对齐 labels 后的 SFT 目标应明确为

```math
\mathcal L_{\rm SFT}
=\frac{1}{|I|}\sum_{i\in I}
-\log p_{\theta,R_i}(x_{i+1}\mid x_{\le i}),
```

其中 I 的选择、R 的训练分布、反传尾长与 mask 共同定义实验。masked loss 只是不计某些位置的损失，不等于禁止注意力跨 padding 或跨文档读取。固定 HF `forward` 未接入其被注释的 `compile_mask`；一旦采用 packing，需核实实际训练路径的注意力 mask，不能用正确的 label mask 代替它。由此，预训练到后训练的对接至少要保存深度采样、初态、截断边界、mask 与读出协议，才能判断性能改变来自目标还是接口。[原文 §3.3、Appendix C；源码 mask 与 labels 分支。]

## 与 TaH2 对照时把退出观测的成本计入预算

Huginn 的逐轮退出需要最终归一化、Coda 和词表读出，才能比较预测分布。TaH2 的 decider 也消费当前 hidden 与预测统计。固定 checkpoint 比较自适应计算时，需要同时计入观察、路由和缓存操作。

设单 token decode 的原策略执行 R 轮，仅最后读出一次；退出策略执行 m 轮并读出 k 次。用每轮共享核成本 C_core、每次读出成本 C_read 和额外控制开销 C_ctl 作加性近似，公共 Prelude 成本抵消：

```math
\begin{aligned}
\Delta C&=C_{\rm fixed}-C_{\rm exit}\\
&=(R-m)C_{\rm core}-(k-1)C_{\rm read}-C_{\rm ctl},\\
\Delta C>0&\iff(R-m)C_{\rm core}>(k-1)C_{\rm read}+C_{\rm ctl}.
\end{aligned}
```

观察每轮时 k=m；稀疏检查必须记录真正执行的检查集合。KV 长度、kernel 调度和活动批大小使单次成本变化时，改为逐次实测求和。这一恒等式不为近似系数提供实测依据。训练 lookahead 是另一项开销，部署成本单独记录。

接入 TaH2 风格深度监督时，先固定比较使用的 Coda、缓存历史和可观测状态，再定义相邻轮次收益；改变 KV 补齐策略或读出时机后，两个 loss 对应的条件计算也会变化。

实现对应：[固定 Huginn HF 的逐轮退出与读出](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py)；[TaH2 推理与缓存](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)。训练接口仍须遵循前文 scalar／tuple num_steps 的梯度区别。

# 固定版本的文件与函数索引

来源：[github.com/seal-rg/recurrent-pretraining](https://github.com/seal-rg/recurrent-pretraining)；固定 revision：`1ea7220ec7eb42d13e89db0663df254d0bcdc28e`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [recpre/model_dynamic.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py) | RecurrentGPT.__init__, forward, iterate_forward, core_block_forward, randomized_iteration_sampler, initialize_state | 主训练计算图；2–4–2；concat adapter；锁步可复现采样；前n轮no_grad末k轮有梯度；整段循环结束ln_f |
| [recpre/model_dynamic.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/model_dynamic.py) | SandwichBlock.forward | 归一化结构；norm2(attn(norm1(x))+x)，norm4(mlp(norm3(x))+x) |
| [train.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/train.py) | train_step, get_lr, create_dataloader, validate, load_checkpoint | 训练器与评估入口；microbatch_step传给model.step；fwd/bwd梯度累积；clip；optimizer step；不同recurrence验证；scheduler改外层组，恢复走fabric.load |
| [launch_configs/frontier_nebel_512_nodes.yaml](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/launch_configs/frontier_nebel_512_nodes.yaml) | 配置文件 | 公开主训练运行配方；lr=4e-5；ELLISAdam；weight_decay=4e-5；lockstep_n/k；world_batch_size4096；4096 token；max_tokens4T非真实训练量 |
| [recpre/optim.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/optim.py) | get_param_groups, get_optimizer, SimpleZeroRedundancyOptimizer, ELLISAdam.step, _single_tensor_modded_adamw | 优化器实现；update clipping、atan2更新、decouple_wd lr/init_lr缩放；不能等同标准AdamW；分片组字典身份与lr传递 |
| [scripts/sources.yaml](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/scripts/sources.yaml) | 数据清单 | 预训练来源；与Appendix C及instructionsets.yaml对应；未下载语料 |
| [evaluate_raven/local_lm_eval.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/local_lm_eval.py) | 文件级入口 | 本地修改版模型的lm-eval测试包装；文件明确建议普通评测直接使用lm-eval，不是全部原表的统一复现脚本 |
| [recpre/utils.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/utils.py) | _allreduce_chunk_stream, SimpleFabric.backward/setup_optimizers/load | 分块通信；64MiB fp32 buffer、尾块仍用完整buffer且只回写有效前缀、all_reduce；固定规模数据包对应论文互连hang修复；setup_optimizers原样返回，load调用对象load_state_dict |
| [recpre/init.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/recpre/init.py) | get_factor_table | 训练初始化表；takase项与HF尺度一致；表注释明确FFN gate也可计作out_proj |
| [README.md](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/README.md) | Reproducing Benchmark Scores | 正式任务入口与GSM8K格式；代码原用bigcode，后续HumanEval示例为instruct+chat，不能无注释回填v2协议 |
| [evaluate_raven/misc_benchmark_variants/README.md](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/misc_benchmark_variants/README.md) | 目录身份声明 | 明确这些benchmark variants未用于论文；不将目录内YAML默认映射原表 |
| [evaluate_raven/saturation_eval_dist.py](https://github.com/seal-rg/recurrent-pretraining/blob/1ea7220ec7eb42d13e89db0663df254d0bcdc28e/evaluate_raven/saturation_eval_dist.py) | saturation_eval | 自述面向v3饱和图；显式更新模型config.mean_recurrence后评测；prompt及task依参数，不自动等于上传v2 |

依赖源码：[PyTorch v2.6.0](https://github.com/pytorch/pytorch/tree/v2.6.0)，用于核对参数组创建和恢复的对象语义；论文记录的 pre-release 运行仍需单独固定其实际版本。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [torch/optim/optimizer.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/optim/optimizer.py) | Optimizer.__init__, add_param_group, load_state_dict | 创建时保留组字典；load时deepcopy并重建组；与Huginn wrapper一起检查fresh/resume学习率传递 |

来源：[huggingface.co/tomg-group-umd/huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125)；固定 revision：`bb6621b65e90b6a4b9b29ef88dc83866d450470c`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [config.json](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/config.json) | 配置 | 发布基座结构；h5280、55heads、96head_dim、2/4/2、mean_recurrence32、mean_backprop_depth8、tie_embeddings |
| [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) | _init_func, RavenPreTrainedModel._init_weights, init_glu | 初始化；不同模块尺度；GLU两半不同初始化 |
| [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) | RavenForCausalLM.forward, iterate_forward, core_block_forward | 推理与可训练接口；scalar num_steps进入no_grad路径；labels已shift；mask编译被注释 |
| [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) | HuginnDynamicCache.update | cache身份与缺失KV补齐；展开深度+token位置；同物理层匹配；compress策略含4层hardcode |
| [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) | generate_with_adaptive_compute, _prefill_with_varied_exit_steps, KLExitEvaluator.check | 自适应退出；真正循环内检查；批量等待全部退出；特殊prefill逐token且batch1 |
| [raven_modeling_minimal.py](https://huggingface.co/tomg-group-umd/huginn-0125/blob/bb6621b65e90b6a4b9b29ef88dc83866d450470c/raven_modeling_minimal.py) | generate_speculative | argmax/阈值验收与目标重采样；一token分布反例、异质batch回滚计数及continuous_compute整窗形状边界；未测吞吐 |

# 逐项覆盖核对

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 原生性/参数/拓扑 | §3.1–3.2 pp.3–4, Fig.2 | config.json；RecurrentGPT.__init__；RavenForCausalLM.__init__ | 论文与官方代码已核 |
| sandwich norm/额外ln_f位置 | §3.2 pp.3–4 | SandwichBlock.forward；RecurrentGPT.core_block_forward/iterate_forward/forward；RavenForCausalLM同名函数 | 已核；标出额外ln_f位置文字与代码差别 |
| 初始化/输入注入/输出 | §4.1 p.5 | recpre/init.py；HF _init_func/_init_weights/init_glu/initialize_state | 已核主要路径；发布HF有细粒度GLU初始化逻辑 |
| 随机循环分布/期望 | §3.3 p.4 Eq.(1)–(2), Fig.3 | randomized_iteration_sampler，poisson-lognormal-filling | 已核；32为分布参数，含+1使期望33 |
| TBPTT/Prelude梯度 | §3.3 p.4 | iterate_forward: no_grad前段/有梯度后段；每轮input injection | 已核；完整梯度遗漏项、冻结边界局部损失与非保守更新场反例；锁步深度的条件协方差已推导 |
| 数据/packing/tokenizer | §4.1 p.5 Fig.4, Appendix C pp.35–37 | scripts/sources.yaml/instructionsets.yaml/parquet_to_parquet_tokenizer.py；train.py dataloaders | 论文配方及文件入口已核；未重建语料 |
| 优化与分布式训练 | §4 pp.5–6, Appendix A.2 pp.24–25 | train.py::train_step；recpre/optim.py::ELLISAdam/_single_tensor_modded_adamw；frontier_nebel_512_nodes.yaml | 实际优化算子与分片lr传递已核；8个local_rank契约模拟通过，fresh/resume分开；最终lr矛盾保留，历史有效lr待运行记录 |
| 失败运行与组件消融 | §4.3 p.6 Fig.5–6；§3 | model_dynamic.py多种norm/injection选项 | 已核；联合修复不能冒充单因子因果消融 |
| 关键结果与受控对照 | Tables 3–5 pp.8–9；Table 6 p.24；Figs.7–9 | 标准lm-eval/bigcode及README协议映射；local/variants分开标识 | 主要表格已读；Tables 3–6全部图像核对 |
| prefill/decode/KV/退出 | §6 pp.10–11 | HF HuginnDynamicCache.update；generate_with_adaptive_compute；_prefill_with_varied_exit_steps；KLExitEvaluator | 源码路径已读，batch和hardcode限制已标 |
| 连续latent/自推测解码 | §6.3–6.4 p.11 | generate_minimal/generate_with_adaptive_compute continuous_compute；generate_speculative cache rollback | 固定源码验收规则已核；解析证明其一般不保目标采样分布，并核对异质batch回滚和连续初态形状边界；未运行模型 |
| 路径独立与状态演化 | §7 pp.11–13, Figs.11–12, Appendix Fig.22 | examples/trajectory_analysis.ipynb为官方入口；本页公式是诊断建议 | 原文已核；经验现象与定理分开 |
| 后训练接入陷阱 | 论文未规定HF scalar num_steps/labels API | HF RavenForCausalLM.forward/iterate_forward；prepared_attn_mask=None | 已核scalar全no_grad、labels预shift、mask未接入 |
| AMD编译/通信/运行监控 | Appendix A.2 pp.24–25 | recpre/utils.py::_allreduce_chunk_stream/SimpleFabric.backward；model_dynamic.py::monitor_module | 实际分块路径已读；指标分母与软件版本范围限定 |
| 受控算术及判分协议 | Appendix A.1/Fig.14，p.23 | 50次试验、正确和子串命中；固定仓库未定位完整原图脚本/日志 | 图像已核，保留失败/非单调与热图—折线差异；一位小数不冒充精确计数，命中与最终答案分开，标准误注明iid条件 |
| 评测入口身份及版本 | v2 §5.1–5.3/Fig.8；固定仓库README及脚本说明 | README；local_lm_eval.py；misc_benchmark_variants/README；saturation_eval_dist.py | 已核原文与后续入口差别：variants未用于论文、saturation为v3、local仅测试改动；未执行原评测 |
| 原文数据配方与当前清单 | Fig.4、Appendix C Tables7–8；固定sources.yaml | 原文96项、当前113项；17项差集和1处地址差异；Fig.4全部11类比例 | PDF文本与配置逐项比对；原始数据manifest仍缺 |
| 文件权重、保留量与数据入口 | weighted_file_selection、map_to_block_operation、shift_inputs_and_labels | 复制＋无放回子集、舍入、未flush尾部、4097→4096；D1 | 原函数与有限总体小例核验；不反推历史丢弃率或完整launcher |
| 完整推理变体与配对不确定性 | Table6，p.24 | 全六行、compression预算、KL阈值和边际／差值SE分开；配对题目索引 | 原表与协方差公式已核；原始逐题评分待取得 |
