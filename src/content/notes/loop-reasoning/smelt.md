---
title: "SMELT：三预算匹配、规模律与第二次访问机制"
description: "SMELT研究从头预训练的MoE循环模型：当每token算术FLOPs、非embedding参数存储量和KV-cache容量同时受到控制时，循环结构还能带来什么收益？其固定方案把中间50%的物理层组成连续块，整块执行两遍，属于prelude–recur–coda谱系。Loopie采用每层连续两次的顺序；Huginn的输入注入、随机初态和随机深度也不是SMEL"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921"
sourcePageId: "page_0acf2f6c73c48191b460da5f519a7921"
sourceSha256: 4606e82b5561acf7bb76c2be7d72b2580d520314b68bade584ae45f4aeaa7ab6
series: loop-reasoning
references: []
---

<span id="source-block-e99a791e-f656-472b-9730-08b9adc45a62"></span>

[基座矩阵与七篇论文](/notes/loop-reasoning/model-matrix/) · [架构理论](/notes/loop-reasoning/architecture/) · [预训练](/notes/loop-reasoning/pretraining/) · [后训练](/notes/loop-reasoning/posttraining/)

<span id="source-block-235f8e9a-561d-4b9b-8171-3b4b0f89daa4"></span>

## 研究问题和阅读路线

<span id="source-block-1a954aec-5942-469a-b968-6e289fb8af42"></span>

SMELT研究从头预训练的MoE循环模型：当每token算术FLOPs、非embedding参数存储量和KV-cache容量同时受到控制时，循环结构还能带来什么收益？其固定方案把中间50%的物理层组成连续块，整块执行两遍，属于prelude–recur–coda谱系。Loopie采用每层连续两次的顺序；Huginn的输入注入、随机初态和随机深度也不是SMELT的默认机制。[英文原文 2609.01343v3](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)（§3，pp.5–8）

<span id="source-block-08360ae1-fbd8-464a-afc8-3e2291e4f910"></span>

阅读顺序是计算图与三预算配平、三项设计消融、跨规模与稀疏度的配对训练，再到分别拟合两种架构的规模律。随后看下游收益为何超过validation loss的预测，最后用第二次访问的路由、残差和注意力分析理解可能机制。这样可以把“middle50%×2”放回完整的实验论证中。

<span id="source-block-690e5630-2180-4ea9-9e90-5d929ece49d0"></span>

完整架构与训练栈属于作者的proprietary模型体系，附录A（p.30）说明不能公开；训练语料也为内部数据。官方可运行仓库与checkpoint仍未核验，因此SMELT目前适合作为架构、预算和实验配方的参考，不能直接列为可下载的后训练基座。

<span id="source-block-29b1f337-b8a4-40aa-a6e6-ac2951769756"></span>

## 循环计算图与残差缩放

<span id="source-block-f200625f-475b-4891-af0e-825c98d5f408"></span>

物理层数L，中间块长度m=L/2，前缀长度a=⌊(L−m)/2⌋。前缀执行一次，中间块顺序完整执行r次，后缀一次；默认r=2。论文给出的1-based loop span分别为3–7、4–9、6–15、8–22，对应L=10、12、20、30（附录A）。有效层数为：

<span id="source-block-c0ece68a-0ebc-4191-8d4d-4783045d5c2d"></span>

```math
\begin{aligned}
L_{\rm eff}&=L+(r-1)m,\\
h_{\rm out}&=C_\theta\!\left(R_\theta^{\circ r}(P_\theta(E(x)))\right).
\end{aligned}
```

<span id="source-block-d9536041-7ea9-4827-a0c5-6e3253abd00b"></span>

§6.2式8–9明确对loop span内的每个子层残差乘1/r，而不是只对整个block最终输出除r；MoE的输入已包含缩放后的attention写入。

<span id="source-block-6a2f6cea-a20b-4ee3-94f7-8f87c186ecc0"></span>

```math
\begin{aligned}
u&=h+\frac1r\operatorname{Attn}_\ell(\operatorname{RMSNorm}^{A}_\ell(h)),\\
F_\ell(h)&=u+\frac1r\operatorname{MoE}_\ell(\operatorname{RMSNorm}^{M}_\ell(u)).
\end{aligned}
```

<span id="source-block-ccb81ec7-9bf0-4301-85d5-1d976ad6f2a2"></span>

循环区间之外取r=1。MoE每个token选择top-8专家；第二次访问会重新计算router和Q/K/V。对共享参数应用链式法则，梯度等于各访问位置贡献之和：下式说明梯度如何累加，收敛性还需要后续条件。

<span id="source-block-6b5cbf9b-e748-476e-923e-28a601f7c8b2"></span>

```math
\frac{\partial\mathcal L}{\partial\theta_\ell}
=\sum_{v=1}^{r}\frac{\partial\mathcal L}{\partial h_{\ell,v}^{\rm out}}
\frac{\partial F_{\theta_\ell}(h_{\ell,v}^{\rm in})}{\partial\theta_\ell}
\Bigg|_{h_{\ell,v}^{\rm in}\,\mathrm{fixed}}.
```

<span id="source-block-ab7d8eb2-9659-46bb-bdc3-078a644eedf6"></span>

1/r旨在限制相关共享残差的累计幅度；它本身没有推出Jacobian谱半径<1，也没有保证任意增加r会收敛。论文认为共享层从多深度收到梯度可能帮助优化，这是解释假说；实际最优r与宽度缩减共同决定。

<span id="source-block-148e8f89-fc1e-4f10-8322-a52c74295e96"></span>

### 残差缩放能够推出的条件界

<span id="source-block-200b6c43-11f1-4245-8bbb-08c07a5daefb"></span>

把包含预归一化的 Attention 与 MoE 子映射分别记为 Aℓ、Mℓ。总访问次数为 r 时，单层算子和整个中间块都依赖 r，因为残差步长也变成1/r：

<span id="source-block-553f5435-53ef-4c28-839a-dc01631a8d6a"></span>

```math
\begin{aligned}
G_{\ell,r}
&=(I+r^{-1}M_\ell)\circ(I+r^{-1}A_\ell),\\
R_r&=G_{b,r}\circ\cdots\circ G_{a,r}.
\end{aligned}
```

<span id="source-block-807bf14d-f237-43d5-8df4-b92e1866dd62"></span>

假设沿所有访问的可微分支，两个子映射的 Jacobian 算子范数分别不超过统一的 cᴬℓ、cᴹℓ。此条件要求 top-k 切换点另行处理；论文没有测得这些上界。在条件成立时，链式法则与次乘性给出：

<span id="source-block-1a9ea1f5-19cc-45b5-96ed-37a1072c45b7"></span>

```math
\begin{aligned}
\|J(R_r^{\circ r})\|
&\le\left[\prod_{\ell=a}^{b}
\left(1+\frac{c^A_\ell}{r}\right)
\left(1+\frac{c^M_\ell}{r}\right)\right]^r\\
&\le\exp\!\left[\sum_{\ell=a}^{b}(c^A_\ell+c^M_\ell)\right].
\end{aligned}
```

<span id="source-block-915c4ad4-aa55-41bb-9032-6be6e8aa4f8f"></span>

第二步用了 log(1+u)≤u。这个条件界说明，在固定物理中间块和统一 Jacobian 界下，1/r 可以消除“仅由重复 r 次造成的这一上界的指数增大”；它既不小于1，也不证明实际 Jacobian 变大。标量反例已足够排除收缩保证：

<span id="source-block-07ef53f3-be51-4cc2-a31e-e247f3eabf65"></span>

```math
F_r(h)=(1+\alpha/r)h,\quad\alpha>0
\quad\Longrightarrow\quad
F_r^{\circ r}(h)\to e^\alpha h.
```

<span id="source-block-0c8d1f4d-3940-4aec-83b8-cfab3febd061"></span>

对设计动机还可做一个带明确假设的方差计算：若某一坐标上的 r 个未缩放残差写入有相同方差 σ²，任意两次的相关系数均为ρ，则缩放后累计写入满足：

<span id="source-block-0e390c36-1fdb-4d9b-96e6-644bacad435a"></span>

```math
\operatorname{Var}\!\left(\frac1r\sum_{v=1}^r u_v\right)
=\sigma^2\left[\rho+\frac{1-\rho}{r}\right].
```

<span id="source-block-0c2a88f8-26b9-4465-bbb8-32562fc3deb8"></span>

正相关时，1/√r 缩放会留下随 r 增大的相关项，而1/r 将其控制在有限量级。等方差、等相关只用于解释机制，不是论文对实际层状态的统计假设。这里的结论是依据 [原文 §3.1、§6.2 式8–9](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) 的残差位置作出的条件化推导。

<span id="source-block-9ff79018-47d1-4c33-878c-d9c999d79438"></span>

### 有序残差乘积与有限访问的稳定性

<span id="source-block-a96bc680-19cd-48e8-963a-45ba42f88ff7"></span>

逐层 Jacobian 的特征值不足以判定整个循环的扰动放大。令 ν 按真实执行顺序枚举中间块中的层访问，A′ν、M′ν 分别在该次 attention 输入与已写入 attention 后的 MoE 输入处求导，二者包含对应 RMSNorm 的导数。在 top-k 分支固定的邻域内，原文式8–9给出：

<span id="source-block-c10d0ea1-f69d-4fb2-87fd-93ed9ad9d676"></span>

```math
\begin{aligned}
J_\nu&=(I+r^{-1}M'_\nu)(I+r^{-1}A'_\nu),\\
J_\nu
&=I+r^{-1}(A'_\nu+M'_\nu)+r^{-2}M'_\nu A'_\nu,\\
J_{\rm mid}&=J_{rm}\cdots J_2J_1.
\end{aligned}
```

<span id="source-block-e00c1608-a77a-4262-91d3-25cf7a0c776d"></span>

右乘因子先执行；把 attention 与 MoE 的顺序交换，会把二阶项改成 A′νM′ν，差一个交换子。共享权重也不使两次访问的 A′、M′相等，因为它们在不同隐状态上求值。因此通常不能把 Jₘᵢd 写成某一个固定单层矩阵的幂。

<span id="source-block-e7b0dba3-368a-4bce-974b-edc923800efd"></span>

下面是反驳“每个子更新谱半径小于1，所以组合稳定”的二维线性例子；它不是训练所得 SMELT 的测量。两因子都是非正规矩阵，且可写成步长1/r的残差形式：

<span id="source-block-e0acfabb-3d7b-4967-a0c4-1596d8e45d7a"></span>

```math
\begin{aligned}
U&=\begin{pmatrix}\tfrac12&2\\0&\tfrac12\end{pmatrix},
\quad
V=\begin{pmatrix}\tfrac12&0\\2&\tfrac12\end{pmatrix},\\
\rho(U)&=\rho(V)=\tfrac12,\qquad
VU=\begin{pmatrix}\tfrac14&1\\1&\tfrac{17}4\end{pmatrix},\\
\rho(VU)&=\|VU\|_2=\frac94+\sqrt5\approx4.486,\\
A'&=r(U-I),\qquad M'=r(V-I).
\end{aligned}
```

<span id="source-block-ec9e3e11-7ba0-451a-8396-05b31561d11f"></span>

即使不改变矩阵，非正规 U 也有 ‖U‖₂≈2.118>1，而 Uⁿ 最终趋于零；有限次放大与渐近收敛本来就是不同命题。SMELT 默认只访问两遍，直接相关的是实际有序乘积的有限步增益，而不是假设无限次迭代后的极限。

<span id="source-block-8b2c2e11-92e0-4367-b3fc-fad4bda06e47"></span>

一个足够但较强的条件是：沿所研究轨迹的所有 Jν 共享同一正定度量 W，且 JνᵀWJν≤qν²W。把此不等式逐次应用即可得到：

<span id="source-block-7189dbc9-f250-43f0-bd11-3d2d56b50b23"></span>

```math
\begin{aligned}
W&\succ0,\quad J_\nu^\top WJ_\nu\preceq q_\nu^2W,\quad q_\nu\ge0,\\
\|J_{\rm mid}v\|_W
&\le\left(\prod_{\nu=1}^{rm}q_\nu\right)\|v\|_W,\\
\|J_{\rm mid}\|_2
&\le\sqrt{\kappa_2(W)}
\prod_{\nu=1}^{rm}q_\nu.
\end{aligned}
```

<span id="source-block-f71d0a50-202f-46b8-917c-ecddbdb96124"></span>

该条件强调同一个度量与有序轨迹；每个因子各自找到不同度量并不足够。若条件只沿一条轨迹验证，它约束的是该轨迹的局部微扰；要推广为全局收缩，还须对整个相关状态域统一成立。原文没有给出 W 或这些上界。

<span id="source-block-6a49b1c1-af11-4a06-9934-156d2b343f25"></span>

来源：[SMELT §3.1、§6.2 式8–9](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) 给出残差位置与缩放；有序链式法则、反例及共同度量界是由该计算图导出的条件分析。实际增益探针应作用于完整中间块的两次有序访问，保留共享权重，并分别报告各样本，不能将单个物理层的特征值平均代替它。

<span id="source-block-167a091a-6c0d-466f-9079-120befe4d8ae"></span>

#### 算法 有序循环的局部放大探针

<span id="source-block-f57cc080-ef8b-499f-a785-634056323e2b"></span>

**输入** 固定参数下的中间块映射 Φ=R₂∘R₂、完整 teacher-forced 前缀的中间状态 H、文本位置和文档 mask、随机初始方向数及幂迭代次数。

<span id="source-block-6946d48d-7570-4c5e-82e4-452fcb33fe29"></span>

**输出** 对 ‖DΦ(H)‖₂ 的方向搜索估计、所选方向的有限差分核对，以及路由边界和数值异常记录。

<span id="source-block-d8885a7b-c001-4316-8b1b-572cb24657bb"></span>

**执行前提** 参数与随机性固定，前向按真实 attention→MoE 与层访问顺序执行；JVP／VJP 对 H 求导而不更新参数。报告 top-k 边界，不把跨分支有限差分称为同一局部 Jacobian。

<span id="source-block-ad63543a-c3fe-4efd-a1f9-48e0e38d2d5c"></span>

<figure class="research-embed"><iframe src="/research-assets/ab8cd9669466a5363395343d52c934f6c0462669890e3258874ef7d17cd8a22a.html" title="算法 有序循环的局部放大探针" loading="lazy" sandbox=""></iframe><figcaption>算法 有序循环的局部放大探针 · <a href="/research-assets/ab8cd9669466a5363395343d52c934f6c0462669890e3258874ef7d17cd8a22a.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-87c119d9-9f89-459f-8f28-d487ed975013"></span>

**实现对应** JVP 给出 Jv，VJP 给出 Jᵀw；对 JᵀJ 做幂迭代，每个有限迭代结果 ‖Jv‖₂≤‖J‖₂，只是方向搜索给出的下界估计。出现大于1即可证实该点存在局部放大；未找到大于1不能认证收缩。它检验中间映射，不自动包含 prefix／suffix 或证明训练梯度稳定。完整 SMELT stack 未公开，因此这是参考诊断程序，不是作者仓库函数或已完成的模型实测。

<span id="source-block-f4851d4d-461c-4449-98a1-122138e0ee66"></span>

## 三预算匹配：参数、计算和KV分开记账

<span id="source-block-a53309cd-7003-4d64-bed9-c3d8202cedbd"></span>

§3.1式1定义的是compute-equivalent sparsity。F₀、N₀来自同尺度全专家激活的Baseline对照；F包含每次循环执行和按文档平均长度计的attention成本，不能简单用物理active参数×6替代。

<span id="source-block-f1d2ec96-9dee-4fd2-a075-867fbe52af09"></span>

```math
\begin{aligned}
N_{\rm act}^{\rm eq}&=\frac{F}{F_0}N_0,\\
S&=1-\frac{N_{\rm act}^{\rm eq}}{N}.
\end{aligned}
```

<span id="source-block-8713da04-6afd-4c22-a2a6-3292dcba51dd"></span>

“100M／200M／600M／1.6B”是Baseline active非embedding参数的档位名称，不是每个SMELT实际active参数。S=0的SMELT仍可top8／16专家；S=0表示计算强度相当于全激活对照，不能据此标成dense。

<span id="source-block-6d20f154-9a64-4c4f-9cdc-2e0ff2292920"></span>

重复执行会增加计算，也会为不同执行位置产生各自的 K/V 状态；只共享权重，并不会同时消除这两项开销。因此，要把收益归到循环结构，SMELT 需要同时控制每 token FLOPs、参数存储量和 KV 容量。匹配算法先缩小 hidden 来减少每次执行的计算，再增加专家数补回存储容量，最后调整 head_dim 与 GQA 比例，把 KV 容量也拉回相近水平。

<span id="source-block-7d8e6c7d-5045-45ff-a7af-9243c5d28a2a"></span>

下式按执行位置计算 KV。B 是 batch 大小，T 是已缓存的 token 数，b_elem 是每个元素的字节数；系数 2 分别计入 K 和 V。每个执行位置保存多少状态，取决于 KV-head 数与 head_dim 的乘积。即使两次执行使用同一物理层的投影权重，它们接收的状态不同，缓存仍分别计入。三个 δ_Q 随后分别记录相对 Baseline 的匹配误差。

<span id="source-block-5074d3d6-8600-438c-917c-8351f7529d26"></span>

```math
\begin{aligned}
M_{\rm KV}&=2BT\,b_{\rm elem}\sum_{\nu=1}^{L_{\rm eff}}n_{{\rm KV},\nu}d_{{\rm head},\nu},\\
\delta_Q&=\frac{Q_{\rm SMELT}}{Q_{\rm Base}}-1,\\
Q&\in\{F,N,M_{\rm KV}\}.
\end{aligned}
```

<span id="source-block-d5da9d4c-abd4-49f4-9e6f-6b19dea0bc40"></span>

| 原文配对 | Baseline → SMELT | 匹配含义 |
|---|---|---|
| 200M、S≈95%，§3.2 | L=12；H1280→1056；experts192→288 | 执行层12→18；F1.33→1.37×10⁹/token；N3.87→3.89B |
| 同一配对，附录表10 | F比1.029、KV比1.031 | δF=+2.9%、δKV=+3.1%，不是精确相等 |
| 最大档S≈97% | total53.89→53.90B；SMELT physical active1.103B | SMELT H=1848、L=30、experts504／top8，F比1.008、KV比0.984 |

<span id="source-block-84ce4808-853d-4edf-a894-2bc8be8b4739"></span>

12个稀疏cell的最大绝对误差为FLOPs3.9%、参数1.0%、KV3.6%。S=0对照的离散配置匹配更粗：100M格F比1.081，600M格1.046，不能将“<4%”写成全16格统一结论。规模律按各配置、平均文档长度得到的每token算术FLOPs F（原文称measured；未披露计数/测量工具）进入拟合，再在相同(C,S)坐标比较两曲面，避免把残差误差当作严格相等。（表2、附录A表9–10，pp.6、30–31）

<span id="source-block-d636fb95-4961-404c-8ba9-61e2e6cf9045"></span>

### 为什么需要同时改变宽度 专家池与 KV 几何

<span id="source-block-0aa358bf-f2b7-4596-b2e8-dd3e8779d599"></span>

这三项预算的求和对象不同。N 统计唯一存储的参数，同一层重复执行仍只存一份权重；F 统计实际执行，每调用一次都要付出计算；KV 统计不同执行位置产生的历史状态。专家池也遵循这个区别：一个物理层可以存 Eℓ 个专家，每个有 pᴱℓ 个参数，但每个 token 只执行其中 k=8 个。

<span id="source-block-fcf6f8f0-0de3-4414-9ae3-add976d4abb0"></span>

因此，下面 N 的求和沿物理层展开，并纳入整个专家池；F 的求和则沿展开后的执行顺序展开，只为入选专家支付该次计算。这个区别使“缩小单个专家、扩大专家池”成为可用的配平手段，而非简单把所有预算都按循环次数放大。

<span id="source-block-dadc5026-14dd-42e1-af2b-c1087bfa7de1"></span>

```math
\begin{aligned}
N&=N_{\rm other}+\sum_{\ell=1}^{L}
\bigl(p^A_\ell+E_\ell p^E_\ell+p^{\rm router}_\ell+p^{\rm norm}_\ell\bigr),\\
F&=F_{\rm other}+\sum_{\nu=1}^{L_{\rm eff}}
\bigl(f^A_{\ell(\nu)}+k f^E_{\ell(\nu)}+f^{\rm router}_{\ell(\nu)}\bigr).
\end{aligned}
```

<span id="source-block-9ca16168-777f-4421-9113-a9428d1bc6b8"></span>

这里 f 采用与 F 一致的训练计算口径；是否纳入 activation recomputation 等额外执行须由计数协议明确。注意力项还依赖文档长度。该账本不补造作者未公布的 expert intermediate、head 几何或 FLOP 计数器。

<span id="source-block-7d2cac69-8938-4b76-8c24-b061b987a144"></span>

在专家开销主导、专家形状随 hidden 成比例、路由开销可忽略的近似下，令 η=m/L，g=1+(r−1)η。固定物理层数时，三项匹配分别要求：

<span id="source-block-2f837fab-958f-4248-be82-52a36f9f28b6"></span>

```math
\begin{aligned}
\frac{H}{H_0}&\simeq g^{-1/2}
&&\text{来自 }F\propto gLH^2,\\
\frac{E}{E_0}&\simeq g
&&\text{来自 }N\propto LEH^2,\\
\frac{n_{\rm KV}d_{\rm head}}{n_{{\rm KV},0}d_{{\rm head},0}}
&\simeq g^{-1}
&&\text{来自逐执行位置 KV 匹配}.
\end{aligned}
```

<span id="source-block-1f6bb5dd-eb1c-4384-a622-098d98e2d5af"></span>

中间一半执行两次，使执行量的主项变成原来的 g=1.5。按上面的专家主导近似，宽度缩到约0.816后，单次计算减少，能够支付这部分重复；但每个专家也随之变小，所以专家数需增到1.5倍才能补回总参数。KV 的每执行位置宽度则需约为原来的2/3，单独抵消更多缓存位置带来的存储量。

<span id="source-block-eb923b33-5349-49b0-aa98-7535e7b42f53"></span>

200M 配对的实际宽度比例为1056/1280=0.825、专家比例为288/192=1.5，接近这个主项推导；注意力、router、归一化和离散维度使精确配置偏离近似值。若只缩 hidden 而保持 KV 几何比例不变，KV 总量主项仍约增为√g。由此可见，第三项配平解决的是前两项完成后仍会留下的缓存开销。（[原文 §3.2、附录A表9–10](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-6342f0f4-1e0a-4624-bd03-fd62cfbdd8de"></span>

compute-equivalent sparsity 也不是独立于这份账本的专家比例。对共享同一全激活参考的两个模型，有精确关系：

<span id="source-block-199e41ec-ed26-446f-9c88-3d65d6af43cb"></span>

```math
\begin{aligned}
\frac FN&=\frac{F_0}{N_0}(1-S),\\
\frac{1-S_{\rm SMELT}}{1-S_{\rm Base}}
&=\frac{1+\delta_F}{1+\delta_N}.
\end{aligned}
```

<span id="source-block-6467d9ff-25cc-4802-94b9-c4c0e5dfef0b"></span>

所以三预算近似配平也会使计算等价稀疏度近似一致，但无法把它变成每 token 的唯一专家访问比例。工程上应列出可整除的 hidden、head、专家并行候选，逐一生成执行表、统计实际 F/N/KV，再记录各自 δ；三个相对误差必须分别验收，不能让某项的负误差抵消另一项的正误差。

<span id="source-block-ee7661d0-ca71-4fbf-a6db-b60d4712a85e"></span>

## 三项设计消融：选择依据与反例同列

<span id="source-block-eb334845-bf9e-4677-b402-6e6235de93b4"></span>

所有设计选择在200M档进行，每个配置仅训练一次。DCLM误差条来自10个few-shot评测seed，不涵盖训练seed差异。（§3.3–3.5，表3–5，pp.7–8）

<span id="source-block-df25e9f3-f3e8-4a20-bd8a-89f8387404c0"></span>

| 消融与条件 | 具体数值 | 选择依据 |
|---|---|---|
| span，S≈85%、L=12、两次 | 无loop val1.9445；middle50%=1.9257；full100%=1.9322 | middle50%验证损失最低；DCLM Core24.92→27.57 |
| span，S≈95% | 50% val1.8517、Core29.68；67% val1.8544、Core31.78 | 50%不是所有指标冠军；作者按validation loss选span |
| depth，S≈85%、span50% | Baseline物理12最好val1.9445；SMELT物理／执行12／18最好1.9257；14／21反退1.9433 | 选择相同物理深度，更大有效depth/width |
| visits，S≈85%、span50% | r1／2／3／4的val=1.9445／1.9257／1.9385／1.9360 | 两次最好；3→4并非单调更差 |
| 同一visits对照 | DCLM Completion=2.2887／2.2635／2.2779／2.2820 | lower is better；加循环需为预算匹配继续缩宽 |

<span id="source-block-a88b2c53-2fb3-4724-a56b-e7d00fc4fe96"></span>

因此“r=2最好”是这套模型规模、数据、预算约束下的架构选择结论，不等价于同一个已训练模型推理时r=2永远最好。50%优于full-loop也不能推出每种主干的前后层都绝不该共享。

<span id="source-block-36c6ce51-6f8c-42b2-a44d-5a5c883146c3"></span>

### 训练架构消融与推理轮数干预是两个函数

<span id="source-block-15b9c828-06f4-4f57-bdb5-156aff169989"></span>

令 a(r) 包含为满足预算而随 r 改变的 hidden、专家池与 head 配置。论文的访问次数消融实际比较以下第一个函数：

<span id="source-block-0934c525-efa7-4b1a-a26d-38e96758d558"></span>

```math
\begin{aligned}
V_{\rm design}(r)
&=V\!\left(\operatorname{Train}(a(r),D,\xi_0),r\right),\\
V_{\rm runtime}(r)
&=V(\widehat\theta_{r=2},r).
\end{aligned}
```

<span id="source-block-acd420a2-d041-4bc4-aa16-13e8be8db489"></span>

前者同时改变模型形状并重新学习参数；后者固定 checkpoint，仅改变执行程序。对 SMELT，第二种实验还必须指定残差系数是保持训练的1/2，还是随新 r 改为1/r，否则“增加循环数”本身没有唯一的算法含义。表5只能支持前者在该设置中 r=2 最好。（[原文 §3.3–3.5、表3–5](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-4579d6b0-3d42-425b-9c09-2db59f80adf3"></span>

span 与深度的选择同样是有限候选上的验证集选择，而非全体循环架构的最优性证明：

<span id="source-block-0dd28892-03bd-4389-a31e-0377de22156a"></span>

```math
a^*\in\arg\min_{a\in\mathcal A_{\rm tested}}
\widehat{\mathcal L}_{\rm val}
\bigl(\operatorname{Train}(a,D,\xi_0)\bigr).
```

<span id="source-block-b6cb07c6-493e-46e6-a01a-c8ca91db3a86"></span>

评测 seed 与训练 seed 的区别可由全方差公式严格表达。令 ξ 为训练随机性，η 为 few-shot 示例随机性：

<span id="source-block-0baf4ea3-7609-4f9e-85d0-5b1eff30d343"></span>

```math
\operatorname{Var}_{\xi,\eta}(V)
=\mathbb E_\xi\operatorname{Var}_\eta(V\mid\xi)
+\operatorname{Var}_\xi\mathbb E_\eta[V\mid\xi].
```

<span id="source-block-ace46740-7c37-4662-9e49-f360431e6fc5"></span>

同一个 checkpoint 上的十个 few-shot seed 只探测第一类变化在该训练结果上的条件值，无法估计不同训练运行的第二项。选择 validation loss 最优的50% span，与某个 DCLM 指标更偏好67%并不矛盾：优化对象不同，有限样本评估也有误差。

<span id="source-block-39e79f0f-d95d-47ca-8fde-9611d724b1fa"></span>

## WSD训练与配对实验

<span id="source-block-feee2740-6058-46a4-9bb8-c6a83f260e39"></span>

§3.1使用AdamW与warmup-stable-decay；stable阶段恒定LR训练196075步，global batch256、context4096，约1M tokens/update，总约205B。输入采用document packing和segment-level attention mask，不能让不同文档在同一pack中互相看见。

<span id="source-block-6195f01a-fe07-481e-8df2-9727f92a3aa9"></span>

从stable step 10000、20000、50000、100000、150000、196075各分叉一次余弦decay，每条分叉再用10B fresh tokens，最长约215B且不重复数据。每个同尺度／稀疏度配对使用相同token序列；这把数据顺序这个混杂变量固定住。4档规模×4档S×2架构=32条stable run，6条decay分叉形成96个配对、192个评测端点；这些不是192次独立从头训练。（§4.1）

<span id="source-block-27609bd1-8129-47f1-a68e-1ea47f6f2d86"></span>

### 附录 B：全部 16 格曲线支持什么结论

<span id="source-block-3050cc0d-c3ff-4235-b8e6-754abeda20d7"></span>

原文附录 B（pp.31–33）及 Figs.20–23 补全四个尺度 × 四档稀疏度；每个面板以累计训练 token 为横轴，实线为 stable 轨迹，虚线为六个 cosine-decay 分叉。原文报告各格子的末端 SMELT 低于 Baseline；这不是“全时刻均更低”，也不自动扩展成六个分叉端点全部获胜。图中是训练 loss，与规模律使用的 held-out validation CE 分开。[原文附录 B](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)

<span id="source-block-73efdf25-2498-4719-948a-9ac5f383f69a"></span>

| 原图 | Baseline 尺度档位 | 展示范围与统计单位 |
| --- | --- | --- |
| Fig.20，p.32 | 100M | S=0、约85／95／97%；四个成对训练格 |
| Fig.21，p.32 | 200M | 同四档；stable前缀及六分叉相关 |
| Fig.22，p.33 | 600M | 同四档；单独保留各自实际F |
| Fig.23，p.33 | 1.6B | 同四档；不把尺度档位当作SMELT物理active参数 |

<span id="source-block-93548e8d-7c8d-465a-bca6-e8c881a4ab65"></span>

#### 相同 token 终点与相同 FLOPs 终点

<span id="source-block-377c22d8-b7a1-43ab-bc4e-33380302870d"></span>

固定一个 cell、一个明确的训练分支和一种 loss 观测量，设两条曲线为 L_B(D)、L_S(D)，每 token 算术训练 FLOPs 比为 χ=F_S/F_B。若两个查询 token 数均位于所选 SMELT 分支的定义域，在与 Baseline 的 D token 相同计算量下，SMELT 只能使用 D/χ token。严格的差分关系为：

<span id="source-block-3b5ccedb-8f06-4e19-a95e-07d78767a6db"></span>

```math
\begin{aligned}
G_D(D)&=L_B(D)-L_S(D),\\
G_C(D)&=L_B(D)-L_S(D/\chi),\\
G_C(D)-G_D(D)&=L_S(D)-L_S(D/\chi).
\end{aligned}
```

<span id="source-block-7e1de161-c4ae-4b0e-8620-7ae4f7f2cf71"></span>

如果 χ≥1，且该区间的 L_S 绝对连续、非增并满足 |L'_S|≤Λ，则：

<span id="source-block-af9ade36-4ee3-45ae-bfa2-41d87f6434e2"></span>

```math
0\le G_D(D)-G_C(D)\le \Lambda D(1-1/\chi).
```

<span id="source-block-8b2196b0-2ae6-4499-8c7c-9a67f25bcc78"></span>

所以 G_D 大于右侧误差界才是这一固定曲线比较中 G_C>0 的充分条件。非增或导数界未经数据验证时不能直接使用；真实 minibatch loss 也不保证单调。例如 χ=1.039、D=100、L_S(u)=0.95+10/u、L_B(100)=1.052 时，同token收益为0.002，同计算收益为−0.0019。这是纯标量反例，并非论文数据。Table10 的实际F进入共同(C,S)规模律曲面，正是与这种token轴观察不同的比较；WSD不同分支之间不能直接用上述导数连接。

<span id="source-block-a38c19ba-2416-456d-b02f-d4715722192c"></span>

WSD 分叉的输入是一份完整训练状态，而不只是模型权重。下方算法把共同 stable 前缀、六个分叉和配对数据序列写为明确的执行顺序。

<span id="source-block-bdff393d-b88f-417d-8f97-f2ba42893bad"></span>

这是按论文重建的实验流程，不是公开仓库中的函数。原文没有公开完整优化器超参、专家中间维度／head几何逐格明细、训练stack、语料及完整训练命令；不能补造为“已复现”。验证集是39来源、五大类的token-weighted CE，语料已过滤已知benchmark条目。对routing辅助loss、初始化和梯度裁剪的缺项，本页留为复现缺口，而不把Loopie配置搬来代填。

<span id="source-block-17b33976-103a-4407-aceb-0f3c2b38bc0d"></span>

### 分叉实验的状态与统计单位

<span id="source-block-5052cfb9-bdb0-45b0-b329-38fb25f6a0da"></span>

一个可恢复分叉的状态至少包含模型参数、optimizer moments、学习率阶段、数据游标和 RNG 状态。若从同一权重重新初始化优化器，则得到另一个实验，而不是原 WSD 轨迹的自然分叉。packing 的正确 causal mask 还要求同一段文档，形式为：

<span id="source-block-923579cc-9e41-4e21-8ce8-59dd9c22f1b5"></span>

```math
M_{ij}
=\mathbf1\{j\le i\}\,\mathbf1\{\operatorname{seg}(j)=\operatorname{seg}(i)\}.
```

<span id="source-block-a28e900b-b2bf-431c-b71d-387ad6ba423b"></span>

循环访问保持该文档语义；不同 visit 不意味着新文档或新的文本位置。两架构的配对训练必须使用相同 token 序列、document boundaries 与 loss mask，而不只是相同的数据集名字。（[原文 §3.1、§4.1](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-3d18d486-95fe-4030-ba15-4c91a9087633"></span>

在稀疏 fit 中，cell c 表示一个“规模×稀疏度”组合，一整个 cell 包含两架构及每架构六个分叉：

<span id="source-block-1d579abd-1ae3-471e-b590-3e9f2615c480"></span>

```math
\mathcal D_c
=\{(a,c,j,F_{a,c},S_{a,c},D_{a,c,j},\ell_{a,c,j})
:a\in\{B,T\},\ j=1,\ldots,6\}.
```

<span id="source-block-f423606a-5d25-48bc-8aa3-07efb4f029ac"></span>

稀疏区间有12个这样的 cell，共24条 stable 轨迹、144个端点；另外 S=0 的四格构成其余8条 stable 轨迹和48个端点。相同 cell 的多个分叉共享已训练前缀，所以不应把144个 fit 端点当成144次独立从头训练。这正是原文后续以 cell 为单位重采样的结构依据。

<span id="source-block-1e2129be-1973-4f29-a6e1-d7b3c56019bc"></span>

#### 算法 配对 WSD 训练与六个衰减分叉

<span id="source-block-27b013af-dac7-4fc8-b7d9-889d1ef97b8a"></span>

**输入** 两架构的同一预算 cell、共同 token 流、packing mask、六个分叉步和10B-token衰减长度。

<span id="source-block-af8b8f15-5d7f-4cfd-9efc-92d34e560b06"></span>

**输出** 两条 stable 轨迹、六对分叉端点及训练谱系记录。

<span id="source-block-39a1cde2-ea2f-46f0-81c4-26268200c21d"></span>

**执行前提** 这是按论文重写的实验程序，完整 optimizer 配方和训练 stack 未公开；恢复状态包含 optimizer、RNG、scheduler 阶段及数据游标。

<span id="source-block-10be9866-eb98-4bc4-930b-ffaa4d95b34a"></span>

<figure class="research-embed"><iframe src="/research-assets/9079670f119af25bfa0b0a4cdd9f417c0f568aeec8eba76b75f767447d586378.html" title="算法 配对 WSD 训练与六个衰减分叉" loading="lazy" sandbox=""></iframe><figcaption>算法 配对 WSD 训练与六个衰减分叉 · <a href="/research-assets/9079670f119af25bfa0b0a4cdd9f417c0f568aeec8eba76b75f767447d586378.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-378354cb-6460-4535-aa99-fd22b71ea5c1"></span>

**实现对应** SMELT §3.1、§4.1 的论文语义算法。fresh 不要求各独立分叉彼此全不重叠；这里只保留原文保证的未在 stable 阶段见过，以及配对架构的数据一致性。

<span id="source-block-acce7ebe-7394-400f-a42c-1259481d16c5"></span>

## 规模律拟合与计算预算分配

<span id="source-block-577eb64c-3ff9-46eb-b410-3c7047ac85be"></span>

配对训练给出有限网格上的端点，规模律进一步问：如果总计算预算固定，应该把多少计算花在每个 token 上，又应该训练多少 token？原文式2为 Baseline 和 SMELT 分别拟合六个系数。F 是每 token 训练 FLOPs，D 是训练 token 数，S 是上文的计算等价稀疏度。

<span id="source-block-4fc370b5-fe62-44a9-b800-628aba2c7042"></span>

公式把损失写成拟合底部 E 加上两项可约部分：一项随 F 和 S 变化，一项随 D 变化。总预算 C 把 F 与 D 联系起来——单个 token 越贵，同一预算能训练的 token 就越少。因而两种架构的比较不仅是某个端点 loss 更低，还包括各自如何在模型计算和数据量之间取得最优分配：

<span id="source-block-1352bcd2-f06c-439d-85c7-e4583f79d265"></span>

```math
\begin{aligned}
\mathcal L(F,S,D)&=E+\frac{A(1-S)^b}{F^a}+\frac{K}{D^c},\\
C&=FD.
\end{aligned}
```

<span id="source-block-fb132ee1-a855-4c87-a8fe-07cb200790dd"></span>

固定(C,S)后，先用预算关系把 D 换成由 F 决定的量。增大 F 会降低公式中的第一项，却因训练 token 变少而抬高第二项；最优点就是这两种边际变化恰好抵消的位置。在本节正系数条件下，对 F 求导并令其为0，得到原文式3–4。F* 与 D* 给出分配，L* 则是在该分配下达到的拟合前沿：

<span id="source-block-633b12b0-9a2f-48ce-b0fd-39296ceb54b4"></span>

```math
\begin{aligned}
F^*&=\left[\frac{aA(1-S)^bC^c}{cK}\right]^{1/(a+c)},\\
D^*&=C/F^*,\\
\mathcal L^*(C,S)&=E+q(S)C^{-\gamma},\\
\gamma&=\frac{ac}{a+c}.
\end{aligned}
```

<span id="source-block-8daec7f3-5039-4928-b810-e4ff5942eae0"></span>

| 拟合系数 | Baseline | SMELT |
|---|---:|---:|
| E | 1.4439 | 1.4493 |
| A／K | 1.366×10³／1.975×10⁶ | 1.963×10³／5.264×10⁶ |
| a／b／c | 0.3703／0.1530／0.6594 | 0.3892／0.1460／0.7011 |
| loss RMSE | 0.00554 | 0.00952 |
| γ | 约0.237 | 约0.250 |

<span id="source-block-efe40344-97c4-4215-9d96-9ce98a87a7ad"></span>

上表系数来自每架构72个稀疏网格端点：拟合在 log-loss 上使用 Huber objective（δ=10⁻³），由 L-BFGS-B 优化；S=0 因处于不同拟合区间而排除。数据覆盖的累计 compute 范围为1.3×10¹⁹–2.2×10²¹ FLOPs。前沿指数 γ 更高，或拟合底部 E 略有差别，都不足以推出无限预算下的必胜结论；两个 E 相差约0.005 nats，小于两种 fit 的 RMSE。下式据此定义等损失计算节省；后续数值需区分拟合窗内估计与窗外外推。（§4.2–4.3、表6）

<span id="source-block-cb972e4e-9c28-49fb-be94-0907181ab210"></span>

```math
\begin{aligned}
\mathcal L^*_{\rm Base}(C_{\rm ref},S)
&=\mathcal L^*_{\rm SMELT}(C_{\rm tgt},S),\\
\operatorname{CEGain}&=1-C_{\rm tgt}/C_{\rm ref}.
\end{aligned}
```

<span id="source-block-c5eca07d-9ce6-4445-96af-3757804a5c2f"></span>

表7以SMELT目标预算C_tgt=10²⁰ FLOPs为锚点，反解Baseline达到相同loss的C_ref：S≈85／95／97%的节省为10.0／7.8／6.8%。当SMELT目标预算C_tgt=10²¹ FLOPs时，同样反解C_ref，节省为18.0／15.8／14.7%，95%区间分别[8,28]／[10,25]／[8,25]%。区间来自对12个稀疏cell做2000次bootstrap、每次成对重拟合，不能把共享stable轨迹的端点当独立样本。SMELT目标预算C_tgt=10²² FLOPs时，同口径的23.5／20.9／19.6%是拟合窗外外推，区间更宽，后两项下界为0。

<span id="source-block-80585920-ca04-4019-9dc9-c5ea54cb36d3"></span>

表8的最优tokens／dense-equivalent parameter在10²¹、S≈95%为Baseline78、SMELT75，区间[71,90]／[60,101]重叠；三档点估计差<6%。收益主要体现为近似相同allocation下更低loss，未显示必须大幅重新分配模型／数据预算。算术FLOPs节省也不等于实测wall-clock提速，§7明确列为系统研究缺口。

<span id="source-block-828dea6d-435a-4784-bb96-e3120f843b88"></span>

### 最优分配的存在 唯一性与闭式系数

<span id="source-block-3caccb0b-37a9-4cc6-bd8a-2654cda1e0ed"></span>

假设拟合系数满足 A、K、a、c>0，预算 C>0，0≤S<1。记 U=A(1−S)ᵇ，取 x=log F，则固定预算上的可约损失为两个指数项之和：

<span id="source-block-0f29fa4b-e1e0-4ec9-b1a4-1ac792878c52"></span>

```math
\begin{aligned}
f(x)&=Ue^{-ax}+KC^{-c}e^{cx},\\
f''(x)&=a^2Ue^{-ax}+c^2KC^{-c}e^{cx}>0.
\end{aligned}
```

<span id="source-block-38f9cdae-79e1-4e4d-85c8-70efc10ce81c"></span>

两端的函数值均趋于无穷，故驻点是唯一全局最小值。驻点条件表明最优时容量项与数据项并不通常相等：

<span id="source-block-64e97fb5-b4b9-4476-a866-54b83e74e2a5"></span>

```math
\begin{aligned}
a\,U(F^*)^{-a}&=c\,K(D^*)^{-c},\\
\frac{U(F^*)^{-a}}{K(D^*)^{-c}}&=\frac ca.
\end{aligned}
```

<span id="source-block-86bb9e25-f47e-4219-ab98-284b144e0936"></span>

将已有 F* 代回即可补全前沿系数 q，而不把它留作未定义常数：

<span id="source-block-df99a860-41be-42d9-80b3-89e3f683fba2"></span>

```math
\begin{aligned}
q(S)&=U^{c/(a+c)}K^{a/(a+c)}\\
&\quad\cdot\left[
\left(\frac ca\right)^{a/(a+c)}
+\left(\frac ac\right)^{c/(a+c)}\right].
\end{aligned}
```

<span id="source-block-dceb9e54-a79a-45b6-9e76-f5b7af485968"></span>

分配的弹性及 TPP 随预算的变化也随之确定：

<span id="source-block-2d698d95-c3ce-4045-97cd-bf81809d7089"></span>

```math
\begin{aligned}
\frac{\partial\log F^*}{\partial\log C}&=\frac c{a+c},
&\frac{\partial\log D^*}{\partial\log C}&=\frac a{a+c},\\
\operatorname{TPP}&=\frac{6D^*}{F^*}=\frac{6C}{(F^*)^2},\\
\operatorname{TPP}&\propto
C^{(a-c)/(a+c)}(1-S)^{-2b/(a+c)}.
\end{aligned}
```

<span id="source-block-f0e24387-7b71-4dce-8228-a687d83029d7"></span>

F/6 在这里是 dense-equivalent 的计算换算，含注意力开销时不等于真实 physical active 参数。闭式最优解还放松了维度整除、可选模型网格及硬件限制；实施时须检查最优 F、D、S 是否位于已观测范围和可实现配置内，仅 C 落在拟合窗内不自动满足其余条件。（[原文 §4.2–4.3 式2–4、表6与表8](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)；存在唯一性与弹性为本节推导。）

<span id="source-block-8c05b7ca-5678-4a8d-9a8a-2ad58a537887"></span>

### 等损失反解的定义域与不确定性

<span id="source-block-b6aec8f5-0438-4e83-a20c-38e59c65ef9e"></span>

“节省多少计算”需要先固定同一个质量目标。这里先选定 SMELT 的目标预算，并在其最优前沿上得到损失 ℓ*；再问 Baseline 沿自己的最优前沿需要多少预算才能达到这一损失。两者预算之差占 Baseline 所需预算的比例，就是下式的 G。

<span id="source-block-7566622f-7b65-491d-818a-70c9d314f2a9"></span>

反解中 ℓ* 要高于 Baseline 的拟合底部 E_B：只有这时，Baseline 曲线上才有有限预算与目标损失相交。这个条件属于当前拟合模型的定义域，也决定了每次 bootstrap 重拟合后能否计算同一个统计量：

<span id="source-block-127c042b-3026-4350-b382-cae087a97027"></span>

```math
\begin{aligned}
\ell^*&=E_T+q_T C_T^{-\gamma_T},\\
C_B&=\left(\frac{q_B}{\ell^*-E_B}\right)^{1/\gamma_B},
\qquad\ell^*>E_B,\\
G&=1-\frac{C_T}{C_B}.
\end{aligned}
```

<span id="source-block-4dca6f63-20cf-47d8-a5fb-fe8bbdf4fc55"></span>

若 ℓ* 等于拟合的 Eᴮ，反解仅在无限预算极限成立；若低于 Eᴮ，按这一拟合模型没有有限解。不能在这种抽样上继续计算一个看似正常的实数节省率。G=15.8%对应的计算倍率是1/(1−G)≈1.188，而不是 GPU 速度提高15.8%；此处仍只计训练算术 FLOPs。

<span id="source-block-f9a67904-e0fa-4ec7-ba96-e1ee105f1e42"></span>

```math
\frac{\partial\log C_B}{\partial\ell^*}
=-\frac1{\gamma_B(\ell^*-E_B)}.
```

<span id="source-block-53396a41-771b-4263-9888-bc4434e9f925"></span>

越接近拟合损失底部，同样大小的损失误差造成的反解预算变化越大。这解释了 E 的不确定性为何会影响远端外推。两个架构 E 不相同时，仅有 γᵀ>γᴮ 不足以保证节省率永远随预算增长：

<span id="source-block-3b1525d9-d268-492c-8cd9-6db7cf4de4b2"></span>

```math
\frac{d\log(C_T/C_B)}{d\log C_T}
=1-\frac{\gamma_T}{\gamma_B}
\frac{\ell^*-E_T}{\ell^*-E_B}.
```

<span id="source-block-118a3dc0-9459-4e88-b2ea-abe5b7b63b1e"></span>

右侧为负时 G 随预算增加；两底部相等才退化为单看 γ 的判断。论文更高预算的外推应据其有限窗口和 cell bootstrap 解释，不能改写成无限规模定律。使用表6四舍五入的系数回算，目标10²¹、S=95%得到约15.68%，接近但不精确等于表7的15.8%；正式结果应保留表7，不能用舍入后重算值覆盖原报告。（[原文 §4.3、表6–8](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-feb982da-a9c0-4e46-9b0e-ea2cd7ab8a4e"></span>

### 成组配对 bootstrap 的实际估计对象

<span id="source-block-47fbfeb4-50e5-4c01-b8e5-37749edce612"></span>

原文明确对12个稀疏 cell 有放回抽样，每次用同一抽样重拟合两架构，并重算所需统计量。令 mᶜ 是某次抽样中 cell c 被抽到的次数，对架构 a 的重拟合目标为：

<span id="source-block-4597f930-8e2b-4eb3-b77d-07a5e5170996"></span>

```math
\min_{\eta_a}\sum_{c=1}^{12}m_c\sum_{j=1}^{6}
\operatorname{Huber}_{10^{-3}}\!\left[
\log\mathcal L_{\eta_a}(F_{a,c},S_{a,c},D_{a,c,j})
-\log\ell_{a,c,j}\right].
```

<span id="source-block-dde56c63-7f38-4ddd-ac58-0f50c282fe5b"></span>

η 包含该架构自己的六个拟合参数。m 同时作用于两架构和该 cell 的全部六个分叉，保留配对关系及共享训练前缀。表7 CE Gain 用2000次抽样；表8 allocation 使用200次。拟合初值、bounds、收敛阈值及失败处理仍是参考程序需要明示的输入，原文未逐项给出可直接复用的代码。

<span id="source-block-5232c23b-9918-4ed3-aad3-9a35f8813a0b"></span>

该区间表征在这套网格、数据与拟合程序下改变 cell 权重产生的变化；它没有补出缺失的独立训练 seed，也不覆盖更换语料或更换规模律函数族的全部不确定性。算法应记录优化失败或反解无定义的抽样，不能静默删除后仍宣称得到了完整2000次有效的原文区间。

<span id="source-block-45657910-7724-499f-bb26-455acce009c9"></span>

#### 算法 十二个预算 cell 的配对重拟合

<span id="source-block-4a7af20b-6a0c-4d3b-85cc-6720ce7f0a91"></span>

**输入** 12个稀疏 cell 的两架构六分叉端点、共同拟合配置、统计量类型、目标预算与稀疏度。

<span id="source-block-6d2e52ee-3527-4d2f-914d-ed6d4a40d7f1"></span>

**输出** CE Gain 或同预算 allocation 的抽样统计、百分位区间及全部失败记录。

<span id="source-block-9b668842-cdf4-464f-aaa3-a5f5617aac7e"></span>

**执行前提** CE Gain 采用2000次抽样；论文 TPP 表采用200次。同一抽样对两架构使用相同 cell 权重，整组保留所有分叉。

<span id="source-block-84fb270a-5a0f-472d-93be-98e4a546a2f4"></span>

<figure class="research-embed"><iframe src="/research-assets/4aa9187afa639ff35b0557537a4d2d20c7e510f09824a1283709255fedb7e1aa.html" title="算法 十二个预算 cell 的配对重拟合" loading="lazy" sandbox=""></iframe><figcaption>算法 十二个预算 cell 的配对重拟合 · <a href="/research-assets/4aa9187afa639ff35b0557537a4d2d20c7e510f09824a1283709255fedb7e1aa.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-17b72a38-e32b-436f-90e8-60d72b880c93"></span>

**实现对应** 原文 §4.3 的12-cell有放回抽样；表7为等损失计算节省，表8为同C的最优allocation。bounds/初值/失败处理是参考实现显式接口，不称为作者未披露的实际程序。

<span id="source-block-ee32d9ad-a3ce-4c4b-9e34-f22779c65ffc"></span>

## 下游表现及其与验证损失的关系

<span id="source-block-74d257e0-1ac8-4435-b799-fce7356dd08b"></span>

DCLM Core不是原始accuracy平均；附录E式11对每任务按reference baselineρ作居中归一化，再对22任务等权平均：

<span id="source-block-fee9edff-7755-4848-93e0-621ea04e5459"></span>

```math
\begin{aligned}
c_t&=\frac{a_t-\rho_t}{1-\rho_t},\\
\operatorname{Core}&=\frac1{22}\sum_t c_t.
\end{aligned}
```

<span id="source-block-534eaf43-d79e-42d9-97bf-2ace22c6fac0"></span>

DCLM Completion是九个自由回答任务gold answer上的token-weighted CE；MMLU单列5-shot。few-shot任务用10个随机示例seed，7个确定的zero-shot任务一次；截断保留BOS。默认shot协议和另外的ICL扫描不可混用。（§3.1、附录E）

<span id="source-block-4f596c08-b6e0-47b5-b735-404ee3eda580"></span>

§5.1报告Completion胜96/96配对、Core胜83/96；MMLU只对baseline高于chance至少10百分点的30对统计，胜29/30。进一步仅用96个Baseline端点拟合validation-loss→benchmark的单调sigmoid，再计算SMELT相对这条曲线的残差；聚合残差为Completion +8.0 millinats、Core +0.9百分点、MMLU +1.3百分点。它是超出基线校准曲线预测的证据，不是固定loss下随机对照因果实验。

<span id="source-block-52166d00-062f-4ce0-97d5-6f6adc21f7a5"></span>

§5.2在C=10²¹、S≈95%下固定聚合fit的指数、仅重拟合各域截距：Code节省20.4%，Finance16.8%，Math/STEM16.6%，Knowledge14.9%，Web14.8%。按Baseline来源loss分四档，收益18.0／16.3／13.8／15.9%，不是“越难越受益”的单调规律。

<span id="source-block-bd541e16-1836-4bcf-92d0-b78caa39e99c"></span>

§5.3将每个配对的按长度收益归一化后，长四档512–4096 vs短四档32–256的平均比是1.52；加参数／加专家控制为0.98／0.88。16项few-shot任务平均差距从0-shot的0.9百分点增到有示例的1.9百分点；Dyck在32-shot为29.8% vs26.4%。这些结果支持结构化输入与上下文检索方向的研究定位。

<span id="source-block-ee29b480-39cc-47e7-843b-ae2d46cbf0e8"></span>

### 校准残差与长度增益的可计算定义

<span id="source-block-d8d9fcc7-787e-4968-9286-5d9c8773ba49"></span>

下游分析使用96个 Baseline 端点，包括 S=0；这与只用72个稀疏 Baseline 端点的规模律 fit 是两个样本集合。原文对每个指标拟合单调 sigmoid：

<span id="source-block-b9188ea5-9474-41c8-8327-beb4b04703b9"></span>

```math
\begin{aligned}
\widehat y_m(\ell)
&=b_m+\frac{a_m}{1+\exp[-d_m\alpha_m(\ell-\tau_m)]},\\
\delta_i&=s_m\,[y_i-\widehat y_m(\ell_i)].
\end{aligned}
```

<span id="source-block-dc687d97-4211-44a3-a420-ec6b92e06777"></span>

其中 aₘ、αₘ>0；Completion 使用 dₘ=+1、sₘ=−1，准确率类使用 dₘ=−1、sₘ=+1。这个符号选择使正残差始终表示优于 Baseline 的拟合曲线。残差同时包含模型表现差异和校准误差；它不是随机配对到完全相同 validation loss 后的平均处理效应。MMLU 的29/30又条件于 Baseline 已高于随机水平至少10百分点的端点，分母不能改成96。（[原文 §5.1 式6–7](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-b6dd2de2-90c2-4561-bc6d-0cb67ac280e4"></span>

领域分析固定各架构在聚合数据上得到的 a、b、c，再只拟合领域的 E、A、K。因而领域20.4%等数值是“共享指数约束下的前沿反解”，不是把某个领域的一次 loss 差直接换成节省率。按难度分组时，每个来源的 CE Gain 固定，但其所属四分位可随匹配模型对的 Baseline 来源损失排名而变化。（[原文 §5.2](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-7356e15f-2c62-4e50-88b6-e9359e6f9078"></span>

长度图先在每个匹配对内部归一化，再跨对平均。令 Δᵢᵦ 是第 i 对在长度桶 b 的 token loss 改善，则其图形统计量是：

<span id="source-block-87a6d193-d513-4719-9cfc-d8ee30bdfa44"></span>

```math
\begin{aligned}
\Delta_{i,b}&=\ell_{B,i,b}-\ell_{T,i,b},\\
\overline\Delta_i&=\frac18\sum_{b=1}^{8}\Delta_{i,b},\\
u_{i,b}&=\Delta_{i,b}/\overline\Delta_i,\quad
\overline\Delta_i\ne0,\\
U_b&=\operatorname{mean}_i u_{i,b},\\
R_{\rm length}
&=\frac{\operatorname{mean}_{b=5:8}U_b}
{\operatorname{mean}_{b=1:4}U_b}.
\end{aligned}
```

<span id="source-block-dc1fb68b-e5bb-42f6-b744-5198a0894b35"></span>

原文1.52描述这个归一化增益形状的长短比，不表示长文 loss 下降52%，也不是逐对长短比再取平均。接近零的分母会放大形状统计的不确定性；重建图形应保留原始分母和处理规则。ICL 从0-shot到有示例后的差距扩大，仍是不同已训练架构在共同评测协议下的条件比较，不能单由该趋势识别训练时形成了哪一种内部算法。（[原文 §5.3、图11](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-61b9e364-0f45-4aa4-9831-f648d68f9ec7"></span>

### 校准残差的尺度结构与类别聚合

<span id="source-block-6dc333e8-0089-4b7e-94d5-0d9577cbd041"></span>

图9的核心结果包含随规模变化的结构，不能只保留合并后的三项平均值。下表沿用原文四舍五入值：Completion为millinats，Core为centered score的百分点，MMLU为原始accuracy百分点。这里比较的都是SMELT相对Baseline校准曲线的残差，而非同预算配对的原始分数差。（[§5.1、式6–7、图9a](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-f053ed46-880b-48e7-b702-72c039c53034"></span>

| 模型尺度 | Completion 超额改善 mn | Core 超额改善 pp | MMLU 超额改善 pp |
|---|---:|---:|---:|
| 100M | +1.7 | +0.5 | +0.3 |
| 200M | +6.5 | +0.9 | +1.0 |
| 600M | +11.6 | +1.0 | +2.2 |
| 1.6B | +12.3 | +1.2 | +1.6 |
| 全部端点 | +8.0 | +0.9 | +1.3 |

<span id="source-block-2c967e8a-270c-44c3-88a8-5c2c619ea7ab"></span>

Completion与Core的尺度均值逐档增加；MMLU从600M到1.6B下降，原文仅支持两个较大尺度超过两个较小尺度。Baseline曲线的样本内R²分别为0.997、0.974、0.911；高R²表示该函数在这些端点上的拟合程度，不提供独立预测集的误差或跨训练seed的显著性。

<span id="source-block-c7a7b868-556a-4b2b-a400-37c442b8dfa2"></span>

| 尺度 | 阅读理解 pp | 世界知识 pp | 常识推理 pp | 语言理解 pp | 符号问题求解 pp |
|---|---:|---:|---:|---:|---:|
| 100M | +4.7 | +0.7 | −0.6 | +1.0 | +0.3 |
| 200M | +1.7 | +0.2 | +0.4 | +0.7 | 0.0 |
| 600M | +1.5 | +0.3 | +2.4 | +0.4 | +0.9 |
| 1.6B | +1.3 | +0.1 | +1.0 | +1.0 | +1.9 |
| 全部端点 | +2.3 | +0.3 | +0.8 | +0.7 | +0.7 |

<span id="source-block-9f4c5779-734d-4225-8823-15cece081739"></span>

图9b在每个DCLM任务上分别拟合Baseline sigmoid，再在类别内平均centered residual。总体阅读理解最高；1.6B时符号问题求解最高，但100M常识推理残差为−0.6，200M符号项为0.0。不能把“平均超出loss预测”扩成每个类别或每个端点都为正。（[§5.1、图9b与图注](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-73e5b6fb-f2b3-4d3a-8dc9-c6686aebf9f2"></span>

设cᵢₜ是第i个端点在任务t上的centered score，ĉₜ是只用Baseline拟合的任务曲线，𝒯g为一个任务类别，𝓘s为一个尺度的端点。图9b的聚合可以写为：

<span id="source-block-6a6a455f-d774-4fbc-bb60-eaf5518c3e7a"></span>

```math
\begin{aligned}
\delta_{i,t}&=c_{T,i,t}-\widehat c_t(\ell_{T,i}),\\
\overline\delta_{s,g}
&=\frac1{|\mathcal I_s|}\sum_{i\in\mathcal I_s}
\frac1{|\mathcal T_g|}\sum_{t\in\mathcal T_g}\delta_{i,t}.
\end{aligned}
```

<span id="source-block-1f4913cd-1d86-4ae5-ad10-e06d26bfbc2a"></span>

每尺度含四档稀疏度×六个分叉，即24个端点；All行汇总96个端点，未另拟合一个“All曲线”。类别任务数依次为3、4、4、6、5，所以类别的简单等权平均也不是22任务宏平均。若以原始accuracy差解释centered百分点，还须保留每任务的1/(1−ρₜ)缩放。

<span id="source-block-8489d18f-c1dc-4937-afa4-87c84ed748bd"></span>

```math
\begin{aligned}
\Delta\operatorname{Core}
&=\frac1{22}\sum_t\frac{\Delta a_t}{1-\rho_t},\\
\delta_i^{\rm Core}
-\frac1{22}\sum_t\delta_{i,t}
&=\frac1{22}\sum_t\widehat c_t(\ell_{T,i})
-\widehat C(\ell_{T,i}).
\end{aligned}
```

<span id="source-block-452a7d39-d3f0-40d7-b028-54387ab9b6e9"></span>

第二式中的 Ĉ 是直接拟合聚合 Core 的曲线。右侧保留了“逐任务曲线的平均”与“聚合指标曲线”之间的差，因此先逐任务校准再聚合，一般不等于直接校准聚合指标；图9b按任务数加权后也不必精确还原图9a。

<span id="source-block-c6c010d3-96ed-4859-a604-c87e084c21df"></span>

重建这项分析时，应先固定96个 Baseline 端点及各任务的 ρₜ，分别拟合聚合曲线和22条任务曲线；曲线固定后，再评估 SMELT 残差，最后按既定尺度/类别聚合。Baseline 校准不加入 SMELT 点，类别汇总也沿用既定分母，才能保持上述比较的含义。这是原文操作的数学展开；完整拟合代码与逐端点原始分数尚未公开。

<span id="source-block-535980cd-03ae-41d3-ad3e-1463d29cfd10"></span>

来源：[§5.1、图8–9，附录E式11及表11](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)。上述差分恒等式只澄清报告量；“结构化推理收益随容量增强”仍是作者对这些分组结果的解释。

<span id="source-block-7853438c-2190-47fc-9501-2a4a6eab439a"></span>

### ICL 曲线的配对单位与显著性筛选

<span id="source-block-117a1edc-e1d5-4751-9a07-0851bc362b9f"></span>

图11的ICL扫描使用15项可带示例的DCLM任务加MMLU，共16项；扫描k=0、1、2、4、8，QA Wikidata、ARC Easy、CS Algorithms、Dyck Languages、Operators另到16、32。其平均raw accuracy不能与默认shot协议的centered Core直接相减。附录C图24只展示最大shot处增益为正且配对置换检验p<0.05的14项；检验单位是10个示例选择/排序seed，图中先在两架构均高于chance的同一(scale,sparsity) cell内聚合，再跨cell等权平均。（[§5.3、图11、附录C图24](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-434fcd31-14dd-4b0c-b3a5-d46605ffa15b"></span>

这一聚合顺序可以直接写成公式。令 Aₐ,c,s,t(k) 表示架构 a 在 cell c、评测 seed s、任务 t、shot k 上的 accuracy，其中已按 cell 内规则完成聚合。再固定两架构共同可比较的 cell 集合 𝒞ₜ(k)，便可在这套固定 checkpoint 和资格集合上定义曲线及配对 seed 差：

<span id="source-block-a5d22030-8503-45eb-a12b-c381f4a33da3"></span>

```math
\begin{aligned}
\widehat A_{a,t}(k)
&=\frac1{|\mathcal C_t(k)|}
\sum_{c\in\mathcal C_t(k)}\frac1{10}\sum_{s=1}^{10}A_{a,c,s,t}(k),\\
d_{s,t}(k)
&=\frac1{|\mathcal C_t(k)|}\sum_{c\in\mathcal C_t(k)}
\bigl[A_{T,c,s,t}(k)-A_{B,c,s,t}(k)\bigr],\\
\widehat\Delta_t(k)&=\frac1{10}\sum_s d_{s,t}(k),\\
\mathcal S_{\rm shown}
&=\{t:\widehat\Delta_t(k_{\max,t})>0,\ p_t<0.05\},
\qquad|\mathcal S_{\rm shown}|=14.
\end{aligned}
```

<span id="source-block-4a05fa15-9734-44af-ae2b-f2df84b97eac"></span>

这里没有把原文未说明的cell内资格判定补成源码：chance条件究竟在哪个shot或端点层应用、置换检验采用单侧还是双侧及多重比较处理，仍需作者的分析程序澄清。比较两模型时应共享同一评测seed对应的示例及顺序；若模型各自独立抽示例，便改变了这个配对统计量。

<span id="source-block-9a6736bd-c329-4743-8f91-f564d2ca293a"></span>

不等大小的cell尤其不能直接混成一个端点池。例如一个cell有两项差值0.02、0.04，另一个有四项−0.01，则先cell内平均再跨cell平均得到0.01；端点池平均仅为0.00333。这是聚合顺序的示例，不是论文原始数据。

<span id="source-block-feb56a0d-877f-4ad8-bac7-f370abaa3ce8"></span>

```math
p_{\rm two\mbox{-}sided}
=2^{-10}\sum_{\varepsilon\in\{-1,+1\}^{10}}
\mathbf1\!\left\{
\left|\frac1{10}\sum_s\varepsilon_s d_s\right|
\ge\left|\frac1{10}\sum_s d_s\right|
\right\}.
```

<span id="source-block-dd1ea513-40e3-441d-a13d-d29fc9f61e1d"></span>

上式给出可枚举1024个符号配置的双侧参考定义，条件是零假设下各配对标签可交换且共同资格集合预先固定；它不声称还原了作者未公开的单双侧细节。十个同号非零差值的示例双侧p为2/1024，而十个差值全零时p=1。按最大shot筛选后展示的14条曲线，不能再次用作“这14项是预先指定的独立验证集”的证据，也不能把评测seed显著性解释为换训练seed仍成立。

<span id="source-block-eedbfc78-f96b-4b22-98f2-f7f96aa6cbfd"></span>

要保留这项配对关系，每条评分记录都需标明 task、cell、endpoint、eval_seed 和 shot，并保存实际示例 ID、示例顺序、保留 BOS 后的截断信息，以及两模型的 accuracy 和资格 mask。分析时先按同一 seed 配对，再按 cell 等权汇总，最后执行已声明的置换检验和展示筛选。公开论文界定了这一统计层级，但尚未提供可直接调用的 SMELT 评测函数或完整原始记录。

<span id="source-block-afe5e556-daf1-4242-9c8c-d1fdb46acedf"></span>

## 第二次访问的路由与注意力

<span id="source-block-842ebce8-e7be-40c1-92d4-a3c35a2ac44c"></span>

§6在1M-token held-out样本上，对全部16个实验格进行probe。下面的统计分别回答两个问题：第二次是否访问相同模块，以及它是否读取、写入了相同信息。两者需要不同的测量。

<span id="source-block-192bd1b4-042c-481b-a9ad-fd70bb291bff"></span>

| 测量 | 原文发现 | 能支持什么 |
|---|---|---|
| top8专家交集 | S=0近8个，S≈97%约2–3个；随机期望64/n | 部分复用、部分改路由，不能说第二次总取全新专家 |
| residual更新 | 两次update cosine为正；所有格的四类norm ratio均>1，约1.2–3.5× | 第二次通常更强且部分同向，不是简单抵消第一次 |
| 1.6B、S≈85%同物理层update cosine | 均值0.56；非对应跨visit层均值0.16 | 对齐与共享物理层有关 |
| Q/K vs V | Q/K cosine0.89–0.93；V0.65–0.74 | 检索坐标较稳，读取的内容变化更大 |
| attended top8 token交集 | 同head跨visit56–66%，不同head控制28–34% | 检索位置复用不是纯位置偏置 |

<span id="source-block-3d00bdd9-1f7a-4226-9250-8b775956aefe"></span>

Dyck case选择“单独消融后最增加Dyck perplexity”的attention head；1.6B、S≈85%、4-shot的三处目标中，BOS attention mass从0.60到0.02，示例答案mass从0.24到0.85（图18，p.22）。在一般验证数据上也观察第二次visit降低sink，方向与普通baseline深层sink增加趋势相反。

<span id="source-block-17172fdf-fe23-466e-9bd4-fbb7584042f5"></span>

这是一组描述性与局部消融证据。作者未证明sink减少就是全部性能增益的因果中介；§7明确留待因果分析。第二次仍须重算Q/K/V，“Q/K方向相似”不允许把第一轮缓存直接冒充第二轮KV，也不推出可省略该轮Attention。

<span id="source-block-9c37951f-6475-4e36-aae2-ef46de8006a3"></span>

### 附录 D：跨尺度 sink 证据与可识别的干预

<span id="source-block-3e206567-cfd5-49fc-8c72-662ffbb54d9e"></span>

Fig.25（p.35）把 Fig.19b 的1.6B比较扩展到100M、200M、600M、1.6B，限定 S≈85%及1M-token held-out样本。原文报告 Baseline 的 sink 随深层通常增大，而重复块的第二次访问在各尺度低于第一次。横轴分别标记物理层和展开后的执行层；“同一物理层两次访问”与“较晚执行深度的不同层”须分别对齐。该图描述 segment-start mass，不是整个性能增益的因果中介估计。[§6.4、附录 D、Fig.25](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)

<span id="source-block-e6d2d88e-2037-4328-ae8a-f9ec5f72a605"></span>

sink 统计只告诉我们：attention 把多少概率分给了 segment 起点。要理解它对输出意味着什么，还需要看这些概率乘上的 value。固定一个 query、物理层和 head；在合法 causal keys 中，将该 query 所属 segment 的起点记为 b。使用关闭 attention dropout 时的归一化概率 A，而非 dropout 后随机缩放的权重；此时 A≥0，且各合法 key 的概率和为1。

<span id="source-block-6a5314b8-d404-4818-ab7c-eb63c9e9b17c"></span>

对访问 v=1,2，s_v=A_{v,b} 是起点分到的概率。只要 s_v<1，便可把其他 key 的概率重新归一化，得到它们的 value 条件均值 μ_v。于是 head 输出成为“起点 value”和“其他 key 的平均 value”的加权和；下面再把两次输出之差拆开，分别追踪概率和内容的变化：

<span id="source-block-a1bf2b97-117c-42d2-b8f9-62e4ed0900fe"></span>

```math
\begin{aligned}
\mu_v&=\sum_{j\ne b}\frac{A_{v,j}}{1-s_v}V_{v,j},\qquad
 o_v=s_vV_{v,b}+(1-s_v)\mu_v,\\
o_2-o_1&=(s_2-s_1)(V_{1,b}-\mu_1)\\
&\quad+s_2(V_{2,b}-V_{1,b})+(1-s_2)(\mu_2-\mu_1).
\end{aligned}
```

<span id="source-block-86e88f6d-b0c2-4ac2-8ca9-b5f7b5ae2b32"></span>

第一项把第一次的 value 和非起点条件分布固定，只改变起点分到的概率；它描述单独重分配概率会怎样改变输出。第二项记录起点 value 本身的变化，第三项记录其余 key 的条件均值变化，其中既有其他 key 概率的变化，也有 value 和表示的变化。实际第二次访问会同时改变这些量，所以 s_2<s_1 只给出第一项中一个系数的符号，最终输出方向和答案 loss 仍要看其余向量及后续网络。

<span id="source-block-4c07dc74-e519-42b4-b749-3d1b64919f2f"></span>

如果要单独检验起点概率的作用，可以在同一次访问上固定全部 value 和非起点相对概率，只把起点质量改为预先声明的 s'∈[0,1]。下面的重标定保持总概率为1，也使这项干预的局部输出变化能够精确计算：

<span id="source-block-4ff162db-c6c6-45d9-b4fe-4687cda50fbb"></span>

```math
A'_b=s',\qquad A'_j=\frac{1-s'}{1-s}A_j\ (j\ne b),\qquad
 o'-o=(s'-s)(V_b-\mu).
```

<span id="source-block-a7117686-a6ec-45cc-b8da-c8ad345cafb5"></span>

当 V_b=μ 时，起点与其余 key 提供相同的向量；概率即使大幅从起点移走，当前 head 输出也不变。这个已有的退化情形说明，减少 sink 是否有用取决于被重新加权的内容，而不只取决于概率降了多少。若 s=1，非起点条件分布未定义，则应跳过该次干预或另行规定分配，避免除以零。

<span id="source-block-c1aa353e-3064-42d4-b5e3-6e9e9ddfaefc"></span>

取得局部输出差后，还要乘固定输出投影中该 head 对应的列块 W_{O,h}，并按原模型顺序重算后续状态，才能比较答案 loss 或任务结果。这样得到的是明确 head／query 干预的效果；跨 head、跨尺度的平均 mass 仍应各自汇总，不能直接放入同一个向量公式。

<span id="source-block-5c20beb4-f3f4-4ab4-8d1d-ec719ca9b76d"></span>

#### 算法：固定 KV 的 sink 质量干预

<span id="source-block-c232546b-d7f5-4db9-84c5-2826bc4657c7"></span>

输入为已固定的模型／前缀状态、query／head／物理层／visit、合法key及目标质量；输出为局部恒等式误差、配对任务差与干预范围。以下是由原文问题推导的参考检验，尚非作者公开函数或已完成的模型实验。

<span id="source-block-849c3c19-575a-42a7-8cef-4a997ffad64a"></span>

<figure class="research-embed"><iframe src="/research-assets/97c4b81cdf3f097661e6214f167e125fc810b79175bf7424e924993e03a03db3.html" title="固定KV的sink质量干预参考算法" loading="lazy" sandbox=""></iframe><figcaption>固定KV的sink质量干预参考算法 · <a href="/research-assets/97c4b81cdf3f097661e6214f167e125fc810b79175bf7424e924993e03a03db3.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-46259959-9310-4acd-9087-d03fa52bf2e0"></span>

[循环研究\_原文附录与机制复核\_20261108.zip（原页附件）](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921#46259959-9310-4acd-9087-d03fa52bf2e0)

<span id="source-block-dbd4b326-edc8-47d5-b94e-ac179bd59dd8"></span>

本轮核对包保存完整原表摘录、六篇英文图表编号目录、7份固定源码／配方、F1–F7、预算与sink反例、可选初始化条件分布及13行参考算法。七组检查通过；500组向量恒等式最大误差约8.9×10⁻¹⁶。新增8段原生数学公式；内容ID与原有算法保留，392条内部链接无失效目标。包SHA-256：1314214cd07a9c4a7fa2bc3f868ba96dfee345ed428df2ca47085e03b092dfb1。检查限原函数替身与标量／向量计算；完整模型、GPU、原始配对评分和浏览器排版尚未验收。

<span id="source-block-2bfe4876-9a42-4e7f-9b74-fc97f22e66ae"></span>

### 专家复用与检索相似性的零假设

<span id="source-block-f75fd89f-a32e-47dc-8665-30f0f4583dc0"></span>

令两次选中的专家集合为 S₁、S₂，大小均为 k=8。若每次都从 E 个专家中独立均匀选取 k 个，则交集 O 服从超几何分布：

<span id="source-block-f79df57e-5987-4113-9ad1-c3807fbbe539"></span>

```math
\begin{aligned}
\Pr(O=o)&=\frac{\binom{k}{o}\binom{E-k}{k-o}}{\binom Ek},\\
\mathbb EO&=k^2/E,\\
|S_1\cup S_2|&=2k-O.
\end{aligned}
```

<span id="source-block-c4cf70d8-1ab3-40fe-bce2-8f09c640c65b"></span>

因此交集2–3对应每个 token 两次总共访问13–14个唯一专家，但仍执行16次专家调用。若专家本来就被不均匀使用，即使两次路由独立，交集期望也应改为：

<span id="source-block-e02c3427-420b-40de-a73c-5ed616e44e79"></span>

```math
\mathbb EO=\sum_{e=1}^{E}
\Pr(e\in S_1)\Pr(e\in S_2).
```

<span id="source-block-bafa312d-91a3-4f2d-96eb-d0ec9a783f7c"></span>

超过64/E可以排除均匀独立基线，却不能单独排除由热门专家造成的复用。若要进一步检验跨 visit 的专门依赖，可在同层、相近位置与文档条件内置换第二次路由，保留各专家边际频率；这是一个补充研究设计，论文图12没有报告该置换实验。（[原文 §6.1、图12](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-fe20f81d-1384-4637-8ffd-ffc3647be396"></span>

残差更新的正 cosine 只表示坐标方向部分一致。它与写入大小的关系是代数恒等式：

<span id="source-block-5cdd075a-56ff-43bb-98cd-4a67aee401a9"></span>

```math
\|\Delta_1+\Delta_2\|^2
=\|\Delta_1\|^2+\|\Delta_2\|^2
+2\|\Delta_1\|\|\Delta_2\|\cos(\Delta_1,\Delta_2).
```

<span id="source-block-8f9b1211-4823-40e6-a088-ebf921fa09fc"></span>

这个式子不把两次写入之和当作删去某次 visit 的因果效果：中间其他层和后续非线性也会改变。对 Q/K/V，方向相似与输出近似又是不同问题。设两次 Attention 的概率行向量为 p₁、p₂，value 矩阵为 V₁、V₂：

<span id="source-block-5ca16861-a706-437b-9be3-d5e4992a5828"></span>

```math
p_2V_2-p_1V_1
=(p_2-p_1)V_1+p_2(V_2-V_1).
```

<span id="source-block-a67c8a3e-efb9-4f9e-9e13-a33c1fe69375"></span>

即使概率分布接近，变化的 value 仍可带来不同输出；仅 Q/K 的 cosine 较高还没有给出概率分布的误差界，因为范数和 softmax logit 尺度也重要。这说明“检索位置较稳定、读取内容改变”可以与第二次访问有用同时成立，也说明不能据此直接复用第一轮 KV。（[原文 §6.2–6.3、图13–16](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-20a71a4e-699e-4319-b4a3-343b052a1b58"></span>

若需要因果对象，应在固定 checkpoint 和同一 held-out token 上定义明确干预：

<span id="source-block-83d70b45-7ddd-4275-b4af-eff2985b1620"></span>

```math
\Delta_{\mathcal I}
=\mathbb E\bigl[\ell(f_{\theta,\mathcal I}(x),y)
-\ell(f_\theta(x),y)\bigr].
```

<span id="source-block-76bf037a-c621-426a-8bc6-5fdb00e4a446"></span>

固定第二次专家身份、替换第二次 value、重分配 BOS attention mass 是三种不同干预，必须分别约定 router 权重如何归一化、缓存和剩余计算如何处理。原文选出最影响 Dyck perplexity 的单 head 消融，支持该 head 在该任务上的作用；它没有将整个 SMELT 相对 Baseline 的增益归因于 sink 的因果中介。图18的0.60→0.02还条件于所选 head、示例及目标位置，不能替换成所有 head 的平均。（[原文 §6.4、图18、§7](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-d9c6f00f-dfb8-4826-995f-685bcef3f295"></span>

### RMSNorm 的方向增益与相对写入量

<span id="source-block-a4a99d36-67ee-499a-9f6f-5e29d2f06880"></span>

原文图14的第四项 norm ratio 测的是送入下一子层的 RMSNorm 输出。RMSNorm 有逐坐标可学习 gain，因而归一化后的范数仍可随方向变化。保留有限 ε，令 G=diag(g)、q=‖h‖₂²>0、u=h/√q、H 为 hidden dimension，则有精确恒等式：

<span id="source-block-c2bc1e1f-ec8a-4322-b746-e185416afb29"></span>

```math
\begin{aligned}
n(h)&=\frac{Gh}{\sqrt{q/H+\varepsilon}},\\
\|n(h)\|_2^2
&=H\,\frac{q}{q+H\varepsilon}\,u^\top G^2u,\\
\frac{\|n(h_2)\|_2^2}{\|n(h_1)\|_2^2}
&=\frac{q_2/(q_2+H\varepsilon)}
{q_1/(q_1+H\varepsilon)}
\frac{u_2^\top G^2u_2}{u_1^\top G^2u_1}.
\end{aligned}
```

<span id="source-block-038d8690-cead-4bd6-9355-fc1f4e5806a4"></span>

比值要求分母非零。最后一个因子是 gain 平方的 Rayleigh 商比；同一物理层两次访问共享 G，却可沿不同方向得到不同能量。若 ε=0，或两次 q 都远大于 Hε，径向因子分别严格或近似消失；这时范数变化确实来自方向与各向异性 gain。有限 ε 时不能把尺度作用绝对说成零。

<span id="source-block-6a13cc4e-8031-4d21-9ccd-667867d69a1c"></span>

例如 H=2、g=(1,3)、两次输入分别是两个单位坐标向量，即使输入范数相同，归一化输出的范数比也为3。若所有 gain 的绝对值相等，ε=0时这种纯方向放大就消失。此例解释原文机制的可能性，不以人造 gain 代替实际测量。

<span id="source-block-0e3db67b-2cb2-409e-94bd-5c7e78b2a110"></span>

绝对写入更大与相对于当前 residual 的写入更大，还须分开。定义 a 为两次 update norm 之比，b 为两次 residual norm 之比，则：

<span id="source-block-a9b07c81-c336-42a9-a926-f26ea4f549d7"></span>

```math
\frac{\|\Delta h_2\|_2/\|h_2\|_2}
{\|\Delta h_1\|_2/\|h_1\|_2}
=\frac ab,\qquad
a=\frac{\|\Delta h_2\|_2}{\|\Delta h_1\|_2},\quad
b=\frac{\|h_2\|_2}{\|h_1\|_2}.
```

<span id="source-block-165837c9-36df-47ad-9813-7ae7e4917f38"></span>

a>1不能推出 a/b>1。原文图13在 S≈85% 的四个尺度均报告第二次绝对写入更大，但相对写入在三个较大尺度更一致；100M 的几个早层相近或反转。这一原文例外应与图14按 cell 汇总的1.2–3.5倍一起保留，不能改成每个 token、每层、每项比例均增加。

<span id="source-block-7154150a-c28c-44c5-866b-b9d78fb44279"></span>

工程 probe 应按同一文档／token／物理层／visit 记录 q、h方向、gain、RMSNorm输出、attention写入、MoE写入及写入前 residual，再使用相同聚合协议作跨访问比较。不同问题分别计算绝对 norm ratio、相对 update ratio与方向 Rayleigh 商；先取比再平均和先平均再取比不能混用。原文没有公开这些 probe 的完整归约代码。（[§6.2、图13–14](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)；上式为 RMSNorm 定义的代数推导。）

<span id="source-block-cba4cbaa-e80d-4e1d-a20d-e78af0be10b0"></span>

### 从注意力重合度到可检验的输出误差界

<span id="source-block-86d8c98f-9229-4873-ac61-640f6174289f"></span>

图17除了 top-1／top-8 集合，还给出完整注意力概率的 mass overlap。对同一允许 key 集合上的两次分布 p₁、p₂，这个统计量与总变差有精确关系：

<span id="source-block-7fbf2f66-c81d-4a82-a682-51e5ece8b4e6"></span>

```math
\operatorname{Overlap}(p_1,p_2)
=\sum_j\min(p_{1j},p_{2j})
=1-\frac12\|p_2-p_1\|_1.
```

<span id="source-block-06c9d9bb-bc76-43d1-a6f8-855de2a86741"></span>

证明直接用 min(a,b)=(a+b−|a−b|)/2 并求和。top-8交集只比较排名集合，不能替代这个概率距离；集合完全相同也允许质量在集合内大幅转移。原文56–66%与28–34%是集合比例，不能代入上述 mass-overlap 恒等式。

<span id="source-block-19171e96-56a6-4f4a-b19b-7822e9b84643"></span>

为了把 Q/K 相似提升为条件化误差界，先对同一 query 定义 logits zᵥⱼ=qᵥᵀkᵥⱼ/√d。softmax 对常数平移不变，因此相关的是 logit 改变量的中心化幅度：

<span id="source-block-f2dd135b-4a82-4d5e-ad62-fa209a4ffde3"></span>

```math
\begin{aligned}
\delta&=z_2-z_1,\\
\eta&=\min_c\|\delta-c\mathbf1\|_\infty
=\tfrac12(\max_j\delta_j-\min_j\delta_j),\\
\|p_2-p_1\|_1&\le\min(2,\eta).
\end{aligned}
```

<span id="source-block-f17507a5-3fd8-4468-836a-a401756d52ac"></span>

最后一个不等式可从 softmax Jacobian 严格推出。对任意向量 v，Jₛv 的第 i 项为 pᵢ(vᵢ−Eₚv)，于是其1范数不超过 √Varₚ(v)≤‖v‖∞；沿 z₁ 到 z₂−c1 的线段积分，再对 c 最小化即可。该界允许很宽的上界，不声称实际分布一定接近。

<span id="source-block-04cc8f4a-0ef9-4c43-aa1f-48b0d73e1b75"></span>

由点积展开还得到每个 key 的 logit 误差界；cosine 只有与范数一起出现才提供向量误差控制：

<span id="source-block-be83d761-6b2c-4e9f-905a-c7332194d191"></span>

```math
\begin{aligned}
|\delta_j|
&\le\frac{
\|q_2-q_1\|_2\|k_{1j}\|_2+
\|q_2\|_2\|k_{2j}-k_{1j}\|_2}{\sqrt d},\\
\|q_2-q_1\|_2^2
&=(\|q_2\|_2-\|q_1\|_2)^2\\
&\quad+2\|q_2\|_2\|q_1\|_2
[1-\cos(q_1,q_2)].
\end{aligned}
```

<span id="source-block-e9f20ab2-0781-4d7f-9dd7-7d31010077d8"></span>

反例取一维正数 q₁=q₂=1，两个 key 从(1,2)变成(100,2)：所有对应 Q/K 的 cosine 都为1，但概率从约(0.269,0.731)变成约(1,0)，1范数变化约1.462。故图16的平均 cosine 0.89–0.93不能推出逐 query 的小误差上界。

<span id="source-block-4ae8b471-df6e-48ce-9462-a4316318daa1"></span>

以下先限定单个head h。令V_max,h=max_j‖v_{1,h,j}‖₂，ΔV_max,h=max_j‖v_{2,h,j}−v_{1,h,j}‖₂，η_h为该head先前定义的logit扰动界。W_{O,h}是固定输出投影W_O中对应head的列块，O_v^(h)=W_{O,h}o_{v,h}仅表示该head的输出贡献。设b_h为其投影前上界，完整多头输出由concat后统一投影：

<span id="source-block-43e5368a-a82d-4734-bea4-344302470cd7"></span>

```math
\begin{aligned}
b_h&=V_{\max,h}\min(2,\eta_h)+\Delta V_{\max,h},\\
\|O_2^{(h)}-O_1^{(h)}\|_2&\le\|W_{O,h}\|_2b_h,\\
O_v&=W_O\operatorname{concat}_h(o_{v,h}),\\
\|O_2-O_1\|_2&\le\|W_O\|_2\left(\sum_h b_h^2\right)^{1/2}.
\end{aligned}
```

<span id="source-block-b0015b27-6816-4c19-b3e1-98530cdba84b"></span>

这给出一项可计算的条件验收：同时量化概率变化与 value 变化，再乘输出投影的放大；只控制 Q/K 方向不够。即使该层输出误差小，最终 logits 或正确率仍取决于后续网络的传播，不能省略后缀影响。 多头界来自concat向量的平方范数相加，不要求各head独立。例如两个head各只有一个合法key，value均从0变1，W_O=I₂，则完整输出差为√2；把单head的b=1直接当成完整输出界会失效。

<span id="source-block-4ca8018b-e7a2-4427-aa37-3fdcd5e447ab"></span>

计算上述误差量时，先按同一物理层、head、文本 query 和文档边界对齐两次 probe；随后只提取合法 causal keys 上的 logits，避免对−∞ mask作相减。Q-head 与 KV-head 还需按 GQA 的真实映射对应，再记录 η、mass overlap、value 变化及输出差异。比较以完整重复前向为参照，已有 visit cache 保持不变。

<span id="source-block-8ce00398-8c09-499e-a9f4-acb458ed41b5"></span>

这些步骤将输出误差界转成了可检查的量。SMELT 未公开源码，因此它们只是依据 [§6.3 式10、图16–17](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) 得到的参考验收条件；统计图本身尚不能验证近似 KV 复用。

<span id="source-block-2dabc356-5aa3-4e5d-8c8f-36fbb62fc75c"></span>

## 工程拆解与后训练对接

<span id="source-block-5cd1cec1-5a22-4927-bd6e-430c0512183d"></span>

下面将论文计算图写成参考算法，用来明确输入、状态和执行顺序。完整训练源码尚未公开，因此这些步骤是实现规范，而非可直接调用的官方仓库函数。

<span id="source-block-f7f48cb2-233a-4e6d-b34c-2592c86a1e65"></span>

下方前向算法按 prefix、中间块的全部访问、suffix 构建虚拟执行序列；每个虚拟位置引用一个物理参数对象，并有独立缓存地址。它是论文计算图的参考实现规范，未对应公开 SMELT 仓库。

<span id="source-block-b3046ae0-423c-4a20-b4ff-a8987b1e589e"></span>

训练时按展开图反传到共享参数；增量推理时每个执行位置缓存各自K/V，存储权重仍按物理层共享。精确验证需比较teacher-forced完整序列与增量解码、检查segment mask与位置、统计F/N/KV实际值；这属于未来复现的验收条件，本轮没有运行模型或宣称验收通过。

<span id="source-block-25188790-44df-467d-a3a0-6e1c6a718f82"></span>

后训练研究应把SMELT当配方与实验范式：固定r=2先测SFT／偏好／可验证RL能否转化长上下文和ICL优势，再考虑学动态计算。新停止策略会改变执行轨迹、每visit cache、router负载和预算定义，必须重新做三预算／实际延迟对照。由于未取得公开checkpoint，不能把它和已开放权重模型放进同一个“今天可直接微调”的候选集合。设计消融仅在200M档完成，放大后最优span／r、真实硬件效率、因果机制、完整recipe都保留为开放问题。（§7、附录A）

<span id="source-block-65c5e65a-0ae7-49e1-862d-b100c4473808"></span>

### 论文语义前向的输入与输出契约

<span id="source-block-d0184eb6-8901-43ea-88c9-02fca59e81c7"></span>

令 eν 是展开表中第 ν 个执行位置对应的物理层，cν 是它的残差系数。完整前缀训练与增量解码都使用同一张表：

<span id="source-block-2dcbac21-cbcf-4daf-9d7b-32659989136a"></span>

```math
\begin{aligned}
\mathcal E&=(\mathrm{prefix},
\underbrace{\mathrm{middle},\ldots,\mathrm{middle}}_{r\text{ 次}},
\mathrm{suffix}),\\
c_\nu&=\begin{cases}
1/r,&e_\nu\in\mathrm{middle},\\
1,&\text{其余位置}.
\end{cases}
\end{aligned}
```

<span id="source-block-5405a8c3-bcc7-4752-9d6f-6a87e22eefdb"></span>

参数由物理层索引查找；K/V 与激活追踪按虚拟位置索引。这分别对应“存储共享”与“状态不共享”。训练路径保留完整展开图，通过所有调用向同一个参数累加梯度；增量路径要求每个槽的历史内容来自相同参数、位置和文档 mask。Attention 的缩放先写入 h，再以更新后的 h 计算 MoE，不能把两个未缩放分支并行计算后一起除 r。

<span id="source-block-edeb0ee2-aff2-4865-ab80-72a7aee6d8a7"></span>

公开资料尚未给出完整 router score、辅助损失及每格专家几何，因此算法将 route_top8 与模型构造视为显式接口，不自行填成其他论文的实现。复现前必须先补齐这些接口的权重与配置，再比较完整前缀和增量 logits，以及实际 F/N/KV；公式正确不等于这些工程缺口已经完成。（[原文 §3.1–3.2、§6.2 式8–9、附录A](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)）

<span id="source-block-b50bf5be-63e9-483e-9688-a5bdd8e53aa8"></span>

#### 算法 中间块两次访问的参考前向

<span id="source-block-35582ee2-fde0-4ea1-b966-23c42db46585"></span>

**输入** 物理层、prefix/middle/suffix、总访问数、文本位置、文档 mask 与可选的逐执行位置缓存。

<span id="source-block-0a894fd6-3c69-4653-917b-05d707b82993"></span>

**输出** 下一 token logits 及可选的逐执行位置缓存。

<span id="source-block-16427599-527f-41b5-a083-5c635df42f22"></span>

**执行前提** 默认 r=2，router 和各层几何必须由真实配置补齐。训练保留完整展开图；增量推理要求缓存与当前权重和前缀一致。

<span id="source-block-54ae69d6-b8e8-42ad-9008-c5a93b5f5d29"></span>

<figure class="research-embed"><iframe src="/research-assets/0fffd9f318015371a4de7ad3435dbe90a7d81937e6a8534c84e75886942b1314.html" title="算法 中间块两次访问的参考前向" loading="lazy" sandbox=""></iframe><figcaption>算法 中间块两次访问的参考前向 · <a href="/research-assets/0fffd9f318015371a4de7ad3435dbe90a7d81937e6a8534c84e75886942b1314.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-6cb6f273-566c-49dc-945a-97519590c2a9"></span>

**实现对应** 按 SMELT §3.1、§6.2 与附录A 的计算图重写。无公开官方源码或已加载 checkpoint；算法不补造 router 与具体内核实现。

<span id="source-block-cb77d5fa-ffeb-4b32-b773-6488c6e3ecc9"></span>

## 早退时固定残差系数还是重新缩放

<span id="source-block-a75f97dd-e503-4910-8841-dda916b093a7"></span>

SMELT 的训练访问次数同时进入展开长度与残差缩放。接入 TaH2 式门控时，提前结束已有计算图，与根据新访问次数重设每次更新，是两个不同的操作。

<span id="source-block-5824d8b0-83f1-40e1-9075-cd89a7e09c3f"></span>

用单个线性残差分支说明：训练访问次数 r₀、更新矩阵 A，执行前 k 次后退出与改成 k 次并重设缩放，分别为：

<span id="source-block-4a54f744-ed6c-4ee1-8a2a-6a1788a6c03b"></span>

```math
\begin{aligned}
h_k^{\rm exit}&=(I+A/r_0)^k h_0,\\
h_k^{\rm rescale}&=(I+A/k)^k h_0.
\end{aligned}
```

<span id="source-block-6634d6c0-9354-4891-b8d5-e51b55e7f15d"></span>

取标量 A=0.4、h₀=1、r₀=2、k=1，前者为 1.2，后者为 1.4。差异在一次更新内已出现。真实 attention→MoE 的顺序残差及归一化使差异进一步依赖状态；这个线性例子仅用于区分两种干预。

<span id="source-block-7dd87205-b70c-4159-a70c-900f3d91d1e2"></span>

实际接入须固定训练时残差系数、退出时系数及可读出的合法状态。重新缩放应作为额外设计因素单独比较。原文重训 visits 消融与三预算配平继续保留原始含义；固定 checkpoint 的提前退出和深度外推需要另外建立证据。

<span id="source-block-3fca0ddb-b3e5-4bbb-a6e7-7c423a44c793"></span>

来源：[SMELT §3、§6.2](https://arxiv.org/html/2609.01343v3)；S_FORWARD 已显式给出残差更新顺序；[TaH2 工程](/notes/loop-reasoning/engineering/)提供门控接口。

<span id="source-block-fc55376d-2032-4bce-961d-4ed8e0d5aff7"></span>

## 实现缺口与后训练研究问题

<span id="source-block-b49ea5af-bd61-4541-81fb-2a56b6990c4e"></span>

已完整核对主要理论、全部设计消融、规模律、下游与机制分析以及附录配置／评测协议；表格中的预算误差和排名反例保留。没有把论文伪代码写成作者开源实现，也没有声称模型复现通过。

<span id="source-block-dc6507ed-f8d8-46b0-8307-4ce2164046b6"></span>

- 资源缺口：官方完整训练stack、可下载checkpoint、内部数据和逐格head／expert几何没有取得；完整架构不能发布是作者在附录A的明确说明。

<span id="source-block-cdaf0791-4bdf-4fc2-8531-cdc7bf1bed2b"></span>

- 规模范围：最优中间span和两次访问来自200M档消融，放大后的设计最优性没有独立验证；规模律支持窗外的结论必须标外推。

<span id="source-block-8ae3e6b3-8379-4386-aa37-9602c945eef5"></span>

- 系统性能：FLOPs、参数和KV近似匹配不约束硬件吞吐，需要补microbatch、通信与serial depth的实际计时。

<span id="source-block-0b10ac06-8ba9-4f6d-a02d-c9fe0af3a716"></span>

- 机制因果性：需要对sink、专家重路由与value变化分别干预，检验去掉哪种变化会损害收益；描述性相关不能替代这种干预。

<span id="source-block-4a28b73c-6202-4133-9271-d0cd7a9274b8"></span>

- 后训练资格：等到存在可取得、训练阶段明确的权重，再把它升级成可执行实验候选；现阶段可以先复用三预算控制与评测方法设计其它原生loop模型的实验。

<span id="source-block-4007c7e6-df02-4e30-a8f0-73fc2123698f"></span>

## 原文内容与实现位置

<span id="source-block-51343271-19a5-46fb-9e2b-9eb2b1ae4dfb"></span>

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

<span id="source-block-7df239ab-6267-4038-b3a6-130400b7a791"></span>

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 动机与三预算因果控制 | §1、§3.1–3.2 pp.1–6 | 仅论文配方；明确定义F/N/KV、δ，实际wall-clock不匹配 | 原文已核 |
| middle50%×2、残差1/r位置 | 图1；§3.1；§6.2式8–9；附录A | 数学和论文语义伪代码；明确attention先缩放再送MoE | 原文已核；无公开源码 |
| 规模标签与S=0语义 | §3.1式1；附录A表9–10 | 1.6B是Baseline档位；SMELT最大physical active1.103B；S=0仍top8/16 | 原文已核 |
| 三个设计消融 | §3.3–3.5表3–5 pp.7–8 | span/depth/count明确数值；95%下DCLM与validation排名相反；单训练seed | 原文和表格图像已核 |
| WSD与数据流/分支 | §3.1、§4.1 pp.5、8 | 196075 stable steps，六decay branches，96配对192端点，不当作192独立训练 | 原文已核；训练stack未公开 |
| 规模律与compute-optimal推导 | §4.2–4.3式2–5、表6–8 pp.9–12 | 两独立曲面、Huber logloss、L-BFGS-B、144 sparse endpoints、cell bootstrap、拟合范围和外推界 | 原文和表格图像已核 |
| 下游及超出loss的残差 | §5.1式6–7、图5–9 pp.12–14 | 96/96、83/96、29/30限定；Baseline sigmoid校准；不称因果证明 | 原文已核 |
| domain/length/ICL | §5.2–5.3图10–11 pp.15–17 | Code20.4%；长度1.52对0.98/0.88；Dyck29.8对26.4；非单调难度分位结果 | 原文已核 |
| router/residual/attention/sink机制 | §6图12–19 pp.17–23 | top8交集、norm ratios、Q/K/V cosine、sink0.60→0.02；保留描述性解释边界 | 原文已核 |
| 完整模型配置与预算误差 | 附录A表9–10 pp.30–31 | 稀疏格F3.9/N1.0/KV3.6%；S=0的F8.1%例外；按各配置、平均文档长度得到的每token算术FLOPs F（原文称measured；未披露计数/测量工具）进入拟合 | 原文和表格图像已核 |
| 评测协议 | 附录E式11、表11 pp.35–36 | Core centered score；Completion微平均gold CE；MMLU5shot；10seed；BOS preserving truncation | 原文已核 |
| GitHub和复现实用性 | 附录A implementation note p.30；§7 p.23 | 作者明确完整architecture/training stack不能发布；不制造repo/function/commit；无公开base资格 | 作者明确未公开 |
| 有序残差的有限步稳定性 | §3.1；§6.2式8–9 | 正确顺序的Jacobian乘积、非正规反例、共同度量充分条件；参考JVP/VJP诊断 | 代数及二维数值核对；未作模型探针 |
| RMSNorm方向机制与相对写入 | §6.2图13–14 | 保留epsilon的范数恒等式；100M早层相对写入例外与cell汇总分开 | 原文及100组代数核对；作者probe归约代码未公开 |
| 全概率overlap与attention误差 | §6.3式10、图16–17 | overlap=1−TV；中心化logit界、value/output条件界；cosine=1反例 | 推导及500组小型数值核对；不声称KV近似复用获证 |
| 图9超额收益的尺度结构与聚合 | §5.1式6–7、图8–9；附录E表11 | 全部原值；任务centered residual→类别平均与聚合Core另拟合的区别；每尺度24端点 | 图9 PDF视觉对照及聚合恒等式核验；不伪造逐端点数据 |
| ICL配对单位与14项展示筛选 | §5.3图11；附录C图24 | 16任务扫描、最大shot正增益且p<0.05筛出14；10评测seed、cell内后cell外平均；双侧枚举仅作参考定义 | 分层均值/1024符号置换小算例通过；原始评分与检验单双侧细节未公开 |
| 全16格训练曲线 | 附录B，Figs.20–23，pp.31–33 | 训练loss、同token横轴、stable／分叉和实际F分别登记；共同计算差分及条件界 | 原文文字／编号已核；不把16格末端扩成96分叉全胜 |
| 全四尺度sink与质量干预 | 附录D，Fig.25，pp.34–35；§6.4 | segment起点、物理／执行层对齐；mass／value／条件均值分解与固定KV算法 | 500组向量恒等式核对；无作者代码或模型因果实测 |
