[基座矩阵与七篇论文](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [预训练](https://chatgpt.com/space/page_4f0508753a6081919a82b0c89213fd6c) · [后训练](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

# Ouro：完整共享栈、熵正则退出与论文到代码的边界

**Scaling Latent Reasoning via Looped Language Models，2510.25741v6，54页**研究整栈共享重复的原生循环 Dense 基座。建议按“结构→两阶段训练目标→性能与机制结果→HF实现边界”阅读。Ouro-1.4B与2.6B都在循环结构下预训练；2.6B从1.4B的3T-token原生循环checkpoint复制层、扩到48层后继续训练。因此，这次upcycle仍发生在原生循环模型之间，不是把普通非循环基座经后训练改造成loop。

**原文**：[上传英文PDF](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal)。**作者项目页**：[Ouro](https://ouro-llm.github.io/)。**官方发布代码**：[1.4B revision 574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)、[2.6B revision 1ed0425](https://huggingface.co/ByteDance/Ouro-2.6B/tree/1ed04250da1a9936042725d302e81c8fa2ab5abd)。下文对照论文与以上固定实现；权重载入和训练复现尚未验收。

## 整栈循环与归一化顺序

1.4B是24层、2.6B是48层，宽度均2048，16个MHA头、head_dim=128，SwiGLU中间维5632，词表49152，输入embedding与LM head不共享。发布配置R=4，max_position_embeddings=65536、RoPE theta=10⁶、RMSNorm epsilon=10⁻⁶。所有“full_attention”层在这里仍是**因果完整上下文注意力**；不是后文理论定理使用的无mask注意力。[原文§4.1/Table 2，p.9；两份config.json]

状态从token embedding开始，没有Huginn式的随机初态和每轮固定输入adapter。实际OuroModel.forward每次过完整共享层栈后执行一个输出RMSNorm，所得状态既送gate/LM head，也作为下一轮输入：

```math
\begin{aligned}
h^{(0)}&=E(x),\\
h^{(r)}&=N_{\rm out}\!\left(B_L\circ\cdots\circ B_1(h^{(r-1)})\right),\\
z_i^{(r)}&=W_{\rm LM}h_i^{(r)},\\
\lambda_i^{(r)}&=\sigma(w_g^\top h_i^{(r)}+b_g).
\end{aligned}
```

每个OuroDecoderLayer.forward把子层**分支输出先归一化再相加**：

```math
\begin{aligned}
u&=z+N_2\!\left(\operatorname{Attn}(N_1(z))\right),\\
B(z)&=u+N_4\!\left(\operatorname{MLP}(N_3(u))\right).
\end{aligned}
```

式中的相加顺序与Huginn的N₂(z+Attn(N₁(z)))不同，因而梯度恒等通路和幅度控制位置也不同。阅读源码时，可用四个名称定位子层归一化：input_layernorm、input_layernorm_2、post_attention_layernorm、post_attention_layernorm_2，再单独查loop末norm。论文§4.1的描述较粗，应以这些调用位置展开计算图。发布config中的initializer_range=.02描述HF初始化配置，尚不能用来确定大规模原始预训练的全部初始化逻辑。[modeling_ouro.py::OuroDecoderLayer / OuroModel]

### 推导 残差恒等通路与轮末归一化分开分析

以下雅可比均对完整因果前缀求导，位置与 mask 固定；各因子在其真实中间激活处求值。Ouro 对分支输出做 norm，因此单 block 和整轮算子的导数分别是

```math
\begin{aligned}
J_B&=(I+J_{N_4}J_MJ_{N_3})
(I+J_{N_2}J_AJ_{N_1}),\\
F_\theta&=N_{\rm out}\circ B_L\circ\cdots\circ B_1,\\
J_F&=J_{N_{\rm out}}J_{B_L}\cdots J_{B_1},\\
\frac{\partial h^{(R)}}{\partial h^{(0)}}
&=J_{F,R}\cdots J_{F,1}.
\end{aligned}
```

单 block 的残差通路有显式 I，但整轮末尾的 RMSNorm 仍作用在所有通路上；“存在 residual”不能推出多轮导数不会消失或放大。Huginn 对残差和做 norm，导数中相应的 I 也被外侧 norm 左乘，二者的梯度结构确实不同。[固定版本 [OuroDecoderLayer/OuroModel](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)；这是源码计算图的链式法则。]

还有一个有助于理解设计动机的条件结论。对相同长度、位置与 mask，Ouro 的整轮算子不再读取原始 embedding；如果它对包含所有输入轨迹的共同不变域具有全局一致收缩常数 ρ<1，那么

```math
\|h^{(R)}(x)-h^{(R)}(x')\|
\le\rho^R\|E(x)-E(x')\|\longrightarrow0.
```

此时唯一极限会消除不同输入的差别。Ouro 使用有限轮数，并不需要满足这个全局假设；多吸引域、保留信息的方向或非固定点动力学也不受该结论排除。这个推导说明为何不能把“所有原生 loop 都应收敛到全局唯一固定点”作为统一设计准则。2.6B 的层复制扩容保留原生循环训练身份，却也不能仅凭复制权重宣称扩容前后函数完全相同。[原文 §3.1、§4.3.1；有限轮模型与假设推论须分别理解。]

## 退出概率：hazard、survival、PDF、CDF不能混称

Ouro要把“这个位置还值得算几轮”变成可学习的决策。沿一个预测位置看，第r轮gate接收本轮状态，输出λᵢ⁽ʳ⁾：已经到达这一轮后，现在退出的条件概率。它还不是最终停在第r轮的概率，因为到达这里以前，模型必须在每个较早轮次都选择继续。

下面用三种量描述同一个退出过程：hazard λ表示到达本轮后的条件退出概率，survival S表示尚未退出的概率，PDF p表示最终停在某轮的概率。CDF F累加截至本轮已经退出的质量，Q-exit再用分位数q选出读出轮次：

```math
\begin{aligned}
S_i^{(0)}&=1,&
S_i^{(r)}&=\prod_{j=1}^{r}(1-\lambda_i^{(j)}),\\
p_i(r)&=\lambda_i^{(r)}S_i^{(r-1)}\quad(r<T),&
p_i(T)&=S_i^{(T-1)},\\
F_i(r)&=\sum_{j\le r}p_i(j),&
R_i&=\min\{r:F_i(r)\ge q\}.
\end{aligned}
```

把各轮串起来，前T−1轮依次分走退出质量，末轮直接接收全部剩余质量，所以p的总和恰为1，最后一个gate logit也就不进入末轮PDF。Q-exit读取的是这份累计质量：q较大时，要等更多概率累积后才退出，通常会选到更深轮次。

这里并没有随机抽一个退出轮次。同一输入和固定q给出确定性分位数选择；从PDF采样则是另一种策略。做RL时，两者对应的行为策略与log-prob也要分别定义。[原文§3.2，pp.5–6；OuroForCausalLM.forward的remaining_prob分支]

## Stage I：同时学习LM和退出分布

有了退出分布，训练便需要使它偏向预测更好的轮次，同时让各个深度都得到学习机会。对同一个真实next token，先计算每轮的交叉熵ellᵢ⁽ʳ⁾；若把退出轮次看作按pᵢ抽样，Σᵣpᵢ(r)ellᵢ⁽ʳ⁾就是这次预测的期望损失。Stage I把它与负熵项相加，再对位置和样本聚合。负熵项鼓励分布保留一定宽度，避免训练过早只使用少数轮次：

```math
\begin{aligned}
\ell_i^{(r)}&=-\log\operatorname{softmax}(z_i^{(r)})_{x_{i+1}},\\
\mathcal L_{\rm I}
&=\mathbb E_x\sum_i
\left[\sum_{r=1}^{T}p_i(r)\ell_i^{(r)}
-\beta H(p_i)\right],\\
H(p_i)&=-\sum_r p_i(r)\log p_i(r).
\end{aligned}
```

这项目标让各轮的预测质量参与深度分配，运算顺序因而很关键：每轮先对真实标签算CE，再按p加权。先混合预测概率再取负log，会得到混合模型的NLL；先平均logits再做CE，又是另一项目标。三者通常不同。gate与这些loss都由同一个prefix产生，数据期望也保留它们的这种关联，不能拆成两个期望相乘。

熵项还给出了变分解释。均匀先验满足KL(p∥Unif)=−H(p)+logT，所以两种写法只差参数无关常数；β=1对应通常的负ELBO，其他β则改变正则权重。[原文Eq.(4)与§3.3，pp.6–7，已核页面图像]

均匀先验的作用是防止深度分布过早塌缩，并不显式惩罚计算量。Appendix A在776M、T=4、FineWeb-Edu 20B、固定其他训练条件下比较几何先验，报告均匀先验更好的收敛与深度利用。原文同时说50K tokens/batch与“约40K steps”；20B/50K实际为400K，这是文本内部账目不一致，不能无注释地复制成可复现配置。该附录不是1.4B/2.6B主运行的同规模先验消融。[Appendix A，pp.34–35]

### 推导 从 hazard 到目标梯度与计算偏好

退出分布也能给出计算预算的含义。对于按p随机采样的策略，执行第r轮的前提是此前尚未退出，概率为Sᵣ₋₁；把每轮是否执行的指示量相加，就得到期望轮数。令aⱼ为gate logit、λⱼ=σ(aⱼ)，末轮吸收尾质量后有：

```math
\begin{aligned}
\sum_{r=1}^{T}p_r&=1,\\
\mathbb E_p[R]&=\sum_{r=1}^{T}S_{r-1},\\
\frac{\partial\mathbb E_p[R]}{\partial a_j}
&=-\lambda_j\sum_{r=j+1}^{T}S_{r-1}
\quad(j<T).
\end{aligned}
```

提高某轮退出 logit 会降低该随机策略的期望深度。固定 q 的 Q-exit 则是确定性分位数，不是上述期望；只有 q 独立均匀采样时，逆 CDF 选择才具有分布 p。若每轮成本 c_r 不同，应改为 E[C]=Σ_r c_r S_(r−1)，而非只用平均轮数。[原文 Eq.(3)；`remaining_prob/cumsum/gather`，固定版本 [HF 代码](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)。]

Stage I究竟偏好哪种深度分配，可以先把同一位置的各轮损失ellᵣ视为已给定，暂时只优化p。β>0时，下面是概率单纯形上的严格凸问题。其解把较小loss变成较大的退出权重，温度由β控制：

```math
\begin{aligned}
\min_{p\in\Delta_T}
\left\{\sum_rp_r\ell_r+\beta\sum_rp_r\log p_r\right\}
&=-\beta\log\sum_re^{-\ell_r/\beta},\\
p_r^\ast&=\frac{e^{-\ell_r/\beta}}{\sum_se^{-\ell_s/\beta}}.
\end{aligned}
```

拉格朗日条件ellᵣ+β(log pᵣ+1)+ν=0给出这一softmax形式。β趋近0时，权重集中到损失最小的轮次；β增大时，分配趋于平坦。特别地，给定各轮损失相同且β>0，最优分布就是均匀分布，平均深度为(T+1)/2。因此均匀先验主要防止深度利用过早塌缩，本身没有“尽早退出”的成本偏好。换成先验π并使用βKL(p∥π)，解则与πᵣexp(−ellᵣ/β)成正比。

这个解帮助解释训练压力，但不是实际gate能够逐题访问的答案：ellᵣ用到了真实目标token，gate只能从prefix和当前状态预测退出概率，而且在不同位置间共享参数。[原文 §3.3、Appendix A；优化解为我们的推导。]

β=1 时，令 q_r(y) 为第 r 轮给真实 token 的概率，m(y)=(1/T)Σ_r q_r(y)，后验 ω_r=q_r(y)/Σ_s q_s(y)，则负 ELBO 的差距可以精确写出：

```math
\mathcal L_{\rm I}+\log T
=-\log m(y)+D_{\rm KL}(p\|\omega).
```

这解释了“与均匀深度先验的变分目标只差常数”，也说明它不等于用 learned p 混合预测概率后的 NLL。

把这层关系落实到反向传播，一个gate的变化会同时影响本轮退出质量和所有后续轮次的质量。因而它面对的是“现在停”与“继续后按条件分布停在以后”的代价比较。暂时固定各轮预测器，记cᵣ=ellᵣ+β(log pᵣ+1)，后续条件概率为p(r|R>j)=pᵣ/Sⱼ，便有：

```math
\frac{\partial\mathcal L_{\rm I}}{\partial a_j}
=p_j(1-\lambda_j)
\left[c_j-\sum_{r>j}\frac{p_r}{S_j}c_r\right]
\quad(j<T).
```

方括号中，第一项是现在退出的代价，第二项是继续以后各轮的条件平均代价；正负号决定梯度会压低还是抬高本轮退出logit。多个gate因此通过剩余概率相互联系，而非独立的二分类器。联合训练LM时，还要计入Σᵣpᵣ∇ellᵣ，并沿gate输入的hidden state反传；冻结p或detach全部gate输入会删去其中的依赖。末轮logit未进入PDF，也没有这条Stage I gate梯度。

计算这些概率时，可在log域积累log Sⱼ=Σₖ≤ⱼlogσ(−aₖ)，再令log pⱼ=log Sⱼ₋₁+logσ(aⱼ)，末轮直接取log Sₜ₋₁，从而避免先连乘再取log的下溢。[原文 Eq.(4)；log-domain 计算为等价实现建议，非声称已获得原始训练源码。]

## Stage II：从相邻轮次的损失改善训练gate

Stage I训练出多深度预测后，gate第二阶段转而学习一个更直接的继续信号：刚刚多算一轮，真实标签的预测损失改善了多少。此时冻结LM，把相邻两轮的CE差detach，负改善截为0，再通过sigmoid生成继续计算的软标签w。gate用本轮状态去拟合这个标签：

```math
\begin{aligned}
I_i^{(r)}&=\max\{0,\operatorname{sg}(\ell_i^{(r-1)}-\ell_i^{(r)})\},\\
w_i^{(r)}&=\sigma\!\left(50(I_i^{(r)}-0.005)\right),\\
\mathcal L_{\rm II}
&=\frac1T\sum_{r=2}^{T}\frac1M\sum_i
\operatorname{BCE}\!\left(1-\lambda_i^{(r)},w_i^{(r)}\right).
\end{aligned}
```

w表示希望继续的程度，1−λ则是gate预测的继续概率。沿“这一轮没有改善”的情况代入：I=0，w=σ(−0.25)≈0.438，仍保留相当的继续权重；所以原文“无改善时接近0”的描述与公式并不精确一致。

监督来自已经完成的r−1→r计算，训练的却是到达r之后是否继续。这让gate能从有答案的训练样本学习一种贪心代理，生成时不再读取真实答案loss；它并没有直接估计下一轮r+1的收益。原文求和从r=2开始而分母仍为T，下面也保留这个尺度。[原文Eq.(5)–(6)，p.7]

这一阶段与后面的Reasoning SFT不是同一个“Stage II”。课程应分别叫“gate第二阶段”和“训练流水线的CT/SFT阶段”，避免将退出策略校准、继续预训练和任务对齐混成一个阶段。

### 软标签学习与最优停止的区别

这份软监督怎样移动gate，可以从单个位置看清。r≥2时，w已经detach，训练只调整退出logit a。把BCE展开后，它等价于用退出标签1−w监督σ(a)；导数为预测退出概率与这一标签的差，所以自由拟合该标签时的最优值是λ*=1−w：

```math
\begin{aligned}
\ell_{\rm gate}(a)
&=-w\log(1-\sigma(a))-(1-w)\log\sigma(a),\\
\frac{\partial\ell_{\rm gate}}{\partial a}
&=\sigma(a)+w-1,\\
\lambda^\ast&=1-w.
\end{aligned}
```

例如无改善标签w≈0.4378235，对应的理想退出概率约0.5621765。由于I≥0，逐位置自由最优解不会超过这一退出概率；共享gate还要同时拟合许多位置，所以这不是对每次实际输出施加的硬上限。

分母选择影响的是更新尺度：T−1个标签项除以T，相比除以T−1会把所有梯度乘上(T−1)/T。独立、无正则的优化最优点不变，但有限步训练以及与其他loss合用时的相对权重会改变。[原文 Eq.(5)–(6)。]

同一gate参数在每轮重复使用，因此“哪一轮有标签”与“哪一轮输出会改变”是两件事。原文也给r=T计算BCE：即使推理PDF在末轮直接吸收尾质量、不使用λₜ，这份BCE仍能更新共享gate。r=1虽没有独立标签项，它的输出也会因同一组参数更新而改变。实现冻结和求导时，需要保留这种跨轮共享。

要让“继续”成为成本敏感的最优停止决策，还需要知道未来可能带来什么。到达第r轮后，停止会承担当前预测的条件风险，继续则要支付下一轮成本，并面对之后各轮仍可选择停止的未来风险。若包含prefix、latent、cache和已执行轮数的完整状态Sᵣ满足所需Markov条件，记当前风险为ℓ̄ᵣ(Sᵣ)、下一轮增量成本为cᵣ₊₁，这个比较写成Bellman形式：

```math
\begin{aligned}
V_T(S_T)&=\bar\ell_T(S_T),\\
V_r(S_r)&=\min\left\{
\bar\ell_r(S_r),\ 
 c_{r+1}+\mathbb E[V_{r+1}(S_{r+1})\mid S_r]
\right\}.
\end{aligned}
```

Bellman式比较的是未来风险与成本，原文Stage II使用的则是已经发生的相邻改善。这使后者成为一个容易构造训练标签的贪心代理，同时说明它与最优停止解之间还缺少未来价值估计和每轮实际成本。研究budget-aware后训练时，可以从这个代理出发，考察其输出与未来增益、执行成本是否校准。[原文 §3.4；Bellman 式为对照性的理论定义。]

#### 算法 Ouro 论文目标的两阶段语义

**输入** M>0 个有效监督位置；共享 LM 参数 θ、跨轮共享 gate 参数 φ、最大轮数 T≥2、Stage I 系数 β。

**输出** Stage I 的联合梯度，或 Stage II 仅 gate 的梯度；不包含 optimizer.step。算法描述论文目标，不代替未公开训练调度。

**执行前提** 阶段选择由外部训练计划给定；目标标签只用于训练；末轮退出质量吸收全部剩余概率。

![算法 Ouro 论文目标的两阶段语义](visualize:fde1_bGliZmlsZV9YTV8yUVdwT1VvTE5NNU83WVFRMVFn_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 依据原文 Eq.(3)–(6)，[Ouro 论文](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal) §3.2–3.4。log-domain hazard 是数学等价的数值实现建议。真实原始训练的阶段时序、并行细节与 optimizer 配置仍需对应训练记录；此算法不冒充 HF labels 分支。

## 7.7T训练流水线与稳定性处理

| 阶段 | tokens | context | 循环/正则 | 学习率与用途 |
|---|---:|---:|---|---|
| 1a Stable I | 3T | 4K | R=8，β=.1 | peak 3×10⁻⁴，batch 4M→8M tokens |
| 1b Stable II | 3T | 4K | R=4，β=.05 | 3×10⁻⁴；保留24层1.4B或复制为48层2.6B |
| CT Annealing | 1.4T | 16K | R=4，β=.05 | cosine降到3×10⁻⁵，提升数学/代码与高质量数据 |
| LongCT | 20B | 64K | R=4，β=.05 | 3×10⁻⁵；ProLong-64K |
| Mid-training | 300B | 32K | R=4，β=.05 | cosine降到10⁻⁵，ChatML格式监督材料与旧数据回放 |

总计7.72T，论文约写7.7T。主训练AdamW betas=(.9,.95)，weight_decay=.1，clip=1.0。RoPE base依次10K、10K、40K、1M、1M。早期R=8出现loss spike与梯度振荡后改R=4，同时增batch、降β；这些是运行中的联合稳定性决策，不是“4轮普遍最优”的定理。Ouro论文没有描述Huginn式随机深度+末8轮TBPTT，不能从loop共性自动补入。[原文Table 1与§4.3，pp.8、10–11，表格已图像核对]

语料细节也影响基座选择：初期含Nemotron-CC、MAP-CC、Ultra-FineWeb-zh、OpenCoder、MegaMath；由于SmolLM2词表对中文碎片化，从Stage 2起移除中文。Mid-training由182B处理后监督语料采90B，加30B Stage 1与180B Stage 2回放组成300B，包含问答和CoT样本。因此“Base”不意味着从未见过指令/CoT。训练框架为flame，建立在torchtitan上；公开HF模型文件不是整个数据、优化、分布式训练链路。[§4.2，Tables 3–5]

### 原始数据配方：语料规模、来源预算与回放分别保留

原文 Table3 同时列语料池规模与使用量，Table4／5 列阶段配比；两种表不能混为总训练 token 数。下面单位均为 B tokens，保持原表用词和显示精度。Table3 没有列出 CT 阶段的每一来源，不能只把其 Used 列相加来重建7.72T训练总量。[原文 §4.2、Tables3–5，pp.9–10](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal)

| Table3 来源 | 阶段 | 语料池规模 | 原表 Used |
| --- | --- | ---: | ---: |
| Nemotron-CC | 1 | 6386 | 4404 |
| MAP-CC | 1 | 800 | 780 |
| Ultra-FineWeb-zh | 1 | 120 | 120 |
| OpenCoder-pretrain | 1 | 450 | 450 |
| MegaMath-web | 1 | 247 | 246 |
| MegaMath-high-quality | 2 | 64 | 64 |
| Nemotron-CC-Math-v1 | 2 | 210 | 210 |
| Nemotron-Code | 2 | 53 | 53 |
| Nemotron-SFT-Code | 2 | 48 | 48 |
| Nemotron-SFT-General | 2 | 87 | 87 |
| OpenCoder-Annealing | 2 | 7 | 7 |
| ProLong-64K | 3 | 20 | 20 |
| Mid-training SFT Mix | 4 | 182 | 90 |

| 阶段 | 来源 | 原表 token 比例 |
| --- | --- | ---: |
| 1：合计6T | Nemotron-CC | 73.4% |
| 1 | MAP-CC | 13.0% |
| 1 | Ultra-FineWeb-zh | 2.0% |
| 1 | OpenCoder-pretrain | 7.5% |
| 1 | MegaMath-web | 4.1% |
| 2：合计1.4T | Nemotron-CC-high-quality | 66.5% |
| 2 | Nemotron-CC-Math-v1 | 15.0% |
| 2 | MegaMath-high-quality | 4.6% |
| 2 | OpenCoder annealing | 0.5% |
| 2 | Nemotron Synthetic-Code | 3.8% |
| 2 | Nemotron-SFT-Code | 3.4% |
| 2 | Nemotron-SFT-General | 6.2% |

两阶段比例各合计100%；例如Stage1的6T×73.4%=4.404T，与Table3显示值一致。Stage2的1.4T×4.6%=64.4B，而Table3写64B，保留两表舍入口径，不把差值解释为新的过滤量。语料池容量、带重复的消费量和唯一数据覆盖也不同；随机采样的说明不足以反推出具体有放回／无放回算法。

Stage4的300B由90B新监督语料、30B Stage1回放、180B Stage2回放组成，即来源预算30%／10%／60%；这是预算来源比例，不能据此认为32K ChatML训练与早先阶段具有相同的前缀分布。最终Base已接触问答与CoT，Thinking又单独使用约8.3M条SFT样本；样本数比例也不直接等于监督token比例。公开HF forward没有包含这套历史数据采样器、formatter、过滤日志和源游标恢复实现，这些仍是完整训练复现的缺口。

## Thinking与RL：原文已经报告过失败尝试

Reasoning SFT约8.3M样本：数学3.5M、代码3.2M、科学808K、对话767K；训练2 epochs、32K序列、LlamaFactory、Adam lr=2×10⁻⁵与cosine，产出独立的Ouro-Thinking checkpoint。[原文§4.4/Table 6，pp.11–12]

§4.5值得保留在后训练课程：作者在DAPO-17K上尝试DAPO/GRPO，没有超过最终SFT checkpoint。第一条路径使用固定四轮rollout再模拟早退、按较早轮loss更新，导致行为分布与更新分布失配；第二条路径把rollout和更新都固定四轮，训练可进行但无显著收益。论文提出小模型经大量SFT后剩余收益有限等解释，属于假设，不能转述成已证明原因。它直接给我们的工程问题：生成的token究竟来自哪个深度，更新时复算哪一个分布，KV和退出历史是否一致，必须写入训练样本。原文 p.12；连接[后训练课程](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)。

### 推导 训练阶段改变了什么 分布失配发生在哪里

从 T=8、β=0.1 切换到 T=4、β=0.05 时，计算图深度、退出概率空间和目标标度都改变。假设所有轮次具有相同预测 CE=ell，且 p 均匀，则

```math
\mathcal L_{\rm I}=\ell-\beta\log T.
```

在上面的假设中，每轮预测CE都未改变，总loss仍会因−βlogT而变化。因此比较训练阶段时，要同时看每轮CE、退出熵和平均深度，不能只据total loss判断LM能力是否突然改善。论文同时调整了batch等条件，所以loss spike的缓解反映的是这套联合训练设置。

层复制带来另一种阶段变化：从原生1.4B复制层来初始化2.6B，只规定参数从哪里开始，新的深度算子仍须继续优化。[原文 §4.3、Table 1。]

对 RL 中的深度失配，令 prefix 为 x_<t，实际固定四轮 rollout 的行为概率为 μ_t=p_(old,4)(x_t|x_<t)。若更新使用较早轮 r_t 的预测器，则单 token 的正确分母仍来自实际行为策略：

```math
\begin{aligned}
\rho_t&=\frac{p_{\theta,r_t}(x_t\mid x_{<t})}
{p_{{\rm old},4}(x_t\mid x_{<t})},\\
\rho_{1:N}&=\prod_{t=1}^{N}\rho_t.
\end{aligned}
```

若希望用随机 gate 联合采样深度和 token，行为记录还必须包括真实采样的深度，联合概率变为 gate 概率乘对应深度 token 概率；若用确定性 Q-exit，深度是分位数规则的结果，不能凭空加上“采样 gate 的 log-prob”。后处理四轮 rollout 的 logits 不会追溯改变当时生成 token 的行为概率。仅把更新 loss 换到较早轮，不是完成了 on-policy 对接；即使在支持条件下可做重要性重加权，方差与裁剪偏差仍是另一问题。[原文 §4.5 的第一条失败路径；上述概率式解释其失配。公式中的 prefix 条件还隐含完整执行协议和按各自参数重算的 latent/cache，不能把行为缓存直接代入当前策略。]

第二条固定四轮 rollout/更新消除了这项深度不一致，却仍未超过 SFT；因此“修复 on-policy 即必然带来收益”也不受实验支持。比较 Base、Thinking 与 RL checkpoint 时，需固定可见生成长度、sampling、深度、gate 与 cache 策略，才能把显式 CoT 改善与内部循环策略变化分别归因。[原文 §4.4–4.5、Tables 6–13。]

## 发布实现的逐步执行链与三项差异

1. OuroConfig加载24/48层、R=4、early_exit_threshold=1.0等配置；OuroModel建立一个ModuleList，所有轮次重复调用同一组对象。 构造时还把config.total_ut_steps复制到实例self.total_ut_steps；构造后仅修改config不自动修改实际循环次数。

2. OuroModel.forward计算embedding、cache_position、因果mask、RoPE；依次运行所有current_ut和所有decoder_layer。

3. 每轮末norm，保存hidden_states_list和early_exit_gate输出。OuroAttention.forward用current_ut×num_hidden_layers+layer_idx作为cache键。

4. UniversalTransformerCache.update把当前token的K/V沿序列轴追加到该展开层槽位；max_cache_size=L×R。prefill一次写入多个token，decode一次追加少量token；参数共享没有合并KV槽。 use_cache=False只跳过新建／返回cache的路径；在普通未启用checkpoint包装器的前向中，显式传入的非空cache仍传给attention，后者仍调用update。因此真正不使用历史cache的参考前向应同时设置use_cache=False和past_key_values=None；没有返回cache不代表传入对象未改变。

5. **所有轮次已经算完后**，OuroForCausalLM.forward才构造PDF/CDF，按exit_threshold选择某轮hidden，或按exit_at_step直接选择；exit_at_step使用零基索引，0代表第一轮。

6. labels=None时输出选定轮次、加权logits或末轮logits；labels存在时走compute_expected_logits，再由loss_function计算CE。 发布配置所记 Transformers 4.55.0 的默认 loss 会在内部 shift labels；不能沿用 Huginn 的预 shift 契约。

论文中的退出分布表达了在哪一轮读出的选择；要节省计算，执行器还必须在选中轮次后停止调用共享栈。上面的HF路径先完成所有轮次，再在已有状态里选输出。因此减小q会改变读出，却不会减少该路径已经执行的Transformer层。

带labels时，选择还变成另一种操作：各轮先经LM head得到logits，按退出概率相加，最后只对这个合成结果计算一次CE：

```math
\begin{aligned}
\bar z_i&=\sum_rp_i(r)z_i^{(r)},\\
\mathcal L_{\rm HF}&=\operatorname{CE}(\bar z,y).
\end{aligned}
```

这时被优化的是合成logits给真实token的概率。论文Stage I则先评价每一轮的预测，再优化这些loss的退出加权和及−βH；gate第二阶段还需要冻结LM和改善标签。这些操作均未由当前labels分支完成。两种训练可以使用同一组逐轮状态，却把不同的信用分配信号传回gate与LM。

缓存也沿完整前向保留了全部展开层槽位；当前公开实现没有Table 14中last-step／averaged decode共享策略的专门分支。因此使用这一forward做后训练时，实际采用的是这里的目标和缓存行为。[固定HF revision下 modeling_ouro.py::OuroModel.forward / OuroForCausalLM.forward / OuroAttention.forward / UniversalTransformerCache.update]

作者项目页的Code链接当前指向[rkstgr/LoopLM](https://github.com/rkstgr/LoopLM)，其README明确写v4论文的reimplementation。应把它作为“项目页链接的训练复现”，不能冒充原始7.7T训练仓库，也不能用它填补官方HF与论文的全部差异。官方项目页同时指出vLLM固定四轮、不支持自适应提前退出，并提示原生支持版本边界；部署前需按其固定版本检查，不用模型平台自动生成的通用安装片段推断支持情况。

### 深度变更同时涉及配置、实例和缓存

固定HF的实际循环读取self.total_ut_steps，而新缓存容量读取config.total_ut_steps；已有UniversalTransformerCache又不会因这两个值变化而自动扩容。令L为物理层数，r、ℓ均从0索引：

```math
 i(r,\ell)=rL+\ell,\qquad
0\le r<R_{\rm loop},\quad0\le\ell<L,\qquad C_{\rm slots}\ge LR_{\rm loop}.
```

只把config从4改8，实例仍可只执行4轮；只把实例改8、config仍4，新缓存仍为4L槽，若计算到第5轮就访问槽4L。两项都修改后，旧缓存的容量和较深轮历史仍未自动补齐。复核深度实验应同步记录请求深度、config值、实例值、实际返回的各轮状态数量、缓存槽数及各槽历史长度；改变深度后采用新cache并重算共同前缀，再与完整前缀参考对齐。

L=2的隔离原函数例子确认：config=8而实例未改时执行4轮；反向只改实例时，第5轮访问index=8、capacity=8，触发越界。原Attention.forward的替身检查另确认显式cache在use_cache=False时仍收到一次update。它们限定固定代码的对象与控制流，独立于前文Transformers4.55.0的基类兼容问题，未认证完整模型环境。[固定构造、forward与cache](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)

### 标签对齐 损失归约与重计算

Ouro HF `forward` 将loss计算交给Transformers，因此标签和归约语义要继续追到依赖实现。两个发布配置记录 `transformers_version=4.55.0`；以下契约对应该版本且未覆盖 `loss_function` 的路径，实际复现仍应固定安装版本。`PreTrainedModel` 为 `OuroForCausalLM` 选中 `ForCausalLMLoss`：先把 logits 升为 FP32，再处理 target 位移、flatten、忽略标记与归约。损失阶段的升精度不会追溯重算此前的 gate、PDF 或加权 logits。来源：[Ouro forward](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)、[配置](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json)；Transformers v4.55.0 [loss_function 选择](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/modeling_utils.py)与 [loss_utils.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/loss/loss_utils.py)。

令原始 labels 为 B×N 的 y，默认忽略值为 −100；未显式提供 `shift_labels` 时，源码在右端补一个 −100 再取 `[...,1:]`。因此训练 target 是

```math
\widetilde y_{b,i}=\begin{cases}
y_{b,i+1},&1\le i<N,\\
-100,&i=N.
\end{cases}
```

这与 Huginn HF 要求 labels 预先 next-token 对齐的契约不同。输入 token 为 [a,b,c,eos] 时，Ouro 传同位置 labels 得到 target [b,c,eos,−100]；若直接复用 Huginn 的预移位 labels，则 Ouro 再移一次，变成 [c,eos,−100,−100]。prompt/response loss mask 也随这次位移作用于预测位置，不能只检查 labels 与 input_ids 形状相同。

标签移位完成后，还要核对保留的预测位置与标签是否对应。`logits_to_keep` 经Ouro的 `_select_token_positions` 选择hidden/PDF位置，得到B×K×V logits，却不会同时切片原始labels。若仍传B×N labels且K≠N，默认loss在flatten后就会遇到预测数与target数不一致的问题。

全序列训练使用 `logits_to_keep=0`。若只计算选定位置，先构造全序列next-token target，再按同一组位置取得B×K的 `shift_labels`，或在外部计算已经对齐的loss。顺序在这里很重要：“先切labels再shift”和“先shift再切”不是默认等价的操作，任意非连续位点尤其如此。[固定 `OuroForCausalLM.forward` 与 Transformers `ForCausalLMLoss`。]

设 I 是位移后未被忽略的位置集合，z_i 是该位置实际参与 loss 的加权 logits，D>0 是归约分母。源码语义及其梯度为

```math
\begin{aligned}
L&=\frac1D\sum_{i\in I}-\log\operatorname{softmax}(z_i)_{\widetilde y_i},\\
D&=\begin{cases}|I|,&\texttt{num\_items\_in\_batch}=\texttt{None},\\
D_{\rm supplied},&\text{调用方显式提供分母},\end{cases}\\
\frac{\partial L}{\partial z_{i,v}}
&=\frac{\mathbf1\{i\in I\}}{D}
\left(\operatorname{softmax}(z_i)_v-\mathbf1\{v=\widetilde y_i\}\right).
\end{aligned}
```

`fixed_cross_entropy` 在未给分母时使用 mean，给定分母时使用 sum 后除以它。因而不同有效 token 数的 microbatch 若各自取 mean 再等权平均，优化的是 microbatch 等权目标；全有效 token 目标则按各 microbatch 的有效数加权。对于 M 个 microbatch、有效数 N_m>0 和均值梯度 g_m，两者分别为 M⁻¹Σ_mg_m 与 Σ_mN_mg_m/Σ_mN_m。传入 `num_items_in_batch` 后，调用方还必须与梯度累积和分布式归约的缩放一致，避免再次除以一个已计入的窗口大小。这是后训练目标的归约定义，不会自动把当前加权-logits CE 改成论文 Stage I。[Transformers v4.55.0 `fixed_cross_entropy`。]

原文 Eq.(4) 还明确要求：token averaging 和 loss mask 同时作用于任务项与熵项。重建论文 Stage I 时应使用同一个有效位置集合 I 与归约分母 D，而不是只给 CE 做 response mask。

```math
\begin{aligned}
L_{\rm I,mask}
&=\frac1D\sum_{i\in I}
\left[\sum_rp_i(r)\ell_i^{(r)}-\beta H(p_i)\right],\\
\frac{\sum_{i\in I}\sum_rp_i(r)\ell_i^{(r)}}{D_{\rm CE}}
-\frac{\beta\sum_{i\in I}H(p_i)}{D_H}
&=\frac1{D_{\rm CE}}\sum_{i\in I}
\left[\sum_rp_i(r)\ell_i^{(r)}
-\beta\frac{D_{\rm CE}}{D_H}H(p_i)\right].
\end{aligned}
```

第二式要求两项使用相同 I、正分母，说明分母不一致会把实际正则系数改为 βD_CE/D_H；若熵项还包含被 CE 忽略的 prompt 或 padding 位置，则连正则作用范围也变了，不能仅调整一个 β 修复。依据：[Ouro 原文 §3.3、Eq.(4)，p.6](https://arxiv.org/abs/2510.25741v6)。这项要求属于论文训练目标；当前 HF labels 分支本身没有实现熵项。

重计算由继承层的调用逻辑实现：`OuroDecoderLayer` 继承 `GradientCheckpointingLayer`，其 v4.55.0 `__call__` 在 training 且开启 checkpoint 时调用重计算函数，并把 `use_cache` 设为 False、传入的 past cache 设为 None。Ouro 把 hidden 作为位置参数传入，每次调用的 `current_ut` 等 kwargs 通过 partial 绑定；因此，`OuroModel.forward` 中没有显式 checkpoint 分支，并不表示该功能无效。

训练应显式使用 `use_cache=False` 且不携带外部历史 cache，使进入层前构造的 mask/position 与重放计算一致。checkpoint 通过重算保存梯度路径，不是 Huginn 的 no_grad 前段截断；冻结 embedding 的适配器训练还需满足所选 reentrant 模式的输入梯度条件。[Transformers v4.55.0 [GradientCheckpointingLayer](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/modeling_layers.py) 与 [gradient_checkpointing_enable](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/modeling_utils.py)；Ouro 两个固定模型文件的继承及调用相同。]

### 依赖兼容 自定义缓存必须满足所选基类的接口

发布配置中的 `transformers_version=4.55.0` 是来源记录，不能单独当作固定模型代码在该版本可运行的保证。当前固定 README 同时保留“使用 <4.56、推荐 4.54.1 或更早”和“已合入修复 ≥4.56 缓存兼容问题”的说明；这些文字没有构成一份经过核验的兼容矩阵。本节只核对 **固定 Ouro revision 与 Transformers v4.55.0** 的具体对象契约，不外推其他版本。[固定 README](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/README.md)；[发布配置](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json)。

这一组合的自定义 cache 构造存在确定的 Python 描述符冲突。v4.55.0 的 `Cache.key_cache` / `Cache.value_cache` 是无 setter 的 property；固定 `UniversalTransformerCache.__init__` 却直接执行 `self.key_cache=[]` / `self.value_cache=[]`，并且不调用父类初始化。属性赋值仍受继承来的 data descriptor 控制，因此第一项就抛出 `AttributeError: property 'key_cache' ... has no setter`；跳过 `super().__init__` 不会绕过 property。已将这几段未改动的上游 Python AST 取出执行，复现了该错误；1.4B 与 2.6B 的自定义 cache 类 AST 相同。这是对象接口检查，没有加载权重或执行模型张量前向。[Ouro `UniversalTransformerCache`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)；[Transformers v4.55.0 `Cache`](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/cache_utils.py)。

触发范围可从调用链精确判断：`OuroModel.forward` 在 `use_cache=True` 时调用 `needs_universal_cache`；传入 None 会创建上述类。v4.55.0 的空 `DynamicCache` 也没有该检测所用的 `layer_class_to_replicate` 标志，且其层槽少于 L×T，因而会走替换分支。已有非空且不足 L×T 槽的 cache 则在检测函数中直接报容量错误。`use_cache=False` 且无历史 cache 的训练不经过这条构造路径；这不等于已认证整套无缓存训练环境可运行。

还有一个独立接口责任：v4.55.0 的 `create_causal_mask` 在普通 None／二维 mask 的标准后端路径中调用 `past_key_values.get_mask_sizes(cache_position,layer_idx)`；基类该方法读取 `self.layers[layer_idx]`。Ouro 自定义类只维护 key/value 列表，把 `layers` 留空且没有重写这个方法。因此仅绕过上述 property 赋值仍不足以满足 mask 接口。用未初始化对象单独检查该继承方法，会得到空列表的 IndexError；这只是分离第二个接口前提的诊断，不是把绕过构造当作修复方案。预先提供 mask 字典或某些已准备 mask 的路径会跳过这次 mask 构造，但不会消除前面的构造冲突。[v4.55.0 `_preprocess_mask_arguments/create_causal_mask`](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/masking_utils.py)。

因此缓存等价证明仍是满足完整 KV、位置、mask 等前提的数学命题；实际接入须先固定模型 revision、依赖版本和 cache 类，验证构造、L×T 地址容量、mask 长度、追加与重排，再做全前缀对逐 token 的数值一致性。前述 v4.55.0 loss／checkpoint 公式仍描述相应依赖函数的语义，但不能被解读成该版本整条生成调用链已经验收。其他版本与完整硬件执行在本次核验范围之外。

### Q-exit 的调用契约与有限精度边界

固定发布配置把 `early_exit_threshold` 设为 1.0；`forward` 对传入的 `None` 使用该实例默认值。因此参数组合是否真的进入预期分支，需要按执行优先级判断。

| 调用条件 | 实际选择的分支 | 可审查的用途 |
| --- | --- | --- |
| `labels` 非空 | 无论退出选择参数如何，调用 `compute_expected_logits` 后计算 loss | 不能用 `exit_at_step` 实现固定轮次 SFT |
| `labels=None`，合法 `exit_at_step` | 直接选零基指定轮次 | 强制最终读出应显式选 T−1；仍执行全部循环 |
| 未指定合法步数，实例阈值仍为 1.0，`use_weighted_exit=True` | 先进入 CDF 阈值分支 | 该 flag 单独开启不保证返回加权 logits |
| 显式传 `exit_threshold=None`，实例阈值仍为 1.0 | 回落到实例默认阈值 | `None` 不是关闭阈值策略的指令 |
| 实例的 step 与 threshold 均为 `None`，且无覆盖参数，`use_weighted_exit=True` | 才进入加权 logits 推理分支 | 如需此预测器，应明确配置状态并核验分支 |

数学上，在有限 gate logits、精确实数运算且 0<λᵣ<1 时，r<T 的 CDF 严格小于 1，所以 q=1 选择末轮。浮点执行不具有这项严格保证。以 T=4、前三个 hazard 都为 7/8 为例，

```math
\begin{aligned}
p&=(7/8,\ 7/64,\ 7/512,\ 1/512),\\
F&=(7/8,\ 63/64,\ 511/512,\ 1).
\end{aligned}
```

若 CDF 存储为 BF16 并按最近偶数舍入，511/512 正处于 1 与相邻较小可表示值的中点，会舍入为 1；`cumulative_probs >= 1` 因而可在第三轮首次成立。即使尾部概率在真实数值下仍为 1/512，q=1 也不再等价于显式末轮选择。这个例子是对给定浮点格式的确定性检查，不声称当前模型频繁触发；实际还受 gate 输出 dtype、kernel 和数值影响。源码未在 hazard/PDF/CDF 路径显式升为 FP32，发布配置声明 `torch_dtype=bfloat16`，因此必须保留这项边界。

更一般地，设真实与数值 CDF 的逐点误差不超过 ε。若 r* 为精确退出轮次，且 r*>1 时 F(r*−1)<q−ε、同时 F(r*)≥q+ε，则首次越阈值轮次不变。证明只需利用 CDF 单调性：所有前轮仍低于 q，r* 仍达到 q。阈值附近没有这个余量时，不能仅用“小浮点误差”推断退出决策不变。固定最大深度评测应直接指定 `exit_at_step=T−1`；概率退出实验则记录 dtype、阈值与真实选中轮次。

依据：[原文 §3.2、§5.4.1](https://arxiv.org/abs/2510.25741v6)；固定 [`OuroForCausalLM.forward`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) 与 [`config.json`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json)。精度反例和阈值余量条件是本页推导；它们不改变“当前发布实现先算完所有轮再选输出”的结论。

## 缓存实验：decode结果不能直接外推到prefill

论文§5.4.2区分两阶段：prefill共享导致GSM8K下降超过10点，需保留对应轮次的KV；decode可尝试只用最后一轮、第一轮或平均KV。Table 14的原始结果是：

| decode KV策略 | GSM8K | MATH-500 | 论文所报KV内存减少倍数 |
|---|---:|---:|---:|
| full四轮cache | 78.92 | 82.40 | 1.00× |
| first-step only | 18.73 | 8.43 | 4.00× |
| last-step only | 78.85 | 80.40 | 4.00× |
| averaged | 78.73 | 78.52 | 4.00× |

最后轮策略在GSM8K损失.07点，但MATH-500损失2.00点；平均策略MATH-500损失3.88点。不能统称“完全无损”，也不能从KV减少4倍推出整机显存、峰值prefill显存或端到端延迟都减少4倍。该实验支持针对decode阶段的近似缓存研究；发布HF实现是否真正执行此策略是另一个问题。[p.18，表格已图像核对]

### 推导 三种交叉熵的差距与 KV 减少的分母

固定一个 token，令退出权重为 p_r，第 r 轮完整词表分布为 q_r=softmax(z_r)。定义概率混合 m、归一化几何混合 g 与三种 loss：

```math
\begin{aligned}
m(v)&=\sum_rp_rq_r(v),\\
g(v)&=\frac{\prod_rq_r(v)^{p_r}}
{\sum_u\prod_rq_r(u)^{p_r}}
=\operatorname{softmax}\left(\sum_rp_rz_r\right)_v,\\
L_{\rm step}&=-\sum_rp_r\log q_r(y),\\
L_{\rm prob}&=-\log m(y),\qquad
L_{\rm logit}=-\log g(y).
\end{aligned}
```

它们的差距不只是一个笼统的 Jensen 不等式。对正概率、Σp=1 的同一组预测，令后验重加权 π_y(r)=p_rq_r(y)/m(y)，可直接展开 KL 得到

```math
\begin{aligned}
L_{\rm step}-L_{\rm logit}
&=\sum_rp_rD_{\rm KL}(g\|q_r)\ge0,\\
L_{\rm step}-L_{\rm prob}
&=D_{\rm KL}(p\|\pi_y)\ge0.
\end{aligned}
```

第一个差距与真实标签 y 无关，度量各轮整分布对几何中心的分歧；第二个取决于各轮对当前真实 token 的相对支持。L_prob 与 L_logit 之间没有普遍顺序。由此，HF 的 CE(加权 logits) 不能替代论文的“加权各轮 CE−βH”，其预测器和 gate 的梯度都会不同。[原文 Eq.(4)；固定 [compute_expected_logits 与 labels 分支](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)；恒等式是我们的推导。]

### 梯度反例 HF 加权 logits 可把 gate 推向相反方向

两种目标的区别还会改变gate朝哪边学习。论文损失比较各轮单独预测真实标签的好坏；加权logits的CE则评价这些轮次合成后能否压过错误类别，所以还受各轮错误偏好的影响。下面只保留两个深度的logits z₁、z₂，用p=σ(a)分配第一轮权重、1−p分配第二轮权重。预测器固定，只调整gate；记g为混合logits的softmax、eᵧ为真实标签的one-hot向量：

```math
\begin{aligned}
\frac{\partial L_{\rm HF}}{\partial a}
&=p(1-p)\langle g-e_y,z_1-z_2\rangle,\\
\frac{\partial L_{\rm I}}{\partial a}
&=p(1-p)\left[\ell_1-\ell_2+
\beta\log\frac{p}{1-p}\right].
\end{aligned}
```

取真实类别为第一个token：第一轮给它0.40，第二轮给它0.35，但两轮主要错误分别落在另外两个类别，即q₁=(0.40,0.59,0.01)、q₂=(0.35,0.01,0.64)。令zᵣ=log qᵣ、p=0.8、β=0.05，代入两式，HF导数约+0.221156，论文导数约−0.010275。梯度下降于是降低HF的第一轮权重，却提高论文目标的第一轮权重。

这里第一轮单独预测真实标签更好，而混合目标还会利用两轮对错误类别的不同偏好。这份三词表解析反例说明，换用HF labels分支可以连深度路由的学习方向一起改变，差别远超过总loss的常数平移；数值并非Ouro权重的实测。

实现这两项目标的分岔点在读出之后。`OuroModel.forward`已保存`hidden_states_list/gate_list`，`OuroForCausalLM.forward`据gate形成PDF；`compute_expected_logits`随后把逐轮LM logits乘权重相加，交给一次`loss_function`。要训练论文Stage I，应在这里保留逐轮CE，使用同一有效位置mask加权，再加入退出熵；gate第二阶段则另从冻结LM取得detached改善标签。`labels`与`exit_at_step`控制的内容，并不包括这次目标替换。[原文 §3.3 Eq.(4)、§3.4 Eq.(6)](https://arxiv.org/abs/2510.25741v6)；固定 [`OuroForCausalLM.forward/compute_expected_logits`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)。上述反例经中心差分核验，未运行模型。

工程执行量也应直接从循环结构计数。官方 `OuroModel.forward` 已执行 T×L 个共享 block，之后才选择第 r 轮输出。因此改变 q 可以改变输出分布，却不减少这条实现中的 Transformer block 调用数；只比较“选中轮次”会低估实际成本。Stage I/II 的语义算法与发布 HF forward 应分别保留，不能合并成一份想象中的原始训练实现。[`OuroModel.forward` 与 `OuroForCausalLM.forward`。]

KV 缓存的全展开地址为 (r−1)L+ell。设数据元素占 b 字节、batch B、prompt 长度 n_p、已生成长度 n_d、KV 头数 H_kv、每头宽 d_h，则全缓存内容量为

```math
M_{\rm full}=2bBLH_{\rm kv}d_h\,T(n_p+n_d).
```

若一种实现保留 prompt 的 T 份 KV，只把新增 decode KV 压成一份，则

```math
\begin{aligned}
M_{\rm mixed}&=2bBLH_{\rm kv}d_h(Tn_p+n_d),\\
\frac{M_{\rm full}}{M_{\rm mixed}}
&=\frac{T(n_p+n_d)}{Tn_p+n_d}\in[1,T].
\end{aligned}
```

这个条件计算说明为何必须交代 prefill 结束后究竟保留哪些缓存。它不是对 Table 14 未公开代码的反向猜测：论文报 decode KV 减少四倍，但不足以确定上述混合保留策略是否就是实验实现；全机显存还包括权重与工作区。压缩后的 KV 与每轮原本 K/V 不相等时，质量变化需实测，不能用存储公式证明输出等价。[原文 §5.4.2/Table 14；当前 `UniversalTransformerCache.update` 仍保留完整展开槽位。]

### 缓存等价需要同一个因果计算图

全前缀 prefill 与逐 token cache decode 的一致性可以表述为一个有条件的执行等价命题。固定参数、循环数 T、token 序列、位置与因果 mask；关闭随机 dropout，且每个展开层都保留精确 K/V，不做跨轮替代。在实数运算下，两种执行产生相同的逐轮 hidden 与 logits。工程浮点实现应比较容差，不能要求不同 kernel 逐 bit 相同。

证明按展开层 d=(r−1)L+ℓ 与 token 位置作归纳。某展开层的当前 query 及各个历史 K/V，均由该层之前的同一状态计算；因果 mask 排除了未来 token，因此历史位置的 K/V 不会因以后追加 token 而改变。只要 cache 槽位 d 保存的正是这些历史值，当前 attention 就与全前缀重算相同；残差、MLP 和逐 token RMSNorm 也相同。沿 d 推进即可得到下一层与下一轮一致。这说明参数共享并未改变等价证明，保存正确的展开地址才是关键。

实现入口依次为 `OuroModel.forward` 构造 `cache_position/position_ids`、因果 mask 和 RoPE，`OuroAttention.forward` 以 `current_ut * num_hidden_layers + layer_idx` 读取槽位，`UniversalTransformerCache.update` 沿 token 轴追加 K/V。只要每次 forward 都跑完全部 T 轮，各槽位可保持相同有效前缀长度。输出层在最后选择较浅 hidden，不会删除已计算的深层 KV；所以当前 Q-exit 读出仍可以依赖此前 token 的全部深层缓存。

若改为真正的 token 早退，令历史 token u 仅执行到 Rᵤ 轮，而当前 token t 需要第 r 轮，则该轮第 ℓ 层需要的地址满足

```math
(r,\ell,u)\text{ 可直接读取}\quad\Longleftrightarrow\quad r\le R_u
```

这里按完整轮次停止，且只讨论该策略实际保存的缓存。当 r>Rᵤ 时，源算法中的 K/V 尚未计算。重新补算、使用其他轮的 K/V、忽略该历史位置或将历史 token 多算到 T，分别定义不同的计算或成本；不能把前一节“全计算后选输出”的质量曲线直接当成物理早退系统的质量—成本曲线。KV 精确相同是保持计算图的充分条件；不同 KV 也可能偶然给出同样输出，但不存在一般等价保证。

平均缓存同样需要独立评估，不能解释成对每轮 attention 输出取平均。一个固定 query 的单头反例取 q=1、缩放为 1、两个历史位置，K₁=(0,0)、K₂=(0,log 3)，V₁=V₂=(0,2)，则

```math
\begin{aligned}
\tfrac12[\operatorname{Attn}(q,K_1,V_1)
+\operatorname{Attn}(q,K_2,V_2)]&=1.25,\\
\operatorname{Attn}\!\left(q,\tfrac{K_1+K_2}{2},\tfrac{V_1+V_2}{2}\right)
&=\frac{2\sqrt3}{1+\sqrt3}\approx1.267949.
\end{aligned}
```

差异来自 softmax 非线性；这不评价论文 Table 14 的质量高低，而是说明其 averaged cache 是一种需实测的算子近似。原文没有公开对应完整实验实现，因此不推定它平均的是 RoPE 前还是后、保留多少 prompt 缓存、如何处理额外位置。后续复现应先做固定深度全前缀与 cache decode 的数值等价检查，再逐项引入 early exit、last-step 或 averaged 策略，以识别改变发生在哪一条状态依赖上。

依据：[原文 §5.4.2/Table 14](https://arxiv.org/abs/2510.25741v6)；固定 [`OuroModel.forward`、`OuroAttention.forward`、`UniversalTransformerCache.update` 与 `eager_attention_forward`](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)。执行等价证明、缺失地址条件和平均缓存反例为本页推导；未执行 Ouro 模型或复现原表。

#### 算法 Ouro 发布 forward 的全循环与输出选择

**输入** 输入 token 或 embedding、因果 mask、位置、已有完整展开 KV、T 与 L、labels 或退出选择参数。

**输出** 与所选分支一致的 logits/loss，以及全部已执行轮次产生的 KV。

**执行前提** 固定 HF 1.4B revision；推理选择优先级为合法 exit_at_step、阈值、weighted-exit，最后回退末轮。

![算法 Ouro 发布 forward 的全循环与输出选择](visualize:fde1_bGliZmlsZV9EWUxPdHpVRW9yWXRyMDhDLXgwVmd3_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 逐行语义对应 [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) 中 OuroModel.forward、OuroAttention.forward、UniversalTransformerCache.update、OuroForCausalLM.forward。按输出轮次选择前已经执行全部 T×L 个 block；这不是物理提前终止算法。 labels 分支的内部位移、有效 token 归约和 logits_to_keep 对齐，依赖前述 Transformers loss 契约。

## 性能、外推与消融：明确Base和Thinking

| 对象与设置 | 核对结果 | 原始定位 |
|---|---|---|
| Ouro-1.4B Base，7.7T，R4 | MMLU 67.35，BBH 71.02，GSM8K 78.92，MATH500 82.40 | Table 7，p.13 |
| Ouro-2.6B Base，7.7T，R4 | MMLU-Pro 55.73，BBH 80.46，MATH500 90.85 | Table 8，p.14 |
| 1.4B Base按深度，MMLU 5-shot | R1=41.21，R4=67.45，R8=64.49 | Table 10，p.15 |
| 2.6B Base按深度，MMLU 5-shot | R1=51.55，R4=74.60，R8=72.24 | Table 11，p.15 |
| 1.4B/2.6B Thinking，R4 | AIME24 pass@1 65.0/64.7；AIME25 46.3/50.3 | Table 9，p.14 |

Table 7和Table 10对1.4B MMLU分别写67.35与67.45；本页保留原表差异，不自行合并。Base表与Thinking表评测管线不同：Base的大部分常规任务使用lm-eval-harness，代码任务使用evalplus；但Table 16明确将MATH500列为in-house、5-shot CoT、strict match，不能一概归入lm-eval。Base的GSM8K则为lm-eval、3-shot CoT、strict match。Thinking用in-house harness、统一prompt与LLM-as-judge，默认temperature=1、top_p=.7；AIME另报pass@10。深度超过4并不保证继续提高；表头把5–8称外推，但训练流水线Stage 1a曾用8轮，严谨说法应是“超过后续主训练/最终4轮设置”，不能说模型从未见过8轮。[§5.1–5.3、Appendix C.1]

论文中与Qwen/Gemma等的对比可以呈现参数效率，但训练tokens、数据与训练配方不同，不能由此分离出拓扑的纯因果增益。Fig.5的已训练gate优于未做专门gate训练的gate、hidden-difference阈值与固定深度，支持第二阶段校准；曲线读出的约66%/2.5轮应标近似值，不写成无误差精确测量。

### Thinking 的深度结果 最佳点依赖 checkpoint 与任务

Tables 12–13（p.16）把同一 Thinking checkpoint 在 R=1…8 的结果列出。下面同时保留单轮退化、最终训练深度 R=4、原表所测网格上的最佳点和 R=8 的下降；分数单位均为百分比。不能仅用 R=4 的横向模型对比替代这条深度曲线。[原文 §5.3、Tables 12–13](https://arxiv.org/abs/2510.25741v6)。

| Thinking checkpoint | 任务 | R=1 | R=4 | R=1…8 中的最高报告值 | R=8 |
| --- | --- | ---: | ---: | --- | ---: |
| Ouro-1.4B | OlympiadBench | 2.22 | 71.55 | R=5：72.30 | 66.81 |
| Ouro-1.4B | SuperGPQA | 2.03 | 47.37 | R=5：48.73 | 42.88 |
| Ouro-1.4B | AIME 2024 | 0.00 | 65.00 | R=4：65.00 | 38.67 |
| Ouro-1.4B | AIME 2025 | 0.33 | 46.30 | R=5：47.00 | 38.00 |
| Ouro-2.6B | OlympiadBench | 18.96 | 76.44 | R=4：76.44 | 39.26 |
| Ouro-2.6B | SuperGPQA | 15.66 | 53.68 | R=3：56.70 | 46.84 |
| Ouro-2.6B | AIME 2024 | 3.00 | 64.70 | R=3：70.33 | 39.00 |
| Ouro-2.6B | AIME 2025 | 2.00 | 50.30 | R=3：50.67 | 24.33 |

例如 2.6B 的 AIME 2024 从 R=3 到 R=4 下降 5.63 个百分点，而 OlympiadBench 同一变化增加 0.88 个百分点。这两个原表差值已经排除“该 checkpoint 的所有任务都在 R=4 最佳”；但它们不构成逐题支配关系，也没有给出重复采样误差或显著性结论。若把 checkpoint 记为 c、任务记为 b、评测协议记为 π，表中的最佳轮次只是有限网格上的经验选择：

```math
\widehat r^\ast_{c,b,\pi}\in
\underset{r\in\{1,\ldots,8\}}{\operatorname{argmax}}
\widehat A_{c,b,\pi}(r).
```

这一经验最佳点不等于所有输入上的最优停止策略；也不能把原表点估计当作未来数据的最优深度。Thinking 采用 §5.2／Table 17 的 in-house harness、相同提示模板、固定 rubric 的 LLM-as-judge，默认 temperature=1、top_p=0.7；完整逐题样本、judge 配置与各深度采样记录未在本页取得。论文把后段训练／SFT 设为四轮，但早期 Stage 1a 使用过八轮，故表中 R>4 应理解为超过最终四轮设置。与前文 HF 的“完整执行后选输出”分支比较成本时，仍须记录实际执行量，不能仅由选中轮次推算延迟。

## 机制实验：存储容量、知识操作与样本效率

Capo使用随机合成人物履历，1M–40M GPT-2式模型、20K–500K人物、1000次exposure，以属性CE换算知识bits。相同参数、1轮与4轮都约2 bits/parameter；这是这个受控合成存储任务的经验结果，不能升级成所有自然知识的普遍容量上界。[§6.1、Appendix B.1]

Mano要求直接算模23表达式、无中间CoT，区分记住算术规则与组合运用规则。Fig.6在最大表达式长度24时：3层×4轮为92.2%，3层×1轮11.0%，12层×1轮34.8%；前一对近参数匹配，后一对近展开计算量匹配。它展示循环归纳偏置在该组合任务的优势，不能说同FLOPs必然胜出于所有任务。[p.20，Fig.6；Appendix B.2]

三跳QA进一步比较6层×1/2/4轮、不同唯一训练样本量与24层×1的iso-FLOP基线，观察循环模型学习更快、使用更少不同样本；但“相同训练tokens”不等于相同训练FLOPs。Appendix B.3额外提供iso-FLOP比较，且2轮在部分少数据设置弱于24层基线。应连同限制保留，而不是只摘“循环总是更好”。[Fig.7，p.21；Appendix B.3，pp.36–38]

### 结果刻画 Capo 的比特定义与三类受控比较

Capo 的“约 2 bits/parameter”有明确的任务内计量定义。设 N 个合成人物、P 个模型参数，p₁ 为每个人姓名 token 的 NLL 总和再对人物平均，p₂ 为全部属性值 token 的 NLL 总和再对人物平均；两者单位都是 nats，不是平均每 token CE。原文给出

```math
\begin{aligned}
\mathcal R(F)
&=\frac{N}{P\log2}
\big[\log N_0+\log S_0-p_1-p_2\big],\\
\mathcal R_{\max}(F)
&=\frac{N}{P\log2}
\big[\log(N_0/N)+\log S_0\big],\\
N_0&=400\cdot400\cdot1000,\\
S_0&=2(12\cdot28\cdot200)\cdot200\cdot300\cdot100\cdot263.
\end{aligned}
```

这里 log 为自然对数。完全记住数据仍需表示 N 个等概率姓名的不确定性，所以最大值对应 p₁=log N、p₂=0，而非两个 loss 都为零。该归一化计量依赖合成空间和评测条件，不能把经验斜率外推为所有自然知识或实数权重的通用信息容量上界。[原文 Appendix B.1 Definition 1，pp.35–36。]

Mano 则在有限域 F₂₃ 上计算表达式树；“规则是否已存储”和“能否组合执行这些规则”是不同目标。对给定表达式分布，定义 A(L,R)=Pr[模型输出等于表达式的模 23 值]。同为三层时比较 R=1 与4，主要控制独立参数；比较三层×四轮与十二层×一轮，主要控制展开层数和近似运算量。后者不控制独立参数数，前者不控制计算量。论文的两种比较应分别解释，不能同时称为“所有预算都相等”。[§6.2/Fig.6、Appendix B.2。]

附录B.2（p.36）还明确扫描学习率{5×10⁻⁵,10⁻⁴,2×10⁻⁴,5×10⁻⁴}、运行3个随机训练seed并报告best performance；这不是已报告的跨seed均值／标准误。最终如何联合选择LR、seed与checkpoint尚需日志，不能由best自行还原。评测限定最难表达式长度ℓ=L，使用单token exact match。

附录另列AdamW的β=(0.9,0.98)、ε=10⁻⁶、weight decay=0.1及全局梯度裁剪1.0；warmup1000步后cosine降至峰值LR的0.1，global batch128、context1024、BF16并对pack内不同表达式作attention隔离。80K／110K／200K为原文列出的不同难度训练预算，其与正文图中各运行的逐项映射仍需配置与日志，不能自行填成同一协议。

### Mano 的前缀例子需要按语法解析

原文附录B.2给出的运算token序列为“− × a b + c d”，但随后配的中缀说明为ab+(c−d)。二元前缀语法应递归消费算子及左右子树，因此前者实际表示ab−(c+d)。在F₂₃中，两者并不相同：

```math
\operatorname{eval}_{23}(\mathtt{-\ *\ a\ b\ +\ c\ d})=[ab-(c+d)]_{23},\qquad
[ab+(c-d)]_{23}-[ab-(c+d)]_{23}=[2c]_{23}.
```

例如a=b=c=d=1时，前缀式为22，原文中缀式为1。这是原文记号冲突，尚不能说明作者训练数据生成器或实验分数有误；完整受控任务生成代码仍未取得。

参考解析步骤为：从算子序列读一个token；数值则返回其有限域值，算子则递归解析左子树、再解析右子树，并依次执行对应运算；最后检查序列完整消费、二元运算符数与len标签一致、无缺失操作数。本文采用这一明确语法作为数学参考，不伪称已经还原作者生成器。附录中L同时涉及长度上限及模型深度表述，不能据符号重用把正文三层／十二层对照和附录所有运行自动视为同一配置。

三跳 QA 的样本效率也需要两个横轴。令 U 为不同训练问题数、D 为处理过的总 tokens、F_train 为训练计算量；重复 exposure 会使同一 U 对应不同 D。定义达到目标准确率 a 所需的 U_a 或 F_a，才能分别陈述数据效率或计算效率。原文 Appendix B.3 的 iso-FLOP 对照补上了主图中训练计算不同的因素，同时保留部分少数据设置下两轮模型弱于非共享基线的结果。[§6.3/Fig.7、Appendix B.3。]

在主模型深度比较中，逐题正确率变化仍是“改对数减改错数”除以总题数；MMLU 总平均上升不保证每个科目、每个样本或 R>4 都单调。Base/Thinking 的不同提示与评分协议、早期曾在 R=8 预训练、Table 7/10 的数值差别都应随结果保存。跨 Qwen/Gemma 的同参数比较说明一种可获得的质量—参数配置，不分离训练数据、tokens 和拓扑各自的因果贡献。[Tables 7–13、Appendix B.4/Table 15。]

### MMLU 分科证据 相对增幅与百分点分别报告

Appendix B.4/Table 15（pp.38–39）进一步比较 57 个 MMLU 子领域的 Loop 1 与 Loop 4，用于检验“循环更有利于知识组合”的经验解释。下面按原表行保留基础准确率、绝对变化和原表的相对增幅，不把相对百分比当作新增的正确题比例。

| 子领域 | Loop 1 accuracy | Loop 4 accuracy | 绝对变化，百分点 | 相对增幅，原表 % |
| --- | ---: | ---: | ---: | ---: |
| elementary_mathematics | 0.3095 | 0.7910 | +48.15 | +155.5556 |
| formal_logic | 0.2381 | 0.5794 | +34.13 | +143.3333 |
| global_facts | 0.3600 | 0.3900 | +3.00 | +8.3333 |
| moral_scenarios | 0.2436 | 0.2626 | +1.90 | +7.7982 |

绝对变化由原表显示的四位小数准确率计算，存在原始数值舍入误差。令 a_{s,r} 为子领域 s 在轮次 r 的准确率，a_{s,1}>0 时，两种变化分别为

```math
\Delta_s^{\rm pp}=100(a_{s,4}-a_{s,1}),\qquad
G_s^{\%}=100\frac{a_{s,4}-a_{s,1}}{a_{s,1}}.
```

例如 elementary_mathematics 的原表 +155.5556% 是相对于 Loop 1 的增长，绝不是增加 155.5556 个百分点。准确率栏只有四位小数，从显示值重算会得到约 +155.5735%，与原表最后几位不同可以由舍入解释；本页分别保留原表相对增幅与从显示值计算的绝对差，不伪造更多精度。

分科结果也没有逐轮单调性：college_chemistry 的 Loop 1/2/3/4 分别为 0.2600/0.5000/0.5500/0.5000，Loop 3→4 回落 5.00 个百分点。B.4 的大小排序与原文知识组合假设相容，但这些子领域同时存在初始准确率、难度、样本数和题型差异；相对增幅又以各自 Loop 1 分数为分母，因此不能由分科排序单独识别“记忆容量”和“知识操作能力”的因果贡献。Table 15 表题未独立标明所用 checkpoint，本页不通过相似均值猜定其身份，也不把四行摘要当作 MMLU 总分的重算。[原文 Appendix B.4/Table 15](https://arxiv.org/abs/2510.25741v6)。

## 图可达性定理：存在性、表示与计算成本

定理固定一个隐藏知识图G编码到参数，输入提供上下文图G_ctx。n个token、每个宽度2n；上半部为单位阵I，下半部为R₀=I∨A_ctx。用**无mask全注意力**和显式阈值算子LN_tau(X)=1{X>1/(2n)}，不是普通LayerNorm或RMSNorm。特定构造的注意力做布尔路径合成，MLP注入隐藏图邻接矩阵A：

```math
\begin{aligned}
R_{r+1}&=R_r\lor R_r^2\lor A,\\
C&=I\lor A_{\rm ctx}\lor A,\\
C^{\,2^{r-1}}&\preceq R_r\preceq C^\ast\quad(r\ge1),\\
L&=1+\left\lceil\log_2\max\{1,D\}\right\rceil.
\end{aligned}
```

这里 ∨ 为逐元素布尔或，矩阵乘方使用布尔半环，C* 为传递闭包。D是可达点对中最大的有限最短路长度，允许长度0路径。L轮得到传递闭包并在更多轮保持正确。证明使用有限注意力尺度β=log(4n²)：被选key权重至少1/n，未选key总泄漏至多1/(4n)，正阈值消除softmax漏出的假阳性；零阈值不成立。[原文Theorem 1，p.22；Theorem 2与证明，pp.38、40–41]

这是构造性存在定理。每轮稠密计算O(n³)、状态O(n²)，O(logD)描述顺序轮数，不是总工作量；输入已是邻接列编码，edge-list转换成本未包含。它未证明因果mask下成立，未证明标准norm下成立，未证明SGD能学到该构造，也未证明宽度2n对所有表示都必要。不能拿它直接解释真实Ouro全部性能优势。

### 构造证明 注意力如何完成布尔平方

沿用原文“列为 token”的约定，固定隐藏图 A 与输入上下文图 A_ctx。完整状态 H=[I;R] 的每个 token 宽度是 2n。下面全部矩阵块为 n×n；取 β=log(4n²)、τ=1/(2n)，构造

```math
\begin{aligned}
K&=\beta\begin{bmatrix}I&0\\0&0\end{bmatrix},&
Q&=\begin{bmatrix}0&I\\0&0\end{bmatrix},\\
V&=\begin{bmatrix}0&0\\0&I\end{bmatrix},&
W_1&=\begin{bmatrix}0&0\\A&0\end{bmatrix},\qquad W_2=I_{2n}.
\end{aligned}
```

原文 attention 的 score 定义不含默认的 1/√d 缩放，因而 HᵀKᵀQH=βR。若用默认 scaled-dot-product kernel，须显式设 scale=1，或将 K 再乘 √(2n) 抵消 head_dim=2n 的缩放；否则给定 β 下的泄漏界不能直接沿用。第 j 个 query 的选中 key 集合为 S_j={k:R_kj=1}，d_j=|S_j|≥1；对所有 key，

```math
\begin{aligned}
\alpha_{k,j}&=\frac{\exp(\beta R_{k,j})}
{d_je^\beta+n-d_j},\\
k\in S_j&\Longrightarrow\alpha_{k,j}\ge1/n,\\
\sum_{k\notin S_j}\alpha_{k,j}
&\le ne^{-\beta}=1/(4n)<\tau.
\end{aligned}
```

value 的下半部是 R 的第 k 列。若存在中间点 k 使 R_ak R_kj=1，第 a 个输出坐标至少收到 1/n 的贡献，超过阈值；若不存在这样的中间点且残差 R_aj=0，所有贡献只能来自未选 key，总量不超过 1/(4n)。因此阈值后恰为 R∨R²。上半部不被 attention/MLP 写入，仍为 I；MLP 借助该 I 将 A 写入下半部，于是下一轮恰为 R∨R²∨A。正阈值提供的判别余量至少为 1/(4n)，零阈值会把 softmax 泄漏误判为边。

再令 C=I∨A_ctx∨A。第一轮覆盖 C，且没有虚构路径。若第 r 轮覆盖 C 的 2^(r−1) 次幂，平方就覆盖 2^r 次幂；真实路径的拼接仍是真实路径，故

```math
\begin{aligned}
C^{2^{r-1}}\preceq R_r\preceq C^\ast
&\Longrightarrow
C^{2^r}\preceq R_{r+1}\preceq C^\ast,\\
2^{L-1}\ge\max\{1,D\}
&\Longrightarrow R_L=C^\ast.
\end{aligned}
```

C 含自环，短路径可用零长度步骤补齐到幂次长度；D 是最大的有限最短路距离。到达闭包后，后续平方与插入 A 不再改变它。这同时证明覆盖性、正确性与继续迭代不出错，而非只给“翻倍增长”的直觉。

工程上，构造可以用 bool 矩阵乘积加 OR 做参考核验：初始化 I∨A_ctx，反复 R←R∨R²∨A，到指定轮数后读取 R_ts。若事先不知道 D，直接取不需要求图直径的保守上限 L=1+ceil(log₂ max{1,n−1})；不能把计算 D 的代价隐藏在算法输入中。用 dense attention 按给定参数实现时，每轮通常 O(n³) 算术、O(n²) 状态；使用这一保守轮数的总工作为 O(n³(1+log max(1,n−1)))。以 D 表示的较紧轮数是正确性的保证界，不是免费获得 D 的子程序。构造依赖无 mask 全注意力、输入邻接列编码、随 n 增长的宽度和显式布尔阈值，不能替换成实际因果语言模型的普通 RMSNorm 后仍宣称证明成立。[原文 Theorem 2 与完整证明，Appendix B.5 pp.38、40–41；矩阵与推导按原文整理。]

## Faithfulness、安全与状态证据的限度

论文在1000个Quora Question Pairs上用逐层线性probe与跨轮答案一致率观察决策是否还会改变：A[2,4]=361，即36.1%；A[2,3]=551，即55.1%。前几轮未固定最终答案，与更多循环修改暂定答案一致。但论文明确没有干预latent推理，只使用observational proxy。所以“真实因果faithfulness已证明”过强；逐轮可读出、决策可变、推理步骤正确、反事实因果影响是不同命题。[§7.2/Fig.9，pp.24–25]

HEx-PHI用330个问题、11类、GPT-4o评分，harmful rate定义为最高分5的占比；Thinking R4的1.4B/2.6B分别.009/.003。Base用greedy、最多128新tokens；Thinking用temperature1/top_p.7、8192新tokens，协议不同。结果显示该评测中的安全改善，不构成所有场景或攻击下的保障。latent PCA分离、原生draft–verify与提前安全筛查是论文分析/讨论内容，后两者没有在本次核对的HF forward中实现为完整部署系统。[§7.1–7.3，pp.22–26]

## Scaling附录：保留反例、拟合范围和目标差异

Appendix D在FineWeb-Edu 20B、53M到1.36B、最大轮次2/4/8等小规模实验，拟合模型规模N、数据D和最大循环T_m对总目标的关系：

```math
\mathcal L_{\rm total}
=E+\frac{A}{(N+a)^\alpha}
+\frac{B}{(D+b)^\delta}
+\frac{C}{(T_m+c)^\gamma}.
```

全数据拟合R²=.9596；固定T_m分别拟合step-wise loss时，T_m=2/4/8的R²=.8898/.8146/.795，且作者剔除某些随数据增多反而loss增加的点。总目标包含退出加权与熵项，某一轮CE则是另一指标；不能把两者的拟合曲线当同一个LM困惑度。[pp.42–50]

附录也报告相同展开深度的非共享Standard总体高于LoopLM；它拥有更多独立参数，不能拿来否定参数效率结论，也不能把文章概括成“loop在所有控制条件下胜出”。Table 18的模型规模标签170M/340M/680M/1.3B与实验说明53M/134M/374M/778M/1.36B不完全一致，应该回到训练记录查清，不能擅自重映射。Appendix E的留出模型/数据拟合给出有限范围的泛化证据，不能直接外推到7.7T主运行或保证推理深度无限增长有益。

## 后训练研究的落点与尚待获得的实现

适合优先研究的对象是：每轮可读出的预测器、固定四轮SFT基线、gate与LM分开的校准、深度/答案联合策略的on-policy rollout、推理预算下正确率与实际FLOPs/延迟，以及prefill/decode的不同cache语义。先解决当前HF“选择输出但不短路计算”与论文训练目标差异，才能让质量—计算前沿可解释。

仍缺原始7.7T完整训练仓库与运行记录来核实gate两阶段的具体时序、初始化、并行/重计算实现、先验消融账目、某些表间差异及Table 14缓存共享代码。上述缺口不影响已核对的官方forward语义，但必须在可复现性清单中保留为待核，不能用通用伪代码冒充已经公开的实现。返回[基座矩阵](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093)。

### 结果边界 观测量 反事实效应与拟合预测分开定义

对 Quora 观测实验，令 Y_i^(r) 为第 i 个问题在第 r 轮读出的标签，则 agreement 矩阵测量的是

```math
\widehat A_{rs}
=\sum_{i=1}^{N}\mathbf 1\{Y_i^{(r)}=Y_i^{(s)}\}.
```

它描述读出的改变，不使用正确答案，也不等于干预内部状态之后最终输出的改变。线性 probe 的 AUC 同样测量该 probe 类的可预测性；低 AUC 不能排除更强解码器已经能预测最终答案，高 AUC 也不证明后续计算没有因果作用。若定义反事实效应，需要明确被干预的完整状态、保持不变的上下文/cache，以及后续相同算子的执行，例如比较 Y_T(do(S_r=s')) 与 Y_T(do(S_r=s))。论文没有做这项干预，因此 agreement 与 probe 是有信息的观测代理，尚未估计这样的因果效应。[原文 §7.2/Fig.9。]

安全实验的 harmful rate 为 N⁻¹Σ_i1{judge_score_i=5}，与五级平均分是不同统计量；模型、采样、输出长度与 judge 都属于其条件。主表中的低 harmful rate 支持给定 HEx-PHI 协议的结果，不能由 PCA 的可分形状替代，或在 Base/Thinking 的不同生成协议之间直接解释为纯深度因果效应。[原文 §7.1/Fig.8。]

规模律则需要明示被拟合的量。Stage I 的退出加权 CE 与熵目标满足

```math
\min_r\ell_r-\beta\log T
\le L_{\rm total}\le\max_r\ell_r.
```

所以 exp(L_total) 不是普通 LM perplexity；固定轮次 CE、预测概率混合 NLL、含熵 total loss 应分别拟合。对于论文的幂律形式，若深度项系数 C>0、指数 γ>0，则

```math
\begin{aligned}
\frac{\partial\widehat L}{\partial T_m}
&=-\frac{C\gamma}{(T_m+c)^{\gamma+1}},\\
\widehat L(T_a)-\widehat L(T_b)
&=C\left[(T_a+c)^{-\gamma}-(T_b+c)^{-\gamma}\right].
\end{aligned}
```

这些是拟合函数在给定参数下的单调性与边际收益，不能推出某个实际 checkpoint 的准确率逐轮单调，也没有计入多一轮的真实系统代价。R²=1−SSE/SST 衡量对选定 loss 数据的误差比例：全点拟合的 R²、留出模型规模、留出数据长度、留出最大轮数的预测应分别记录。相邻 checkpoint 共享一条训练轨迹，不能当成独立重复训练样本；被剔除的异常点与模型规模标签不一致，也都属于拟合数据定义而非可以删去的细节。[原文 Appendix D–E。]

因此后训练接入记录应分三层保存：实际行为策略与计算图、目标函数及梯度边界、质量与真实成本的评测协议。原始 7.7T 训练源码缺失的部分不能由 HF forward 或项目页链接的 v4 reimplementation 代填；本页的论文目标算法是语义规格，发布 forward 算法则是已核代码的行为。[原文 §4–§7；固定 [HF 源码](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) 与前述来源身份核对。]

## 与 TaH2 的概率换元及执行差异

Ouro 用条件退出概率 λ，TaH2 用条件继续概率 g。固定同一个最大深度 M，若两者具有相同逐轮状态与读出，并令 g=1−λ，则截断停止分布相同：

```math
\begin{aligned}
g_r&=1-\lambda_r,\qquad S_r=\prod_{j=1}^{r}g_j,\\
p_r&=S_{r-1}(1-g_r),\quad r<M,\\
p_M&=S_{M-1},\qquad \sum_{r=1}^{M}p_r=1.
\end{aligned}
```

这一代数对应要求共同的截断终点。TaH2 在实际停止深度吸收剩余质量；若提前到 m 轮停止，最后权重是 S_(m−1)，不再保留原 M 下的尾部。只有状态计算和混合读出同时一致，相同权重才意味着相同输出。

条件概率阈值与累计概率阈值对应不同决策。例如每轮 λ=0.3，累计退出概率在第二轮达到 1−0.7²=0.51，而逐轮条件退出概率仍为 0.3。用阈值 0.5 时，CDF 规则在第二轮停止，g<0.5 的继续规则仍会继续。此例演示判据差异，不指定发布模型默认阈值。

工程移植依次明确返回量是 continue logit、hazard、PDF 还是 CDF，实际停止时如何吸收尾部，读出采用概率还是 logits 混合，停止是否真正阻止后续执行，以及缓存是否遵循新路径。TaH2 论文中从 Qwen3-Base 改造的 Ouro 对照与本页原生 Ouro 权重具有不同训练历史。

来源：[Ouro 固定发布计算图](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)；[TaH2 §3–4、Appendix A.2](https://arxiv.org/html/2609.35748v1)。换元是对两种定义的推导；具体执行见 OU2 与 [TaH2 内部循环](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811)。

# 固定版本的文件与函数索引

来源：[huggingface.co/ByteDance/Ouro-1.4B](https://huggingface.co/ByteDance/Ouro-1.4B)；固定 revision：`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [config.json](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/config.json) | 配置 | 发布1.4B；24层、2048hidden、5632intermediate、16heads、R4、threshold1、untied embedding |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | OuroDecoderLayer.forward / GradientCheckpointingLayer继承 | 真实层计算图；attention/MLP输出先norm再加残差，共四个norm；checkpoint由继承的__call__执行 |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | OuroModel.forward | 全栈重复；所有UT轮先完整计算；每轮末norm，保存hidden和gate |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | OuroAttention.forward | KV索引；current_ut * num_hidden_layers + layer_idx |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | UniversalTransformerCache.update | 缓存写入；L×R槽位，沿token维cat；不是last-step/average复用 |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py) | OuroForCausalLM.forward, compute_expected_logits | 输出与训练loss；CDF后gather；zero-based exit_at_step；默认threshold优先于weighted flag；labels总是CE(加权logits)，无熵项/StageII；labels位移/归约交给TF依赖，logits_to_keep未同步选择labels |

来源：[huggingface.co/ByteDance/Ouro-2.6B](https://huggingface.co/ByteDance/Ouro-2.6B)；固定 revision：`1ed04250da1a9936042725d302e81c8fa2ab5abd`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [config.json](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/config.json) | 配置 | 发布2.6B；48层，其余关键维度与1.4B相同；R4 |
| [modeling_ouro.py](https://huggingface.co/ByteDance/Ouro-2.6B/blob/1ed04250da1a9936042725d302e81c8fa2ab5abd/modeling_ouro.py) | OuroModel.forward, OuroForCausalLM.forward, UniversalTransformerCache.update | 推理；相同主结构与完整多轮输出选择范式 |

依赖源码：两个 Ouro 发布配置记录 [Transformers v4.55.0](https://github.com/huggingface/transformers/tree/v4.55.0)；以下固定到该版本解释 loss、checkpoint 和 cache 接口。缓存构造已发现具体冲突，不能把该版本字段或单个函数语义当作整条调用链兼容保证；实际运行环境仍须锁定并验收。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [loss/loss_utils.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/loss/loss_utils.py) | ForCausalLMLoss, fixed_cross_entropy | logits升FP32；内部next-token shift；-100忽略；mean或sum/num_items_in_batch；显式shift_labels路径 |
| [modeling_utils.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/modeling_utils.py) | PreTrainedModel.loss_function, gradient_checkpointing_enable, _set_gradient_checkpointing | 选择ForCausalLM loss；向继承层设置重计算函数；默认reentrant及PEFT输入梯度处理 |
| [modeling_layers.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/modeling_layers.py) | GradientCheckpointingLayer.__call__ | training且开启时checkpoint；partial绑定每次kwargs；关闭use_cache并清空past cache |
| [cache_utils.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/cache_utils.py) | Cache.key_cache/value_cache/get_mask_sizes；DynamicCache | v4.55只读property与Ouro直接赋值冲突；mask方法依赖layers；版本字段不构成兼容保证 |
| [masking_utils.py](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/masking_utils.py) | _preprocess_mask_arguments, create_causal_mask | 标准后端None/2D mask路径读取cache mask大小；预制mask与字典分支须分别判断 |

来源：[github.com/rkstgr/LoopLM](https://github.com/rkstgr/LoopLM)；核验范围：main（只核README身份，未固定commit或读取其训练源码，不充当官方训练证据）。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [README.md](https://github.com/rkstgr/LoopLM/blob/main/README.md) | 项目身份声明 | 作者项目页链接的训练复现；明确称2510.25741v4 reimplementation；不是原始训练仓库的核实证据 |

# 逐项覆盖核对

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 原生性/upcycling/拓扑 | §3.1, §4.3.1, Fig.4, Tables1–2 | config.json 24/48物理层，OuroModel.forward共享ModuleList | 论文与官方发布代码已核 |
| 完整结构/初始化/归一化 | §4.1 p.9 | OuroDecoderLayer.forward四norm、OuroModel.forward末轮norm、config initializer_range | 已核；不能与Huginn的残差后norm混淆；原始初始化待训练仓库 |
| hazard/PDF/CDF/Q-exit | §3.2 pp.5–6 Eq.(3) | OuroForCausalLM.forward remaining_prob/cumsum/gather | 论文/源码已核；默认参数优先级、末轮吸收余量与BF16 q=1边界；显式末轮选择和概率退出分开 |
| Stage I期望CE与熵 | Eq.(4), §3.3 pp.6–7 | HF labels分支实际是CE(加权logits)，非论文目标 | 差异已明确；相同预测器下gate梯度方向相反的三词表反例经中心差分核验；原文要求任务/熵共用mask及归约，分母不一致改变有效β |
| Stage II detached改善gate | Eq.(5)–(6) p.7 | 当前HF未实现该训练loss | 原文与数学下界已核；w_min≈.438；无训练源伪装 |
| 先验消融 | Appendix A pp.34–35 Fig.10 | 项目页链接reimplementation不是原始运行 | 已核；20B/50K与40K steps账目矛盾 |
| 7.7T阶段/优化/稳定性 | §4 Tables1–5 pp.8–11 | 官方发布模型config；原始flame训练记录未获 | 论文细节已核并标复现缺口 |
| Thinking SFT与失败RL尝试 | §4.4–4.5 p.12 Table6 | 论文LlamaFactory；DAPO/GRPO路径仅原文证据 | 完整保留；区别Base/Thinking及off-policy失配 |
| prefill/decode KV/动态退出 | §5.4.2 p.18 Table14 | OuroAttention.forward cache idx；UniversalTransformerCache.update；所有轮先算后选择 | 已核重大差异；完整展开缓存的条件等价证明、物理早退缺失地址及平均KV非线性反例；Table14图像核对 |
| 基座性能与深度外推 | Tables7–13 pp.13–16；C.1 Tables16–17 p.42 | 发布R4配置；原始各评测器区分 | 已核Thinking全部64个深度分数，保留R1/最佳点/R8与任务差异；Base MATH500为in-house；Table7/10差异及曾训练R8继续保留 |
| 知识容量/Mano/三跳QA/MMLU分科 | §6 pp.19–22, AppendixB.1–B.4/Table15 pp.38–39 | 无原始受控实验代码已核证据；仅论文设定和分科表 | iso-param/iso-FLOP范围保留；分科relative增幅与百分点分开，舍入相容性已核；保留R3→R4下降，非机制因果识别 |
| 可达性存在定理 | Theorem1 p.22；B.5 pp.38,40–41 | 理论构造非真实因果Ouro代码 | 假设、阈值、证明步骤、成本/不成立推广完整保留 |
| faithfulness与安全 | §7 pp.22–26 Figs8–9 | probe/readout为论文实验；非HF部署保证 | 观测证据非因果证明；评估协议限定 |
| Scaling及反例 | AppendixD–E pp.42–54 | 无原始拟合脚本已核 | 公式/拟合R²/剔除点/更多参数Standard对照/表标签差异已保留 |
| GitHub来源身份 | 项目官网Code链接 | rkstgr/LoopLM README自称v4 reimplementation | 已确认不可当原始7.7T官方训练release |
| HF后训练labels与重计算依赖 | 固定Ouro发布代码及配置TF4.55.0；Eq.(4) mask要求 | ForCausalLMLoss/fixed_cross_entropy；GradientCheckpointingLayer.__call__ | 已核内部shift、ignore mask、分母、logits_to_keep对齐与继承重计算；小张量契约模拟/梯度差分通过，非模型训练 |
| 固定依赖的缓存兼容 | 发布config/README；Transformers v4.55.0 Cache及masking_utils | UniversalTransformerCache.__init__、needs_universal_cache、继承get_mask_sizes | 原始AST复现无setter构造错误；空layers另有mask契约缺口；1.4B/2.6B一致；其他版本及完整模型未运行 |
| 原始语料池与阶段来源预算 | §4.2、Tables3–5，pp.9–10 | Table3全部13来源；Stage1／2全部比例；Stage4新监督与回放分开 | 原文及舍入算术已核；HF发布不包含历史采样器与数据恢复栈 |
