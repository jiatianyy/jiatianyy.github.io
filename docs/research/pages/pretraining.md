本课沿一次训练更新，解释原生循环模型怎样从输入数据形成可交付的检查点。先定义监督与深度分布，再推导共享参数的完整梯度和截断梯度，随后连接优化器、数据打包与分布式执行，最后核对各模型配方及后训练交接。

[研究总览](https://chatgpt.com/space/page_e25611bb2c00819195e9c2f172dda6da) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [后训练与接口](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

> [!CALLOUT]
>
> **先读一次更新，再对照六篇配方。** 第一遍沿目标、梯度和 P1 算法走通更新流程；第二遍检查初始化、稳定性、数据与优化器。读完应能解释每一步为何这样计算，并填写基座的预训练与后训练交接表。

# 从数据到可用基座

| 环节 | 要解析的计算与工程 | 学习产物 |
| --- | --- | --- |
| 数据与样本 | 数据来源和配比、去重、tokenizer、packing、长度与预测 mask | 数据到监督 token 的流程与配置来源 |
| 模型初始化 | 参数共享、状态初始化、输入注入、归一化和尺度 | 参数表、状态表、初始化说明 |
| 循环与监督 | 深度取值或采样、监督所在轮次、读出和退出目标 | 训练深度分布与目标函数 |
| 反向与优化 | 完整或截断反传、重计算、梯度累积、损失分母、优化器与调度 | 单步计算图和梯度路径 |
| 稳定性诊断 | 状态范数、梯度范数、损失、跨轮相似度、MoE 路由与退出分布 | 现象、可能机制、论文干预证据 |
| 系统执行 | DP／TP／专家并行、控制流一致性、通信、显存和吞吐 | 训练系统图及成本口径 |
| 检查点与交接 | 权重、配置、tokenizer、optimizer、数据游标和随机状态 | 推理载入与训练恢复各自需要的字段 |

# 训练目标与梯度计算

## 监督目标与深度分布

先看模型在一次训练中究竟收到什么信号。一条样本虽然整段送入网络，每个预测位置仍只能使用注意力规则允许的历史，来预测下一个 token。监督 mask 再从这些预测中选出需要计分的位置。因此，前向读到哪些信息和 loss 对哪些输出提出要求，是两项分别设定的规则。

固定一个更新窗口及其中已经完成 packing 的样本。样本 i 含 T+1 个 token，位置 t 的 logits 对应目标 x_(t+1)；m 表示这个目标是否计入损失。下面先按一次 next-token 对齐写出预测分布与单位置 CE：

```math
\begin{aligned}
y_{i,t}&=x_{i,t+1},\qquad 1\le t\le T,\\
m_{i,t}&\in\{0,1\},\qquad N=\sum_{i,t}m_{i,t}>0,\\
p_{\theta,r}(\,\cdot\mid x_{i,\le t})
&=\operatorname{softmax}(z_{i,t,r}),\\
\ell_{i,t,r}&=-\log p_{\theta,r}(y_{i,t}\mid x_{i,\le t}).
\end{aligned}
```

padding、没有目标的位置及不要求预测的提示位置不计分。拼接独立文档时，还要分别处理边界上的注意力可见性和预测目标。实现中只应建立一次 next-token 对应：如果上游已经构造移位标签，又交给会内部移位的 loss，就会重复移位。因此，先确认哪一层负责把位置 t 的输出与下一个 token 对齐。

训练深度 K 可以固定，也可以从给定分布 q 采样。下面先假设 q 与参数无关，深度权重也不依赖参数；权重和为 1 是本节为了可比较不同 K 所采用的约定，不是所有论文都必须使用的配方。

```math
\begin{aligned}
L(\theta;X,K)
&=\frac1N\sum_{i,t}m_{i,t}
  \sum_{r=1}^{K}w_r(K)\,\ell_{i,t,r},\\
w_r(K)&\ge0,\qquad \sum_{r=1}^{K}w_r(K)=1,\\
\mathcal J(\theta)
&=\mathbb E_{X\sim\mathcal D}\,\mathbb E_{K\sim q(\cdot\mid X)}[L(\theta;X,K)].
\end{aligned}
```

末轮监督取最后一轮权重为 1；等权多轮监督取各轮权重为 1/K。不除以 K 的多轮求和会让更深的样本获得更大的总损失权重。辅助损失、退出熵、路由均衡项及其分母另写，不能借用语言模型损失的名称隐藏这些选择。

在可交换微分与期望的条件下，参数独立的 q 允许直接对采样后的损失求梯度。可学习的深度分布则还包含采样概率的导数；仅对已执行轨迹反传一般遗漏第二项。

```math
\begin{aligned}
\nabla_\theta\mathbb E_{K\sim q_\theta}[L_\theta(K)]
&=\mathbb E_{K\sim q_\theta}[\nabla_\theta L_\theta(K)]\\
&\quad+\mathbb E_{K\sim q_\theta}
 [L_\theta(K)\nabla_\theta\log q_\theta(K)].
\end{aligned}
```

此式假设离散深度分布具有固定支持；状态依赖的停止策略还要展开其动作概率。退出权重下的逐轮交叉熵期望，也不同于先混合各轮概率再计算负对数。必须先确定训练目标，再讨论实现是否正确。

## 共享参数的完整梯度

以全前缀训练的共享块为例，将架构课中的注入张量 U 向量化为 u，并把 prelude 与向量化操作合记为 P。“固定输入”指一次前向中各轮使用同一个 u，u 本身仍可依赖可训练参数。

H 包含完整循环状态，h 是它的向量化，G 是读出函数；跨轮可微缓存或第二路状态也应纳入 H。下面先固定随机初始化、dropout 和离散路由轨迹，只在可微位置求导。

```math
\begin{aligned}
u&=P_\theta(X),\qquad
h_0=I_\theta(u,\xi),\\
h_{r+1}&=F_\theta(h_r,u),\qquad
z_r=G_\theta(h_r),\\
D_u&=\frac{\mathrm d u}{\mathrm d\theta},\qquad
D_r=\frac{\mathrm d h_r}{\mathrm d\theta}.
\end{aligned}
```

同一组 θ 在多个轮次出现，意味着一次参数改动会沿几条路径影响最后的预测。它会改变第一轮输出，而这个改变又传到第二轮；它也会直接改变第二轮所用的运算，后续轮次依此类推。若每轮还重新读取可训练的输入 u，就又多了一条经过 u 的路径。

为把这些影响分开，下面把已有状态和 u 暂时固定，记本轮直接使用参数的偏导为 B；再用 D 表示状态对参数的总导数。所有模块都以总参数向量 θ 为自变量，未使用的分量偏导为零，因此这套记法也覆盖参数分组和 embedding／读出绑权。

```math
\begin{aligned}
J_r&=\frac{\partial F_\theta}{\partial h}(h_r,u),\\
B_r&=\left.\frac{\partial F_\theta}{\partial\theta}\right|_{h_r,u},\qquad
C_r=\frac{\partial F_\theta}{\partial u}(h_r,u),\\
D_0&=\left.\frac{\partial I_\theta}{\partial\theta}\right|_u
       +\frac{\partial I_\theta}{\partial u}D_u,\\
D_{r+1}&=J_rD_r+B_r+C_rD_u.
\end{aligned}
```

递推式的三项正好对应刚才的三条路径。J_rD_r 把此前已经积累的参数影响带过本轮；B_r 记录本轮直接使用 θ 的影响；C_rD_u 记录本轮重新读取可训练输入的影响。Huginn 式重复注入在后续各轮都保留第三条路径，只在起点输入的结构则没有后续的重复注入项。

继续把 D_r 展开，就能看到每次调用的贡献还要经过哪些后续轮次。下面的雅可比乘积 Φ 将这些后续变换连在一起：

```math
\begin{aligned}
\Phi_{r,s}&=J_{r-1}J_{r-2}\cdots J_s,\qquad
\Phi_{s,s}=I,\\
D_K&=\Phi_{K,0}D_0\\
&\quad+\sum_{j=0}^{K-1}\Phi_{K,j+1}(B_j+C_jD_u).
\end{aligned}
```

这是沿依赖路径展开链式法则得到的恒等式。共享参数在每个出现位置都贡献梯度，再按后续雅可比传回；不能只计算最后一次使用，也不能把某一轮梯度机械乘以 K。若各轮读出损失只经 h 和读出参数依赖 θ，则总梯度为：

```math
\begin{aligned}
L&=\sum_{r=1}^{K}L_r,\\
\nabla_\theta L
&=\sum_{r=1}^{K}D_r^\top\nabla_{h_r}L_r\\
&\quad+\sum_{r=1}^{K}
 \left.\nabla_\theta L_r\right|_{h_r}.
\end{aligned}
```

最后一项包含读出层的直接梯度。参数若同时出现在输入、循环块和读出层，其各条路径相加；“冻结骨干”“只训练门控”“截断若干轮”应当分别落实到这些路径，而不是用一句“没有梯度”概括。

## 截断反传与重计算

循环很深时，保存整条反向图会占用较多激活内存。截断反传保留完整的前向计算，却只追踪最后一段计算对参数的依赖。设总深度为 K，只保留最后 b 轮，边界就是 c=K−b；本节先只在末轮计算监督损失。

当 c>0 时，先算出第 c 轮状态，再把这次得到的数值交给后段使用。求后段梯度时不再追问这个边界状态怎样随 θ 变化，但后 b 轮仍使用同一组可训练参数，也仍可直接读取可训练的 u。这就是下式中边界导数置零，而 B_r 与 C_rD_u 继续保留的原因：

```math
\begin{aligned}
\widetilde h_c(\theta;\bar\theta)&=h_c(\bar\theta),\qquad
\widetilde D_c=0\quad(c>0),\\
\widetilde D_{r+1}
&=J_r\widetilde D_r+B_r+C_rD_u,
\qquad c\le r<K.
\end{aligned}
```

严格地说，先在当前参数 θ̄ 计算边界状态，再把它作为常量构造后段的局部替代目标；上式对 θ 求导，最后在 θ=θ̄ 处取值。工程中的 sg／detach 正是这个约定，不是一个既处处等于恒等映射、经典导数又处处为零的普通函数。

因此，截断并没有把共享 core 永久冻结：同一个参数在最后 b 轮再次被使用，仍会收到这些调用的梯度。省去的是跨越边界、经过前 c 轮的依赖，所以相对于完整目标，这通常是有偏的梯度近似。若 b=K，则根本没有需要切开的前段，应从可微初始化开始；此时仍 detach 初态，就会额外删去初始化和起点输入的学习路径。

为节省前段激活，算法先在 no_grad 下计算前 c 轮，再重新计算带图的 prelude，供后段直接注入使用。重算必须复用原 prelude 的随机性，并保持精度与训练模式，使数值输入不变。只把 no_grad 下的 u 原样交给后段，会同时删去 C·D_u 路径。仅在起点输入的模型没有这一后段路径；若 embedding 与读出绑权，仍可能通过读出获得梯度。

激活重计算保存边界输入，并在 backward 时重建中间运算；在相同输入、参数、随机性和副作用约定下，它保持被选计算图的梯度。对已截断的图使用重计算，也只能保持该截断图的梯度。PyTorch 官方说明了前后两次函数调用不一致会导致错误梯度，默认确定性检查也不等于完整数值等价验证。[checkpoint 文档](https://docs.pytorch.org/docs/2.9/checkpoint.html)

对循环实现尤其要核查：重算是否重复追加 KV，是否改变路由随机数，是否原地修改作为边界输入保存的状态。参考算法将循环更新视为返回新状态的纯函数；采用可变缓存的实现需要单独证明其重算语义。

## 状态稳定性可以证明到哪里

模型能否稳定运行更多轮，可以先从一个较小的问题入手：参数与输入不变时，初态中的一点扰动会怎样传到第 K 轮？在当前轨迹附近，一轮的影响由该轮雅可比描述；经过多轮，就要按实际执行顺序把这些矩阵相乘。共享权重仍会遇到不同状态，因此每一轮的局部雅可比也可能不同。

下面先讨论固定有限 K 的局部线性化，所有范数采用同一诱导算子范数。它给出扰动传播的出发点；要进一步得到整个状态域上的保证，随后还需要统一 Lipschitz 条件。

```math
\begin{aligned}
\delta h_K&=\Phi_{K,0}\delta h_0
+o(\|\delta h_0\|),\\
\|\Phi_{K,0}\|
&\le\prod_{r=0}^{K-1}\|J_r\|.
\end{aligned}
```

只有在相关状态域内存在统一 Lipschitz 上界时，才能把局部描述提升为该域上的扰动界。若固定输入的更新映射还把某个完备状态域映回自身，并具有小于 1 的统一 Lipschitz 常数，压缩映射结论才适用。这个充分条件讨论迭代收敛，不保证语言建模损失下降，也不保证答案正确。

各轮雅可比的谱半径分别小于 1，仍不足以控制不同雅可比的乘积。一个精确的线性反例是：

```math
A=\begin{pmatrix}0&2\\0&0\end{pmatrix},
\qquad
B=\begin{pmatrix}0&0\\2&0\end{pmatrix},
\qquad
\rho(A)=\rho(B)=0.
```

```math
BA=\begin{pmatrix}0&0\\0&4\end{pmatrix},
\qquad
(BA)^n\binom01=4^n\binom01.
```

这里两个非正规矩阵交替作用会放大扰动；共享参数的非线性模型也会因状态不同而产生不同的局部雅可比。实际训练还涉及参数更新、数据变化、归一化和离散路由。因此，应把状态范数、跨轮变化、梯度范数与损失联合解释，再对照论文的初始化、残差缩放、学习率和深度调度干预。

## 全局损失分母与一次更新

设数据并行组有 W 个副本，一次更新累积 A 个微批次。n 的计数在整个更新窗口开始前确定；S 是本副本本微批次的有效目标交叉熵之和。这里采用普通 DDP 的平均梯度约定，且所有副本完成同一更新窗口。

```math
\begin{aligned}
N&=\sum_{j=1}^{W}\sum_{a=1}^{A}
\sum_{i,t}m^{(j,a)}_{i,t},\\
S_{j,a}&=\sum_{i,t}m^{(j,a)}_{i,t}\,
\ell^{(j,a)}_{i,t,K},\\
L^{\mathrm{backward}}_{j,a}&=\frac{W}{N}S_{j,a}.
\end{aligned}
```

```math
\frac1W\sum_{j=1}^{W}\sum_{a=1}^{A}
\nabla_\theta L^{\mathrm{backward}}_{j,a}
=\frac1N\sum_{j=1}^{W}\sum_{a=1}^{A}\nabla_\theta S_{j,a}.
```

式子中的 W 抵消了普通 DDP 对梯度的平均，因此最后得到整个更新窗口的 token 平均梯度，无需再除以微批次数。若反向图已经截断，同样的归约关系作用于相应截断梯度。

这里的分母直接决定训练在重视哪些预测。假设两个微批次分别只有 1 个和 3 个有效目标，先在批内取平均、再把两批等权平均，会让第一批唯一的目标占一半权重，另一批的三个目标各占六分之一；按全窗口 token 平均，四个目标则各占四分之一。因此，换一种“平均”就可能换掉训练目标。

实际计数不应把 TP 副本上的同一 token 重复算入。通信 hook、分片优化器或流水线若使用另一种缩放约定，需要相应重推 W/N 因子。[DDP 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html)

下面是末轮 CE、最后 b 轮反传的语义参考算法，采用全前缀 teacher forcing。它用于把公式对应到控制流；Huginn、Ouro 的退出目标、多轮监督、MoE 辅助项及各自分布式策略需要在此基础上逐项写出，不能直接认定为同一实现。

MaskedCESum 只对有效目标计算或累加损失，并使用稳定的 log-softmax／交叉熵实现。无效标签若采用 ignore_index，必须由损失内核处理；先用无效索引 gather，再乘零 mask，不能保证正确。

中间微批次使用 no_sync 时，forward 和 backward 都需位于该上下文中；最后一个微批次同步。全局 N 为零时全组跳过更新；某副本本地没有有效目标而全局 N 大于零时，仍须保留通信顺序和可微零损失，不能局部提前返回。[no_sync 的作用域](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync)

### 算法 P1 一次原生循环训练更新

**输入** 同一更新窗口的 A 个微批次、模型与优化状态、参数独立的深度分布 q、反传上限 k≥1、DP 副本数 W、梯度阈值 τ。

**输出** 更新后的参数与优化状态，或全组一致的未更新状态；记录执行深度、反传深度、损失分母和实际更新计数。

**P1 的计算范围。** 使用末轮 CE、完整前缀因果计算，循环块返回新状态；若有额外可微缓存，H 表示包含它们的完整状态。多轮损失、可学习 q、退出目标和 MoE 辅助损失需要另外加入，P1 不代表六篇论文各自的完整实现。

**随机性与截断。** prelude 重算复用 ξ_u，初始化使用 ξ_0。仅当 c>0 时截断；c=0 时保留初态的全部可微依赖。

**同步。** 普通 DDP 对梯度取平均；各副本使用相同微批次数、通信顺序和采样深度。TP／专家并行的额外 collective 需按具体框架展开。

**非有限值与故障。** 若 backward 能正常完成但产生 NaN／Inf，先完成既定通信，再全组跳过更新。运算异常或设备错误由训练器统一中止和恢复。

![算法 P1 一次原生循环训练更新](visualize:fde1_bGliZmlsZV9sbVhjaV9KSFdVSUR2dTFiS1otYm53_FileDrive_45288dfe06608191af94334ba8699511)

## 一个可手算的梯度核对

用三个标量参数分开代表 core、输入与初始化。它保留共享参数、重复注入和截断边界三种依赖，足以检查最容易漏掉的路径。

```math
\begin{aligned}
h_0&=\beta,\qquad u=\alpha,\\
h_{r+1}&=\theta h_r+u,\qquad
K=3,\qquad L=\tfrac12 h_3^2,\\
(\theta,\alpha,\beta)&=(\tfrac12,1,0),\qquad h_3=\tfrac74.
\end{aligned}
```

```math
\begin{aligned}
h_3&=\theta^3\beta+(\theta^2+\theta+1)\alpha,\\
\nabla_{(\theta,\alpha,\beta)}L
&=\left(\tfrac72,\tfrac{49}{16},\tfrac7{32}\right).
\end{aligned}
```

若只保留最后一轮，则 h₂=3/2 被视为常量，直接注入仍有梯度；若连注入也被 detach，输入参数的这条路径就消失。三种图的前向值相同，梯度不同。

```math
\begin{aligned}
\nabla L_{\text{末一轮}}
&=\left(\tfrac{21}{8},\tfrac74,0\right),\\
\nabla L_{\text{末一轮且注入截断}}
&=\left(\tfrac{21}{8},0,0\right).
\end{aligned}
```

检查实现时，先固定该例的状态、随机性及截断位置，再核对输出和三组梯度。重计算版本应与其对应的原计算图一致；完整反传、末轮截断和额外截断输入三个图不应被要求具有相同梯度。这里的数值是上述标量公式的结果，不是任何真实模型的实验成绩。

| 数学对象 | 工程表示 | 必须保持的语义 |
| --- | --- | --- |
| 输入、移位目标与监督 mask | 整数 [B,T+1]、整数 [B,T]、布尔 [B,T] | 目标只移位一次，packing 边界与任务定义一致 |
| H 与注入 u | [B,T,d] 或完整状态元组 | 截断 H 与是否保留 u 的图分别控制 |
| K、b 与 c | 整数和执行记录 | 执行深度、反传深度、截断边界分开记录 |
| S 与 N | FP32 损失和、整型全局计数 | 全窗口统一分母，DP 平均补偿仅应用一次 |
| 模型与优化状态 | 参数、优化器、调度、随机与数据状态 | 推理载入所需内容与原训练恢复所需内容分别保存 |

## 把训练配方还原成实际更新算子

完整梯度说明参数怎样影响目标，但实际更新还取决于截断和优化器。需要分别写出三个对象：论文定义的训练目标、backward 得到的梯度向量，以及 optimizer 最终施加的参数增量。下表用具体模型说明它们为何可能不同；详细证明、固定源码与最小反例见各专页。

| 基座 | 新补的数学结论 | 必须核对的执行事实 |
| --- | --- | --- |
| [Huginn](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790) | 截断更新可为非保守向量场；在固定参数、条件独立与有限二阶矩假设下，锁步深度保留公共梯度噪声 | 前 n 轮 no_grad，尾 k 轮带图；边界每次重新计算；深度按 microbatch/step 锁步而非整个累积窗口只抽一次 |
| [Done Right I](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8) | 共享参数先合并所有调用贡献，再进入同一套 AdamW moments；逐轮更新后平均一般不等价 | 初始化名 gaussian 实际为 ±3σ 截断正态；主体、embedding 与 head 的默认尺度分开；共享 core 不按轮次重新初始化 |
| [Loopies](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a) | microbatch 均值平均与全局 token 均值是两个目标；mask 去掉直接 CE，不自动切断输入表示梯度 | `calculate_per_token_loss` 在 schedule 与 finalize 两处改变归约；不从默认值反推论文历史运行 flag |
| [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 残差稳定性由真实有序 Jacobian 乘积控制；RMSNorm gain 改变方向放大与绝对写入，但不保证相对写入同比增加 | attention→MoE、层访问、残差系数和 gain 同时进入局部映射；探针固定参数、随机性与 top-k 分支 |
| [SMELT附录B／D](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 训练loss的同token差不等于同FLOPs差；sink降低须与value及非起点条件均值分解 | 全16格曲线、四尺度sink、固定KV的质量干预；参考算法与作者未公开probe分开 |
| [TaH2初始化对照](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811) | 正态score权重经sigmoid一般为logistic-normal；均值1/2不等于独立、均匀或校准 | 普通Base后训练的工程对照；可选uniform_init、默认关闭、真实dtype／cast和初始化随机对象明确 |

截断反传可以严格表示为“本次冻结边界”的局部导数。设当前参数为 theta₀，无梯度边界为 s_c(theta₀)，则每次前向重新定义一个局部损失：

```math
\widetilde L_{\theta_0}(\theta)
=\ell_\theta\bigl(F_\theta^{\circ b}(s_c(\theta_0)),y\bigr),\qquad
G(\theta_0)=\left.\nabla_\theta\widetilde L_{\theta_0}(\theta)\right|_{\theta=\theta_0}.
```

这里省略固定输入与后段注入的记号；真实导数仍保留它们。局部损失随着 theta₀ 变化，所以不能仅凭该表达式断言存在一个固定标量 L 使所有参数点都有 G=∇L。Huginn 专页用共享线性递推给出混合偏导不相等的反例。优化器的收敛解释必须与实际更新场对应。

反向图里可以有多次使用 θ 的位置，但优化器面对的是同一份参数。各次调用的反向贡献先累积到一起，才形成这次更新使用的梯度 G；AdamW 的 m、v 也随这一个合并后的梯度更新，而不是为每个循环轮次各建一套状态。

将第 r 次调用的贡献记为 g_r，学习率记为 eta，Done Right 等采用 AdamW 的路径可写为下式。Huginn 使用的 ELLISAdam 则按其专页的实际规则展开：

```math
G=\sum_rg_r,\qquad
(\theta',m',v')=\operatorname{AdamW}(\theta,m,v,G;\eta).
```

梯度裁剪、moment 平滑和逐坐标归一化都在合并后的向量上起作用；它们通常不与“对各轮分别操作后再平均”交换。是否除以循环数取决于损失本身是总和还是平均，不能为保持名义学习率而擅自增加一项缩放。

可以用三个检查区分这些对象：对同一小例分别计算完整梯度、截断梯度和优化器增量；用不等长 microbatch 检查归约权重；逐次核对模块调用采用的初始化与残差尺度。这些检查无需完整预训练，却能避免把错误的目标或尺度带入后训练。

## 配置值如何真正进入更新核

配置写了某个学习率，并不意味着实际更新就使用了它。需要沿“配置→调度器→外层 optimizer→本地 optimizer→更新核”追踪字段的传递。记 k 为参数组、r 为 rank、s 为更新编号，b_k 为该组规定的学习率倍率；下面比较本地更新实际使用的值与调度器预期值：

```math
\eta^{\rm used}_{s,k,r}
\overset{?}{=}
\eta^{\rm schedule}_{s}\,b_k.
```

| 案例 | 本次确认的固定源码事实 | 训练配方卡增加什么 |
| --- | --- | --- |
| [Huginn 的分片优化器](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790) | SimpleZeroRedundancyOptimizer 对非 0 本地 rank 的主 weight 组复制字典；train_step 重绑外层 lr，wrapper.step 未转发该超参数。fresh 的 embedding／norm 组与主 weight 分片因此可能接收不同调度值 | 分别记录外层和实际本地组的 lr、init_lr、衰减和计数；fresh 与 resume 分开验证。当前发布代码不能证明历史 800B 运行的完整学习率轨迹 |
| [Done Right Part I 的更新事务](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8) | 论文配方 loss_rescaling=False、scale=1；BF16 计算与 FP32 reduce-scatter；根 FSDP 对共享参数去重后归约全局范数并裁剪 | 保留归约、裁剪、step、清梯度的顺序；通用动态缩放分支另列。有效 token 加权训练目标与 microbatch 等权日志分别命名 |
| [Ouro 发布后的 loss 接口](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b) | Transformers 4.55.0 默认 ForCausalLMLoss 在内部移位 labels、用 FP32 CE，并允许显式归约分母；它仍是发布实现的加权-logits CE，不等于原文 Stage I | 交接时记录 labels 是否已经移位、选择的 logits 位点、有效 mask、分母和安装版本；重建原文目标时，CE 与熵使用相同监督范围 |

P1 中的抽象步骤由此变成具体检查：“设置学习率”要确认改动的字段确实被更新核读取；“对有效位置求平均”要确认分布式分母；“全局裁剪”要确认范数覆盖完整模型梯度。三篇专页列出对应文件、参数组分支、归约公式和小型检查。这些证据说明固定代码的行为，尚不能恢复历史训练轨迹。

## 数据打包同时决定可见域与监督域

预训练目标还需显式带上 attention 可见域 A 和目标位置 mask。设一个 packed row 在 shift 前为 u₀,…,u_S，每个目标 token 的监督标志为 m；有效预测位置只对齐一次。下面的条件概率使用模型实际的 A，不默认等于“整个前缀可见”。分母为 0 的样本按训练器约定跳过或作零贡献，不能直接代入除法。

```math
\begin{aligned}
x_t&=u_t,&y_t&=u_{t+1},&\mu_t&=m_{t+1},\\
\mathcal L(\theta;u,A,\mu)
&=-\frac{\sum_{t=0}^{S-1}\mu_t\log p_\theta(u_{t+1}\mid u_{\le t};A)}
{\sum_{t=0}^{S-1}\mu_t}.
\end{aligned}
```

Done Right 发布路径提供了具体反例：Part I 的 `multi_segments=True` 按输入 BOS 构造 attention segment，却不删除 EOS→下一 BOS 的监督；best-fit 长文续块没有自动补 BOS，和另一短文拼进同一 bin 时甚至可能共享同一 attention segment。因此“分段 attention”不等于“跨原始文档完全隔离”，也不等于“跨边界 CE 已屏蔽”。官方示例启用 bestfit，通用默认仍为 simple；当前发布 loader 与历史训练顺序的区别继续保留。[完整调用链、源码与三个已执行小例](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8)

工程上先由 feature builder 生成 token 与逐 token 监督标志，再由 packer 切块、拼接并统一 shift，最后从实际输入边界构造 A。每个抽查 row 同时保存 input、target、target mask 与 segment，分别检查“预测时能看到什么”和“哪次预测计入 loss”。接 SFT 时监督标志跟随被预测的 assistant token，而非当前输入 token；更换边界监督、重加 BOS 或改 segment 元数据都会改变训练函数，应登记为新的配方。

## 七篇材料的数据配方与执行链

数据配方只有追到实际输入序列才算明确：来源、权重单位、模板、过滤、打包、监督和恢复共同决定训练对象。下表按材料列出这些环节及其实现边界。

| 材料 | 原始配方与本轮落实 | 实现依据或仍缺的条件 |
| --- | --- | --- |
| [Huginn](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790#a8b80346-6e18-4634-a673-38c19d445e78) | Fig.4 全11类比例；附录96来源与当前113项配置；文件倍率、舍入及打包保留率；D1 | 原文与固定脚本已核；cooldown脚本入口不能直接冒充800B历史数据launcher |
| [Ouro](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b) | Tables3–5的语料池、Used和两个阶段全比例；Stage4新监督与回放30%／10%／60% | 原文阶段来源预算完整保留；公开HF实现尚未提供历史采样器、formatter及源游标恢复 |
| [Done Right I](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8#ba6de3ee-0ee9-4b39-8050-3c0ea2cab73f) | 42项正权重合计1；token预算结转、有限样本长度下的累计误差界；D2 | 当前release真实refill与state方法已核；旧论文loader顺序差异保留 |
| [Loopie](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a) | Stage2七类预算与SPT响应mask已展开；Stage1的3T与570B×4仍并列登记 | 不把未知论文launcher推定为Done Right的quota实现；token预算冲突继续待原日志解释 |
| [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 同cell配对使用相同token序列，stable与六个fresh-token分叉分别记账 | 原文未公开完整训练语料与训练栈；不能补成其他论文的数据配方 |
| [LoopMoE](https://chatgpt.com/space/page_13f174d58dd08191b37781d5c4da6099) | Dolma3Mix；3B共同200B子集、相同文档顺序及packing，9B为300B | 原论文§4.1已核；实现／权重及精确子集ID、source-order和packing工件仍待取得 |
| [TaH2](https://chatgpt.com/space/page_b39954c20a348191b95d45a766f24ef2#9f477f57-3017-463b-858f-093b6c18790b) | 原工具轨迹按episode展开；完整历史输入、当前段监督；offset边界与长度策略；D3 | 原函数体＋替身tokenizer已核；真实模板、BPE、工具环境和GPU训练另验收 |

数据配比有三个层次：文件被选中的倍率、输入 token 的预算，以及有效目标的权重。D1 分析文件选择的期望与方差；D2 说明预算结余如何使累计误差有界；D3 区分重复输入与一次监督。比较训练收益前，应保存各来源的实际输入数、目标数、重复数和恢复状态。

# 六篇模型的训练配方

| 材料 | 已落实的专有训练链路与推导 | 论文与实现的关键边界 |
| --- | --- | --- |
| [Huginn](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790) | 初始化与 RMSNorm 导数、随机深度精确矩、末 8 轮梯度遗漏项与条件界、锁步采样、ELLISAdam；算法 HG1 给出更新顺序 | 配置 32 与均值 33、参照深度、最终 LR、首个边界跳过 step、running_init、HF 梯度接口分别核对 |
| [Done Right I](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8) | Part I 完整反传、注入参数导数与 H/L 联合 Jacobian、router 分段梯度、MoE aux 归一化；算法 D1 追到微批次目标 | Part II 不混入；论文 LR sweep、旧 loader、release 配方与恢复状态分开 |
| [Loopies](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a) | 共享调用梯度、SmallInit 矩、恒定 LR 与两阶段 PT、SPT 有效标签分母、GSPO 梯度与动态筛选算法 | 匹配实测 step 时间；token 预算歧义、SPT／RL 专用训练脚本缺口保留 |
| [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 残差 1/r 的条件界、三预算约束、WSD stable 与六分叉算法、规模律唯一最优解及配对 bootstrap | 算术 F 与耗时分开；完整训练栈、部分初始化／router 配置未公开；参考算法明确来源范围 |
| [LoopMoE](https://chatgpt.com/space/page_13f174d58dd08191b37781d5c4da6099) | IterAdaLN 完整 Jacobian 与零初始化首步梯度、专家并集期望、Dolma3Mix、固定四轮与 AdamW／cosine | 完整 loss、detach／recompute、router、并行与权重未获官方实现；参考前向显式接收缺失契约 |
| [Ouro](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b) | 退出分布期望 CE−熵的最优形式与 hazard 梯度、Stage II BCE 导数、7.7T 与 Thinking；OU1 给出两阶段目标 | HF labels 使用另一种目标；OU2 忠实记录先执行全循环再选输出；原始训练栈与物理早退仍有缺项 |

P1 给出通用目标和同步规则，模型专页则解释具体训练器。Huginn 的 HG1 包括首个累积边界跳过 step 及 running_init；Done Right 的 D1 区分全局有效标签分母和 auxiliary 平均；Loopie 的 SPT 用 response mask 确定监督量。迁移时先选定目标，再检查梯度、随机状态与归约是否一致。

# 两个完整交接案例

## Ouro 的计算能力与退出机制

Ouro 的两阶段训练说明了交接时需要保留什么。Stage I 联合训练语言模型与退出门控，目标是逐轮 CE 的期望减去退出分布熵。Stage II 冻结 LM，取相邻两轮 CE 改善的非负部分 I，并停止这项改善量的梯度，用它构造 gate 的训练目标。

Stage II 使用贪心代理 w=σ(50(I−0.005))。由于 I≥0，w 的下界约为 0.438，因此“无改善时目标约为零”并不是该公式的含义。[hazard／PDF／CDF、两阶段目标、7.7T 与 Thinking 链路，以及官方 HF forward 的对照](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b)见专页。

训练目标明确后，还要看各轮状态在生成时怎样通过 KV 缓存复用。Ouro 分别研究 prefill 与 decode：prefill 需要保留轮次对应关系，不能直接套用 decode 的最后轮或平均复用方案。这把预训练形成的状态、部署时的缓存规则和实际成本联系起来。[英文第 18 页](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal)

## 同一组退出权重对应三种不同目标

以 Ouro 为具体核对案例。令 aᵣ 为单个位置的退出概率，zᵣ 为第 r 轮 logits，y 为目标 token，且 aᵣ≥0、Σᵣaᵣ=1。逐轮交叉熵的期望、logits 加权后交叉熵、预测概率混合后的负对数似然分别为：

```math
\begin{aligned}
L_{\mathrm{step}}&=\sum_{r=1}^{R}a_r\,\operatorname{CE}(z_r,y),\\
L_{\mathrm{logit}}&=\operatorname{CE}\!\left(\sum_{r=1}^{R}a_r z_r,y\right),\\
L_{\mathrm{prob}}&=-\log\!\left[\sum_{r=1}^{R}a_r\,\operatorname{softmax}(z_r)_y\right].
\end{aligned}
```

Ouro v6 式 (4) 使用 L_step 再减去退出分布的熵正则。已核 HF 实现的 `OuroForCausalLM.forward(labels=…)` 则先调用 `compute_expected_logits`，对应 L_logit，且该分支未实现论文的熵项。两者不能互相代替；调用一个可返回 loss 的模型接口，并不等于复现论文训练目标。[论文 §3.3](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal) · [固定版本模型代码](https://huggingface.co/ByteDance/Ouro-1.4B/blob/574fa66cb8bf5abdc979642d01cf2b79b16bfab1/modeling_ouro.py)

这一区别可直接由凸性核对：log-sum-exp 的凸性给出 L_logit≤L_step，−log 的凸性给出 L_prob≤L_step；一般并不相等，也不能由此推出两种模型训练后的性能优劣。例如 z₁=(3,0)、z₂=(0,1)、a=(0.3,0.7)、y=第一类时，三者依次为 0.933859、0.598139、0.746482。这是教学数值核对，用于说明目标和梯度已改变。

## Loopie 的 Base 到 Thinking

Loopies 给出预训练、高质量数据阶段、Base、SPT、数学／代码 RL 和 Thinking 的链路。该配方的高质量阶段仍使用恒定学习率；SPT 与 SFT 使用同一目标 token CE，规模、采样和监督位置需要分别记录。[专页已补初始化、数据配比、优化器、SPT loss mask 与预算、GSPO 序列比率、DAPO 裁剪与动态过滤](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a)。

# 源码阅读入口

| 案例 | 已固定版本的源码入口 | 实际调用链／缺口 |
| --- | --- | --- |
| [Huginn](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790) | [recurrent-pretraining@1ea7220](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e) | train_step→RecurrentGPT.forward→iterate_forward→core_block_forward；优化器、配置、通信与 HF 单列 |
| [Done Right I](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8) | [xllm-loop@755c7b3](https://github.com/ifm-ai/xllm-loop/tree/755c7b3a91081fe0f83500eabd53bd2490b8b745) | train_paper_part1→recipes／training_config→get_model_cls→forward_backward_window；native loader／evaluator |
| [Loopies](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a) | [Megatron@cb2daa2](https://github.com/zitian-gao/Megatron-LM/tree/cb2daa267d3c5458e75751b3f73a17672fe7a55a) | pretrain_gpt→GPTModel→TransformerBlock→TransformerLayer→router；另核 HF 与 vLLM |
| [Ouro](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b) | [HF 1.4B@574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1) | OuroForCausalLM→OuroModel→OuroDecoderLayer／Attention／Cache；发布 forward 不是完整原始训练链 |
| [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 原论文和附录 A | 完整架构／训练栈明确不能发布；页面算法标明为论文语义重建 |
| [LoopMoE](https://chatgpt.com/space/page_13f174d58dd08191b37781d5c4da6099) | 正式论文、arXiv 与作者主页 | 尚未核实归属明确的官方实现；精确函数、配置与权重不作猜测 |

各专页的源码入口均指向固定 commit 或配置 revision，公式与张量对应到具体函数。模型卡能示范加载，可训练代码能执行更新，完整状态才能恢复原训练轨迹，三者需要分别核对。LaTeX 工程用于论文文本和公式核对，模型实现使用上表来源。[材料说明](https://chatgpt.com/library?file_id=libfile_ff2d41241da481919b240ff4f2c8d8ef&account_id=personal)

# 预算与证据登记

Loopie 按实测 optimizer-step 时间匹配训练预算。SMELT 则近似控制算术 FLOPs/token、非 embedding 参数和 KV，并用各配置在平均文档长度下的算术 F 拟合规模律。原文虽称 F 为 measured，却未披露计数工具，因此这里不将其视为 GPU profiler 测量。具体匹配误差、分叉来源与预算锚点见 [Loopie](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a)和 [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921)。

Loopies v3 的 Stage 1 总述与细述存在 token 预算口径待解释：第 13 页写 3T，第 14 页写 570B unique tokens × 4 epochs。正式重建配方时需对照代码与作者说明，暂不将其中任一数字固定成执行配置。

## 训练预算、调度与验证分母分开记

TaH2 的 TPP=2 指训练 token 预算与所采用参数量口径之比；不是张量并行度。训练输入 tokens、监督目标 tokens、共享骨干的重复调用数和唯一文本量分别记录；epoch 重复不能算成新的独立数据。[原文 §5.1／A.1 与纠正后的配方表](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb)

| 对象 | 固定实现中如何得到 | 接回哪份证据 |
| --- | --- | --- |
| 训练预算 TPP | 训练 tokens／模型参数量；另报完整训练 FLOPs | 原文 4B／8B 的约 8B／16B tokens，与 release 实际 token 数分开 |
| 学习率 horizon | sampler 每 epoch 更新窗口数 × 向上取整的 epochs | [参数组与调度课](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647)：max_steps 提前退出不重建日程 |
| 训练目标分母 | 当前更新窗口中实际监督位置，按实际 DP 归约 | 只补偿一次梯度平均；不把 TP 副本重复计数 |
| eval 目标分母 | helper 先统计整份处理后数据；执行端另截断 batch 数 | [T12](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c#b445b124-6da5-430c-896a-5b1c0faf4646)：需要核对实际执行的移位监督数 |

验证日志同样需要明确分母。对 TaH2 的固定 eval 路径，下面的式子比较日志 NLL 与实际执行集合的平均损失 L_E。其中 U_D 是 helper 在整份数据上统计的未移位标签数，V_E 是执行集合的移位目标数。等式限定实数算术，用来解释计数流程怎样影响日志数值，不用于追溯判定原论文的验证曲线。

```math
L_{\rm raw}=\frac{V_{\mathcal E}}{U_D}L_{\mathcal E},\qquad U_D>0,\;V_{\mathcal E}>0.
```

代码分组、学习率更新顺序和验证估计量共同构成可重建的配方；本轮八组 CPU 替身检查与源码快照已附于 T12，完整模型与并行执行仍须另行验收。

# 从原生预训练交接到 TaH2 式适配

交接对象包含权重、状态初始化、默认深度、读出、归一化、位置与缓存规则，以及实际梯度路径。TaH2 的共同 Qwen 起点提供普通基座适配案例；原生循环权重的交接则逐模型保留这些规则。

## 单轮是否退回原模型有可检查的条件

设普通模型为 f_θ，循环包装的单轮模型为 f̃_(θ,ψ)。若第一轮直接使用同一 embedding 和 backbone，attention 仅包含同一因果历史，位置、归一化、读出相同，无 dropout 或随机状态一致，updater 尚未调用且混合质量为 1，则精确算术下：

```math
\widetilde f_{\theta,\psi}(x;M=1)=f_\theta(x).
```

这是逐层计算相同的归纳结论。不同 kernel、dtype 和并行归约只支持数值容差内的比较；若模板、权重阶段或 mask 不同，前提已不满足。TaH2 Standard 配方的单轮组件见固定 [`iter_decider.py`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py) 与 [`input_updater.py`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/input_updater.py)。

Huginn 等原生循环模型的 R=1 是该模型自身较短的计算路径，不自动对应另一个普通预训练 Transformer。Ouro、Loopie、SMELT 和 LoopMoE 的深度改变也依各自训练分布解释。

## 后训练开销与部署开销分别交接

```math
C_{\rm total}=C_{\rm pre}+C_{\rm post}+N_{\rm served}\,\mathbb E[C_{\rm infer}].
```

这是统一计量口径下的生命周期成本分解。采用已发布基座时，新增预算主要在后两项；方法比较仍报告实际后训练前向、反向与无梯度 lookahead。相同数据和 epoch 不推出相同后训练 FLOPs。TaH2 部署不执行训练 lookahead；其两份成本账本见 [TaH2 工程](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133) 和 [性能评测](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c)。

# 第一份训练配方卡

先对照 Huginn 与 Ouro 的数据、目标、深度、反传、初始化、优化、退出和缓存，独立复述一次更新。重点说明论文目标、发布接口与实际梯度为何可能不同。随后用 Done Right I 的受控配方区分因素，再加入三个 MoE 案例。

```codex-prompt
请沿本页推进原生循环预训练学习。先定义移位目标、监督 mask 和深度分布，推导共享参数、可训练输入注入、初始化与读出的完整梯度。用标量例子核对完整反传、末 b 轮截断与输入被额外 detach 的差别，再按算法 P1 追踪一次更新的全局分母、随机性、同步和优化状态。随后对 Huginn 与 Ouro 核对其专有配方、退出目标和公开源码。公式采用原生数学块；算法采用论文式行号与真实操作公式。教学计算与真实模型验收分别报告。
```
