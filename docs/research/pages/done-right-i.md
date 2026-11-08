[基座矩阵与七篇论文](https://chatgpt.com/space/page_06fdf6bccaac8191b412533a0628a093) · [架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8) · [预训练](https://chatgpt.com/space/page_4f0508753a6081919a82b0c89213fd6c) · [后训练](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)

# 研究问题和阅读路线

Done Right I 用受控实验拆解 Ouro 与 Huginn 的差别：循环放在哪里、每轮如何写入输入，以及循环状态如何初始化和组织。它包含一组从头训练的 dense／MoE 配方和对照模型，并非单一架构。阅读时先用第1至3节连接模型、递推和一次训练更新，再看第4至5节的消融与路由结果，最后读第6至8节的执行链、发布差异和后训练接口。

资料版本为英文论文 2610.23036v1（27页）和官方 xllm-loop 提交 755c7b3a91081fe0f83500eabd53bd2490b8b745。结论限于论文的模型、训练预算和评测协议；多种子与更大尺度仍是 §5 的后续工作。[原论文](https://chatgpt.com/library?file_id=libfile_1060bf6ca4348191bb4b729c27a0bf69&account_id=personal)；[固定版本官方发布说明](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/README.md)

## 1 模型配置和比较预算

稠密模型的主比较使用28个物理块和112次逻辑块执行。730M是不含embedding的物理参数尺度；约2.9B表示展开后的参数应用量，而不是2.9B份独立参数。

MoE配置在全部28个物理层使用路由FFN：25个routed experts、top-2、expert中间宽度2432。按Table 6的non-embedding口径，循环模型驻留8.0B参数，每次物理遍历激活793.9M，展开后约有3.2B次active parameter applications；112层独立MoE对照则驻留32.0B参数、激活3.2B。[原论文 §3.1、§4.1、Table 6，pp.5、15、24](https://arxiv.org/pdf/2610.23036v1#page=5)

| 模型/变体 | Prelude / 共享核心 / Coda | 状态与写入 | 物理块 / 逻辑块 |
| --- | --- | --- | --- |
| Ouro | 0 / 28 块 × 4 轮 / 0 | 初始状态为 token embedding；无写入 | 28 / 112 |
| Middle | 8 / 12 块 × 8 轮 / 8 | 初始状态为 prelude 输出；无写入 | 28 / 112 |
| Middle + prelude injection | 同 Middle | 初始状态为 prelude 输出；每轮对角写入 | 28 / 112 |
| Huginn（本文配方） | 同 Middle | 独立随机初态；每轮对角写入 | 28 / 112 |
| Shared H/L | 同 Middle | 两个状态、共享同一个 12 块核心；2 个高层周期，每周期 3 次低层更新后 1 次高层更新 | 28 / 112 |
| Feedforward D28 / D112 | 无深度共享 | 常规逐层计算 | 28 / 28；112 / 112 |

各变体共用 PreNorm decoder block：宽度 1536，24 个 Q heads / 6 个 KV heads，每头 64；RoPE base 10⁶；dense FFN 是宽度 4352 的 SwiGLU。一次 block 先作 attention residual update，再作 FFN/MoE residual update，不能把循环跨越的整组 block 简化成“重复一次 FFN”。这些设置与 recipes.py 的 _model_fields 对应。[原论文 §3.1](https://arxiv.org/pdf/2610.23036v1#page=5)；[recipes.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/recipes.py)

### 执行预算的精确记账

把一次前向写成物理层索引序列 σ。先只对 decoder 堆栈记账：Θ_all 是该堆栈所有驻留参数张量，包含未被当前 token 选中的专家；Aᵥ(x) 是给定 token 在第 v 次调用激活的参数集合。张量身份必须去重，不能把相同数值的独立参数合并。embedding、head 与 injection 参数另列。计算量沿实际调用序列累加，还取决于序列长度与路由。

```math
\begin{aligned}
\sigma&=(\ell_1,\ldots,\ell_D),\\
P_{\rm resident,stack}&=\sum_{p\in\Theta_{\rm all}}\operatorname{numel}(p),\\
P_{\rm visited,stack}(x)&=\sum_{p\in\bigcup_{v=1}^{D}\mathcal A_v(x)}\operatorname{numel}(p),\\
P_{\rm applications,stack}(x)&=\sum_{v=1}^{D}\sum_{p\in\mathcal A_v(x)}\operatorname{numel}(p),\\
C_{\rm forward}(s)&=\sum_{v=1}^{D}C_{\ell_v}(s)+C_{\rm injection}(s)+C_{\rm head}(s).
\end{aligned}
```

因此 28×4 与 8+12×8+8 都有 D=112，但第一式中的参数共享方式不同；D 相等本身不是逐算子 FLOPs 完全相等的证明。尤其加入 diagonal injection 后，多出一个 d×d 投影和两组 d 维参数。以 d=1536 代入，精确增加 2,362,368 个参数；固定源码按前向加反向每个 multiply-add 计 6 FLOPs 的代理口径，8 次 injection 另计每 token 113,246,208 FLOPs。该算术结果来自代码公式，不是硬件时间测量。

```math
\begin{aligned}
\Delta P_{\rm inj}&=d^2+2d,\\
\Delta C_{\rm inj,train}/\text{token}&=6R d^2.
\end{aligned}
```

Q1 的受控结论是：在论文给定的训练预算与评测协议下，改变共享跨度及两端独立层的配置产生了任务分数差。它没有单独识别“循环在中间”这个位置变量与“为输入和输出保留不共享层”这两个机制的贡献。

来源：[Table 1、§3.1–3.3](https://arxiv.org/pdf/2610.23036v1#page=5)；[`build_loop_execution_schedule`、`_diagonal_input_injection_parameter_count`、`_input_injection_tflops_per_token`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)。以上计数为源码公式的直接展开。

## 2. 完整递推：共享参数不等于共享状态

设输入 embedding 为 X，prelude 输出 E，循环状态为 Z。P、R、C 分别是 prelude、共享 core、coda；Z 与 E 的形状均为 batch × token × hidden。词表读出接在 coda 及最终归一化之后。每轮复用 R 的同一参数对象，同时保存自己的激活供反向传播。

```math
\begin{aligned}
E&=P_\theta(X),\\
Z_0&=\phi(E,\xi),\\
\widetilde Z_r&=W_\theta(Z_r,E),\\
Z_{r+1}&=R_\theta(\widetilde Z_r),\\
p_\theta(\cdot\mid x_{\le t})
&=\operatorname{softmax}\!\left[
V_\theta\,\operatorname{Norm}\!\left(C_\theta(Z_T)_t\right)
\right].
\end{aligned}
```

循环状态承接上一轮的计算，输入表示 E 则来自题目、示例或规格等原始输入。diagonal injection 在每次进入共享 core 之前决定两件事：上一轮状态 Z 留下多少，以及输入表示重新写入多少。公式中的 α 逐通道控制前者，δ 逐通道控制后者，W_in 则是可学习的输入投影。这种写入比简单 Z+E 多了可学习系数；它采用逐通道相加，也与原始 Huginn 的任意 concat adapter 不同：

```math
\begin{aligned}
W_\theta(Z,E)&=\alpha\odot Z+\delta\odot(W_{\rm in}E),\\
\delta&=\operatorname{softplus}(b_\delta),\\
\alpha&=\exp\!\left[-\delta\odot\exp(a)\right].
\end{aligned}
```

官方 InputInjection 把这些量实现为 A_log、dt_bias 和 input_adapter。初始化 dt=1、exp(A_log)=1、W_in=I，所以输入投影最初保持原表示，而旧状态的保留系数已经是 exp(−1)。三个参数都交给 AdamW，对角系数也参与 weight decay。

0<α<1 说明每个通道在写入这一步都会衰减旧状态；随后，衰减后的状态还要经过共享 core。若 core 放大了同一个扰动，完整一轮仍可能放大。因此，下面的 Jacobian 把两步按顺序相乘：右侧 diag(α) 描述写入，左侧 core Jacobian 描述写入之后的计算。要证明整轮收缩，需要约束这个乘积的范数；仅有 α 的取值范围尚不足以保证任意深度收敛或稳定。[原论文 §3.4.2，pp.8–10](https://arxiv.org/pdf/2610.23036v1#page=8)；[looped.py / InputInjection](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)

```math
J_{r}=
\frac{\partial R_\theta(\widetilde Z_r)}{\partial\widetilde Z_r}
\,\operatorname{Diag}(\alpha).
```

共享 H/L 的状态增加与权重增加是不同操作。令 R̄ 表示“先写入 E，再过同一个 R”；H 初值为 E，L 初值按每 token 的 N(0,I/d) 抽样。论文固定 H=2 个周期、L=3 次低层更新，完整依赖为：

```math
\begin{aligned}
\bar R_\theta(S;E)&=R_\theta(W_\theta(S,E)),\\
Z_H^0&=E,\qquad Z_L^{0,0}=\xi,\\
\xi&\sim\mathcal N(0,I/d).
\end{aligned}
```

```math
\begin{aligned}
Z_L^{i,j+1}&=\bar R_\theta(Z_L^{i,j}+Z_H^i;E),
\quad j=0,1,2,\\
Z_H^{i+1}&=\bar R_\theta(Z_H^i+Z_L^{i,3};E),\\
Z_L^{i+1,0}&=Z_L^{i,3},\quad i=0,1.
\end{aligned}
```

最后将第二次 H 更新输出送入 coda。L 在两个周期之间延续，而不是重新采样；跨状态相加发生在 injection 之前。共 6 次 L 更新与 2 次 H 更新，仍执行 8 次共享 core。原文只报告这一 shared-module H/L 消融，不能把它的负结果推广为所有 HRM 式独立 H/L 模块都无效。[原论文 §3.4.3、Fig.7、Table 4，pp.11–13](https://arxiv.org/pdf/2610.23036v1#page=11)

### 注入既改变状态传播，也提供直接的输入梯度

以下固定随机初态和一个光滑路由区域，将整个 batch × token × hidden 张量向量化。Dₐ 与 Dδ 表示在各 token 上重复的对角矩阵，W 表示同样广播的输入投影；Jᴿ 是 core 对其输入的 Jacobian。除了已有的状态 Jacobian，还需保留每轮直接读入 E 的通路。

```math
\begin{aligned}
A_r&=J_r^{R}D_{\alpha},\qquad B_r=J_r^{R}D_{\delta}W,\\
\frac{\partial Z_{r+1}}{\partial E}
&=A_r\frac{\partial Z_r}{\partial E}+B_r.
\end{aligned}
```

若某个包含轨迹的凸不变区域内，所有状态 Jacobian 的算子范数不超过 q<1，输入通路不超过 b，则链式法则给出下面的条件上界。它解释了为什么“初态影响衰减”与“持续依赖输入”可以同时成立；论文并未估计满足该假设的 q。

```math
\left\|\frac{\partial Z_R}{\partial E}\right\|
\le q^R\left\|\frac{\partial Z_0}{\partial E}\right\|
+b\frac{1-q^R}{1-q}.
```

对一个通道，令 a=exp(A_log)、δ=softplus(bδ)、α=exp(−δa)、u=(WE)。源码中的可学习衰减并非常数；反向时下列项必须进入共享参数的每次使用所贡献的梯度。

```math
\begin{aligned}
\frac{\partial W_\theta(z,E)_j}{\partial b_{\delta,j}}
&=\sigma(b_{\delta,j})\bigl(u_j-a_j\alpha_j z_j\bigr),\\
\frac{\partial W_\theta(z,E)_j}{\partial A_{\log,j}}
&=-\delta_j a_j\alpha_j z_j,\\
\frac{\partial W_\theta(z,E)_j}{\partial W_{jq}}
&=\delta_j E_q.
\end{aligned}
```

### H/L 的稳定性必须分析联合状态

令 G(s;E)=R(W(s,E))。以 (H,L) 为联合状态，一次 L 更新保留 H，一次 H 更新保留 L。下面的 A、B 是在各自实际输入处求得的 G 的 Jacobian，不能先把它们替换成同一个数值矩阵。

```math
\begin{aligned}
J_{L,j}&=\begin{pmatrix}I&0\\A_j&A_j\end{pmatrix},\\
J_H&=\begin{pmatrix}B&B\\0&I\end{pmatrix},\\
J_{\rm cycle}&=J_HJ_{L,3}J_{L,2}J_{L,1}.
\end{aligned}
```

即使 G 对单个输入是收缩，两状态间的相加仍可能放大联合扰动。一个由上述 schedule 得出的标量反例是 G(s)=a·s、a=0.6、每周期三次 L 更新：

```math
J_{\rm cycle}
=\begin{pmatrix}
a+a^2+a^3+a^4&a^4\\
a+a^2+a^3&a^3
\end{pmatrix},\qquad
\rho(J_{\rm cycle})\approx1.4310>1.
```

这是数学反例，不是训练模型的实测特征。它说明 shared H/L 消融应同时记录两个状态的扰动增长与更新范数；单看注入 α<1 或单次 core 的范数不能推出整个 H/L 系统稳定。

来源：[§3.4.2–3.4.3](https://arxiv.org/pdf/2610.23036v1#page=9)；[`InputInjection._decayed_state_and_update` 与 `Huginn._run_recurrences`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)。上述 Jacobian、界与反例均由已核实计算图推导。

## 3 从数据到一次参数更新

Part I 的 forward 没有截断循环梯度：Huginn._run_recurrences 和 LoopedTransformer.forward 对所有调用保留计算图，直到末轮 coda / LM head 形成 token loss。文件中的 DepthControlledHuginn 属于 Part II，不能将它的随机深度或 no_grad warmup 机制套到 Part I。共享参数的梯度由所有调用位置累积；改变循环次数既改变前向分布，也改变梯度路径，不能只改一个推理参数就假定训练分布保持不变。[looped.py / 两类 Part I forward](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)

MoE 路由实际实现为 float32 logits → sigmoid score → top-2 选择 → 对入选的原始 score 重新归一化。计算负载均衡项时又把所有专家 sigmoid score 归一化。令 N 是本次路由调用的 token 数，E_num=25；f 是 top-2 选择频率，p̄ 是全专家归一化分数均值，则代码中的 dot 辅助项为：

```math
\begin{aligned}
s_{t,e}&=\sigma(w_e^\top h_t),\\
S_t&=\operatorname{Top2}(s_t),\\
g_{t,e}&=\frac{s_{t,e}}{\sum_{j\in S_t}s_{t,j}},
\quad e\in S_t.
\end{aligned}
```

```math
\begin{aligned}
f_e&=\frac{1}{2N}\sum_{t=1}^{N}\mathbf 1[e\in S_t],\\
\bar p_e&=\frac1N\sum_{t=1}^{N}
\frac{s_{t,e}}{\sum_j s_{t,j}},\\
L_{\rm bal}&=E_{\rm num}\sum_e f_e\bar p_e.
\end{aligned}
```

训练器把每次实际 MoE 调用的辅助项求和，再除以本次 forward 的实际 MoE 调用数；不是只除以 28 个物理层。Part I MoE 系数为 0.001，z-loss 系数为 0，没有 router bias、shared expert。梯度累积窗口内，语言模型 loss 按有效目标 token 总数归一化；aux 按 microbatch 数与执行层数归一化。[router.py / TopKRouting.forward、calc_probs](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/moe/router.py)；[train.py / forward_backward_window](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py)

```math
L=L_{\rm token}
+10^{-3}\frac{1}{N_{\rm MoE\ calls}}\sum_{c=1}^{N_{\rm MoE\ calls}}L_{\rm bal}^{(c)}.
```

论文在 TxT360 上使用长 8192、全局 512 序列，即每步 4,194,304 tokens；扫描学习率 4×10⁻⁴、6×10⁻⁴、8×10⁻⁴ 并报告最佳结果。当前发布配方固定 6×10⁻⁴、AdamW β=(0.9,0.98)、ε=10⁻⁸、weight decay=0.1、global grad clip=1、200-step warmup、cosine 至峰值的 0.01，seed=1、BF16。336b target 停在 80,000 步，实际 335,544,320,000 tokens；500b target 是 119,210 步、500,002,979,840 tokens。两者都使用 119,210 步 horizon 的同一 schedule，336b 不是重新缩短的 cosine schedule。[原论文 §3.2、Appendix A](https://arxiv.org/pdf/2610.23036v1#page=6)；[recipes.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/recipes.py)

数据先经过混合、tokenize和packing。发布的[data.md](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/data.md)列出42个有顺序、有权重的来源，其中一部分prepared source未公开。`common-base.example.json`使用bestfit packing和512-sequence buffer；tokenizer的来源revision固定，`materialize-tokenizer.py`去掉自动BOS，再由loader给文档添加BOS/EOS。

文档边界和监督目标需分开处理。`multi_segments=True`按输入中的BOS位置分段；长文续块没有补BOS时，不能保证按原始文档隔离。完整文档之间的EOS→BOS预测目标也不会自动删除，具体规则与小例见下节。

当前loader在线tokenize、mix、pack的顺序与论文旧版不同，因此公开代码不能直接逐步重放原训练的样本序列。

### 从文档到监督对齐：attention 分段与目标 mask 是两套契约

以下针对固定 revision 的发布数据路径。Part I recipes 锁定 multi_segments=True；官方 common-base.example.json 选择 bestfit、buffer_size=512，但 packing_type 属于调用方配置，通用默认是 simple，不能把所有运行都写成 bestfit。train_paper_part1.py 把 base 交给 training_config，再由 train.main→build_data_loader→MultiSourceDataLoader 选择 packer。发布说明已提示新 loader 与历史训练顺序不同，因此下列核验不反推原论文每次训练实际出现了哪些文档边界。[发布示例](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/common-base.example.json)；[配置归属](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/training.py)；[数据构建入口](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/build.py)

| 执行阶段 | 固定源码行为 | 对训练目标的含义 |
| --- | --- | --- |
| DefaultFeatureBuilder.build | 调 tokenizer.encode(text,bos=True,eos=True)，target_mask=None | 普通预训练文本默认监督所有可形成的下一 token 目标 |
| BestFitPacker.pack | 每个 bin 容量为 S+1；长文按步长 S 切成至多 S+1 个 token 的片段；片段按长度降序 best-fit | 相邻长文片段重叠一个 token，以保留相邻预测对；续块不自动补 BOS，也不传递旧块 hidden |
| _flatten_chunks | 连接片段及 target_mask；跨片段首 token 的屏蔽代码仍是注释 TODO | 打包边界不会自动从监督中删除 |
| pack 后的 x/y/mask | x 取前一位、y 取后一位，mask 同时取后一位；padding 的目标置 −100 | mask 必须与被预测 token 对齐，不能与当前输入 token 的标签混用 |
| _begin_forward→get_segment_arg | 仅根据输入中的 BOS 位置产生 attention segment；source_infos 不进入此接口 | attention 可见域与 CE 监督域分别确定，前者不能自动修正后者 |

令打包后未填充序列为 u₀,…,uₙ₋₁，逐 token 监督标志为 m；预训练未提供 mask 时令 m 全为 1。对 t<n−1，packer 的实际对齐如下；padding 另令 μ=0。这里 −100 是固定源码的 ignore index，token-loss 分母继续使用有效 μ 的总数。

```math
x_t=u_t,\qquad \mu_t=m_{t+1},\qquad
y_t=\begin{cases}
u_{t+1},&\mu_t=1,\\
-100,&\mu_t=0.
\end{cases}
```

在 context parallel size=1、multi_segments=True 下，等价的因果可见域可写为下面的 segment 条件；每个 packed row 起点也形成一个起始 segment。它没有规定 μ 的值，故“禁止跨文档 attention”与“取消跨文档预测目标”不能互相替代。

```math
s_t=\sum_{j=0}^{t}\mathbf1[u_j=\mathrm{BOS}],
\qquad
A_{t,j}=\mathbf1[j\le t]\,\mathbf1[s_j=s_t].
```

**完整文档边界仍保留 EOS→BOS 目标。** 例如 S=5 时拼接 [BOS,11,EOS] 与 [BOS,21,EOS]，得到 x=[BOS,11,EOS,BOS,21]、y=[11,EOS,BOS,21,EOS]，mask 为全真。第四个输入 BOS 开始新 segment；第三个位置从旧文档的 EOS 预测新 BOS 仍计入 CE。这是 packer 的已核行为，仓库现有 test_best_fit_predicts_bos_across_document_boundaries 和 simple 对照都明确检查了该跨边界目标。[packer.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/buffer_assembler/packer/packer.py)；[原有目标测试](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/tests/data/test_packer_targets.py)

**长文续块还需要检查 segment 标记。** 当前 best-fit 不为截出的续块补 BOS。取 S=6，长文 [BOS,31,32,33,34,35,36,37,EOS] 的尾块为 [36,37,EOS]；另有短文 [BOS,21,22,EOS]。源码按长度降序将短文与该尾块装入同一满 bin，实际输出如下。示例 token ID 10 表示 BOS、12 表示 EOS；其余只是彼此区分的整数。

| 来自固定 packer 的对象 | 实际小例输出 |
| --- | --- |
| x | [10,21,22,12,36,37] |
| y | [21,22,12,36,37,12] |
| target mask | 全真 |
| 按 BOS 累加的 segment | [1,1,1,1,1,1] |
| source_infos 中的顺序 | 短文、长文续块 |

因为续块开头 36 不是 BOS，segment 构造不会在此切开；该续块能够 attention 到前面的短文，EOS→36 这一拼接目标也保留。这个反例限制了“multi_segments=True 总能隔离原始文档”的说法：每个需要隔离的新片段必须有可识别边界，或由额外 segment 信息显式传入。这里直接执行了固定源码的 NumPy packer，并对照 segment 构造函数确认可见域；未运行 Transformer，也没有据此推测论文分数受到多大影响。[BestFitPacker.pack 与 _slice_chunk](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/buffer_assembler/packer/packer.py)；[_begin_forward](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)；[Transformer.get_segment_arg](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/transformer.py)

接 SFT 时，ChatTemplateFeatureBuilder 先从 assistant labels 生成逐 token target_mask，packer 再取后一 token 的 mask，与 y 同步对齐；目标是预测的 assistant token，而非“输入位置本身属于 assistant”。普通与 fused 输出头都接收已经对齐的 y，不在头内再次 shift。若希望改变拼接边界的监督，或为所有续块提供完整文档隔离，应把 target mask 与 attention segment 两项分别实现和验收；这会改变发布路径的训练函数，应记录为新的后训练数据契约，不能称为原流程的无影响整理。[feature_builder.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/dataset_streamer/feature_builder/feature_builder.py)；[templator.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/dataset_streamer/templator/templator.py)；[TransformerOutputLayer](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/transformer.py)；[fused output](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/fused_blocks/transformer/transformer.py)

### 在线数据配比：token 预算、结余额与恢复状态

固定 release 的 data.md 按原顺序列出42个不同来源，权重均为正且合计1；来源权重表和两个公开 TxT360 组件的固定 revision 已收入检查包。当前 loader 的一次 refill 以 B=buffer_size×seq_len 为预算，Part I 示例为512×8192=4,194,304个位置／rank；这不是一次 optimizer update 的 global batch。BufferAssembler 为每个来源分配 b_d=w_dB，读取完整的有效样本直到达到本次目标，再结转超额量。[原始42项与版本](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/data.md) · [真实 refill 调用链](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/buffer_assembler/buffer_assembler.py#L132-L248)

设 c_{d,t−1} 为已超额读取、可抵扣当前预算的 token 结余额；q 是本次剩余目标，A 是本次真正接受的完整样本 token 和。源码两个分支统一满足下式：q≤0时 A=0；q>0时接受样本直到 A≥q。拒绝的样本不进入 A。

```math
q_{d,t}=b_d-c_{d,t-1},\qquad
c_{d,t}=A_{d,t}-q_{d,t}
=c_{d,t-1}+A_{d,t}-b_d.
```

将等式对 t 累加，得到有限 refill 次数的精确记账。若 c_{d,0}=0，且每个被接受的样本长度属于1,…,L_d，则归纳可得0≤c_{d,t}<L_d。因此累计偏离由一个样本的最大长度控制，不随 refill 次数线性累积。均匀有界结论依赖样本长度上界；无此上界时只保留恒等式。

```math
\sum_{t=1}^{T}A_{d,t}=Tb_d+c_{d,T}-c_{d,0},\qquad
0\le\sum_{t=1}^{T}A_{d,t}-Tb_d<L_d.
```

在这些条件和实数算术下，令 π̂ 为累计接受 token 中的来源比例，可以精确说明其与配置权重的差别；真实浮点核对使用声明的容差。

```math
\widehat\pi_{d,T}-w_d
=\frac{c_{d,T}-w_d\sum_jc_{j,T}}
{TB+\sum_jc_{j,T}}.
```

这三式统计的是 build_features 之后、packer 之前接受的 token，尚不是最终参与 CE 的目标。过滤会改变各来源内部的条件分布；pending 中尚未输出的片段，以及后续的长文分块、重叠、padding 和 assistant mask，又会改变输出位置与监督目标。因此，来源读取量符合 quota，并不保证最终 CE 按同一比例加权。要重建相同的数据顺序，还需保留并行收集后的拼接规则：结果虽按 future 完成时间收集，最终仍按 source_targets 的固定顺序拼接，再执行相同 RNG shuffle。

#### 算法 D2 来源预算结转与数据状态恢复

**输入** 固定来源、权重、B、tokenizer／feature builder、packer、RNG及消费位置。

**输出** packed 序列、接受 token 量、结余额、active 槽起点与恢复核验记录。

![算法 D2 来源预算结转与数据状态恢复](visualize:fde1_bGliZmlsZV9JVk1LQ2kyT25nYjVTV0pGNGl1OC13_FileDrive_45288dfe06608191af94334ba8699511)

**性质** 对应当前发布 loader 的控制流。八组新检查中的 D4–D6 直接执行原方法：100次 refill 核对结转恒等式，15次串行／线程池对照，以及恢复后连续20条输出对照；数据流与 pair-packer 是明确替身，尚未验证真实 bestfit、分布式I/O或模型训练。

恢复时保存的是 active 槽消费前的源游标、packer pending、token credits、打乱前 RNG 和槽内 popping_idx；inactive 的预取结果作为性能缓存丢弃，恢复后重建。只保存后台线程当前游标会跳过未消费数据，只恢复模型 RNG 也不足以重建来源预算。原 get_state／set_state 的这些字段已逐项核对。[完整恢复路径](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/buffer_assembler/buffer_assembler.py#L258-L356)

上述结论属于当前 release；发布文档明确旧论文 loader 的批次顺序不同，因而“当前数据状态可恢复”不等于“取得原论文逐批轨迹”。

### 初始化必须按张量核对：同名 Gaussian 并非无截断正态

发布配方把主体权重设为 init_mode=gaussian、init_std=0.02；但 get_init_fn 的 gaussian 分支实际调用 trunc_normal_，截断在指定标准差的 ±3 倍。这里 0.02 是底层正态的尺度参数，不能不加限定地写成截断后样本的精确标准差。embedding 与输出 head 的 init_embed_std / init_logits_std 为 None，构造函数则回退到 1/√1536，约 0.02552；两者并未继承主体的 0.02。[recipes.py::_model_fields](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/recipes.py)；[utils.py::get_init_fn](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/utils.py)

```math
W_{ij}\sim\mathcal N(0,\sigma^2)\ \text{conditioned on}\ |W_{ij}|\le3\sigma,
\qquad
\operatorname{Var}(W_{ij})
=\sigma^2\left(1-\frac{6\varphi(3)}{2\Phi(3)-1}\right)
\approx0.97334\,\sigma^2.
```

φ、Φ 分别是标准正态密度与分布函数；上式是理想连续采样分布的方差，不是对 BF16 权重做过实测。源码中真正独立的初始化对象如下，不能以一个“所有参数正态初始化”覆盖。

| 张量或状态 | 固定配方的实际构造 | 对循环执行的含义 |
| --- | --- | --- |
| Attention 的线性投影、dense SwiGLU 三个投影 | ±3σ 截断正态，σ=0.02 | 同一物理层初始化一次；各轮复用其数值 |
| MoE router 与各 expert 的三个投影 | 同一 get_init_fn；显式 σ=0.02 覆盖 fan-in 回退 | 专家可被重复访问；不会按逻辑调用重新抽权重 |
| Token embedding 与 vocabulary head | 截断正态，σ=1/√1536 | 两个独立构造的参数对象；不能默认 tied embedding |
| 对角输入写入 | W_in=I；δ=1、α=e⁻¹ | 不是随机线性投影或零初始化残差分支 |
| normal recurrent state | 训练时 randn/√d；评估为确定性 hashed normal | 它是每次 forward 的状态；与截断的权重初始化分开 |

主体投影的初始化参数没有按 loop_times 再除以 √R；默认 BaseResidual.forward 返回分支输出与 residual 之和。这个结论只针对所核 Part I 配方与默认 residual 配置，不替其它 xLLM 模型作保证。Layer_id 传给残差对象也不意味着默认分支按深度缩放。[MultiheadAttention](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/multihead_attention.py)；[NormalizedFeedForwardNetwork](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/normalized_feedforward_network.py)；[Expert](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/moe/expert.py)；[Transformer 的 embedding/head 构造](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/transformer.py)；[BaseResidual](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/residual/base.py)

相同单次初始化分布与相同逻辑深度，不足以推出 tied 与 untied 网络具有相同的初始化动力学。对有限二阶矩的标量残差分量 Δᵣ，有下列恒等式。各轮复用参数且 hidden 递推，跨轮协方差一般不能删除；甚至 untied 层的残差也不因权重独立就自动独立。

```math
\operatorname{Var}\!\left(\sum_{r=1}^{R}\Delta_r\right)
=\sum_{r=1}^{R}\operatorname{Var}(\Delta_r)
+2\sum_{r<s}\operatorname{Cov}(\Delta_r,\Delta_s).
```

作为边界思想实验，同一个零均值标量被加 R 次时方差为 R²σ²；R 个相互独立、同方差的标量之和才是 Rσ²。因此本页不由权重尺度直接宣布循环稳定，也不自行补上源码不存在的 1/√R 缩放。实际稳定性仍需状态范数、跨轮相关性、梯度与 clipping 记录。上述分布与协方差分析是对已核初始化的数学解释，不是预训练实测。

### 离散选专家与连续 mixture weight 具有不同梯度

在 top-2 与第 3 名之间存在严格分数间隔的局部区域，入选集合 S 不变。令 aⱼ 为 router logit，sⱼ=σ(aⱼ)，gⱼ 为入选后归一化权重，M 为 MoE 聚合输出，v 为传回该输出的损失梯度。专家输出暂按给定 hidden 固定，则 router logit 的任务梯度为：

```math
\begin{aligned}
\frac{\partial g_e}{\partial a_j}
&=\mathbf1[j\in S]\,g_e(\mathbf1[e=j]-g_j)(1-s_j),\quad e\in S,\\
\frac{\partial L_{\rm task}}{\partial a_j}
&=\mathbf1[j\in S]\,g_j(1-s_j)
\left\langle v,E_j(h)-M(h)\right\rangle.
\end{aligned}
```

未入选专家的任务梯度在这条 router 路径上为零；hidden 的总梯度仍包括入选专家对输入的导数。源码没有对 top-k 身份使用直通估计器。分数并列处不可把上述局部导数解释为离散选择的可微概率模型。

辅助项使用全部 25 个专家的归一化 sigmoid 分数 p，而频率 f 由离散计数产生，反向时不求导。N 为该次 router 调用实际接收的位置数，包含哪些 padding 位置由输入与实现决定，不能直接替换成 LM 的有效目标数。

```math
\frac{\partial L_{\rm bal}}{\partial a_{t,j}}
=\frac{25}{N}\,p_{t,j}(1-s_{t,j})
\left(f_j-\sum_e f_ep_{t,e}\right).
```

因此未入选专家也能通过辅助项获得 router 梯度。`TopKRouting.backward` 正是先累加入选分数梯度与全专家辅助梯度，再通过 sigmoid backward 和线性投影回传；它不是“top-2 后只训练这两个 router 行”。

### 一次更新的两个分母不可合并

在 context parallel size=1、W 个数据并行副本按均值归约梯度、每个副本累积 M 个 microbatch 的记法下，发布实现对应下列目标。n 是有效 next-token 目标数，C 是该 forward 实际执行的 MoE 调用数。

```math
\begin{aligned}
L_{\rm update}
&=\frac{\sum_{w,m,t}a_{wmt}\ell_{wmt}}
{\sum_{w,m,t}a_{wmt}}\\
&\quad+\frac{10^{-3}}{WM}\sum_{w,m}
\frac{1}{C_{wm}}\sum_{c=1}^{C_{wm}}L_{{\rm bal},wmc}.
\end{aligned}
```

语言模型项按全窗口的有效目标数归一化；auxiliary 项对 microbatch 及逻辑 MoE 调用取均值。有效目标数不同的 microbatch 并没有因而获得不同 auxiliary 权重。源码还显式把 context parallel size 纳入其局部分母；迁移训练器时应保留 gather 与梯度归约约定，而不是再机械除一次累积步数。全部循环调用保持计算图，activation checkpoint 仅在反向时重算，不能替换为 detach。

来源：[`TopKRouting.forward/backward`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/moe/router.py)；[`forward_backward_window`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py)；[`_run_layers` 与 `num_aux_loss_layers`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)。局部导数由这三个实现的目标推导。

### 共享调用只产生一套 AdamW 状态，不能逐轮优化后再平均

同一组权重在不同轮次处理不同状态，最后却只形成一个训练目标。计算梯度时，可以先在纸面上把 R 次调用各记成一个独立坐标 θ₁,…,θᵣ，得到展开损失 L̃，再把这些坐标同时设回同一个 θ。对 θ 做一次微小改动，就同时改动了所有调用位置；链式法则因而要求把各位置传回的梯度相加。

这里的 gᵣ 包含该次调用经后续计算传到最终目标的完整伴随量，不是只看当前层输出的局部误差。各轮输入可以不同，梯度方向也可以相反；共享的是参数对象，而非这些调用对损失的贡献：

```math
L(\theta)=\widetilde L(\theta,\ldots,\theta),\qquad
g_r=\left.\nabla_{\theta_r}\widetilde L\right|_{\theta_1=\cdots=\theta_R=\theta},\qquad
\nabla_\theta L=\sum_{r=1}^{R}g_r.
```

这一步求和已经是所定义损失对共享参数的导数。任务损失此前已按有效目标数归一化，auxiliary 也按自己的调用数归约；再额外除以 R，会改变更新所用的梯度尺度。实现应先让所有调用把贡献累积到同一参数，再对完整模型梯度做全局裁剪，最后交给 AdamW。

令 c 为全局裁剪对完整模型施加的同一个系数。在 Part I 的 loss_rescaling=False、state.scale=1 固定配方下，进入 AdamW 的是 g̃=cΣgᵣ，而非 R 份分别裁剪的梯度。下面的一阶矩记录这个合成梯度的滑动平均，二阶矩记录其逐坐标平方的滑动平均；后者因此包含不同调用之间的交叉项。通用动态缩放分支的执行方式另见下节。

```math
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)\widetilde g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)\widetilde g_t^{\odot2},\\
\left(\sum_r g_r\right)^{\odot2}
&=\sum_r g_r^{\odot2}+2\sum_{r<s}g_r\odot g_s,\\
\theta_{t+1}&=(1-\eta_t\lambda)\theta_t
-\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\end{aligned}
```

交叉项正是“先汇总再优化”与“每轮各自优化再平均”分开的地方。取首次更新、θ=0、无裁剪和 weight decay、ε=0，两个调用的标量梯度为 2 与 −1。共享参数先合成一个正梯度，Adam 给出 −η 的更新；若两轮各有自己的 moments，它们先各自形成方向相反、幅度相同的 Adam 更新，再平均便得到 0。实际 ε=10⁻⁸ 只使后一结果接近 0，两种算法仍不等价。

普通 SGD 的独立坐标更新若随后欧氏投影到对角子空间，会产生平均梯度；AdamW 则在投影之前已对各自的二阶矩作了非线性处理。因此，不能把这个 SGD 几何关系直接用作 AdamW 的学习率换算。实现共享 core 或共享 adapter 时，下一张表中的一套参数、一套 moments 和一次 optimizer update 必须同时保留。

| 真实执行点 | 被归约或修改的对象 | 移植时必须保留的次序 |
| --- | --- | --- |
| train.py::main 的数据窗口 | 先读取 M 个 microbatch 并将其放到 GPU | 整个窗口交给 forward_backward_window；512=W×M×本地 batch size |
| looped.py::_run_layers | 反复调用 self.layers[layer_id] | 共享 Parameter，保留各次调用的激活依赖 |
| train.py::forward_backward_window | 已按有效目标与 auxiliary 约定缩放的 microbatch 目标 | backward 累积到同一参数；未用 no_sync 跳过前 M−1 次通信；不在每轮执行 optimizer.step |
| train.py 的 clip_grad_norm_ | 整个累积窗口的整模型梯度 | 根 FSDP 按张量身份去重并归约分片范数；Part I scale=1、阈值为 1；先裁剪再更新 |
| rescale_grads → AdamW.step | Part I scale=1 时 rescale_grads 直接返回；随后更新一套一、二阶矩 | weight decay 每次 optimizer update 施加一次，不按 core 调用次数重复 |
| scheduler.step → zero_grad → after_optimizer_step | 更新学习率调度，清空窗口，再调用可选模型 hook | 不把循环次数当作 optimizer step 或 warmup 计数；hook 每次 optimizer update 一次 |

源码依据：[looped.py::_run_layers](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)；[train.py::forward_backward_window 与更新主循环](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py)；[build_optimizer / rescale_grads](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/optim/optimizer.py)。导数恒等式已用双调用标量网络的有限差分核对；Adam 反例为代数检验，未执行模型训练。共享 adapter 的后训练同样必须遵守这一参数与更新契约。

### 固定配方与通用动态缩放分支必须分开

Part I 的发布配方锁定 BF16 计算、FP32 gradient reduce-scatter、loss_rescaling=False；training_config 还固定 model/context parallel size 均为 1。因此这里的 W 是数据并行大小，global_batch_size=512 必须能被 W×M 整除，本地 microbatch size 为 512/(W×M)。默认模型配置选择 FSDP1；fully_sharded_size=None 使其在数据并行组内分片。forward_backward_window 没有用 no_sync 包住前 M−1 次 backward，不能把“梯度累积”自动解释为“只在最后一次反向通信”。[recipes.py 的 _COMMON_TRAINER_FIELDS](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/recipes.py)；[training_config](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/training.py)；[build_model_fsdp1](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/build.py)

裁剪由根 FSDP 对象执行。它先按参数对象去重，将各 rank 的分片梯度范数以 FP32 计算并归约；未分片参数只计一次。对已经完成数据并行梯度平均的整模型梯度 g，固定配方 τ=1 的实际缩放如下，ε_clip=10⁻⁶。函数返回裁剪前范数 G，而 optimizer 接收裁剪后的梯度。

```math
G=\left(\sum_{p\in\Theta_{\rm unique}}\|g_p\|_2^2\right)^{1/2},
\qquad
c=\begin{cases}
1,&G\le\tau,\\
\displaystyle\frac{\tau}{G+\epsilon_{\rm clip}},&G>\tau,
\end{cases}
\qquad
\widetilde g_p=cg_p.
```

这次归约跨的是参数分片，和前面的数据并行梯度平均是两个操作。若两个分片的标量梯度为 3、4，忽略 ε_clip 时全局裁剪得到 (0.6,0.8)；各 rank 独立裁到 1 会得到 (1,1)，其整模型范数仍为 √2。共享 core 调用次数也不能再乘进 G；那些调用的贡献已经进入同一参数的累积梯度。[FullyShardedDataParallel.clip_grad_norm_](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/distributed/fully_sharded_data_parallel.py)

> [!CALLOUT]
>
> **仅针对另行启用的通用动态缩放分支。** 固定提交的 train.py 在 grad_norm<0.05 且 scale<4096 时先把 scale 加倍，再用新 scale 还原本轮梯度；未触发裁剪时，这会把按旧 scale 累积的梯度再缩小一半。其 overflow 条件为 grad_norm>500 或 NaN；当新 scale 未触发下限异常时，会在 zero_grad 之前 continue，下一次 while 又读取新窗口，因而源码没有实现“清空失败梯度并重试同一批数据”的事务。上述分支被 Part I recipe 禁用，不能据此推断论文训练经历了这些事件。移植到后训练或 FP16 时，应分开保存本轮 s_used 与下轮 s_next，并明确失败时清梯度、数据重试和 scheduler 计数的规则。[train.py 更新主循环](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py)；[rescale_grads](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/optim/optimizer.py)

### 训练日志的平均不等于优化目标

令 n_m>0 为第 m 个 microbatch 在全部数据并行 rank 上的有效目标数，ℓ̄_m 为这些目标的平均 token loss，M 为累积窗口长度。忽略分母中防零的 10⁻⁶ 后，实际用于反向的语言模型目标与 optim.avg_loss 的记录量分别为：

```math
L_{\rm token}=\frac{\sum_{m=1}^{M}n_m\bar\ell_m}{\sum_{m=1}^{M}n_m},
\qquad
L_{\rm log}=\frac1M\sum_{m=1}^{M}\bar\ell_m.
```

forward_backward_window 用全窗口分母构造 train_loss，却返回按各 microbatch 自己分母归一化的 detached loss；主循环再对这些返回值取均值，随后跨 rank 平均。auxiliary 单独记录为 avg_aux_loss，并未加进 avg_loss。若保留数值 ε，跨 W 个 rank 平均后相应分母为 Σn_m+Wε 与 n_m+Wε；它不改变两种聚合的区别。[forward_backward_window 与 optim_metrics](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py)

用窗口内的总体协方差记法，可精确刻画忽略 ε 时的差异。n̄ 与 ℓ̄ 分别是 n_m、ℓ̄_m 对 M 个 microbatch 的等权平均。

```math
L_{\rm token}-L_{\rm log}
=\frac{\operatorname{Cov}_{M}(n,\bar\ell)}{\bar n},
\qquad
\operatorname{Cov}_{M}(n,\bar\ell)
=\frac1M\sum_m(n_m-\bar n)(\bar\ell_m-\bar\ell).
```

例如 n=(1,9)、ℓ̄=(4,1)，有效 token 加权目标为 1.3，日志均值为 2.5。所有 microbatch 有效目标数相同是两者相等的充分条件；一般条件是上述协方差为零。比较改变 packing、mask 或 accumulation 的训练曲线时，应保存 token-loss 分子与有效目标数后汇总；不能把日志均值的变化直接当成优化目标的变化。这里核对的是训练日志路径，不改写论文下游评测分数或其他独立验证器的指标定义。

## 4. Dense 消融：每个结论都带比较预算

下表统一为论文 730M non-embedding dense、336B tokens（表中称 460 TPP）、28 个物理块、112 次逻辑执行。选择的任务同时覆盖收益与退步；完整十任务结果在 Tables 2–4。所有数字均为原表原始分数，已对 PDF 排版核对，避免用已四舍五入的差值反推原始分数。

| 控制变量与来源 | 指标 | 控制模型 | 改动后 | 可支持的结论 |
| --- | --- | ---: | ---: | --- |
| Q1：Ouro → Middle，Table 2 p.7 | MATH500 | 20.80 | 32.80 | 仅移到 sandwich envelope 可带来很大的数学收益 |
| 同上 | GSM8K | 54.97 | 62.02 | 此设置有多步数学收益 |
| 同上 | BBH-CoT | 45.83 | 48.30 | 推理收益同时出现在另一个任务族 |
| 同上 | DROP | 36.38 | 38.99 | 对 passage 信息抽取亦改善 |
| 同上 | HumanEval+ | 29.88 | 28.66 | 不能概括为所有任务均改善 |
| Q2：Middle 无写入 → prelude 写入，Table 3 p.9 | BBH-CoT | 48.30 | 54.94 | 上下文访问与示例利用改善 |
| 同上 | HumanEval+ | 28.66 | 34.15 | 代码规格任务有收益 |
| 同上 | MATH500 | 32.80 | 29.20 | 持续输入写入有数学代价 |
| 同上 | GSM8K | 62.02 | 59.51 | 代价不局限于一个数学基准 |
| Q2：Ouro 无写入 → raw embedding 写入，Table 3 | HumanEval+ | 29.88 | 32.32 | 写入收益不能全部归为 prelude 的功劳 |
| 同上 | MATH500 | 20.80 | 19.20 | raw-token injection 也可能降低数学分数 |
| Q3：输入初态 → 随机初态，Table 4 p.12 | ARC-C | 37.60 | 40.94 | 随机初态有个别收益 |
| 同上 | MATH500 | 29.20 | 27.40 | 随机初态不是普遍增益 |
| 同上 | MBPP+ | 47.35 | 45.24 | 也可能损伤代码任务 |
| Q3：输入初态单状态 → shared H/L，Table 4 | DROP | 40.38 | 42.18 | 双状态有任务相关收益 |
| 同上 | MATH500 | 29.20 | 22.00 | 最大退步集中于数学 |
| 同上 | GSM8K | 59.51 | 57.32 | 不能据层次结构名称推断更强推理 |

Q1 在 42B、84B、168B、336B 四个预算上追踪；MATH500 与 BBH-CoT 的方向跨预算保持。Q2 的结果说明“持续保留输入”与“允许状态远离初始表示”之间存在任务相关权衡，但论文没有通过机制实验分离写入强度、写入位置与状态吸引效应。Q3 中输入初态胜过随机初态六项；随机初态超过一分的收益两项、损失四项；shared H/L 超过一分的收益三项、损失三项。完整 Huginn 仍高于 Ouro 十项，这不意味着构成 Huginn 的每个组件都各自提高十项。[原论文 §3.4–3.5、Figs.5–9](https://arxiv.org/pdf/2610.23036v1#page=7)

评测口径应随表保留：MATH500 是 4-shot EM、500 题、4096/2048 输入/输出上限；GSM8K 是 8-shot answer accuracy、1319 题、2048/512；BBH-CoT 是 3-shot EM、6511 题、4096/1024；DROP 是 3-shot F1、9535 题、2500/64；HumanEval+ 是 0-shot EvalPlus pass@1、164 题、1024/512；MBPP+ 是 3-shot EvalPlus pass@1、378 题、3096/1024。生成均 greedy single sample、无 chat template。论文 Appendix B 还记录 Q3 三个 MATH500 行使用固定 Minerva prompt 变体；因此不宜将这些小差异包装为已完全消除 prompt 影响的纯机制结论。[原论文 Table 7、Appendix B，pp.24–25](https://arxiv.org/pdf/2610.23036v1#page=24)

### Q1–Q3 的完整十任务证据

上面的速查保留各项收益与退步的解释；下面补齐原表所有端点和 feedforward 对照。Q1 的 D28 只匹配约 730M non-embedding resident 参数，D112 只匹配 112 次逻辑执行、约 2.9B 独立参数；不能把二者同时当成 resident 与 compute 双匹配的对照。各循环模型均为 336B training tokens，原表称 460 TPP。以下 Δ 一栏逐字保留论文显示值。

**Table 2：循环位置。** Ouro 为 28×4；Middle 为 8+12×8+8，均无重复输入写入。[原表 p.7](https://arxiv.org/pdf/2610.23036v1#page=7)

| 指标 | D28 | D112 | Ouro | Middle | 原表 Δ：Middle−Ouro |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARC-C | 36.39 | 43.78 | 39.40 | 38.45 | −0.94 |
| HellaSwag | 41.73 | 50.28 | 44.71 | 45.01 | +0.30 |
| MMLU | 39.86 | 54.85 | 46.32 | 47.27 | +0.95 |
| BBH-CoT | 34.42 | 55.91 | 45.83 | 48.30 | +2.47 |
| TriviaQA | 25.21 | 38.83 | 28.51 | 28.60 | +0.09 |
| DROP | 31.04 | 40.88 | 36.38 | 38.99 | +2.61 |
| MATH500 | 12.80 | 47.60 | 20.80 | 32.80 | +12.00 |
| GSM8K | 29.11 | 64.22 | 54.97 | 62.02 | +7.05 |
| HumanEval+ | 21.34 | 42.07 | 29.88 | 28.66 | −1.22 |
| MBPP+ | 35.19 | 51.06 | 41.01 | 43.12 | +2.12 |

**Table 3：Middle 内的写入对照。** 输入初态均为 prelude 输出，比较关闭写入与每轮写入 prelude。[原表 p.9](https://arxiv.org/pdf/2610.23036v1#page=9)

| 指标 | Middle 无写入 | Middle prelude 写入 | 原表 Δ：写入−无写入 |
| --- | ---: | ---: | ---: |
| ARC-C | 38.45 | 37.60 | −0.86 |
| HellaSwag | 45.01 | 45.37 | +0.36 |
| MMLU | 47.27 | 49.79 | +2.53 |
| BBH-CoT | 48.30 | 54.94 | +6.63 |
| TriviaQA | 28.60 | 29.42 | +0.82 |
| DROP | 38.99 | 40.38 | +1.39 |
| MATH500 | 32.80 | 29.20 | −3.60 |
| GSM8K | 62.02 | 59.51 | −2.51 |
| HumanEval+ | 28.66 | 34.15 | +5.49 |
| MBPP+ | 43.12 | 47.35 | +4.23 |

**Table 3：Ouro 内的写入对照。** 初态均为 token embedding，比较关闭写入与每轮写入 raw embedding。它与上一表共同保留了输入源和写入次数不同这一跨族限制。

| 指标 | Ouro 无写入 | Ouro raw-token 写入 | 原表 Δ：写入−无写入 |
| --- | ---: | ---: | ---: |
| ARC-C | 39.40 | 39.06 | −0.34 |
| HellaSwag | 44.71 | 45.65 | +0.94 |
| MMLU | 46.32 | 48.11 | +1.79 |
| BBH-CoT | 45.83 | 47.63 | +1.80 |
| TriviaQA | 28.51 | 28.53 | +0.02 |
| DROP | 36.38 | 37.29 | +0.91 |
| MATH500 | 20.80 | 19.20 | −1.60 |
| GSM8K | 54.97 | 55.50 | +0.53 |
| HumanEval+ | 29.88 | 32.32 | +2.44 |
| MBPP+ | 41.01 | 45.50 | +4.50 |

**Table 4：固定 Middle 和 prelude 写入后的状态对照。** 随机初态列是完整 Huginn；H/L 列是共享 recurrent body 的双状态版本，不是已发布的 split H/L 变体。[原表 p.12](https://arxiv.org/pdf/2610.23036v1#page=12)

| 指标 | 输入初态 | 随机初态 | Shared H/L | 原表 Δ：H/L−输入初态 |
| --- | ---: | ---: | ---: | ---: |
| ARC-C | 37.60 | 40.94 | 39.83 | +2.23 |
| HellaSwag | 45.37 | 45.47 | 45.90 | +0.53 |
| MMLU | 49.79 | 48.50 | 49.49 | −0.30 |
| BBH-CoT | 54.94 | 54.43 | 55.12 | +0.18 |
| TriviaQA | 29.42 | 28.97 | 30.17 | +0.75 |
| DROP | 40.38 | 40.63 | 42.18 | +1.80 |
| MATH500 | 29.20 | 27.40 | 22.00 | −7.20 |
| GSM8K | 59.51 | 60.73 | 57.32 | −2.19 |
| HumanEval+ | 34.15 | 32.32 | 35.37 | +1.22 |
| MBPP+ | 47.35 | 45.24 | 44.71 | −2.64 |

原表 Δ 与显示分数的直接相减是两个记录。例如 Q1 ARC-C 的原表 Δ=−0.94，而 38.45−39.40=−0.95；Q2 Middle 的 BBH 原表 Δ=+6.63，而 54.94−48.30=+6.64。本页保留二者各自口径，不用重新相减的数覆盖原表。若两个分数及其差均由同一未舍入值分别四舍五入至 0.01，则下面的误差界成立；现有 0.01 差异与该解释相容，但无法凭表格恢复未舍入分数。

```math
\begin{aligned}
\bar s_i&=s_i+e_i,\qquad
\bar\Delta=(s_1-s_0)+e_\Delta,\\
|e_i|,|e_\Delta|&\le0.005
\quad\Longrightarrow\quad
|\bar\Delta-(\bar s_1-\bar s_0)|\le0.015.
\end{aligned}
```

补齐其余四项评测定义：ARC-C 为 0-shot、1,165 题、按答案 token 数归一化的选择准确率，输入 cap 512；HellaSwag 为 0-shot、10,042 题、未按长度归一化的最高似然选择准确率，cap 256；MMLU 为 5-shot、14,042 题、micro accuracy，cap 2,048；TriviaQA 为 5-shot、11,313 题、F1，输入/输出 cap 为 512/24。前三项通过候选答案似然评分，不生成答案。TriviaQA 的少量保留轨迹早于 newline stopping，但 prompt、输出 cap 与 F1 scorer 相同。不能把本篇 TriviaQA 的 F1 与 LoopMoE 的 exact match 当成同一指标；Q3 的 MATH500 prompt 变体边界仍按上文保留。[Table 7、Appendix B](https://arxiv.org/pdf/2610.23036v1#page=24)

### 把每个性能结论写成条件差分

记训练过程输出 θ=T(a,D,η,ξ)，其中 a 为结构、D 为训练 token 预算、η 为其余训练配方、ξ 为训练随机性；τ 固定 prompt、shots、截断、生成与评分协议。对任务 j 的比较量是两个已训练模型在同一协议下的分数差，而不是结构本身脱离训练过程的性质。

```math
\begin{aligned}
\widehat\Delta_j(a_1,a_0\mid D,\eta,\xi,\tau)
&=\widehat S_j(T(a_1,D,\eta,\xi);\tau)\\
&\quad-\widehat S_j(T(a_0,D,\eta,\xi);\tau).
\end{aligned}
```

若两条训练实际使用不同随机性或按学习率 sweep 各选最佳，则在记号中分别写出它们，不能省略后宣称获得同 seed 的配对实验。论文 Q1–Q3 的原表是在指定 recipe 和预算上的受控比较；现有结果不足以给出跨训练 seed 的期望效应与置信区间。

以 Q1 的 MATH500 为例，32.80−20.80=12.00 个百分点；相对基线的增幅才是 12/20.8≈57.69%。这两种单位不能混写。Q2 中上下文相关任务的正差与 MATH500/GSM8K 的负差属于同一组证据，应保留任务向量，而非仅以均值认定输入注入必然有益。

```math
\widehat{\boldsymbol\Delta}
=(\widehat\Delta_1,\ldots,\widehat\Delta_J),\qquad
\overline\Delta=\frac1J\sum_{j=1}^{J}\widehat\Delta_j.
```

跨任务平均默认给每个 benchmark 相同权重；它既不是按所有题数合并的 micro accuracy，也不是自动代表真实使用分布的期望收益。Q3 的随机初态和 shared H/L 结果同样应理解为各自有限配方的条件差分，不能据此推出所有随机状态或双状态体系的优劣。

原始结果与口径：[Tables 2–4、Appendix B](https://arxiv.org/pdf/2610.23036v1#page=7)。以上定义用于精确解释原有数字，不增补论文未报告的训练重复或统计显著性。

### Q2 识别的是整个写入算子，衰减与外部信息尚未各自识别

Table 3 在每个拓扑内部比较 no write 与 diagonal write，先固定了该族的循环布局。接下来需要看清这个开关到底改了什么：关闭时，core 直接收到 z；开启时，core 收到衰减后的旧状态与新输入的和。下面把“开启后的输入”减去“关闭后的输入”，得到两项：一项改变旧状态留下的量，另一项改变输入源写入的量。这是在进入 recurrent core 之前逐坐标成立的分解。

```math
W(z,e)-z=(\operatorname{Diag}(\alpha)-I)z
+\operatorname{Diag}(\delta)W_{\rm in}e.
```

因此，Table 3 中上下文任务改善、部分数学任务下降，反映的是这两项同时改变后的训练结果。即便初始化 W_in=I、δ=1，α=e⁻¹ 也已让旧状态少留下一部分。重新读取 prompt 可能改变结果，衰减已有状态也可能改变结果；现有对照给出的是整套已训练写入机制的条件性能差，并未把两种作用分别测出。

若想解释任务间的差异，下一步应让“是否衰减旧状态”与“是否加入输入源”分别变化。下面的四个算子正是为此设置：保留原论文的两个角点，再补上只衰减和只写入的对照，才能比较各自作用及其交互。[论文 §3.4.2、Table 3](https://arxiv.org/pdf/2610.23036v1#page=9)；[InputInjection.forward](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)

若研究目标进一步是分离这两个机制，可定义下面的四种算子。a 控制旧状态衰减，b 控制外部写入；它们是后续控制设计，不是原论文已有的四个实验。

```math
W_{ab}(z,e)=\bigl[(1-a)I+a\operatorname{Diag}(\alpha)\bigr]z
+b\operatorname{Diag}(\delta)W_{\rm in}e,
\qquad a,b\in\{0,1\}.
```

原有 no-write 与 write 对应角点 (0,0) 和 (1,1)。若 Y_ab 是相同训练与评测契约下的结果，原文只给出这条对角差；未报告两个混合角点就无法唯一求出衰减主效应、写入主效应及二者交互。实现混合角点时还需声明未使用参数是否保留、冻结、参与 weight decay，以及如何控制初始化和参数预算，不能只改 forward 后把旧 checkpoint 当作重新预训练的对照。

```math
Y_{11}-Y_{00}
=(Y_{10}-Y_{00})+(Y_{11}-Y_{10}),\qquad
\Gamma=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
```

跨族比较还增加了输入源差异：Ouro 写 raw embedding，Middle 写经 prelude 编码的 e，且写入次数分别为 4 与 8。Table 3 的 MATH500 写入差为 −1.60 与 −3.60 个百分点，二者再差为 −2.00；这只能描述两套指定策略的响应不同，不能单独估计“contextualized source 的因果效应”。原文已经把写入强度、位置与状态相互作用列为尚未分离的解释；上述算子分解把这个边界明确到可实施的控制变量。[Table 3、§3.4.2 末段](https://arxiv.org/pdf/2610.23036v1#page=10)

## 5. MoE 主结果、路由观测与因果干预

Table 5（p.15）统一为 500B training tokens，looped 模型为 8.0B resident / 793.9M 每物理遍历 active，feedforward 为 32.0B / 3.2B。三者逻辑深度及 active compute 对齐，但 resident 容量不相同；不要把它写成三者总参数也匹配。

| 指标 | Feedforward 112 层 | Ouro MoE | Huginn MoE |
| --- | ---: | ---: | ---: |
| ARC-C | 53.22 | 51.33 | 50.47 |
| HellaSwag | 57.30 | 54.82 | 55.10 |
| MMLU | 67.13 | 64.03 | 66.81 |
| BBH-CoT | 71.76 | 69.05 | 68.85 |
| TriviaQA | 59.02 | 50.98 | 51.04 |
| DROP | 56.81 | 57.70 | 59.65 |
| MATH500 | 53.40 | 49.80 | 53.40 |
| GSM8K | 80.82 | 78.92 | 83.62 |
| HumanEval+ | 56.10 | 48.17 | 49.39 |
| MBPP+ | 60.85 | 58.99 | 59.52 |

Huginn 对 Ouro 为八胜两负；feedforward 总体仍领先七项，Huginn 在 DROP / GSM8K 领先、MATH500 持平。由此支持“以显著更少 resident 参数维持有竞争力的性能”，不支持“循环全面胜过 feedforward”。MoE 比较直接比较完整 Ouro/Huginn 配方，不能把 dense Q1–Q3 的组件归因自动视为已在 MoE 分别重复验证。[原论文 Table 5、§4.4、§5](https://arxiv.org/pdf/2610.23036v1#page=15)

路由证据分为观测与干预两层。先看观测：500B 时 normalized balancing loss 为 Huginn 1.571、FF 1.652、Ouro 1.899。Huginn 在同口径下的辅助目标值更低，作者据此判断路由更均衡；不过，该标量不是离散专家计数的方差，严格解释见下文。

另一项观测直接检查专家身份是否随轮次变化。48 输入的小型 probe（六任务各八题、输入 cap 1024）在每个相邻迭代有 47,928 个有效 token–layer 决策，top-1 平均切换率为 49.4%。它表明不同轮次确实使用不同专家；要判断这些切换是否有益，还需下一项干预。[原论文 Fig.10、Fig.11、Appendix C](https://arxiv.org/pdf/2610.23036v1#page=16)

第二层证据来自冻结专家身份的干预：第 1 轮正常选择专家，第 2–8 轮固定使用同 token、同物理层在第 1 轮的 top-2 身份，同时按当前 hidden 重新计算并归一化这两个专家的权重。这样保留了连续 mixture-weight 变化，去掉了离散 expert-identity 变化。完整匹配的选择题评测中，MMLU 从 67.7 到 56.5，ARC-E 从 80.4 到 70.8，ARC-C 从 49.1 到 40.8；其余 BoolQ / PIQA / COPA 也下降。这是在该干预和任务范围内支持“迭代特异专家选择具有功能贡献”的因果证据。

这组结果有独立的汇总口径：此处 MMLU 是 57 subject accuracy 的非加权均值，不能直接与 Table 5 的 micro accuracy=66.81 合并。原 Fig.12 的 gap 标签文字与负号方向存在表述歧义，因此这里直接报告 original→frozen 原始值。[原论文 Fig.12、Appendix C，pp.17、25](https://arxiv.org/pdf/2610.23036v1#page=17)

Fig.13 的 dense/MoE gap 汇总只覆盖指定的可比任务子集和共同端点；4.96→1.71 的均值缩小不能冒充 Table 5 十任务均值，图中也未列 MATH500。

### MoE 缩小 feedforward 差距的原始证据

Fig.13 比较的是两个模型族各自相对 feedforward 的差距。对训练 token 预算 T、任务 j 和固定评测协议 τ，令正 gap 表示 feedforward 更高分、负 gap 表示 Huginn 更高分。下面的第二个差分小于零，表示该任务上 MoE 的 gap 比 dense 小；它不要求 Huginn MoE 已超过 feedforward MoE。

```math
\begin{aligned}
g_{f,j}(T;\tau)
&=S_j(D^{\,f}_{112};T,\tau)-S_j(\mathrm{Huginn}^{\,f};T,\tau),
\quad f\in\{\mathrm{dense},\mathrm{MoE}\},\\
\Delta^{\mathrm{gap}}_j
&=g_{\mathrm{MoE},j}-g_{\mathrm{dense},j},\\
\bar g_{f,\mathcal J}
&=\frac{1}{|\mathcal J|}\sum_{j\in\mathcal J}g_{f,j}.
\end{aligned}
```

**Fig.13(a)：共同端点的九任务。** 预算沿用该面板图注的“shared endpoint”标记；它不含 MATH500，不能用上方 500B 的 Table 5 代替此图的 MoE 端点。下表完整保留图中 gap、Δ 及 Average，单位均为百分点。[原图与图注 p.18](https://arxiv.org/pdf/2610.23036v1#page=18)

| 指标 | Dense gap | MoE gap | 原图 Δ gap |
| --- | ---: | ---: | ---: |
| ARC-C | 2.84 | 4.21 | +1.37 |
| HellaSwag | 4.81 | 2.05 | −2.76 |
| MMLU | 6.35 | 0.20 | −6.15 |
| BBH-CoT | 1.48 | −0.82 | −2.30 |
| TriviaQA | 9.86 | 6.97 | −2.89 |
| DROP | 0.25 | −2.88 | −3.13 |
| GSM8K | 3.49 | 0.68 | −2.81 |
| HumanEval+ | 9.75 | 5.49 | −4.26 |
| MBPP+ | 5.82 | −0.53 | −6.35 |
| 原图 Average | 4.96 | 1.71 | −3.25 |

九项中有八项缩小差距，ARC-C 是反例：gap 由 2.84 增至 4.21。BBH、DROP 与 MBPP+ 的 MoE gap 在该端点为负；其余项目仍为正。因而“均值由 4.96 缩至 1.71”保留了总体改善，也保留了任务和训练端点相关的排序变化。

**Fig.13(b)：固定七任务随预算变化。** 任务集合为 ARC-C、HellaSwag、MMLU、BBH-CoT、TriviaQA、DROP、HumanEval+；相对 panel(a) 再去掉 GSM8K 和 MBPP+。所有列均为这一相同集合的 gap 均值。

| 训练 tokens | Dense 平均 gap | MoE 平均 gap | 原图 Δ gap |
| --- | ---: | ---: | ---: |
| 42B | 1.40 | −2.20 | −3.60 |
| 84B | 2.74 | −1.56 | −4.30 |
| 168B | 3.48 | −0.32 | −3.80 |
| 336B | 5.05 | 2.17 | −2.87 |

四个预算的 MoE gap 都小于 dense gap。按 panel(a) 的显示值仅取这七项，得到 dense≈5.0486、MoE≈2.1743，舍入后与 panel(b) 的 336B 末点 5.05、2.17 一致；这个核对说明两个任务集合的均值不能混用。末行原图 Δ=−2.87，而显示值 2.17−5.05=−2.88，继续按前文规则保留原图与显示值计算各自的精度口径。

另一个原文汇总来自 **500B Table 5 的四项数学与代码任务**：MATH500、GSM8K、HumanEval+、MBPP+。Huginn MoE 平均为 61.48，feedforward MoE 为 62.79；这是该四任务集合的平均分，既不是 Fig.13 的九任务 gap，也不是其七任务训练曲线。[§4.4、Table 5](https://arxiv.org/pdf/2610.23036v1#page=16)

上述 gap 差分是四个已训练端点的代数比较。只有预算、任务集合与协议固定时，才可把它解释为该设置下相对差距的变化。它没有单独识别“只增加稀疏路由”的机制效应；原文也仍要求多规模、多 seed 和逐组件 MoE 对照。因此后训练实验应保存各模型原始任务向量、checkpoint token 数与汇总权重，再比较 gap。[§5](https://arxiv.org/pdf/2610.23036v1#page=18)

### 三种路由统计回答三个不同问题

热图对每个 token 是否进入 top-2 计一次，所以同层同轮的专家激活率之和为 2。训练辅助项中的 f 则按 2N 个路由槽位归一化，和为 1。top-1 切换率只检查第一名身份；从有序选择 (e₁,e₂) 变为 (e₂,e₁) 会计为切换，但专家集合完全没有改变。

```math
\begin{aligned}
q_{\ell,r,e}&=\frac1{N_{\rm valid}}\sum_t
\mathbf1[e\in S_{\ell,r,t}],\qquad \sum_e q_{\ell,r,e}=2,\\
\operatorname{Switch}_{r}
&=\frac1{N_{\rm decisions}}\sum_{\ell,t}
\mathbf1[e^{(1)}_{\ell,r,t}\ne e^{(1)}_{\ell,r+1,t}].
\end{aligned}
```

令 u 为 25 维均匀分布。负载辅助项可以精确改写如下；它测量频率偏差与平均概率偏差的内积，而不是频率偏差的平方范数。

```math
L_{\rm bal}=25\langle f,\bar p\rangle
=1+25\langle f-u,\bar p-u\rangle.
```

均匀路由给出 1，但 1 不是这类目标的一般下界。例如构造 230 个位置：70% 的位置让前两个 sigmoid score 为 0.5001、其余为 0.5；另外 30% 在其余 23 个专家上均匀轮换 top-2，入选 score 为 0.9、其余为 0.0001。前两个专家各占 35% 的槽位，直接计算却得到 L_bal≈0.7982。这个合法 top-2 数学反例不涉及论文模型；它说明原始 1.571/1.652/1.899 应与实际 counts、最大负载及负载变异系数共同解读。

### 冻结的是专家身份，反事实 hidden 仍须独立演化

切换率说明专家身份确实变化，却还没有回答变化是否影响预测。冻结干预只限制“可选择哪些专家”：固定 checkpoint、同一输入和相同初始随机状态后，normal 与 frozen 在第一轮相同；此后 frozen 一直使用第一轮的专家集合，但仍根据自己当前的 hidden 重新计算 mixture 权重。

下面第一式固定集合，第二式保留入选专家之间的连续权重调整。两条轨迹从第二轮起可以产生不同 hidden，后面的 attention 和路由分数也各自继续演化。若把 normal 的 hidden 送给 frozen，或连第一轮权重也一并冻结，测量的就不再是这个“仅固定身份政策”的实验。

```math
\begin{aligned}
S^{\rm frozen}_{\ell,r,t}&=S^{\rm frozen}_{\ell,1,t},\quad r\ge2,\\
g^{\rm frozen}_{\ell,r,t,e}
&=\frac{\sigma(w_e^\top h^{\rm frozen}_{\ell,r,t})}
{\sum_{j\in S^{\rm frozen}_{\ell,1,t}}
\sigma(w_j^\top h^{\rm frozen}_{\ell,r,t})},
\quad e\in S^{\rm frozen}_{\ell,1,t}.
\end{aligned}
```

前面六项选择题的下降说明：在这个固定训练模型、这组任务和这条冻结政策下，允许后续轮次重新选择专家，对最终预测有功能贡献。这个总响应包含状态、attention 和 routing score 的连锁变化；它回答的是整条身份政策是否重要。若要定位哪一层、哪一次切换产生收益，还需要更局部的干预；若改为从头训练一个永不切换的模型，又是在问另一件事。

MMLU 还需按其原有汇总方式计算：先在每个 subject 内对同一道题比较 normal 与 frozen 的正确性，再平均为该 subject 的准确率差，最后对 57 个 subject 等权平均。下一式保持了这一先题内配对、再学科平均的顺序。

```math
\widehat\Delta_{\rm macro}
=\frac1{57}\sum_{s=1}^{57}\frac1{n_s}\sum_{i=1}^{n_s}
\bigl(c^{\rm normal}_{s,i}-c^{\rm frozen}_{s,i}\bigr).
```

下面的身份冻结算法是 Appendix C 的可执行语义重建；公开仓库中已核验的是 router 计算，尚未定位原实验的独立复跑入口。

来源：[Figs.10–12 与 Appendix C](https://arxiv.org/pdf/2610.23036v1#page=25)；[`TopKRouting.forward/backward`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/moe/router.py)。内积恒等式与反例为本页推导，未改动论文原始结果。

#### 算法 冻结专家身份的配对干预

**输入** 固定 checkpoint、完整选择题评测集、相同 prompt/截断/初态；正常和冻结两个模式。

**输出** 逐题 correct 差；任务 micro accuracy 差；MMLU 57-subject macro 差。

**执行前提** Appendix C 语义重建。新增 route hook 须实现于已核验 TopKRouting 接口；不声称存在作者原始独立复跑脚本。

![算法 冻结专家身份的配对干预](visualize:fde1_bGliZmlsZV9WNjM4eXlZS1ZSQ2NHYk93OGRhNF9n_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 原论文 Appendix C、Fig.12；路由运算对应固定提交 xllm/modules/moe/router.py::TopKRouting。干预同时改变后续 hidden 轨迹，因此结果是整个冻结身份政策的总效应。

## 6 训练与生成的执行顺序

以下均为固定提交中的实际符号。已完成源码逐项阅读；未执行训练、加载大权重或复跑论文分数。

| 实现位置 | 实际职责 | 核验要点 |
| --- | --- | --- |
| train_paper_part1.py / main | 解析 recipe、target、base config，再调用 train.main | --print-config 只展开展示；base 不得覆盖 recipe-owned 字段 |
| xllm/paper_part1/recipes.py / get_recipe、Recipe.optimizer_config、_model_fields | 17 个 recipe 的架构、优化器、循环与精度字段 | arch=transformer 也可能因 loop_times>1 构造成循环模型，不能只按 arch 名判断是否原生 loop |
| xllm/paper_part1/training.py / training_config、stop_step、activated_parameters | 合并配置并按 endpoint 或 TPP 算 stop step | TPP 分母包含 embedding、router、active experts，循环共享权重只计一次 |
| xllm/models/build.py / get_model_cls | 分派 Transformer、LoopedTransformer、Huginn | Part I 未启用 huginn_depth_control |
| xllm/models/looped.py / build_loop_execution_schedule、LoopedTransformer.forward | 把物理层索引排成实际逻辑执行顺序 | 反复调用同一 ModuleList；不是复制四份参数 |
| 同文件 / Huginn.forward、_run_recurrences、_recur、InputInjection.forward | prelude→初始化→递推→coda；精确处理 H/L 与注入 | 每次状态更新写入一次，split module 内 repeat 不额外写入 |
| train.py / forward_backward_window；xllm/optim/optimizer.py / build_optimizer | token/aux 归一化、backward、AdamW 与梯度裁剪 | 辅助项按逻辑 MoE 调用数平均；所有模型参数统一 weight decay |
| xllm/paper_part1/artifacts.py / verify_artifact、load_artifact | 校验 manifest、文件集合、hash、索引、BF16 shard | load_native_model 再严格校验 state_dict 名称和 shape |
| xllm/paper_part1/native_inference.py / load_native_model、generate_native | 原生构模与生成 | 推理关闭训练 fused block/output，使用完整前缀重算 |
| xllm/paper_part1/eval_protocol.py / PAPER_PROTOCOL；evaluator.py / evaluate_native_task | 绑定任务、指标、cap、题数与样本级执行 | 当前 schema_version=2，还包含论文主表之外的 MMLU-Pro，不能将整个 suite 当成原文十任务均值 |
| xllm/paper_part1/export.py / export_checkpoint | 从训练 checkpoint 导出 BF16 artifact | 导出格式不含 optimizer / dataloader state |

[固定提交源码树](https://github.com/ifm-ai/xllm-loop/tree/755c7b3a91081fe0f83500eabd53bd2490b8b745)；[模型实现](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)；[训练配置](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/training.py)；[原生推理](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py)；[评测协议](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/eval_protocol.py)

Part I 的 _begin_forward 明确拒绝非空 KV cache；generate_native 每生成一个 token 都对当前完整前缀重新 forward。因此，当前发布路径尚未实现分层分轮的 KV serving。输出头每次只读末轮经 coda 的状态，代码也没有可直接监督或调用的通用早退头。

完整前缀重算还要求初态保持一致。训练时 normal 初态使用 randn / √d；eval 时则对维度、token position、batch row 做哈希，再经 erfinv 构造确定性近似 Gaussian。重新打包 batch、截断输入或改变样本行号，都可能改变该评估初态，缓存等价测试必须固定这些条件。

H/L 训练另有一项随机轨迹约定：forward 先生成的单状态随后不被使用，_run_recurrences 再抽一次随机数初始化 L。源码明确保留两次 draw，以维持发布模型的 seed 轨迹；整理代码时不能无记录地删掉第一次抽样。[looped.py / _begin_forward、initialize_recurrent_state、_hashed_normal_state](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)

### 无缓存生成的成本应沿真实前缀长度求和

考虑单条长度 T₀ 的输入，生成 G 个 token，且没有提前 EOS。以每个完整前缀的投影及 FFN 成本 aDT、attention 成本 bDT² 作为简化模型，则当前 native 路径的成本为下式。a、b 吸收宽度和 kernel 常数，D 是逻辑执行深度；这是复杂度记账，不是实测 latency 拟合。

```math
\begin{aligned}
C_{\rm full}&\approx D\left(a\sum_{g=0}^{G-1}(T_0+g)
+b\sum_{g=0}^{G-1}(T_0+g)^2\right),\\
\sum_{g=0}^{G-1}(T_0+g)&=GT_0+\frac{G(G-1)}2,\\
\sum_{g=0}^{G-1}(T_0+g)^2
&=GT_0^2+T_0G(G-1)
+\frac{G(G-1)(2G-1)}6.
\end{aligned}
```

因此“只生成一个新 token”在该实现中仍会重做旧前缀所有轮次。KV 改造要证明缓存路径与完整重算路径的 logits 在给定数值容差内一致，不能只比较共享的投影权重。必要条件还包括同一 logical layer/iteration、位置编码、segment mask 以及初态。

eval 初态是 ξ(b,t,j)，依赖 batch row、token 位置和 hidden 维度；并非仅依赖样本的内容。配对评测必须固定 row 与截断方式。若将样本重新排序后只逆置最终输出，初态已改变，不能把输出差一律归因于缓存错误。

```math
\xi_{b,t,j}=\operatorname{HashNormal}(b,t,j),\qquad
\max_{t,v}|z^{\rm cache}_{t,v}-z^{\rm full}_{t,v}|\le\varepsilon_{\rm num}
```

后一式是待实现缓存路径的验收条件；现有发布路径仍明确拒绝非空 cache。

源码对应：[`generate_native`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py)；[`_begin_forward`、`_hashed_normal_state`、`initialize_recurrent_state`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)。

### 返回截断不等于停止执行

固定发布函数 generate_native 不检查 EOS，也不接收任务 stop strings；它完成固定位置循环后，才按每条 prompt 的长度切出 max_gen_len 个 token。evaluator._generation_result 随后调用 task.postprocess：例如 MBPP 再按 [DONE] 截断，BBH 和数学任务也在 decode/postprocess 中处理 EOS 或分隔字符串。因此 Appendix B 的“在 EOS 或任务标记处停止”描述保留答案的语义，不能直接作为当前发布生成器已经提前停止 forward 的证据。[generate_native](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py)；[_generation_result](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/evaluator.py)；[MBPP postprocess](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/tasks/mbpp.py)

| 执行阶段 | 当前发布函数的行为 | 对计费与评测的含义 |
| --- | --- | --- |
| prompt 处理 | 加 BOS 后按 max_prompt_len 保留末尾；用 token≠pad_id 标记已给定位置 | 长度要按截断后实际 token 数计；prompt 内若含 pad_id，需单独核对该掩码语义 |
| 共同生成区间 | 从最短 prompt 长度开始，到最长 prompt 长度加 G 之前结束 | 异长 batch 的循环次数可以大于 G |
| 每个位置 | 对所有行的完整当前前缀 forward；再用 torch.where 保留尚未结束的原 prompt token | 已给定的 prompt 位置也参与该次模型调用与候选 token 计算 |
| 返回 | 每行只取自身 prompt 之后的 G 个 token；remove_prompts=False 时返回 prompt 加这 G 个 token | 短行多算的尾部被丢弃；默认返回长度固定为 G |
| 任务后处理 | 全部生成完成后才 decode、截断和评分 | 显示答案变短不意味着省下后续 forward |

设 batch 大小为 B，截断后的非空 prompt 长度为 Tᵢ，生成上限为 G>0；假定给定 prompt 内无 pad_id，以确保 prompt_mask 与真实给定区域一致。固定函数没有中途异常时，调用次数和每条短行额外生成但未返回的 token 数如下。这里的次数是源码控制流的精确计数。

```math
\begin{aligned}
T_{\min}&=\min_i T_i,\qquad T_{\max}=\max_i T_i,\\
N_{\mathrm{forward}}&=G+T_{\max}-T_{\min},\\
N_{\mathrm{discard},i}&=T_{\max}-T_i.
\end{aligned}
```

沿用上节每行完整前缀的简化成本 aDT+bDT²，当前整批路径的操作记账为下式。它不是 GPU latency 测量，也没有计入 batch 对 kernel 效率的影响。

```math
C_{\mathrm{batch}}
\approx BD\sum_{c=T_{\min}}^{T_{\max}+G-1}
\left(ac+bc^2\right).
```

两个流程小例可直接区分“调用”和“返回”：单条 T=2、G=4，即便每次 argmax 都是 EOS，仍调用四次，前缀长度依次为 2、3、4、5，返回四个 EOS ID；任务 decode 后可变成空文本。异长 T=(2,5)、G=3 时，调用六次，前缀长度为 2,…,7，但每行仅返回三个 token；短行最后三次写入的 token 被丢弃。这两个结果通过执行固定 generate_native 函数的控制流、以 NumPy 张量与确定性返回值替代模型计算得到；未运行 Transformer 或推断硬件时延。

原生 evaluator 逐样本执行生成，且 _generation_result 要求恰好返回一条 generation，所以异长 batch 的额外调用属于通用 API 边界，不能归因于论文逐题评测。单样本固定生成上限与随后文本截断的区别仍然适用。实现真正的 EOS 早停时，应在生成循环中记录完成状态并实际终止或跳过计算；只截断返回值或设置显示长度不会省下模型调用。若进一步压缩 batch，须保持前文 hashed normal 初态所依赖的稳定行身份，否则可能改变尚未完成样本的预测函数。[evaluator._evaluate_items 与 _generation_result](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/evaluator.py)

## 7 论文实验与公开发布的差异

| 核对项 | 原论文 | 当前官方 release | 对后续研究的含义 |
| --- | --- | --- | --- |
| 研究范围 | 报告 dense Q1–Q3 与完整 MoE 架构比较 | 17 recipes / 25 checkpoints，含 FF 对照与额外消融 | 不是“17 个都属于论文已完整评测的原生 loop base” |
| 独立 H/L 模块 | p.13 说明仍在评估，未纳入现有比较 | dense_split_hl 与 dense_split_hl_x2 已提供 | 必须单列新增发布变体，不能给它们套用 Table 4 的分数 |
| Split H/L 执行深度 | 主文 shared H/L 是 112 次块执行 | split 把 12 层分成 6 层 H/6 层 L；repeat=1 为 64 次，repeat=2 为 112 次 | 两者 checkpoint tensor layout 相同仍不等于计算图或 FLOPs 相同 |
| 学习率 | 报告从三点 sweep 取最佳 | release Recipe 固定 6e-4 | 用该固定配方复跑，不代表重新完成论文 LR sweep |
| 训练/发布环境 | 论文 PyTorch 2.8.0+cu128 与旧 xLLM | 发布说明检验 2.11.0+cu128、FlashAttention 3.0.0 | 数值和 batch ordering 可不同 |
| 数据与恢复 | 原训练轨迹未完整随权重公开 | 部分数据不公开；current loader 顺序不同；restore_rng_state=False | 不能承诺重现原逐步轨迹或 bitwise resume |
| 参数/TPP | 主表多用 non-embedding 参数尺度 | target 的 activated_parameters 包含 embedding/head 等 | 论文“460 TPP”不能直接当 release 的 460tpp target |
| 权重文件 | 论文分数所对应训练产物 | BF16、config、tokenizer、manifest，无优化器或 dataloader 状态 | 可作为新后训练的初始化；不能原样续接原 optimizer trajectory |
| 路由探针 | Appendix C 描述身份冻结实验 | 本次未定位到与该实验对应的独立公开运行入口 | 已验证路由函数，未验证此实验的端到端复跑脚本 |

split 的深度数字来自已核验执行顺序的直接计算：prelude/coda 共 16 次，加 2×(3+1) 次 H/L 更新，每次 6×repeat 层，得到 16+48×repeat。这是源码导出的计数，不能伪装为原论文单列的测量值。[原论文 p.13](https://arxiv.org/pdf/2610.23036v1#page=13)；[release README](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/README.md)；[Huginn.__init__ 与 _run_recurrences](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)

## 发布检查点的不可变版本与清单

按固定release README列出的25个仓库逐项读取Hub元数据，已记录全部完整revision；它们属于17个recipe，包含非循环对照和论文之外新增H/L，不是25个同种原生Base。表中显示短SHA，链接使用完整不可变版本；此处固定下载身份，尚未下载权重或运行native loader。

| recipe／身份 | 336b revision | 500b revision |
| --- | --- | --- |
| dense-d28（非循环对照） | [0ae1ce77](https://huggingface.co/IFM/LoopedLM-P1-dense-d28-336b/tree/0ae1ce772b7126a31de22f1daab6ee12aac8ae99) | — |
| dense-d112（非循环对照） | [6792a315](https://huggingface.co/IFM/LoopedLM-P1-dense-d112-336b/tree/6792a315287b01127b9b6d509a66cdd6bb06f939) | [ce5e8e62](https://huggingface.co/IFM/LoopedLM-P1-dense-d112-500b/tree/ce5e8e62c2c0192a17347fc82e748be018ea3925) |
| dense-ouro | [e87db19b](https://huggingface.co/IFM/LoopedLM-P1-dense-ouro-336b/tree/e87db19b5e9d5bc188ca2230e0299f260ba579ee) | — |
| dense-middle | [38a1464c](https://huggingface.co/IFM/LoopedLM-P1-dense-middle-336b/tree/38a1464ce82cbb68e3d088fc2b87e5b2c5427482) | — |
| dense-ouro-raw-injection | [a218b790](https://huggingface.co/IFM/LoopedLM-P1-dense-ouro-raw-injection-336b/tree/a218b790436c34985202a8ccb80c20ec42775982) | — |
| dense-middle-prelude-injection | [dfaf77f4](https://huggingface.co/IFM/LoopedLM-P1-dense-middle-prelude-injection-336b/tree/dfaf77f443c9887cf5fae13bbbd88848df8d274e) | — |
| dense-huginn | [e8778059](https://huggingface.co/IFM/LoopedLM-P1-dense-huginn-336b/tree/e87780591bf62f453299f5056bb7f51884d84b84) | [567818c8](https://huggingface.co/IFM/LoopedLM-P1-dense-huginn-500b/tree/567818c8e6ec0fb0802d3527e0283520d3e63844) |
| dense-shared-hl | [38302825](https://huggingface.co/IFM/LoopedLM-P1-dense-shared-hl-336b/tree/38302825459491e1889cd3b6482211b6440e48dc) | — |
| dense-split-hl（额外发布H/L） | [2cdd3ba8](https://huggingface.co/IFM/LoopedLM-P1-dense-split-hl-336b/tree/2cdd3ba8bcc86e93e7e8d8c01ab9b91a0c8180a5) | — |
| dense-split-hl-x2（额外发布H/L） | [53dd852f](https://huggingface.co/IFM/LoopedLM-P1-dense-split-hl-x2-336b/tree/53dd852f9d5bdacd80ad75cd80bbb4302b29d6f4) | — |
| moe-ff112（非循环对照） | [8e3d306e](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-336b/tree/8e3d306e67db0f627423f91aa7cadfebc154ee5d) | [cffe259b](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-500b/tree/cffe259bd2109d02d04540c5f9ae1034fa84d2c9) |
| moe-ff112-prelude-injection（非循环对照） | [39ecb399](https://huggingface.co/IFM/LoopedLM-P1-moe-ff112-prelude-injection-336b/tree/39ecb399fec15c1db95cbfa335ab03375bc32810) | — |
| moe-ouro | [14186ab2](https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-336b/tree/14186ab2cc6d12bf1beed10efa140f68e930b68d) | [f0e9c8af](https://huggingface.co/IFM/LoopedLM-P1-moe-ouro-500b/tree/f0e9c8af22c10f34ac716c52495ceb00525f3556) |
| moe-middle | [068ab150](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-336b/tree/068ab1500dd280112a1907f4a18a368674647fec) | [42842590](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-500b/tree/428425900ffd38465ea3de584b14cff2fc682d48) |
| moe-middle-input-injection | [162f8675](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-input-injection-336b/tree/162f8675f1df99bb5a65036a96bcc5390305148b) | [8ea8d694](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-input-injection-500b/tree/8ea8d694864dc0a32a8d4a26c4e40808bc9a6a7a) |
| moe-huginn | [070bb945](https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-336b/tree/070bb945e91c37e1f4641a69217407552e99049d) | [f1321d52](https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b/tree/f1321d5272a11348d5093ad7ab1d310eb9de2e11) |
| moe-shared-hl | [174aacad](https://huggingface.co/IFM/LoopedLM-P1-moe-shared-hl-336b/tree/174aacad5b8d1f5cd512fe5461e2774bb9da5755) | [821826cd](https://huggingface.co/IFM/LoopedLM-P1-moe-shared-hl-500b/tree/821826cd5ffec7c70a6922f2b4b009977351755b) |

两个矩阵候选另实际读取config、artifact_manifest和Safetensors index，共6份小文件并校验Git blob／SHA-256。dense-huginn-336b为e87780591bf62f453299f5056bb7f51884d84b84，moe-huginn-500b为f1321d5272a11348d5093ad7ab1d310eb9de2e11；分别声明258／286个state tensor，1／4个分片，index payload bytes为2,998,060,032／17,566,503,936。这些是序列化张量负载口径，不是非embedding参数量或真实GPU峰值显存。

两份config均为8层prelude、12层共享核重复8次、8层coda，正常执行112次块调用；MoE候选为25专家、top-2。保存的fused/recompute开关不等于native推理最终生效配置，仍按load_native_model的覆盖路径核对。native权重布局不能直接交给tomg-group Huginn的HF类加载。

完整加载验证从固定 repo／revision 开始：下载 manifest 列出的全部权重、config、tokenizer 及文档后，先运行固定源码的 verify_artifact，再严格比对 tensor 名称／shape／dtype，最后加载并比较前缀与生成结果。当前已完成的范围是小文件的真实字节核验，以及权重 manifest／Hub 声明之间的一致性检查；完整 verify_artifact、严格张量加载和生成对照仍未执行。[固定release说明](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/README.md) · [原生工件校验与加载](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/artifacts.py)

[循环研究\_执行契约与发布版本复核\_20261108](library-file:fde1_bGliZmlsZV9EZjhCejk0T0RYMHFXRTVWMUJTWXJR_FileDrive_45288dfe06608191af94334ba8699511)

核对包包含25个固定revision、两个候选的6个实际小文件、Ouro／Loopie／TaH2原函数探针、八页写回记录及目录保留检查。独立解压后的H1–H7全部通过；不包含模型权重或真实模型／GPU实验。文件SHA256：`f1e5f834594aabe66068bf003c33dcd8770e60c66fada30744eceedf174013fe`。

### 相同权重文件不等于相同训练状态

至少区分三个对象：前向函数所需的权重与结构、一次 optimizer update 所需的优化器和调度状态、重放同一数据轨迹所需的随机数及采样器状态。配置中的 schedule 也是前向函数的一部分，所以 tensor layout 相同的 split 与 split×2 仍可表示不同函数。

```math
\begin{aligned}
\mathcal F&=(\theta,\text{config},\text{tokenizer},\sigma),\\
\mathcal U&=(\mathcal F,\text{optimizer moments},\text{scheduler},\text{step}),\\
\mathcal R&=(\mathcal U,\text{RNG states},\text{data cursor},\text{packing state}).
\end{aligned}
```

发布 BF16 权重可构造新的后训练初始函数；仅有这些权重不能逆推出 optimizer moments 或原数据 cursor。H/L 中被丢弃的第一次 normal draw 对前向数值没有直接贡献，却会改变后续 draw 和整个训练随机序列。因此做“无功能变化”的重构时，应分别检验前向等价、梯度等价和随机轨迹等价，不能用一次 logits 相同替代全部三者。

模型加载契约还包括 `strict=True` 权重键核验、tokenizer/model vocabulary 一致、禁用训练专用 fused block，以及明确数值精度。以下算法依据固定提交的共享 H/L 分支重建，保留这次被丢弃的抽样；它没有将 Part II 的随机深度或截断梯度混入 Part I。

来源：[`load_native_model`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py)；[`Huginn.forward` 与 `_run_recurrences`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)；[release 的工件说明](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/README.md)。

#### 算法 共享 H/L 的完整前向与微批次目标

**输入** token、对齐目标、mask；共享 8–12–8 配方；两次 H 周期、每周期三次 L 更新。

**输出** 末轮 logits 或 token loss、MoE 辅助项及 112 次层调用的执行记录。

**执行前提** 固定提交的 Part I shared_hl；cache=None；有效目标数大于零。所有循环保留计算图；checkpoint 仅重算并保存 RNG 语义。

![算法 共享 H/L 的完整前向与微批次目标](visualize:fde1_bGliZmlsZV9wMERfNHRkZ053LUFuVjhVa05wNXZR_FileDrive_45288dfe06608191af94334ba8699511)

**实现对应** 由固定提交 looped.py 的 Huginn.forward、_run_recurrences、_run_layers、_finish_forward 与 train.py 重建。训练 normal 初态消耗两次独立 draw；eval 使用源码确定性 hashed normal。不是 Part II 的截断深度算法。

## 8 后训练接口与研究问题

可选基座应先固定 recipe、token endpoint、输入写入、初态和执行 schedule；同族至少保留 Middle / Middle+injection / Huginn 的区别。SFT、偏好训练或 RL 都可以从已发布权重开启新的训练过程，但不能把现有原生生成接口当成已兼容任意 Trainer、PEFT 或高效 rollout engine。

后训练首先要保留共享约束。给共享core添加adapter时，每轮应复用同一组adapter参数；每轮各配一份会改变参数预算和模型结构。若要早退，还需检查中间态经过同一个coda/head后是否可读，并学习相应停止策略；固定末轮训练没有提供这种校准。

MoE还要同时记录expert load、离散专家身份切换和mixture权重，观察后训练是否把有功能贡献的迭代路由压成固定选择。由原文权衡出发，一个具体问题是：怎样保留prompt和规格的访问收益，同时减少持续注入对数学状态演化的干扰？效果应由质量与实际延迟曲线检验。这些是论文及实现导出的研究接入点，尚无已证明有效的后训练方案。

### 后训练先固定共享约束，再定义可读出状态

对同一个 core 加一组 adapter ψ 时，参数只储存一次，但梯度来自每次执行。若每轮新增独立 ψᵣ，新增参数约增至 R 倍，并已改变原模型的权重共享约束。

```math
\nabla_{\psi}L
=\sum_{r=1}^{R}
\left(\frac{\partial G_{\theta,\psi}(Z_{r-1};E)}{\partial\psi}\right)^{\!\top}
\bar Z_r,
```

其中状态伴随量已包含该轮之后所有依赖；不能只对最后一轮 adapter 求导。对 shared H/L，原训练读出的是完成 H 更新后的状态，分别发生在第 4 与第 8 次 core 调用。将 L 更新后的状态直接送入 coda 是新增读出分布，不能与“提前一个 H 周期停止”视为同一政策。

可用固定深度权重 q 为指定合法出口训练可读出性，下面给出一种后训练目标接口。teacher 是固定的末轮模型，所有项只对有效目标计分；该目标是研究设计，并非论文已做的训练。

```math
\begin{aligned}
p_{r,\psi}&=\operatorname{softmax}(\operatorname{Head}(C(Z^{\rm exit}_{r,\psi}))),\\
L_{\rm read}&=\mathbb E\sum_{r\in\mathcal E}q_r
\left[-\log p_{r,\psi}(y)
+\lambda\operatorname{KL}(p_R^{\rm teacher}\Vert p_{r,\psi})\right].
\end{aligned}
```

出口集合、调用 schedule、初态、tokenizer 和 task protocol 应固定在实验配置中；真正省下的计算应按停止后的实际 core/coda/head 调用计数验证。当前末轮 head 与无缓存生成接口不会自动完成中间态校准或按需跳过执行。

依据：[`Huginn._run_recurrences`、`_finish_forward`](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py) 与[论文局限](https://arxiv.org/pdf/2610.23036v1#page=19)。目标与验收条件为本页提出的后训练接口。

## H/L 深度监督必须按可读出的宏步定义

TaH2 每次完整 backbone 执行后获得词表分布。将相邻深度标签用于 H/L，先确定读出依赖哪个状态。对仅由 H 读出的变体，若一个宏步包含若干 L 更新及一次 H 更新，应比较宏步端点：

```math
\begin{aligned}
(H_{k+1},L_{k+1})&=\Phi_\theta(H_k,L_k,e),\\
q_k&=\operatorname{softmax}(R_\theta(H_k)),\\
\delta_k&=-\log q_k(y)+\log q_{k+1}(y).
\end{aligned}
```

L 子步中 H 不变时，立即读出看不到 L 为后续 H 准备的信息。最小反例：初态 H=L=0，目标 1，损失 (H−1)²；先令 L←1，损失仍为 1，再令 H←L，损失变为 0。完整宏步有收益，第一子步的即时读出收益却为零。

若允许 L 子步退出，需要新增并训练读取 (H,L) 的接口，或定义足够观察跨度的监督；这改变了原来的读出契约。有限前瞻只覆盖所选窗口，不能替代更远未来的收益估计。

停止单位记录为完成的宏步数及各类子调用数。macro-step 成本包括全部 L/H 调用、注入、router 和读出；将其换算到同一成本口径后才与 TaH2 整栈迭代比较。合法退出点继续依固定代码的 _run_recurrences 与 _finish_forward 确定。

来源：[xllm/models/looped.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py)；[TaH2 §4.2](https://arxiv.org/html/2609.35748v1#S4.SS2)。反例说明观察跨度，不是 H/L 模型性能实验。

# 源码文件与关键函数

来源：[github.com/ifm-ai/xllm-loop](https://github.com/ifm-ai/xllm-loop)；固定 revision：`755c7b3a91081fe0f83500eabd53bd2490b8b745`。

| 文件直达 | 关键符号 | 已核职责与细节 |
| --- | --- | --- |
| [train_paper_part1.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train_paper_part1.py) | main | 训练CLI入口；get_recipe→training_config→train.main；base不得覆盖recipe-owned字段 |
| [xllm/paper_part1/recipes.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/recipes.py) | Recipe.optimizer_config; _model_fields; get_recipe | 精确配方；17 recipes；6e-4、200warmup、cosine119210；BF16；MoE dot aux0.001、zloss0 |
| [xllm/paper_part1/training.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/training.py) | training_config; stop_step; activated_parameters | 配置与预算；512×8192，全data parallel；含embedding的active TPP；336b=80000，500b=119210 |
| [xllm/models/build.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/build.py) | get_model_cls | 真实模型分派；arch=huginn选Huginn；arch=transformer且loop>1/注入选LoopedTransformer |
| [xllm/models/looped.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py) | build_loop_execution_schedule; LoopedTransformer.forward | 共享调度；物理索引复用；Ouro全栈循环；各逻辑调用保留独立激活 |
| [xllm/models/looped.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py) | InputInjection.__init__; _decayed_state_and_update; forward | 对角注入；dt=softplus，a=exp；decay=exp(-dt*a)；W初始identity；全部weight decay |
| [xllm/models/looped.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/looped.py) | Huginn.forward; initialize_recurrent_state; _hashed_normal_state; _run_recurrences; _recur | single/shared-HL/split-HL与初始化；prelude→state→core→coda；H/L加和在注入前；repeat每次注入一次；eval deterministic hashed state |
| [xllm/models/transformer.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/transformer.py) | Transformer.num_parameters | 参数口径；输入embedding+输出head均计入；MoE total/activated分开；nonembedding论文口径不等于训练TPP |
| [xllm/modules/moe/router.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/moe/router.py) | TopKRouting.forward; calc_probs | 路由与auxiliary；sigmoid→top2→入选归一化；全expert归一化用于dot负载loss |
| [train.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/train.py) | forward_backward_window; main | 训练loss与梯度；token numerator按全累积窗口分母；aux按microbatch和实际MoE调用数平均；PartI full graph；日志另按microbatch平均；scale分支与clip/step/zero_grad顺序 |
| [xllm/optim/optimizer.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/optim/optimizer.py) | build_optimizer; rescale_grads | AdamW直接model.parameters()统一weight decay；rescale_grads在scale=1时直接返回，否则逐梯度除scale |
| [release/paper-part1/data.md](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/data.md) |  | 数据与tokenizer协议；42来源配比；部分prepared来源无公开下载；指定tokenizer revision与BOS处理 |
| [release/paper-part1/common-base.example.json](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/common-base.example.json) |  | caller-owned配置；数据/日志/checkpoint/tokenizer/gradient accumulation；bestfit与buffer512 |
| [xllm/paper_part1/artifacts.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/artifacts.py) | verify_artifact; load_artifact | 工件校验；文件集合/hash/index/config和BF16shard校验 |
| [xllm/paper_part1/native_inference.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py) | load_native_model; generate_native | 原生加载与生成；strict state dict；关闭训练fused实现；full-prefix recompute；无cache |
| [xllm/paper_part1/eval_protocol.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/eval_protocol.py) | PAPER_PROTOCOL; paper_reported_metrics | 评测契约；schema2，shots在tasks、caps/metrics/cardinality固定；包含MMLU-Pro扩展 |
| [xllm/paper_part1/eval_native.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/eval_native.py) | evaluate_native_suite; _evaluate_native_suite_from_root | 评测链；模型/评测器/suite/data hash；加载一次后逐task执行 |
| [xllm/paper_part1/evaluator.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/evaluator.py) | evaluate_native_task; _evaluate_items | 样本级评测；调用原生生成或token loss；结果resume与身份约束 |
| [xllm/paper_part1/export.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/export.py) | export_checkpoint | 导出权重；训练checkpoint→BF16 artifact；不含optimizer/dataloader状态 |
| [xllm/utils.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/utils.py) | get_init_fn | gaussian 实为 ±3σ 截断正态；未指定 std 时取 1/√dim |
| [xllm/modules/residual/base.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/modules/residual/base.py) | BaseResidual.forward | 默认返回 h+residual；rc_id 不构成深度缩放 |
| [xllm/distributed/fully_sharded_data_parallel.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/distributed/fully_sharded_data_parallel.py) | FullyShardedDataParallel.clip_grad_norm_ | 根FSDP按参数对象去重；FP32本地范数、跨分片归约、未分片只计一次；返回裁剪前norm，εclip=1e−6 |
| [xllm/data/dataset_streamer/feature_builder/feature_builder.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/dataset_streamer/feature_builder/feature_builder.py) | DefaultFeatureBuilder.build; ChatTemplateFeatureBuilder.build | 预训练文本加 BOS/EOS 且 target_mask=None；SFT 从 assistant labels 生成逐 token mask，再交 packer 对齐 |
| [xllm/data/buffer_assembler/packer/packer.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/data/buffer_assembler/packer/packer.py) | BestFitPacker.pack; _slice_chunk; _flatten_chunks | bin 容量 S+1，长文步长 S；续块不补 BOS；连接后对齐 x/y/mask；边界目标屏蔽仍是注释 TODO |
| [xllm/models/transformer.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/models/transformer.py) | Transformer.get_segment_arg; TransformerOutputLayer.forward | attention 仅按 BOS 构造 segment；输出头直接使用已对齐 targets，不再次 shift |
| [tests/data/test_packer_targets.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/tests/data/test_packer_targets.py) | test_best_fit_predicts_bos_across_document_boundaries; test_best_fit_sets_sft_masked_targets_to_ignore_index | 仓库原有边界目标和 SFT mask 测试，支持当前 packer 契约；不等同于历史原论文数据轨迹 |
| [native_inference.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/native_inference.py) 与 [evaluator.py](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/xllm/paper_part1/evaluator.py) | generate_native; _generation_result | 共同位置区间完整重算，无 EOS/stop-string break；每行返回 G 个 token 后才 task.postprocess；evaluator 单样本生成与通用异长 batch 分开 |

# 逐项覆盖核对

核验日期：2026 年 11 月 8 日（北京时间）。各行分别记录原文依据、实现对应和当前证据状态；具体运行与复现以实际验收记录为准。

| 检查项 | 原文／来源位置 | 实现对应 | 当前核对状态 |
| --- | --- | --- | --- |
| 研究目标与模型选择矩阵 | §1–2、Table 1，pp.2–5 | recipes.py 注册17配方；get_model_cls区分FF与原生loop；release列25 checkpoints | 论文与官方源码已核对，FF对照单列 |
| 模型几何和参数口径 | §3.1、§4.1、Table 6，pp.5、15、24 | _model_fields；Transformer.num_parameters；Huginn.execution_schedule | 已核对；730M/8B为non-embedding尺度，与release TPP含embedding口径区分 |
| 对角输入写入及稳定性边界 | §3.4.2，pp.8–10 | InputInjection.__init__/_decayed_state_and_update/forward；dt与a正参数化、W identity init | 已落实完整公式、初始化与Jacobian；未将通道衰减误写成整体收缩证明 |
| Q1 topology envelope | Table 2、Fig.5，pp.7–8 | dense_ouro与dense_middle；28×4与8+12×8+8 | 主结果原表图像核验；覆盖收益及代码任务退步、四训练预算 |
| Q2 input injection | Table 3、Fig.6，pp.9–10 | dense_ouro_raw_injection / dense_middle_prelude_injection；每轮写入源分别为embedding/prelude | 原表图像核验；上下文与数学权衡明确 |
| Q3 random initialization | Table 4、Fig.8，pp.12–13 | initialize_recurrent_state；训练N(0,1/d)，eval _hashed_normal_state | 论文与实现差异明确；未将随机初态判作必需 |
| Q3 shared H/L与新版split H/L | §3.4.3、Fig.7、Table 4，pp.11–13 | Huginn._run_recurrences；shared_hl/split_hl、huginn_split_module_repeats | 已核对；论文结论仅shared-module；新增split不套用Table4分数，源码计数64/112分开 |
| 训练目标与反向路径 | §3.2、Appendix A，pp.6、24 | PartI forward全图；train.forward_backward_window；TopKRouting dot auxiliary；num_aux_loss_layers | 已落实实际normalization、0.001 auxiliary、无router bias与zloss；不套用PartII截断 |
| 优化器与训练预算 | §3.2、Appendix A，pp.6、24 | Recipe.optimizer_config；training_config/stop_step；80k/119210停止、同119210 scheduler | 已核对；论文LR sweep与release固定6e-4区分 |
| 数据/tokenizer/恢复条件 | §3.2；release/paper-part1/data.md | 42 sources、bestfit512、materialize-tokenizer.py移除自动BOS；restore_rng_state=False | 部分数据与旧loader缺口保留；不声称原轨迹复现 |
| MoE主比较与FF参照 | Table 5、Fig.13、§4.4，pp.15–18 | moe_ouro/moe_huginn/moe_ff112；25 top2 experts, width2432 | Table5全部10任务已落地且图像核验；匹配compute与不匹配resident容量明确 |
| 负载、专家切换、冻结干预 | Figs.10–12、Appendix C，pp.16–17、25 | TopKRouting score/select/normalize已核对；未定位独立冻结干预运行入口 | 观测与因果干预分开；49.4% probe与MMLU macro/micro口径保留 |
| native artifact与KV推理 | release README / native_inference.py | verify_artifact/load_artifact/load_native_model/generate_native；_begin_forward拒绝cache | 已核对源码，无KV且完整前缀重算；未运行模型 |
| 评测协议和范围 | Table 7、Appendix B，pp.24–25 | PAPER_PROTOCOL schema2；evaluate_native_suite/evaluate_native_task | 已核对shots/caps/metrics；MATH Q3提示变体、MMLU-Pro新增均明确 |
| 后训练接口 | §5局限与本页源码导出分析 | 共享core adapter、末轮coda/head、route diagnostics、初始化状态与新optimizer | 研究接入点已列；未声称SFT/RL/早退/serving已实现 |
| 初始化的真实分布 | release recipes；utils.py；Transformer 构造 | get_init_fn 使用 ±3σ 截断正态；主体 .02，embedding/head 回退 1/√d；BaseResidual 不按 R 缩放 | 函数构造路径与截断方差已核；权重初始化和 recurrent-state 初始化分开 |
| 共享梯度与 AdamW 状态 | Part I 完整反向；train.py；optim/optimizer.py | 多次调用先累积梯度，再全局裁剪、还原 scale、更新同一组 moments；weight decay 每 optimizer step 一次 | 链式求和有限差分误差约 1.5e−11；独立 Adam 后平均不等价的反例已验证 |
| Q2 的机制识别边界 | §3.4.2、Table 3，pp.8–10 | InputInjection 同时改变 state decay 与 input write；跨族还改变 source 和写入次数 | 原表对角差保留；混合角点为新增研究控制，未冒充原文实验 |
| Part I 更新控制流与全局裁剪 | release recipes 与固定源码 | BF16、FP32 reduce-scatter、MP=CP=1、loss_rescaling=False；根 FSDP clip→AdamW→scheduler→zero_grad→hook | 全局分片范数与每 rank 独立裁剪的区别已核；未启用动态分支单列，未归因于论文实验 |
| 优化目标与训练日志聚合 | train.py::forward_backward_window 与 optim_metrics | 有效 token 加权反向；日志按 microbatch 均值；auxiliary 单列 | 协方差恒等式与 1.3/2.5 反例已核；含 ε 的 DP 归约公式一致，不改写下游分数 |
| Dense Q1–Q3 完整任务与原表差值 | Tables 2–4，pp.7、9、12；Table 7 | 40 行对照、150 个数值逐项核原表；Δ 原值与显示分数相减分开 | 补齐全部十任务、D28/D112 与所有组件端点；0.01 差异与独立舍入相容，不覆盖原表 Δ |
| Packing、监督对齐与片段可见域 | 发布示例、recipes.py、data/packer、looped.py、transformer.py | Part I multi_segments=True；示例 bestfit；目标在 packing 后 shift；segment 仅识别 BOS | 直接执行固定 packer 的完整文档、长文续块和 SFT 三个小例；续块无 BOS 的跨文档可见域已明确；未推测历史分数影响 |
| MoE 与 feedforward gap 的完整原图证据 | §4.4、Fig.13，pp.16、18 | 无新增训练运行；按原图分别保留九任务共同端点与七任务四预算 | 全部图中值核对；8/9 缩小、ARC-C 反例、336B 七任务与500B 四任务平均口径分别保留 |
| native 生成次数与答案截断 | native_inference.py::generate_native；evaluator.py::_generation_result；task.postprocess | 固定位置循环无 EOS break；异长 batch 为 G+Tmax−Tmin 次调用，每行仍只返回 G 个 token | 固定函数 AST 提取后以 NumPy 替身验证 EOS、异长 batch 和 remove_prompts=False 三例；无 Transformer/GPU 时延推断 |
| 当前release数据源与quota | release data.md；BufferAssembler._refill | 42项正权重和为1；结余额与累计接受token的望远镜恒等式；D2 | 固定源码、100次refill核验；有限长度上界为条件 |
| 数据状态恢复与并行次序 | _tokenize_sources_to_budget_parallel、get_state／set_state | source-order回收、active起点、pending、credit、RNG与pop索引 | 15次串并行及恢复后20条替身序列通过；真实packer／分布式I/O未运行 |
