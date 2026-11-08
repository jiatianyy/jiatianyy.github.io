本课沿保存、加载、导出和自主生成，追踪同一份模型计算。先区分参数格式、数值精度与恢复状态，再检查位置、缓存和执行后端，逐层确认训练端定义在生成端怎样落实。

[返回学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
> **阅读目标：**建立训练端与生成端的状态对照表。先看每个阶段保留了什么，再沿加载分支和生成步骤定位差异。

# 从导出到生成

1. **明确保存内容。** `save_model_tp` 导出骨干、updater 和 decider 的 BF16 权重。启用完整训练状态保存时，DCP 另存 FP32 主权重和优化器状态。[保存路径](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py)

2. **加载计算规则。** 推理端还需要 tokenizer、`tah_config.json`、`input_updater.bin` 和 `iter_decider.bin`。循环深度来自配置，不能从目录名推断。

3. **处理前缀并推进生成。** prefill 建立前缀状态，随后逐 token decode。每一步区分内部执行轮数与追加到文本中的 token 数。

4. **比较执行实现。** 固定导出权重、token 前缀、深度及读出规则，先比较一个位置的分布，再扩大到完整生成。

# 精度、恢复与执行一致性

## 精度与权重版本

所读训练路径保留 FP32 主权重、使用 BF16 计算，生成导出则把权重转换为 BF16。主权重精度、前向计算精度与导出精度分别设置。比较训练端和推理端时，应先固定权重版本，避免把舍入差异归因于循环算法。

## 模型导出与训练恢复

用简化状态映射区分训练恢复与模型导出。`O_k` 为优化器状态，`η_k` 为 scheduler 状态，`D_k` 为采样器与数据游标，`R_k` 为随机状态，`T` 为live tokenizer与chat template，`C` 为最终配置：

```math
\begin{aligned}
S_k&=(\theta_{32},O_k,\eta_k,k,D_k,R_k,T,C),\\
E(S_k)&=(Q_{\rm BF16}(\theta_{32}),T_{\rm export}(T),C_{\rm export}(C,T)).
\end{aligned}
```

导出会舍入且不含完整训练状态，不能由 `E(S_k)` 反演 `S_k`。

存在 DCP 完整训练状态时，该恢复路径加载 FP32 主权重与优化器，并按 `global_step` 推进新建 scheduler；未保存全部 RNG 状态，不能据此声称后续轨迹逐步一致。

实际验收比较同一下一批样本的损失、梯度和参数增量，并核对样本身份。[状态保存与恢复](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L497)

恢复能力首先取决于保存了哪些状态。固定训练器区分以下恢复分支；4B 示例配方设置 `save_only_model=True`，默认不会写出 `optim` 目录中的 DCP 主参数与优化器状态。 [保存与恢复分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L497-L586) · [4B 保存配置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml#L68-L75)

| checkpoint 内容 | 恢复行为 |
| --- | --- |
| 权重与 simple_trainer_state.pt，并有 optim 目录 | 加载 DCP 中 FP32 主权重与优化器；按 global_step 重建 scheduler 进度 |
| 有 trainer state，无 optim | 继续记录中的步数与 scheduler 进度，但使用新优化器和已舍入的 BF16 导出权重 |
| 无 trainer state | 只加载权重，从 step 0 开始 |

恢复入口先检查 `windows_per_epoch` 是否变化，再由 `global_step` 推导起始 epoch 和需要跳过的窗口数。这只能核对窗口数量；数量相同仍可能对应不同的数据内容、顺序或配置。要恢复完整训练轨迹，还需固定这些外部输入，并处理该路径未保存的 RNG 状态。

## 导出还会改写 EOS/PAD 元数据

T 表示训练时正在使用的 tokenizer 及模板。固定版本的 `save_model_tp` 先保存 tokenizer，再检查是否有 chat template。若有模板，它渲染一条空 user 消息，不添加 generation prompt；随后编码，并从后向前寻找最后一个 special ID。找到后，将这个 ID 及对应 token 写入四份已存在 JSON 的 EOS/PAD 字段：`config`、`tokenizer_config`、`special_tokens_map` 和 `generation_config`。这些磁盘文件的修改不会同步改动训练中的 tokenizer 对象或 collator。

因此E不仅舍入权重，也包含明确的元数据变换；它采用的是源码中的模板启发式，不能保证任意自定义模板的最后special token都是正确结束符。缺少模板或未找到special ID时不执行这一改写；未存在的配置文件也不是无条件新建。

MiniSGL scheduler从导出目录加载tokenizer的EOS，并实际用于文本终止。因此应记录live EOS/PAD、导出后四文件的值、重新加载tokenizer的值，以及服务最终采用的停止ID；不能只检查权重相等或模板文本相同。[导出逻辑](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L296-L328) · [scheduler读取与终止](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/scheduler/scheduler.py#L149)

隔离原改写AST的例子中，原live EOS/PAD为999／998，渲染文本的最后special ID为303；四份保存JSON改为303或其token字符串，live对象仍为999／998，其他字段保留。检查使用toy tokenizer与本地JSON，没有加载模型或运行服务。

## 数值差异与路由变化

硬门控会把小数值误差变成执行图变化。设 gate 概率为 `g_v`、统一阈值为 `τ`，参考路由上的最小阈值距离为 `γ`。在此前决策相同、对应输入状态可比较且没有资源覆盖路由时：

```math
\begin{aligned}
a_v&=\mathbf1\{g_v>\tau\},\\
\gamma&=\min_{v\ {\rm visited}}|g_v-\tau|,\qquad
e_g=\max_v|g_v-\widehat g_v|,\\
e_g<\gamma&\ \Longrightarrow\ a_v=\widehat a_v.
\end{aligned}
```

证明是逐节点比较阈值两侧，再按执行顺序归纳。`γ=0` 时没有此保证。先重放固定路由以测数值差异，再开放自主门控，才能定位首个变化来自数值还是执行规则。

# 对照源码

固定版本：TaH `78ddf9a2c25713fced03a736c5746161f6ee590c`。

- [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py)：读 `save_model_tp`、`_save_train_state` 和恢复分支，分别列出模型导出与训练状态。

- [miniSGLang README](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md)：确认加载格式、执行模式与后端支持范围。

- [tah_qwen3.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/models/tah_qwen3.py)：对照训练模型中的 updater、decider 与输出组合。

- [engine.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py)：追踪 prefill、decode 和状态推进。

## 按需补充系统知识

遇到多卡与重计算时，再回到 [tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 和训练循环；用 [PyTorch FSDP2 教程](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)补齐分片、通信与参数生命周期，用 [Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)理解时间和显存记录。实际运行时须另行对齐所装 PyTorch 版本。

# 训练与生成的一致性

| 检查顺序 | 固定条件 | 对照对象与通过依据 |
| --- | --- | --- |
| 输入与状态 | 相同 token、padding、逻辑位置、空缓存或同一前缀 | 有效位置和 KV 可见性相同 |
| 导出与加载 | 同一导出版本及 dtype | 骨干、updater、decider 与循环配置均加载；参数名和形状对应 |
| 固定深度前向 | 关闭随机采样，固定深度及读出 | 逐轮 hidden、目标 token log-probability 与完整小词表参考分布在所定容差内 |
| 自回归缓存 | 相同前缀，每步追加相同 token | 整段前向与 prefill 后逐步 decode 在相同语义下对应 |
| 自主路由 | 同一阈值与门控输入 | 记录概率差异及阈值距离，再解释路由是否改变 |
| 完整生成 | 相同采样设置和随机条件 | 输出、停止原因与深度计数可追溯；微小数值差异可能放大为轨迹差异 |

## DUO 的可见性与执行等价

把内部节点记作 `(b,t,m,ℓ)`，分别表示样本、原始文本位置、循环深度与层；本文深度从 1 开始，代码常从 0 开始。实际执行集合与 DUO 的深度／位置可见性为：

```math
\begin{aligned}
\mathcal E&=\{(b,t,m):1\le m\le d_{b,t}\},\\
\mathcal V(b,t,m)&=\{(b,s,j)\in\mathcal E:s\le t,\ j\le m\}.
\end{aligned}
```

这里的集合规定每层应读取哪些逻辑 KV 条目；压缩数组下标不能代替原始 `t`，内部循环不推进文本位置。prefill 必须保存 prompt 实际执行深度对应的 KV。

**执行等价的条件与证明**

要比较训练端和生成端，先让它们处理同一个固定前缀，并重放同一条实际路由。我们关心的不是两个缓存数组长得是否一样，而是同一个逻辑节点 (b,t,m,ℓ) 是否接收了同样的输入：原始 token embedding、位置编码，以及 DUO 规则允许读取的 KV。

沿这个节点向后追踪：输入状态和运算相同，本层的输出及新增 KV 就相同；进入下一轮时，updater 又从相同的原始 embedding 和末层 hidden 得到相同输入。这样逐层、逐轮推进，便可一直比较到各轮预测和最终混合读出。两后端可以采用不同的执行安排，只要都遵守同一依赖图的合法拓扑序，这个归纳过程就成立。

因此，精确算术下的等价命题要求相同权重、有效 token、位置编码、updater 的原始 embedding 锚点、实际路由、可见 KV 和混合读出，并关闭随机算子。有限精度下则沿同一条计算链声明数值容差并验收。下面的对照算法先固定路由，是为了在执行图尚未分叉时定位数值差异，再检查自主 gate 的行为。

普通 causal 模型的缓存结论不能直接替代这些条件。固定深度只固定了路由；一端取末轮 logits、另一端取多轮混合概率仍然不同。先在共同目标 ID 上比较 target-only 与全分布参考，再拓展到 selected-token 或完整分布。

## 算法 4 训练与生成的执行对照

```math
\begin{aligned}
e_h&=\max_v\|h_v-\widehat h_v\|_\infty,\\
e_p&=\max_{t,a}|\log p_t(a)-\log\widehat p_t(a)|.
\end{aligned}
```

**过程** BackendParity(export, effective\_config, prefixes)

**输入** 模型导出、实际生效配置与固定前缀。

**输出** 固定路由的两端数值差异、自主 gate 对照及采样对照记录。

![算法 4 操作与数学公式](visualize:fde1_bGliZmlsZV9VbVVxWDBlR21qXzRZVTVMZG5SYk1R_FileDrive_45288dfe06608191af94334ba8699511)

这是拟实施的对照算法，尚无两端通过结果。完整回答相同不足以替代数值对照；同随机种子也不足以保证不同调度下随机数与逻辑动作一一对应。

# miniSGLang 的具体入口

从 [tah_qwen3.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/models/tah_qwen3.py) 的 `load_tah_weights`、`compute_iter_embeds` 追踪组件加载和回填，再读 [engine.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py) 的 `forward_batch`、`forward_mixed_tah_dispatch`、`forward_mixed_tah_finalize`，定位活动 token 的执行和收尾。

采样记录分别声明温度、top-k／top-p、EOS、ignore_eos 和最大长度；概率字段注明来自原模型还是采样分布，停止规则按实际后端支持范围记录。prompt 与 completion 深度分列，核对 `usage.prompt_iter_counts` 与 `usage.iter_counts`。[引擎说明](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md)

固定提交的三处执行边界需写进有效配置与 trace：

- **有效 decider：**HF eval 可选择 `eval_iter_decider`，miniSGL 加载 `iter_decider.bin` 并允许运行时覆盖。[HF eval](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py#L582)

- **门控模式：**decode 的 `sample` 模式可用 Bernoulli 覆盖阈值，DUO prefill 此处仍直接使用 threshold gate。[decode](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503) · [prefill](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/tah_duo_prefill.py#L151)

- **资源覆盖：**DUO 页池不足时会把部分继续行强制提前采样。[资源覆盖](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L600)

因此同一 checkpoint 与阈值尚不足以保证相同路由，应拟补记录 `effective_decider / effective_route / forced_sample`。

## 同名读出模式的执行条件

even_mix 说明如何组合已执行分布，但 HF 与 MiniSGL 对“执行到哪里”的选择不同。固定源码的行为如下：

| 模式 | HF TaHForCausalLM | MiniSGL |
| --- | --- | --- |
| stop_prob_mix | 有效 decider 决定实际深度，最后执行轮吸收剩余质量 | DUO 采用 learned decider；decode 可选 threshold 或 sample，并可能资源强制停止 |
| even_mix | 对实际执行的 d 轮等权平均，路由仍由有效 decider 与 override 决定 | decode 忽略 decider，强制执行最大深度 M，再取 M 轮等权平均 |
| causal 与 even_mix 的 prefill | 依模型调用和输入标签条件执行 | 专用 uniform prefill 只让最后一个 prompt 位置循环，临时 KV 随后释放 |
| same_iter | 训练模型有此可见性分支 | loader 明确拒绝该模式 |

```math
\begin{aligned}
q_{\rm HF}^{\rm even}(v)
&=\frac1{d_t}\sum_{m=1}^{d_t}q_{m,t}(v),\\
q_{\rm MiniSGL}^{\rm even}(v)
&=\frac1M\sum_{m=1}^{M}q_{m,t}(v).
\end{aligned}
```

只有实际深度和各深度状态同时对应时，这两式才能作为同一读出比较。官方 uniform 路径搭配 AlwaysIterDecider 与当前 token recurrence，正好提供固定深度；不要将 learned decider 下的任意 HF even_mix 自动视为相同运行模式。[HF 累计与计数归一化](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L567-L682) · [MiniSGL 固定深度覆盖](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L557) · [官方支持路径](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md#L32-L41)

uniform prefill 能只重算最后一个 prompt 位置，是因为 causal 可见性使它读不到更早位置的额外深度；保存临时的同位置 KV 因而足够。DUO 的可见性不满足这个前提。检查分派时还需留意：运行时没有禁止 `duo+even_mix`，而 prefill 先进入 even_mix 分支，仍调用上述 causal helper。因此，这个组合须经修正和对照后，才能纳入 DUO 后端等价性的结论。这里指出的是静态分支的适用边界；官方 README 描述的标准路径仍分别为 DUO learned 和 causal uniform。[分派顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L389-L415) · [uniform prefill](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/tah_duo_prefill.py#L356-L403)

## 从继续候选到本轮采样

DUO stop_prob_mix 的 dispatch 同时构造两种读出：继续分支只把本轮停止质量加入累计，立即采样分支把当前剩余质量全部交给本轮。令 Lambda 为先前深度的 log 累计，R 为进入本轮的剩余质量：

```math
\begin{aligned}
\Lambda_{\rm continue}
&=\operatorname{logaddexp}
\left(\Lambda,\log\max(R(1-g),\epsilon)+\log q_m\right),\\
\Lambda_{\rm sample}
&=\operatorname{logaddexp}
\left(\Lambda,\log\max(R,\epsilon)+\log q_m\right),\\
R_{\rm next}&=Rg.
\end{aligned}
```

引擎先对整批行生成采样候选，finalize 等待继续掩码传回后，只提交停止行的候选。页池不足时，部分原本继续的行改为提交已经算好的 sample 分支，所以其尾部质量仍被吸收。DUO 的各深度 KV 保留到请求结束；causal uniform 的本 token 额外 KV 可在该 token 结束时释放。[dispatch 与 finalize](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L670)

融合图中的实际算子次数与语义上需要的次数也要分开。固定 `_run_tah_capture_body` 路径先对整批计算 updater，再用 `torch.where(iterN_mask, updated, base)` 选输入；因此首轮行也计算了随后被丢弃的 updater 输出。末层后的 decider 同样在图中执行，最大深度约束在图外再应用，所以末次执行仍可能付出 gate 计算。论文按必要调用计的 updater 次数 d−1 与 gate 次数 min(d,M−1)，不能直接代替这个融合图的实际算子账本。[融合图主体](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/graph.py#L209-L246) · [图外深度限制](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L503-L515)

原始 embedding 的存储也有明确边界：prefill 的 `forward_with_hidden` 给骨干传 `embeds.clone()`，因为 fused residual norm 会原地改写其输入；原始 embeds 留给 updater 与 decider。融合 decode 用 torch.where 产生骨干输入，同时保留 base 锚点。删除这些复制或选择操作时，必须重新证明原始锚点没有被后续层覆盖。[prefill 的别名处理](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/models/tah_qwen3.py#L245-L264) · [decode 输入选择](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/graph.py#L209-L246)

在非贪心模式中，为最终继续的行计算后又丢弃采样候选，也会影响随机数与逻辑请求的对应方式；只给两个实现同一 seed，不能保证它们消耗同一随机序列。

### 算法 T4 DUO 生成调度与终止

**输入** 当前请求批次、请求状态池、主与循环 KV 页表、深度上限、有效 gate 模式和文本采样参数。

**输出** 继续内部循环的请求、真正新增的文本 token、实际深度及终止状态。

**前置条件** DUO 与 stop_prob_mix；prefill 已按 DUO 规则建立真实前缀状态；CUDA graph 批大小受支持；内部深度从源码零起始转换到数学一起始记号。

![算法 T4 DUO 生成调度与终止](visualize:fde1_bGliZmlsZV93SVpsOTBKb0RVaVNlY0F0OGVLUmN3_FileDrive_45288dfe06608191af94334ba8699511)

**源码** [`forward_mixed_tah_dispatch` 与 `finalize`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L440-L679) · [`_process_mixed_data`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/scheduler/scheduler.py#L309-L350)

## 资源强停改变多少当前 token 概率

页池不足时，固定 finalize 保留批次中前若干个继续候选，将超额候选提交为本轮采样。它使用已经计算好的停止分支，而不是忽略请求或继续执行。因此服务负载能改变实际停止深度及后续 KV。[强停与批次顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L600-L653)

可以对**同一入态、同一共同前缀链**建立一个条件误差界。先采用温度 1、无截断、未 clamp 的 stop_prob_mix。比较在 d<M 强停的 P_d 与沿同一链继续到某个 D>d 的 P_D。令 A 为前 d−1 轮累计，R_d 为进入 d 的剩余质量，Q 为从 d+1 到 D 的归一化尾部混合：

```math
\begin{aligned}
P_d&=A+R_dq_d,\\
P_D&=A+R_d(1-g_d)q_d+R_{d+1}Q,\\
P_d-P_D&=R_{d+1}(q_d-Q),\\
\operatorname{TV}(P_d,P_D)
&=R_{d+1}\operatorname{TV}(q_d,Q)\le R_{d+1},\\
\operatorname{TV}(p,q)&=\tfrac12\sum_v|p(v)-q(v)|.
\end{aligned}
```

R_{d+1}=R_dg_d，而非单独的本轮 gate。该上界在 q_d 与 Q 支持不相交时达到；例如首轮 g=0.3、q₁=(1,0)、q₂=(0,1)，强停与继续的 TV 恰为 0.3。只知道 gate 接近阈值，不能给出很小的强停误差；已有硬阈值反例正是这一等式的特例。

源码对权重设下限并由 sampler 归一化。固定某一条路线的理想权重 w，若只加入下限 epsilon，令 delta 为增加的总质量，则对某个归一化分布 B：

```math
\begin{aligned}
\delta&=\sum_j(\epsilon-w_j)_+,\\
P^{\rm floor}&=\frac{P+\delta B}{1+\delta},\\
\operatorname{TV}(P^{\rm floor},P)&\le\frac{\delta}{1+\delta}.
\end{aligned}
```

delta=0 时取原分布即可。对强停和继续两条路线分别计算 delta，再用三角不等式把两项加入 R_{d+1}。这个界只覆盖权重下限和归一化；混合缓冲舍入、温度和 top-k／top-p 必须另行分析，不能把结果直接称为实际服务误差上界。[权重下限](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L548-L561) · [三个损失分支](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647)

**从局部概率到任务行为还需要什么。** 若 TV≤eta，每个词的概率差至多 eta；贪心首名相对第二名的间隔严格大于 2eta 时，首名才由此保证不变。P=(0.501,0.499)、Q=(0.499,0.501) 的 TV 仅为 0.002，贪心结果已相反。目标词概率没有正下界时，小 TV 也不保证小 NLL 差：P_y=epsilon、Q_y=epsilon² 时，TV 可趋近 0 而两者 NLL 差趋于无穷。

对于最多 H 个文字动作，令 eta_t∈[0,1] 是对全部共同文字历史统一的非随机上界。只有包含各自真实内部状态后验的条件分布都满足 TV≤eta_t，才可通过逐步最大耦合得到：

```math
\operatorname{TV}(P_{1:H},Q_{1:H})
\le1-\prod_{t=1}^{H}(1-\eta_t)
\le\sum_{t=1}^{H}\eta_t.
```

EOS 后采用相同吸收规则。DUO 强停会改变下一 token 的缓存，所以当前单入态界并不自动提供上述逐历史前提；不能只把本次剩余质量相加就宣称最终正确率损失有界。后训练的预算门控应分别评估当前分布改变、未来状态改变与完整任务收益。

**对应的实现核对。** 每次停止记录原始 gate、进入轮次的 remaining、原始继续候选、真正继续结果及覆盖原因；同时保留请求顺序、可用循环页槽、KV 历史标识和采样参数。固定入态比较 P_d 与 P_D 时，要从共同前缀的只读快照分叉，不能在强停后的缓存上直接补算后续文字再称为同一路径对照。完整服务比较继续见[系统页](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c)。

## 温度作用于哪一个分布

采样器收到的是混合 log 值。先做带温度的 softmax，再应用 top K／top P；正温度 theta 下，未截断的采样分布是混合概率的幂次归一化：

```math
\pi_\vartheta(v)=
\frac{\widetilde q(v)^{1/\vartheta}}
{\sum_u\widetilde q(u)^{1/\vartheta}}.

```

因此“先把每轮 logits 除以温度再混合”通常是不同算法。若 top K／top P 最终允许的支持集为 S，则进一步将这个分布限制到 S 并重新归一化；记录模型原分布的 logp 与实际行为分布的 logp 时必须说明温度及截断支持。全批贪心时源码走 argmax；混合批次中的贪心行则通过温度 10⁻⁶ 进入采样路径，两者不应被默认视为逐位一致。[Sampler](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/sample.py#L25-L88)

### 固定依赖中的 top-k 与 top-p 顺序

TaH2 原文主评测使用 temperature=0.6、top-p=0.95、top-k=20，因此前面的温度 1、无截断强停界需要继续经过采样变换，才能用于理解该评测设置。固定仓库 `.[tah2]` 锁定 `flashinfer-python==0.6.4`；`sample_from_probs_impl` 同时收到 k、p 时不传 `filter_apply_order`。该版本的默认值是 `top_k_first`：先调用 `top_k_renorm_probs`，再对归一化结果调用 `top_p_sampling_from_probs`。这里核查的是 v0.6.4 对应提交 `f1e6fdcb8f65104047697f022b5d055ef022d763`，不以滚动更新的 API 网页代替版本证据。[原文 §5.1](https://arxiv.org/html/2609.35748v1#S5.SS1) · [依赖锁定](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/pyproject.toml) · [TaH 采样分派](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/sample.py) · [FlashInfer 固定实现](https://github.com/flashinfer-ai/flashinfer/blob/f1e6fdcb8f65104047697f022b5d055ef022d763/flashinfer/sampling.py)

令 z 是引擎实际交给 sampler 的一行有限 log 值，已包含当前路线的权重下限与累计结果；不把源码的“混合权重 clamp”改写成“每个词概率 clamp”。在正温度 T、严格无并列且无浮点边界歧义的参考语义下，K 是温度分布的前 k 个词；在 K 内重新归一化后，再选达到 nucleus 阈值 p 的最短前缀 S。最终只需在 S 上归一化 z/T：

```math
\begin{aligned}
u(v)&=\frac{e^{z_v/T}}{\sum_w e^{z_w/T}},
&u_K(v)&=\frac{u(v)\mathbf1\{v\in K\}}{u(K)},\\
m&=\min\left\{j:\sum_{i=1}^{j}u_K(v_{(i)})\ge p\right\},
&S&=\{v_{(1)},\ldots,v_{(m)}\},\\
b(v)&=\frac{e^{z_v/T}\mathbf1\{v\in S\}}{\sum_{w\in S}e^{z_w/T}}.
\end{aligned}
```

正温度不改变精确实数下的排序，但会改变 nucleus 的累计概率和最终支持。例：u=(0.45,0.30,0.20,0.05)、k=2、p=0.55；先 top-k 得到 (0.60,0.40)，再 top-p 只保留第一词，b=(1,0,0,0)。若在原始 u 上同时判定 top-k 与 top-p，前两词都会保留，得到 (0.60,0.40,0,0)，两者 TV=0.40。这是顺序差异的构造例，不是对论文模型输出的测量。

**参考语义的边界。** FlashInfer 使用无需显式全排序的 GPU 采样；固定版本的上游测试也允许 k 边界并列与浮点舍入影响保留数量，并明确指出 `softmax→top_k_renorm` 与 `top_k_mask→softmax` 不保证逐位一致。因此排序参考算法只在严格间隔条件下解释理想支持；临界值必须记录实际 kernel、dtype 和执行结果，不能用 CPU 排序顺序宣称已验证 GPU 等价。[采样内核](https://github.com/flashinfer-ai/flashinfer/blob/f1e6fdcb8f65104047697f022b5d055ef022d763/include/flashinfer/sampling.cuh) · [固定测试中的舍入与并列说明](https://github.com/flashinfer-ai/flashinfer/blob/f1e6fdcb8f65104047697f022b5d055ef022d763/tests/utils/test_sampling.py)

### 停止误差经过温度与支持变化后的条件界

下面分析停止误差经过采样变换后会怎样变化；这是本课程对读出接口的补充推导。取同一入态上的两行 sampler 输入 z、z′，使用相同温度 T>0，分别得到支持集 S、S′ 和行为分布 b、b′。记共同支持为 I，各分布落在 I 以外的概率为 a、a′，二者较大值为 r。I 为空时 TV=1；I 非空时，记 D_I 为共同支持上 log 值差的振幅，得到下式。该结论并非 TaH2 论文宣称的定理：

```math
\begin{aligned}
I&=S\cap S',&a&=b(S\setminus I),&a'&=b'(S'\setminus I),\\
r&=\max(a,a'),
&D_I&=\frac{\max_{v\in I}(z_v-z'_v)-\min_{v\in I}(z_v-z'_v)}{T},\\
r&\le\operatorname{TV}(b,b')
\le r+(1-r)\tanh\!\left(\frac{D_I}{4}\right).
\end{aligned}
```

**证明。** 分两步处理共同支持和独占支持。先把 b、b′ 条件化到 I；归一化只减去常数，因此两者 log 概率比的振幅仍为 D_I。对任意共同支持分布 μ、ν，记概率比 μ/ν 的最小值、最大值为 L、U。在 L<1<U 时，这个比值在 ν 下的均值为 1；用凸函数的弦界得到下方第一行的第一个不等式，再固定 U/L、优化 L，得到第二个不等式。μ=ν 时直接取 0。最后，共同支持上可匹配的质量，至少是条件分布可匹配质量的 (1−r) 倍；独占支持的概率 a、a′ 又分别给出 TV 的下界，从而得到结论。

```math
\begin{aligned}
\operatorname{TV}(\mu,\nu)
&\le\frac{(U-1)(1-L)}{U-L}
\le\frac{\sqrt{U/L}-1}{\sqrt{U/L}+1}
=\tanh\!\left(\frac{\log(U/L)}4\right),\\
\sum_v\min\{b(v),b'(v)\}
&\ge(1-r)\sum_{v\in I}\min\{b(v\mid I),b'(v\mid I)\}.
\end{aligned}
```

S=S′ 时 r=0，只剩温度放大的 log 差振幅；更低 T 使该界通常更松。I 只有一个词时，条件分布完全相同，TV 恰为 r。证明要求 S、S′ 非空，I 非空时两侧在 I 上都有正质量；有限 log 值的实数 softmax 满足这个条件，浮点下溢须另行检查。该式允许 z、z′ 来自实际含权重 floor 的读出，但其结论仍是这些输入经理想概率变换后的条件分布界；浮点 kernel 偏差、DUO 未来缓存变化和整个任务正确率都不在这个证明里。它需要两个读出及支持信息，是离线对照证书，不能只凭当前 gate 在线免费获得。

也可把此前的 TV 界接到固定支持 S 上。设 u、u′ 是两种未截断的温度分布，u(S)、u′(S)>0，则归一化限制满足：

```math
\operatorname{TV}\!\left(u(\cdot\mid S),u'(\cdot\mid S)\right)
\le\min\left\{1,\frac{\operatorname{TV}(u,u')}{\max\{u(S),u'(S)\}}\right\}.
```

例如 u(S)≥u′(S)，在 S 上拆分差值后，L1 分子至多为 Σ_S|u−u′|+u(S)−u′(S)；S 外的 L1 至少为该质量差，故分子至多是全词表 L1，得到分母 u(S)。该结论要求同一 S；支持不同时须使用上面的独占质量项，不能把两个不同集合各自的保留质量直接代入。

**top-p 本身可以不连续。** 固定 p=0.6、0<ε<0.1，取 Pε=(0.6+ε,0.25−ε,0.15)、Qε=(0.6−ε,0.25+ε,0.15)。原分布 TV=2ε→0，而且邻近 ε=0 时排序没有并列；Pε 的 nucleus 只有第一词，Qε 的 nucleus 包含前两词，因此采样分布 TV=(0.25+ε)/0.85→5/17≈0.2941。小的原分布误差不能自动推出小的截断后误差，问题也不只发生在 argmax 并列处。

支持稳定可以用明确间隔作充分条件。若两个温度分布相差至多 η∞，参考分布第 k 与 k+1 名差距大于 2η∞，则 top-k 集合不变。进一步要求 K 内排序不变，令 δ_K=TV(u_K,u′_K)，C_j 为参考 u_K 前 j 项累计；若 C_{m−1}+δ_K<p<C_m−δ_K，则 nucleus 的长度 m 也不变。p=1 的无 nucleus 截断分支单独处理；处在等号或浮点容差内应标记为边界待核。

### 算法 T11 采样变换与支持审计

**输入** 同一文字前缀及内部入态的两种 sampler 输入 z、z′；每行真正生效的 T、k、p；后端版本与 dtype；可选的已采样词 y。读出可以来自前文同一共同链的强停/继续对照，但必须单独记录其缓存快照身份。

**输出** 两种参考行为分布、选中词的行为 logp、支持变化质量 r、实际 TV、条件上界和临界排序/累计标记。若只有部分 top-k 或缺少全词表归一化，不伪造完整分布与证书。

**前置条件** 正温度、有限输入、未掺入其他 logits processor；有效参数已经过服务端解析。贪心全批分支另作 argmax 与间隔检查；混合批次的贪心行使用实际生效的 10⁻⁶。参考计算采用明确的并列规则并记录边界，但边界样本不因此获得 GPU 等价认证。

![算法 T11 采样变换与支持审计](visualize:fde1_bGliZmlsZV9kWjJvTHd4T2VPeGVveldYbTZJajZn_FileDrive_45288dfe06608191af94334ba8699511)

**算法定位与核验。** 这是新增的离线审计算法，不是仓库现有函数，也不改写 FlashInfer 的高效采样内核。全排序参考实现每行 O(V log V)、存储 O(V)，不会被计入论文的模型 matmul FLOPs。已用小规模 CPU 概率例验证顺序反例、支持变化反例及上下界，并从两个固定 Python 函数的原 AST 捕获默认调用顺序；没有运行 CUDA kernel、加载模型或测量论文 benchmark。实际部署对照仍需接入真实张量与该版本 GPU 环境。

采样之后还要判断回答是否结束：停止内部循环只表示可以输出下一个文本 token。scheduler 追加该 token 后，才检查 `can_decode` 与 EOS；`ignore_eos=True` 时忽略 EOS 终止。这里使用的 EOS 来自导出后重新加载的 tokenizer，需要与前文的元数据改写结果对应，不能仅记录训练中 tokenizer 对象的 EOS。固定 MiniSGL 尚未实现 stop sequences、多候选 completions 或 presence/frequency penalties。[文本终止](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/scheduler/scheduler.py#L326-L350) · [服务能力范围](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/README.md#L49-L61)

## 加载字段与版本兼容

MiniSGL 的 TaH loader 接受 Qwen3MLPUpdater，以及 AlwaysIterDecider 或 Qwen3MLPIterDecider；causal 模式要求 even_mix，非 even_mix 要求 learned decider。加载后还会在 CUDA graph capture 之前应用深度、读出与阈值覆盖。[组件加载](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/models/tah_qwen3.py#L316-L444) · [运行时覆盖](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/engine/engine.py#L313-L363)

Updater 的 norm 开关按 checkpoint init_args、配置与默认值逐级读取；服务端 updater 内部固定使用 SiLU 和 epsilon=10⁻⁶，不能据此认为训练端任意 hidden_act 或 rms_norm_eps 修改都能无改动导入。默认发布配方与这两个默认值一致；自定义组件需扩展 loader 并重新对照。[服务端 updater](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/minisgl/models/tah_qwen3.py#L160-L199)

官方 1.7B 模型卡的示例仍写 import 路径 tah.model.tah_model；本页固定提交的 TaH2 实现在 tah2.model.tah_model，两个目录对应不同代际。复现时应优先采用与固定提交一致的入口，并在 manifest 固定下载 revision；不能混用旧示例 import 和新训练配置。[官方模型卡](https://huggingface.co/nics-efc/TaH2-1.7B-max2) · [固定 TaH2 模型类](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)

## 发布检查点的固定版本与加载契约

2026 年 11 月 8 日核对官方 Hub 元数据及以下不可变版本的小文件。三个循环 checkpoint 都保存 DUO、learned decider、阈值 0.5、top-k=2048、Triton、stop_prob_mix 与 memory_lean_fp32_reductions=true；区别中的 max_iter 分别为 2／4／8。Standard 保存普通 Qwen3 骨干，不含三个 TaH 侧文件。发布版本固定并不证明其数值复现了论文对应训练运行。

| 1.7B 发布对象 | 不可变 revision | 保存深度与组件 |
| --- | --- | --- |
| [Standard](https://huggingface.co/nics-efc/TaH2-1.7B-Standard/tree/bbf0d82946c8e242ab2dc1f3be20408e317562ec) | bbf0d82946c8e242ab2dc1f3be20408e317562ec | 单次骨干；无 tah_config.json、updater／decider bin |
| [max2](https://huggingface.co/nics-efc/TaH2-1.7B-max2/tree/e965dff2bd539b5be8087126b7d26bcb47e1b159) | e965dff2bd539b5be8087126b7d26bcb47e1b159 | max_iter=2；骨干与两个侧模块 |
| [max4](https://huggingface.co/nics-efc/TaH2-1.7B-max4/tree/817b7bd1822f0b17dbb469c525c31892de1d475b) | 817b7bd1822f0b17dbb469c525c31892de1d475b | max_iter=4；骨干与两个侧模块 |
| [max8](https://huggingface.co/nics-efc/TaH2-1.7B-max8/tree/88dc9b1d8d3f58e3c94dc8b794a78337f38d3a65) | 88dc9b1d8d3f58e3c94dc8b794a78337f38d3a65 | max_iter=8；骨干与两个侧模块 |

四份 config.json 相同，不能借此判断权重相同或恢复 TaH 计算图。Hub 的 1,720,574,976 个 safetensors 参数只统计骨干；两个侧模块另存 bin。按固定源码的张量定义，完整循环模型的参数账本为：

```math
P_{\rm full}=1{,}720{,}574{,}976+20{,}979{,}712+25{,}176{,}064=1{,}766{,}730{,}752.
```

以上加法是源码计数与发布清单的对应，尚未反序列化 bin 并逐张量复核。四份 chat_template.jinja 字节相同；max8 的 tokenizer_config.json 与 max2 字节格式不同，但解析后的 JSON 内容相同。模板、tokenizer.json 的版本及真正 token 化结果仍应分别登记。

### 配置合并的精确语义

固定 [TaHForCausalLM.from_pretrained](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) 先通过本地 os.path 读取 tah_config.json，再加载骨干及侧模块。应先用 snapshot_download 的 revision 参数得到完整本地快照；仅把 Hub ID 与 revision 传入骨干加载器，不能替代侧文件的本地解析。

当保存配置 s 和用户配置 p 均存在时，_merge_tah_config 逐字段执行以下规则；非空字典整项替换，不递归合并：

```math
\operatorname{merge}(s,p)_k=\begin{cases}p_k,&p_k\ne\mathrm{None}\ \land\ p_k\ne\{\},\\s_k,&\text{otherwise}.\end{cases}
```

`TaHConfig(max_iter=12)` 会携带其他字段的非空默认值，因此覆盖范围超过深度：DUO 会变成 causal，Triton 会变成 sdpa，`memory_lean_fp32_reductions` 会从 `True` 变成 `False`。合并时，`False` 会覆盖保存值，空字典会保留旧值；若仅提供 `iter_decider_kwargs={"threshold":0.6}`，则会替换整份 kwargs。只修改深度时，可先读完整保存配置，再用 `dataclasses.replace(saved, max_iter=12)`，并逐字段核对其余值。这里说明加载配置的控制方法，深度 12 是否训练过、是否有效仍需另外确认。

组件重载分为构造参数和权重两部分。`_load_iter_decider_weights` 用 `setdefault` 将最终 `max_iter` 注入 decider 的 `init_args`，已显式存在的 `max_iter` 仍优先。[load_registered_component](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/component_io.py) 使用非空 `init_args` 替代保存的 `init_args`，过滤同名 `state_dict` 键，再以 `strict=False` 加载；它没有检查返回的 missing／unexpected keys。因此，加载成功本身不足以确认侧模块权重齐全，`strict=False` 也不会自动忽略尺寸冲突。验收应分别记录键、shape、有效阈值与深度，并核对 wrapper 配置和重建后组件的属性，不能只记录 `tah_config`。

### 重载 decider 时未继承的 memory-lean 标志

需要分别检查首次构造与权重重载。`_build_iter_decider` 首次构造时，会把 wrapper 的 `memory_lean_fp32_reductions` 注入组件 kwargs；`from_pretrained` 随后调用 `_load_iter_decider_weights` 重建 decider，后者只补 `max_iter`。三个发布的 `tah_config` 都把 wrapper 标志保存为 `True`，但 `iter_decider_kwargs` 中没有这个键；非空 kwargs 又会整项覆盖 bin 中的 `init_args`。因此在这条 HF wrapper 重载路径中，即使模拟 bin 的 `init_args` 保存了 `True`，实际 decider 仍会回到类默认值 `False`。

未挂载vocab_parallel_ops时，这会从top-k logits加FP32 logsumexp的memory-lean路径切换为完整softmax再top-k；挂载vocab ops时优先走独立分支，不能把结论外推所有后端。两种表达式在有限logits的实数算术下等价，有限精度、分配量与性能仍需实际张量检查。此处未运行真实Tensor或模型。

保留预期分支的方法是：复制完整保存配置，在最终 `iter_decider_kwargs` 中显式设置 `memory_lean_fp32_reductions=True`，随后核对 wrapper、有效 decider 的属性及 vocab ops 是否启用。单独修改 wrapper 字段不足以确定组件分支。隔离原始 `_build_iter_decider`、`_load_iter_decider_weights` 与 `load_registered_component` 的检查结果为：首次构造 `True`、缺省重载 `False`、显式 kwargs 控制组 `True`；三者 `max_iter` 均为 8。R1 的“有效配置”应记录这些实测属性。 [固定加载器](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) · [实际分支](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_decider.py)

### 算法 R1：固定发布快照并保留执行配置

输入为 repo、完整 revision、固定代码和依赖、显式覆盖字段集合及验收适配器；输出为下载对象、最终配置、组件核对和模型运行四类分开的状态。下面是按原函数契约整理的验收算法；键／shape 报告和后端对齐记录属于待接入的验收能力，不是假称仓库已经返回这些记录。

![算法R1：固定发布快照并保留执行配置](visualize:fde1_bGliZmlsZV9zdDhyZzJxeGhkMHpPNnNZR0lfMVR3_FileDrive_45288dfe06608191af94334ba8699511)

本轮 C1–C8 使用真实 dataclass 与隔离的原始 loader 函数核对：四个 revision、23 个小文件的 Git blob／SHA-256、9 份固定源码、配置覆盖反例、完整配置替换和 decider 深度注入均通过。组件读取使用 torch.load 与组件替身；没有下载、反序列化或执行模型权重，没有运行 tokenizer、GPU 或服务。原有 T1–T12、D1–D3 与模型级验收清单继续保留。

[循环研究\_收尾核对\_20261108.zip](library-file:fde1_bGliZmlsZV9oMjJZT1c5eENPVF9lUVN4LUx3SDJn_FileDrive_45288dfe06608191af94334ba8699511)

核对包含39个文件：四版本清单与23份小文件、9份固定源码、C1–C8脚本／结果、15行R1、24页引用／公式结构及保存核对。包SHA-256：f8790eb5367722cc499a145d1fdc497289c859b05848def495b20d328bac611d。最终390条内部链接无失效目标，原有内容块ID和46份附件引用均保留；浏览器排版、完整权重与GPU模型仍未实跑。

训练 ceiling 与推理 ceiling 的区分、同一 M8 checkpoint 在推理 4／8／12 轮的完整结果，见[TaH2 论文结果与实现](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb)。运行时修改 max_iter 改变的是固定权重的执行上限，不等同于重新以该深度训练的模型。

# 交付与验收

- [ ] `export_manifest.json`：四个官方发布 revision 与小文件摘要已固定；仍需完整权重字节校验、侧模块键／shape、有效配置及实际加载记录。

- [ ] `prefill_decode_trace.json`：同一个短前缀在各步的 token、位置、KV、深度及读出对应。

- [ ] `backend_parity.json`：固定深度与自主路由分阶段比较，解释首个出现差异的位置。

- [ ] `generation_record.jsonl`：文本、token、停止原因、采样设置、版本与两阶段循环成本可回查。

继续到[分布式执行与性能评测](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c)时，沿用这些记录比较速度与显存；先确认执行含义，再判断优化收益。

# 动手检查

为一个极短前缀制作训练端与生成端的对照表，先在源码层面填清五项：

| 检查项 | 需要对应的内容 |
| --- | --- |
| 输入 | 相同 token 与前缀边界 |
| 权重 | 同一版本与转换精度 |
| 位置 | position id 与位置编码规则 |
| 缓存 | K/V 的建立、可见范围和更新 |
| 深度与读出 | 执行轮数、路由规则、输出组合 |

**理解检查**

当两个后端输出不同，能提出逐层缩小范围的检查顺序，说明应先固定什么、观察什么。只有比较条件明确，生成结果才能支持研究判断。

以后引入新的接口或循环方式时，这张对照表也应更新，使训练端学到的计算能够在生成端执行。实际数值一致性仍需后续运行验证。

# 围绕本课讨论

```codex-prompt
请用同一个短前缀对照 TaH2 训练模型与 miniSGLang。先带我完成输入、权重、位置、缓存和深度读出的对应表，再解释 BF16 导出与完整训练恢复的区别。只在遇到必要边界时展开 FSDP、重计算和 profiler。
```

# 继续学习

[上一课：一次内部循环](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811) · [学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133) · [下一课：接入在线蒸馏](https://chatgpt.com/space/page_e434e46c533c819192afedba5f10a140)
