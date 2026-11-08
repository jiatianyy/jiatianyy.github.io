一次参数更新由多条样本、多个微批次的损失共同形成。本课先用可手算的批次说明输入、移位标签和梯度如何对应，再定义全局监督分母，最后沿 TaH2 源码追到参数组、学习率和 optimizer.step。循环与路由的内部计算由下一课展开。

[返回学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
>
> **阅读主线**：样本与标签 → 标量损失 → 梯度累积与归约 → 参数变化。先完成手算示例，再读更新算法和源码调用链；参数组、调度及分组等价的条件按需深入。

# 监督位置与损失

## 一个可以手算的批次

下面的数字是教学用 token ID，不是真实 tokenizer 的输出。`2` 表示结束符，`0` 表示 padding。甲的前两个 token 是问题，回答包含 `21、22`；乙的回答只有 `41`。

| 张量 | 样本甲 | 样本乙 |
| --- | --- | --- |
| `input_ids` | `[11,12,21,22,2]` | `[31,32,41,2,0]` |
| `attention_mask` | `[1,1,1,1,1]` | `[1,1,1,1,0]` |
| `labels` | `[-100,-100,21,22,2]` | `[-100,-100,41,2,-100]` |
| 移位后的标签 | `[-100,21,22,2,-100]` | `[-100,41,2,-100,-100]` |

`attention_mask` 标记有效输入位置；`labels` 中的 `-100` 表示该目标不参与相应的预测损失。它们解决的是不同问题。

甲的第一个回答 token `21`，由输入 token `12` 所在位置的输出预测。这个位置能够读取问题前缀，因果注意力阻止它看到右侧的 `21`。训练时真实回答也出现在输入中，是为了让后续位置学习给定正确前缀预测下一个 token，这就是 teacher forcing。

送入 TaH2 的 labels 与原始文本位置对齐。预处理先确定 assistant 的监督区域，collator 再补齐长度；模型的 TaHForCausalLM._shift_labels 随后左移标签，并在末尾补 −100。调用方保留这次内部移位，不应预先再移一次。[预处理](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) · [标签移位](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)

问题 token 本身不作为监督目标，预测这些 token 的损失被屏蔽；最后一个问题位置仍承担首个回答 token 的预测损失。更早的问题表示还可经后续注意力影响答案损失，因此需分别追踪目标位置、计算依赖与参数梯度。

## 接到真实调用链

从训练入口 `main_tp` 看模型、数据和并行配置怎样接起来，再进入 `train_model_tp`。第一遍先追踪模块之间传递什么。[训练入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py)

1. `CustomTaHDataCollator.__call__` 补齐输入、标签和 attention mask；部分元数据会被移除，身份 trace 在 collator 外保存。

2. `TaHForCausalLM.forward` 准备移位标签、嵌入、位置与缓存，执行内部循环。

3. 模型汇总各轮结果，调用目标的 `final_loss_func`，返回 `outputs.loss`。

4. 训练器对损失反向传播，并完成优化器更新。

TaH2 的前向包含更新器、继续决策、深度标签和跨轮预测组合。官方 4B 示例配置采用 `NextTokenPredLoss` 与 `IterDeciderLoss`，权重分别为 `1.0` 和 `0.05`。下面的手算只展开 token 预测分量。[官方示例配置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml)

## 多个微批次形成一次更新

沿甲、乙两条样本继续看。甲有三个有效目标，乙有两个；先只考虑 token 预测项，并固定分组前后的预测和目标。假如显存只够逐条前向，我们可以先算甲的损失，再算乙的损失，但希望这两次计算仍组成同一个“五个 token 等权”的更新。

用后面练习的数字，甲的每个目标损失为 1，乙的每个目标损失为 4，则 S甲=3、S乙=8，五个目标的平均是 11/5=2.2。如果先求各样本均值再平均，会得到 (1+4)/2=2.5，那是在给两条样本相同总权重。这里选择 token 平均，因此应先汇总损失和，并让两个微批次共同使用分母 5：

```math
L_{\mathrm{token}}=\frac{S_{\mathrm{甲}}+S_{\mathrm{乙}}}{5}.
```

### 更新窗口的 token 目标

把一次 optimizer 更新的微批次集合记作窗口 `W`。设数据并行数为 `P`，rank `r` 的第 `k` 个微批次有效目标集合为 `I_rk`。这里的预测概率可来自 TaH2 跨深度混合；不能根据字段名 `logits` 就假设它是完整词表原始 logits。

```math
\begin{aligned}
S_{r,k}&=\sum_{(b,t)\in I_{r,k}}-\log p_\theta(\widetilde\ell_{b,t}\mid x_{b,\le t}),\\
N&=\sum_{r,k}|I_{r,k}|,\qquad L_W=\frac{\sum_{r,k}S_{r,k}}N.
\end{aligned}
```

### DP 平均梯度的补偿

把同一次更新分到 P 个数据并行 rank 后，每个 rank 只计算自己的那部分损失和。若各 rank 都对 S_rk/N 反向，而 reducer 最后又把梯度取平均，就会在全局 token 分母 N 之外再除一次 P，整个梯度因此缩小 P 倍。

所以，在这种平均归约约定下，每个微批次反向的是 P·S_rk/N：先补上 P，归约时再除掉 P，恰好留下全部目标损失梯度之和除以 N。下式展示的就是这一步抵消。若 reducer 本来求和，就不需要乘 P；TP ranks 共同处理同一份数据，也不应把同一目标重复计入 N。

```math
\begin{aligned}
\frac1P\sum_r\sum_k\nabla_\theta\!\left(\frac{P S_{r,k}}N\right)
&=\frac{\sum_{r,k}\nabla_\theta S_{r,k}}N\\
&=\nabla_\theta L_W.
\end{aligned}
```

上面的等式说明的是：同一个目标函数的各部分梯度可以相加。为此，窗口内不能先更新参数，分母必须固定，每条样本的预测、路由、标签和权重也必须在分组前后保持不变。满足这些条件，逐微批次反向就只改变计算的安排。

TaH2 的在线标签还会使用当前 forward 的统计。改变微批次划分，可能改变覆盖 cutoff、监督标签或权重，于是两次运行实际求导的目标也变了。后文会单独分析这部分；检查分组等价时，要先固定这些量，再验证下面源码如何统计全局分母。

### 源码的监督分母

固定源码先取得更新窗口，统计未移位 `labels`，仅跨 DP 求全局分母，再逐微批次调用模型并反向 `outputs.loss * dp_size`。已经使用全局分母后，再除累积次数会重复缩小梯度。

首标签全为 `-100` 时，源码计数与真实移位监督数相同；修改格式时必须重新验证。[训练循环](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L743)

```math
N_{\rm shifted}=N_{\rm raw}
-\sum_{z\in W}\mathbf1\{\ell_1(z)\ne-100\}.
```

## 算法 2 一次参数更新

下面给出具有全局 token 分母的参考更新算法。零监督窗口分支是拟补的防护，固定训练循环在该处未显式实现；只在本 rank 为零时跳过 collective 会破坏并行协议。

**过程** TokenUpdate(W\_local, dp\_group, optimizer, scheduler)

**输入** 本 rank 的更新窗口 W\_local、DP 通信组、优化器、scheduler，以及约定的梯度归约语义。

**输出** 分子、分母、梯度范数与参数变化记录；正常分支完成一次更新，N=0 时返回无更新记录。

![算法 2 操作与数学公式](visualize:fde1_bGliZmlsZV9DWDJ3WDBsV2dKdm5lbzFxMlJ3RXZR_FileDrive_45288dfe06608191af94334ba8699511)

## 梯度最终更新谁

`build_optimizer` 选择 `requires_grad=True` 的参数，按配置分组并设置学习率、权重衰减。判断一个模块是否在学习，要一起检查它是否可训练、是否进入优化器，以及损失是否有梯度路径到达它。

更新窗口内依次反向传播各微批次，之后同步需要同步的梯度、记录梯度范数、裁剪梯度，再执行：

```python
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

优化器改变参数，scheduler 推进学习率日程，`zero_grad` 清除本次累积的梯度。训练日志对 loss 做 `detach()` 只是保存数值，不等于把此前用于训练的损失切断。

## 参数组与学习率：从配方到真正更新

build_optimizer 先选择 requires_grad=True 的参数，再建立下表中的参数组。衰减名单由 get_parameter_names 按运行时 ALL_LAYERNORM_LAYERS 的模块类型排除规则生成，并剔除名称含 bias 的参数。判断 no-decay 应检查实际类注册和最终名单；仅凭参数名含 norm 或张量为一维，无法确定其分组。[TaH 参数组与调度](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L82-L221) · [Transformers v4.57.3 的名称判定](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/trainer_pt_utils.py)

| 判定点 | 固定实现 | 核验记录 |
| --- | --- | --- |
| 特殊学习率 | `training.separate_lr` 的键按 `component_name + '.'` 匹配参数完整名前缀 | 保存完整参数名、组件前缀及匹配数；不存在的组件仅打印 warning |
| 衰减与否 | 每个组件再分 decay／no-decay；no-decay 的 weight_decay=0 | 记录实际模块类型；剩余参数使用默认 LR 与同一衰减规则 |
| 默认配方 | 1.7B YAML 未设置 separate_lr；LR=4×10⁻⁵、weight_decay=0.01 | AdamW 未显式配置时 β=(0.9,0.999)、ε=10⁻⁸；CUDA 可用时尝试 fused |
| 分片后的真实组 | `split_dtensor_param_groups` 在 scheduler 构造前执行 | 以最终 `optimizer.param_groups` 为准；每个可训练参数恰好进入一个组 |

自定义特殊前缀若嵌套，当前循环不会在下一特殊组中跳过已匹配名称，因此必须检查参数对象的唯一性。`requires_grad`、组归属、非零梯度和实际参数改变量仍分别验收；本次未加载权重来生成真实参数库存。

### 日程长度与停止步数是两个字段

令 W 为 sampler 的 `windows_per_epoch`，E 为向上取整后的 epochs，H 为创建 scheduler 时的训练步数。固定 1.7B recipe 的最小倍率 μ=0.1、warmup 比例为 0.03；下式适用于 0≤s≤H，默认 cosine 半周期，s 是 scheduler 当前索引。

```math
\begin{aligned}
E&=\lceil\mathrm{num\_train\_epochs}\rceil,\quad H=WE,\quad w=\lfloor0.03H\rfloor,\\
\eta_g(s)&=\eta_{g,0}\lambda(s),\\
\lambda(s)&=\begin{cases}
s/\max(1,w),&s<w,\\
0.1+0.45\left[1+\cos\!\left(\pi\dfrac{s-w}{\max(1,H-w)}\right)\right],&s\ge w.
\end{cases}
\end{aligned}
```

学习率日程总长 H 来自完整 epoch 计划；正的 training.max_steps 只控制训练循环何时退出，不会让 create_scheduler 重建 H，因此提前停止时通常尚未到最低 LR。Transformers v4.57.3 中最终绑定、支持 min_lr_rate 的 cosine helper，在 H=100、w=3 时给出 s=0／3／100 的倍率 0／1／0.1，s=40 仍高于0.1。退出阈值在 optimizer.step 和 global_step 递增之后判断；恢复时已经达到上限的情形见下文。[真实 H 与提前停止阈值](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L433-L447) · [固定调度函数](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/optimization.py#L324-L380)

### 恢复时已达 max_steps 的首个更新

这个额外更新分支有三个前提：`M=training.max_steps` 为正数，恢复步数 g₀≥M，且完整 epoch 计划仍有可消费窗口。固定循环在首个 `optimizer.step` 前没有检查上限，而是先更新、令 g₁=g₀+1，再判断是否停止。因此恢复到 40 且 `max_steps=40` 时，可能再更新到 41；恢复到 45、上限为 40 时，则可能到 46。恢复到 39、上限为 40，更新一次到 40 属于正常边界。以上均假设更新未发生异常；epoch 已耗尽或 `max_steps=0` 时，不触发这里描述的分支。

| 首个窗口前状态 | 固定源码次序下的结果 | 需要保证的预算语义 |
| --- | --- | --- |
| g₀<M且有窗口 | 更新后检查 | 正常逐步逼近阈值 |
| g₀≥M>0且仍有窗口 | 可能额外更新一次 | 若要求“不再更新”，须在首个optimizer.step前退出 |
| 无剩余窗口 | 不进入更新 | 不由这个分支产生额外更新 |

恢复的参考顺序是先载入进度，检查剩余窗口与正 max_steps，再决定是否为下一次更新建立计算图。停止上限与 scheduler horizon 分别保存，不能共用一个 H。这是针对固定源码更新后检查行为的参考流程，上游实现并未因此改动。[恢复与更新后检查](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L544-L850)。最小检查保留原 AST 的 step、递增和 guard 次序，使用 toy optimizer 与一个可用窗口；没有执行真实训练。

新建默认 LambdaLR 会先初始化 s=0；warmup 非空时首个更新可使用零 LR。每次 `optimizer.step()` 后再 `scheduler.step()`，训练日志读取其后的 `get_last_lr()[0]`，对应推进后的第一组 LR，不能代替本次各组实际使用的 LR。记录更新前全部组的 LR、更新后 LR、scheduler 索引和 global_step，才能还原更新轨迹。[PyTorch v2.9.1 初始化](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/lr_scheduler.py#L139-L147)

### 验证目标也要接回同一更新记录

训练窗口与验证循环分别确定损失分母：训练按实际窗口计数，固定 eval 则先统计整份处理后数据，再把 batch 数截成 DP 的整数倍。目标也需分别注明：联合训练含决策损失，默认 eval 使用 NextTokenPredLoss；两者再与采样生成的任务指标区分。[数据计数链](https://chatgpt.com/space/page_b39954c20a348191b95d45a766f24ef2#23d1af89-77e4-4c85-a0d0-d3abf5b46601) · [损失恒等式与 T12](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c#b445b124-6da5-430c-896a-5b1c0faf4646)。

# 对照源码

本课固定使用 TaH 提交 `78ddf9a2c25713fced03a736c5746161f6ee590c`。按下表阅读，每次先定位列出的函数，再跟踪其输入和返回值。

| 阅读顺序 | 源码入口 | 本次要看清 |
| --- | --- | --- |
| 1 | [SFT_TaH.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py) | `main_tp` 怎样接起模型、数据与训练器 |
| 2 | [data_prepare.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) 与 [data_collator.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/data_collator.py) | 监督区域与批次补齐 |
| 3 | [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) | `_shift_labels`、`forward`、最终损失调用 |
| 4 | [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) | 分母、参数组、反向传播与更新 |
| 5 | [loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py) | token 目标与辅助目标分别怎样计算 |

[Full CoT 基础讲解](https://chatgpt.com/space/page_40d82c92ebc48191b0bf707bb236c682)可以按需补充因果移位、teacher forcing 与 prompt 梯度。

# 参数范围与精度

参考配方联合训练骨干、updater 与 decider。先从 `build_optimizer` 导出的参数组核对每项的名称、数量、学习率与 weight decay，再记录反向后的梯度范数和更新后的参数差值。`requires_grad=True`、进入优化器和实际有梯度是三个分别检查的条件。

## 可选的 LoRA 对照

LoRA 作为后续参数范围对照：学习 [PEFT LoRA 配置](https://huggingface.co/docs/peft/v0.15.0/en/package_reference/lora)中的 target modules、rank、scale 与额外保存模块，并核对合并前后的生成分布。TaH2 示例本身不能仅因同仓库旧 TaH 使用 LoRA，就被当作已接入 LoRA。增加 PEFT 时另行固定兼容版本。

## 主参数与计算精度

[tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 FSDP2 路径保留 FP32 主参数，以 BF16 前向计算和 FP32 归约；模型导出精度另由保存路径决定。把这些设置写入更新记录，避免只用一个“BF16 训练”标签涵盖全部状态。

# 更新记录与分组边界

## 实现验收记录

沿 `main_tp → preprocess_dataset → CustomTaHDataCollator → TaHForCausalLM.forward → final_loss_func → backward → optimizer.step` 保留一份实际记录。数据来源见[数据构建与版本固定](https://chatgpt.com/space/page_b39954c20a348191b95d45a766f24ef2)。

| 记录 | 验收所回答的问题 |
| --- | --- |
| `update_trace.json` | 每个 loss 的分子、分母、系数和更新窗口对应哪些样本 |
| `parameter_groups.json` | 骨干、updater、decider 实际哪些参数被训练 |
| `gradient_and_delta.json` | 预期模块收到梯度，更新后的参数变化可解释 |
| token 目标的分组对照 | 固定预测和路由，比较合批与累积的 token 目标梯度；完整辅助目标单独分析 |

## 在线标签对分组等价的限制

前面的梯度累积等式要求分组前后的目标相同，TaH2 的在线标签却可能受微批次划分影响。具体来说，覆盖 cutoff 使用当前这次 forward 在 DP 范围内收集的正 CE 改善量；动态 BCE 权重也使用该次 forward 的标签统计，定义如下。

对启用 `dyn_pos_weight` 的深度，BCE 正类权重按该次 forward 的正负标签数计算；下式范围在该固定源码中为 `w_min=0.02, w_max=50`，静态 `pos_weight` 分支另按配置。

```math
w_d^+=\begin{cases}
\operatorname{clip}(n_d^-/n_d^+,w_{\min},w_{\max}),&n_d^+n_d^->0,\\
1,&\text{某一类为空}.
\end{cases}
```

改变微批次划分可能改变 cutoff、标签和权重。验证严格分组等价时先冻结这些量；研究原在线算法时，将微批次划分作为算法配置记录。[标签统计](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L370) · [辅助损失](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L430)

这些记录用于后续实现验收；本页手算示例用于先理解计算关系。

# 动手检查

先只考虑 token 预测项，并假定分组前后的预测相同。甲的三个目标，每个损失都是 `1`；乙的两个目标，每个损失都是 `4`。如果两条样本分别求平均后再平均，得到什么？它与全局 token 平均有什么区别？

## 参考答案

全局 token 平均为：

```math
\frac{3\times1+2\times4}{5}=2.2.
```

两个微批次的均值再平均为 `(1+4)/2=2.5`。后一种写法给两条样本相同总权重，因此较短样本中的每个 token 获得更大权重。首先要明确研究希望给每条样本相同权重，还是给每个监督 token 相同权重。

## 归一化与输出张量

上述 `L_W` 只展开 token 分量。加入辅助项时，为每项分别保留分子、分母、系数与标签生成时刻；不要把不同归一化目标合成一个无法追溯的 loss 数值。

启用 `target_gather_mix` 且提供标签时，运行时可只汇总目标 token 的混合 log 概率，张量形状可能为 `[B,T,1]`。名为 `logits` 的字段不能默认是 `[B,T,V]` 的原始 logits。[运行时](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)

### 概率质量与三个损失分支

同一输出字段进入不同损失分支后，可能得到不同的标量损失。为看清差别，先固定实际执行深度 d、第 m 轮的归一化分布 q_m，以及理想停止权重 ω_m。源码的两条累计路径都会把权重下限设为 ε=10⁻¹⁰；记加上下限后的混合为 q̃、总质量为 Z，目标词 y 的累计 log 值为 λ_y：

```math
\begin{aligned}
\bar\omega_m&=\max(\omega_m,\epsilon),\\
\widetilde q(v)&=\sum_{m=1}^{d}\bar\omega_mq_m(v),\\
Z&=\sum_v\widetilde q(v)=\sum_m\bar\omega_m,\\
\lambda_y&=\operatorname{logsumexp}_{m\le d}
\left(\log\bar\omega_m+\log q_m(y)\right).
\end{aligned}
```

在 stop_prob_mix 下，finalize 没有额外除以 Z。target 路径采用 FP32 标量累计；全词表路径还将 log 权重及累计缓冲转换到模型 dtype，需另计舍入。[两条累计与 finalize](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L682)

| NextTokenPredLoss 分支 | 消费方式 | 精确算术下单个有效目标的损失 |
| --- | --- | --- |
| target_gather_mix，形状为 [B,T,1] | 直接取负目标 log 值 | −log q̃(y) |
| 全词表且无 vocab TP | 将累计值传给 cross_entropy | −log q̃(y)+log Z |
| 全词表且 vocab TP，logits_are_log_probs=True | nll_from_log_probs，直接 gather | −log q̃(y) |

这里的“无 vocab TP”也包括仅对骨干做 TP、词表未分片的配置。原始 logits 的 TP cross entropy 是另一个分支。只有在 Z=1 且相应数值一致时，表中三条才给出同一 NLL。[NextTokenPredLoss](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

未 clamp 的权重和为 1，且只考虑精确算术的权重下限影响时，有：

```math
\begin{aligned}
0&\le Z-1
=\sum_{m=1}^{d}(\epsilon-\omega_m)_+
\le d\epsilon,\\
L_{\rm CE}-L_{\rm target}
&=\log Z,\qquad
0\le\log Z\le\log(1+d\epsilon).
\end{aligned}
```

这是单个目标的条件界，不覆盖 BF16 缓冲舍入、global logsumexp 的误差或路由变化。对教学权重 (0,0.3,0.7)，Z=1+10⁻¹⁰，两个精确公式只差约 10⁻¹⁰；因此这个分支差别应准确记录，也不能据此推断实际能力下降。若实现把混合 log 值用于 KL 或行为概率，应先决定是否显式归一化，并将该决定纳入目标定义。

target-only 节省的是混合累计缓冲：从完整词表规模降为每位置一个数。每轮目标概率仍需要全词表归一化，decider 仍需要全局 top K 概率，因此它不表示去掉了 LM head 或词表归约。

### 算法 T3 读出质量与损失路径核对

**输入** 小词表逐深度 logits、同一实际路由、目标 ID、gate logits、累计 dtype 与 epsilon。

**输出** 理想及源码下限参考分布、质量 Z、target 与完整分布损失的差异、按固定分支检查的梯度。

**前置条件** 固定路由、标签和各层输入；先用 stop_prob_mix；不把数值对照自动扩大为整段生成等价。

**性质** 这是针对源码三条分支的参考核对算法；使用全词表小例建立基准，再逐项替换成目标 gather 或词表分片算子。

![算法 T3 读出质量与损失路径核对](visualize:fde1_bGliZmlsZV8tT0kxM1ZEVlE4NW8zbW1YM1NMX29R_FileDrive_45288dfe06608191af94334ba8699511)

**源码** [`accumulate_iter` 与 `_accumulate_target_mix`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L647) · [损失分派](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

本例首标签为 `-100`，移位前后的有效目标数相同。训练器实际统计未移位的 `labels`；修改数据格式时，应重新检查首位置和监督边界。

# 围绕本课讨论

```codex-prompt
请从本页的两条短样本开始教学。先让我定位第一个回答 token 的监督位置，再解释 mask、全局 token 分母和梯度累积。之后沿固定版本的 TaH2 源码跟踪一次参数更新，把教学简化与实际实现分别说清；每次展开一个计算边界。
```

# 继续学习

[学习地图](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133) · [下一课：一次内部循环](https://chatgpt.com/space/page_31a843d260588191a252ae63fa65f811)
