---
title: "一次参数更新"
description: "一次参数更新由多条样本、多个微批次的损失共同形成。本课先用可手算的批次说明输入、移位标签和梯度如何对应，再定义全局监督分母，最后沿 TaH2 源码追到参数组、学习率和 optimizer.step。循环与路由的内部计算由下一课展开。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647"
sourcePageId: "page_6a7d786183d0819192c20a043a45d647"
sourceSha256: fc5197e0e6cf06e996b8956379c87b1bd60754eb9271be1d2bc6e3c1a58b023a
series: loop-reasoning
references: []
---

<span id="source-block-f34d9508-9c54-4703-92ed-404f24d6e06a"></span>

一次参数更新由多条样本、多个微批次的损失共同形成。本课先用可手算的批次说明输入、移位标签和梯度如何对应，再定义全局监督分母，最后沿 TaH2 源码追到参数组、学习率和 optimizer.step。循环与路由的内部计算由下一课展开。

<span id="source-block-d0c7df90-ad0d-43ab-b1ed-de6a71420a8f"></span>

[返回学习地图](/notes/loop-reasoning/engineering/)

<span id="source-block-aecc7e14-61a6-4983-a818-cc7c92d90ee6"></span>

> [!CALLOUT]
>
> **阅读主线**：样本与标签 → 标量损失 → 梯度累积与归约 → 参数变化。先完成手算示例，再读更新算法和源码调用链；参数组、调度及分组等价的条件按需深入。

<span id="source-block-72d697ef-4dd1-422e-952e-cc3a06e066e5"></span>

## 监督位置与损失

<span id="source-block-fd8f0e5b-dd3a-4412-9b0b-5361d2035bc3"></span>

### 一个可以手算的批次

<span id="source-block-dda1b367-7806-4c31-aaac-fa47a2d9768f"></span>

下面的数字是教学用 token ID，不是真实 tokenizer 的输出。`2` 表示结束符，`0` 表示 padding。甲的前两个 token 是问题，回答包含 `21、22`；乙的回答只有 `41`。

<span id="source-block-2c2e1a87-2e02-4fd0-bdc0-555902e8b130"></span>

| 张量 | 样本甲 | 样本乙 |
| --- | --- | --- |
| `input_ids` | `[11,12,21,22,2]` | `[31,32,41,2,0]` |
| `attention_mask` | `[1,1,1,1,1]` | `[1,1,1,1,0]` |
| `labels` | `[-100,-100,21,22,2]` | `[-100,-100,41,2,-100]` |
| 移位后的标签 | `[-100,21,22,2,-100]` | `[-100,41,2,-100,-100]` |

<span id="source-block-53216446-a9ed-436a-9b5b-bee26a95afe7"></span>

`attention_mask` 标记有效输入位置；`labels` 中的 `-100` 表示该目标不参与相应的预测损失。它们解决的是不同问题。

<span id="source-block-535bfcfc-2072-4e52-b260-290cf6ee3da3"></span>

甲的第一个回答 token `21`，由输入 token `12` 所在位置的输出预测。这个位置能够读取问题前缀，因果注意力阻止它看到右侧的 `21`。训练时真实回答也出现在输入中，是为了让后续位置学习给定正确前缀预测下一个 token，这就是 teacher forcing。

<span id="source-block-81a579cf-bff7-4e76-b5cf-0fe9882b84a3"></span>

送入 TaH2 的 labels 与原始文本位置对齐。预处理先确定 assistant 的监督区域，collator 再补齐长度；模型的 TaHForCausalLM._shift_labels 随后左移标签，并在末尾补 −100。调用方保留这次内部移位，不应预先再移一次。[预处理](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) · [标签移位](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)

<span id="source-block-ccc3a347-e943-4936-91e5-5235d4ad3432"></span>

问题 token 本身不作为监督目标，预测这些 token 的损失被屏蔽；最后一个问题位置仍承担首个回答 token 的预测损失。更早的问题表示还可经后续注意力影响答案损失，因此需分别追踪目标位置、计算依赖与参数梯度。

<span id="source-block-55592ff8-29c7-4cb6-87da-fa8a76a26ae9"></span>

### 接到真实调用链

<span id="source-block-73c1736c-a2f2-4b60-9210-96d93239122f"></span>

从训练入口 `main_tp` 看模型、数据和并行配置怎样接起来，再进入 `train_model_tp`。第一遍先追踪模块之间传递什么。[训练入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py)

<span id="source-block-1cd4e3fc-1952-416b-81a0-70d05625f842"></span>

1. `CustomTaHDataCollator.__call__` 补齐输入、标签和 attention mask；部分元数据会被移除，身份 trace 在 collator 外保存。

<span id="source-block-dc5580b5-49ac-40c0-a279-877089894787"></span>

2. `TaHForCausalLM.forward` 准备移位标签、嵌入、位置与缓存，执行内部循环。

<span id="source-block-18f306c8-710c-4cc5-9010-5e99ca129f93"></span>

3. 模型汇总各轮结果，调用目标的 `final_loss_func`，返回 `outputs.loss`。

<span id="source-block-925ed0ed-05a9-44f9-abcc-f7a8ac44e51a"></span>

4. 训练器对损失反向传播，并完成优化器更新。

<span id="source-block-b31b9bd1-d21c-4473-89c1-c00da00aa125"></span>

TaH2 的前向包含更新器、继续决策、深度标签和跨轮预测组合。官方 4B 示例配置采用 `NextTokenPredLoss` 与 `IterDeciderLoss`，权重分别为 `1.0` 和 `0.05`。下面的手算只展开 token 预测分量。[官方示例配置](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/recipes/qwen3_4b/sft_tah.yaml)

<span id="source-block-aa05a699-fb0c-440e-825a-7ad18c1537d2"></span>

### 多个微批次形成一次更新

<span id="source-block-396faf20-72cf-411d-9bf0-aec2900e755f"></span>

沿甲、乙两条样本继续看。甲有三个有效目标，乙有两个；先只考虑 token 预测项，并固定分组前后的预测和目标。假如显存只够逐条前向，我们可以先算甲的损失，再算乙的损失，但希望这两次计算仍组成同一个“五个 token 等权”的更新。

<span id="source-block-4766dd0a-c5d6-4e5c-b913-f4780c43a0da"></span>

用后面练习的数字，甲的每个目标损失为 1，乙的每个目标损失为 4，则 S甲=3、S乙=8，五个目标的平均是 11/5=2.2。如果先求各样本均值再平均，会得到 (1+4)/2=2.5，那是在给两条样本相同总权重。这里选择 token 平均，因此应先汇总损失和，并让两个微批次共同使用分母 5：

<span id="source-block-c9c4b26b-0c6f-4a49-be05-763760d87673"></span>

```math
L_{\mathrm{token}}=\frac{S_{\mathrm{甲}}+S_{\mathrm{乙}}}{5}.
```

<span id="source-block-d6383857-5844-41ac-94cf-b834f11380ba"></span>

#### 更新窗口的 token 目标

<span id="source-block-498dd8eb-0b06-4cc5-ad01-4682877b5f12"></span>

把一次 optimizer 更新的微批次集合记作窗口 `W`。设数据并行数为 `P`，rank `r` 的第 `k` 个微批次有效目标集合为 `I_rk`。这里的预测概率可来自 TaH2 跨深度混合；不能根据字段名 `logits` 就假设它是完整词表原始 logits。

<span id="source-block-841c9e84-80a1-46b6-900e-6c6713d8d7ac"></span>

```math
\begin{aligned}
S_{r,k}&=\sum_{(b,t)\in I_{r,k}}-\log p_\theta(\widetilde\ell_{b,t}\mid x_{b,\le t}),\\
N&=\sum_{r,k}|I_{r,k}|,\qquad L_W=\frac{\sum_{r,k}S_{r,k}}N.
\end{aligned}
```

<span id="source-block-4a9106a1-5668-4562-ba16-bc603f30fe2f"></span>

#### DP 平均梯度的补偿

<span id="source-block-b7a24f44-1cba-456e-8d72-67f2d7706fbd"></span>

把同一次更新分到 P 个数据并行 rank 后，每个 rank 只计算自己的那部分损失和。若各 rank 都对 S_rk/N 反向，而 reducer 最后又把梯度取平均，就会在全局 token 分母 N 之外再除一次 P，整个梯度因此缩小 P 倍。

<span id="source-block-e024afd1-a18a-49ac-9444-499fecc40231"></span>

所以，在这种平均归约约定下，每个微批次反向的是 P·S_rk/N：先补上 P，归约时再除掉 P，恰好留下全部目标损失梯度之和除以 N。下式展示的就是这一步抵消。若 reducer 本来求和，就不需要乘 P；TP ranks 共同处理同一份数据，也不应把同一目标重复计入 N。

<span id="source-block-fbe62f12-242a-43bb-81a1-7c9c96d454a0"></span>

```math
\begin{aligned}
\frac1P\sum_r\sum_k\nabla_\theta\!\left(\frac{P S_{r,k}}N\right)
&=\frac{\sum_{r,k}\nabla_\theta S_{r,k}}N\\
&=\nabla_\theta L_W.
\end{aligned}
```

<span id="source-block-f5eb8f40-a55c-4977-bf4b-8d6dce94810d"></span>

上面的等式说明的是：同一个目标函数的各部分梯度可以相加。为此，窗口内不能先更新参数，分母必须固定，每条样本的预测、路由、标签和权重也必须在分组前后保持不变。满足这些条件，逐微批次反向就只改变计算的安排。

<span id="source-block-d9d300d5-83d9-4a10-9f9e-495fa66c5552"></span>

TaH2 的在线标签还会使用当前 forward 的统计。改变微批次划分，可能改变覆盖 cutoff、监督标签或权重，于是两次运行实际求导的目标也变了。后文会单独分析这部分；检查分组等价时，要先固定这些量，再验证下面源码如何统计全局分母。

<span id="source-block-14d0af18-dbea-43df-8f26-aa14f524a60b"></span>

#### 源码的监督分母

<span id="source-block-717eaf80-9865-4069-84b3-98b0cee1b01a"></span>

固定源码先取得更新窗口，统计未移位 `labels`，仅跨 DP 求全局分母，再逐微批次调用模型并反向 `outputs.loss * dp_size`。已经使用全局分母后，再除累积次数会重复缩小梯度。

<span id="source-block-0b0655df-22aa-409a-91a7-59089ab1c28d"></span>

首标签全为 `-100` 时，源码计数与真实移位监督数相同；修改格式时必须重新验证。[训练循环](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L743)

<span id="source-block-00299fa0-4c4c-4acd-9839-44dfb29f110a"></span>

```math
N_{\rm shifted}=N_{\rm raw}
-\sum_{z\in W}\mathbf1\{\ell_1(z)\ne-100\}.
```

<span id="source-block-5faff74d-0e64-45b8-9747-7899707829db"></span>

### 算法 2 一次参数更新

<span id="source-block-f4c119fb-d64b-46df-aac0-eb9b5beedc27"></span>

下面给出具有全局 token 分母的参考更新算法。零监督窗口分支是拟补的防护，固定训练循环在该处未显式实现；只在本 rank 为零时跳过 collective 会破坏并行协议。

<span id="source-block-84dfbdb3-1362-47ae-b5c9-4513e4c48efd"></span>

**过程** TokenUpdate(W\_local, dp\_group, optimizer, scheduler)

<span id="source-block-c7bbe321-6a89-423f-831e-157e80e2ab9e"></span>

**输入** 本 rank 的更新窗口 W\_local、DP 通信组、优化器、scheduler，以及约定的梯度归约语义。

<span id="source-block-9ab0d7d3-781e-49d2-bcd6-8030889fb282"></span>

**输出** 分子、分母、梯度范数与参数变化记录；正常分支完成一次更新，N=0 时返回无更新记录。

<span id="source-block-063e5bdc-d77f-4d25-a9e8-b215417868f5"></span>

<figure class="research-embed"><iframe src="/research-assets/3f5f43eeb7ebde4ab4dd9f28badb8257b484dac9585a8fc6feb43f5959a86825.html" title="算法 2 操作与数学公式" loading="lazy" sandbox=""></iframe><figcaption>算法 2 操作与数学公式 · <a href="/research-assets/3f5f43eeb7ebde4ab4dd9f28badb8257b484dac9585a8fc6feb43f5959a86825.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-d281b766-64d6-4139-b409-d8535c2eb181"></span>

### 梯度最终更新谁

<span id="source-block-f00d5c3e-ff77-4353-a1fa-0ef23acf98dd"></span>

`build_optimizer` 选择 `requires_grad=True` 的参数，按配置分组并设置学习率、权重衰减。判断一个模块是否在学习，要一起检查它是否可训练、是否进入优化器，以及损失是否有梯度路径到达它。

<span id="source-block-769ac782-0c92-47ba-a319-65ed89881b62"></span>

更新窗口内依次反向传播各微批次，之后同步需要同步的梯度、记录梯度范数、裁剪梯度，再执行：

<span id="source-block-ff101068-4b78-4e00-84c3-2d8832fc0b70"></span>

```python
optimizer.step()
lr_scheduler.step()
optimizer.zero_grad(set_to_none=True)
```

<span id="source-block-cc17c0b0-5a1e-46d3-86cf-5f759ad0eafa"></span>

优化器改变参数，scheduler 推进学习率日程，`zero_grad` 清除本次累积的梯度。训练日志对 loss 做 `detach()` 只是保存数值，不等于把此前用于训练的损失切断。

<span id="source-block-673e822e-b486-411f-9aee-6fe5de3c2e6c"></span>

### 参数组与学习率：从配方到真正更新

<span id="source-block-b78f8d59-6e84-4380-80a2-71122e1c4b80"></span>

build_optimizer 先选择 requires_grad=True 的参数，再建立下表中的参数组。衰减名单由 get_parameter_names 按运行时 ALL_LAYERNORM_LAYERS 的模块类型排除规则生成，并剔除名称含 bias 的参数。判断 no-decay 应检查实际类注册和最终名单；仅凭参数名含 norm 或张量为一维，无法确定其分组。[TaH 参数组与调度](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L82-L221) · [Transformers v4.57.3 的名称判定](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/trainer_pt_utils.py)

<span id="source-block-20143220-ffcb-4a16-aea9-885058703ab7"></span>

| 判定点 | 固定实现 | 核验记录 |
| --- | --- | --- |
| 特殊学习率 | `training.separate_lr` 的键按 `component_name + '.'` 匹配参数完整名前缀 | 保存完整参数名、组件前缀及匹配数；不存在的组件仅打印 warning |
| 衰减与否 | 每个组件再分 decay／no-decay；no-decay 的 weight_decay=0 | 记录实际模块类型；剩余参数使用默认 LR 与同一衰减规则 |
| 默认配方 | 1.7B YAML 未设置 separate_lr；LR=4×10⁻⁵、weight_decay=0.01 | AdamW 未显式配置时 β=(0.9,0.999)、ε=10⁻⁸；CUDA 可用时尝试 fused |
| 分片后的真实组 | `split_dtensor_param_groups` 在 scheduler 构造前执行 | 以最终 `optimizer.param_groups` 为准；每个可训练参数恰好进入一个组 |

<span id="source-block-4783a74d-b1d2-457b-8e55-f891fde56a85"></span>

自定义特殊前缀若嵌套，当前循环不会在下一特殊组中跳过已匹配名称，因此必须检查参数对象的唯一性。`requires_grad`、组归属、非零梯度和实际参数改变量仍分别验收；本次未加载权重来生成真实参数库存。

<span id="source-block-5065ffd6-353e-4fb4-9387-8c8f1735f7ea"></span>

#### 日程长度与停止步数是两个字段

<span id="source-block-f9b59bcd-3379-4939-98e2-84a02eb81fb3"></span>

令 W 为 sampler 的 `windows_per_epoch`，E 为向上取整后的 epochs，H 为创建 scheduler 时的训练步数。固定 1.7B recipe 的最小倍率 μ=0.1、warmup 比例为 0.03；下式适用于 0≤s≤H，默认 cosine 半周期，s 是 scheduler 当前索引。

<span id="source-block-0f26394e-c64a-49e5-a674-921878111681"></span>

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

<span id="source-block-f33f630d-8a0c-4e5b-8188-ac54cb9ef098"></span>

学习率日程总长 H 来自完整 epoch 计划；正的 training.max_steps 只控制训练循环何时退出，不会让 create_scheduler 重建 H，因此提前停止时通常尚未到最低 LR。Transformers v4.57.3 中最终绑定、支持 min_lr_rate 的 cosine helper，在 H=100、w=3 时给出 s=0／3／100 的倍率 0／1／0.1，s=40 仍高于0.1。退出阈值在 optimizer.step 和 global_step 递增之后判断；恢复时已经达到上限的情形见下文。[真实 H 与提前停止阈值](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L433-L447) · [固定调度函数](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/optimization.py#L324-L380)

<span id="source-block-40c0110e-2422-4175-9399-3d5d06fa633e"></span>

#### 恢复时已达 max_steps 的首个更新

<span id="source-block-33f1083e-e242-4be3-943c-957115b38288"></span>

这个额外更新分支有三个前提：`M=training.max_steps` 为正数，恢复步数 g₀≥M，且完整 epoch 计划仍有可消费窗口。固定循环在首个 `optimizer.step` 前没有检查上限，而是先更新、令 g₁=g₀+1，再判断是否停止。因此恢复到 40 且 `max_steps=40` 时，可能再更新到 41；恢复到 45、上限为 40 时，则可能到 46。恢复到 39、上限为 40，更新一次到 40 属于正常边界。以上均假设更新未发生异常；epoch 已耗尽或 `max_steps=0` 时，不触发这里描述的分支。

<span id="source-block-d1339ab1-d5a5-44ff-9729-44535bc37adf"></span>

| 首个窗口前状态 | 固定源码次序下的结果 | 需要保证的预算语义 |
| --- | --- | --- |
| g₀<M且有窗口 | 更新后检查 | 正常逐步逼近阈值 |
| g₀≥M>0且仍有窗口 | 可能额外更新一次 | 若要求“不再更新”，须在首个optimizer.step前退出 |
| 无剩余窗口 | 不进入更新 | 不由这个分支产生额外更新 |

<span id="source-block-c7fe0fa1-4f43-44fa-bf9d-a93b24b110d5"></span>

恢复的参考顺序是先载入进度，检查剩余窗口与正 max_steps，再决定是否为下一次更新建立计算图。停止上限与 scheduler horizon 分别保存，不能共用一个 H。这是针对固定源码更新后检查行为的参考流程，上游实现并未因此改动。[恢复与更新后检查](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py#L544-L850)。最小检查保留原 AST 的 step、递增和 guard 次序，使用 toy optimizer 与一个可用窗口；没有执行真实训练。

<span id="source-block-56c82b50-25fc-4a37-8149-c5232e306aa1"></span>

新建默认 LambdaLR 会先初始化 s=0；warmup 非空时首个更新可使用零 LR。每次 `optimizer.step()` 后再 `scheduler.step()`，训练日志读取其后的 `get_last_lr()[0]`，对应推进后的第一组 LR，不能代替本次各组实际使用的 LR。记录更新前全部组的 LR、更新后 LR、scheduler 索引和 global_step，才能还原更新轨迹。[PyTorch v2.9.1 初始化](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/lr_scheduler.py#L139-L147)

<span id="source-block-b7ec2218-4af1-4074-8995-be5ae7fee47e"></span>

#### 验证目标也要接回同一更新记录

<span id="source-block-d52dbd4d-51eb-44f2-8ce3-1da2cb893d48"></span>

训练窗口与验证循环分别确定损失分母：训练按实际窗口计数，固定 eval 则先统计整份处理后数据，再把 batch 数截成 DP 的整数倍。目标也需分别注明：联合训练含决策损失，默认 eval 使用 NextTokenPredLoss；两者再与采样生成的任务指标区分。[数据计数链](/notes/loop-reasoning/data/#source-block-23d1af89-77e4-4c85-a0d0-d3abf5b46601) · [损失恒等式与 T12](/notes/loop-reasoning/systems-evaluation/#source-block-b445b124-6da5-430c-896a-5b1c0faf4646)。

<span id="source-block-2efebef0-2a41-43c2-b889-72da1b941cd1"></span>

## 对照源码

<span id="source-block-0c52d9d4-2199-4ec5-ae0c-5d6c1af4b979"></span>

本课固定使用 TaH 提交 `78ddf9a2c25713fced03a736c5746161f6ee590c`。按下表阅读，每次先定位列出的函数，再跟踪其输入和返回值。

<span id="source-block-3e0b852c-7bce-404d-a333-260176f8d8e1"></span>

| 阅读顺序 | 源码入口 | 本次要看清 |
| --- | --- | --- |
| 1 | [SFT_TaH.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py) | `main_tp` 怎样接起模型、数据与训练器 |
| 2 | [data_prepare.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) 与 [data_collator.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/data_collator.py) | 监督区域与批次补齐 |
| 3 | [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) | `_shift_labels`、`forward`、最终损失调用 |
| 4 | [simple_loop.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/simple_loop.py) | 分母、参数组、反向传播与更新 |
| 5 | [loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py) | token 目标与辅助目标分别怎样计算 |

<span id="source-block-a6d3133d-30ed-452c-b68b-668ceb33eeb6"></span>

[Full CoT 基础讲解](/notes/loop-reasoning/full-cot/)可以按需补充因果移位、teacher forcing 与 prompt 梯度。

<span id="source-block-3885a313-257b-4f92-9f41-54b7bd48fbf5"></span>

## 参数范围与精度

<span id="source-block-3f5cae13-368a-4968-a1ca-bf0fe8d8b8b0"></span>

参考配方联合训练骨干、updater 与 decider。先从 `build_optimizer` 导出的参数组核对每项的名称、数量、学习率与 weight decay，再记录反向后的梯度范数和更新后的参数差值。`requires_grad=True`、进入优化器和实际有梯度是三个分别检查的条件。

<span id="source-block-9bfad8c9-9ac1-4b13-96f3-4f8b4fd5b246"></span>

### 可选的 LoRA 对照

<span id="source-block-faff8226-20ff-46fb-a962-f861622836f3"></span>

LoRA 作为后续参数范围对照：学习 [PEFT LoRA 配置](https://huggingface.co/docs/peft/v0.15.0/en/package_reference/lora)中的 target modules、rank、scale 与额外保存模块，并核对合并前后的生成分布。TaH2 示例本身不能仅因同仓库旧 TaH 使用 LoRA，就被当作已接入 LoRA。增加 PEFT 时另行固定兼容版本。

<span id="source-block-1af2e38b-c1c2-45d8-a556-c8c324e209a2"></span>

### 主参数与计算精度

<span id="source-block-c7e8d139-0157-48fc-99f4-da65ed330191"></span>

[tp.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/tp.py) 的 FSDP2 路径保留 FP32 主参数，以 BF16 前向计算和 FP32 归约；模型导出精度另由保存路径决定。把这些设置写入更新记录，避免只用一个“BF16 训练”标签涵盖全部状态。

<span id="source-block-94531f7c-37e7-4a4c-aa83-aee47a438d04"></span>

## 更新记录与分组边界

<span id="source-block-8b07ed13-eab0-4048-a558-811b7ebb0e60"></span>

### 实现验收记录

<span id="source-block-a6ae6a6e-4c96-4f6d-9993-331dfd18f54c"></span>

沿 `main_tp → preprocess_dataset → CustomTaHDataCollator → TaHForCausalLM.forward → final_loss_func → backward → optimizer.step` 保留一份实际记录。数据来源见[数据构建与版本固定](/notes/loop-reasoning/data/)。

<span id="source-block-810a74f1-9fc5-4fd0-9b86-8d0204f1ca46"></span>

| 记录 | 验收所回答的问题 |
| --- | --- |
| `update_trace.json` | 每个 loss 的分子、分母、系数和更新窗口对应哪些样本 |
| `parameter_groups.json` | 骨干、updater、decider 实际哪些参数被训练 |
| `gradient_and_delta.json` | 预期模块收到梯度，更新后的参数变化可解释 |
| token 目标的分组对照 | 固定预测和路由，比较合批与累积的 token 目标梯度；完整辅助目标单独分析 |

<span id="source-block-bb0a061d-7510-4b7f-a7ec-c15468df8566"></span>

### 在线标签对分组等价的限制

<span id="source-block-15cb0ac2-fb06-47b8-aab5-2f2bc57b4658"></span>

前面的梯度累积等式要求分组前后的目标相同，TaH2 的在线标签却可能受微批次划分影响。具体来说，覆盖 cutoff 使用当前这次 forward 在 DP 范围内收集的正 CE 改善量；动态 BCE 权重也使用该次 forward 的标签统计，定义如下。

<span id="source-block-8a1cd674-5287-4e1f-a9bf-6655e644a065"></span>

对启用 `dyn_pos_weight` 的深度，BCE 正类权重按该次 forward 的正负标签数计算；下式范围在该固定源码中为 `w_min=0.02, w_max=50`，静态 `pos_weight` 分支另按配置。

<span id="source-block-79d5c39b-2351-41a6-9cc3-62110fc0781b"></span>

```math
w_d^+=\begin{cases}
\operatorname{clip}(n_d^-/n_d^+,w_{\min},w_{\max}),&n_d^+n_d^->0,\\
1,&\text{某一类为空}.
\end{cases}
```

<span id="source-block-51c616e8-9226-4b35-a68c-a4948a0b4192"></span>

改变微批次划分可能改变 cutoff、标签和权重。验证严格分组等价时先冻结这些量；研究原在线算法时，将微批次划分作为算法配置记录。[标签统计](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/iter_label.py#L370) · [辅助损失](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L430)

<span id="source-block-d9764b78-4ee5-4b5f-a412-8dfc18b17f32"></span>

这些记录用于后续实现验收；本页手算示例用于先理解计算关系。

<span id="source-block-58d105a1-c857-4922-8818-b84055bf0f98"></span>

## 动手检查

<span id="source-block-b66489ee-55b8-4ac1-a0ae-12a24ac05d14"></span>

先只考虑 token 预测项，并假定分组前后的预测相同。甲的三个目标，每个损失都是 `1`；乙的两个目标，每个损失都是 `4`。如果两条样本分别求平均后再平均，得到什么？它与全局 token 平均有什么区别？

<span id="source-block-a10796f1-3bd4-416b-8f54-83ec65ef6e90"></span>

### 参考答案

<span id="source-block-1d6fe33e-2011-448a-b3eb-3dd781fd784e"></span>

全局 token 平均为：

<span id="source-block-a51e116a-77b8-4d56-8e6b-4c797dba24e5"></span>

```math
\frac{3\times1+2\times4}{5}=2.2.
```

<span id="source-block-aa4a4eb1-0299-42e8-b541-d7b3d9213f16"></span>

两个微批次的均值再平均为 `(1+4)/2=2.5`。后一种写法给两条样本相同总权重，因此较短样本中的每个 token 获得更大权重。首先要明确研究希望给每条样本相同权重，还是给每个监督 token 相同权重。

<span id="source-block-9d2159df-203b-415e-82dc-3aa1071b866b"></span>

### 归一化与输出张量

<span id="source-block-023a3016-a543-4e4e-b70b-37a5af673efe"></span>

上述 `L_W` 只展开 token 分量。加入辅助项时，为每项分别保留分子、分母、系数与标签生成时刻；不要把不同归一化目标合成一个无法追溯的 loss 数值。

<span id="source-block-ed315a13-aa0b-45ea-b044-c493e3258396"></span>

启用 `target_gather_mix` 且提供标签时，运行时可只汇总目标 token 的混合 log 概率，张量形状可能为 `[B,T,1]`。名为 `logits` 的字段不能默认是 `[B,T,V]` 的原始 logits。[运行时](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py)

<span id="source-block-450e1456-95aa-463f-a580-5585ef442425"></span>

#### 概率质量与三个损失分支

<span id="source-block-47fc6136-8ac7-46aa-913c-e58c71c4887b"></span>

同一输出字段进入不同损失分支后，可能得到不同的标量损失。为看清差别，先固定实际执行深度 d、第 m 轮的归一化分布 q_m，以及理想停止权重 ω_m。源码的两条累计路径都会把权重下限设为 ε=10⁻¹⁰；记加上下限后的混合为 q̃、总质量为 Z，目标词 y 的累计 log 值为 λ_y：

<span id="source-block-6d084efd-c241-4219-9950-f1fc805c7fea"></span>

```math
\begin{aligned}
\bar\omega_m&=\max(\omega_m,\epsilon),\\
\widetilde q(v)&=\sum_{m=1}^{d}\bar\omega_mq_m(v),\\
Z&=\sum_v\widetilde q(v)=\sum_m\bar\omega_m,\\
\lambda_y&=\operatorname{logsumexp}_{m\le d}
\left(\log\bar\omega_m+\log q_m(y)\right).
\end{aligned}
```

<span id="source-block-833e56ce-efda-4170-9158-d861a65530af"></span>

在 stop_prob_mix 下，finalize 没有额外除以 Z。target 路径采用 FP32 标量累计；全词表路径还将 log 权重及累计缓冲转换到模型 dtype，需另计舍入。[两条累计与 finalize](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L682)

<span id="source-block-2890b27d-494c-4c67-9e6b-f4ce2cdb6c77"></span>

| NextTokenPredLoss 分支 | 消费方式 | 精确算术下单个有效目标的损失 |
| --- | --- | --- |
| target_gather_mix，形状为 [B,T,1] | 直接取负目标 log 值 | −log q̃(y) |
| 全词表且无 vocab TP | 将累计值传给 cross_entropy | −log q̃(y)+log Z |
| 全词表且 vocab TP，logits_are_log_probs=True | nll_from_log_probs，直接 gather | −log q̃(y) |

<span id="source-block-69688451-c8a3-4305-9c26-cc91bbab656c"></span>

这里的“无 vocab TP”也包括仅对骨干做 TP、词表未分片的配置。原始 logits 的 TP cross entropy 是另一个分支。只有在 Z=1 且相应数值一致时，表中三条才给出同一 NLL。[NextTokenPredLoss](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

<span id="source-block-f978b161-e265-4f0c-88fa-8af3f1c5c4bc"></span>

未 clamp 的权重和为 1，且只考虑精确算术的权重下限影响时，有：

<span id="source-block-21f2d234-9a0b-40c0-9e17-a403ef0b8dbc"></span>

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

<span id="source-block-fd408537-b875-4933-96e8-e9dd2a8f0752"></span>

这是单个目标的条件界，不覆盖 BF16 缓冲舍入、global logsumexp 的误差或路由变化。对教学权重 (0,0.3,0.7)，Z=1+10⁻¹⁰，两个精确公式只差约 10⁻¹⁰；因此这个分支差别应准确记录，也不能据此推断实际能力下降。若实现把混合 log 值用于 KL 或行为概率，应先决定是否显式归一化，并将该决定纳入目标定义。

<span id="source-block-26d7fc64-4581-4407-b6c8-95e905b73bf7"></span>

target-only 节省的是混合累计缓冲：从完整词表规模降为每位置一个数。每轮目标概率仍需要全词表归一化，decider 仍需要全局 top K 概率，因此它不表示去掉了 LM head 或词表归约。

<span id="source-block-a377ea8b-5457-42fc-8fdb-39d983ece6cc"></span>

#### 算法 T3 读出质量与损失路径核对

<span id="source-block-22ebc1db-757f-44b3-a7b2-8cb4e8c730f3"></span>

**输入** 小词表逐深度 logits、同一实际路由、目标 ID、gate logits、累计 dtype 与 epsilon。

<span id="source-block-5432f79f-a542-401f-a2ef-621f8102aee5"></span>

**输出** 理想及源码下限参考分布、质量 Z、target 与完整分布损失的差异、按固定分支检查的梯度。

<span id="source-block-6ee9f831-8fb4-4913-910b-73d3570a8136"></span>

**前置条件** 固定路由、标签和各层输入；先用 stop_prob_mix；不把数值对照自动扩大为整段生成等价。

<span id="source-block-2b57bf61-30dc-42cd-ab25-c9f329e54ca7"></span>

**性质** 这是针对源码三条分支的参考核对算法；使用全词表小例建立基准，再逐项替换成目标 gather 或词表分片算子。

<span id="source-block-f9323d6f-6e24-43db-a022-51de406d330c"></span>

<figure class="research-embed"><iframe src="/research-assets/f253119efe8be6dd1991301b70124e1d740ed21d500d2419e65f68b40ed078cf.html" title="算法 T3 读出质量与损失路径核对" loading="lazy" sandbox=""></iframe><figcaption>算法 T3 读出质量与损失路径核对 · <a href="/research-assets/f253119efe8be6dd1991301b70124e1d740ed21d500d2419e65f68b40ed078cf.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-3213b957-2290-4d14-882f-19fb6e732ef1"></span>

**源码** [`accumulate_iter` 与 `_accumulate_target_mix`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_runtime.py#L520-L647) · [损失分派](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py#L39-L91)

<span id="source-block-dda6ce33-b417-4833-a8d9-b46dd8a05c4c"></span>

本例首标签为 `-100`，移位前后的有效目标数相同。训练器实际统计未移位的 `labels`；修改数据格式时，应重新检查首位置和监督边界。

<span id="source-block-7869dea0-b1a1-45c8-939e-fe9acc8a1697"></span>

## 围绕本课讨论

<span id="source-block-e52ac7b6-0d46-4a42-9d1f-12f2a628e823"></span>

```text
请从本页的两条短样本开始教学。先让我定位第一个回答 token 的监督位置，再解释 mask、全局 token 分母和梯度累积。之后沿固定版本的 TaH2 源码跟踪一次参数更新，把教学简化与实际实现分别说清；每次展开一个计算边界。
```

<span id="source-block-d20ad11a-0713-4d09-8090-baf0a877208f"></span>

## 继续学习

<span id="source-block-2fde0e76-83c9-46ce-afe0-dce7e6272b0f"></span>

[学习地图](/notes/loop-reasoning/engineering/) · [下一课：一次内部循环](/notes/loop-reasoning/internal-loop/)
