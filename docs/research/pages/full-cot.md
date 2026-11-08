用一条 Full CoT 样本贯通 next-token 训练：输入怎样对齐目标、哪些位置计入损失，以及梯度怎样回到 prompt 的表示。先看位置表和损失公式，再对照一次更新代码，最后区分监督屏蔽与输入条件。它是框架学习的可选基础材料。

[返回研究入口](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

# 输入位置与下一 token 目标

先固定同一条样本，后面所有公式和代码都沿它展开。输入依次是问题、正确解题过程、FINAL 答案和 EOS：用 `p1 p2 p3` 表示问题，`r1 r2` 表示过程，`a` 表示答案。训练时，这些 token 都在输入张量中；labels 则另外说明哪些预测需要计入损失。`FINAL` 在示意中只占一格，这些符号表示位置，不是实际 tokenizer 的切分。先看两列怎样对齐：

| 位置 | input_ids | labels |
| ---: | --- | --- |
| 0 | p1 | -100 |
| 1 | p2 | -100 |
| 2 | p3 | -100 |
| 3 | r1 | r1 |
| 4 | r2 | r2 |
| 5 | FINAL | FINAL |
| 6 | a | a |
| 7 | EOS | EOS |

模型在每个输入位置给出的是“下一个 token”的预测，而不是对该位置 token 重新作答。因此，位置 2 的输入是 `p3`，它的输出却要与位置 3 的标签 `r1` 比较。这就是 `labels[:, 1:]` 与前一位置 logits 对齐的原因。`labels[2] = -100` 屏蔽的是从 `p2` 预测 `p3` 的损失，并没有屏蔽 `p3` 输出对 `r1` 的预测。把对齐关系逐行写出来，就得到下表。

| 当前输入位置 | 对齐的下一 token 目标 | 是否计入损失 |
| --- | --- | --- |
| p1 | p2 | 否 |
| p2 | p3 | 否 |
| p3 | r1 | 是 |
| r1 | r2 | 是 |
| r2 | FINAL | 是 |
| FINAL | a | 是 |
| a | EOS | 是 |
| EOS | 本样本没有下一 token | 不参与本次移位损失 |

# 从隐状态到损失

现在只看预测 `r1` 的那一次计算。模型在 `p3` 位置形成隐状态 `h_2`，语言头把它映射为词表上的 logits，再由 softmax 得到每个候选词的概率；损失随后查询其中 `r1` 的概率。隐状态承载供预测使用的表示，本身并不是词表概率。下面用一般位置 t 写出同一过程：`h_t` 经语言头得到 `z_t`，再归一化；对长度为 T 的序列，`m_j` 说明位置 j 的标签是否有效。

```math
z_t=W_U h_t,\qquad p_t=\operatorname{softmax}(z_t)
```

```math
\mathcal L=\frac{\sum_{t=0}^{T-2}m_{t+1}\left[-\log p_t(x_{t+1})\right]}{\sum_{t=0}^{T-2}m_{t+1}}.
```

公式中的 t+1 正是刚才的移位：在 t=2 处，读出分布 p₂ 预测输入中的 `r1`，是否计分由位置 3 的 mask 决定。本例最后留下五项误差，分别对应 r1、r2、FINAL、a 和 EOS，再除以这五个有效目标。分母不统计未监督的 prompt，也不统计没有下一目标的末尾 EOS 读出。这里假定有效目标数大于 0；空监督批次按训练器约定跳过或返回零贡献。跨样本、microbatch 和设备怎样聚合，仍由实际框架规定。

# 因果移位与一次参数更新

代码把同一对齐关系落实到张量：从 logits 去掉末尾没有目标的一项，从 labels 去掉开头没有前一位置预测的一项，余下位置逐一配对。交叉熵中的 `ignore_index=-100` 去掉不计分的目标，随后 backward 计算梯度，optimizer.step 用它更新参数。示例由外部代码负责移位和损失，所以调用模型时不传 labels。它只覆盖标准 logits 输出，省略混合精度、分布式聚合、梯度累积和 scheduler；概率混合等其他读出需要按实际输出含义构造损失。

```python
optimizer.zero_grad()
logits = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
).logits                           # [B, T, V]

prediction = logits[:, :-1, :]     # [B, T-1, V]
target = labels[:, 1:]             # [B, T-1]
loss = F.cross_entropy(
    prediction.reshape(-1, prediction.shape[-1]),
    target.reshape(-1),
    ignore_index=-100,
)
loss.backward()
optimizer.step()
```

# 监督屏蔽与梯度传播

沿着这条样本看，预测 r1 的位置是 p3，所以 r1 的损失会先传到 p3 的隐状态，再沿生成这个状态的计算回到可训练参数。这解释了一个容易误会的地方：把 prompt 标签设为 `-100`，只是不要求模型预测问题里的词，并没有把问题的表示从计算中拿掉。误差仍可从回答目标出发，经过参与回答计算的 prompt 表示继续回传。

预测 r2 时，模型还可以通过因果注意力读取 p1、p2、p3 的 K/V，回答误差也可能沿这些路径回传。最后一个 prompt 位置 p3 还直接承担首个回答 token r1 的预测。因此，“prompt 不计分”并不意味着生成其表示的参数没有梯度。整数 token ID 本身不求梯度，能够更新的是由这些 ID 生成表示并继续计算的可训练参数。

真正决定路径是否保留的，是参数冻结、detach，以及缓存怎样接入计算图。比如，仅改变标签 mask，不会自动切断某份 K/V 与上游参数的联系；若某处显式 detach，则要从那个位置重新判断可回传范围。因此，标签 mask 说明误差从哪里产生；要判断误差传到哪些参数，还要继续看实际计算图。

# 正确前缀与自产前缀

为什么完整过程已在输入中，仍可训练“预测过程”的能力？关键是因果可见性：预测 r1 的位置只能看到 p1、p2、p3；预测 r2 的位置则已经能看到正确的 r1。Teacher forcing 为每个目标提供正确的历史前缀，因果 mask 阻止它读取目标自身及之后的信息。于是，各位置可以在同一轮前向中并行计算，却仍分别学习下一 token 预测。

部署时，这条历史不再由数据提供。模型先从问题预测第一个 token，再把自己实际生成的 token 接回输入，才继续预测下一个。如果第一步没有生成正确的 r1，后续就不再是在刚才训练 r2 的那个正确前缀上计算。teacher-forced loss 衡量给定正确历史时的预测，完整解题还依赖模型能否自己产生并继续使用历史；前者下降并不自动保证后者改善。KD 与 OPD 的学习也需追踪前缀来自哪里。

## 只监督答案是否等价于 Direct

再沿同一条样本看一个更细的区别：如果把 r1、r2 的标签改成 `-100`，但仍把正确的 r1、r2 放在输入里，预测答案 a 的位置仍然能够读取这段过程。更一般地，即使只保留答案监督，只要正确过程仍在输入中，训练就仍以这段过程为条件，不能自动视为没有过程输入的 Direct 训练。两者可以对相同答案计分，模型获得的信息却不同。下面用条件分布把这一点写清楚。

### 条件分布与理论下界

将问题 p1、p2、p3 合记为 X，提供的正确过程记为 C，答案记为 A。保留 C、屏蔽过程标签时，答案预测仍以 X、C 为条件；Direct 则只以 X 为条件。因此，应比较下面两个目标，而不只比较它们的监督 mask。为看清信息条件本身的差异，下面限定有限离散分布，并允许模型表示任意条件概率；实际有限容量、有限数据的训练未必达到相应理论下界。

```math
\mathcal L_{\mathrm{answer\ only}}
=\mathbb E[-\log p_\theta(A\mid X,C)],
\qquad
\mathcal L_{\mathrm{Direct}}
=\mathbb E[-\log q_\theta(A\mid X)].
```

在这个理想模型类中，真实条件分布本身也是可选模型。交叉熵分解为条件熵与期望 KL 后，选择真实条件分布就能把 KL 降到零，所以两类输入条件各自对应一个最优条件熵。把它们相减，得到的恰是过程 C 对答案 A 提供的条件互信息：

```math
\inf_p\mathcal L_{\mathrm{answer\ only}}=H(A\mid X,C),
\qquad
\inf_q\mathcal L_{\mathrm{Direct}}=H(A\mid X),
```

```math
H(A\mid X)-H(A\mid X,C)=I(A;C\mid X)\ge0.
```

取一个极端但可手算的例子：所有问题 X 相同，答案 A 是等概率二元变量，提供的 C 又恰好等于 A。看见 C 的模型已经知道答案，最优 CE 因而为 0；只看 X 的模型仍面对两个等概率答案，最优 CE 为 ln 2，约 0.693147。监督都只保留答案，也没有消除输入信息的差别。反过来，若 C 完全由 X 决定，条件互信息可为零，模型容量、计算过程和优化难度仍可能不同；因此，这个理论差值不能直接当作实际性能增益。

所以，工程记录不能只保存“哪些标签被屏蔽”。还要保存实际输入 token、attention mask、移位后的 labels 和监督位置数，才能还原一次预测看见了什么、又因什么目标产生误差。评测时进一步确认 C 是人工提供、教师生成还是模型自产，才知道测的是哪种条件下的能力。当前 TaH2 的移位与损失入口见 [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py)、[loss.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/loss.py)。

# 历史源码索引

以下保留原审计包 `final12_audit_20261103/` 中的源码索引，供需要追溯旧实现时查阅。当前框架建设按新选择的参考源码推进。

| 需要核对的问题 | 原包中的入口 |
| --- | --- |
| Full 与 Direct 的 target、prompt mask 和 EOS | `evidence/matrix/contracts.py` 的 target 与 encode |
| labels 怎样与下一 token 对齐 | `evidence/matrix/official/tah2/model/tah_model.py` 的标签移位 |
| 有效位置、概率累计与损失归一化 | `evidence/matrix/official/tah2/model/loss.py`，随后连接 `train/simple_loop.py` |

上述索引来自 [P002 独立审阅第 2 节及证据索引](https://chatgpt.com/library?file_id=libfile_a90ac85ecfb88191985c7ae410c74ac0&account_id=personal)。它们描述历史执行内容；后续源码学习注明实际选用的版本。

```codex-prompt
请从本页的输入与监督对齐开始教学。先解释最后一个 prompt 位置怎样预测第一个回答 token，再解释 prompt 的 labels 为 -100 为什么不意味着它没有梯度。把公式、张量位置和当前框架结构对应起来；涉及具体实现时读取实际选用的源码，不把教学示例当作已执行代码。沿我的疑问补充必要背景，再回到框架的模块与调用关系。
```
