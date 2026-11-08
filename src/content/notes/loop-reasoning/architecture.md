---
title: "循环架构设计理论"
description: "理解循环架构，先要回答一次预测执行了哪些计算。本课先用 Done Right I 的三个设计轴定位模型，再定义状态、执行顺序、读出和缓存，最后用消融与预算条件解释实验结果。Ouro、Huginn、逐层循环与 MoE 模型提供具体对照。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8"
sourcePageId: "page_d88b723779a48191b389ec0da58ab6e8"
sourceSha256: 579d7eb0cb76b5f05b24651072e983c8584e3ce627038319d9d4dd7530547768
series: loop-reasoning
references: []
---

<span id="source-block-cc761836-a221-4984-ae07-b0344f87a6ed"></span>

理解循环架构，先要回答一次预测执行了哪些计算。本课先用 Done Right I 的三个设计轴定位模型，再定义状态、执行顺序、读出和缓存，最后用消融与预算条件解释实验结果。Ouro、Huginn、逐层循环与 MoE 模型提供具体对照。

<span id="source-block-51ba43a1-3cf4-4ef0-a92c-da9504bbe193"></span>

[研究总览](/notes/loop-reasoning/overview/) · [基座矩阵](/notes/loop-reasoning/model-matrix/) · [原生预训练](/notes/loop-reasoning/pretraining/)

<span id="source-block-933eb763-9015-4092-86bf-f648c03f8c85"></span>

> [!CALLOUT]
>
> **阅读顺序：设计轴 → 状态与前向 → 读出与缓存 → 算法 → 结果比较。** 第一遍先重建 Ouro 和 Huginn 的完整计算图；第二遍再检查梯度之外的执行细节与比较条件。目标是解释组件做了什么，以及实验支持哪一种解释。

<span id="source-block-671f068b-d60d-4d7a-9115-36d04c0a75bc"></span>

## 推荐阅读顺序

<span id="source-block-dd53a9ba-41a8-489a-b48a-4f8c7bb5e112"></span>

| 单元 | 核心问题 | 阅读与学习产物 |
| --- | --- | --- |
| 完整前向与拓扑 | 一次预测究竟执行了哪些模块 | Done Right I 对照 Ouro、Huginn；各画一张完整前向图并写出展开次序 |
| 状态与输入注入 | 原始输入与迭代状态如何分工 | 对照初始化、注入、归一化与残差；把动机和消融证据逐项对应 |
| Dense 与 MoE | 参数共享和专家稀疏选择怎样共同改变计算 | Loopies、SMELT、LoopMoE；分别登记总参数、每步激活和整条轨迹访问的参数 |
| 位置与 KV | token 深度与循环深度如何进入可见性和缓存 | 重建 prefill 与 decode 两张读写表；分析复用条件 |
| 组件消融 | 去掉组件后究竟改变了什么 | 完成一份组件卡：动机、作用机制、消融、代价、替代解释 |
| 跨模型比较 | 优势依赖哪个预算和训练条件 | 建立分条件性能表，不把不同设置下的最好成绩合成排行榜 |

<span id="source-block-cd13bf38-3466-4ce4-83d3-5595b24d3d4e"></span>

## 三个设计轴

<span id="source-block-6dbeb3b7-a248-4ef5-877a-85829aedb3de"></span>

“让模型循环”还没有规定一个完整架构。以整块重复为例，我们至少还要决定：每轮是否重新提供题目表示，上一轮状态怎样进入下一轮，哪些模块只在开始或结束时执行。这些选择会改变模型实际使用的信息，即使重复的层完全相同，也可能得到不同计算。

<span id="source-block-c2682a09-7c48-405d-8596-0147f9c6a6c1"></span>

Done Right I 因此把设计拆为 iteration envelope、input injection 与 recurrent-state design，分别对应迭代作用范围、输入注入和循环状态组织。Ouro、Huginn 可以放在这三个轴上看作不同组合；模型是否从头进行循环预训练、是否使用 MoE，则是另外的问题。[英文论文第 3 至 4 页](https://chatgpt.com/library?file_id=libfile_1060bf6ca4348191bb4b729c27a0bf69&account_id=personal)

<span id="source-block-c71fc173-25d9-458f-a906-9447804a4f4c"></span>

| 设计轴 | 首先写清什么 | 要解释的差异 |
| --- | --- | --- |
| 迭代作用范围 | 重复整栈、中间共享块，还是每层先局部重复 | 输入和输出各经过哪些一次性模块，执行顺序如何变化 |
| 输入注入 | 原始输入只在起点进入，还是每轮重新进入 | 注入位置、加法或拼接、投影与归一化 |
| 循环状态组织 | 单路状态、多个状态及其读写关系 | 初始化、跨轮保留、状态交互与输出读出 |

<span id="source-block-9fe592d0-0c7f-4e5d-aaf6-37621d3daf6f"></span>

## 数学模型与执行语义

<span id="source-block-87150cf9-578c-4db0-915f-7252598ebb0c"></span>

### 数学对象与完整状态

<span id="source-block-154d086a-f443-4e7d-8544-bafc6ba4d850"></span>

为了描述一次循环，可以设想在第 r 轮结束时暂停模型。继续运行所需的量，除了当前隐藏表示，还可能包括已经写入的 KV、下一轮哪些位置仍然活动，以及多路状态的内容。我们把这些跨轮保留的量合称为完整状态 S。这样，后面讨论“继续计算”或“复用缓存”时，就有了明确的对象。

<span id="source-block-ec3f9b31-c31a-4c48-831b-7e38e41e3680"></span>

固定模型版本和执行规则后，批大小、文本长度、隐藏宽度和词表大小分别记为 B、T、d、V；i 表示样本，t 表示文本位置，ℓ 表示物理参数层，r 表示循环轮次，K 表示实际深度。t 增长意味着读入下一个 token；r 增长意味着文字前缀不变时继续内部计算。

<span id="source-block-f2543713-fd40-4e62-93ec-43f0a6da09d8"></span>

```math
\begin{aligned}
x&\in\{0,\ldots,V-1\}^{B\times T},&
E&=\operatorname{Embed}_{\theta}(x),\\
E,\ H^{(r)},\ U&\in\mathbb R^{B\times T\times d},&
\mathcal S^{(r)}&=(H^{(r)},\mathcal C^{(r)},\mathcal M^{(r)}).
\end{aligned}
```

<span id="source-block-2a60424e-7e91-479c-9a11-909d27587c12"></span>

E 是原始嵌入，U 是可选的固定输入表征，H 是当前循环表示。花体 C 包括 KV、地址及有效性；花体 M 包括继续执行所需的其他状态，例如活动位置和多路状态。随机初始化记为 ξ。这里为便于比较，先使用相同宽度；若实际宽度不同，投影必须显式写进计算图。

<span id="source-block-7367c332-3435-4df6-b3be-aa364b2eef2e"></span>

```math
\begin{aligned}
\mathcal S^{(0)}&=\operatorname{Init}_{\theta}(E,U;\xi),\\
\mathcal S^{(r+1)}
&=\mathcal F_{\theta,r}
 \bigl(\mathcal S^{(r)},U;\mathcal V_r\bigr),\\
z_{i,t}^{(K)}&=\mathcal G_{\theta}
 \bigl(\mathcal S^{(K)}\bigr)_{i,t},\\
q_{i,t}^{(K)}&=\operatorname{softmax}\bigl(z_{i,t}^{(K)}\bigr).
\end{aligned}
```

<span id="source-block-908802dd-374a-40f5-bd97-3ab2902cbb57"></span>

这组递推把一次预测分成三件事：初始化状态，反复更新状态，再从所得状态读出词表分布。G 包含实际的 coda、归一化和词表读出；位置 t 的分布用来预测下一个 token。共享 θ 说明重复使用同一组参数，r 仍可通过步数编码或可见性规则影响每轮计算。

<span id="source-block-63a9b49b-b9b4-40f1-a4d1-7954c8de8be5"></span>

因此，后面分析迭代性质时，需要同时说明参数、输入和执行规则是否固定。只有将外部依赖固定并把继续运行所需的量纳入状态，才得到一个确定的迭代系统；具体网络中的归一化、路由和缓存仍由各模型给出。

<span id="source-block-971d781b-e4c9-4ed7-a574-daa834cc5111"></span>

### 拓扑与输入注入怎样改变计算

<span id="source-block-905601e6-d1f4-4d91-a36e-cfc25e88c1c4"></span>

在固定可见性和位置规则、把注意力的内部状态操作包含在 F 中的约定下，可以先只画隐藏表示路径。下面的式子是结构示意，归一化、残差、输出头和退出机制须继续按各论文展开。

<span id="source-block-4504f704-98f4-460f-9a1f-ef7e5e789232"></span>

```math
\begin{aligned}
\text{Ouro style:}\qquad
&H^{(0)}=E,\quad H^{(r+1)}=F_{\theta}(H^{(r)}),\\
\text{Huginn style:}\qquad
&U=P_{\theta}(E),\quad H^{(0)}=I(\xi),\\
&H^{(r+1)}=F_{\theta}\bigl(A_{\theta}(H^{(r)},U)\bigr).
\end{aligned}
```

<span id="source-block-67127b26-345c-4ff5-b0e3-3138d9fed3ef"></span>

Ouro 用整栈共享计算迭代输入表示；Huginn 在 prelude 与 coda 之间迭代共享 core，并通过适配器反复注入固定的 prelude 表征。这里 I 表示论文规定的随机状态初始化，A 表示其输入适配；实际随机尺度和归一化不能由示意式推断。[Ouro 第 5 至 8 页](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal)；[Huginn 第 2 至 6 页](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal)

<span id="source-block-08efb97b-95eb-40cd-bc55-ab48afcf8242"></span>

输入注入给下一轮提供了两条信息来源：一条是上一轮加工后的状态，另一条是仍可直接访问的题目表征 U。状态会随着循环改变，但 U 在这次前向中保持相同，因此后续计算不必只通过已经多次变换的状态来读取题目信息。这就是“保留问题信息”在计算图中的具体含义；它在任务上是否有益，再由相应消融判断。

<span id="source-block-972beb13-dc68-4a21-9c0e-57cfab850648"></span>

这条通路本身也是设计选择。LoopMoE 采用 prefix–loop block–suffix，却取消每轮 prefix 重注入。因此，仅知道模型重复中间块，还不足以确定它每轮怎样访问原输入。[LoopMoE 第 3 至 6 页](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal)

<span id="source-block-75d5e644-06c2-4385-99cc-ec3b1e7b673e"></span>

### 轮次条件化对应不同的设计目标

<span id="source-block-b6e3c720-f4ad-4733-86cd-3a2e987492b6"></span>

Huginn 结合重复输入注入、随机初态和不显式输入轮次，目标是学习可继续运行更多步的迭代算子。论文观察到固定点、轨道和漂移，这些现象尚不构成全局收敛证明。

<span id="source-block-c9a2687d-6bb0-485a-82c8-5ed98c2e2a89"></span>

LoopMoE 则用 IterAdaLN 同时依赖轮次和当前 token 状态，在固定四轮训练及预算控制下增强各轮的功能差异。两种设计解决的问题不同，不能据某一篇的消融推断所有模型都应采用同一种轮次编码。[Huginn §3.1、§7](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal) · [LoopMoE §3.3、§5](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal)

<span id="source-block-26d84034-645c-4804-bf78-920116c3dcdd"></span>

```math
\begin{aligned}
\text{Huginn-style:}\quad&H^{(r+1)}=F_\theta(H^{(r)},U),\\
\text{iteration-conditioned:}\quad&H^{(r+1)}=F_\theta(H^{(r)};c_{r}),\qquad
c_r=C_\theta(r,H^{(r)}).
\end{aligned}
```

<span id="source-block-be06b6a6-3bf9-4cf1-a921-a7ca93c08443"></span>

第一行在固定 U 下反复调用同一个状态映射；第二行让映射显式依赖轮次。两者都共享参数，因此“共享参数”并不要求每轮使用完全相同的状态更新函数。有轮次条件化，既不能据此推出稳定性，也不能据此排除深度外推；还要检查训练深度、未见轮次上的条件编码及实测结果。

<span id="source-block-ac4b36c7-863a-4788-b85a-3052a496a2f2"></span>

### 执行顺序决定复合函数

<span id="source-block-d97800a3-3467-4d11-b034-145c4cf076e7"></span>

先只看两层。整块重复两遍时，模型依次执行第一层、第二层，再回到第一层、第二层；逐层重复两遍时，则先连续执行两次第一层，再连续执行两次第二层。后一个方案中，第二层第一次接到的输入已经经过两次第一层变换，所以即使权重和调用次数相同，后续状态也可能不同。

<span id="source-block-cba5e3a8-5ca1-42db-bc23-96e12081013b"></span>

下面把这一区别写成任意 L 层、每层调用 K 次的复合函数。这里先假设每个物理层都是固定函数，以便单独考察执行顺序：

<span id="source-block-cd7b61ef-a24d-4652-90af-74ee7067cf81"></span>

```math
\begin{aligned}
F_{\rm block}
&=(f_L\circ\cdots\circ f_1)^{\circ K},\\
F_{\rm layer}
&=f_L^{\circ K}\circ\cdots\circ f_1^{\circ K}.
\end{aligned}
```

<span id="source-block-440e90d4-9680-486f-8afc-d7eb8ed2a7c3"></span>

两者拥有相同的参数集合，也可以具有相同的参数调用次数，但通常不是同一个函数。Loopie 的“每层执行两次再进入下一层”属于后一种顺序；SMELT 重复中间一半层，其循环范围另行限定。[Loopies 第 4 至 8 页](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[SMELT 第 5 至 8 页](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)

<span id="source-block-7d92ca57-87ef-4a77-b3ae-1f690056b78e"></span>

反例只需两个线性层。令先执行 M、再执行 N，并取 K=2：

<span id="source-block-a0dead8b-78e2-429a-b071-490a5ceee24c"></span>

```math
\begin{aligned}
M&=\begin{pmatrix}1&1\\0&1\end{pmatrix},&
N&=\begin{pmatrix}1&0\\1&1\end{pmatrix},\\
(NM)^2&=\begin{pmatrix}2&3\\3&5\end{pmatrix},&
N^2M^2&=\begin{pmatrix}1&2\\2&5\end{pmatrix}.
\end{aligned}
```

<span id="source-block-de5b1c50-3a15-4669-bd40-3867e95d26b9"></span>

同一输入 (1,0) 在两种顺序下分别得到 (2,3) 与 (1,2)。因此，相同参数量与展开层数不能证明拓扑等价；这个反例也不说明哪种顺序的任务性能更好。实际模型若还有轮次条件化、跨轮 KV 或多路状态，应比较完整的状态转移算子。

<span id="source-block-69eaa4e6-1cc3-447b-a658-54ebfc8ca92e"></span>

### MoE 的路由与参数使用

<span id="source-block-d71ed917-00d3-4f4c-bfa7-cc4ed5b645b5"></span>

MoE 让循环多出一项选择：这一轮调用哪些专家。router 读取当前隐藏表示，而隐藏表示已经被上一轮更新，因此共享同一套路由器和专家权重，仍可能在不同轮次选出不同专家。理解这一点后，就能区分“反复使用同一组专家”和“在后续轮次访问更多专家”这两种计算。

<span id="source-block-3544ea6b-0a18-4205-bed2-c9cc0a5ba486"></span>

先以 top-k 选择后在入选集合内重新归一化的路由为参照。e 是专家编号，a 是 router 分数；下面的集合记录选中了谁，α 记录它们怎样共同形成本轮输出：

<span id="source-block-0fdb97ba-e4b3-4a40-9b6a-bd1c60f9e830"></span>

```math
\begin{aligned}
a^{(r)}&=R_{\theta}\bigl(h^{(r)}\bigr),\qquad
\mathcal I^{(r)}=\operatorname{TopK}\bigl(a^{(r)},k\bigr),\\
\alpha_e^{(r)}
&=\frac{\exp a_e^{(r)}}{\sum_{j\in\mathcal I^{(r)}}\exp a_j^{(r)}}
\quad(e\in\mathcal I^{(r)}),\\
\operatorname{MoE}\bigl(h^{(r)}\bigr)
&=\sum_{e\in\mathcal I^{(r)}}
 \alpha_e^{(r)}\,E_{\theta_e}\bigl(h^{(r)}\bigr).
\end{aligned}
```

<span id="source-block-770edd53-93f5-49e2-b2c4-48bca2a4e8e4"></span>

这是一种明确的参考定义。具体模型还需核对 router 归一化顺序、shared experts、容量截断、负载均衡项及轮次条件化；这些选择不能从“MoE”三个字推出。[Loopies](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)、[SMELT](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)、[LoopMoE](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal)分别作为实例。

<span id="source-block-c4c47480-4b26-4cd3-8da1-7163c247d479"></span>

在不重叠的模块粒度上，令 P(n) 是第 n 次模块调用使用的参数标量索引集合；同一共享参数标量跨轮或以别名出现时使用同一索引。一次计算轨迹实际访问过多少不同参数，与这些参数累计被调用多少次是两个量：

<span id="source-block-a14912c2-283e-4a7f-a44c-b11cab1ad603"></span>

```math
\begin{aligned}
P_{\rm visited}
&=\left|\bigcup_{n\in\mathcal E}P(n)\right|,\\
A_{\rm param}
&=\sum_{n\in\mathcal E}|P(n)|.
\end{aligned}
```

<span id="source-block-d3778404-e265-4535-96ef-ba0a6f8a7b94"></span>

花体 E 是执行过的模块调用集合。循环复用可保持已访问参数量不变而增加累计调用量；MoE 也可能在后续轮次访问新的专家。全部驻留参数还包括本轨迹没有访问的专家。累计参数调用量只是一种记账方式，注意力长度、通信、激活和硬件效率仍使它区别于 FLOPs 与延迟。

<span id="source-block-b0f39915-d346-4068-a24a-2fe40d025584"></span>

### 读出分布与退出规则

<span id="source-block-67889214-fd17-418e-a912-c6c8a29fcdba"></span>

得到各轮状态后，还要确定怎样形成最终预测：可以读取固定深度 K 的末轮输出，可以按状态决定何时停止，也可以混合多轮预测。为看清这些选择的差异，先固定同一上下文，把第 r 轮 logits 记为 zᵣ，相应分布记为 qᵣ，再取一组非负且和为 1 的权重 wᵣ：

<span id="source-block-964fb672-d9b8-470d-9773-dbe0fcf59262"></span>

```math
\begin{aligned}
q_r(y)&=\operatorname{softmax}(z_r)_y,\\
w_r&\ge0,\qquad\sum_r w_r=1,\\
q_{\rm prob}(y)&=\sum_r w_rq_r(y),\\
\bar z&=\sum_r w_rz_r,\\
q_{\rm logit}(y)&=\operatorname{softmax}(\bar z)_y\\
&=\frac{\prod_r q_r(y)^{w_r}}
 {\sum_v\prod_r q_r(v)^{w_r}}.
\end{aligned}
```

<span id="source-block-d4ba119e-9ed8-49fa-9595-a47f9ed76141"></span>

如果先把每轮 logits 转为概率再平均，每一轮都拿出一份归一化分布参与结果；如果先平均 logits 再做 softmax，合并的则是归一化之前的分数。式子的最后一行说明，后一种操作相当于把各轮概率做加权几何组合，再重新归一化。这个等式可由代入 softmax、消去各轮归一化常数得到。

<span id="source-block-d0d97930-9627-42a6-a803-a9da14703cb1"></span>

用已有的二分类例子看，两轮分布为 (0.9,0.1) 和 (0.5,0.5)：等权概率混合得到 (0.7,0.3)，平均 logits 得到 (0.75,0.25)。同样叫“综合两轮预测”，实际给正确词的概率已经不同。后面定义训练损失或采样器时，必须延续这里选定的读出方式，并明确返回的是 logits、log-probability 还是混合概率。

<span id="source-block-a77c0cde-e717-42e3-9d18-0c73c983f333"></span>

在给定上下文和固定随机性下，若各轮预测已确定，先按权重 w 选择轮次，再按该轮分布采样 token，对轮次边缘化后恰好得到概率混合。硬阈值只规定执行到哪一轮；停止后取末轮预测，还是混合已执行轮次，仍需另行定义。[TaH2 的停止权重与读出实例](/notes/loop-reasoning/internal-loop/)

<span id="source-block-6df7e426-8ac0-4b14-bd8b-317a3747f839"></span>

选择退出信号时，还要区分状态稳定与任务正确。参数共享、跨轮变化变小和正确率提高并不相互等价：恒等更新可以从第一轮起完全稳定，却始终读出错误答案。因此，增加循环是否有效，最终仍要看固定评测协议下的质量与成本曲线。

<span id="source-block-bb4636df-a73f-4a7e-aeaf-ab76167af96e"></span>

#### 读出、采样器与评测器共同定义观测结果

<span id="source-block-0f226966-f88c-43cb-88c9-137a26de1631"></span>

结构比较需要把最终读出之后的两步也固定：从模型读出到实际文字行为的采样变换，以及从完整输出到分数的判分函数。TaH2 的固定 MiniSGL 依赖按“温度→top-k 归一化→top-p”执行；更换顺序可以改变支持与输出分布。[生成课 T11](/notes/loop-reasoning/generation/)已经给出支持改变时的误差界与反例。它不改变 Dense／MoE 或循环拓扑分类，但改变所比较的实际策略。

<span id="source-block-d741b1d3-2dc0-42eb-a40e-3df54c236bf7"></span>

评分规则也会改变结论。Huginn Fig.14 用“正确和的字符串是否出现在完整输出中”判定算术命中，并非最终答案精确匹配。Done Right Tables 2–4 给出十任务结果，其中 TriviaQA 使用 F1，不能直接与另一论文的 exact match 合并。比较性能时，应同时固定模型计算、实际采样与评分规则。[Huginn 判分与原图](/notes/loop-reasoning/huginn/) · [Done Right Q1–Q3 与指标](/notes/loop-reasoning/done-right-i/)。

<span id="source-block-8e349148-a706-465f-9c3a-395afe2f97ea"></span>

### KV 的可见性与复用条件

<span id="source-block-4abe3183-9692-4d62-abb5-6db604537ae7"></span>

缓存复用的是已经完成的一次计算。对于固定模型、位置规则和随机性下的因果前向，新 token 不改变旧位置的状态时，旧 K/V 就可以保存下来，供后续位置直接读取；这一前提的完整形式将在下面的等价命题中写出。

<span id="source-block-404d8914-4514-4a31-885f-41e4c0d41a68"></span>

循环模型增加了一个需要辨认的维度：同一个物理层会被访问多次，而各次访问的输入状态可能不同。同一组投影权重算出的 K/V，并不因此就是同一份结果。所以缓存地址需要注明样本、文本位置、物理层和循环轮次；多路状态或额外阶段还要增加相应标签。下式中的大写 K/V 是键和值，执行深度仍记为 K，缓存整体记为花体 C。

<span id="source-block-6c2fdf20-b1f8-4215-aa3b-2f37784925bf"></span>

```math
\mathcal C[i,t,\ell,r]
=\bigl(\mathsf K_{i,t,\ell,r},\mathsf V_{i,t,\ell,r},
       \pi_{i,t,\ell,r},\operatorname{valid}\bigr).
```

<span id="source-block-2b9a96a9-bc71-4c22-a4ae-4ea32fafdc37"></span>

π 是实际位置编码标识；轮次增加并不自动要求文本位置编码增加。对查询节点 n，可见的来源 m 必须同属一个样本、满足文本因果性、K/V 已按依赖图构造，并满足模型自己的深度、层和分支规则：

<span id="source-block-956adfe5-9f03-440a-83df-2fe61cddd425"></span>

```math
\begin{aligned}
\mathcal V_{\mathcal R}(n)=
\{m:\;&i_m=i_n,\ t_m\le t_n,\ \operatorname{ready}(m)=1,\\
&\operatorname{valid}(m)=1,\
 \operatorname{Allowed}_{\mathcal R}(m,n)=1\}.
\end{aligned}
```

<span id="source-block-59e5c9e0-dada-4460-ae36-fa2efe12f353"></span>

R 表示固定的执行规则。当前查询自身的 K/V 可以在注意力聚合前构造并进入 ready 集合；ready 不要求整个当前层已经执行结束。Allowed 负责区分同轮、历史轮、分支及物理层，不能统一替换成一个深度三角形。实际有效查询还应有非空的合法键集合。

<span id="source-block-4042c31e-8131-4674-958e-447fe44c4a6f"></span>

```math
s_{nm}=q_n^\top k_m/\sqrt{d_k}.
```

<span id="source-block-6eac0f4f-23a2-458d-9ab7-8aaecc70c719"></span>

```math
\begin{aligned}
\alpha_{nm}^{\rm attn}
&=\frac{\exp s_{nm}}
 {\sum_{j\in\mathcal V_{\mathcal R}(n)}\exp s_{nj}},\\
\operatorname{Attn}_{\mathcal R}(n)
&=\sum_{m\in\mathcal V_{\mathcal R}(n)}
  \alpha_{nm}^{\rm attn}v_m.
\end{aligned}
```

<span id="source-block-b24be9f7-62d3-4240-8bec-c8c4898497d0"></span>

带 attn 上标的 α 在可见键集合上归一化，MoE 权重则在专家集合上归一化。上式已将位置变换包含在 q、k 中，并以没有额外位置偏置的注意力为参照；使用相对偏置时，需要在分数中补齐。

<span id="source-block-79b1c7b6-67fd-43e8-804e-67b0624c1837"></span>

复用已算出的 K/V 是缓存计算结果；跨轮合并、平均或替换 K/V 则改变读取规则，需要重新定义参考计算。Ouro 的 prefill 与 decode 是具体案例。[Ouro 第 18 页](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal)

<span id="source-block-413c4e0a-5b8b-473b-982f-a8657a51ac8a"></span>

**条件命题：缓存解码与完整前缀重算何时相同。** 固定参数版本、深度或可重放的路由、位置和可见性规则，并固定同一组随机初态及随机算子抽样。再要求新增后缀不改变旧位置的状态，缓存保存的值及其来源与参考计算一致，当前节点执行相同算子。则在精确算术下，缓存解码与该参考计算得到相同 logits。

<span id="source-block-46c34ff4-6b7e-4160-a39e-f7fe8debbd7d"></span>

证明按当前计算图的拓扑次序归纳：初始输入及缓存来源相同；某节点的全部前驱结果相同，配合相同算子与可见性，其结果也相同；一直推进到读出节点即可。实际浮点实现按数值误差容限比较。若前缀重算重新采样旧位置的初态，或 MoE 容量截断随批组成改变，上述前提不满足，便不能要求逐点 logits 相等。

<span id="source-block-012930ea-e0a0-4312-980c-29d9cc6a90da"></span>

算法 A2 正是这条命题的执行版本：先检查复用前提，再复用旧节点；对发生变化的依赖重算。更新模型参数、修改深度或位置规则后，必须重新判断缓存有效性，不能沿用旧张量而只改配置。

<span id="source-block-fff8c2c4-d83a-4dd8-92eb-79b89c32d77a"></span>

把缓存等价命题用于真实模型之前，首先要确认缓存对象能够按约定工作。例如，Ouro 固定自定义 cache 与 Transformers v4.55.0 存在无 setter 属性及 `get_mask_sizes/layers` 接口冲突。这属于特定版本的对象协议问题，不改变上述条件命题。[Ouro 依赖约定](/notes/loop-reasoning/ouro/)。运行 A2 前，应依次检查对象构造、地址容量、mask 尺寸、追加和重排，再比较数值结果。

<span id="source-block-98a3f35e-79a6-4530-bde3-3359a601629d"></span>

### 从数学对象到执行算法

<span id="source-block-dd5cee27-21a3-4eac-bb03-46450c281693"></span>

算法 A1 把前面的状态递推真正执行一遍：先形成输入表示和初态，再按轮次注入输入、调用共享 core，最后经过 coda／读出得到 logits。这里固定深度、只保留单路循环表示并在末轮读出，因此可以先专心看清一次调用怎样接到下一次；逐层循环、双路状态和自适应退出需要改写相应调度。

<span id="source-block-c50c7b41-60e9-4115-9e6a-cd026f12bef3"></span>

A2 进一步回答生成时怎样利用已经算过的前缀。它先检查旧缓存是否仍对应同一个参考计算，复用有效结果，再补算当前缺少或已经失效的节点。两份算法由此连接起来：A1 说明完整计算应该得到什么，A2 说明在满足复用条件时，怎样避免重复完成其中一部分计算。

<span id="source-block-c85bf7d9-50fb-4eb9-882f-4c380180f34c"></span>

| 数学对象 | 工程表示 | 必须保持的关系 |
| --- | --- | --- |
| H、U 与有效位置 | `[B,T,d]` 逻辑张量及 padding mask | 固定输入表征与当前循环表示分开保存 |
| 参数共享 | 多次调用同一模块参数 | 权重相同，激活与调用位置仍各自独立 |
| 花体 C 与可见集合 | 分层分轮 KV、position、valid、版本元数据 | 每个键能追溯到实际生产节点，mask 与参考图一致 |
| MoE 路由 | top-k 索引、权重、dispatch 和 gather | 压紧后仍能还原到原样本、位置与轮次 |
| 每轮读出 | 明确类型的 logits／logp／probability | softmax、混合与目标提取的顺序正确 |
| 执行轨迹 | 轮次、层次、形状、路由与 KV 地址记录 | 跟踪不改动模型状态；成本统计含真实额外计算 |

<span id="source-block-a3ab8a6a-9d02-4b84-85d0-5ce65343c6f6"></span>

读源码时先找张量生产者、消费者和状态修改点，再核对框架中的相应调用。所有原论文组件都应能落到这张表中的具体位置；缺少对应关系时，先补计算图，再讨论组件的理论解释。

<span id="source-block-5016c0e6-1a8c-42ea-8b13-782604f0cfca"></span>

#### 算法 A1 固定深度共享块的参考前向

<span id="source-block-2878ec9e-5516-4f0f-ad89-0d4854323b12"></span>

**输入** token 批、有效位置 mask、固定执行深度 K、参数 θ、执行规则 R、固定初始化抽样 ξ

<span id="source-block-4abeed5f-8cd1-4c63-9813-d96b37069c49"></span>

**输出** 末轮 logits、完整执行状态、各轮路由与 KV 读写轨迹

<span id="source-block-83af311b-cc57-4f1a-844a-c0a31bc81aec"></span>

**执行约定 A1** 单路循环表示，固定 K，仅末轮读出；Prefix、Inject、Core、Readout 按具体模型定义。R 内含位置与可见性规则；多路状态、逐层循环和自适应退出另写调度。

<span id="source-block-f822108f-6c70-409e-9717-eae090d3d9a7"></span>

<figure class="research-embed"><iframe src="/research-assets/6d2ce0d8cba28ac1aeb95e3ec73c75e1748a51bcf6964510edfddd65bc60ea62.html" title="算法 A1 固定深度共享块的参考前向" loading="lazy" sandbox=""></iframe><figcaption>算法 A1 固定深度共享块的参考前向 · <a href="/research-assets/6d2ce0d8cba28ac1aeb95e3ec73c75e1748a51bcf6964510edfddd65bc60ea62.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-8276e461-ace9-48dc-a0b8-229f2a709d60"></span>

#### 算法 A2 按参考计算图执行增量解码

<span id="source-block-a7345732-3c7f-4b34-984c-033757dfe876"></span>

**输入** 完整 token 前缀（含历史 token 与本次新 token）、已有完整边界状态及缓存、参数版本、执行规则 R、相同的初始化与路由抽样

<span id="source-block-3b152ccb-5458-40fb-ab2a-d5c01dea474a"></span>

**输出** 新位置 logits、有效的新缓存及读写轨迹

<span id="source-block-efee7bd3-f653-48dc-8bf3-c50b31c1bbac"></span>

**执行约定 A2** 固定深度或可重放路由，因果参考图有限且无环。缓存版本、位置、随机性与参考计算匹配；遇到前缀状态变化则重算。这里描述语义，不规定融合算子或服务端调度。

<span id="source-block-daa2e87f-e214-4925-8b2a-20dc5e0dbd1d"></span>

<figure class="research-embed"><iframe src="/research-assets/be3360e418079f1fba61cf83ea4c3e1ce0b0d9d3da0bcd3c38bd3130de0d752f.html" title="算法 A2 按参考计算图执行增量解码" loading="lazy" sandbox=""></iframe><figcaption>算法 A2 按参考计算图执行增量解码 · <a href="/research-assets/be3360e418079f1fba61cf83ea4c3e1ce0b0d9d3da0bcd3c38bd3130de0d752f.html" target="_blank" rel="noopener">展开完整算法 ↗</a></figcaption></figure>

<span id="source-block-d742ec69-766a-496d-9c14-9c984178d80a"></span>

## 用 TaH2 区分结构名称 训练历史与可见性

<span id="source-block-8037b3ee-15aa-4931-b8aa-568f4f2eae10"></span>

TaH2 给普通预训练模型增加整栈循环、输入 updater、token-level decider 和扩展 duo-causal attention。它与六篇原生循环材料共同用于架构分析；相同的共享层范围仍未确定输入注入、缓存可见性、深度监督和读出分布。

<span id="source-block-dbbe43fc-930e-4b6e-b249-e8626485326f"></span>

令 a 为架构，ξ 为训练随机性，θ_Qwen 为共同普通预训练起点。两种权重形成过程分别是：

<span id="source-block-3bed2c97-7ca3-4bab-be69-616bba0f0b46"></span>

```math
\begin{aligned}
\theta_a^{\rm native}&=T_{\rm pre}^{a}(D_{\rm pre},\eta_{\rm pre},\xi),\\
\theta_a^{\rm adapted}&=T_{\rm post}^{a}(\theta_{\rm Qwen},D_{\rm post},\eta_{\rm post},\xi).
\end{aligned}
```

<span id="source-block-2b7e3c23-3321-494a-959f-c47a4cc89bd7"></span>

TaH2 论文中的 Ouro/Huginn 对照属于第二行。其结果回答在指定 Qwen 起点、数据和配方下如何引入循环，不能替代第一行原生 Ouro/Huginn 发布权重的性能证据。[TaH2 §3、§5.1、Appendix A.2](https://arxiv.org/html/2609.35748v1)

<span id="source-block-b3f6f0fd-0101-47e7-af63-bf78f939ba1f"></span>

### 相同循环深度不保证相同依赖图

<span id="source-block-cba20dcc-2d22-4222-bbc5-a52ea2f9e909"></span>

以主路径查询位置 (t,m) 为例，TaH2 DUO 允许读取已实际执行、满足 s≤t 且 j≤m 的 (s,j) 条目，同位置较早循环的 KV 也是合法历史。普通逐虚拟层缓存读取对应虚拟层的因果历史；两种依赖图不能仅靠更换 cache 索引互相替代。

<span id="source-block-8461b3d6-e96a-47d6-a4e5-51dcdf4ae528"></span>

```math
\mathcal V_{\rm DUO}(t,m)=\{(s,j):s\le t,\ j\le m,\ (s,j)\text{ 已在主路径执行}\}.
```

<span id="source-block-16657cfa-8b4a-4310-8533-ccdfe92e37fa"></span>

训练 lookahead 另有可见性例外：不能相互读取其他 stopped token 的 lookahead 条目。正确并行执行必须是目标依赖图的拓扑序，并保持输入、位置、可见集合和随机状态；shape 一致不足以保证计算等价。详细 mask 和 side-pass 规则见 [一次内部循环](/notes/loop-reasoning/internal-loop/)，固定图与增量计算继续由 A1/A2 解释。

<span id="source-block-eec14bf0-cfd3-4659-9e63-0fedf51bd0d3"></span>

### 完整算子决定组件是否可替换

<span id="source-block-9b2d901d-c3b4-4c77-904f-032a8d834679"></span>

判断组件能否替换，需要比较完整模型，而不只是某个线性子式。归一化、共享专家、缓存和读出都可能影响等价性。下面四个例子分别说明：局部代数变形能证明什么，又在哪一步需要额外条件。

<span id="source-block-d127cef5-b9a8-4fcc-ba36-5df1d6e5d0b7"></span>

| 结构问题 | 能严格得到什么 | 需要避免的外推 |
| --- | --- | --- |
| LoopMoE 的 MoE 分支 gate | 在未归一化的 raw expert sum 中，标量 gate 可分配进各专家系数；完整条件见 [LoopMoE](/notes/loop-reasoning/loopmoe/) | 不能据此断言，在 normalized router、shared experts 或 post-RMSNorm 存在时，外部分支 gate 仍可被吸收并删除；张量可以融合计算不等于参数可静态折叠 |
| Done Right 的输入写入消融 | 当前替换同时改变旧状态衰减和新输入通路；见 [Done Right I](/notes/loop-reasoning/done-right-i/) | 单组替换不能分别识别两个机制的贡献；需要额外的交叉控制 |
| Ouro 的浅层读出与物理早退 | 全展开 KV 下可按依赖归纳证明 prefill/decode 等价；实际 token 早退会产生未计算的历史深层 KV 地址；见 [Ouro](/notes/loop-reasoning/ouro/) | 先算全部轮再选输出的质量曲线，不自动是物理短路后的质量—成本曲线 |
| SMELT 的跨轮表示与注意力相似 | 全概率质量 overlap 与 TV 有精确关系，Q/K 到 attention 输出需要范数与扰动条件；见 [SMELT](/notes/loop-reasoning/smelt/) | 平均 cosine 或共享专家身份相似，不能单独证明 KV 可直接复用 |

<span id="source-block-5d843908-2b5b-4275-8397-fb5c62dc8827"></span>

以 RMSNorm 为例，考虑把正标量 alpha 从归一化前移到归一化后。先省略两边相同的逐维可训练 gain，只比较缩放与归一化的顺序。记 r²=||x||²/d，并保留 epsilon：

<span id="source-block-cf1540b9-b27b-4044-a399-4a95621e12d7"></span>

```math
N(\alpha x)=\frac{\alpha x}{\sqrt{\alpha^2r^2+\epsilon}},\qquad
\alpha N(x)=\frac{\alpha x}{\sqrt{r^2+\epsilon}}.
```

<span id="source-block-2cfb6706-5307-43da-81f1-517549e5d1a4"></span>

对非零 x，即使取 epsilon=0，也只有 N(alpha x)=N(x)，并非 alpha·N(x)。因此“乘进专家系数后再归一化”和“归一化后乘分支 gate”一般不是同一个更新。具体结构是否仍有其他可调尺度补偿，应明确允许改变哪些参数与路径，不能仅据分配律认定 gate 冗余。[LoopMoE 原文 A.3 与完整条件核对](/notes/loop-reasoning/loopmoe/)

<span id="source-block-074837d0-7bba-477e-8a86-0b74863f113e"></span>

课程练习据此增加一个要求：每次声称某组件能删除、缓存能复用或循环能重排，都先写出包含归一化、路由、残差及状态读写的完整复合算子；给出相等所需的假设，再尝试构造最小反例。组件消融的性能证据与函数等价的数学证明分别保留。

<span id="source-block-53a57c5f-0c4d-4acc-a3cf-0936921965a0"></span>

## 六篇结构落到实际实现

<span id="source-block-424749dc-8a3d-4e5d-892e-e2c35e28991f"></span>

| 论文专页 | 要复述的结构机制 | 对照实现时检查什么 |
| --- | --- | --- |
| [Huginn](/notes/loop-reasoning/huginn/) | 固定 prelude 注入、随机初态、共享 core；残差和后的 norm | concat adapter、逐层 SandwichBlock、整段循环末 ln_f；训练与 HF 调用链 |
| [Ouro](/notes/loop-reasoning/ouro/) | 整栈共享；分支输出先 norm 再加 residual；逐轮 gate 和读出 | OuroModel 先执行全部循环，OuroForCausalLM 再选输出；L×R 的 KV 地址 |
| [Done Right I](/notes/loop-reasoning/done-right-i/) | 三轴受控变化、对角写入、shared H/L 完整更新顺序 | build_loop_execution_schedule、InputInjection、_run_recurrences；Part I／II 分开 |
| [Loopies](/notes/loop-reasoning/loopie/) | 每层的 Attention＋MoE 访问两次；同层权重共享 | 训练物理层 checkpoint、每次重新路由；HF 同槽写入与 vLLM 虚拟层地址分别核验 |
| [SMELT](/notes/loop-reasoning/smelt/) | prefix→middle→middle→suffix；两个子层各自缩放残差 | 论文语义算法、每个执行位置的 KV 与每次调用的算术预算；无公开作者函数 |
| [LoopMoE](/notes/loop-reasoning/loopmoe/) | iteration＋token 条件；MoE 条件使用 attention 更新后的状态 | 动态 norm、attention-only gate、零初始化边界、MLA／FFN 配平；具体实现仍缺 |

<span id="source-block-5a68e591-afcc-4e71-a24b-89025980b4a4"></span>

缓存复用条件要落实到地址与控制流。Loopie vLLM 以“物理层×访问次数＋访问编号”区分缓存，HF 路径另需验证；Ouro 在全部循环完成后选择输出；Done Right I 原生生成不使用增量 KV。Huginn 的深层 KV 补齐、压缩与连续 latent 属于额外策略，需分别定义其参考计算，不能自动套用精确缓存等价命题。各专页列出对应文件和分支。

<span id="source-block-3e5b5f44-26df-44bc-b658-51bfb99248c3"></span>

## 用消融形成可检验的解释

<span id="source-block-142608a6-8e4e-4eda-9da1-ac6f45870278"></span>

每个设计都要回答同一组问题：它试图解决哪一个具体困难；改变了计算图中的哪条依赖；论文用什么对照支持解释；还有哪些同时变化的因素；如果换一个拓扑，理由是否仍然成立。组件卡保持简短，把详细推导接在真正需要的地方。

<span id="source-block-364b6a1f-9845-4a9a-8d44-4e36bada2937"></span>

| 组件卡字段 | 要留下的内容 |
| --- | --- |
| 机制 | 状态、参数或读出规则发生的具体变化 |
| 证据 | 论文版本、节或图表、对照设置、指标与预算 |
| 解释边界 | 实验观察、作者解释、我们的待检验猜测分别记录 |
| 迁移问题 | 在另一个基座上成立需要哪些额外条件 |

<span id="source-block-639e60df-0f92-48c2-85c3-a5b23424aac6"></span>

## 性能比较的口径

<span id="source-block-8dd92015-0376-4e91-b8a3-117ec2adb85e"></span>

参数量、训练 FLOPs、训练墙钟时间、推理 FLOPs、延迟和 KV 内存是不同约束。先列原论文各自的比较，再讨论哪些条件已足够接近，可以支持跨模型结论。Loopies 的训练预算比较和 SMELT 的 FLOPs／参数／KV 匹配是两个不同案例，应各自保留原口径。[Loopies 英文第 6 至 8 页](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal)；[SMELT](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)

<span id="source-block-1583790c-13e5-4954-8567-f3c9a480099e"></span>

| 每条结果必填 | 作用 |
| --- | --- |
| 模型与权重版本 | 区分 Base、指令模型与推理后训练模型 |
| 训练数据和 token 数 | 判断数据差异是否解释成绩差异 |
| 总参数与激活参数 | 判断容量、稀疏度与重复计算的关系 |
| 训练预算及匹配项 | 写明 FLOPs、墙钟时间或其他约束 |
| 推理设置和成本 | 深度、采样、输出长度、prefill／decode 与实测硬件 |
| 评测协议 | 数据版本、提示、评分、样本数和统计不确定性 |

<span id="source-block-6b7fb9b8-ccb6-4d90-927d-0252e63d8e6b"></span>

### 保留控制条件的性能对照

<span id="source-block-7f4afd06-5b7c-4c35-9447-e987e65a8706"></span>

| 对照问题 | 原论文中的具体结果 | 可以回答什么 |
| --- | --- | --- |
| [Huginn：同数据、180B 端点](/notes/loop-reasoning/huginn/) | ARC-C：固定深度 26.96；循环模型 R=32 为 29.18，R=1 为 23.72 | 该对照中实际执行循环有贡献；固定深度 baseline 不是 800B 端点 |
| [Ouro：同一 Base 的深度改变](/notes/loop-reasoning/ouro/) | 1.4B MMLU 5-shot：R4=67.45，R8=64.49 | 超过最终四轮设置未必更好；早期训练曾用八轮 |
| [Done Right：730M Dense、336B](/notes/loop-reasoning/done-right-i/) | Q1 Ouro→Middle：MATH500 20.80→32.80；HumanEval+ 29.88→28.66 | envelope 有任务相关收益；同预算不等于各任务都提升 |
| [Loopie：四组实测 step 时间匹配](/notes/loop-reasoning/loopie/) | 八任务平均提升 1.1／0.6／1.7／2.2 个百分点 | 硬件效率与结构联合配方的收益；不据此宣称等 FLOPs |
| [SMELT：96 组近似三预算匹配](/notes/loop-reasoning/smelt/) | DCLM Completion 胜 96／96；Core 胜 83／96 | 两类指标的配对证据；共享训练轨迹、误差及统计范围须保留 |
| [LoopMoE：9B、300B 同数据预算](/notes/loop-reasoning/loopmoe/) | 九任务平均 Vanilla 41.22→LoopMoE 44.03 | 该尺度平均收益；两个尺度、单 seed 不足以建立一般规模律 |

<span id="source-block-1223207a-86fe-42cd-b482-932163a4fb83"></span>

这些结果分别保留各论文的 checkpoint、提示和指标定义。跨论文进一步比较时，须先对齐 MATH／MATH500、MMLU micro／macro、CoT 与生成协议、训练数据和推理成本；表中数值不合成为统一排行榜。各专页同时保留性能退步、组件反例与原文限制。

<span id="source-block-f92d3da5-8446-4fc2-98f6-5a6066c701c0"></span>

### 结果差异究竟估计什么

<span id="source-block-a9765ea5-1f4a-4f95-9a53-4bfac330928b"></span>

把架构、训练配方和训练随机性共同写入训练算子。下式只是定义评测对象，不假设训练可微。固定检查点的配对结果与对训练随机性取期望的架构效应，是两个不同的量。

<span id="source-block-30a3f9d0-d87a-4286-a260-98a7d02d37f0"></span>

```math
\begin{aligned}
\theta_{a,\xi}&=\mathcal T(a,\mathcal D,\eta,\xi),\\
\Delta(\xi_1,\xi_0;\kappa)
&=\mathbb E_{x\sim\mathcal E}
 [u(\theta_{1,\xi_1},x;\kappa)-u(\theta_{0,\xi_0},x;\kappa)],\\
\Delta_{\rm train}(\kappa)
&=\mathbb E_{\xi_1,\xi_0}[\Delta(\xi_1,\xi_0;\kappa)].
\end{aligned}
```

<span id="source-block-b5840a53-7bcf-4024-87c3-f9a37adcb5bd"></span>

这里 a 指待比较的架构，D 是训练数据，η 是优化与数据顺序等配方，ξ 是训练随机性，κ 固定提示、深度、解码和评分协议；u 是逐题评分，E 是评测总体。改变架构时若同时改变宽度、专家数或 batch，比较对象就是整套配方。Loopie 的硬件配方与 SMELT 的三预算配方都应这样描述。只有一个训练 seed 时，更多评测题目能减小条件于该检查点的评测误差，不能给出训练 seed 方差。[Loopie 的预算与结果](/notes/loop-reasoning/loopie/) · [SMELT 的配对训练谱系](/notes/loop-reasoning/smelt/)

<span id="source-block-1f291cf2-b9fd-4d9d-8dc0-4be45ffc56d1"></span>

在两检查点使用同一组独立评测样本、同一评分单位的前提下，令 dᵢ 为逐题得分差，可计算配对均值及其条件标准误。若 u 为 0／1 正确率，b 是仅模型 1 答对的题数，c 是仅模型 0 答对的题数：

<span id="source-block-7e0cddd7-9fab-411f-a12c-9b7cca9f5193"></span>

```math
\begin{aligned}
d_i&=u(\theta_1,x_i;\kappa)-u(\theta_0,x_i;\kappa),\\
\widehat\Delta&=\frac1n\sum_{i=1}^{n}d_i,\\
\widehat{\operatorname{SE}}(\widehat\Delta)
&=\sqrt{\frac{\sum_i(d_i-\widehat\Delta)^2}{n(n-1)}},\qquad n>1,\\
\widehat\Delta_{\rm binary}&=\frac{b-c}{n}.
\end{aligned}
```

<span id="source-block-da6fb3bb-26c9-4a2a-aab1-0bef07bc2443"></span>

上述标准误需要逐题记录；只有论文汇总分数，无法恢复 b、c 或逐题配对差异。此外，同一训练轨迹分出的多个端点彼此相关，重采样时应保留分组。SMELT 专页给出了具体的重采样单元与算法。

<span id="source-block-12725a6f-c900-4445-9002-ec2cc13f8637"></span>

任务平均还要固定权重。宏平均给每个任务相同权重，微平均给每个题目相同权重；两者只有在任务样本数相同等条件下才一致。

<span id="source-block-ae159f47-e52f-49cf-a98e-ca49ec06199e"></span>

```math
\begin{aligned}
\operatorname{Acc}_{\rm macro}
&=\frac1J\sum_{j=1}^{J}\operatorname{Acc}_j,\\
\operatorname{Acc}_{\rm micro}
&=\frac{\sum_j n_j\operatorname{Acc}_j}{\sum_j n_j}.
\end{aligned}
```

<span id="source-block-6adb9860-706d-4800-94cf-708d457ab9b0"></span>

Done Right I 的 MMLU 主表与路由冻结实验使用的汇总口径需要分别保留；LoopMoE 的九任务平均与五任务消融平均也不应直接相减。对应专页把差分限制在同一任务、同一列、同一协议内。[Done Right I](/notes/loop-reasoning/done-right-i/) · [LoopMoE](/notes/loop-reasoning/loopmoe/)

<span id="source-block-c144024c-82cd-45ce-8bf6-a9cbb1ac284f"></span>

### 原始分项、校准与聚合必须一起比较

<span id="source-block-f8fb1a4a-a6c7-4a9a-9a9f-4c34901f2c53"></span>

结构的性能结论还依赖“怎样把分项变成总分”。[LoopMoE 专页](/notes/loop-reasoning/loopmoe/)已补齐 Table 9 的全部九任务和 Table 10 的全部 27 个 BBH 子任务；后者的等任务均值 26.4133／27.6989／28.2744 与原表 Average 26.12／27.33／27.94 不同。原分数保留，聚合权重和有效样本数仍需原始评测产物解释，不能用自行重算的均值替换原表。

<span id="source-block-e5537cb7-8c95-4cdb-a17e-22f7775798ea"></span>

[SMELT 专页](/notes/loop-reasoning/smelt/)列出 Fig.9 中各规模、五类任务的原值，以及 Appendix C 图24的展示筛选条件。这里还需区分两种校准方式：先分别校准各类别再加权平均，或直接校准聚合分数 Core。若 Core 恰好是同一批任务的加权均值，这两种残差的差就完全来自校准曲线，而非原始任务分数：

<span id="source-block-491e1a1b-c6ad-4fce-b527-748358dc2821"></span>

```math
\begin{aligned}
C_{\rm core}&=\sum_jw_jC_j,\qquad\sum_jw_j=1,\\
r_{\rm task}&=\sum_jw_j\bigl(C_j-\widehat f_j(L)\bigr),\\
r_{\rm core}&=C_{\rm core}-\widehat f_{\rm core}(L),\\
r_{\rm task}-r_{\rm core}
&=\widehat f_{\rm core}(L)-\sum_jw_j\widehat f_j(L).
\end{aligned}
```

<span id="source-block-01374860-8a4b-4d97-b8a0-c9bd7e068d6a"></span>

因此比较表应附 `metric_definition`、任务权重、chance 校正、校准拟合单位、训练与评测 seed、筛选规则和原始任务版本。SMELT 的 14/16 ICL 展示结果以最大 shot 处正增益且配对置换 p<0.05 为筛选条件；配对单位是十个示例选择／排序评测 seed。它们不能直接代替十次独立训练，也不能把筛选后的平均值写成未经筛选任务全集的平均收益。原始表图数值与方法边界均在两篇专页逐项标注。

<span id="source-block-e8e12cc3-26b8-4c8b-9bba-958658b212e8"></span>

#### 具体结果中的评测协议差异

<span id="source-block-1fb3f022-5b60-4218-b1a2-4ba7f03fb5d7"></span>

| 材料 | 已保存的补核 | 比较时保留的条件 |
| --- | --- | --- |
| [Huginn](/notes/loop-reasoning/huginn/#source-block-db68a91e-e5ba-4f58-941d-9ba1631643bf) | 原文 v2、后续 v3 饱和入口和明确未用于论文的 task variants 分开 | 论文版本、harness/task、完整 prompt、few-shot、答案过滤器共同定义结果 |
| [Ouro](/notes/loop-reasoning/ouro/#source-block-0753f7f2-2b31-4056-ba0b-e41f5d65e3f6) | Thinking Tables12–13 的单轮、R4、网格最佳点与R8退化 | Base／Thinking 协议与 checkpoint 分开；有限网格最佳轮数不等于输入级停止策略 |
| [Done Right I](/notes/loop-reasoning/done-right-i/#source-block-6524266e-7ca5-45c9-b4c1-905d758d032b) | Fig.13 九任务 gap、七任务预算曲线和500B四任务汇总；native生成的实际调用数 | 训练端点与任务集合分别固定；返回文本截断不等于实际停止 forward |

<span id="source-block-bedf14c0-9cad-4b69-9f77-78aa233bdc35"></span>

同样，架构比较中的验证 NLL 必须对应同一有效监督集合。[TaH2 T12](/notes/loop-reasoning/systems-evaluation/#source-block-b445b124-6da5-430c-896a-5b1c0faf4646)把 raw 分子、helper 分母、执行后移位目标数和显示日志分开；它是实现核查，不能据此直接重判历史模型的原表成绩。

<span id="source-block-9b3b3020-7ada-4a2e-86d5-24624ed7ed4d"></span>

## 用完整前向检查理解

<span id="source-block-1dcffce5-6cbe-4f07-b789-fa8afb9bcb4f"></span>

以 Done Right I 的三轴为框架，对照 Ouro 与 Huginn 重建前向过程。结合 A1／A2，画出计算图，列出状态与缓存的读写，再解释执行顺序为何不可交换、缓存等价需要哪些条件。把这两类模型讲清楚后，再进入 MoE 和逐层循环。

<span id="source-block-0ddf908c-b375-49d6-a3c4-52bbd637ac27"></span>

```text
请从本页第一课开始，以 Done Right I 的三轴对照 Ouro 与 Huginn。按数学对象与假设、计算图与推导、算法 A1/A2、张量与源码对应的顺序讲解。明确 t、ℓ、r、状态、输入注入和缓存，复述块循环不等价于逐层循环的反例，以及缓存解码等价命题的全部前提。算法采用论文式行号和真实公式排版。原论文的设计动机与消融分别给出来源，教学推导与模型实测分别说明。
```

<span id="source-block-0d1a05cc-0995-448b-bf7d-3caac2141050"></span>

## 核心材料

<span id="source-block-54cc5e80-05f6-47fe-ba85-3c92fa7efd88"></span>

[Done Right I](https://chatgpt.com/library?file_id=libfile_1060bf6ca4348191bb4b729c27a0bf69&account_id=personal) · [Ouro](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal) · [Huginn](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal) · [Loopies](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal) · [SMELT](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) · [LoopMoE](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal)
