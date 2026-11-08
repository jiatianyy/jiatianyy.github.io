---
title: "循环推理研究总览"
description: "理解循环推理，可以先盯住一次预测：模型已经读到了当前上下文，在输出下一个词之前，还能怎样继续计算？循环模型让同一段网络再次处理已经更新的表示。参数可以保持不变，但输入给这组参数的状态变了，因而计算仍可能继续推进。研究的关键是，这些额外步骤是否把已有信息组织成了对任务更有用的表示，以及这种收益需要付出多少成本。"
draft: true
approved: false
kind: research
date: 2026-11-08
lang: zh-CN
sourceUrl: "https://chatgpt.com/space/page_e25611bb2c00819195e9c2f172dda6da"
sourcePageId: "page_e25611bb2c00819195e9c2f172dda6da"
sourceSha256: f111c4ea5cb98a3a9336913320e94a339961cb767926080674a8638ef271ba90
series: loop-reasoning
references: []
---

<span id="source-block-9477296d-22ed-4c43-a802-743e9f0d2d01"></span>

理解循环推理，可以先盯住一次预测：模型已经读到了当前上下文，在输出下一个词之前，还能怎样继续计算？循环模型让同一段网络再次处理已经更新的表示。参数可以保持不变，但输入给这组参数的状态变了，因而计算仍可能继续推进。研究的关键是，这些额外步骤是否把已有信息组织成了对任务更有用的表示，以及这种收益需要付出多少成本。

<span id="source-block-37af4ca3-0e48-473e-a47f-2b7510bfc10b"></span>

六篇原生循环模型论文从架构和预训练入手，研究怎样让模型一开始就学会这种重复计算。TaH2 则从普通预训练基座出发，通过后训练增加循环。把它们放在一起，是为了逐步回答三个相连的问题：额外计算怎样发生，训练怎样让这些计算有用，部署时又怎样决定算到哪里。具体基座和配方沿着这些问题选择。

<span id="source-block-189441ec-b25c-4e31-b276-0072f843822a"></span>

> [!CALLOUT]
>
> **建议从架构课开始。** 先回答模型怎样循环，再看循环能力怎样训练形成，最后研究后训练要改变什么。已有基础时，可直接从下方入口进入论文专页或 TaH2 工程课。

<span id="source-block-48c53b41-7552-41ef-963c-7a99b770203f"></span>

## 学习入口

<span id="source-block-57e3f8dc-db46-4057-b2ed-a355759237e9"></span>

| 阅读入口 | 要解决的问题 | 读完应能说明 |
| --- | --- | --- |
| [基座矩阵与七篇论文](/notes/loop-reasoning/model-matrix/) | 哪些是原生循环模型，哪些权重可用于研究 | 架构类别、检查点阶段、实现入口与可用条件 |
| [循环架构设计理论](/notes/loop-reasoning/architecture/) | 模型从输入到输出怎样计算 | 状态、前向顺序、组件作用与公平比较 |
| [原生循环模型预训练](/notes/loop-reasoning/pretraining/) | 循环能力怎样形成，训练怎样稳定执行 | 目标、完整与截断梯度、一次更新和训练交接 |
| [循环推理后训练](/notes/loop-reasoning/posttraining/) | 后训练应改善哪一种具体行为 | 问题诊断、监督与梯度、可部署的门控及成功判据 |
| [TaH2 与后训练工程](/notes/loop-reasoning/engineering/) | 论文方法怎样变成更新与生成 | 从数据、内部循环到蒸馏、RL 和分布式执行的调用链 |

<span id="source-block-b45b0feb-7e57-42f4-9bf1-208185a733f3"></span>

## 第一轮学习顺序

<span id="source-block-38108a19-1e5b-490a-a307-9bfe78282117"></span>

1. **先选阅读对象。** 用矩阵区分论文结构、模型实现与公开权重，并确认检查点属于 Base、SPT 还是 Thinking 阶段。

<span id="source-block-ba84f8f0-ae64-4183-a5f3-3d8f432f5e3a"></span>

2. **重建前向计算。** 以 Done Right I 的设计轴为框架，对照 Ouro 与 Huginn，再加入 Loopie 的逐层循环及 MoE 案例。

<span id="source-block-a4aa1445-4e43-4f9c-8c71-925124ff491f"></span>

3. **追踪一次训练更新。** 理解深度、梯度和稳定性的关系，再用 Loopie 与 Ouro 连接预训练和后训练。

<span id="source-block-3247810d-9ff0-4b52-94fe-b152d2444b57"></span>

4. **定义后训练问题。** 结合 TaH2，分别描述原生基座与普通基座改造的能力差距；固定可见信息、计算成本和梯度路径，再选择监督、蒸馏、偏好优化或 RL。

<span id="source-block-fc91757f-4d06-4203-a861-c4669551191c"></span>

从[架构第一课](/notes/loop-reasoning/architecture/)进入。先用 [Done Right I 的三轴与消融](/notes/loop-reasoning/done-right-i/)理解设计选择，再用 [Huginn 的共享核](/notes/loop-reasoning/huginn/)和 [Ouro 的整栈循环](/notes/loop-reasoning/ouro/)追踪状态、归一化、读出与缓存。A1／A2 分别给出参考前向与增量解码的步骤。

<span id="source-block-86265cf5-53bf-44a0-bdb8-4fad97ca5029"></span>

## 七篇原文与实现专页

<span id="source-block-bf37422f-d161-434a-aaaf-e31fe563fca0"></span>

| 专页 | 固定原文版本 | 实现与工件入口 |
| --- | --- | --- |
| [Huginn](/notes/loop-reasoning/huginn/) | 2502.05171v2 | [官方训练仓库 1ea7220](https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e)；HF权重另固定版本 |
| [Ouro](/notes/loop-reasoning/ouro/) | 2510.25741v6 | [官方1.4B发布实现 574fa66](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)；完整官方训练栈未取得，第三方复现单列 |
| [Done Right I](/notes/loop-reasoning/done-right-i/) | 2610.23036v1 | [xllm-loop 755c7b3](https://github.com/ifm-ai/xllm-loop/tree/755c7b3a91081fe0f83500eabd53bd2490b8b745)；17配方／25检查点分别标记 |
| [Loopie](/notes/loop-reasoning/loopie/) | 2607.16051v3 | [Megatron cb2daa2](https://github.com/zitian-gao/Megatron-LM/tree/cb2daa267d3c5458e75751b3f73a17672fe7a55a)；Transformers／vLLM固定版本见专页 |
| [SMELT](/notes/loop-reasoning/smelt/) | 2609.01343v3 | 原文附录A说明完整栈不公开；保留论文语义算法 |
| [LoopMoE](/notes/loop-reasoning/loopmoe/) | EMNLP 2026，2026.emnlp-main.2160 | [ACL原文](https://aclanthology.org/2026.emnlp-main.2160/)；官方仓库／权重尚未核验，同名作品不借用 |
| [TaH2](/notes/loop-reasoning/tah2/) | 2609.35748v1 | [TaH 78ddf9a](https://github.com/thu-nics/TaH/tree/78ddf9a2c25713fced03a736c5746161f6ee590c)；四个发布revision及R1见生成课 |

<span id="source-block-2b1dabed-8e07-4ea3-8a3c-df5fb71762dd"></span>

各论文专页从问题与结构进入，随后展开训练、主结果和消融，最后连接实现与适用边界。需要核对细节时，可用页内的章节、图表及附录索引定位；TaH2 的 12 张表、13 幅图和附录 A–E 也有逐项对应。

<span id="source-block-03360061-9862-46f7-973f-ff6ec3b7c3d9"></span>

## 核心结论与成立范围

<span id="source-block-0885956f-6837-460f-8cab-ceee7f8aed87"></span>

下表概括每篇材料应如何理解。完整数学推导、原始结果、评测协议和源码位置见各专页。

<span id="source-block-ca6bd241-bf31-4d26-ae8c-ca12cfb0000b"></span>

| 材料 | 要理解的核心结论 | 工程阅读重点 |
| --- | --- | --- |
| [Huginn](/notes/loop-reasoning/huginn/) | 共享核反复更新状态，每轮注入固定输入。前向深度与反传深度分别定义；观察到路径独立，并不保证全局收缩或答案正确。 | 从随机深度、末轮反传追到参数更新，再检查数据打包、KV 与退出规则。原论文96来源与YAML113项的差异见专页。 |
| [Ouro](/notes/loop-reasoning/ouro/) | hazard 决定停止分布，两阶段训练使用不同目标。知识组合收益与存储容量分别解释；可达性定理依赖其构造条件，Base／Thinking 使用各自评测协议。 | 对照13个数据来源及阶段配比，区分论文目标与HF加权logits；再追踪缓存、输出选择、真实早退和依赖接口。 |
| [Done Right I](/notes/loop-reasoning/done-right-i/) | Q1–Q3比较指定配方下的差异。输入写入、初始化和H/L状态分别分析；路由统计与冻结专家干预提供不同层次的证据。 | 先读十任务和MoE分组结果，再看42项数据权重、D2预算结转与恢复、共享AdamW、梯度裁剪及无KV生成。 |
| [Loopie](/notes/loop-reasoning/loopie/) | 每层执行两次，并按实测step时间匹配训练预算。PT、SPT、RL分别塑造不同阶段的能力，训练效率与最终Thinking成绩需分别比较。 | 三个代码fork承担不同职责。重点对齐loss分母、序列GSPO与通用GRPO、筛选补样、采样参数传递和每次访问的缓存。 |
| [SMELT](/notes/loop-reasoning/smelt/) | 三种预算近似匹配；6.8–18.0%的节省来自拟合前沿。训练分叉和ICL展示筛选决定统计单位，机制图尚不足以证明因果。 | 按有序前向、WSD分叉、16格曲线与配对重拟合阅读，再分析四尺度sink和固定KV干预。完整作者训练栈未公开。 |
| [LoopMoE](/notes/loop-reasoning/loopmoe/) | IterAdaLN按轮次与token状态调制，gate具有非对称作用。专家容量并集与实际调用量不同；组件交互、跨规模归因及BBH聚合各有条件。 | 先固定router、shared experts和调制张量的定义，再比较训练吞吐与decode时延。原BBH汇总与等任务重算并列，官方工件仍待核实。 |
| [TaH2](/notes/loop-reasoning/tah2/) | 在普通Qwen3 Base上后训练循环计算。单轮CE改善、混合预测收益、联合KV干预与最优停止是不同对象；53.1%指准确率—计算斜率的相对增幅。 | 从12表、13图和附录A–E追到T1–T12与D3，再核对参数组、学习率、eval分母和R1的四个发布版本。可选初始化的分布、dtype与共享相关性单列。 |

<span id="source-block-89f94248-a3ce-47cf-8f81-4c42a96d4e71"></span>

理解这些差异，还需要知道数据怎样形成训练序列。[预训练课](/notes/loop-reasoning/pretraining/)和 [TaH2 数据页](/notes/loop-reasoning/data/)分别说明文件采样倍率、token 预算、保留序列与有效监督量；[后训练课](/notes/loop-reasoning/posttraining/)讨论如何据此选择监督、蒸馏或 RL；[框架接口](/notes/loop-reasoning/framework/)则说明如何记录实际使用的模型配置和恢复状态。

<span id="source-block-f30e246d-0da5-40b8-af14-c9d71dcfc58a"></span>

## TaH2 论文与后训练工程

<span id="source-block-5962dbc6-e288-4a66-9a27-33dc3e692698"></span>

先在[TaH2 论文结果与实现](/notes/loop-reasoning/tah2/)理解方法与实验，再沿[TaH2 与后训练工程](/notes/loop-reasoning/engineering/)追踪调用链。重点包括 updater／decider 的 Jacobian、DUO 可见集合、混合预测与损失分派，以及生成调度、蒸馏和 RL 的概率接口、分布式反传与恢复状态。

<span id="source-block-5aae5c46-b39d-47ee-b99a-a2e86fc93758"></span>

[数据与监督](/notes/loop-reasoning/data/) · [一次内部循环](/notes/loop-reasoning/internal-loop/) · [完整 RL](/notes/loop-reasoning/reinforcement-learning/) · [研究设想](/notes/loop-reasoning/research-ideas/)

<span id="source-block-a94d6baa-60bf-4019-a6fd-a1768425f190"></span>

## 数学与算法的讲解方式

<span id="source-block-897580fc-18c4-4e15-94ae-f8b9d7e52b05"></span>

每个核心问题按同一顺序讲解：先定义数学对象与假设，再推导目标和梯度，然后给出算法步骤、张量与源码对应，最后用例子检查结论。命题的成立条件就近说明；实验观察、作者解释和待检验猜测分别表述。

<span id="source-block-18f8451e-56af-4e82-aa45-d875f4f9a007"></span>

| 理论与实现的对应 | 各课必须讲清的内容 |
| --- | --- |
| 状态与条件分布 | 定义域、索引、形状，以及历史信息和随机性的来源 |
| 目标与梯度 | 原始目标、估计器、代理损失；共享参数、输入注入与读出的梯度路径 |
| 参考算法 | 输入输出、初始化、执行顺序、循环与分支、停止条件和状态更新 |
| 实现语义 | mask、detach、KV 生命周期、随机数、归约与权重同步 |
| 结论与检验 | 证明或反例、源代码对应、教学数值核对与真实模型证据 |

<span id="source-block-8b633231-9d94-474e-89c4-04a0b91bc318"></span>

算法采用论文式行号与缩进，输入输出、条件、控制流和来源放在相邻正文中。既有三条课程、六模型、TaH2、T1–T12、D1–D3、SMELT 探针与 R1 加载算法均可按名称检索。算法正文、公式和注释分别使用 18px、22px、15px；源码与 MathML 检查、浏览器渲染检查分别记录。

<span id="source-block-04e86871-8203-47f7-bd2c-efe42e27e59d"></span>

## 尚待补证与运行验收

<span id="source-block-bc4ced45-ce06-445e-9504-98058f11312b"></span>

阅读论文、理解数学、核对源码和复现实验是不同层次。下表集中列出会影响模型选用或结论解释的证据缺项，以及补齐它们所需的条件。

<span id="source-block-d7a886d5-8934-4f57-9067-468078252872"></span>

| 材料 | 尚未获得的关键证据 | 解除条件 |
| --- | --- | --- |
| Huginn | 历史800B数据序列；固定本地参数组lr与外层重绑定的模型级轨迹；部分评测变体的论文归属 | 原始数据manifest／launcher和实际更新记录；明确评测版本 |
| Ouro | 完整官方训练栈；指定HF与Transformers组合的缓存兼容及数值验收 | 相容环境或明确修补版本；真实前向、缓存和退出trace |
| Done Right I | 当前loader不能直接重放论文旧数据顺序；BF16发布文件不含完整训练状态 | 历史数据顺序／优化器等状态；native模型加载与后训练验收 |
| Loopie | Stage1预算说法冲突、preview具体阶段、论文专用SPT／GSPO launcher；HF缓存路径未实跑 | 作者日志／勘误、权重阶段说明及真实启动配置；端到端KV对照 |
| SMELT | 完整架构、训练栈和公开权重不可用；逐端点原始数据与部分probe归约未公开 | 作者可授权工件或另行实现并明确身份；保留论文语义参考 |
| LoopMoE | 尚未核验可归属的官方仓库／权重；router精确契约、BBH聚合权重 | 官方发布与配置；task版本、有效样本数及aggregation记录 |
| TaH2 | 四个权重对象与发布数据revision已固定；完整权重／Arrow字节、有效组件配置、监督重计和模型级数值／性能仍待验收 | 按R1载入并固定有效配置；补齐benchmark manifest，执行既有后端和成本对照 |

<span id="source-block-5fed6794-6255-4808-9a75-3be80df508d5"></span>

阅读时可沿三条课程推进，遇到具体差异再回查专页。参数组、学习率、截断梯度、mask 与分母、采样、缓存及收益干预的详细分析，均放在相应计算环节。

<span id="source-block-a75c1b0a-f2ff-416e-a42d-caf31633b609"></span>

## 当前进度

<span id="source-block-26c9c0ad-6d82-491b-affe-2e3e0bcefdc1"></span>

| 验收层次 | 当前状态 | 证据入口 |
| --- | --- | --- |
| 原文与数学 | 七篇核心结构、训练目标、主结果、消融与附录均有逐项索引；推论附条件与反例，未公开项保留 | 七篇专页末尾覆盖表；三条课程 |
| 固定源码 | 函数、梯度、mask、KV、参数组、日程、采样与数据执行已对应；通用入口和论文历史运行分开 | 各专页函数索引、T1–T12、D1–D3 |
| 内容补实 | 原表、附录、既有推导与版本保留；补Ouro深度／cache、Loopie两类mask、TaH2导出EOS与恢复停止边界，以及Done Right全25发布revision | 模型专页与生成／更新／系统页；数学、原函数替身、实际小文件和未执行模型分别登记 |
| 引用与公式结构 | 本轮重读24页，393条内部链接／锚点及附件逐项可达；修正后的sink算法源码回读一致；既有149处MathML与原生围栏检查范围分别保留 | 核对包保留逐项快照与范围；不替代公式正确性或浏览器渲染验收 |
| 模型与性能 | 不以静态检查和小型替身替代权重载入、梯度、真实语料、分布式或GPU性能结果 | 各工程页原有未完成验收项 |

<span id="source-block-c76503f8-ba21-4c33-9c4d-b03b5e2f9287"></span>

阅读进度与实现进度分别记录：材料已公开、页面已整理和模型已复现，各自需要不同证据。具体候选见[基座可用性和已知边界](/notes/loop-reasoning/model-matrix/)。

<span id="source-block-8e29fb71-6e93-40ff-b609-c2a0770b3421"></span>

## 研究记录与维护

<span id="source-block-eacbe07c-36f8-4bad-89bb-8897992610f2"></span>

2026 年 11 月 7 日建立当前结构，11 月 8 日进入内容收尾。总览汇总当前结论和剩余证据，详细来源与核对包留在七篇专页及工程课程；[研究进展总览与历史记录](https://chatgpt.com/library?file_id=libfile_ec54e573cc948191bd1306c6d6922230&account_id=personal)保留历次决定、实验审计、Q1／Q2／Q3 和交接来源。历史实验不绑定当前基座与配方选择。

<span id="source-block-f24c5985-5554-4af2-92ce-1a48d43f0489"></span>

所属研究项目：[Loop & Hybrid Reasoning Research](https://chatgpt.com/g/g-p-6ae1e739d1b881918d1022bf250562c1/project)。

<span id="source-block-f1a4a77a-c1d5-4e6d-b8b4-8eb45d711a24"></span>

定时任务“研究进展周日汇总（四周试行）”已保存并启用：2026 年 11 月 1、8、15、22 日，北京时间 21:00。结果更新原总览，并在当前 dot 对话通知。每次更新后沿用此入口；不会自动新建四个研究对话或启动实验。
