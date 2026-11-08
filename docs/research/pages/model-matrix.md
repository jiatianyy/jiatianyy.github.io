选择基座时，我们实际要继承一套已经形成的计算方式，而不只是下载一份权重。循环发生在哪些层、每轮怎样读取输入、状态怎样保留，都会决定后训练能改什么。同样是整栈共享，原生 Ouro 与在 Qwen3 上改造的 TaH2，也有不同的训练起点、状态更新和读出规则。

因此，下面先用结构矩阵看清各模型怎样计算，再确认公开检查点处于哪个阶段，最后检查目标、梯度、缓存与生成接口。代码和权重各自固定版本；实际加载、梯度及生成结果用来判断这套计算能否在研究中使用。

[研究总览](https://chatgpt.com/space/page_e25611bb2c00819195e9c2f172dda6da) · [进入架构理论](https://chatgpt.com/space/page_d88b723779a48191b389ec0da58ab6e8)

> [!CALLOUT]
>
> **按四步选型。** 先比较架构，再确认权重阶段与固定版本，随后核对执行接口，最后估算预算。论文提供的理论价值与检查点的工程可用性分别判断。

# 架构与训练来源矩阵

“原生”表示循环结构参与了预训练；仅仅拥有预训练权重不满足这个条件。Done Right I 是一组受控模型与非循环对照，不能作为单一结构名称。

| 论文 | 原生循环与稠密性 | 循环作用范围 | 输入和状态的关键区别 |
| --- | --- | --- | --- |
| [Huginn](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal) | 是；Dense | prelude–recurrent core–coda | 单独随机初始化状态，每轮注入固定 prelude 表征 |
| [Done Right I](https://chatgpt.com/library?file_id=libfile_1060bf6ca4348191bb4b729c27a0bf69&account_id=personal) | 原生循环组与非循环对照；Dense 和 MoE | 整栈、中间块及多种状态变体 | 分别控制输入注入、状态初始化与共享模块 H/L；新增发布的 split H/L 单列 |
| [Loopies](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal) | 是；MoE | 每层局部执行两次再进入下一层 | layer-loop；沿 Qwen3-MoE 设计从头训练，不是改造 Qwen3 权重 |
| [SMELT](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) | 是；MoE 配方 | 中间一半层组成整块，整块执行两次 | 循环残差缩放及预算重配；不直接等同完整 Huginn |
| [LoopMoE](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal) | 是；MoE | prefix–loop block–suffix | 取消每轮 prefix 重注入，以 IterAdaLN 区分循环步 |
| [Ouro](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal) | 是；Dense | 共享整栈循环 | embedding 初始化，逐轮 LM 读出和退出门控 |
| [TaH2](https://arxiv.org/html/2609.35748v1) | 非原生循环改造；Dense | 完整 Qwen3 骨干共享、token 级自适应重复 | 首轮原 embedding；后续学习 updater 融合原 embedding 与 hidden；DUO 跨轮 KV；停止权概率混合 |

拓扑以 Done Right I 的三个轴展开：迭代作用范围、输入注入、循环状态组织。Loopie 的逐层执行顺序单列；Dense／MoE、训练深度和退出策略都是另外的维度。固定两轮的原生模型也不能据此默认支持任意深度外推。证据位置：Huginn 第 2 至 6 页；Done Right I 第 3 至 6 页；Loopies 第 4 至 8 页；SMELT 第 5 至 8 页；LoopMoE 第 3 至 6 页；Ouro 第 5 至 8 页。

TaH2 是普通 Transformer Base 的循环推理后训练实例。它与 Ouro 都有整栈共享这一拓扑特征，但训练来源、状态更新、DUO 可见性和读出不同；不能仅凭拓扑同族替换权重或算法。论文里的 Ouro／Huginn 对照由同一 Qwen3 Base 改造，不是原生发布模型的直接横评。[TaH2 结果与基线身份](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb) · [完整工程课程](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

# 公开检查点候选

下面登记公开发布入口和模型阶段，尚未进行本地加载、梯度或显存验收。它们是可继续核查的候选，不是已确定的唯一研究基座。

| 具体检查点 | 结构 | 发布阶段与许可证 | 接入时首先核对 |
| --- | --- | --- | --- |
| [huginn-0125](https://huggingface.co/tomg-group-umd/huginn-0125) | Dense；Huginn | 原生 Base；Apache-2.0 | 自定义实现、循环步数、状态与 KV |
| [Ouro 1.4B](https://huggingface.co/ByteDance/Ouro-1.4B) | Dense；整栈循环 | Base；Apache-2.0 | 深度、退出、prefill 与 decode |
| [Ouro 2.6B](https://huggingface.co/ByteDance/Ouro-2.6B) | Dense；整栈循环 | Base；Apache-2.0 | 与 Thinking 权重分别登记 |
| [IFM dense-huginn-336b](https://huggingface.co/IFM/LoopedLM-P1-dense-huginn-336b/tree/e87780591bf62f453299f5056bb7f51884d84b84) | Dense；Huginn 配方 | Part I 原生预训练检查点；Apache-2.0 | 固定revision=e8778059…；native xLLM loader；端点与权重分开 |
| [IFM moe-huginn-500b](https://huggingface.co/IFM/LoopedLM-P1-moe-huginn-500b/tree/f1321d5272a11348d5093ad7ab1d310eb9de2e11) | MoE；Huginn 配方 | Part I 原生预训练检查点；Apache-2.0 | 固定revision=f1321d52…；完整／非embedding口径与MoE执行分开 |
| [Loopie 6B-A0.6B preview](https://huggingface.co/zitian-gao/Loopie-6B-A0.6B-preview) | MoE；逐层循环 | 数学与科学推理 preview；MIT | 对应 Base／SPT／RL 的具体阶段待核 |
| [Loopie 20B-A2B preview](https://huggingface.co/zitian-gao/Loopie-20B-A2B-preview) | MoE；逐层循环 | 数学与科学推理 preview；MIT | 对应 Base／SPT／RL 的具体阶段待核 |
| [TaH2 1.7B max2](https://huggingface.co/nics-efc/TaH2-1.7B-max2/tree/e965dff2bd539b5be8087126b7d26bcb47e1b159) | Dense；普通 Qwen3 Base 上全栈循环改造 | 已后训练；Apache-2.0 | 非原生 Base；revision=e965dff…；DUO／gate／mix 与固定 TaH2 入口 |
| [TaH2 1.7B Standard](https://huggingface.co/nics-efc/TaH2-1.7B-Standard/tree/bbf0d82946c8e242ab2dc1f3be20408e317562ec) | Dense；单次骨干对照 | 同源 Standard 后训练权重；Apache-2.0 | revision=bbf0d829…；与 max2 保持数据和评测口径对应 |
| [TaH2 1.7B max4](https://huggingface.co/nics-efc/TaH2-1.7B-max4/tree/817b7bd1822f0b17dbb469c525c31892de1d475b) | Dense；普通 Base 循环改造 | 已后训练；Apache-2.0 | 保存深度4；revision=817b7bd…；完整版本与加载契约见生成课 |
| [TaH2 1.7B max8](https://huggingface.co/nics-efc/TaH2-1.7B-max8/tree/88dc9b1d8d3f58e3c94dc8b794a78337f38d3a65) | Dense；普通 Base 循环改造 | 已后训练；Apache-2.0 | 保存深度8；revision=88dc9b1…；推理改深度与重新训练分开 |

Ouro 候选的主要运行障碍是依赖兼容性。固定发布版本的 `UniversalTransformerCache` 与 Transformers v4.55.0 存在构造时的属性冲突，还缺少 mask-size／layers 接口约定；配置中的 `transformers_version` 不能保证兼容。这一结论只适用于该版本组合，候选仍属于原生 Dense Base。[固定依赖、触发路径与最小验证](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b)。

TaH2 的 Standard、max2、max4、max8 已固定到官方不可变 revision，配置和侧文件清单已核对。三个循环版本是在普通 Qwen3 Base 上后训练得到的权重；Standard 是同源单遍对照。完整四版本表、模型卡旧 import、配置覆盖反例和算法 R1 见[加载与生成课](https://chatgpt.com/space/page_0626be2568f8819187b536e037f9a6a9)。权重尚未实际载入和数值验收。

Done Right I 的[固定版本发布指南](https://github.com/ifm-ai/xllm-loop/blob/755c7b3a91081fe0f83500eabd53bd2490b8b745/release/paper-part1/README.md)列出 17 个配方、25 个检查点，包含非循环对照和论文之外发布的 split H/L。全部仓库已记录不可变 revision，表中的 Dense／MoE 候选另核对了 6 份 config、manifest 和 index 小文件。

接入时还需注意：原生生成会重算完整前缀且不使用 KV 缓存；BF16 发布文件不含 optimizer 和数据加载状态；当前 loader 的数据顺序与论文旧版不同。权重尚未加载。[完整调用链、论文与 release 差异及后训练接入条件](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8)。

Loopie 的两个 preview 由论文直接链接，但模型卡未明确它们处于 Base、SPT 还是 RL 阶段，因此暂不视为纯 Base。两份配置在 head_dim、top-k 概率归一化和 embedding tying 上也有差异。[两个 HF revision、三个代码 fork 及各自的 KV 处理](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a)见专页。

公开 Megatron 通用 RL 入口使用 token ratio，并不支持丢组后的补样。论文中的 GSPO／DAPO 配方还需专用启动链，不能仅凭 `train_rl.py` 的存在判断已经完整实现。

SMELT 附录 A 明确说明完整架构和训练栈不能公开。LoopMoE 的 ACL、arXiv 和作者主页中，尚未核实归属明确的官方实现或权重，同名 Foil 项目已排除。因此，两篇目前用于学习理论与实验范式，不登记为可直接后训练的公开 Base。详见 [SMELT 的资源与复现边界](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921)和 [LoopMoE 的官方来源与缺项](https://chatgpt.com/space/page_13f174d58dd08191b37781d5c4da6099)。

# 后训练前需要固定的执行条件

拿到候选权重后，还要把一次后训练更新真正走通。例如，同样叫 labels 的输入，有的接口要求调用方先移位，有的会在 loss 内部移位；同样叫输出概率，也可能指给定一条内部路线的分布，或对路线求和后的分布。若这些含义没有对齐，训练代码即使能运行，也可能优化了另一个目标。下表把这些差异对应到具体模型，便于在接入时逐项确认。

| 候选或理论材料 | 已核到的条件 | 开始后训练前要定下的接口 |
| --- | --- | --- |
| Huginn | 截断梯度、锁步深度与 speculative 分支的条件保留；分片本地组与外层 lr 赋值还存在独立字典边界 | 训练／带图深度、ELLISAdam及实际组学习率；fresh／resume；普通生成或明确命名的推测策略 |
| Ouro | labels、step、threshold与weighted exit有优先级；默认loss内部移位；物理早退与全展开缓存不同 | 论文或HF目标；依赖版本；选择后target、mask和分母；真实短路与缺失KV |
| Done Right I | 共享调用进入一套AdamW；Part I scale=1及全局FSDP裁剪；token加权目标与microbatch日志不同 | checkpoint recipe、参数共享、loss分母、实际裁剪／step事务和日志估计量 |
| Loopies | per-token flag与prompt间接梯度保留；公开RL是token ratio，动态过滤开关被拒绝；preview阶段待核 | Base／SPT／RL身份；mask与辅助损失；序列GSPO归约、响应等权和持续补样 |
| SMELT | 有序残差、gain、检索重合度的条件保留；Fig.9校准层级和ICL展示筛选已补；完整栈未公开 | 用于参考架构、预算与机制诊断；记录任务权重、chance校正及评测seed，不假定已取得公开Base |
| LoopMoE | 完整gate不能由Eq.10直接删除；动态调制不能静态折叠；BBH原Average与等任务均值不同 | 官方实现／权重仍待归属证据；router、shared experts、norm和shape；评测聚合权重单列 |
| TaH2 | 路由、停止混合、文字边缘和真实访问概率不同；三种续算收益分别定义；eval 执行集合与预统计分母可能错配 | checkpoint与有效decider；读出／logp、服务覆盖、独立快照；实际参数组／scheduler horizon；eval出现记录、U_D与V_E |

需要检查某项条件时，可回到[七篇专页](https://chatgpt.com/space/page_e25611bb2c00819195e9c2f172dda6da)和[后训练交接表](https://chatgpt.com/space/page_860625e31d1c8191a35d6ab14da2abbb)。论文来源、代码与权重版本、训练阶段和许可证分别登记；通过静态核对的候选仍需运行验收。

# 参数与预算要分别记账

总参数、非 embedding 参数、一次物理遍历的激活参数、跨轮参数应用量和实测 FLOPs 含义不同。尤其不能把模型名中的训练 token 端点当作参数规模，或用激活参数直接估计需要存储的全部权重。

同一篇论文也可能使用多种参数统计。LoopMoE 同时报告 FLOPs 等效活跃参数与物理活跃参数，SMELT 也区分 physical active 与 compute-equivalent sparsity。因此，每条比较都要注明原论文的统计定义，并核对具体检查点配置；仅凭模型简称无法判断所需资源。[LoopMoE 第 6、13 页](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal) · [SMELT 第 5 页与附录 A](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal)

SMELT 的“100M／200M／600M／1.6B”是 Baseline active 非 embedding 规模档位；S=0 也不能自动解释成 SMELT 为 Dense。Loopie 的主要训练比较按实测 optimizer-step 时间匹配，SMELT 则近似控制算术 FLOPs、参数和 KV；两种成本约束必须分别保留。具体数值与误差见 [SMELT 预算表](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) 和 [Loopie 硬件配方](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a)。

训练预算还要与并行配置区分。TaH2 的 [TPP=2](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb)表示训练 token 预算与参数量之比；TP 表示张量并行度，DP 表示实际数据并行度，数据配置还另有整除要求。[T12 的八组检查](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c#b445b124-6da5-430c-896a-5b1c0faf4646)只验证控制流和损失算术，检查点能否加载、训练和生成仍按前表的运行证据判断。

# 七篇论文的内容索引

论文专页按问题与贡献、计算图、训练、结果与消融、实现、推理接口展开。六篇材料的 83 项核对、47 组数学与实现说明及 13 段逐行算法可逐项查阅；TaH2 专页对应原文 12 张表、13 幅图和附录 A–E，并连接各工程课中的固定函数。

| 专页 | 已落实的论文核心 | 实现边界与后续核对 |
| --- | --- | --- |
| [Huginn](https://chatgpt.com/space/page_e2a842b7e2dc81919ba8e52efff12790) | 2–4–2、SandwichBlock、初始化、随机深度与末 8 轮反传、路径独立与状态轨迹、KV／连续计算／自推测 | 官方训练与 HF 版本分别固定；scalar num_steps、labels 对齐、mask 和缓存策略逐项说明 |
| [Ouro](https://chatgpt.com/space/page_d4e92e84c2508191a9de921ca171a48b) | 整栈共享、hazard／PDF／CDF、两阶段 gate 目标、7.7T 与 Thinking、机制实验、可达性定理、faithfulness 与 scaling 附录 | 官方 HF 的 loss、退出执行与 KV 不等于论文全部机制；项目页链接的训练复现单列 |
| [Done Right I](https://chatgpt.com/space/page_4b1be7042b8c8191b0c8b435a140eef8) | 三轴 Q1–Q3、对角写入、shared H/L、Dense 和 MoE 主表、路由观测与冻结干预 | 17 recipes／25 checkpoints 含 FF 和新增 split H/L；native 无 KV，恢复状态和数据顺序有边界 |
| [Loopies](https://chatgpt.com/space/page_29412e31ceac8191b5ac7a933efd966a) | 逐层两次、实测 step 时间配方、PT→Base→SPT→RL、数据和超参、主结果与消融 | Megatron／Transformers／vLLM 各自追到函数；preview 阶段未明，HF／vLLM 缓存路径分别核验 |
| [SMELT](https://chatgpt.com/space/page_0acf2f6c73c48191b460da5f519a7921) | 中间块两次、残差缩放、三预算误差、WSD 分叉、规模律、域／长度／ICL 与第二次访问机制 | 完整架构与训练栈明确不公开；保留论文语义算法，不能当作已开放 Base |
| [LoopMoE](https://chatgpt.com/space/page_13f174d58dd08191b37781d5c4da6099) | IterAdaLN 全部计算顺序、非对称 gate、容量配平、完整组件消融、路由与调制统计、效率 | 作者官方实现和权重仍未核验；排除同名作品，精确 router／训练／KV 接口留有明确缺项 |
| [TaH2](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb) | 同源 Qwen3 后训练基线、DUO、updater／decider、动态深度标签、停止混合；全部主表、消融、扩展与成本 | 固定源码的路由覆盖、辅助探测、损失分派、缓存与 MiniSGL 执行；论文与 release 统计及成本口径分别保留 |

# 六篇材料入口

| 材料与固定论文版本 | 英文与中文 | LaTeX 源码 |
| --- | --- | --- |
| Huginn 2502.05171v2 | [英文](https://chatgpt.com/library?file_id=libfile_6d154ab975d48191aa062920173ef911&account_id=personal) · [中文](https://chatgpt.com/library?file_id=libfile_b8dfe8757dc081918e7fe18d22bd0f33&account_id=personal) | [中英文工程](https://chatgpt.com/library?file_id=libfile_b141bb3260dc8191b906b226e2186c01&account_id=personal) |
| Done Right I 2610.23036v1 | [英文](https://chatgpt.com/library?file_id=libfile_1060bf6ca4348191bb4b729c27a0bf69&account_id=personal) · [中文正文与英文附录](https://chatgpt.com/library?file_id=libfile_7cbbe217b54c8191bb446ce9905eda20&account_id=personal) | [中英文工程](https://chatgpt.com/library?file_id=libfile_2410551480d88191a5a4dc1f24629d71&account_id=personal) |
| Loopies 2607.16051v3 | [英文](https://chatgpt.com/library?file_id=libfile_13d199dd3da88191b5c11b17f1a02a66&account_id=personal) · [中文初稿](https://chatgpt.com/library?file_id=libfile_8b9cd4eca358819192f10a31adac22b9&account_id=personal) | [中英文工程](https://chatgpt.com/library?file_id=libfile_1b4a5cb3fa5881919dd7b71a819d9059&account_id=personal) |
| SMELT 2609.01343v3 | [英文](https://chatgpt.com/library?file_id=libfile_441e8247c16c8191bf405aeee8adbe38&account_id=personal) | [英文工程](https://chatgpt.com/library?file_id=libfile_a0ec3bd20a188191913c40d6651653bf&account_id=personal) |
| LoopMoE EMNLP 2026 | [英文](https://chatgpt.com/library?file_id=libfile_4368bc33ef9c81918ea2c25e99cd9d35&account_id=personal) · [中文正文](https://chatgpt.com/library?file_id=libfile_004a872603608191ae136e07cc90a89f&account_id=personal) | [中文重排工程](https://chatgpt.com/library?file_id=libfile_3d78e34b10048191ac1eba01d6ea2a07&account_id=personal) |
| Ouro 2510.25741v6 | [英文](https://chatgpt.com/library?file_id=libfile_fb17828dba3c819189295587bcf1500b&account_id=personal) · [中文正文与英文附录](https://chatgpt.com/library?file_id=libfile_dbc79166c4f88191a5880609306633c2&account_id=personal) | [中文重排工程](https://chatgpt.com/library?file_id=libfile_f73606b0b8fc81919778e7df1caf5aec&account_id=personal) |

本节链接的 LaTeX 源码包用于论文排版；训练和推理代码见前面的独立入口。Loopies 中文初稿存在训练预算与测试时预算混译，相关结论以英文为准。Ouro 与 LoopMoE 的中文重排工程也不是官方英文 TeX。[目录与使用说明](https://chatgpt.com/library?file_id=libfile_ff2d41241da481919b240ff4f2c8d8ef&account_id=personal) · [源码归档索引](https://chatgpt.com/library?file_id=libfile_25a371460fec8191abe896188f25b643&account_id=personal)

# 从候选进入实际研究

| 核验层次 | 完成所需的证据 |
| --- | --- |
| 论文与结构 | 对应英文版本、计算图、参数共享与状态规则 |
| 权重与实现 | 真实发布地址、许可证、Base／SPT／Thinking 阶段与 revision |
| 本地可用 | 固定依赖后的载入、确定性前向和保存回读 |
| 后训练可用 | 可训练参数、梯度、目标计算与生成概率的一致性 |
| 比较可用 | 明确数据、训练与推理预算，实际测量与评测记录 |

实际基座由研究问题、权重阶段、可用接口和实测资源共同决定。确定候选后，依次完成上表的加载、后训练和公平比较检查；理论材料与已可运行的模型继续分别使用。
