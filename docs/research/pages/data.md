一条原始问题要经过教师回答、对话模板、监督区间和数据划分，才能成为训练批次。本课沿这条转换链定义数学对象和算法，再说明如何固定环境与数据版本；同一份产物将用于一次参数更新及后续蒸馏。

[返回后训练研究工作台](https://chatgpt.com/space/page_90a0e38077148191b6c40007fbb15133)

> [!CALLOUT]
>
> **阅读主线**：先把原始样本写成编码映射，再区分三种 mask，最后计算样本配比对应的监督 token 权重。数学定义与算法是主线；环境配置、源码索引和验收清单按需查阅。

# 运行环境与版本

TaH2 的固定提交是 `78ddf9a2c25713fced03a736c5746161f6ee590c`。该版本的 [pyproject.toml](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/pyproject.toml) 给出 `torch==2.9.1`、`transformers==4.57.3`、`triton==3.5.1`；TaH 使用另一组依赖。环境记录还应包含 Python、CUDA、驱动、GPU 和内核编译器版本。参考代码使用 Linux CUDA 环境，第一次内核编译与正式性能测量分开计时。

版本清单拟保存：代码 commit、模型 revision、tokenizer 文件与 chat template 的哈希、数据 revision 与划分哈希、最终展开的配置、依赖锁定结果。阅读源码时固定版本；升级依赖后重新核对缓存和分布式行为。[chat template 官方说明](https://huggingface.co/docs/transformers/v4.57.3/en/chat_templating)用于补充控制 token 和模板边界。

论文训练配方、各规模教师与发布数据的样本数／token 数对应表见[TaH2 论文结果与实现](https://chatgpt.com/space/page_51505cd26b54819180fb3ee0632da7bb)。其中 1.7B release 的 eval 为 955 条，论文写 1,000 条；三遍历 token 算术值与论文约数也保留各自口径。数据重建时应绑定具体 revision，不能把两组统计当成同一次历史运行。

## 发布数据快照与实际监督口径

官方数据仓库 nics-efc/TaH2-amteam-tool 固定为 revision e9088ae272a81e2fce720e585a005a1547964cfd。当前证据包括 Hub 的 176 项文件元数据，以及 17 份小文件的实际内容：README、三个 mix_report、六份 dataset_info、六份 state 和 provenance。小文件的 Git blob 与 SHA-256 已对应；Arrow 训练和验证分片尚未下载、重计。 [固定快照](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/tree/e9088ae272a81e2fce720e585a005a1547964cfd)

| 发布规模 | train行数 | train输入tokens，README／报告 | 报告train_supervised | eval行数 | train／eval Arrow分片 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1.7B | 273,195 | 1,099,413,406 | 943,087,650 | 955 | 19／1 |
| 4B | 638,817 | 2,586,857,725 | 该报告未列 | 1,000 | 44／1 |
| 8B | 1,282,124 | 5,192,783,145 | 该报告未列 | 1,000 | 88／1 |

1.7B 的 `mix_report` 分别统计输入 token 和受监督 token，后者约占前者的 85.781%。将报告数值乘以三次遍历，可得名义输入量 3,298,240,218、报告监督量 2,829,262,950。这两项都不直接代表实际训练执行量，也不直接代表标签移位后的 loss 分母：预处理、标签首位、DP 整除、packing 和实际更新数还会影响计数。应沿本页执行链与 T12 逐步核对；报告字段本身不等于重新扫描 Arrow 后的结果。

8B mix_report的total_tokens=5,196,736,781同时覆盖train与eval；它比README的train_tokens多3,953,636，不应把total写成训练量。4B报告明确按来源分层采样、seed=1234并做一次来源内长度交换；eval写为复制8B，固定发布清单中两者eval Arrow的声明LFS摘要、大小和state fingerprint一致。这是发布声明的一致性，尚未用实际Arrow字节独立证明。

[provenance.json](https://huggingface.co/datasets/nics-efc/TaH2-amteam-tool/blob/e9088ae272a81e2fce720e585a005a1547964cfd/provenance.json)映射 173 个工件；另外三项是 README、provenance 自身和仓库属性文件。它列出三份来源身份：TaH-plus/am-220k-8b@5148233e99acb5f4c37e9e7842d432dc23011b03、TaH-plus/amteam-tool@64a65e9f636d7d6bd554f3503949a33596d9ec97、TaH-plus/am-prompt-220k@d5f9019d1fa0fbbf28e4cd2408e88e1d6f02f534。这些映射说明发布工件的来源声明；来源仓库可访问性、教师模型版本和论文历史生成运行仍需各自的证据。

快照核对顺序是：固定repo与revision→逐个split按state._data_files的顺序列举分片→将文件名、字节数和声明摘要与Hub清单／provenance交叉核对→实际下载后再校验字节与重计行／输入／监督tokens→预处理与batch形成后另记最终有效labels。Dataset fingerprint不是完整文件内容的密码学证明；未下载状态下不把声明摘要标成已实测通过。

[循环研究\_收尾修正与数据快照核对\_20261108.zip](library-file:fde1_bGliZmlsZV9aZVcxanlUVXJnVGdwMm9HM2ZDVUdB_FileDrive_45288dfe06608191af94334ba8699511)

本轮收尾包：17份固定数据小文件、176项清单与173项provenance核对；Mano原文记号／协议、零质量来源、多头界与sink概率域修正；原始decider重载分支和显式控制组；G1–G8均通过。24页、393条内部链接与核对包保存前50份附件可达；修正后的算法源码回读一致。31文件包SHA-256：d158a03ea9c09f81cbf05584e9e2b8e568fc6d79a7185dc8e74f4ed97d299eea。检查未涉及Arrow有效载荷、真实Tensor／模型、GPU性能或浏览器排版。

## 处理缓存身份与内容版本

处理缓存的 key 由文件名和字节数、tokenizer 词表大小、chat template SHA1、预处理版本及截断和选择参数构成。源码有意省略绝对路径与 mtime，使同一数据可以跨挂载路径复用缓存。最终 SHA1 并不补足输入身份：每个数据文件的内容摘要、完整 token 到 ID 映射的摘要均未进入 key。 [缓存身份字段](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L100-L216)

令 F 表示实际编码结果、K 表示这些元数据构成的缓存键，一份可复用缓存需要满足：

```math
K(D,v,\tau,c)=K(D',v',\tau',c')
\ \Longrightarrow\
F(D,v,\tau,c)=F(D',v',\tau',c').

```

该实现只在数据与 tokenizer 的外部版本不变等约定下支持这一假设。同名、同大小但内容改变的数据文件，或词表大小与模板相同而 token ID 映射改变的 tokenizer，都可能保持相同 key 却改变编码结果。这是 key 未纳入相关字段的后果，不需要发生哈希算法碰撞。研究 manifest 应独立固定数据内容摘要、tokenizer 文件摘要与模板摘要；这些输入改变时切换处理缓存目录或使旧缓存失效。

meta.json 在 save_to_disk 完成后最后写入，因此它是该实现的完成标记；文件存在并不额外证明当前源文件内容与旧缓存逐字节一致。[完成标记与加载](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L219-L222)

# 原始样本怎样进入模型

## 源码入口

| 阶段 | 固定版本源码 | 必须追踪的结果 |
| --- | --- | --- |
| 教师回答与工具轨迹 | [regenerate.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/data/regenerate.py) 的 `generate`、`tool_rows`、`build` | 提示、回答、工具观测、截断状态；教师版本和生成参数由实验元信息补充 |
| 原始对话编码 | [data_prepare.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py) 的 `_tokenize_messages` | 模板后的 token 与 assistant 监督位置 |
| 已编码数据 | 同文件 `preprocess_pretokenized_batch` | `real_token` 与 `mask` 转为 `input_ids`、`labels` |
| 长度和批次 | 同文件的预处理与 [data_collator.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/data_collator.py) | 截断或过滤、有效长度、padding 与监督标签 |
| 因果目标 | [tah_model.py](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/model/tah_model.py) 的 `_shift_labels` | 模型内部只移位一次，记录首个回答 token 的预测位置 |

## 两种输入格式的不同语义

`_build_preprocess_fn` 先检测 conversations_key，再检测 real_token。若数据同时带有两种列，实际会进入原始对话分支，不能仅凭存在预分词列就认为绕过了模板。[分派顺序](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L429-L460)

| 输入分支 | 固定实现的处理 | 数据契约 |
| --- | --- | --- |
| 原始对话 | human/user 映射为 user，gpt/assistant 映射为 assistant，保留 system；缺字段或未映射角色被跳过 | 原生 tool/function 角色不在映射表中；工具观测需明确适配编码 |
| 已分词 real_token 与 mask | mask 的 Python 真值决定目标 token 或 −100 | 验证等长、合法 token ID 与二值 mask；代码本身没有自动补齐这些断言 |
| cutoff | 先截取前 max_length 个 token 与对应标签或 mask | 截断后可能失去全部监督目标，随后样本被丢弃 |
| filter | 超过 max_length 整条丢弃 | 来源保留率及监督长度分布随阈值变化 |

工具观测可以通过预分词 build 路径写入 real_token，再用 mask 区分输入与监督；因此 raw 分支缺少 tool 角色支持，并不等于发布工具数据被丢弃。切换 raw 与预分词格式时，应沿两条预处理路径检查角色、token 和 mask 的对应关系。[角色归一化](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L37-L43) · [消息转换](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L225-L239) · [两种预处理](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L349-L427)

原始对话编码还只监督索引大于 0 的 assistant turn；仅有首条 assistant 消息的样本不形成该路径的有效监督。模板渲染之前的角色过滤和截断之后的监督检查，都会改变最终保留分布。[模板监督边界](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L252-L281)

## 编码映射与三种 mask

把数据处理写成固定的编码映射。下面的 `mᶦⁿ` 标记有效输入，`mˢᵘᵖ` 标记被监督的目标 token，`τ,v` 分别固定模板与 tokenizer。

RL 的 action mask 另标记策略生成且需要计分的动作；工具观测可以是有效输入而不属于动作。三种 mask 形状相同也不能互换。

```math
\begin{aligned}
E_{\tau,v}(z)&=(x_{1:T},m^{\rm in}_{1:T},m^{\rm sup}_{1:T},\ell_{1:T}),\\
\ell_t&=\begin{cases}x_t,&m^{\rm sup}_t=1,\\-100,&m^{\rm sup}_t=0.\end{cases}
\end{aligned}
```

## 因果预测与监督位置

先看下一课会继续使用的甲样本：输入是 `[11,12,21,22,2]`，前两个 token 是问题，`21、22` 是回答，`2` 是结束符。数据端把监督标记放在目标 token 自己的位置，因此原始 `labels` 是 `[-100,-100,21,22,2]`。

模型却要从左侧前缀预测右侧的下一个 token。预测 `21` 的输出产生在输入 `12` 所在位置；随后，输入 `21` 的位置预测 `22`，输入 `22` 的位置预测结束符 `2`。最后一个输入位置没有下一个目标。于是模型内部把标签左移一次，得到 `[-100,21,22,2,-100]`，三个有效预测位置由此确定。

这解释了为什么要在移位后定义计分集合 I(z) 和监督数 n(z)：它们统计承担预测损失的输出位置。最后一个问题位置虽然属于 prompt，却负责预测首个回答 token；问题是否作为输入、某个输出是否计分，需要分别看输入 mask 和监督标签。TaH2 保留数据端的原始对齐，并在模型内部完成这一次移位，调用方无需预先再移位。

```math
\begin{aligned}
\widetilde\ell_t&=\begin{cases}\ell_{t+1},&t<T,\\-100,&t=T,\end{cases}\\
I(z)&=\{t:\widetilde\ell_t\ne-100\},\qquad n(z)=|I(z)|.
\end{aligned}
```

普通 `[B,T]` batch 已沿 batch 维隔离样本。该版本的长度分组用于减少 padding；若将多条样本串接为 sequence packing，才需另行规定段边界、段内因果可见性与 position。padding 的计算安全处理也与目标是否计分分开检查。

## 算法 1 编码与数据审计

下面是拟用于审计的参考算法；源码已有过滤和编码，并未实现下列全部断言。

`_tokenize_messages` 多次渲染模板，再根据各次渲染的长度切出监督区间。要使这些长度对应正确的 token 位置，需要检查：逐轮渲染结果，以及使用 `add_generation_prompt=True` 的渲染结果，是否都是完整渲染结果的 token 级前缀；还需检查 `prompt_len <= len(ids)`。字符位置不能直接代替 token 位置。

预编码路径用 `zip(real_token, mask)`，需要额外防止长度不等被静默截短。[编码与预处理](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L252)

**过程** EncodeAndAudit(z, split\_manifest, tokenizer, template)

**输入** 原始样本 z、冻结划分清单、tokenizer、模板，以及已声明的生成与过滤或截断规则。

**输出** 训练批次与 collator 外的身份映射；无有效监督目标时返回拒绝记录。

![算法 1 操作与数学公式](visualize:fde1_bGliZmlsZV9iNU0wbGtlUjdfTEZEYmFYTjhlWmlB_FileDrive_45288dfe06608191af94334ba8699511)

collator 不保留所有附加字段：它明确移除 `positions / data_id / raw_idx / cross_entropy / length`，轨迹身份应在入批前另存。输入、标签与 mask 的对齐要在 padding 后再次核对。[collator](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/train/data_collator.py#L96)

# 去重与留出按问题身份组织

[regenerate.py 的 `build`](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/data/regenerate.py#L197) 按完整 `real_text` 的 BLAKE2b 摘要去重，并支持排除已有数据。相同问题搭配不同教师回答可能产生不同摘要，因此仍需增加问题 ID、规范化问题和题族层面的重叠检查。

| 拟定字段 | 用途 |
| --- | --- |
| `sample_id / problem_id / family_id` | 区分一条轨迹、同一问题与同源题族 |
| `source / source_revision / split` | 固定来源及训练、调优、最终评测身份 |
| `teacher_revision / generation_config` | 追溯显式轨迹由谁、按什么设置生成 |
| `tokenizer_hash / template_hash` | 判断旧 token 与新模型是否仍兼容 |
| `input_ids / labels / length` | 进入训练的实际数据 |
| `truncated / grading_status` | 区分未完成回答、验证器错误与答案错误 |

划分先按问题或题族固定，再从各划分生成不同回答。调优集用于选择配置；最终评测集用于冻结之后的结论。组合留出与知识留出分别定义，不能仅通过随机拆行宣称实现。

## 验证集从发布行数到实际监督位置

原始 release 行数、预处理行数、整除后的行数、真正执行的样本出现次数，需要分别登记。这里保留有序样本及其重复次数；使用 pad 时，重复出现的原始 ID 不能被误计为新的独立题目。

| 边界 | 固定实现的操作 | 需要保存的计数与身份 |
| --- | --- | --- |
| 发布数据 → 可用序列 | raw 对话缺少 assistant、截断后无监督，或按长度 filter，均可能被过滤；预编码分支也检查 mask | 发布行数、处理后行数 N₀、保留／过滤 ID 与原因 |
| 序列 → 常规训练入口的 eval 数据 | `_apply_even_strategy` 按配置 `data.dp=q` 做 drop／pad；drop 删除末尾，pad 重复开头 | 有序出现记录、处理后行数 N、重复次数；不能只保存集合 |
| eval 数据 → DataLoader | 顺序分批，每批最多 b 条；最后一批可不满 | batch 总数 B 与每批出现记录 |
| DataLoader → 实际 eval | 只执行前 B* 个 batch，再按 batch 索引模实际 DP 数 P 分给各 rank | 执行／遗漏出现记录、移位后监督 token 数与实际分子 |

对常规训练入口的 drop 分支，配置 q>1 时有下式；这里 q 是数据配置，P 是由实际 world size 和 TP 数确定的 DP 数，固定 `main_tp` 没有把后者回写为 `data.dp`。

```math
\begin{aligned}
N&=q\left\lfloor\frac{N_0}{q}\right\rfloor,\qquad
B=\left\lceil\frac Nb\right\rceil,\\
B_*&=P\left\lfloor\frac BP\right\rfloor,\qquad
N_{\mathrm{executed}}=\min(N,bB_*).
\end{aligned}
```

因此“行数能被 q 整除”不足以保证“batch 数能被 P 整除”。若预处理后恰为 955 行、q=8、drop，则先保留 952 行；b=1、P=8 时全部执行，b=2、P=8 时只执行 944 行。这是条件算例，未把 README 的 955 条直接当作已实测的预处理结果。

`eval_only=True` 在此函数中提前返回，跳过下面的整除处理；常规训练分支的 train drop 则延后到 sampler 的每 epoch shuffle 之后。两者都不能套用 eval 的“先截前缀”流程。[预处理与过滤](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L334-L428) · [整除策略与提前返回](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L466-L698) · [实际 DP 拓扑与数据入口](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/train/SFT_TaH.py#L50-L149)

**pad 的适用范围。** `_apply_even_strategy` 只追加一次 range(pad_count)，没有循环取模。设原非空行数为 N₀，缺口为 q−(N₀ mod q)；所有追加索引都小于 N₀ 时，这条路径才能补齐。N₀=1、q=8 时会产生 1…6 等越界索引，并非重复第0条七次。八组检查的第8组以检查索引的 dataset 替身执行原函数，复现了这一边界；真实数据框架和分布式执行仍需验收。[源码整除函数](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/tah2/utils/data_prepare.py#L466-L504)。

这条计数链继续接到[并行与评测页的验证损失推导及算法 T12](https://chatgpt.com/space/page_d887d8f1a41881918ecff055e7613c9c)：损失的分子与分母必须对应同一批实际监督位置。

## 工具轨迹怎样展开成训练记录

固定 regenerate.py 的 tool_rows 以 user 消息划分 episode。它先丢弃最早异常 episode 及后续历史，再为每个有 assistant 输出的完整 episode 生成一条记录：输入是截至该段结束的完整前缀，监督 spans 只取本段 assistant；早期回答、用户和工具观测继续留在输入。由此可区分“重复处理历史上下文”和“重复监督历史回答”。[tool_rows／tokenize_row／build](https://github.com/thu-nics/TaH/blob/78ddf9a2c25713fced03a736c5746161f6ee590c/script/tah2/data/regenerate.py#L28-L137)

下面估算 episode 展开后的输入量和直接监督量。记固定系统／工具头占 C 个 token，第 i 个 episode 新增 L_i 个 token，其中有效 assistant 目标有 a_i 个，共有 k 个 episode。C 与 L_i 都使用同一 tokenizer 和模板定义。以下公式要求模板保持历史前缀一致，并且不截断、不过滤、不去重：

```math
N_{\rm input}=kC+\sum_{i=1}^{k}(k-i+1)L_i,\qquad
N_{\rm supervised}=\sum_{i=1}^{k}a_i.
```

第i段的文本因后续前缀而重复进入输入k−i+1次，但只在本段记录中计直接CE。被mask的历史仍可经K/V影响当前目标，梯度并不因此被detach；损失mask与注意力／梯度可见性分别核验。真实模板若不是前缀一致的，应直接从每条编码记录统计，而非套用此简式。

`tokenize_row` 用 `offset_mapping` 将字符跨度转换成 token mask。若某个 token 与监督 span 相交，却未完整落在 span 内，整条候选返回 `None`。长度处理区分两条路径：工具前缀超过 16384 token 时整条过滤；普通非工具 build 调用同一函数，但允许 cutoff。没有有效 mask 的候选也会被过滤，最后再按 `real_text` 摘要检查 seen 和 exclude。这些步骤都会改变保留率与任务比例。摘要标识的是完整渲染文本，问题或题族的留出仍需另外检查。

### 算法 D3 工具 episode 的展开与监督

**输入** messages、对应assistant生成元信息、tools、固定模板、tokenizer和排除集。

**输出** real_text／real_token／mask／data_id，以及输入、监督、过滤和去重计数。

![算法 D3 工具 episode 的展开与监督](visualize:fde1_bGliZmlsZV9VRVQ3VTl2bmtBbDFSNHdaRm9XcHZ3_FileDrive_45288dfe06608191af94334ba8699511)

**性质** 按固定函数重写；D7–D8用原函数体和字符tokenizer替身验证跨度、长度、两段前缀展开及只监督当前段。未运行真实BPE、工具环境或模型梯度。

## 本轮数据核对与可重复检查

本轮另把原生预训练的真实数据配方接回本课：Huginn的原文96来源、当前113项配置、文件倍率和尾部处理；Done Right的42来源、token预算结转与完整数据状态恢复；Ouro的阶段来源预算与监督回放。它们是不同的混合机制，不能都写成一个未经解释的“采样权重”。完整清单、14份固定源码、三个算法源文件和D1–D8检查结果在本轮检查包中。

[循环研究数据配方与执行核对：原始清单、固定源码与D1–D8检查](library-file:fde1_bGliZmlsZV9wSzZnaUo4aDF2RXpiamo2Y2xwT3lB_FileDrive_45288dfe06608191af94334ba8699511)

检查包保留Huginn原文96项完整清单、Fig.4比例、113项YAML及全部差异，Done Right的42项原始权重顺序，14份固定源码、许可证、三个算法源文件及D1–D8结果。直接运行 python3 check_data_pipeline.py；依赖NumPy与PyYAML。检查采用替身数据／tokenizer／packer，范围与未运行部分记录在README及JSON中。

# 数据配比与目标权重

## 三种平均目标

令先抽来源 `d~α`，再从保留后的来源分布 `P_d` 抽样。记一条样本的监督损失和为 `Sθ(z)`、监督数为 `n(z)>0`。必须先说明希望平均样本、合并后的 token，还是各来源的 token 风险：

```math
\begin{aligned}
J_{\rm sample}(\theta)&=\sum_d\alpha_d\,\mathbb E_d[S_\theta(z)/n(z)],\\
J_{\rm token}(\theta)&=\frac{\sum_d\alpha_d\,\mathbb E_dS_\theta(z)}
{\sum_d\alpha_d\,\mathbb E_dn(z)},\\
J_{\rm source}(\theta)&=\sum_d\alpha_d\,
\frac{\mathbb E_dS_\theta(z)}{\mathbb E_dn(z)}.
\end{aligned}
```

## 从 token 权重反解采样概率

合并 token 平均时，来源的有效权重由平均监督长度改变。若希望指定 token 权重 `q_d`，且各来源分布固定、平均监督数有限且为正，可反解样本采样权重：

```math
\begin{aligned}
q_d&=\frac{\alpha_d\,\mathbb E_dn}{\sum_j\alpha_j\,\mathbb E_jn},\\
\alpha_d&=\frac{q_d/\mathbb E_dn}{\sum_jq_j/\mathbb E_jn}.
\end{aligned}
```

例如两个来源等概率抽样，监督长度固定为 1 与 9，每 token 损失为 1 与 3：样本平均是 2，合并 token 平均是 2.8，长来源占 90% 的 token 权重。

有限批次的随机和之比一般不是总体期望之比的无偏估计；不要与比值的期望混写。

这些分布均指过滤、截断之后。若先混合再全局拒绝无效样本，保留的来源比例还会乘以各来源保留率；若先选来源再在来源内部重采样，来源概率可保持不变。清单同时记录样本数、输入 token、监督 token、保留率与实际计算成本，明确预算单位。

## 配方案例与延伸阅读

第一版只接入可复现的固定配比，再把在线调度作为替换采样器的研究扩展。对照 [Marin Quick RL loop](https://github.com/marin-community/marin/blob/f5e8a9cfe9d9d50b79fd03e4ed0ed0087edf627d/docs/recipes/quick-rl-loop.md) 学习“固定选择产物—固定 SFT 初始化—前后评测”的组织方式。其默认 40 H100 和 25 次 RLOO-N 更新是案例规模，不是两张 A100 的启动配置。

# 交付与验收

使用同一份小型开发样本面板，保留短回答、长回答、多轮或工具记录、截断边界。它用于排查工程语义，不能代替最终能力评测。下面的文件名是待实现产物约定。

- [ ] `manifest.json`：全部版本、划分与样本身份可追溯。

- [ ] `batch_trace.json`：每个位置的 token、角色、label、attention mask、移位后目标和有效分母。

- [ ] `data_audit.json`：各来源长度、监督量、过滤率和问题重叠报告。

- [ ] 同一清单重建后，token、labels 和样本顺序相同；若不同，能定位到模板、数据或随机状态的变化。

验收以这些实际记录为据。当前材料已建立入口和数据约定，尚未运行数据构建。

# 围绕本课讨论

```codex-prompt
请沿固定版本 regenerate.py 和 data_prepare.py，追踪一条原始样本到训练 labels 的全过程。先解释模板和监督边界，再核对去重究竟按什么身份进行。最后用同一小样本面板设计 manifest、batch_trace 和 data_audit；把源码已有行为与拟补功能分开。
```

[下一步 一次参数更新](https://chatgpt.com/space/page_6a7d786183d0819192c20a043a45d647)
