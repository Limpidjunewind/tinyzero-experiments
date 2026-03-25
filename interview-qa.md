# Interview Q&A

> 按天积累，面试前通读一遍。
> 格式：问题 → 标准答案 → 如有漏洞标注 ⚠️

---

## Day 2（2026-03-19）：任务设计 + 奖励函数 + 训练曲线

**Q: 你们训练的是什么任务？为什么选这个任务？**
> A: Countdown 数字游戏——给定目标数和 6 个数字，用四则运算凑出目标。
> 选这个任务是因为答案有客观标准（可验证），不需要训 Reward Model，
> 同时复杂度足够高，模型需要多步推理，是复现 DeepSeek-R1 思想的最小单元。

**Q: 奖励函数怎么设计的？有没有 reward hacking？**
> A: Rule-Based Reward，三档打分（0 / 0.1 / 1.0）。
> 0 = 没有 `<answer>` 标签；0.1 = 格式对但答案错；1.0 = 完全正确。
> 格式分 0.1 用来引导模型学会结构化输出，10:1 的比例确保模型不会停在"只套格式"的局部最优。
> Rule-Based 天然没有 reward hacking——模型无法欺骗一个固定规则。

**Q: 为什么 format_score 设成 0.1 而不是 0 或 0.5？**
> A: 设成 0 则模型没有信号学习输出格式，思维链涌现会更难更慢。
> 设成 0.5 则格式和正确答案权重相当，模型会在"套格式"这个局部最优里停太久，不再努力解题。
> 0.1 让正确答案的信号比格式信号强 10 倍，格式是引导，不是目标。

**Q: prompt template 为什么要区分 base 和 instruct？**
> A: Instruct 模型在 SFT 阶段被训练成识别特定 chat template 格式标记（如 `<|im_start|>`）。
> 如果用不匹配的 prompt 格式，模型输入分布与训练分布不一致，输出会混乱。
> Base 模型没有这个约束，直接续写纯文本即可。

**Q: 为什么用 PPO 而不是 SFT？**
> A: SFT 需要大量人工标注的正确示范。Countdown 的解题路径太多，人工标注成本极高。
> PPO 让模型自己探索，只需要"对不对"这一个验证信号，不需要知道"怎么对"。
> 这也是 DeepSeek-R1 的核心思路：用可验证的奖励替代人类示范。

**Q: entropy 曲线说明了什么？**
> A: 训练分两个阶段。第一阶段熵急剧下降，模型收敛到"输出格式化标签"这一固定策略；
> 第二阶段熵回升，模型在掌握格式之后开始探索多样的推理路径。
> 如果熵一直降到极低不回升，是策略崩塌的信号，模型只会一招。

**Q: KL 散度为什么会出现负值？**
> A: PPO 用的是 log ratio 估计值（`log π_current - log π_ref`），不是严格数学意义上的 KL。
> 负值表示当前模型对某些 token 比初始模型更不确定（概率降低了）。
> 真实 KL ≥ 0，但这个 per-sample 估计量可以为负，整体均值接近 0 说明模型没有跑偏。

**Q: SFT 和 RL 训练的本质区别是什么？**
> A: SFT 是 teacher forcing，每个 token 都有正确标签，用 cross-entropy loss 监督，模型学的是模仿训练数据的分布。
> RL 只有最终结果有 reward，中间过程靠模型自己探索，学的是"什么行为导致好结果"。
> 关键区别：SFT 学"输出什么样子"，RL 学"做什么能赢"。

**Q: 为什么 `<think>` 推理格式不能只靠 SFT 训练？**
> A: SFT 训练 think 需要有标注好的推理过程数据，而这些数据获取成本极高（人工标注贵、GPT生成质量有限）。
> 更深的问题：SFT 训出来的 think 是表面格式模仿，think 内容和 answer 对错没有因果连接。
> RL 里模型自己生成 think，正确的 think 带来更高 reward，因果链真实存在，推理能力才是真的涌现出来的。

**Q: 为什么要先 SFT 再 RL？直接从 base model 上 RL 不行吗？**
> A: RL 需要一个靠谱的初始策略，否则模型随机探索的空间太大，reward 信号极稀疏，训练无法收敛。
> SFT 给模型一个起点：学会基本的 chat 格式和指令跟随能力，RL 再在此基础上优化输出质量。
> 实证：DeepSeek R1-Zero 跳过 SFT 直接 RL，虽然推理能力涌现了，但输出混乱（中英混杂、格式不稳定）。正式版 R1 加了 cold-start SFT 阶段才稳定。

**Q: Parquet 是什么格式？TinyZero 为什么用它？**
> A: Parquet 是列式存储格式，同一列的数据存在一起，相比 CSV（行式）读取指定列更快、压缩率更高。
> TinyZero 训练时只需要读 question 和 answer 两列，列式读取可以跳过其他列。
> answer 列就是 ground_truth 的来源，通过 DataProto 传入 RewardManager，最终传给 compute_score。

**Q: instruct 模型为什么必须保持 chat template 格式（如 `<|im_start|>`）？破坏了会怎样？**
> A: Instruct 模型 SFT 时训练数据全是特定 chat template 格式，模型权重里编码了"看到 `<|im_start|>assistant` → 应该生成回答"这个条件概率。
> 破坏格式 = 推理时输入分布偏离训练分布（distribution shift），模型缺少触发"回答模式"的 token 信号。
> 结果不是"答不出来"，而是输出质量大幅下降：乱说、重复、风格不对。

---

**Q: ground_truth 是怎么从数据文件传到打分函数的？**
> A: 四个站点：
> 1. `data_preprocess/countdown.py` 把 `{"target": 24, "numbers": [3,4,6,8]}` 存进 parquet
> 2. DataProto 加载时，这个 dict 进入 `non_tensor_batch`（因为不是 tensor，无法放 TensorDict）
> 3. `RewardManager.__call__` 遍历每个样本，用 `data_item.non_tensor_batch['reward_model']['ground_truth']` 取出来
> 4. 传给 `compute_score(solution_str, ground_truth)`，验证数字合法性 + 计算正确性，返回 0 / 0.1 / 1.0

---

## Day 3（2026-03-21）：GAE + clip loss

**Q: GAE 的 λ 参数控制什么？λ=0 和 λ=1 分别退化成什么？**
> A: λ 控制 bias-variance tradeoff。λ 越大，用到的步数越多，方差越大但偏差越小。
> λ=0 退化成 **TD(0)**：只用单步 TD 误差 `r_t + γV(s_{t+1}) - V(s_t)`，偏差大但方差小。
> λ=1 退化成 **Monte Carlo**：用整条轨迹的真实回报，无偏但方差大。

**Q: 为什么 PPO 用 GAE 而不是直接用 TD error δ(t)？**
> A: δ(t) 只看一步，太短视。比如 t=0 的 token 没有即时 reward，δ(0) 可能是负的，但它导向了最终的正确答案。
> 用 δ(0) 更新会错误惩罚这个 token。
> GAE 把多步 TD error 加权求和，功劳往前传，让 t=0 的 token 也能感知到最终 reward 的贡献。

**Q: PPO clip loss 为什么取 max 而不是直接用 clip 后的 loss？**
> A: advantages 有正有负，clip 在不同方向起作用。取 max 能自动处理两种情况——
> 不管 ratio 往哪个方向越界，max 都会选更保守（更大）的 loss，梯度被压制，不需要额外判断 advantages 的符号。

**Q: GRPO 和 PPO 的 advantage 计算有什么区别？**
> A: PPO 用 Critic 提供 V(t)，通过 GAE 计算每步的 advantage，需要额外训练一个 Critic 模型。
> GRPO 对同一个 prompt 采样多条回答，用组内均值当 baseline，(score - mean) / std 得到 advantage，不需要 Critic。
> 好处：省掉 Critic，训练更简单；代价：需要每个 prompt 多次采样，batch 构造成本更高。

---

## Day 4（2026-03-21）：PPO 训练主循环

**Q: PPO 训练的主循环有哪几步？**
> A: 9 步：generate sequences → repeat align → compute valid tokens → compute values → compute scores → compute rewards → compute advantages → update critic → update actor。
> GRPO 模式下跳过 compute values 和 update critic。

**Q: score 和 reward 有什么区别？为什么不直接用 score？**
> A: score 是纯任务分（0/0.1/1.0），reward = score - kl_coef * KL(t)。
> 只用 score 模型只管答对，不管有没有跑偏，可能 reward hacking 或语言能力退化。
> KL 惩罚把"不偏离参考模型"也变成优化目标，两个信号一起约束模型。

**Q: KL 惩罚的公式是什么？物理意义是什么？**
> A: KL(t) = log π_current(token_t) - log π_ref(token_t) = log(新模型概率/参考模型概率)
> KL > 0 说明新模型比参考模型更偏爱这个 token（跑偏了），扣分。
> reward = score - kl_coef * KL，kl_coef 由 AdaptiveKLController 自动调节。

**Q: critic warmup 是什么？为什么需要？**
> A: 训练开始时先只更新 Critic，等 Critic 稳定后再开始更新 Actor。
> Critic 刚初始化时 V(t) 估计很差，advantage 质量低，这时更新 Actor 反而有害。

**Q: repeat align 这一步做了什么？为什么需要？**
> A: GRPO 每个 prompt 采样 n 条回答，generate_sequences 生成 n 条 response，但原始 batch 每个 prompt 只有 1 条。
> batch.repeat(n) 把每个 prompt 复制 n 份，才能和 n 条 response 一一对应后合并。

**Q: 为什么 reward 只放在 response 的最后一个 token，而不是每个 token 都有？**
> A: 这是 sparse reward 设计——整个 rollout 只有在序列结束时才知道答案对不对，中间每步没有即时反馈。
> 类比下棋：只有终局才知道赢没赢，中间每步棋没有分数。
> 正因为 reward 稀疏，GAE 才有存在的意义：把终点信号通过 λ 加权往前传播，让每个 token 都能感知到最终结果的贡献。

**Q: PPO 为什么是 on-policy 的？训练时不能复用历史数据吗？**
> A: PPO 每次 generate 完就用这批数据更新，更新完模型变了，旧数据的分布就和新模型不匹配了，不能再用。
> 这是 on-policy 的核心约束：只用当前模型自己生成的数据更新自己。
> PPO 的妥协：同一批数据可以跑几个 mini-batch epoch，但靠 clip 限制每次更新幅度，保证新旧策略不偏离太远。

**Q: RewardManager 的两条路径分别是什么？TinyZero 用的哪条？**
> A: 两条路径：
> 1. model-based RM：batch 里已有 `rm_scores`（由神经网络打分），直接返回。适合开放式任务，答案没有唯一标准。
> 2. rule-based RM：按 `data_source` 路由到对应规则函数（如 countdown.compute_score），代码直接验证答案。适合有确定答案的任务。
> TinyZero 用 rule-based，countdown 答案可验证，不需要训 Reward Model。

**Q: 训练用了多卡，怎么协调 Actor、Ref、Critic 三个模型？**
> A: 用 Ray 做分布式调度。三个模型分别部署为 Ray Worker，各自持有模型权重跑在不同 GPU 上。
> 主进程通过 RPC 向 Worker 发命令（"你去 generate"、"你去算 log prob"），Worker 执行完把结果返回。
> Actor 和 Ref 共用同一个 Worker 类（ActorRolloutRefWorker），靠 role 参数区分行为：Actor 有 optimizer 会更新梯度，Ref 冻结只做推理。

---

## Day 5（2026-03-21）：分布式训练 + fsdp_workers.py

**Q: DDP 和 FSDP 有什么区别？各自适合什么场景？**
> A: DDP（Data Parallel）：每卡存完整模型，处理不同 batch，梯度 allreduce 后同步。显存需求 = 完整模型大小，模型放得下时用。
> FSDP（Fully Sharded Data Parallel）：参数、梯度、optimizer state 全部切碎（shard），每卡只存 1/N，需要时临时 allgather 拼回来，用完丢掉。显存需求降为 1/N，大模型必须用。

**Q: FSDP 的 CPU offload 是什么？为什么不默认开？**
> A: 把暂时不用的参数/梯度/optimizer state 挪到 CPU RAM，GPU 显存实在不够时的兜底方案。
> 不默认开的原因：CPU ↔ GPU 传输带宽远低于 GPU 内存带宽，每次 offload/load 都要等数据搬运，训练速度大幅下降。

**Q: 训练用 FSDP，推理用 vLLM，两者怎么共存？**
> A: FSDP 把参数切碎分散在多卡，vLLM 需要完整参数加载。矛盾由 `rollout_sharding_manager` 解决：
> 进入推理前，allgather 把 FSDP 碎片拼成完整权重交给 vLLM；推理结束，重新切碎回 FSDP 状态准备下一轮训练。

**Q: vLLM 为什么比普通推理快？PagedAttention 是什么？**
> A: 普通推理 KV Cache 需要预分配连续显存，容易产生碎片，利用率低。
> PagedAttention 借鉴操作系统内存分页思想，把 KV Cache 切成小页动态分配，避免显存碎片，同一时刻可以并行处理更多请求，吞吐量大幅提升。

**Q: generate_sequences 里 old_log_probs 是什么时候算的？为什么不等 Actor 更新后再算？**
> A: 在 vLLM 生成完 response 之后、Actor 更新之前立刻算，用的是**当前 Actor**（还没更新）的参数。
> 这就是 fit() 第 1 步的输出 old_log_probs，后面用来算 KL（和 Ref 比）和 clip ratio（和新 Actor 比）。
> 如果等 Actor 更新后再算，old 和 new 就是同一个模型，clip ratio 恒为 1，PPO 的约束机制完全失效。

---

## Day 6（2026-03-26）：对比实验 + 训练诊断

**Q: 你跑了几组实验？结果是什么？**
> A: 三组：Qwen2-0.5B（PPO）、Qwen2.5-1.5B（PPO + GRPO）、Qwen2.5-3B（PPO）。
> 0.5B：score 卡在 0.10，从未出现 `<think>`，RL 完全无效。
> 1.5B PPO：score 0.3–0.5，出现空 `<think>`，找到了格式捷径但没有真实推理。
> 1.5B GRPO：score 卡在 0.10，150 步未收敛，没有 Critic 信号更难起步。
> 3B PPO：score 0.3–0.4，`<think>` 有真实内容，step ~250 entropy 坍塌。
> 结论：三个规模跨越三个门槛——无 think → 空 think → 有内容的 think。

**Q: pg_loss 归零和 entropy collapse 哪个先发生？因果关系是什么？**
> A: pg_loss 先归零，entropy 随后坍塌，之前搞反了。
> 因果链：模型在 step ~100 找到"空 think + format reward"捷径 → pg_loss 归零（策略停止更新）→ 输出固定后 entropy 随之坍塌。
> entropy collapse 是策略停止更新的结果，不是原因。

**Q: entropy 和 pg_loss 归零后 score 还在涨，说明模型还在学习吗？**
> A: 不是。entropy ≈ 0、pg_loss ≈ 0 后 score 继续涨是方差降低的假象：
> rollout 方差下降，原来就会的题开始稳定答对，score 自然上升。
> 但策略没有改进，模型没有在学新东西。entropy 和 pg_loss 是比 score 更诚实的诊断信号。

**Q: 3B 模型比 1.5B 更快 entropy 坍塌，怎么解释？**
> A: 反直觉但合理：能力越强 → 越快找到稳定策略（如空 think 捷径）→ PPO 越快强化这个策略 → entropy 越快锁死。
> 能力强在这里是更快失去探索性的代价，不是优势。

**Q: 你的实验对 cold start 有什么结论？**
> A: outcome reward 的有效性依赖模型的初始能力。模型需要在随机探索阶段就能偶尔答对，reward 信号才能起作用。
> 0.5B 做不到这一点，RL 完全无法激励推理链。
> cold start（先 SFT 再 RL）是必要条件，不是可选优化——它给模型一个探索起点，让 reward 信号有机会生效。

**Q: 实验中怎么判断 Critic 是否过拟合？**
> A: 观察 values 和 returns 的关系。如果 Critic 过拟合，values 会收敛到接近 returns，advantage ≈ 0，Actor 梯度消失。
> 实测 values ≈ 0、returns ≈ 0.3，advantage 始终有值，说明 Critic 正常工作，没有过拟合。
> `score ≈ rewards` 只说明 KL 惩罚极小，不代表 advantage ≈ 0，两者不能混淆。

