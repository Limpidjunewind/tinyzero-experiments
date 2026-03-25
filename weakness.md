# Weakness

> 测验暴露的薄弱点。修复后划掉，面试前通读确认全部清零。

---

- [x] reward 数据流：`ground_truth` 从 parquet → DataProto → compute_score 链路不清晰 ✅ 已掌握（2026-03-19）
- [ ] 分布式名词体系（FSDP/DDP/shard/allgather/allreduce/CPU offload）：逻辑链能串起来，但单独被问某个词可能反应慢，需多用几次形成条件反射
- [ ] `generate_sequences` 里 `old_log_probs` 时序：是"生成完立刻算"，不是"Actor 更新后算"。面试被追问 log_prob 从哪来时注意区分
