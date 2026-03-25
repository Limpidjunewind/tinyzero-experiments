# TinyZero Todo

> 目标：LLM 算法工程师面试（字节/阿里/百度），时间节点 2026年3-5月

---

## 当前状态（2026-03-21）

- Day 1 ✅ 完成：第一个 SLURM job 已提交，wandb 曲线正常，Aha Moment Step 22 出现
- Day 2 ✅ 完成：reward 数据流全链路搞清楚，weakness 清零
- Day 3 ✅ 完成：GAE 手算验证，clip loss，GRPO vs PPO
- Day 4 ✅ 完成：main_ppo.py RewardManager 路由，Ray 分布式，fit() 8步主循环
- Day 5 ✅ 完成：fsdp_workers.py，DDP/FSDP/vLLM/offload，ActorRolloutRefWorker 结构

---

## 本周计划

- [x] Day 3：`verl/trainer/ppo/core_algos.py` — GAE 搞清楚；clip loss
- [x] Day 4：`main_ppo.py` + `ray_trainer.py` fit() 全流程 — PPO dataflow 8步
- [x] Day 5：`fsdp_workers.py` — Actor/vLLM Worker 结构（了解即可）
- [ ] Day 6：对比实验 — PPO vs GRPO，跑起来对比曲线
- [ ] Day 7：包装面试材料，模拟追问 5 轮

---

> 面试问题见 interview-qa.md｜薄弱点见 weakness.md
