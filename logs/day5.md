# Day 5 — fsdp_workers.py：分布式训练与推理

## 多卡并行：DDP vs FSDP

| | DDP | FSDP |
|---|---|---|
| 参数存储 | 一卡| 每张卡一片|
| 显存占用 |大| 小|
| 适用场景 |小模型/大卡/单卡| 大模型/小卡/多卡|

FSDP 全称：fully-sharded-data-parallel

## CPU Offload

作用：显存装不下的时候就放在这里

触发条件（为什么不默认开）：gpu-cpu传输速度慢

## vLLM

是什么：推理的框架

核心技巧 PagedAttention：是：KV Cache 像 OS 管内存一样按页动态分配，避免显存碎片，提高利用率 → 更多请求可以并行 → 吞吐量高。

## rollout_sharding_manager

解决的矛盾：推理的时候需要完整的参数

进入推理前做了什么：把参数拼起来

推理完后做了什么：把参数拆开存放

---

## ActorRolloutRefWorker 结构

### __init__ 做了哪三件事

1. 几张卡怎么分
2. 设置角色标志
3. 设置offload标志

### generate_sequences 骨架（5步）

1. 如果有offload，从cpu中把参数搬去gpu
2. 推理生成responses
3. 如果是actor 还需要算出log- probability
4. 如果有offload，数据搬回到cpu
5. 清空kv cache，清空显存

### update_actor 骨架（4步）

1. 如果有offload，从cpu中把参数搬去gpu
2. 更新参数
3. lr_scheduler.step() — 学习率调度器更新学习率（比如每步线性衰减）。
4. 如果有offload，搬回cpu

---

## 遗留问题

- 无 / （填问题）
