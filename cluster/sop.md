这份 SoC 集群开发 SOP 将帮助你建立起一套“工业级”的工作流，既能享受本地 VS Code 的丝滑，又能稳健地调度集群顶级的 A100/H100/H200 算力。

---

### 🚀 SoC 集群高效开发 SOP (Standard Operating Procedure)

我们将整套流程分为：**环境补给**、**补给线验证**、**交互调试**、**生产发射** 四个环节。

#### 第一阶段：环境补给 (Daily Start)

每天正式写代码前，确认这三项处于“就绪”状态。

1. **建立连接通路**：确保 SoC VPN 已连接（或处于校园网内）。
2. **激活本地编辑器**：打开本地 VS Code 文件夹 `TinyZero_Local`。
3. **确认自动同步**：
* 建议开启 VS Code 的 **Auto Save**（设置 -> `afterDelay`）。
* 检查 `sftp.json` 的 `"uploadOnSave": true` 是否开启。
* *口诀：本地是真理，保存即发货。*



#### 第二阶段：补给线验证 (Heartbeat Check)

通过原生终端（Mac Terminal）建立指挥部，不要使用 VS Code 内部的远程连接。

1. **登录前台**：
```bash
ssh jingwenl@xlogin1.comp.nus.edu.sg  # 或 xlogin0, xlogin2

```


2. **验证路径**：
```bash
cd /home/j/jingwenl/TinyZero
ls -l  # 确认看到的文件和本地 VS Code 里的一致

```



#### 第三阶段：交互调试循环 (Development Loop)

当你需要修改代码、观察输出、解决 Bug 时，使用交互式模式。

1. **本地修改**：在 VS Code 里改代码，`Cmd + S` 存盘。
2. **申请算力**：在终端通过 `srun` 申请一张临时显卡（比如 H100 的切分版）：
```bash
# 申请 1 小时算力进行调试
srun -p gpu -G h100-47 --time=01:00:00 --pty bash

```


3. **运行程序**：进入计算节点后（提示符改变），激活环境跑代码：
```bash
conda activate tinyzero
python main.py --debug

```


*若报错，回到 VS Code 修改再保存，终端直接重新运行。*

#### 第四阶段：生产发射 (Production Launch)

当代码已经调通，准备跑长达数小时甚至数天的训练时。

1. **编写任务书**：在本地 VS Code 新建 `train.slurm`（保存即同步）。
2. **一键发射**：在原生终端执行：
```bash
sbatch train.slurm

```


3. **断开连接**：此时你可以直接输入 `exit` 并**关闭电脑**，任务会在云端后台持续运行。

---

### 🛠 必备工具箱 (Command Cheat Sheet)

| 命令 | 作用 | 备注 |
| --- | --- | --- |
| `sinfo` | 查看全服算力剩余 | 看看 A100/H200 哪台空着 |
| `squeue -u jingwenl` | 监控我的任务状态 | 查看任务是在排队(PD)还是在跑(R) |
| `scancel <JOB_ID>` | 紧急停止任务 | 发现代码写错时及时止损，省点算力 |
| `sacct` | 查看历史任务 | 看看之前的任务为什么失败了 |

### ⚠️ 避坑红线 (Safety Rules)

* **严禁在前台（xlogin）跑程序**：登录节点只用来传文件、写 SLURM 脚本、看日志。任何超过 1 分钟的 Python 进程都可能导致你的 VS Code 再次卡死。
* **双向同步注意**：如果你在终端用 `vim` 改了云端文件，一定要回到 VS Code 在该文件上右键点 **`SFTP: Download`**，把改动拉回本地。
* **路径陷阱**：记住主目录是 `/home/j/jingwenl/`，中间那个 `/j/` 缺一不可。