# 全自动 Loop 简介

本文档简要说明本项目里“全自动 Loop”是怎么做到的，方便后续 AI 或人工继续使用。

## 1. 它解决什么问题

普通聊天式 AI 有两个限制：

- 一轮对话做完后不会自动继续工作。
- 长时间工作后可能丢失上下文，或者忘记前面实验结果。

所以我们做了一个外部 Loop：不是让服务器一直跑一个大脚本，而是在本地反复启动新的 Codex 会话。每一轮 Codex 都重新读取项目里的持久化文档、实验日志和任务队列，再决定下一步做什么。

核心思想：

```text
外部脚本负责“反复启动 AI”
AI 每一轮负责“读上下文 - 提假设 - 做实验 - 写日志 - 给继续/停止信号”
```

## 2. 相关文件

Loop 入口脚本：

```text
scripts/codex_universal_tg_agent_loop.py
scripts/codex_universal_tg_agent_loop.ps1
scripts/codex_universal_tg_agent_loop.sh
```

远程命令辅助脚本：

```text
scripts/remote_tg_command.py
```

任务与记忆文件：

```text
AGENTS.md
docs/research/universal-tg-task-queue.md
docs/research/universal-tg-iteration-log.md
results/universal_single_regressor/scoreboard.json
```

实验结果目录：

```text
results/universal_single_regressor/
logs/
```

## 3. 每一轮 Loop 做什么

每一轮大致流程：

1. 本地 Loop 脚本启动一个新的 Codex exec 会话。
2. 新会话读取 `AGENTS.md`，获得工作协议、指标、服务器规则和停止条件。
3. 读取任务队列：

```text
docs/research/universal-tg-task-queue.md
```

4. 读取近期实验日志：

```text
docs/research/universal-tg-iteration-log.md
```

5. 读取当前积分榜：

```text
results/universal_single_regressor/scoreboard.json
```

6. 选择一个明确假设或一个小任务。
7. 修改代码或配置，或通过远程服务器运行一个受控实验。
8. 把命令、数据、指标、结论写回实验日志。
9. 更新结果文件或 scoreboard。
10. 最后输出一个信号：

```text
TG_CONTINUE
TG_BLOCKED
TG_CONVERGED
```

外部 Loop 脚本根据这个信号决定是否继续下一轮。

## 4. 为什么它不会完全依赖聊天记忆

关键点是：每轮都重新从磁盘读上下文。

它依赖的不是上一轮聊天记忆，而是这些持久化文件：

- `AGENTS.md`：规则和目标。
- `universal-tg-task-queue.md`：下一步任务池。
- `universal-tg-iteration-log.md`：历史实验记录。
- `scoreboard.json`：当前最好结果。
- `results/`：每个实验的输出。

所以即使重新开一个 AI，只要这些文件还在，它也能恢复到相近的工作状态。

## 5. 本地控制、服务器实验

设计上是：

```text
本地 Windows：负责启动 Codex、思考、写日志、改代码
远程服务器：负责跑耗时训练和评估实验
```

服务器信息：

```text
SSH: sheng-xiang@100.64.0.4
远程目录: ~/Tgprediction
远程 Python: /home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python
```

重要规则：

- 远程命令必须先 `cd ~/Tgprediction`。
- 不要操作 `~/Tgprediction` 之外的路径。
- 不要把密码写进仓库文件。

## 6. 为什么不用交互式 SSH

AI 通常不会稳定处理交互式密码输入，所以项目里准备了 Paramiko 远程命令脚本：

```text
scripts/remote_tg_command.py
```

PowerShell 里先设置临时环境变量：

```powershell
$env:TG_REMOTE_PASSWORD = "密码只放当前会话，不要提交"
```

测试远程连接：

```powershell
python scripts\remote_tg_command.py --use-paramiko "cd ~/Tgprediction && pwd"
```

测试远程 Python：

```powershell
python scripts\remote_tg_command.py --use-paramiko "cd ~/Tgprediction && /home/sheng-xiang/miniconda3/envs/llm4graphgen/bin/python -V"
```

以后所有远程命令都可以写成：

```powershell
python scripts\remote_tg_command.py --use-paramiko "cd ~/Tgprediction && <你的命令>"
```

## 7. 怎么启动 Loop

推荐在本地 PowerShell 里启动：

```powershell
cd "C:\Users\24020\Desktop\Tg预测项目"

$env:TG_REMOTE_PASSWORD = "你的服务器密码"

powershell -ExecutionPolicy Bypass -File scripts\codex_universal_tg_agent_loop.ps1 `
  -MaxHours 5 `
  -MaxRounds 20 `
  -RoundTimeoutMinutes 45
```

参数含义：

- `MaxHours 5`：最多运行 5 小时。
- `MaxRounds 20`：最多启动 20 轮 AI。
- `RoundTimeoutMinutes 45`：每一轮最多 45 分钟。

## 8. 怎么看运行情况

查看 Loop 日志：

```powershell
Get-Content logs\codex_universal_tg_agent_loop.nohup.log -Tail 120
```

查看实验日志：

```powershell
Get-Content docs\research\universal-tg-iteration-log.md -Tail 120
```

查看任务队列：

```powershell
Get-Content docs\research\universal-tg-task-queue.md
```

查看当前最好结果：

```powershell
Get-Content results\universal_single_regressor\scoreboard.json
```

查看远程是否有训练进程：

```powershell
python scripts\remote_tg_command.py --use-paramiko "cd ~/Tgprediction && pgrep -af train_universal_tg_single_regressor.py"
```

查看 GPU：

```powershell
python scripts\remote_tg_command.py --use-paramiko "nvidia-smi"
```

## 9. 如何停止

如果是在 PowerShell 前台跑，直接 `Ctrl+C`。

如果远程有后台训练，需要先查进程：

```powershell
python scripts\remote_tg_command.py --use-paramiko "cd ~/Tgprediction && pgrep -af train_universal_tg_single_regressor.py"
```

然后按 PID 停止：

```powershell
python scripts\remote_tg_command.py --use-paramiko "kill <PID>"
```

不要乱用 `kill -9`，除非普通 `kill` 无效。

## 10. 它什么时候会自己停

每轮结束时 AI 会输出一个信号。

```text
TG_CONTINUE
TG_BLOCKED
TG_CONVERGED
```

含义：

- `TG_CONTINUE`：还有独立实验值得继续。
- `TG_BLOCKED`：遇到缺数据、环境坏掉、连续无提升等阻塞。
- `TG_CONVERGED`：达到目标，例如所有任务 R² 都达到预设阈值。

外部 Loop 看到 `TG_CONTINUE` 会继续下一轮；看到 `TG_BLOCKED` 或 `TG_CONVERGED` 应停止。

## 11. 这个 Loop 的优点和限制

优点：

- 每轮重新读文件，不完全依赖聊天记忆。
- 能把长任务拆成多轮受控实验。
- 每轮都要求写日志，方便复盘。
- 本地负责推理，服务器只负责跑实验。

限制：

- 它不能保证一定做出好结果，只能保证按协议持续迭代。
- 如果任务队列或实验日志写得差，后续轮次也会被误导。
- 如果远程密码没设、服务器断线、GPU 被占用，Loop 会卡住或阻塞。
- 如果 Codex CLI 不可用，Loop 不能启动新 AI 会话。

## 12. 给接手 AI 的一句话说明

```text
这是一个外部控制的多轮 AI 迭代系统。不要指望单次聊天记住所有上下文；每轮必须重新读取 AGENTS.md、任务队列、实验日志和 scoreboard。远程实验通过 scripts/remote_tg_command.py 调用服务器，并且只能在 ~/Tgprediction 下工作。每轮只做一个明确假设或小任务，写入日志后用 TG_CONTINUE / TG_BLOCKED / TG_CONVERGED 结束。
```

