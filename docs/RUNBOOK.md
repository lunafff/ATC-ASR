# Runbook / 实验追踪表

安装故障修复（2026-11-17）：用户Windows Python3.11自带pip在cu121单一索引安装时拒绝typing_extensions包名元数据，继而找不到flit_core。run.ps1已改为先从PyPI升级pip，以PyPI为主索引、cu121为额外索引，并精确指定torch==2.5.1+cu121；普通依赖从PyPI安装，随后pip check。此次失败在训练前，无需删除runtime、venv或pip缓存；拉取修复后重跑即可。

## Windows A6000 入口（用户已确认）

目标 Windows + A6000，可访问HF。安装Git、NVIDIA驱动、Python3.11后，用 `git clone -b codex/asr-round2 https://github.com/lunafff/ATC-ASR.git asr-round2` 克隆本轮独立仓库，在目录内执行 `powershell -ExecutionPolicy Bypass -File .\run.ps1`。后续更新用 `git pull --ff-only origin codex/asr-round2`，本机提交后使用 `git push`。入口验证Python3.10–3.12、建立venv、显式安装CUDA12.1版PyTorch，再运行完整流水线。分阶段用 `-Stage prepare/train/final/package`。Python路径用 `-Python 'C:\路径\python.exe'` 指定。查看日志：`Get-Content .\runtime\logs\launcher.log -Tail 30 -Wait`。Windows不要执行下文Linux的nohup命令。

下面保留跨平台流程表；其中bash命令为Linux备选，Windows使用上述PowerShell等效入口。

2026-11-17 状态：代码已实现，真实 CUDA smoke 和正式实验均未运行。不要将代码完成等同于实验完成。

| ID | 阶段/脚本 | 输出 | 当前状态 |
|---|---|---|---|
| P00 | run.sh 安装隔离环境；pipeline.py 记录环境/代码 | runtime/provenance | 待GPU机器执行 |
| P01 | prepare.py 下载固定revision模型/数据、导出音频、去重划分 | models / dataset / audio / manifests | 待执行 |
| S00 | engine.py smoke：2样本 CE+KD+adv，2条 dev 生成 | runs/smoke | 待执行；失败阻断后续 |
| R01 | CE，seed42，5 epochs | runs/supervised | 待执行 |
| R02 | CE+KD，T2，alpha .5 | runs/kd | 待执行 |
| R03 | CE+KD+adv .01 | runs/kd_adv_0.01 | 待执行 |
| R04 | CE+KD+adv .1 | runs/kd_adv_0.1 | 待执行 |
| E00 | teacher、初始student、R01–04统一测试 | evaluation/*.jsonl，summary.json/csv | 待全部组完成 |
| Z00 | pipeline.py package | results_return.tar.gz | 待执行 |

现场步骤：确认驱动 `nvidia-smi`、Python `python3.11 --version`、80GB可用磁盘；拉取固定 commit，执行 `bash run.sh`。需要后台运行时可用 `nohup bash run.sh > launcher.console.log 2>&1 &`，之后用 `tail -f runtime/logs/launcher.log` 查看。

目录职责：`models/` 下载原始权重；`dataset/` 原始 parquet；`audio/` 导出的音频；`manifests/` 冻结样本清单和摘要；`runs/<组>/best/` dev最优权重及processor；`latest.pt` 完整 epoch 断点，含学生/D/优化器/RNG；`train.jsonl` 损失；`evaluation/` 最终预测与指标；`provenance/` 配置、源码摘要、版本；`logs/` 启动输出；`failure.json` 异常。

恢复：同一代码/配置重跑 `bash run.sh`。中断 epoch 从头重算，该 epoch 之前已写入的损失日志会保留，分析时按 epoch/sample 去重；以 dev_epoch 文件及 complete.json 判定真正完成。下载使用HF缓存断点续传。不要在训练期间打包/修改配置。

故障：先读 failure.json 和 launcher.log；OOM 不会暗中改变 batch/数据/精度，需记录后归档再改配置。下载失败检查网络/镜像/HF_TOKEN；驱动问题在GPU机处理。超长 target 错误需要确认样本后另开配置，不能丢EOS。磁盘不足保留现有日志后先腾出空间。失败回传：`bash run.sh --stage package`。

回传最少文件：`runtime/results_return.tar.gz`。不需要回传 latest.pt、best 权重或原始音频；如后续要部署学生，再单独传送所选 best 目录。运行后状态以 `runtime/status.json`、各组 `complete.json` 及 `evaluation/summary.json` 为准，本静态表不伪造完成状态。
