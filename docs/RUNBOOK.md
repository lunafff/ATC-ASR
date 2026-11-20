# Runbook / 实验追踪表

## 2026-11-21 本地下载复用修复

下载前按 configs/assets.json 中固定revision的上游文件大小、SHA256或Git blob摘要检查本地文件。优先当前目录，随后查找 runtime/base、runtime/small、runtime_old*（含旧单学生及双学生目录）。匹配文件通过硬链接复用，跨盘时复制；不修改旧文件。全部匹配时不会调用HF下载API，缺失/损坏文件才下载。数据只要求7个固定parquet，README不再请求。两个学生仍生成各自独立的音频、清单、训练结果。

当前失败在下载阶段，拉取后直接重跑 run.ps1，不删除或再次归档 runtime。仅在对应学生尚无训练产物时允许更新源码锁并保留旧源码摘要；已开始训练的目录仍禁止混用新版源码。

本机E盘空间不足，Git元数据及src/configs/tests/docs已迁移至C:/Users/lx/.codex/experiment-archives/中的round2_git_storage、round2_working_storage，通过目录联接保留原工作区路径；不影响远程仓库或GPU机器目录。

**2026-11-20 决定：旧第二轮整体作废。下方完成状态仅为历史运行记录，不作为有效基线或方法结论；改用混合版教师后，CE、KD、两个对抗组及评测全部重跑。**

耗时预算（同一 A6000、当前 5 epoch 配置）：原包文件时间显示 smoke 完成于 11-18 01:51，整轮完成于 12:47，训练与最终评测合计约 10 小时 56 分。按相邻组完成时间估计（包含加载、Dev 评测和保存）：CE 约 51 分钟，KD 约 3 小时 10 分，两个对抗组各约 3 小时 10 分，最终评测和打包约 35 分钟。混合版架构相同，预计计算部分约 11 小时，计入下载与数据准备预留 12–14 小时；网络较慢另计。以上依据归档时间估算，不是逐阶段精确计时。

## 2026-11-20 base + small，各4组，共8组

最终范围：`openai/whisper-base`（配置 configs/base.json）和 `openai/whisper-small`（配置 configs/default.json）分别完成 CE / KD / KD+对抗0.01 / KD+对抗0.1，合计8组。每组5 epoch、seed42、lr1e-5、batch1、累积8、T2、KD权重0.5、D lr1e-4。两个配置只在学生名称及revision上不同。数据及划分算法未改变。前述耗时属于旧 base 实验，不能作为8组的总耗时承诺。

默认 `run.ps1` 串行执行 base→small，失败立即停止；两者的 smoke、配置锁、源码锁、检查点、完成标记、日志和结果包分别位于 runtime/base 与 runtime/small。可用 `-Student base` 或 `-Student small` 单独续跑。配置/源码不变时，直接重复原命令即可恢复，不要重复归档新运行目录。

| 学生 | 正式训练组 | 当前状态 | 回传包 |
|---|---|---|---|
| base | CE / KD / KD+adv0.01 / KD+adv0.1 | 待服务器运行 | runtime/base/results_return.tar.gz |
| small | CE / KD / KD+adv0.01 / KD+adv0.1 | 待服务器运行 | runtime/small/results_return.tar.gz |

本地核对了固定版本的模型配置、特征配置及完整 tokenizer.json：教师32+32层/128 mel/51866词，学生12+12层/80 mel/51865词；51865个学生token均存在一对一教师映射。本地未验证CUDA实际前向/反向，服务器入口先运行真实smoke（包括CE/KD/D/G）再进行四组训练。

默认教师改为 `jlvdoorn/whisper-large-v3-atco2-asr-atcosim`，revision `efd96bd7862709d2d6aeb55621934a9965db8ed7`。下方已完成记录均属于旧 ATCO2 单语料版教师。混合版的 KD / 对抗结果需重跑，不能沿用旧检查点。另修复对抗组小数名称导致预测文件互相覆盖的问题。

GPU 机器必须在当前任务停止后更新；先将旧 runtime 改名留存，再启动新运行。不要把旧 `models/teacher`、`complete.json` 或 `best/` 复制回新 runtime。以下命令假定本机更改已推送：

```powershell
git pull --ff-only origin codex/asr-round2
if ($LASTEXITCODE -ne 0) { throw 'Git pull failed' }
if (Test-Path -LiteralPath .\runtime) {
    $archivePath = Join-Path $PWD ('runtime_old_atco2_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
    Move-Item -LiteralPath (Join-Path $PWD 'runtime') -Destination $archivePath -ErrorAction Stop
}
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

新运行在 runtime/base 和 runtime/small 中完整执行。两个结果包请分别命名为 base_results_return.tar.gz、small_results_return.tar.gz 后回传。原始结果页保留旧配置数值；本轮8组结果尚未生成。E盘空间耗尽后，本机旧回传包已原样移至 C:/Users/lx/.codex/experiment-archives/round2_20261120/results_return.tar.gz，SHA256 校验一致。

## 以下为旧单学生实验的历史运行记录（已作废）

安装故障修复（2026-11-17）：用户Windows Python3.11自带pip在cu121单一索引安装时拒绝typing_extensions包名元数据，继而找不到flit_core。run.ps1已改为先从PyPI升级pip，以PyPI为主索引、cu121为额外索引，并精确指定torch==2.5.1+cu121；普通依赖从PyPI安装，随后pip check。此次失败在训练前，无需删除runtime、venv或pip缓存；拉取修复后重跑即可。

## Windows A6000 入口（用户已确认）

目标 Windows + A6000，可访问HF。安装Git、NVIDIA驱动、Python3.11后，用 `git clone -b codex/asr-round2 https://github.com/lunafff/ATC-ASR.git asr-round2` 克隆本轮独立仓库，在目录内执行 `powershell -ExecutionPolicy Bypass -File .\run.ps1`。后续更新用 `git pull --ff-only origin codex/asr-round2`，本机提交后使用 `git push`。入口验证Python3.10–3.12、建立venv、显式安装CUDA12.1版PyTorch，再运行完整流水线。分阶段用 `-Stage prepare/train/final/package`。Python路径用 `-Python 'C:\路径\python.exe'` 指定。查看日志：`Get-Content .\runtime\logs\launcher.log -Tail 30 -Wait`。Windows不要执行下文Linux的nohup命令。

下面保留跨平台流程表；其中bash命令为Linux备选，Windows使用上述PowerShell等效入口。

2026-11-20 复核：RTX A6000 于 2026-11-18 12:47（北京时间）完成 smoke、四组训练、测试和打包。指标、逐条复算及证据限制见 [实验结果与评审](EXPERIMENT_RESULTS_20261120.md)。

| ID | 阶段/脚本 | 输出 | 当前状态 |
|---|---|---|---|
| P00 | run.ps1 安装环境；pipeline.py 记录环境/代码 | runtime/provenance | 完成；提交 6fae99d |
| P01 | prepare.py 下载、导出、划分 | models / dataset / audio / manifests | 完成；7286/804/2026 条 |
| S00 | engine.py smoke | runs/smoke | 通过 |
| R01 | CE，seed42，5 epochs | runs/supervised | 完成；最优 epoch 4，test WER 4.285% |
| R02 | CE+KD，T2，alpha .5 | runs/kd | 完成；最优 epoch 4，test WER 5.102% |
| R03 | CE+KD+adv .01 | runs/kd_adv_0.01 | 完成；最优 epoch 5，test WER 7.134%（仅汇总） |
| R04 | CE+KD+adv .1 | runs/kd_adv_0.1 | 完成；最优 epoch 5，test WER 6.061% |
| E00 | teacher、初始student、R01–04统一测试 | evaluation/*.jsonl，summary.json/csv | 完成；0.01 组逐条预测被命名碰撞覆盖 |
| Z00 | pipeline.py package | results_return.tar.gz | 已回传并复核 |

现场步骤：确认驱动 `nvidia-smi`、Python `python3.11 --version`、80GB可用磁盘；拉取固定 commit，执行 `bash run.sh`。需要后台运行时可用 `nohup bash run.sh > launcher.console.log 2>&1 &`，之后用 `tail -f runtime/logs/launcher.log` 查看。

目录职责：`models/` 下载原始权重；`dataset/` 原始 parquet；`audio/` 导出的音频；`manifests/` 冻结样本清单和摘要；`runs/<组>/best/` dev最优权重及processor；`latest.pt` 完整 epoch 断点，含学生/D/优化器/RNG；`train.jsonl` 损失；`evaluation/` 最终预测与指标；`provenance/` 配置、源码摘要、版本；`logs/` 启动输出；`failure.json` 异常。

恢复：同一代码/配置重跑 `bash run.sh`。中断 epoch 从头重算，该 epoch 之前已写入的损失日志会保留，分析时按 epoch/sample 去重；以 dev_epoch 文件及 complete.json 判定真正完成。下载使用HF缓存断点续传。不要在训练期间打包/修改配置。

故障：先读 failure.json 和 launcher.log；OOM 不会暗中改变 batch/数据/精度，需记录后归档再改配置。下载失败检查网络/镜像/HF_TOKEN；驱动问题在GPU机处理。超长 target 错误需要确认样本后另开配置，不能丢EOS。磁盘不足保留现有日志后先腾出空间。失败回传：`bash run.sh --stage package`。

回传最少文件：`runtime/results_return.tar.gz`。不需要回传 latest.pt、best 权重或原始音频；如后续要部署学生，再单独传送所选 best 目录。运行后状态以 `runtime/status.json`、各组 `complete.json` 及 `evaluation/summary.json` 为准，本静态表不伪造完成状态。
