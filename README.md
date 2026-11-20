# 第二轮 ASR 轻量化实验（2026-11-17）

本目录是独立交付单元。只用 Git 传代码与配置；权重、数据、检查点和日志全部写入被忽略的 `runtime/`。历史实验不覆盖。本机不训练。完整计划见 [实验设计](docs/EXPERIMENT_PLAN.md)，逐项运行记录见 [RUNBOOK](docs/RUNBOOK.md)。

学生为 **openai/whisper-base 和 openai/whisper-small 两种多语言模型**，各自从预训练权重独立运行 CE、KD、KD+对抗0.01、KD+对抗0.1，**合计8组正式训练**。base revision `e37978b90ca9030d5170a5c07aadb050351a65bb`，small revision `973afd24965f72e36ca33b3055d56a652f456b4d`。共同教师为 **jlvdoorn/whisper-large-v3-atco2-asr-atcosim 混合版**，revision `efd96bd7862709d2d6aeb55621934a9965db8ed7`。两个配置除学生名称与版本外完全一致。旧 base + 单语料教师实验整体作废。此次英语训练和评测不能证明中文能力保持。

## GPU 机器一键执行

用户目标环境：Windows、A6000、可访问 Hugging Face。安装 Git、NVIDIA 驱动和 Python 3.11（推荐）；支持 Python 3.10–3.12，不支持仅笼统的任意 Python 3.x。磁盘预留至少 80GB；显存和耗时以 smoke 为准，未在本机实测。脚本创建独立虚拟环境，安装 CUDA 12.1 版 PyTorch 2.5.1 和固定依赖，不安装系统驱动，也不要求另装完整 CUDA Toolkit。驱动需要支持该 CUDA 运行时。首次下载较慢。

本机将代码 push 到远程仓库；GPU 机器首次在 PowerShell 执行：

```powershell
git clone -b codex/asr-round2 https://github.com/lunafff/ATC-ASR.git asr-round2
cd asr-round2
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Windows 默认 `run.ps1` 顺序完成 base 四组，再完成 small 四组，两个学生各先做 smoke。目录分别为 `runtime/base/`、`runtime/small/`，数据清单、下载、检查点、完成标记和日志均隔离；划分算法及数据版本相同。回传 **`runtime/base/results_return.tar.gz` 和 `runtime/small/results_return.tar.gz` 两个包**，请分别命名为 base_results_return.tar.gz、small_results_return.tar.gz。每包包含逐条预测、汇总CSV/JSON、损失、配置和版本，不含权重或音频。

重跑同一命令会跳过已完成组，从最近完整 epoch 恢复未完成组。单独运行可用 `-Student base` 或 `-Student small`；自定义 `-Config` 必须显式选择一个学生，且模型名称必须匹配。旧单学生的 `runtime/provenance/config.json` 存在时入口拒绝启动，请先停掉旧进程并归档整个旧 runtime。

若 Python Launcher 中没有 3.11，可用 `-Python 'C:\路径\python.exe'` 显式指定兼容解释器。镜像可设置 `$env:HF_ENDPOINT='https://hf-mirror.com'`；凭据使用机器自身的 `HF_TOKEN` 环境变量，不写进 Git。

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage prepare
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage train
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage final
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage package
```

旧 Linux 入口 `bash run.sh` 仍只直接运行单个配置；本轮8组一键入口是 Windows 的 `run.ps1`。

默认 `all` 顺序执行上述流程，smoke 异常立即停止。修改代码、配置或种子开展新一轮之前，将整个 `runtime` 移到另一归档目录；脚本拒绝混用不同代码/配置。不要在运行期间 git pull 修改代码。

## 远程 Git 同步

本目录已初始化独立仓库，分支 `codex/asr-round2`，不包含旧项目历史。仅提交本轮代码、配置和文档；runtime 已被忽略。远程 origin 使用 https://github.com/lunafff/ATC-ASR.git。

```bash
# 本机，首次配置并推送
git remote add origin https://github.com/lunafff/ATC-ASR.git
git push -u origin codex/asr-round2
# 本机，后续开发提交后
git push
# GPU 机器，等待当前实验结束后再更新
git pull --ff-only origin codex/asr-round2
```

修改训练代码后不能接着用旧 runtime 训练；需先归档 runtime。Windows首次执行可用 `nvidia-smi`、`py -3.11 --version` 检查驱动和解释器；入口会验证 Python 版本。

## 验证状态

本地完成 Python 语法检查和7项无依赖契约测试，包括两个学生配置一致性、进程间运行目录隔离和对抗组结果文件命名。本地未运行真实模型前向/反向或 CUDA smoke。GPU 机器为每个学生先执行真实 smoke 后才继续其四组实验。历史代码审查记录见 docs/EXPERIMENT_CODE_REVIEW.md。
