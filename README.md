# 第二轮 ASR 轻量化实验（2026-11-17）

本目录是独立交付单元。只用 Git 传代码与配置；权重、数据、检查点和日志全部写入被忽略的 `runtime/`。历史实验不覆盖。本机不训练。完整计划见 [实验设计](docs/EXPERIMENT_PLAN.md)，逐项运行记录见 [RUNBOOK](docs/RUNBOOK.md)。

学生统一采用 **openai/whisper-small 多语言版，约 244M 参数、12 层 encoder + 12 层 decoder**，revision `973afd24965f72e36ca33b3055d56a652f456b4d`；教师采用 **jlvdoorn/whisper-large-v3-atco2-asr-atcosim 混合版**（32+32 层，约 1.55B），revision `efd96bd7862709d2d6aeb55621934a9965db8ed7`。四个平行组保持 CE、KD、KD+对抗0.01、KD+对抗0.1，超参数和数据划分不变。旧 base + 单语料教师实验整体作废。学生全部微调，不复制教师层、不冻结旧 turbo 编码器。此次英语陆空通话数据训练和英语评测不能证明中文能力保持。

## GPU 机器一键执行

用户目标环境：Windows、A6000、可访问 Hugging Face。安装 Git、NVIDIA 驱动和 Python 3.11（推荐）；支持 Python 3.10–3.12，不支持仅笼统的任意 Python 3.x。磁盘预留至少 80GB；显存和耗时以 smoke 为准，未在本机实测。脚本创建独立虚拟环境，安装 CUDA 12.1 版 PyTorch 2.5.1 和固定依赖，不安装系统驱动，也不要求另装完整 CUDA Toolkit。驱动需要支持该 CUDA 运行时。首次下载较慢。

本机将代码 push 到远程仓库；GPU 机器首次在 PowerShell 执行：

```powershell
git clone -b codex/asr-round2 https://github.com/lunafff/ATC-ASR.git asr-round2
cd asr-round2
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

本仓库仅含本轮新代码，不包含原项目旧文件。重跑同一命令会跳过完成的实验，未完成训练从最近完成的 epoch 恢复；中断 epoch 会从头重跑。结果：`runtime/results_return.tar.gz`，回传此文件即可，包含逐条预测、汇总 CSV/JSON、损失日志、配置、数据清单与版本信息，不含权重和音频。

若 Python Launcher 中没有 3.11，可用 `-Python 'C:\路径\python.exe'` 显式指定兼容解释器。镜像可设置 `$env:HF_ENDPOINT='https://hf-mirror.com'`；凭据使用机器自身的 `HF_TOKEN` 环境变量，不写进 Git。

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage prepare
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage train
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage final
powershell -ExecutionPolicy Bypass -File .\run.ps1 -Stage package
```

仍保留 Linux 入口 `bash run.sh`，不是 Windows 的默认入口。

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

本地完成 Python 编译检查和4项无依赖契约测试；本地没有 PyTorch，因此没有声称通过真实模型前向/反向或 CUDA smoke。GPU 机器启动时自动执行真实 smoke 后才继续完整实验。正式运行前的状态及审查意见见 docs/EXPERIMENT_CODE_REVIEW.md。
