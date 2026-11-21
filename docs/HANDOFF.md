# 跨电脑接续交接（2026-11-21）

新会话先读此文件，再读 RUNBOOK.md、EXPERIMENT_PLAN.md 和两个学生配置。本文代替完整聊天记录；不包含凭据。后续工作结束时更新当前状态。

## 仓库和分工

- Git: https://github.com/lunafff/ATC-ASR.git ，分支 codex/asr-round2。
- 此仓库是独立实验交付目录，不包含原Windows工作区中全部研究资料、历史代码或本地ARIS skills。
- Mac/开发机：修改、检查、提交和push。Windows GPU机器：pull并运行。当前pipeline要求NVIDIA CUDA，不能直接在Mac上训练。
- GPU机器：Windows，RTX A6000，Python3.11.4，PyTorch2.5.1+cu121。路径 E:\zyfbs\ATC-ASR\ATC-ASR。
- 不要在服务器训练进程运行时pull或改源码。两台开发电脑切换前，提交并push；另一台开始前先pull，避免互相覆盖。

## 已确定的实验，不要误改

- 教师：jlvdoorn/whisper-large-v3-atco2-asr-atcosim 混合版，efd96bd7862709d2d6aeb55621934a9965db8ed7。不能误用仅atco2-asr版。
- 学生同时保留两种：base（e37978b90ca9030d5170a5c07aadb050351a65bb）与small（973afd24965f72e36ca33b3055d56a652f456b4d），均为多语言版。
- base和small各独立跑 CE / KD / KD+对抗0.01 / KD+对抗0.1，共8组正式实验，另各自先做smoke；同一A6000顺序运行base再small。
- configs/base.json 是base；configs/default.json 是small。两配置仅学生名称和revision不同。
- 同一混合数据集 jlvdoorn/atco2-asr-atcosim，revision 21820a50f2f4d0a8f90838feb4efe88c3db31f95；发布validation作本轮Test，train按音频哈希切出约10% Dev，去重后7286/804/2026条。
- 5 epochs，seed42，AdamW lr1e-5，batch1，累积8；KD T2、权重0.5；D lr1e-4。FP32、英文贪心、无VAD。未经用户要求不改实验参数。
- 旧base+单语料教师的第二轮整体作废，仅保留历史文档，不能将其4.285% WER作为本轮正式基线或混合版结果。

## 当前状态和下一步

用户报告新流程已进入训练，手动中断后要求续训和监控。加入监控后源码锁阻止恢复；最新已推送修复 e1072b9（上一版2c89ee2，下载复用版7421171，双学生版dd02ee3）。截至此交接，尚未收到服务器成功恢复的确认，也没有新8组的结果包。优先确认续训状态，不能宣称训练已完成。

现在支持完整epoch边界恢复；中断epoch会从头重跑，第一轮未保存则该组从头开始。恢复学生、优化器、D、投影和CPU/CUDA随机状态；已完成组跳过。重复epoch日志归档至 interrupted_logs，曲线去重。

源码兼容由 configs/monitor_compatibility.json 严格核对已发布旧/新源码hash，支持已知LF/CRLF及路径分隔符差异。不要删除sources.json或随意放宽检查。新增训练源码改动会触发保护，即便只修改监控，也要重新审查并更新迁移摘要。当前修复已通过17项CPU测试、3个历史版本LF/CRLF迁移检查，未在本机运行CUDA续训。

## 服务器操作（保留现有runtime）

当前runtime已是新双学生结构，不要再次归档、删除或清空。旧Legacy runtime问题已经处理过。

```bat
git pull --ff-only origin codex/asr-round2
```

拉取成功后，在CMD或PowerShell均可执行：

```bat
powershell -ExecutionPolicy Bypass -Command "$env:HF_ENDPOINT='https://hf-mirror.com'; & .\run.ps1"
```

也可单独续跑 -Student base 或 -Student small。HF直连超时，现用第三方镜像；数据README曾503。现已按固定资产清单验大小和摘要，自动复用当前目录、另一学生和runtime_old*中的模型/数据；匹配则不联网，缺失才下载，不请求README。

若源码仍被拒绝，向用户获取 runtime/base/provenance/source_mismatch.json（或small对应文件）定位；不要让用户重训或删除保护文件。

## 观察与回传

- 两学生数据、checkpoint、状态、日志和结果分别在 runtime/base、runtime/small，runtime被Git忽略，pull不会传训练状态。
- 日志 runtime/<学生>/logs/launcher.log；各组 runs/<组>/train.jsonl、latest.pt、best/、complete.json。
- 每20样本打印时间、进度、速度、ETA与loss；每200样本/epoch更新 training_curves.svg 和 training_curves.html，浏览器页面每30秒刷新。包含loss/CE/KD/G/D、Dev WER和RTF。
- 单独绘图（无CUDA需求）：python src/training_monitor.py --work-dir runtime/base；small同理。
- 回传 runtime/base/results_return.tar.gz 和 runtime/small/results_return.tar.gz；重命名为base_results_return.tar.gz和small_results_return.tar.gz避免覆盖。两个包都不随Git传输，需另发给Mac。
- 旧Windows开发机因E盘满，旧结果包及部分工作区物理存储移至C盘并以目录联接保留路径；这只是旧机本地设置。Mac应正常git clone，不复制这些联接。

## Mac首次接续

```bash
git clone -b codex/asr-round2 https://github.com/lunafff/ATC-ASR.git
cd ATC-ASR
python3 -B -m unittest discover -s tests -v
```

单元测试只需Python标准库，不用为开发检查安装CUDA环境。若python3尚未安装，先安装Python3.11。训练环境保留在Windows服务器，不复制.venv。ARIS skills需在Mac单独安装；仓库已有实验计划、runbook和结果文档，不依赖完整对话。原项目其他未入此仓库的研究资料需要另行同步。

## 用户偏好

中文沟通。结果记录要求简洁：模型版本、训练参数、准确率/WER、推理指标、简短分析，不加完整性复核长篇。先核对模型/语料匹配和异常结果，再做研究结论。不要将base误替换掉：本轮必须base和small都跑。
