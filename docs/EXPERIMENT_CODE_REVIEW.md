# 实验代码审查记录（2026-11-17）

Windows适配增补：用户确认Windows/A6000/HF可访问，新增run.ps1，显式安装CUDA12.1 PyTorch轮子并验证Python3.10–3.12。PowerShell AST语法解析成功，Python编译与4项测试再次成功；未在本机安装或运行CUDA训练。Git仓库仅纳入本轮文件。

采用 ARIS experiment-bridge 的新鲜上下文审查：原生 collaboration reviewer，gpt-6-astra，xhigh，fork_turns=none；同模型家族，结论属于 provisional，不是跨家族独立背书。当前安装包找不到 reviewer_route.py，故没有声称已运行该辅助解析器；使用技能规定的 host default 原生审查路径，审查任务 /root/code_review。

已核实核心训练：冻结教师、词表字符串映射、分别提取128/80 mel、保留EOS、KD T²、D/G梯度路径、尾部梯度累积；未发现已确认阻塞错误。没有执行CUDA验证。

确认问题及处置：

1. 最终评测中断于指标保存和 weight_bytes 补写之间，恢复会跳过并造成汇总 KeyError：已修复，仅当预测和指标均存在且含 weight_bytes 才跳过。复核确认。
2. Whisper 4.46.3 返回序列会删尾部 pad/EOS，不能按最后token判断截断；输出保留前缀，因此长度阈值改为 max_new_tokens + prefix长度。未启用 output_scores，避免额外拷贝污染RTF。
3. 非阻塞：中断epoch重跑会有重复损失日志；runbook明确按epoch/sample去重，完成状态由检查点/complete文件判定。
4. 非阻塞：环境记录覆盖；已改为保存首次快照并逐次追加带时间戳的 invocation 环境及pip记录。

本地检查：Python compileall成功；4项纯Python契约测试成功（词表置换、多余教师词、不支持token报错、无语音标记别名、规范化）；git diff --check无错误。本机 Python 3.14、已有venv无torch，不冒充目标Python3.11/CUDA通过。真实模型加载、前向反向、显存、CUDA smoke待目标机器执行。smoke自动检查损失/学生梯度有限且非零、教师无梯度；失败停止。

真实远端tokenizer JSON核对：student 51865，teacher 51866，1507个token ID不同；学生唯一缺失标记为 `<|nocaptions|>`，教师新增 `<|nospeech|>` 和 `<|yue|>`。已显式将 nocaptions 映射至 nospeech（同为无语音标记），其余按字符串匹配，并检查映射一一对应。新增回归测试；未知缺失标记继续报错，不按位置猜测。

剩余限制：公开教师训练数据不可核实；本轮单种子、utterance级划分，实体无标注，不能据此宣称独立泛化或实体级收益。

Final fresh-review follow-up: alias mapping, prefix-adjusted cap, environment snapshots and evaluation resume checked; no new confirmed blockers. CUDA smoke remains pending.
