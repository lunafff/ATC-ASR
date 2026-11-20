# 实验设计

研究问题：244M Whisper-small 学生在相同数据和训练预算下，KD 能否优于纯监督；对抗分布匹配是否在 KD 之上进一步改善 WER，较强系数是否反而损害迁移。当前是单种子探索，不作统计显著性或实体准确率提升结论。

模型选择：学生 https://huggingface.co/openai/whisper-small （多语言，约244M）；教师 https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr-atcosim （混合语料版）。按用户要求完全替换 base，保留四组原有超参数和数据划分，不增加第二个学生。旧实验整体作废。教师模型卡列出 Apache-2.0，但训练数据描述不完整；其卡片 WER 不能直接当成本实验基线。模型及数据 revision 已在配置固定。实测参数量、权重字节数由评测脚本记录。

教师 128 mel、学生 80 mel，各自独立提取同一音频特征。teacher 51866 词、student 51865 词，通过 token 字符串构造映射（已确认的 nocaptions/nospeech 无语音标记使用显式别名）；教师 logits 在学生词表支持集上重新归一化，明确不是直接对完整不同词表做 KL。共享标注前缀、转录、EOS 均映射到教师坐标后 teacher forcing；教师无梯度。没有层对齐损失。

损失：CE 为人工转录的逐 token 交叉熵；KD = T² mean KL(p_teacher^T || p_student^T)，T=2；总损失 (1-alpha) CE + alpha KD + lambda_adv G，alpha=.5。batch=1，因此没有 padding 位置；保留 EOS，去除开头 decoder-start，标签超过 256 token 直接报错而不静默截断。默认每 8 条累计一次梯度，最后不足 8 条按实际条数平均，学生梯度裁剪 1。

对抗：固定随机 64 维投影将 soft token distribution 映射到共同空间，MLP D 判断 teacher-real / student-fake。D 最小化 BCE(D(detach teacher),1)+BCE(D(detach student),0)；学生最小化 BCE(D(student),1)，此阶段冻结 D，梯度仍通过 D 回到学生。教师始终冻结，CE 始终锚定人工真值。**这是教师/学生分布匹配消融，不是原 TODO 中教师输出/人工真值的鉴别，也不是实体加权。**实体无标注，相关损失和指标留空。

每组均从同一预训练学生和 seed 42 开始，5 epochs，AdamW lr 1e-5。D lr 1e-4，lambda=0.01/0.1。每个 epoch 在 dev 上贪心生成算 WER，保留最低 WER（平局保留最早）；完整组全部训练结束才测试，不能根据 test 选择 lambda。当前不使用 scheduler 或 early stopping，各组训练预算相同。单卡顺序运行，不自动租卡或远程提交。

数据：下载既有公共 jlvdoorn/atco2-asr-atcosim 的官方 train/validation parquet。官方 validation 定为本轮 held-out test；训练部分按解码后音频内容哈希固定 90/10 train/dev。跨全部划分去重，test 优先；同音频不同标注直接报错。仅接受非空文本和 (0,30] 秒音频，其他记录及数量保存。不是 speaker/session 独立划分，也不能排除公共教师接触过测试集；本轮只能作此公共数据上的探索，独立本地保留集验证是后续必要工作。数据下载/使用须遵守原数据提供者许可，不随代码再分发。

统一评测：Transformers、float32、batch=1、greedy、English transcribe、无 timestamps、无提示词、无 VAD、无量化。规范化：英文小写、去撇号、其他标点转空格、合并空白；不映射数字或同义读法。WER 对人工真值，存 0–1 比例（可超过1）；报告 RTF、参数量、权重字节、相邻重复词计数、触及长度上限数量。RTF 排除磁盘加载，包含特征与生成，先 warmup 再 CUDA synchronize。相邻重复词仅为诊断，合法重复也会计入。此轮不作生产端到端延迟结论。

算力预算：本机未实测，不编造 GPU 小时估计。先 2 样本 smoke（包含 CE/KD/D/G 和 2 条 dev 生成），完整 4 组 × 5 epochs，另 6 个模型最终测试。训练每20样本打印进度，每样本落损失 JSONL；smoke 通过只证明链路可运行。检查 GPU 显存与单轮耗时后可以中断；如果需修改预算，应归档 runtime 后另开配置。
