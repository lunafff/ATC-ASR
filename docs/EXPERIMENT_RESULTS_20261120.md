# 第二轮 ASR 轻量化实验结果（已作废）

记录：2026-11-20。**本轮实验整体作废，不作为正式实验结果、基线或方法结论引用。以下数据仅保留作历史记录。**

配置更正：本轮教师误用了 ATCO2 单语料版；应使用 `jlvdoorn/whisper-large-v3-atco2-asr-atcosim` 混合版。下列数值保留为旧配置结果，混合版尚未运行。本轮蒸馏对比不能代表匹配语料教师下的效果。

## 模型与训练参数

| 角色 | 模型版本（HF revision） | 参数量 | 权重文件 |
|---|---|---:|---:|
| 教师 | `jlvdoorn/whisper-large-v3-atco2-asr` · `f7a0ddd` | 1.543B | 3.220 GB |
| 学生 | `openai/whisper-base` · `e37978b` | 72.59M | 0.290 GB |

- 数据：`jlvdoorn/atco2-asr-atcosim`（`21820a5`，ATCO2-ASR 与 ATCOSIM 混合）。从发布的 `train` 划出约 10% 作 Dev，发布的 `validation` 作本轮 Test；Train / Dev / Test 为 7,286 / 804 / 2,026 条。
- 学生训练：seed 42，5 epoch，AdamW，学习率 `1e-5`，batch 1，梯度累积 8；按最低 Dev WER 选最佳 epoch。教师冻结。
- 损失：CE 为纯监督基线；KD 温度 `T=2`，权重 `α=0.5`；对抗项权重 `λ=0.01` 或 `0.1`，鉴别器学习率 `1e-4`。

## 识别与推理指标

| 模型 / 训练组 | 最优 epoch | Dev WER | Test WER | Test RTF ↓ | Test 耗时（秒） |
|---|---:|---:|---:|---:|---:|
| ATCO2 large-v3 教师 | — | — | 15.642% | 0.12209 | 1,020.11 |
| 初始 Whisper-base | — | — | 86.666% | 0.02363 | 197.41 |
| Whisper-base + CE | 4 | **4.575%** | **4.285%** | **0.02191** | **183.05** |
| Whisper-base + CE + KD | 4 | 5.598% | 5.102% | 0.02278 | 190.30 |
| Whisper-base + CE + KD + 对抗（λ=0.01） | 5 | 6.217% | 7.134% | 0.02458 | 205.39 |
| Whisper-base + CE + KD + 对抗（λ=0.1） | 5 | 5.950% | 6.061% | 0.02335 | 195.08 |

推理口径：RTX A6000、FP32、batch 1、英文贪心解码、最多 256 个新 token、无 VAD。RTF = 推理耗时 / 音频时长；计时含特征提取和生成，不含读取与重采样。

## 简短分析

- 旧配置下 CE 比 KD 的 Test WER 低 0.817 个百分点，两个对抗配置也未改善 WER；CE 推理速度约为教师的 5.57 倍。这些仅为作废实验的历史观测，下一轮包含 CE 在内全部重跑。
- [作者的 WhisperATC 项目](https://github.com/jlvdoorn/WhisperATC#fine-tuning)区分 ATCO2、ATCOSIM 和混合语料版。本轮教师是 ATCO2 版，学生则直接在混合语料上适配。这个训练语料差异可能解释部分 WER 差距；单次混合集评测无法确定它和转录格式各贡献多少。[教师模型卡](https://huggingface.co/jlvdoorn/whisper-large-v3-atco2-asr)报告 17.0374% WER，但未写明评测划分和口径，不能与本轮 15.642% 直接比较。
- 下一轮若继续做 KD，先在 Dev 按 ATCO2 / ATCOSIM 分开看教师 WER，抽查数字、呼号等转录格式，再决定是否更换教师。若评估作者的混合语料版，先核对评测样本未被它用于训练或选模。
