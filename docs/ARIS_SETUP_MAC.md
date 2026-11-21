# Mac：下载并安装 ARIS 的 Codex skills

本仓库只记录下载/安装方法，不打包或复制Windows本机skills。ARIS官方仓库：

https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep

使用官方Codex适配包 `skills/skills-codex` 及安装器 `tools/install_aris_codex.sh`，不要误装Claude专用入口。依据：官方README的Codex说明和上述安装器帮助，2026-11-21核对。

## 给新Mac上的Codex执行

用户已要求为本项目下载并安装ARIS skills。读取HANDOFF.md后，可直接按下述方法执行，不需要用户重新逐条输入命令。先确认当前工作目录是ATC-ASR仓库，Git和Python3可用；只安装技能，不启动训练，不改实验配置。若已有ARIS下载目录，复用并核对来源，不覆盖无关文件夹。

在ATC-ASR仓库根目录，用macOS终端执行：

```bash
set -e
project_dir="$(pwd)"
aris_dir="$HOME/Developer/ARIS"
test -f "$project_dir/docs/HANDOFF.md"
mkdir -p "$HOME/Developer"

if [ ! -e "$aris_dir" ]; then
    git clone https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep.git "$aris_dir"
fi

# 已存在目录时，先核对来源和安装器；不要覆盖不明目录。
git -C "$aris_dir" remote -v
test -f "$aris_dir/tools/install_aris_codex.sh"
test -d "$aris_dir/skills/skills-codex"

bash "$aris_dir/tools/install_aris_codex.sh" "$project_dir" \
    --aris-repo "$aris_dir" --all --quiet --no-doc

git -C "$aris_dir" rev-parse HEAD
test -f "$project_dir/.agents/skills/analyze-results/SKILL.md"
test -f "$project_dir/.agents/skills/experiment-plan/SKILL.md"
test -f "$project_dir/.agents/skills/run-experiment/SKILL.md"
test -d "$project_dir/.agents/skills/shared-references"
```

官方安装器把技能链接到本项目 `.agents/skills/`，同时管理 `.aris/` 安装记录。保留 `~/Developer/ARIS`，否则链接失效。生成的本机链接和安装记录不提交到ATC-ASR；使用明确文件列表提交项目改动，不要直接 `git add -A`。

安装后下一轮对话即可使用skills；若当前会话未显示，重新打开项目并新建对话。可让Codex调用analyze-results分析回传包。第三方搜索、外部评审等可选技能可能需要额外工具或凭据；按实际任务配置，不复制Windows密钥。

## 后续更新

由Mac上的Codex先检查ARIS工作区没有本地修改，再执行：

```bash
git -C "$HOME/Developer/ARIS" pull --ff-only
bash "$HOME/Developer/ARIS/tools/install_aris_codex.sh" "$PWD" \
    --aris-repo "$HOME/Developer/ARIS" --all --add-new --quiet --no-doc
```

每次记录上游提交号。该方法下载当时的官方版本，不保证与旧Windows本机的87个skills快照完全相同。
