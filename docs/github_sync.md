# GitHub 同步操作指南

本文档用于把 `E:\EXPS\bratshgnn` 同步到 GitHub，并让服务器通过 GitHub 获取本地最新代码。

## 0. 当前仓库状态

- 当前本地仓库还没有 commit。
- 当前没有配置 GitHub remote。
- 本项目包含私有数据、训练输出、缓存和本地路径配置，默认不建议上传。
- 已通过 `.gitignore` 忽略 `privatedata/`、`outputs/`、`node_modules/`、`__pycache__/`、权重文件、医学影像文件和 `configs/private_*.yaml`。
- `configs/private_t1_t2_t1ce.example.yaml` 是可上传模板；服务器上使用时复制成 `configs/private_t1_t2_t1ce.yaml` 并填写真实路径。

## 1. 在 GitHub 创建空仓库

推荐仓库名：

```bash
bratshgnn
```

创建时建议：

- Visibility: 如果包含未发表研究代码，优先选 `Private`。
- 不要勾选自动生成 README、.gitignore、license，因为本地已有 README，且我们已经添加 `.gitignore`。

创建后你会得到一个远端地址，二选一：

```bash
git@github.com:<your_name>/bratshgnn.git
```

或：

```bash
https://github.com/<your_name>/bratshgnn.git
```

服务器长期同步更推荐 SSH 地址。

## 2. 本地首次提交并推送

在 Windows PowerShell 中进入项目：

```powershell
cd E:\EXPS\bratshgnn
```

确认哪些文件会被提交：

```powershell
git status --short
```

建议先检查忽略规则是否生效：

```powershell
git status --ignored --short
```

首次提交：

```powershell
git add .
git status --short
git commit -m "Initial bratshgnn project"
```

绑定 GitHub 远端。把 `<your_name>` 改成你的 GitHub 用户名或组织名：

```powershell
git remote add origin git@github.com:<your_name>/bratshgnn.git
git branch -M main
git push -u origin main
```

如果你使用 HTTPS：

```powershell
git remote add origin https://github.com/<your_name>/bratshgnn.git
git branch -M main
git push -u origin main
```

## 3. 服务器首次拉取代码

在服务器上选择一个代码目录，例如：

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:<your_name>/bratshgnn.git
cd bratshgnn
```

创建 Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

创建服务器私有配置：

```bash
cp configs/private_t1_t2_t1ce.example.yaml configs/private_t1_t2_t1ce.yaml
```

然后编辑 `configs/private_t1_t2_t1ce.yaml`，填入服务器真实的 `data_root` 和 `label_xlsx`。该文件默认被 `.gitignore` 忽略，不会上传到 GitHub。

## 4. 日常同步流程

### 本地修改后推送到 GitHub

```powershell
cd E:\EXPS\bratshgnn
git status --short
git add .
git commit -m "Describe your change"
git push
```

### 服务器更新到最新代码

```bash
cd ~/projects/bratshgnn
git status --short
git pull --ff-only
```

如果服务器上没有改代码，只跑实验，`git pull --ff-only` 是最稳妥的更新方式。

## 5. 推荐工作规则

1. GitHub 只同步代码、配置模板、文档和小型脚本。
2. 原始影像、标签表、训练输出、模型权重和私有路径配置不要进 GitHub。
3. 服务器上只保留本地私有文件，例如 `configs/private_t1_t2_t1ce.yaml`、数据目录和输出目录。
4. 每次本地推送前先运行 `git status --short`，确认没有误加入数据或权重。
5. 如果确实需要同步大模型权重，单独启用 Git LFS，不要直接用普通 Git 管理。

## 6. 常见问题

### 服务器 pull 时提示本地有修改

先查看修改：

```bash
git status --short
git diff
```

如果这些修改是服务器临时实验，不应该入库，可以复制到别处后再处理。不要直接 `git reset --hard`，除非你明确确认这些修改可以丢弃。

### 误把大文件加入暂存区

如果还没 commit：

```bash
git restore --staged <file>
```

然后确认 `.gitignore` 是否覆盖该文件类型。

### 需要服务器自动更新

最简单可控的方式是先手动：

```bash
git pull --ff-only
```

如果后续需要自动化，可以再加 cron 或 GitHub webhook。但研究代码建议先用手动更新，避免训练中途被自动改代码。

