# 服务器测试与结果回传

本文档说明如何在服务器端固定测试代码，并把结果同步回本地，方便在本地查看 `metrics.csv`、`metrics.json`、日志和解释性图片。

## 1. 固定测试入口

当前固定测试入口是：

```bash
python evaluate.py --config <config.yaml> --checkpoint <best.pt>
```

默认行为：

- 如果不传 `--combo`，会测试配置中所有非空模态组合。
- 如果传 `--combo t2 t1ce`，只测试指定组合。
- 测试阈值会优先读取 checkpoint 中保存的 `calibrated_threshold` 和 `threshold_calibration`。
- 输出默认写入 `<output_dir>/evaluate_only/`。

因此服务器端测试代码是固定的，推荐统一使用脚本：

```bash
bash scripts/server_evaluate_and_pack.sh \
  --config configs/private_t1_t2_t1ce.yaml \
  --checkpoint outputs/private_t1_t2_t1ce_run/checkpoints/best.pt \
  --run-name private_t1_t2_t1ce_eval
```

如果测试默认 BraTS 配置：

```bash
bash scripts/server_evaluate_and_pack.sh \
  --config configs/default.yaml \
  --checkpoint outputs/demo_run/checkpoints/best.pt \
  --run-name default_eval
```

如果只测试一个模态组合：

```bash
bash scripts/server_evaluate_and_pack.sh \
  --config configs/default.yaml \
  --checkpoint outputs/demo_run/checkpoints/best.pt \
  --run-name combo_t2_t1ce_eval \
  --combo t2 t1ce
```

脚本会生成：

```text
outputs/server_eval/<run_name>/
outputs/server_eval/<run_name>.tar.gz
```

关键文件：

- `evaluate.log`: 完整测试日志
- `command.txt`: 本次测试命令
- `run_metadata.txt`: 服务器、commit、Python 版本、输出路径
- `evaluate_only/metrics.csv`: 适合本地直接看表
- `evaluate_only/metrics.json`: 完整指标
- `evaluate_only/<combo>/`: 每个模态组合的解释性输出和图片

## 2. 本地拉取服务器结果

在本地 PowerShell 中执行：

```powershell
cd E:\EXPS\bratshgnn

.\scripts\fetch_server_results.ps1 `
  -Server "user@server_ip" `
  -RemoteArchive "/home/user/projects/MMhgnn/outputs/server_eval/private_t1_t2_t1ce_eval.tar.gz"
```

下载后结果会放到：

```text
E:\EXPS\bratshgnn\server_results\
```

其中解压后的主表通常是：

```text
server_results\<run_name>\evaluate_only\metrics.csv
```

## 3. 推荐日常流程

### 本地改代码并推送

```powershell
cd E:\EXPS\bratshgnn
git add .
git commit -m "Update evaluation workflow"
git push
```

### 服务器更新代码

```bash
cd ~/projects/MMhgnn
git pull --ff-only
```

### 服务器运行固定测试并打包

```bash
bash scripts/server_evaluate_and_pack.sh \
  --config configs/private_t1_t2_t1ce.yaml \
  --checkpoint outputs/private_t1_t2_t1ce_run/checkpoints/best.pt \
  --run-name private_t1_t2_t1ce_eval
```

### 本地拉取结果

```powershell
.\scripts\fetch_server_results.ps1 `
  -Server "user@server_ip" `
  -RemoteArchive "/home/user/projects/MMhgnn/outputs/server_eval/private_t1_t2_t1ce_eval.tar.gz"
```

## 4. 注意事项

- 不建议把 `outputs/` 提交到 GitHub；实验结果走 `scp` 拉回本地更合适。
- 每次测试建议用明确的 `--run-name`，例如 `baseline_seed42_eval`、`private_t1_t2_t1ce_eval`。
- 如果服务器会继续训练，不要开自动 `git pull`，避免训练中途代码变化。
- 本地解读结果时优先看 `metrics.csv`，再看每个组合目录下的解释性图片。

