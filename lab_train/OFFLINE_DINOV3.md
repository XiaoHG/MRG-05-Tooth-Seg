# 离线 lab_train：官方 DINOv3 牙科预训练（渐进式）

服务器不能联网；所有训练所需内容已经放进 `lab_train/`：

```text
lab_train/
  third_party/dinov3/                  # 官方 DINOv3 代码（无需 clone/联网）
  offline_wheels/                      # Linux x86_64 / Python 3.11 离线 wheels
  requirements-offline.txt
  weights/dinov3-vitb16-official.pth   # 已转换的官方格式权重（327 MB）
  pretrain/official_dinov3_patch/      # ImageFolder 补丁 + 3 份配置
    config_dental_vitb_224_diag.yaml   # Stage 0：2,220 步诊断
    config_dental_vitb_224.yaml        # Stage 1：224/96 主训练
    config_dental_vitb_512.yaml        # Stage 2：512/128 高分辨率适应
  pretrain/audit_pretrain_data.py      # 数据审计
  pretrain/build_dedup_view.py         # 按 SHA256 建去重视图（硬链接）
  setup_offline_dinov3.sh              # 离线安装 + 打补丁 + 生成配置
  run_official_dinov3.sh               # 启动训练
```

> 不要复制 `lab_train/dataset/` 和 `lab_train/output/` 到服务器：
> 服务器已有 55,559 张源文件和 `dataset/pretrain/pretrain_manifest.json`。
> 本地 `dataset/pretrain/pretrain_manifest.json` 是 0 张图的占位文件，覆盖会破坏服务器清单。
> 本地 `output/` 是旧的实验产物，也不需要复制。

## 0. 复制与安装

```bash
# Mac
rsync -av --exclude 'dataset/' --exclude 'output/' \
  lab_train/ cs_cs_qjh@ecs-instance-fd926d86:~/MRG-05-Tooth-Seg/lab_train/

# 服务器
ssh cs_cs_qjh@ecs-instance-fd926d86
cd ~/MRG-05-Tooth-Seg/lab_train
conda activate toothseg
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

bash setup_offline_dinov3.sh --data-root ~/MRG-05-Tooth-Seg/lab_train/dataset/pretrain_dedup
```

> 如果此时 `dataset/pretrain_dedup` 还没创建，setup 会打印
> `WARNING: data root does not exist`，但仍会把正确路径写入配置；下一步创建即可。
> 先建数据再 setup 也可以，顺序不影响最终配置。

`setup_offline_dinov3.sh` 会：

1. 离线安装 `omegaconf/ftfy/regex/torchmetrics/submitit/termcolor/scikit-learn/...`；
   `--no-deps`，**不会覆盖服务器的 CUDA torch**；
2. 给 `third_party/dinov3` 打补丁（ImageFolder、strict=False、softmax centering、
   截断图容错、logging 修复）；
3. 生成 `offline_configs/*.yaml`，自动填入官方权重路径和 data root；
4. 验证 `from dinov3.data.loaders import make_dataset`。

## 1. 数据审计、去重、删除坏图

先确认 `output/data-audit/image_audit.csv` 存在（旧服务器上已有）。没有就重跑：

```bash
cd ~/MRG-05-Tooth-Seg/lab_train
python pretrain/audit_pretrain_data.py \
  --data-root dataset \
  --eval-root ../dataset/positive \
  --output output/data-audit
```

建去重视图（每个 SHA256 保留一份），并删除两张截断 JPEG：

```bash
python pretrain/build_dedup_view.py \
  --data-root dataset \
  --audit output/data-audit/image_audit.csv \
  --out dataset/pretrain_dedup \
  --delete-corrupt
```

预期输出：

```text
unique_kept: 42616
corrupt: 2
deleted corrupt files: 2/2
```

- `--delete-corrupt` 只删除严格解码失败的 2 张 AlphaDent 图；
- 去重视图用硬链接，不占额外空间；
- 之后所有配置的 `train.dataset_path` 都指向 `dataset/pretrain_dedup`。

## 2. Stage 0：2,220 步诊断（约 25 分钟）

```bash
cd ~/MRG-05-Tooth-Seg/lab_train
CONFIG=$PWD/offline_configs/config_dental_vitb_224_diag.yaml \
OUTPUT=$PWD/output/official-dental-diag-224 \
bash run_official_dinov3.sh
```

启动前脚本会硬性检查 `centering: softmax`，否则拒绝启动。

监控：

```bash
tail -f output/official-dental-diag-224/nan_logs/logs/log.txt
nvidia-smi -l 2
```

**验收门槛**（不达标就停，不要进入 Stage 1）：

新配置 K=32768，均匀目标的交叉熵是 `ln(32768) = 10.397`。

| 指标 | 期望 |
| --- | --- |
| `dino_*_crops_loss` @ step ~100 | < 10.39（证明 softmax 路径生效） |
| `dino_*_crops_loss` @ step 2,220 | 比 step 100 至少低 0.2 nat，最好 < 10.2 |
| `ibot_loss` | 温度固定 0.07 后稳定 < 4.8 |
| 显存峰值 | 20–23 GB（batch 96 @ 224） |

如果 DINO loss 从第一步起就贴在 `~10.40`，说明 softmax 没生效或配置没被重新生成。

## 3. Stage 1：224/96 主训练（120 epochs，约 5.8 小时）

```bash
cd ~/MRG-05-Tooth-Seg/lab_train
CONFIG=$PWD/offline_configs/config_dental_vitb_224.yaml \
OUTPUT=$PWD/output/official-dental-vitb-224 \
bash run_official_dinov3.sh
```

- 42,616 张，`OFFICIAL_EPOCH_LENGTH = 134`（`ceil(42616/320)`）；
- batch 320，120 epochs = 16,080 steps；
- 峰值 LR = `0.000054 × 4 × sqrt(320/1024) = 1.21e-4`；
- 实测显存 21.6 GB、1.31 s/it（约 5.8 小时）；
- 每 500 步保存 DCP，`max_to_keep: 10`，另保留每 5k 步的 checkpoint。

## 4. Stage 2：512/128 高分辨率适应（约 30–45 分钟）

Stage 1 正常跑完后，**用同一个 OUTPUT 续训**：

```bash
cd ~/MRG-05-Tooth-Seg/lab_train
CONFIG=$PWD/offline_configs/config_dental_vitb_512.yaml \
OUTPUT=$PWD/output/official-dental-vitb-224 \
RESUME=1 \
bash run_official_dinov3.sh
```

- 只改 `crops`、`batch_size_per_gpu`、`optim.epochs`；
- `train.OFFICIAL_EPOCH_LENGTH` 保持 134（它是 schedule 单位，不是每 epoch 图数）；
- `epochs=144` → 总 19,296 步，Stage 1 结束在 16,080，剩 3,216 步；
- 起始 LR ≈ 1.3e-5，衰减到 1e-6，属于高分辨率适应的低 LR 区间；
- 数据 loader 会继续用同一个 output 的 checkpoint，不会重新初始化。

> 如果 Stage 1 改过 batch / epoch length，Stage 2 必须同步：
> `Stage2.OFFICIAL_EPOCH_LENGTH = Stage1.OFFICIAL_EPOCH_LENGTH`，
> `Stage2.epochs = Stage1.epochs × 1.2`。
> 否则 resume 的 iteration 可能超过 Stage 2 的 `max_iter`，训练会直接结束或
> LR 停留在 1e-6。

> 不要用 `--no-resume` 跑 Stage 2；`RESUME=1` 会去掉该参数。

## 5. 显存与 batch 对照

以下是服务器上 50 步探针的**实测值**（224/96，6 local crops，32768 prototypes，Gram off）：

| 分辨率 / local | batch | `max mem` | 稳态 `time/it` | 备注 |
| --- | ---: | ---: | ---: | --- |
| 224 / 96 | 96 | 7.9 GB | 0.64 s | 诊断用 |
| 224 / 96 | 192 | 13.8 GB | 0.79 s | OOM 时 |
| 224 / 96 | 256 | 17.7 GB | 1.06 s | OOM 时 |
| **224 / 96** | **320** | **21.6 GB** | **1.31 s** | **默认，约 86%** |
| 224 / 96 | 352 | ~23.5 GB | — | 风险偏高 |
| 512 / 128 | 24 | 未实测 | — | Stage 2 默认 |

线性拟合约为 `max_mem(MB) ≈ 2070 + 61 × batch`。192 以后吞吐稳定在约 243 img/s，
再增大 batch 不会更快，只会减少优化步数。

OOM 时按 `batch` 回退，同时改 `OFFICIAL_EPOCH_LENGTH` 和 `optim.lr`：

| batch | `OFFICIAL_EPOCH_LENGTH` | `optim.lr`（峰值 ≈1.2e-4） |
| ---: | ---: | ---: |
| 320 | 134 | 0.000054 |
| 256 | 167 | 0.000060 |
| 224 | 191 | 0.000064 |
| 192 | 222 | 0.000070 |

例如回退到 256：

```bash
CONFIG=$PWD/offline_configs/config_dental_vitb_224.yaml \
OUTPUT=$PWD/output/official-dental-vitb-224-b256 \
bash run_official_dinov3.sh \
  train.batch_size_per_gpu=256 \
  train.OFFICIAL_EPOCH_LENGTH=167 \
  optim.lr=0.00006
```

> Stage 2 的 `OFFICIAL_EPOCH_LENGTH` 故意不改成 `ceil(42616/24)=1776`：
> 它必须与 Stage 1 相同，才能让 cosine 尾段落在正确的 iteration 上。

## 6. 监控与产物

日志：

```text
output/official-dental-vitb-224/nan_logs/logs/log.txt   # 官方文本日志
output/official-dental-vitb-224/training_metrics.json   # JSONL，每 10 步一行
output/official-dental-vitb-224/stdout.log              # torchrun 控制台
output/official-dental-vitb-224/ckpt/<iter>/            # DCP checkpoint
```

快速看 loss：

```bash
grep "Training" output/official-dental-vitb-224/nan_logs/logs/log.txt | tail -3
```

## 7. 常见启动错误

- `ValueError: ... gram.it_load_ema_teacher must be non-negative`
  → 配置已加 `gram.it_load_ema_teacher: 0`、`gram.rep_update: false`；
  **重新运行 `setup_offline_dinov3.sh`**。
- `ImageFolder:root=./dataset` 被解析到 `third_party/dinov3/dataset`
  → `run_official_dinov3.sh` 从 `lab_train/` 启动；setup 写入绝对路径。
- `TypeError: must be str, not DictConfig`
  → setup 的 patch 会修 logging formatter；重新运行 setup。
- `AssertionError` in `build_schedulers`（smoke）
  → `SMOKE=1` 会把总步数压到 100；脚本同时覆盖 warmup/freeze 为 0。
- `OSError: image file is truncated`
  → setup 会在 `decoders.py` 开启 `LOAD_TRUNCATED_IMAGES`；ImageFolder 也会跳过坏图。
  仍建议用 `build_dedup_view.py --delete-corrupt` 清掉源文件。
- `FileNotFoundError: '/PATH/TO/weights/dinov3-vitb16-official.pth'`
  → `offline_configs/` 里的配置是从 patch 目录手工拷贝的，占位符没替换。
  重跑 `setup_offline_dinov3.sh`；新版 `run_official_dinov3.sh` 也会检测
  `dataset_path` / `resume_from_teacher_chkpt` 中的 `/PATH/TO/` 并自动重建。
- DINO loss 停在 ~10.40（K=32768 的均匀基线）
  → `centering` 不是 softmax，或 `offline_configs` 是旧的。重新跑 setup，并用
  `grep centering offline_configs/*.yaml` 确认。
- 启动被 `ERROR: ... does not set train.centering: softmax` 拒绝
  → 这是有意的保护；确认配置正确后再启动。只有明确知道 batch 很大时才用
  `ALLOW_SINKHORN=1`。
