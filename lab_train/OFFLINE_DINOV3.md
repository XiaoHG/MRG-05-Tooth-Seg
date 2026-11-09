# 离线 lab_train：官方 DINOv3 牙科预训练

服务器不能联网；所有训练所需内容已经放进 `lab_train/`：

```text
lab_train/
  third_party/dinov3/             # 官方 DINOv3 代码（无需 clone/联网）
  offline_wheels/                 # Linux x86_64 / Python 3.11 离线 wheels
  requirements-offline.txt
  weights/dinov3-vitb16-official.pth  # 已转换的官方格式权重（327 MB）
  pretrain/official_dinov3_patch/     # ImageFolder/strict=False 补丁 + 224/512 配置
  pretrain/audit_pretrain_data.py     # 数据审计
  setup_offline_dinov3.sh             # 离线安装 + 打补丁 + 生成配置
  run_official_dinov3.sh              # 启动训练
```

> 不要复制 `lab_train/dataset/` 和 `lab_train/output/` 到服务器：
> 服务器已有 42,616 张数据和对应的 `dataset/pretrain/pretrain_manifest.json`；
> 本地 `dataset/pretrain/pretrain_manifest.json` 是 0 张图的占位文件，覆盖会破坏服务器清单。
> 本地 `output/` 是旧的 224/512 实验产物（约 5.5 GB），也不需要复制。

## 1. 推荐复制方式

在 Mac 上执行：

```bash
rsync -av --exclude 'dataset/' --exclude 'output/' \
  lab_train/ cs_cs_qjh@ecs-instance-fd926d86:~/MRG-05-Tooth-Seg/lab_train/
```

这样会同步：

- 代码、`third_party/dinov3`、`offline_wheels`、`weights`（含新的官方权重）；
- 不会碰服务器已有的 `lab_train/dataset` 和 `lab_train/output`。

如果你坚持 `cp -r` 整个目录，**先备份服务器清单**：

```bash
ssh cs_cs_qjh@ecs-instance-fd926d86 \
  'cp ~/MRG-05-Tooth-Seg/lab_train/dataset/pretrain/pretrain_manifest.json \
       ~/MRG-05-Tooth-Seg/lab_train/dataset/pretrain/pretrain_manifest.json.bak'
```

然后复制时手动排除 `dataset/`、`output/`。

## 2. 服务器安装

```bash
ssh cs_cs_qjh@ecs-instance-fd926d86
cd ~/MRG-05-Tooth-Seg/lab_train

conda activate toothseg
python -V
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

bash setup_offline_dinov3.sh --data-root ~/MRG-05-Tooth-Seg/lab_train/dataset
```

`setup_offline_dinov3.sh` 会：

1. 离线安装 `omegaconf/ftfy/regex/torchmetrics/submitit/termcolor/scikit-learn/...`；
   `--no-deps`，**不会覆盖服务器的 CUDA torch**；
2. 给 `third_party/dinov3` 打 ImageFolder + backbone-only resume 补丁；
3. 生成 `offline_configs/*.yaml`，自动填入官方权重路径和 data root；
4. 验证 `from dinov3.data.loaders import make_dataset`。

## 3. 数据审计

```bash
python pretrain/audit_pretrain_data.py \
  --data-root dataset \
  --eval-root ../dataset/positive \
  --output output/data-audit
```

确认：

- 约 42,616 张，来源 AlphaDent / COde / OMNI；
- 无 X 光（你说当前没有）；
- 与下游评测集无哈希重叠（你说 test 不在 42k）；
- 无损坏/重复。

## 4. 100 步 smoke

```bash
SMOKE=1 bash run_official_dinov3.sh
```

检查：

```bash
nvidia-smi
tail -f output/official-dental-vitb-512-smoke/log.txt
```

- 显存 < 25 GB；
- DINO / iBOT / KoLeo / Gram loss 有限；
- steps/s 可接受。

OOM 时编辑 `offline_configs/config_dental_vitb_512.yaml`：

```yaml
train:
  batch_size_per_gpu: 4
  OFFICIAL_EPOCH_LENGTH: 10654   # ceil(42616/4)
crops:
  local_crops_number: 4
```

或先跑 224：

```bash
CONFIG=$PWD/offline_configs/config_dental_vitb_224.yaml bash run_official_dinov3.sh
```

## 5. 正式训练

```bash
bash run_official_dinov3.sh
```

默认：

```text
config = offline_configs/config_dental_vitb_512.yaml
output = output/official-dental-vitb-512
nproc  = 1
```

多卡：

```bash
NPROC=4 bash run_official_dinov3.sh
# 同时把 config 的 OFFICIAL_EPOCH_LENGTH 改为 ceil(42616 / (8 * 4)) = 1332
```

`run_official_dinov3.sh` 默认 `--no-resume`；续训时编辑脚本去掉该参数，或直接跑官方命令。

## 6. 关键参数

| 项 | 512 | 224 |
| --- | ---: | ---: |
| global / local | 512 / 128 | 224 / 96 |
| batch/GPU | 8 | 16 |
| `OFFICIAL_EPOCH_LENGTH` | 5327 | 2664 |
| epochs | 15 | 15 |
| KoLeo | 0.05 | 0.05 |
| iBOT | 1.0 | 1.0 |
| Gram | on（EMA teacher） | on（EMA teacher） |
| activation checkpointing | on | on |

- KoLeo 官方 0.1；单卡 batch 8 用 0.05，避免 P0 的近正交特征。
- 最佳 checkpoint 可能早于 final；每 1k–2k 迭代看下游 probe。
- 官方训练保存 DCP checkpoint（`output/.../ckpt/<iter>/`）；导出成 HF backbone 后才能用
  `eval_checkpoints.py` 做线性 probe。

## 7. 已验证

- HF → 官方权重：175 个 ViT 张量，CLS 余弦 **0.9999998**；
- 训练 checkpoint 176 个张量（含 `backbone.*` 和 rope buffer）；
- ImageFolder + strict=False 补丁已在官方 repo 上实际打过并通过语法检查；
- 数据审计脚本在 141 张氟斑牙上实测通过。
