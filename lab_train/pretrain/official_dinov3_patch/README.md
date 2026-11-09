# 官方 DINOv3 训练适配（iBOT + KoLeo + Gram）

本目录把官方 `facebookresearch/dinov3` 训练代码适配到牙科 42,616 张口内图，解决三件事：

1. 官方代码只内置 ImageNet/ADE20K/COCO/NYU，**没有通用 ImageFolder**；
2. 官方训练需要 **官方格式** 的 teacher checkpoint，而现有权重是 HF/ModelScope 格式；
3. 单卡 25 GB 需要 activation checkpointing、较小 batch 和合理的学习率。

本地已验证：

- HF → 官方权重转换：175 个 ViT 张量，缺失仅 `rope_embed.periods`（计算 buffer），
  CLS 特征余弦 **0.9999998**；
- 转换后的训练 checkpoint 含 `backbone.*` 前缀和 rope buffer，共 176 个张量；
- ImageFolder patch 与 `strict=False` 补丁语法检查通过。

---

## 1. 克隆并打补丁

```bash
git clone --depth 1 https://github.com/facebookresearch/dinov3.git /opt/dinov3
cd /opt/dinov3
pip install -r requirements.txt
pip install -e .
```

打补丁（二选一）：

```bash
# 在项目根目录执行
python lab_train/pretrain/official_dinov3_patch/patch_official_dinov3.py --repo /opt/dinov3
```

补丁做三件事：

- 复制 `image_folder.py` → `dinov3/data/datasets/image_folder.py`；
- 在 `datasets/__init__.py` 注册 `ImageFolder`；
- 在 `loaders.py` 增加 `ImageFolder` 数据字符串分支；
- 把 `init_fsdp_model_from_checkpoint` 的 `load_state_dict` 改为 `strict=False`，
  允许只加载 backbone、heads/centers 保持新初始化。

验证：

```bash
grep -n ImageFolder /opt/dinov3/dinov3/data/loaders.py
grep -n strict=False /opt/dinov3/dinov3/checkpointer/checkpointer.py
```

---

## 2. 数据审计与准备

42,616 张来自 AlphaDent + COde-Dataset + OMNI_COCO。

```bash
cd ~/MRG-05-Tooth-Seg/lab_train

python pretrain/audit_pretrain_data.py \
  --data-root dataset \
  --eval-root ../dataset/positive \
  --output output/data-audit
```

重点看：

- `summary.md` 的来源分布；
- **灰度/低饱和（疑似 X 光）比例**：下游氟斑牙/疾病是 RGB 口内图，建议先只用 RGB；
- `eval_overlap`：氟斑牙 141 张、公开 720 张、下游 test 不能进预训练；
- 重复文件（SHA256）、零字节、解码失败。

官方训练数据字符串：

```text
ImageFolder:root=/PATH/TO/dental_pretrain
```

该目录可以是嵌套结构，类别标签会被忽略；建议只放 **RGB 口内图**，并排除下游 test。

---

## 3. 权重转换（HF → 官方）

在服务器上执行（需要官方 repo 以生成 rope buffer）：

```bash
cd ~/MRG-05-Tooth-Seg/lab_train

python pretrain/convert_hf_dinov3_to_official.py \
  --hf-model weights/dinov3-vitb16 \
  --out weights/dinov3-vitb16-official.pth \
  --official-repo /opt/dinov3 \
  --validate-image ../dataset/positive/mild/mild_1.png \
  --img-size 224
```

期望输出：

```text
load_state_dict: missing=1 unexpected=0
missing (first 10): ['rope_embed.periods']
Saved training checkpoint: .../dinov3-vitb16-official.pth (176 tensors, backbone.* prefix)
CLS cosine=0.99999982  max_abs_diff=0.00104666
```

> `rope_embed.periods` 是计算 buffer，缺失不影响；保存的训练 checkpoint 已经包含它。

---

## 4. 修改配置路径

编辑：

```text
lab_train/pretrain/official_dinov3_patch/config_dental_vitb_512.yaml
```

把两处 `/PATH/TO/...` 换成真实路径：

```yaml
train:
  dataset_path: ImageFolder:root=/data/dental_pretrain

student:
  resume_from_teacher_chkpt: /data/weights/dinov3-vitb16-official.pth
```

其他关键参数：

| 参数 | 512 配置 | 224 配置 |
| --- | ---: | ---: |
| global / local | 512 / 128 | 224 / 96 |
| batch/GPU | 8 | 16 |
| `OFFICIAL_EPOCH_LENGTH` | 5327 | 2664 |
| `optim.epochs` | 15 | 15 |
| `optim.lr`（缩放前） | 3e-4 | 3e-4 |
| KoLeo | 0.05 | 0.05 |
| iBOT | 1.0 | 1.0 |
| Gram | on（EMA teacher） | on（EMA teacher） |
| activation checkpointing | on | on |

`OFFICIAL_EPOCH_LENGTH` 公式：

```text
ceil(42616 / (batch_size_per_gpu × world_size))
```

如果使用 4 卡、每卡 batch 8，则 `OFFICIAL_EPOCH_LENGTH = ceil(42616 / 32) = 1332`。

---

## 5. 启动训练

单卡 25 GB：

```bash
cd /opt/dinov3

torchrun --nproc_per_node=1 dinov3/train/train.py \
  --config-file /path/to/config_dental_vitb_512.yaml \
  --output-dir /data/output/official-dental-vitb-512 \
  --no-resume
```

多卡（推荐，官方 recipe 原本就是大 batch）：

```bash
torchrun --nproc_per_node=4 dinov3/train/train.py \
  --config-file /path/to/config_dental_vitb_512.yaml \
  --output-dir /data/output/official-dental-vitb-512 \
  --no-resume
```

如果 512 + iBOT/Gram 在 25 GB 下 OOM：

1. `train.batch_size_per_gpu: 8 → 4`（同时把 `OFFICIAL_EPOCH_LENGTH` 改为 10654）；
2. `crops.local_crops_number: 6 → 4`；
3. 先跑 `config_dental_vitb_224.yaml` 验证 recipe，再上 512；
4. 确认 `train.checkpointing: true` 已开启；
5. 如果仍 OOM，使用多卡或 ViT-S/16。

---

## 6. 训练监控与 checkpoint

官方训练输出为 DCP checkpoint（`output_dir/ckpt/<iter>/`）。建议同时监控：

- DINO / iBOT / KoLeo / Gram 各项 loss；
- 显存峰值、steps/s；
- 每 1k–2k 迭代保存的 checkpoint；
- 下游 probe（需要先把官方 DCP 导出成 HF 格式，见下节）。

**不要只看总 loss。** 之前 P0 的教训是 loss 下降、下游下降。  
至少监控：pair cosine（不要低于 0.1）、participation ratio、teacher entropy/used dims、颜色/灰度稳定性。

---

## 7. 下游评估（LoRA / 全量 / 检测）

训练完成后：

1. 导出官方 teacher backbone 为 HF 格式（下一步实现 `export_official_teacher.py`）；
2. 用 `eval_checkpoints.py` 做线性 probe（快速诊断）；
3. 用 LoRA / 全量微调在氟斑牙 5 级、口腔疾病 6 类上对比 base vs OralDINO；
4. 检测任务 A3：DINOv3 + ViT-Adapter/FPN + DETR，base vs OralDINO，frozen/LoRA/full；
5. 固定 held-out test、多种子、同协议比较。

> 当前 `eval_checkpoints.py` 读取的是我们自己的 `{"backbone": ...}` checkpoint；
> 官方 DCP 需要先导出，不能直接喂。

---

## 8. 注意事项

- 官方 recipe 假设 **大 batch + 多卡 + 长 schedule**；单卡 25 GB 是妥协版。batch 8 下 KoLeo 用 0.05，不要用 0.1。
- 42k 对 ViT-B 仍然偏小；建议先跑 224 配置验证，再上 512；最佳 checkpoint 可能早于 final，必须用下游 probe 早停。
- COde 若含 X 光片，先只保留 RGB 口内图；不要与下游 test 混在一起。
- 下游如果是氟斑牙/疾病分类，预训练数据要包含对应的纹理/病灶图像；纯检测图不够。
- 如果官方 recipe + LoRA/检测仍不如 base，说明 42k 数据/当前路线不足，应转向更多样数据或结构先验/检测模块，而不是继续堆同质数据。
