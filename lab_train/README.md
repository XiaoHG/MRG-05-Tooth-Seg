# lab_train 服务器训练包

本目录是一个**自包含的单类别牙齿检测训练包**，可直接整体拷贝到目标服务器，按本文件完成一次环境安装后即可开始训练。当前项目是单类别 `tooth`（类别 ID 0）的 YOLO 格式边界框检测，不是像素级分割，也不输出 FDI 牙位编号或氟斑牙临床分级。

## 本目录内容

```text
lab_train/
  pyproject.toml              # 项目依赖定义（pip install -e ".[train]" / ".[pretrain]"）
  cli/main.py                 # 训练 / 验证 / 推理统一入口
  src/toothseg/               # 核心包（数据、模型目录、pipeline、各后端适配器）
  weights/                    # 预训练权重（yolo11n/s、yolo26n/m、yolov8m）
  pretrain/                   # DINOv3 域内预训练（数据准备 + 训练脚本 + 配置）
  dataset/                    # 多源公开数据集（按数据集命名，直接放口内图像）
    AlphaDent/
    COde-Dataset/
    DentalAI/
    Images_Data_Set/
    Mendeley-dataset-19.4.2024/
    OMNI_COCO/
    Oral-Diseases/
    segmentation-teeth-images-and-masks/
    pretrain/                 # 预训练去重清单（脚本生成，不放图像）
```

数据组织：`dataset/` 下除 `pretrain/` 外，每个目录以一个公开数据集命名，直接把该数据集的口内图像放到对应目录（可平铺或子目录嵌套，脚本会递归扫描）。预训练数据准备脚本会自动发现这些目录、跨目录 SHA256 去重后生成 `dataset/pretrain/pretrain_manifest.json`，训练时据此加载图像。检测训练（v13 及以前）所用的 `before/after` 标签数据在根项目 `dataset/OMNI_COCO/` 下，不属于本服务器预训练包。

## 目标服务器环境（已确认）

| 项 | 值 |
| --- | --- |
| GPU | 原目标：NVIDIA RTX PRO 6000 Blackwell，约 96 GB；当前实际训练环境可用显存约 25 GB（配置见 3.4） |
| 驱动 | 595.84，CUDA 版本 13.2（驱动支持的最高 CUDA 运行时） |
| CPU | 2× Intel Xeon Gold 6530，128 线程 |
| 内存 | 约 1 TB |
| 系统 | x86_64 Linux（容器环境） |
| 现状 | 仅装了驱动，`python` / `conda` / `pip` / `nvcc` 均未安装 |

> **关键点**：RTX PRO 6000 属于 Blackwell 架构（计算能力 `sm_120`），PyTorch 必须使用 **CUDA 12.8+ 的官方 wheel（`cu128` 或 `cu130`）**，且 PyTorch ≥ 2.7 才原生支持 sm_120。旧 wheel（`cu124`、`cu126` 等）缺 sm_120 内核，会报 `no kernel image is available for execution on the device`。驱动为 595.84 / CUDA 13.2，属于 R580+ 分支，可直接使用 `cu130` wheel。

---

## 一、环境安装（首次一次性）

**推荐：一键脚本**（自动完成 Miniconda → conda 环境 → PyTorch → 项目依赖 → GPU 验证）：

```bash
cd lab_train
bash setup_env.sh                 # 默认官方源（快），只装 .[train]
bash setup_env.sh --mirror cn     # 启用国内镜像（清华 TUNA + 阿里云 PyTorch）
bash setup_env.sh --mirror tuna   # 仅清华 TUNA（conda/pip/miniconda），PyTorch 官方
bash setup_env.sh --mirror aliyun # 仅阿里云（pip + PyTorch wheels）
bash setup_env.sh --external      # 额外装 RT-DETRv2-L / RF-DETR-M 后端
```

镜像默认关闭（`--mirror none`，走官方源）；只有明确传入 `--mirror` 才切换镜像。`cn` = 全开国内镜像（Miniconda/conda/pip 走清华 TUNA，PyTorch wheel 走阿里云 `mirrors.aliyun.com/pytorch-wheels`），镜像缺 `cu130` 时自动回退官方 `download.pytorch.org`。可通过环境变量覆盖：`CUDA=cu128 bash setup_env.sh`（回退较旧稳定 wheel）、`ENV_NAME=myenv bash setup_env.sh`、`MIRROR=cn bash setup_env.sh`。

以下为手动安装步骤（脚本内部等价逻辑，供排查用）。

### 1. 安装 Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
# 重新登录 shell，或执行： source ~/.bashrc
```

### 2. 创建并激活环境

```bash
conda create -n toothseg python=3.11 -y
conda activate toothseg
```

### 3. 安装 PyTorch（Blackwell / sm_120，CUDA 12.8+）

优先使用 `cu130`（驱动 595.84 支持）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

若 `cu130` 索引不可用或需锁定较早稳定版，可回退到 `cu128`：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 4. 安装项目训练依赖

在 `lab_train` 目录内执行：

```bash
cd lab_train
pip install -e ".[train]"
```

`.[train]` 会安装 `ultralytics`、`pyyaml`、`numpy`、`pillow` 等训练/验证依赖（`torch`/`torchvision` 已在上一步装好，不会被降级）。

如需 RT-DETRv2-L 或 RF-DETR-M 后端，额外安装：

```bash
pip install -e ".[external]"
# rfdetr 若未随上一步自动装好，可单独：
pip install rfdetr
```

### 5. 验证 GPU 可用

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('capability', torch.cuda.get_device_capability()); print('arch_list', torch.cuda.get_arch_list())"
```

期望输出：版本字符串带 `+cu130`（或 `+cu128`）后缀、`cuda_available=True`、`capability=(12, 0)`、`arch_list` 中包含 `sm_120`。若版本字符串没有 `+cu1xx` 后缀，说明装成了 CPU 版或错误 CUDA 版，需重装。

---

## 二、查看可用模型

```bash
python cli/main.py models
```

列出 8 个正式对比模型及其后端就绪状态：Ultralytics（`yolo11n.pt`、`yolo11s.pt`、`yolov8m.pt`、`yolo26m.pt`、`rtdetr-l.pt`）、torchvision（`fasterrcnn_resnet50_fpn`、`retinanet_resnet50_fpn`）、Transformers（`rtdetrv2-l`）。RF-DETR-M（`rf-detr-m`）仍可训练，但其原生 `validate` 不接受 `--split`，未纳入固定 test 集正式对比。

---

## 三、训练

统一入口为 `python cli/main.py train`，`--data` 传入数据版本根目录，CLI 会自动在实验目录下生成运行时 `dataset.yaml`（不依赖数据集内已有的 yaml）。

### 冒烟测试（每个新模型先跑 1 epoch）

```bash
python cli/main.py train \
  --data dataset/OMNI_COCO/after \
  --model yolo11n.pt \
  --epochs 1 --imgsz 1024 --batch 1 --device 0 \
  --project output/smoke --name smoke-yolo11n
```

冒烟测试通过后再上正式训练。

### 正式训练

Ultralytics 系列（`--batch -1` 由 Ultralytics 自动选择；96 GB 显存通常可选较大 batch）：

```bash
python cli/main.py train \
  --data dataset/OMNI_COCO/after \
  --model yolo11n.pt \
  --epochs 50 --imgsz 1024 --batch -1 --device 0 \
  --project output/train --name omni-coco-after-yolo11n
```

替换 `--model` 与 `--name` 即可训练其它模型：`yolo11s.pt`、`yolov8m.pt`、`yolo26m.pt`、`rtdetr-l.pt`。

torchvision 后端（Faster R-CNN / RetinaNet，检测头以 batch 1 训练，`--batch` 建议固定为 2）：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model fasterrcnn_resnet50_fpn --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-fasterrcnn

python cli/main.py train --data dataset/OMNI_COCO/after --model retinanet_resnet50_fpn --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-retinanet
```

Transformers / RF-DETR 原生后端：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model rtdetrv2-l --epochs 50 --imgsz 1024 --batch 1 --device 0 --project output/train --name omni-coco-after-rtdetrv2l

python cli/main.py train --data dataset/OMNI_COCO/after --model rf-detr-m --epochs 50 --imgsz 1024 --batch 1 --device 0 --project output/train --name omni-coco-after-rfdetrm
```

### 训练参数说明

| 参数 | 说明 |
| --- | --- |
| `--data` | 数据版本根目录，如 `dataset/OMNI_COCO/after` 或 `.../before` |
| `--model` | 模型名，见上表 |
| `--weights` | 可选，用已有权重初始化；同时给 `--model` 时权重优先 |
| `--epochs` | 训练轮数 |
| `--imgsz` | 输入尺寸，默认 1024；显存不足改 640 |
| `--batch` | `-1` 自动；torchvision / 原生后端建议显式 1 或 2 |
| `--device` | 单卡服务器用 `0` |
| `--project` / `--name` | 输出目录 `output/<project>/<name>` |
| `--lr` | 可选初始学习率（RT-DETRv2 默认 1e-5） |

---

## 四、验证与测试评估

验证训练所用的 `val` 集：

```bash
python cli/main.py val \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --data dataset/OMNI_COCO/after
```

在固定、独立、带人工标签的 `test` 集上评估（模型确定后使用）：

```bash
python cli/main.py val \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --data dataset/OMNI_COCO/after \
  --split test
```

`--split` 支持 `train` / `val` / `test`，默认 `val`，透传到 Ultralytics、torchvision、RT-DETRv2-L 三个后端。注意 RF-DETR-M 原生 `validate` 读自身物化的 `valid` 视图、不接受 `--split`。

---

## 五、输出位置

每个实验在 `output/<project>/<name>/` 下至少保留：

```text
weights/best.pt   weights/last.pt
args.yaml         results.csv
.val 的验证图 / 训练日志
```

训练完成后应整体归档实验目录（含 `args.yaml`、`results.csv` 与权重），不要只保存 `best.pt`，以便保留数据版本、训练参数与验证结果。

---

## 六、注意事项

- **Blackwell 依赖**：必须使用 `cu128`/`cu130` 且 PyTorch ≥ 2.7 的 wheel，并确认 `arch_list` 含 `sm_120`；否则训练时会报 `no kernel image is available`。
- **网络**：`pip install` 与首次训练需联网；`weights/` 已内置 `yolo11n.pt`、`yolo11s.pt`、`yolo26n.pt`、`yolo26m.pt`、`yolov8m.pt`。`rtdetr-l.pt`、RT-DETRv2-L（HuggingFace 权重）、RF-DETR-M（官方 `rf-detr-medium.pth`）未内置，首次使用需联网下载或手动传入 `--weights`。
- **不覆盖已完成实验**：每次实验使用不同的 `--name`，避免覆盖既有结果。
- **数据版本一致性**：before/after 或不同模型的比较必须使用相同的图像、标签与评估协议；调参用 `val`，最终用 `--split test` 做一次独立评估，勿反复使用 test 指标。
- **磁盘**：当前 overlay 根分区剩余约 241 GB，训练输出较大时建议写到 `/mnt/dev-share-heli` 共享盘（286 TB）。
- **项目定位**：本包是科研检测基线，不是临床诊断系统。

---

## 七、DINOv3 域内预训练（v15 / 论文 A·A2）

A2 的目标是在多源口内自然图像上对 DINOv3 做**域内 continue pretraining**，得到「OralDINO」预训练骨干，供下游检测 / 氟斑牙分级复用。自监督预训练只用图像、不看标签，因此先把多源口内图像去重后喂给 DINO 训练。

本节按以下顺序展开：**① 版本选择 → ② 权重下载与校验 → ③ 环境/数据准备、冒烟与正式训练 → ④ 输入输出与产物 → ⑤ 安装依赖 → ⑥ 注意事项与排错**。

本目录内置了 `pretrain/` 包：

```text
pretrain/
  prepare_pretrain_data.py   # 扫描图像目录 → SHA256 去重 → pretrain_manifest.json
  audit_pretrain_data.py     # 预训练数据审计（来源/重复/损坏/灰度/评测泄漏）
  convert_hf_dinov3_to_official.py # HF DINOv3 → 官方 DINOv3 teacher checkpoint
  pretrain.py                # 单机 DINO 简化训练循环（P0/control 用）
  eval_checkpoints.py        # 对已保存 checkpoint 做冻特征线性 probe
  config.example.json        # 通用超参参考
  config.512-25g.json        # 25 GB 显存 + 512 分辨率基础配置
  config.512-25g-42k-p0.json # 42k + P0 配方（已停用，仅作失败对照）
  config.512-25g-42k-control.json # 42k + 保守对照配方
  official_dinov3_patch/     # 官方 DINOv3 (iBOT/KoLeo/Gram) 训练适配与配置
```

### 1. DINOv3 版本选择分析

DINOv3（Meta FAIR，[论文 arXiv:2508.10104](https://arxiv.org/abs/2508.10104) / [仓库 facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)）是一族自监督视觉骨干，**所有 ViT 变体统一 patch size 16、4 个 register token、RoPE 位置编码**，并额外加入 iBOT（掩码图像建模）+ KoLeo（正则）+ Gram anchoring 多阶段调度。注意：DINOv3 **没有 `vitg14`**——那是 DINOv2 的命名；DINOv3 的最大 ViT 是 ViT-7B/16。

Web 数据（LVD-1689M，1.689B Instagram 图）的 ViT 变体：

| HF 模型 id | 骨架 | 参数量 | 嵌入维 | FFN | 备注 |
| --- | --- | --- | --- | --- | --- |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | ViT-S/16 | 21M | 384 | MLP | 最小 |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | ViT-S+/16 | 29M | 384 | SwiGLU | — |
| **`facebook/dinov3-vitb16-pretrain-lvd1689m`** | **ViT-B/16** | **86M** | **768** | MLP | **本包默认** |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | ViT-L/16 | 300M | 1024 | MLP | 数据量大时升级 |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` | ViT-H+/16 | 840M | 1280 | SwiGLU | — |
| `facebook/dinov3-vit7b16-pretrain-lvd1689m` | ViT-7B/16 | 6716M | 4096 | SwiGLU | 蒸馏教师/超大规模 |

另有 ConvNeXt 系列（`-convnext-tiny/small/base/large-pretrain-lvd1689m`）和卫星影像系列（`-vitl16/vit7b16-pretrain-sat493m`，与本项目无关）；小模型由冻结的 ViT-7B 教师硬蒸馏得到。

**本项目选 ViT-B/16（86M / 768 维）作为 OralDINO 默认骨干，理由：**

1. **规模匹配**：下游牙齿检测基线（YOLO11s / RT-DETR-L 等）backbone 量级在几十 M 到百 M，ViT-B 的 86M 与之匹配，便于后续 `backbone` 直连检测头或 LoRA/冻结微调；ViT-S（21M）容量偏小，ViT-L（300M）对当前数据量偏重。
2. **数据量匹配**：当前唯一口内图约 6k（A1 清单），愿景约 7 万张。自监督预训练在少量域内数据上，B 比 L/7B 更不易欠拟合、训练更快、显存占用小（96 GB 卡可轻松以 `--batch 32` 甚至更大跑 B）。
3. **可升级路径**：当 COde / Oral-Diseases 等全部到位、唯一图逼近 7 万后再切 `vitl16`（300M）不迟；切换只需改 `--model`，脚本与流程不变。

### 2. 权重下载详细流程（门控）

DINOv3 权重在 HuggingFace 上**门控（gated）**，须「申请访问 → 审批 → 鉴权 → 下载」四步，不能直接 `pip`/`wget`。

**① 申请访问（一次性，可能需数天审批）**

1. 打开模型页：`https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`。
2. 登录 HF 账号，点击页面上的「Request access / Agree」表单：填写姓名、出生日期、国家、机构、职业（Student / Research Graduate / AI researcher 等），勾选接受 **DINOv3 License**（`https://ai.meta.com/resources/models-and-libraries/dinov3-license`）。
3. 等待邮件确认（通常数天内）。**未获批前下载会报 403 GatedRepoError。**

**② 本地鉴权**

```bash
pip install -U "huggingface_hub[cli]"
hf login                       # 交互式，粘贴带 read 权限的 token
# 或非交互（服务器/CI 推荐）：
export HF_TOKEN="hf_..."
```

> 注意：细粒度 token（fine-grained）默认无 gated 仓库读权限，需在 HF 后台显式勾选「Read access to contents of all public gated repos」；旧的 `HF_TOKEN` 环境变量会覆盖 `hf login` 的缓存，排查时先 `env | grep HF_TOKEN`。

**③ 下载权重（推荐离线物化，避免训练时反复联网）**

```bash
huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local-dir weights/dinov3-vitb16 \
  --resume-download
```

下载到本包 `weights/` 后，`transformers.AutoModel.from_pretrained` 可直接用本地路径加载（`pretrain.py` 的 `--model` 传 `weights/dinov3-vitb16` 即可，无需再联网）。

**④ 验证与排错**

```bash
hf whoami                                  # 确认登录身份
python -c "from transformers import AutoModel; m=AutoModel.from_pretrained('facebook/dinov3-vitb16-pretrain-lvd1689m', token='hf_...'); print(m.config.hidden_size)"
# 期望输出 hidden_size = 768
```

- **401**：token 缺失/过期/无效 → 重新 `hf login` 或更新 `HF_TOKEN`。
- **403 / GatedRepoError**：未接受 license 或 token 无 gated 读权限 → 回步骤①。
- **地区限制 / 申请被拒**：DINOv3 License 受贸易管制条款约束；`HF_ENDPOINT` 镜像只改变下载端点，**不会绕过 gated 审批**。若官方申请被拒，优先用下面的 ModelScope 镜像（HF Transformers 格式，仓库内附 `LICENSE.md`）。

**⑤ 备选：ModelScope 镜像（HF 门控被拒时）**

ModelScope 上的 `facebook/dinov3-vitb16-pretrain-lvd1689m` 与官方 HF 仓库同为 **Transformers 格式**（`architectures: ["DINOv3ViTModel"]`），可直接被 `AutoModel.from_pretrained` 加载。推荐用 `uv` 隔离环境下载，不污染服务器环境：

```bash
uv run --no-project --with modelscope python - <<'PY'
from modelscope import snapshot_download
snapshot_download(
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    local_dir="weights/dinov3-vitb16",
)
PY
```

或使用 ModelScope CLI：

```bash
modelscope download \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --local_dir weights/dinov3-vitb16
```

下载完成后目录应包含：

```text
weights/dinov3-vitb16/
  config.json                # architectures: DINOv3ViTModel, hidden_size: 768
  model.safetensors          # ≈327 MB
  preprocessor_config.json
  LICENSE.md                 # DINOv3 License
```

**⑥ 加载校验（必须通过后再训练）**

```bash
python - <<'PY'
import torch
from transformers import AutoModel
m = AutoModel.from_pretrained("weights/dinov3-vitb16")
print(type(m).__name__, m.config.hidden_size)
m.eval()
with torch.no_grad():
    out = m(pixel_values=torch.randn(1, 3, 224, 224))
print("forward:", out.last_hidden_state.shape)  # [1, 201, 768]
PY
```

期望输出：

```text
DINOv3ViTModel 768
forward: torch.Size([1, 201, 768])
```

> `201 = 1 CLS + 4 register tokens + 196 patches`；若 `hidden_size` 不是 768 或模型类型不是 `DINOv3ViTModel`，说明下载的不是 ViT-B/16，或 `transformers` 版本过旧（需 `>=4.56`）。
>
> 注意：`timm/vit_base_patch16_dinov3.lvd1689m` 虽然也是非门控的，但它是 **timm 格式**（`architecture: vit_base_patch16_dinov3`），不能直接传给 `transformers.AutoModel`；本项目 `pretrain.py` 用的是 Transformers 格式，请优先用上面的 ModelScope 镜像或官方 HF 文件。

### 3. 架构安排与训练流程

**当前 `pretrain.py` 实现的是「精简 DINO 参考实现」**，训练目标与 DINOv3 同源但未含其完整多阶段调度：

```text
原始图像
  ├─ 2 个全局裁剪（global crop，scale 0.4~1.0）
  └─ N 个局部裁剪（local crop，scale 0.05~0.4）   # --local-crops 默认 6
        │
        ▼
   student（ViT backbone + DINOHead） ──┐
        │                              │ EMA（动量 0.996→1.0 余弦）
   teacher（ViT backbone + DINOHead） ◄─┘   停止梯度
        │
        ▼
DINO 交叉熵损失：teacher 输出经「中心化 + 温度锐化」作为软标签，
student 输出做 log-softmax 对齐（跳过同一视图的自身对比）
        │
        ▼
AdamW + 余弦 LR（线性 warmup）+ AMP → 更新 student → EMA 回灌 teacher
```

- `DINOHead`：`Linear(768→2048)→GELU→Linear(2048→256)` + l2 归一化 + weight-norm `Linear(256→out_dim=65536)`，是标准 DINO 投影头。
- **与官方 DINOv3 的差异**：本实现**未含** iBOT（掩码预测）、KoLeo 正则、Gram anchoring 与高分辨率/多阶段调度，属「够用的 continue-pretraining 脚手架」。要复现论文级预训练，应改用官方仓库在 `resume_from_teacher_chkpt` 指到本项目的 OralDINO 断点、或在官方 `facebookresearch/dinov3` 上直接以口内图像跑 continue pretraining（该仓库同样门控、要求 `transformers>=4.56.0`、`timm>=1.0.20`、PyTorch>=2.7.1）。

#### 3.1 环境准备

服务器（RTX PRO 6000 Blackwell / cu130）：

```bash
cd lab_train
bash setup_env.sh --pretrain
conda activate toothseg

python -c "import torch, torchvision, transformers; \
print(torch.__version__, torchvision.__version__, transformers.__version__); \
print('cuda:', torch.cuda.is_available())"
```

要求：

- `torch >= 2.7.1`（Blackwell `sm_120` 需 cu130/cu128 wheel）
- `torchvision >= 0.20`
- `transformers >= 4.56`（DINOv3 的 `dinov3_vit` 架构支持；本包 `pyproject.toml` 已固定该下限）
- 可选 `timm >= 1.0.20`（仅在改用 timm 格式权重时需要）

本地无 GPU 的流程验证（macOS / CPU）：

```bash
pip install -e ".[pretrain]"

# 或用 uv 隔离环境，避免污染现有环境：
uv run --no-project --with torch --with torchvision --with transformers \
  --with pillow --with numpy python pretrain/pretrain.py ...
```

> CPU 只适合跑 1–3 步冒烟；正式训练必须在 CUDA 上执行。

#### 3.2 数据准备（服务器上的真实布局）

`prepare_pretrain_data.py` 以 `lab_train/` 为根目录。把各数据集的图像直接放到对应目录（一个数据集一个目录，递归扫描，子目录无所谓）：

```text
lab_train/dataset/
  OMNI_COCO/              # 直接放图像
  AlphaDent/
  COde-Dataset/
  Oral-Diseases/
  ...
  pretrain/               # 只放生成的清单，不放图像
```

生成去重清单：

```bash
cd lab_train

# 默认：自动发现 dataset/ 下所有数据集目录（除 pretrain/）
python pretrain/prepare_pretrain_data.py --validate-images

# 或只扫指定数据集（调试 / 分批训练时）：
python pretrain/prepare_pretrain_data.py --validate-images \
  --images dataset/OMNI_COCO dataset/AlphaDent \
  --output dataset/pretrain/pretrain_manifest.json
```

> `--validate-images` 会完整解码每张图，把截断/损坏文件写入清单的 `corrupt` 字段并排除，避免训练时 DataLoader worker 因 `OSError: image file is truncated` 崩溃。扫描 5k–6k 张图通常只需几十秒到几分钟；不加该参数时只做 SHA256 去重，不检查图像完整性。

期望输出示例：

```text
Scanned 14191 files across 7 dir(s)
  - dataset/OMNI_COCO
  - dataset/AlphaDent
  ...
Unique images after SHA256 dedup: 5759
Manifest written to: dataset/pretrain/pretrain_manifest.json
```

若发现坏图，会额外打印 `Skipped corrupt images: N`。

检查清单：

```bash
python - <<'PY'
import json
m = json.load(open("dataset/pretrain/pretrain_manifest.json"))
print("base_dir:", m["base_dir"])
print("seen:", m["seen"], "unique:", m["unique"])
print("first:", m["images"][0]["path"])
PY
```

- `base_dir` 应为 `lab_train/` 的绝对路径；
- `images[].path` 应为 `dataset/<name>/...` 相对路径（可随整个 `lab_train/` 拷贝到服务器）；
- 若 `unique == 0`，检查图像是否真的放进了 `dataset/` 下；仓库里的 `pretrain_manifest.json` 是占位文件（0 张），**必须在服务器上重新生成**；
- 若 `corrupt` 非空，先检查这些文件；它们不会进入训练。即使不重新生成清单，`pretrain.py` 也会在运行时跳过无法解码的图并打印 `[warn] skip unreadable image`，但建议用 `--validate-images` 生成干净清单；
- `--images` 指向 `lab_train/` 外部时，清单会写绝对路径（可用，但不可移植）。

#### 3.3 冒烟测试（正式训练前必须做）

GPU 冒烟（20 步，验证模型加载、前向/反向、EMA、梯度累积、日志和 checkpoint 保存）：

```bash
cd lab_train

python -u pretrain/pretrain.py \
  --model weights/dinov3-vitb16 \
  --manifest dataset/pretrain/pretrain_manifest.json \
  --config pretrain/config.512-25g.json \
  --output output/pretrain/smoke-dinov3-vitb16-512 \
  --max-steps 20 \
  --batch 4 --grad-accum 2 \
  --local-crops 2 --local-imgsz 128 \
  --warmup-steps 5 --save-every 1000 --log-every 1 \
  --device cuda:0 --amp
```

期望日志：

```text
Pretraining on <N> unique images, micro-batch 4 x grad-accum 2 (effective 8), 20 steps, imgsz 512 / local 128, model weights/dinov3-vitb16
[step 0/20] loss=10.xxxx lr=... momentum=0.9960 epoch=0.xx gpu=xxxxMB
Finished. Final teacher checkpoint: output/pretrain/smoke-dinov3-vitb16-512/teacher_checkpoint.pt
Metrics CSV: output/pretrain/smoke-dinov3-vitb16-512/metrics.csv
Text log: output/pretrain/smoke-dinov3-vitb16-512/train.log
```

CPU 冒烟（无 GPU 时，3 步即可）：

```bash
python pretrain/pretrain.py \
  --model weights/dinov3-vitb16 \
  --manifest dataset/pretrain/pretrain_manifest.json \
  --output output/pretrain/smoke-dinov3-vitb16-cpu \
  --imgsz 128 --local-imgsz 64 --batch 2 --local-crops 1 \
  --max-steps 3 --warmup-steps 1 \
  --save-every 1000 --log-every 1 --num-workers 0 --device cpu
```

> `--device cuda:0` 在无 CUDA 的机器上会自动回退到 CPU；但 `--amp` 仅对 CUDA 有意义，CPU 冒烟不要加。`--num-workers 0` 用于 macOS 或 DataLoader 排查。

无论 GPU 还是 CPU 冒烟，都会自动生成 `metrics.csv` 和 `train.log`；不需要额外配置。

**checkpoint 校验（两种方式）**

只检查结构：

```bash
python - <<'PY'
import torch
ckpt = torch.load(
    "output/pretrain/smoke-dinov3-vitb16-512/teacher_checkpoint.pt",
    map_location="cpu", weights_only=False,
)
print("step:", ckpt["step"], "model:", ckpt.get("model"))
print("backbone tensors:", len(ckpt["backbone"]))
print("head tensors:", len(ckpt["head"]))
print("center:", tuple(ckpt["center"].shape))
PY
```

装回 `DINOv3ViTModel`（期望 `missing 0 / unexpected 0`）：

```bash
python - <<'PY'
import torch
from transformers import AutoModel
ckpt = torch.load(
    "output/pretrain/smoke-dinov3-vitb16-512/teacher_checkpoint.pt",
    map_location="cpu", weights_only=False,
)
m = AutoModel.from_pretrained("weights/dinov3-vitb16")
missing, unexpected = m.load_state_dict(ckpt["backbone"], strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
PY
```

#### 3.4 正式训练（25 GB 显存 + 512 分辨率推荐）

**核心思路**：全局 512、局部 128（**不要把 local crop 也设成 512**），用 `--batch` 控制单卡显存，用 `--grad-accum` 控制有效 batch。

有效 batch = `batch × grad_accum`；学习率按 DINO 常用缩放 `5e-4 × effective_batch / 256`。

**推荐配置（`pretrain/config.512-25g.json`）**

| 参数 | 值 | 说明 |
| --- | --- | --- |
| `imgsz` | 512 | 全局裁剪 512×512 |
| `local_imgsz` | 128 | 局部裁剪 128×128（关键：不能用 512，否则显存爆炸） |
| `local_crops` | 6 | 2 global + 6 local |
| `batch` | 8 | 单卡 micro-batch；25 GB 下先试 8 |
| `grad_accum` | 4 | 有效 batch = 32 |
| `lr` | 6.25e-5 | `5e-4 × 32 / 256` |
| `warmup_steps` | 500 | 约 3% 总步数 |
| `max_steps` | 17500 | 约 100 epochs（5598 张、有效 batch 32） |
| `out_dim` | 65536 | 基线；后续可做 16384 的消融 |
| `save_every` | 2000 | 约 8 个断点 ≈ 3.2 GB |
| `log_every` | 10 | 每 10 个优化步写一次日志 |
| `amp` | true | 必须开 |

**启动命令**：

```bash
cd lab_train

python -u pretrain/pretrain.py \
  --model weights/dinov3-vitb16 \
  --manifest dataset/pretrain/pretrain_manifest.json \
  --config pretrain/config.512-25g.json \
  --output output/pretrain/oraldino-vitb16-512 \
  --device cuda:0 \
  2>&1 | tee output/pretrain/oraldino-vitb16-512.stdout.log
```

- `--config` 提供默认值，CLI 仍可覆盖（例如 `--batch 4 --grad-accum 8`）。
- `python -u` 让 stdout 不缓冲；`tee` 额外保存完整控制台输出；脚本本身还会自动写 `metrics.csv` 和 `train.log`。
- 输出目录不要复用：每次正式训练用新的 `--output`，避免覆盖已有 checkpoint 和日志。

**显存不够时（保持有效 batch 32）**：

```bash
--batch 4 --grad-accum 8
```

如果仍然 OOM，按顺序调整：

1. `--local-crops 6 → 4`
2. `--local-imgsz 128 → 96`
3. 最后才降 `--imgsz 512 → 448`

**训练量怎么定**：

```text
steps_per_epoch = ceil(unique_images / effective_batch)
max_steps = target_epochs × steps_per_epoch
```

| unique | effective batch | steps/epoch | 50 epochs | 100 epochs | 200 epochs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5598 | 32 | 175 | 8,750 | 17,500 | 35,000 |
| 5598 | 64 | 88 | 4,400 | 8,750 | 17,500 |
| 4166 | 32 | 131 | 6,550 | 13,100 | 26,200 |

建议：

- 第一次正式跑先定 **100 epochs**（5598 张、有效 batch 32 → 17,500 步）；
- 在 5k / 10k / 17.5k 步各做一次下游线性评估或检测微调，如果还在提升，再延长到 200 epochs；
- 少于 50 epochs（< 8,750 步）只能作为流程验证，不建议写进论文。

**学习率与有效 batch 的关系**：

```text
lr ≈ 5e-4 × effective_batch / 256
```

| effective batch | 建议 lr |
| ---: | ---: |
| 16 | 3.1e-5 |
| 32 | 6.25e-5 |
| 64 | 1.25e-4 |

> 之前 `batch=16, lr=1.25e-4` 是有效 batch 16 对应 lr 的 4 倍；新配置 `batch 8 × grad_accum 4 = 32`，lr 用 6.25e-5。

**多种子实验**：

```bash
for seed in 0 1 2; do
  python -u pretrain/pretrain.py \
    --model weights/dinov3-vitb16 \
    --manifest dataset/pretrain/pretrain_manifest.json \
    --config pretrain/config.512-25g.json \
    --output output/pretrain/oraldino-vitb16-512-seed${seed} \
    --seed ${seed} \
    --device cuda:0 \
    2>&1 | tee output/pretrain/oraldino-vitb16-512-seed${seed}.stdout.log
done
```

**训练中检查**：

```bash
nvidia-smi
tail -f output/pretrain/oraldino-vitb16-512/train.log
head -5 output/pretrain/oraldino-vitb16-512/metrics.csv
```

日志字段：`step,epoch,loss,lr,momentum,elapsed_sec,steps_per_sec,samples_per_sec,gpu_mem_mb`。

每 `log_every` 个优化步写一次；step 0、每次保存 checkpoint、最后一次都会写。崩溃时已经写入的行不会丢。

> 如果 loss 长期不降或出现 NaN：先把 `--lr` 降一半，再降低 `--batch`/`--imgsz`，并确认 AMP 是否稳定。

**42,616 张数据：P0 配方（已停用）→ 官方 DINOv3 recipe（当前推荐）**

> **P0 42k 实测结果（2026-11-08）**：teacher 最佳 QWK **0.315**（10k），student 最佳 **0.381**（5k），公开 6 类 **0.613/0.574**；base DINOv3 为 **0.718/0.820**。P0 特征 pair cosine 仅 **0.03**，KoLeo 0.1 过强导致近正交/不稳定，**不要使用 P0 checkpoint**。
>
> 下一步改用官方 `facebookresearch/dinov3` 的 **iBOT + KoLeo + Gram** recipe，从 base continue-pretraining；数据审计、权重转换、ImageFolder 适配、224/512 配置见 [`official_dinov3_patch/README.md`](official_dinov3_patch/README.md)。下面的 P0 命令与训练量仅作为历史记录和消融参考。

- 数据：**42,616 张**（AlphaDent + COde-Dataset + OMNI_COCO），有效 batch 32（P0 历史配置）。
- 1 epoch = `42616 / 32 ≈ 1,332` 步。
- P0 历史计划：30 epochs ≈ 40,000 步；实测最佳在 5k–10k，说明早停比跑满更重要。

| 配置 | out_dim | temp_teacher | center_ema | KoLeo | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `config.512-25g-42k-p0.json` | 16384 | 0.05 | 0.99 | 0.1 | 历史失败，仅参考 |
| `config.512-25g-42k-control.json` | 65536 | 0.04 | 0.9 | 0 | 旧配方消融 |

P0 代码改动：

- 新增 `--koleo-weight`（默认 0）；P0 配置用 0.1，抑制码本/特征坍缩；
- center EMA 改为**每个优化步更新一次**（grad-accum 感知），不再每个 micro-batch 更新；
- 新增 `--save-student`，checkpoint 同时保存 `student_backbone` / `student_head`（约 700 MB/个）；
- 新增 `eval_checkpoints.py`，用下游线性 probe 选 checkpoint。

数据准备（先在服务器确认 42,616 张与来源分布）：

```bash
cd ~/MRG-05-Tooth-Seg/lab_train
python pretrain/prepare_pretrain_data.py --validate-images

python - <<'PY'
import json, collections
m = json.load(open("dataset/pretrain/pretrain_manifest.json"))
print("unique:", m["unique"], "corrupt:", len(m["corrupt"]))
print(collections.Counter(p["path"].split("/")[1] for p in m["images"]))
PY
```

训练：

```bash
cd lab_train

python -u pretrain/pretrain.py \
  --model weights/dinov3-vitb16 \
  --manifest dataset/pretrain/pretrain_manifest.json \
  --config pretrain/config.512-25g-42k-p0.json \
  --output output/pretrain/oraldino-vitb16-512-42k-p0 \
  --device cuda:0 \
  --batch 16 \
  --grad-accum 2 \
  2>&1 | tee output/pretrain/oraldino-vitb16-512-42k-p0.stdout.log
```

- 显存：batch 8 × grad-accum 4 = 有效 32，约 11 GB（25 GB 内）。
- 如果 GPU 吞吐偏低但显存有余，可试 `--batch 16 --grad-accum 2`（有效 32 不变，减少累积次数）；OOM 则退回 8×4。
- checkpoint：`--save-every 5000` + `--save-student` ≈ 8 个 × 700 MB ≈ 5.6 GB；`--save-every 10000` 可减半。

下游 probe（每 5,000 步的 checkpoint 都可以评）：

```bash
# 服务器上需要一份 141 张氟斑牙数据；不要把评测集放进 dataset/（会被预训练扫描）
python pretrain/eval_checkpoints.py \
  --run-dir output/pretrain/oraldino-vitb16-512-42k-p0 \
  --data-root ../eval/positive \
  --imgsz 512 --device cuda:0
```

- 默认评 teacher backbone；加 `--student` 评 student backbone。
- 输出 `probe_results.csv`，按 QWK 自动打印最佳 checkpoint。
- 如果 QWK 在某个 checkpoint 后连续下降，停止训练并用该 checkpoint；当前脚本不支持 resume，停止后会浪费最后一个 5k 区间。

**训练量参考（42,616 张、有效 batch 32）**

| epochs | steps | 说明 |
| ---: | ---: | --- |
| 10 | 13,320 | 快速验证 |
| 20 | 26,640 | 最小正式量 |
| **30** | **39,960** | **推荐首跑（40k）** |
| 50 | 66,600 | 若 probe 仍提升再延长 |

时间估算：先用 100 步测 `steps_per_sec`（写在 `metrics.csv`）。按 0.2–0.5 steps/s，40k 步约 **22–55 小时**。

**官方 DINOv3 recipe（当前推荐，替代 P0）**

完整步骤见 [`official_dinov3_patch/README.md`](official_dinov3_patch/README.md)。核心命令：

```bash
cd ~/MRG-05-Tooth-Seg/lab_train

# 1. 数据审计（来源/重复/损坏/灰度/评测泄漏）
python pretrain/audit_pretrain_data.py \
  --data-root dataset \
  --eval-root ../dataset/positive \
  --output output/data-audit

# 2. HF DINOv3 → 官方 teacher checkpoint（已验证 CLS 余弦 0.9999998）
python pretrain/convert_hf_dinov3_to_official.py \
  --hf-model weights/dinov3-vitb16 \
  --out weights/dinov3-vitb16-official.pth \
  --official-repo /opt/dinov3

# 3. 给官方 repo 打 ImageFolder + backbone-only resume 补丁
python pretrain/official_dinov3_patch/patch_official_dinov3.py --repo /opt/dinov3

# 4. 编辑 config_dental_vitb_512.yaml 里的两个 /PATH/TO/...，然后启动
torchrun --nproc_per_node=1 /opt/dinov3/dinov3/train/train.py \
  --config-file ~/MRG-05-Tooth-Seg/lab_train/pretrain/official_dinov3_patch/config_dental_vitb_512.yaml \
  --output-dir output/pretrain/official-dental-vitb-512 \
  --no-resume
```

关键差异（相对 P0）：

- 使用官方 **iBOT + KoLeo + Gram**，不是最小 DINO；
- 从 base DINOv3 继续预训练，不是从头；
- 512 全局 / 128 局部，batch 8，activation checkpointing；
- KoLeo 用 **0.05**（不是 0.1），并监控 pair cosine；
- 15 epochs 起步，每 1k–2k 迭代看下游 probe，早停；
- 数据审计后再训练；COde 若含 X 光，先只用 RGB 口内图。

#### 3.5 断点、日志、续训与注意事项

- **日志自动保存**：每次训练自动生成
  - `output/pretrain/<run>/metrics.csv`：逐 step 指标（`step,epoch,loss,lr,momentum,elapsed_sec,steps_per_sec,samples_per_sec,gpu_mem_mb`）；
  - `output/pretrain/<run>/train.log`：与控制台一致的文本日志。
  两者在每次写入后 `flush`，训练崩溃或中断时已写入的部分不会丢。`--log-every 10` 表示每 10 个优化步写一次；step 0、每次保存 checkpoint、最后一次都会写。
- `--save-every N` 会写 `ckpt_stepN.pt`（teacher backbone + head + center），最终写 `teacher_checkpoint.pt`。
- `--save-student` 会额外保存 `student_backbone` / `student_head`，checkpoint 约 **700 MB**（默认只存 teacher 时约 400 MB）。
- **当前 `pretrain.py` 没有实现 `--resume`**：中断后重新启动会从 step 0 开始，`ckpt_step*.pt` 仅用于下游评估/手动恢复，不能直接续训。长训练前请确认 `--max-steps`、显存和任务时限。
- 42k 配置：`--save-every 5000` + `--save-student` ≈ 8 个文件 × 700 MB ≈ 5.6 GB；磁盘紧张时改 `--save-every 10000`。
- `--num-workers` 默认 4；如果 DataLoader 报错或卡住（尤其是 macOS），用 `--num-workers 0` 排查。
- 用 `eval_checkpoints.py` 对保存的 checkpoint 做下游 probe；结果写入 `probe_results.csv`，按 QWK 选最佳 checkpoint。
- 若只需要最终骨干，可以在训练完成后只保留 `teacher_checkpoint.pt`，按需删除中间 `ckpt_step*.pt`。

### 4. 输入输出与产物

**端到端数据/产物流：**

```text
[输入] dataset/{AlphaDent,COde-Dataset,DentalAI,Images_Data_Set,
               Mendeley,OMNI_COCO,Oral-Diseases,segmentation-teeth}/**/*.jpg
        │  （图像直接放对应目录，脚本递归扫描）
        ▼
prepare_pretrain_data.py  ──► dataset/pretrain/pretrain_manifest.json
        │                       { base_dir, unique, seen, images:[{path,sha256,size}] }
        ▼
pretrain.py  ──► output/pretrain/<name>/
                     ├─ args.json               # 本次完整超参记录（可复现）
                     ├─ metrics.csv             # 逐 step 指标（loss/lr/epoch/速度/显存）
                     ├─ train.log               # 文本日志（与控制台一致）
                     ├─ ckpt_step<N>.pt         # teacher（+ student，若 --save-student）
                     ├─ teacher_checkpoint.pt   # 最终 OralDINO 骨干（下游用 backbone 字段）
                     └─ probe_results.csv       # eval_checkpoints.py 的下游 probe 结果
        │
        ▼  [下游 v16·A3]
线性评估 / LoRA / 冻结微调 —— 检测（YOLO/RT-DETR）与氟斑牙分级
```

- **`pretrain_manifest.json`**：`unique`（去重后唯一图数）、`seen`（扫描文件数）、`images`（逐图相对路径 + SHA256 + 大小）；路径相对本包根目录，整体拷贝到服务器后依然有效。
- **`args.json`**：包含 `effective_batch`（= `batch × grad_accum`）和 `dataset_size`，便于复现步数/epoch 换算。
- **`metrics.csv` / `train.log`**：训练一开始就创建，逐次 flush；用于画 loss 曲线、算吞吐、检查显存。
- **`teacher_checkpoint.pt`**：字典含 `backbone` / `head` / `center` / `step` / `model`，下游评估直接取 `backbone` 字段作为 OralDINO 初始化权重。
- **`--save-student`**：checkpoint 额外包含 `student_backbone` / `student_head`；可用 `eval_checkpoints.py --student` 对比 teacher/student。
- **`probe_results.csv`**：`eval_checkpoints.py` 对每个 checkpoint 输出的 accuracy / macro-F1 / QWK。

> 数据准备（放置图像、生成/校验清单）的完整命令见 **3.2**；这里只说明产物结构。

### 5. 安装预训练依赖

`setup_env.sh` 已支持 `--pretrain`（额外装 `transformers`，且不降级已装的 cu130 PyTorch）：

```bash
bash setup_env.sh --pretrain          # 或已装环境后： conda activate toothseg && pip install -e ".[pretrain]"
```

### 6. 注意事项

- **PyTorch ≥ 2.7.1**：官方 DINOv3 要求，且 Blackwell `sm_120` 依赖 cu130/cu128 wheel（本目录 `setup_env.sh` 已保证）；`transformers>=4.56.0`、`timm>=1.0.20`。
- **25 GB 显存 + 512 分辨率**：用 `pretrain/config.512-25g.json`（global 512 / local 128 / `batch 8 × grad_accum 4 = effective 32` / lr 6.25e-5）。不要把 `local_imgsz` 设成 512；显存不够时先改 `--batch 4 --grad-accum 8`。
- **P0 防坍缩**：42k 配置用 `out_dim 16384`、`temp_teacher 0.05`、`center_ema 0.99`、`koleo_weight 0.1`；center EMA 每个优化步更新一次。`--save-student` 会让 checkpoint 从 ~400 MB 增到 ~700 MB。
- **日志**：每次训练自动写 `metrics.csv` + `train.log`；建议同时用 `python -u ... 2>&1 | tee <run>/stdout.log` 保留完整控制台输出。
- **数据量**：当前唯一口内图约 6k 张（见根项目 A1 清单），对 65536 维 codebook 偏小，`--out-dim` 可调小、`--lr`/`--max-steps` 需按数据量校准；后续下载 COde / Oral-Diseases 后重跑数据清单并加大训练量，再评估是否升到 `vitl16`。
- **门控与 license**：权重仅限授权用途，不得商业化/临床部署；研究用请遵守 DINOv3 License 条款。

### 7. 常见错误排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `GatedRepoError: 403` / `Access denied ... requires approval` | HF 门控未获批 | 用 ModelScope 镜像（见 2.⑤），或用机构邮箱重新申请 |
| `401 Unauthorized` | `HF_TOKEN` 缺失/过期 | `hf login` 或更新 `HF_TOKEN` |
| `KeyError: 'dinov3_vit'` / `Unrecognized model` | `transformers` 版本过低 | 升级到 `>=4.56` |
| `Manifest has no images` | 清单为空或路径不对 | 确认图像在 `lab_train/dataset/<name>/`，重新生成清单 |
| `FileNotFoundError`（DataLoader 找不到图） | 清单路径与图像位置不一致 | 在 `lab_train/` 根目录运行；不要把清单和图像分开拷贝 |
| `OSError: image file is truncated` | 数据集中有截断/损坏 JPEG | 用 `prepare_pretrain_data.py --validate-images` 重新生成清单；运行时也会自动跳过并打印警告 |
| `CUDA out of memory` | batch/imgsz/local-crops 过大 | 先降 `--batch` 并提高 `--grad-accum`（保持有效 batch），再降 `--local-crops`/`--local-imgsz`，最后降 `--imgsz`；开启 `--amp` |
| 找不到 `metrics.csv` / `train.log` | 训练未启动或输出目录不同 | 脚本会自动写到 `<output>/`；确认 `--output` 路径和磁盘权限 |
| DataLoader 卡住/报错 | worker 进程问题（macOS 常见） | 先用 `--num-workers 0` 排查 |
| `pin_memory ... MPS` 警告 | CPU/MPS 环境 | 无害，可忽略 |
| `torch.cuda.amp` FutureWarning | PyTorch 新版弃用提示 | 无害；后续可迁移到 `torch.amp` |
| loss 为 NaN / 不下降 | lr 过大、AMP 不稳定、数据异常 | 降 lr、关 AMP 试跑、检查清单里是否有损坏图像 |
| `non-monotonic index ... ._pack-*.idx` | macOS 在外部卷生成 AppleDouble 文件 | 把 `.git` 下 `._*` 移出后重试 Git 操作 |

### 8. 权重与 License 说明

- DINOv3 权重受 **DINOv3 License** 约束；协议允许在相同条款下再分发，但必须附带协议副本（`LICENSE.md`）。
- 基于 DINO Materials 发表论文时，需在文中声明/致谢使用 DINOv3。
- 模型权重、预训练 checkpoint（每个约 400 MB）和 `lab_train/dataset/` 已在本项目 `.gitignore` 中排除，不要提交到 Git。
