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
| GPU | NVIDIA RTX PRO 6000 Blackwell，约 96 GB 显存（97887 MiB） |
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

本节按四块展开：**① 版本选择分析 → ② 权重下载流程 → ③ 架构与训练流程 → ④ 输入输出与产物**。

本目录内置了 `pretrain/` 包：

```text
pretrain/
  prepare_pretrain_data.py   # 扫描图像目录 → SHA256 去重 → pretrain_manifest.json
  pretrain.py                # 单机 DINO continue-pretraining 训练循环
  config.example.json        # 超参参考（model/imgsz/batch/lr/EMA 温度等）
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
- **地区限制**：DINOv3 License 对受制裁司法管辖区不开放，若所在网络无法直连，可用镜像端点 `export HF_ENDPOINT=https://hf-mirror.com`（并配合对应 token）尝试；最终能否下载以 Meta 授权为准。

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

**启动命令**

```bash
HF_TOKEN="hf_..." python pretrain/pretrain.py \
  --config pretrain/config.example.json \
  --manifest dataset/pretrain/pretrain_manifest.json \
  --output output/pretrain/oraldino-vitb
```

关键超参（`config.example.json` 或 CLI 覆盖）：`--model facebook/dinov3-vitb16-pretrain-lvd1689m`（或本地 `weights/dinov3-vitb16`）、`--imgsz 224`（显存富余可提 518）、`--batch 32`（96 GB 可加大）、`--lr 1e-4`、`--max-steps`。

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
                     ├─ ckpt_step<N>.pt         # teacher backbone+head+center 定期断点
                     └─ teacher_checkpoint.pt   # 最终 OralDINO 骨干（下游用 backbone 字段）
        │
        ▼  [下游 v16·A3]
线性评估 / LoRA / 冻结微调 —— 检测（YOLO/RT-DETR）与氟斑牙分级
```

- **`pretrain_manifest.json`**：`unique`（去重后唯一图数）、`seen`（扫描文件数）、`images`（逐图相对路径 + SHA256 + 大小）；路径相对本包根目录，整体拷贝到服务器后依然有效。
- **`teacher_checkpoint.pt`**：字典含 `backbone` / `head` / `center` / `step` / `model`，下游评估直接取 `backbone` 字段作为 OralDINO 初始化权重。

**准备数据命令**

```bash
# 默认自动发现 dataset/ 下所有数据集目录（除 pretrain/），跨目录 SHA256 去重
python pretrain/prepare_pretrain_data.py

# 也可显式指定只扫部分数据集目录：
python pretrain/prepare_pretrain_data.py \
  --images dataset/AlphaDent dataset/COde-Dataset dataset/OMNI_COCO \
  --output dataset/pretrain/pretrain_manifest.json
```

### 5. 安装预训练依赖

`setup_env.sh` 已支持 `--pretrain`（额外装 `transformers`，且不降级已装的 cu130 PyTorch）：

```bash
bash setup_env.sh --pretrain          # 或已装环境后： conda activate toothseg && pip install -e ".[pretrain]"
```

### 6. 注意事项

- **PyTorch ≥ 2.7.1**：官方 DINOv3 要求，且 Blackwell `sm_120` 依赖 cu130/cu128 wheel（本目录 `setup_env.sh` 已保证）；`transformers>=4.56.0`、`timm>=1.0.20`。
- **数据量**：当前唯一口内图约 6k 张（见根项目 A1 清单），对 65536 维 codebook 偏小，`--out-dim` 可调小、`--lr`/`--max-steps` 需按数据量校准；后续下载 COde / Oral-Diseases 后重跑数据清单并加大训练量，再评估是否升到 `vitl16`。
- **门控与 license**：权重仅限授权用途，不得商业化/临床部署；研究用请遵守 DINOv3 License 条款。
