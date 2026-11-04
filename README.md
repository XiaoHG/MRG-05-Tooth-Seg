# MRG-05-Tooth-Seg

## 模型初始化与预训练权重

当前四个新增检测模型的默认初始化方式并不一致：

| 模型 | 默认初始化方式 | 说明 |
| --- | --- | --- |
| Faster R-CNN (`fasterrcnn_resnet50_fpn`) | 从头初始化 | torchvision 构造时使用 `weights=None` 和 `weights_backbone=None`，不会自动下载预训练权重。 |
| RetinaNet (`retinanet_resnet50_fpn`) | 从头初始化 | torchvision 构造时使用 `weights=None` 和 `weights_backbone=None`，不会自动下载预训练权重。 |
| RT-DETRv2-L (`rtdetrv2-l`) | 使用预训练参数 | 未传 `--weights` 时，默认从 Hugging Face 加载 `PekingU/rtdetr_v2_r50vd`，再在本项目数据上训练。 |
| RF-DETR-M (`rf-detr-m`) | 默认从头初始化 | 未传 `--weights` 时，适配器传入 `pretrain_weights=None`。建议正式实验显式提供官方 `rf-detr-medium.pth`。 |

如果训练命令显式传入 `--weights`，则会使用该权重初始化模型。对于 Faster R-CNN、RetinaNet 和 RF-DETR-M，这会把默认的从头初始化改为基于指定权重训练；对于 RT-DETRv2-L，则会从指定的本地权重目录或模型来源加载。这里的“预训练”仅表示模型参数初始化方式，不代表模型已经在本项目的牙齿数据上训练完成。

上述行为对应的实现位于 `src/toothseg/torchvision_adapter.py` 和 `src/toothseg/external_adapter.py`。例如，现有 RF-DETR-M 冒烟实验的 `training_config.json` 中 `pretrain_weights` 为 `null`，说明该次实验没有使用外部预训练权重。

面向口腔图像的单类别牙齿检测项目。当前主任务是使用 YOLO 格式边界框训练牙齿检测模型，输出检测框、可视化图像、检测 JSON 和单颗牙齿裁剪图。

当前项目不是像素级牙齿分割系统，也不输出 FDI 牙位编号或氟斑牙临床分级结果。

## 项目现状

- 检测类别只有 `tooth`，类别 ID 为 `0`。
- 标签格式为 `class x_center y_center width height`，坐标归一化到 `[0, 1]`。
- 当前训练入口基于 Ultralytics，支持通过 `--model` 选择已适配模型，例如 `yolo11n.pt`、`yolo11s.pt`、`yolov8m.pt`、`yolo26m.pt` 和 `rtdetr-l.pt`。
- `--weights` 用于加载已有权重初始化训练；当同时提供 `--model` 和 `--weights` 时，已有权重优先。
- Faster R-CNN、RetinaNet、RT-DETRv2-L 和 RF-DETR-M 已接入独立后端：前两者使用 torchvision 检测适配器，RT-DETRv2-L 使用 Transformers 原生模型接口，RF-DETR-M 使用 RF-DETR 原生包接口。四类模型均使用独立的数据读取、训练、权重和验证入口，不会被错误地当作 Ultralytics 模型运行。Mask R-CNN 还需要真实实例分割掩码，不能用矩形框伪造掩码作为正式分割实验。
- Kaggle 启动器支持当前 OMNI 数据目录，并会在工作目录生成标准 YOLO 数据视图。

## 数据集

当前主数据集位于 `dataset/OMNI_COCO/`，包含 `before` 和 `after` 两个数据版本：

```text
dataset/OMNI_COCO/
  before/
    train/              # 2481 张图像
    val/                # 857 张图像
    test/               # 828 张图像
    labels/train/       # 2481 个标签
    labels/val/         # 857 个标签
    labels/test/        # 828 个标签
  after/
    train/
    val/
    test/
    labels/train/
    labels/val/
    labels/test/
  classes.txt
  data.yaml
  dataset_manifest.json
  split_manifest.json
  notes.json
```

`before` 和 `after` 的 train、val、test 图像与标签均按文件 stem 一一对应。两个版本各有 2481 个训练样本、857 个验证样本和 828 个测试样本。`test` 是有标签的独立评估集，应在模型和参数确定后使用，不应在调参阶段反复使用其指标。

`before/train` 中不得包含 Windows 快捷方式、压缩包或其他非图像文件。数据打包和上传前应检查图像与标签 stem 集合完全一致。

### 标准 YOLO 数据视图

Ultralytics 期望的标准结构是：

```text
dataset-view/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
  data.yaml
```

OMNI 数据版本使用 `train/`、`val/`、`test/` 加 `labels/` 的布局。Kaggle 启动器会把它复制为上述标准视图；不要通过符号链接代替复制，因为 Ultralytics 解析符号链接后可能丢失 `images/labels` 的路径对应关系。

原始 `dataset/raw/` 已删除，不应再根据旧文档执行 raw 数据构建命令。

## 安装

基础依赖：

```bash
pip install -e .
```

训练、验证和推理依赖：

```bash
pip install -e ".[train]"
```

RT-DETRv2-L 的 Transformers 后端和 RF-DETR-M 的原生后端：

```bash
pip install -e ".[external]"
pip install rfdetr
```

开发测试依赖：

```bash
pip install -e ".[dev]"
```

## 数据校验

对于标准 `images/`、`labels/` 数据视图，可运行：

```bash
python cli/main.py validate --data <dataset-view>
```

校验内容包括类别文件、图像/标签 stem、YOLO 标签字段数量、类别 ID 和归一化坐标范围。校验不会修改图像或标签。

对 OMNI 的 `before` 或 `after` 目录，优先使用 Kaggle 启动器生成标准视图后再训练；也可以在本地按相同规则准备 `images/` 和 `labels/` 目录。

## 本地训练

查看模型目录及当前后端适配状态：

```bash
python cli/main.py models
```

使用默认 YOLO11n：

```bash
python cli/main.py train --data dataset/OMNI_COCO/before --model yolo11n.pt --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name omni-coco-before-yolo11n
```

使用同系列更大模型：

```bash
python cli/main.py train --data dataset/OMNI_COCO/before --model weights/yolo11s.pt --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/train --name omni-coco-before-yolo11s
```

使用 YOLOv8m：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model yolov8m.pt --epochs 1 --imgsz 1024 --batch -1 --device 0 --project output/smoke --name omni-coco-after-yolov8m

python cli/main.py train \
  --data dataset/OMNI_COCO/after \
  --model yolov8m.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --project output/train \
  --name omni-coco-after-yolov8m
```

使用 YOLO26m（需要当前安装的 Ultralytics 版本提供该模型）：

```bash
python cli/main.py train \
  --data dataset/OMNI_COCO/after \
  --model yolo26m.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --project output/train \
  --name omni-coco-after-yolo26m
```

### Faster R-CNN / RetinaNet 冒烟测试

Faster R-CNN 和 RetinaNet 使用 torchvision 后端。安装 `.[train]` 后，先用 1 个 epoch 验证数据读取、模型构造、反向传播、权重保存和验证流程：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model fasterrcnn_resnet50_fpn --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name omni-coco-after-fasterrcnn-smoke

python cli/main.py train --data dataset/OMNI_COCO/after --model retinanet_resnet50_fpn --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name omni-coco-after-retinanet-smoke
```

冒烟测试通过后执行完整训练：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model fasterrcnn_resnet50_fpn --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-fasterrcnn

python cli/main.py train --data dataset/OMNI_COCO/after --model retinanet_resnet50_fpn --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-retinanet
```

两类 torchvision 模型的权重为项目统一 checkpoint，包含模型名称和 `state_dict`；验证时使用同一命令入口：

```bash
python cli/main.py val --weights output/train/omni-coco-after-fasterrcnn/best.pt --data dataset/OMNI_COCO/after
```

### RT-DETRv2-L / RF-DETR-M 冒烟测试与训练

RT-DETRv2-L 和 RF-DETR-M 通过各自原生包适配。RT-DETRv2-L 需要安装 `transformers` 并提供可用的 RT-DETRv2 checkpoint；RF-DETR-M 需要安装 `rfdetr`。由于两个原生包的模型构造和输出格式并不相同，CLI 会调用其原生 `train`、`validate/val` 和 `predict` 接口，不会将其权重转交给 Ultralytics。

RF-DETR-M 训练启动时会在实验输出目录生成独立的数据适配视图（`train/images`、`train/labels`、`valid/images`、`valid/labels` 和 `data.yaml`），用于满足 RF-DETR 的数据格式检测；该过程只复制数据，不修改原始 `dataset/` 内容。

先执行 1 个 epoch 冒烟测试，确认原生依赖、权重和数据接口可用：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model rtdetrv2-l --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name omni-coco-after-rtdetrv2l-smoke

python cli/main.py train --data dataset/OMNI_COCO/after --model rf-detr-m --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/smoke --name omni-coco-after-rfdetrm-smoke
```

冒烟测试通过后执行完整训练：

```bash
python cli/main.py train --data dataset/OMNI_COCO/after --model rtdetrv2-l --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-rtdetrv2l

python cli/main.py train --data dataset/OMNI_COCO/after --model rf-detr-m --epochs 50 --imgsz 1024 --batch 2 --device 0 --project output/train --name omni-coco-after-rfdetrm
```

使用 RT-DETR（需要当前 Ultralytics 版本提供对应权重）：

```bash
python cli/main.py train \
  --data dataset/OMNI_COCO/before \
  --model rtdetr-l.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --project output/train \
  --name omni-coco-after-rtdetr
```

用已有权重初始化新实验：

```bash
python cli/main.py train \
  --data dataset/OMNI_COCO/before \
  --model yolo11s.pt \
  --weights weights/yolo11n.pt \
  --epochs 50 \
  --project output/train \
  --name omni-coco-after-finetune
```

`--batch -1` 让 Ultralytics 自动选择 batch。显存不足时，统一降低 `--imgsz` 或显式设置较小的 batch；不同模型的实验应记录这些差异。

## 已有训练结果

当前已完成 `YOLO11n` 在 `before` 和 `after` 数据版本上的 50 epoch 实验，配置为 `imgsz=1024`、自动 batch、CUDA 和 AMP。结果目录分别为：

```text
output/train/omni-coco-before-result/toothseg-output/train/omni-coco-before/
output/train/omni-coco-after-result/toothseg-output/train/omni-coco-after/
```

验证集最佳结果如下：

| 数据版本 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| before | 0.94390 | 0.94688 | 0.96963 | 0.61072 |
| after | 0.96197 | 0.95363 | 0.98193 | 0.62756 |

在当前验证集上，`after` 的四项指标均高于 `before`。这只是验证集对比，最终结论仍应在两个版本统一、固定且带人工标签的 test 集上重新评估。

## 验证与测试评估

验证训练期间使用的 `val` 集：

```bash
python cli/main.py val \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --data dataset/OMNI_COCO/after
```

在固定、独立、带人工标签的 `test` 集上评估（模型和参数确定后使用）：

```bash
python cli/main.py val \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --data dataset/OMNI_COCO/after \
  --split test
```

`--split` 支持 `train` / `val` / `test`，默认 `val`。模型选择和调参使用 `val`，最终模型确定后再用 `--split test` 做一次独立测试评估。该参数透传到三个后端：Ultralytics（`model.val(split=...)`）、torchvision（`evaluate_torchvision_detection(..., split, ...)`）和 RT-DETRv2-L（`_ExternalYoloDataset(data_yaml, split)`）。注意 RF-DETR-M 的原生 `validate` 读取自身物化的 `valid` 视图，不接受 `split` 参数，因此 `--split` 对 RF-DETR-M 无效。

在固定 `test` 集上批量评估全部对比模型并汇总指标：

```bash
python tools/eval_test_set.py --split test
```

脚本按 `tools/eval_test_set.py` 内建的模型清单（8 个正式对比模型）依次评估，结果写入 `output/val-test/results.csv` 和 `README.md`。RF-DETR-M 因原生 `validate` 不支持 `--split` 已从对比清单移除；也可用 `--models yolo11n yolo11s` 只跑子集。

报告中至少保存 precision、recall、mAP50、mAP50-95、训练耗时和推理速度。before/after 或不同模型的比较必须使用相同的图像、标签和评估协议。

## 困难子集语义标注（v13 任务二/三）

在固定 `test` 集上自动初筛困难样本，并人工标注误检/漏检的语义原因（高光 / 金属 / 遮挡 / 门牙 / 其他）：

```bash
# 1. 自动初筛：按 FP+FN 排序，产出 screen.csv / errors.json（RT-DETRv2-L 检测）
python tools/hard_subset_screen.py \
  --weights output/train/omni_coco/rtdetrv2-l/omni-coco-after-rtdetrv2l \
  --split test

# 2. 生成人工标注工作表（CSV + MD，语义列留空待填）
python tools/build_annotation_worksheet.py

# 3. 生成内嵌 overlay 缩略图的 Excel 工作表（在此表上人工标注）
python tools/build_annotation_xlsx.py

# 4. 人工填完 annotation_worksheet.xlsx 后，回写语义列到 CSV
python tools/sync_annotation_xlsx.py

# 5. 汇总误检类型占比
python tools/analyze_annotation.py

# 6. 汇总下游检测基线报告（论文 A·A0）
python tools/build_downstream_report.py
```

第 5 步读取 `output/hard-subset/annotation_worksheet.csv` 中已填写的 `高光 / 金属 / 遮挡 / 门牙 / 其他` 列，输出 `annotation_summary.md` 的逐类型图数/占比与按子集矩阵，作为任务三「误检类型占比」的输入。语义标签需人工对照 overlay 图像（绿色 GT 框、红色预测框）逐张确认，脚本只汇总、不判定语义原因。

第 6 步把固定 test 集下游基线指标（YOLO11 + RT-DETR-L / RT-DETRv2-L）、困难子集每图 FP/FN、困难/易对比与误检类型占比组装为一份报告，写入 `output/downstream-baseline/README.md`，收口 v13 任务三。DN-DETR / DINOv3 等论文级强基线待 A2 预训练阶段接入（见 `draft/README.md`）。

## 多源牙科数据整合（v14 / 论文 A·A1）

A1 产出两样东西：多源公开牙科数据集清单 + 噪声图剔除规则。前者盘点 AlphaDent、COde、Oral-Diseases、OralSDv1、DentalAI 等公开集合并核对本地已有数据；后者按「黑屏 / 过曝 / 模糊 / 无牙」规则剔除无语义噪声图，为 A2 域内预训练准备干净数据。

```bash
# 1. 生成多源数据集清单（registry + 本地自动扫描 + 缺口分析）
python tools/build_dataset_inventory.py

# 2. 噪声图剔除（默认 dry-run，只写报告、不移动文件）
python tools/filter_noise_images.py --images dataset/positive --recursive

# 2a. 加入模型层：用 yolo11n 检测，无牙/仅低置信度检出 1–2 颗即剔除
#（模型推理较慢，建议放 Kaggle/Colab 跑；--weights 传 best.pt）
python tools/filter_noise_images.py \
  --images dataset/omni_coco-alphadent/images/train \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt

# 2b. 确认后执行隔离（把命中图移入 output/a1-data-integration/noise_quarantine/，不删除）
python tools/filter_noise_images.py --images dataset/positive --recursive --apply
```

产物写入 `output/a1-data-integration/`：

- `dataset_inventory.md` / `dataset_inventory.json` — 数据集清单（来源/许可/模态/任务/标注/规模/本地状态/本地数量 + 缺口分析）
- `noise_filter_report.csv` — 逐图特征（亮度/高光占比/清晰度/检出数/置信度）+ 保留/剔除判定与原因
- `README.md` — 噪声剔除规则与原因分布汇总

静态阈值（`--min-brightness`、`--max-highlight-ratio`、`--min-sharpness`）为起始默认值，需在人工标注样本上校准后再用于全量数据；模型层默认 `--min-teeth 2 --min-confidence 0.5`，对应 `draft/README.md` 的「检测不到牙齿或仅低置信度检出 1–2 颗即剔除」。脚本拒绝运行于 `dataset/raw/`，且 `--apply` 只隔离、不删除。

## 域内预训练（v15 / 论文 A·A2，lab_train）

DINOv3 口内照片域内 continue pretraining（OralDINO）在 `lab_train/` 服务器包内完成（RTX PRO 6000 Blackwell / cu130），不在本地跑：

- 环境与命令：`lab_train/README.md`「七、DINOv3 域内预训练（v15 / 论文 A·A2）」
- 数据准备：`lab_train/pretrain/prepare_pretrain_data.py`（跨目录 SHA256 去重 → `pretrain_manifest.json`）
- 训练：`lab_train/pretrain/pretrain.py`（student/teacher EMA + 多裁剪 + 中心化，精简 DINO 参考实现）

自监督预训练只用图像不看标签；DINOv3 权重在 HuggingFace 门控，需 `HF_TOKEN`。`kaggle_train/`、`colab_train/` 不做预训练适配（预训练固定在 lab_train 的 96 GB 卡上）。

## 推理

单张图像：

```bash
python cli/main.py predict \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --image dataset/OMNI_COCO/after/test/example.jpg \
  --output output/predict \
  --device 0
```

目录批量推理：

```bash
python cli/main.py predict-dir \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --image-dir dataset/OMNI_COCO/after/test \
  --output output/predict \
  --conf 0.25 \
  --device 0

python cli/main.py predict-visualize --weights output/train/yolo11n/omni-coco-after-result/toothseg-output/train/omni-coco-after-v2/weights/best.pt --image-dir F:/tooth_dataset/Oral/Images_Data_Set/post-processing/images --output F:/tooth_dataset/Oral/Images_Data_Set/post-processing/predict --conf 0.25 --device 0
```

聚合可视化推理：

```bash
python cli/main.py predict-visualize \
  --weights output/train/omni-coco-after-yolo11n/weights/best.pt \
  --image-dir dataset/OMNI_COCO/after/test \
  --output output/predict-after-yolo11n \
  --conf 0.25 \
  --device 0
```

`predict` 和 `predict-dir` 为每张图像生成原图、检测框叠加图、检测 JSON 和牙齿裁剪图；目录推理还会在输出根目录生成汇总的 `predictions.json`。`predict-visualize` 生成扁平化叠加图和聚合的 `predictions.json`。每条检测记录包含 `class_id`、`class_name`、`confidence`、像素坐标 `box_xyxy` 和与训练标签对应的归一化 `box_normalized_xywh`，便于和标注及不同模型结果进行对比。

### 将可视化推理结果导入 Label Studio 复核

`prediction-to-labelstudio` 读取 `predict-visualize` 生成的聚合 `predictions.json`，拷贝其中引用的原图，并将单类别 `tooth` 检测框转换为 Label Studio 可编辑的 `predictions`。它不会使用 `_overlay.png` 叠加图，也不会修改推理目录中的原图或 JSON。目标目录非空时默认拒绝执行，只有明确传入 `--overwrite` 才会重建目标目录。

```powershell
python cli/main.py prediction-to-labelstudio `
  --predict output/predict/dental_fluorosis/yolo11n `
  --output output/predict/dental_fluorosis/labelstudio/before
```

输出包括 `images/` 原图副本、`tasks.json`、`label_config.xml`、`manifest.json` 和导入说明 `README.md`。本机 Label Studio 启动前，将 `LOCAL_FILES_SERVING_ENABLED` 和 `LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED` 都设为 `true`，将 `LOCAL_FILES_DOCUMENT_ROOT` 和 `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT` 都指向上述 `before` 目录，然后关闭全部旧进程并重新启动 Label Studio。项目使用 `label_config.xml`，把 `before/images` 添加为 Local Files source storage，但不要执行同步，然后导入 `tasks.json`。已有框只是待人工复核的牙齿检测预测，不是像素级分割、FDI 牙位、氟斑牙分级或人工真值。

### 可视化人工复核后的 YOLO 检测框

Label Studio 导出的 `images/`、`labels/` 和 `classes.txt` 可通过 `visualize-labels` 生成黄色检测框叠加图。命令只读取原图和标签，在单独的输出目录写入派生图片与哈希清单，不会覆盖源文件；输出目录非空时默认拒绝执行。

```powershell
python cli/main.py visualize-labels `
  --data output/predict/dental_fluorosis/labelstudio/after `
  --output output/predict/dental_fluorosis/labelstudio/after/visualization
```

生成的 `visualization/manifest.json` 记录图像数、检测框数、源图和源标签 SHA-256，可用于确认可视化前后的来源未被修改。这里展示的是人工复核后的单类别牙齿边界框，不是像素级分割掩码或氟斑牙分级结果。

批量推理遇到无法解码的图片时会跳过该图片并在聚合 JSON 中记录 `error`，不会中断剩余任务。`--device 0` 才会显式使用第一块 CUDA GPU；图片读取、绘制和保存仍由 CPU 完成。

## Kaggle 训练

Kaggle 训练文件位于 `kaggle_train/train.py`。建议从 GitHub 下载最新版本，而不是继续使用 Kaggle Dataset 中的旧副本：

```python
!wget -q -O /kaggle/working/train.py \
  https://raw.githubusercontent.com/XiaoHG/MRG-05-Tooth-Seg/main/kaggle_train/train.py
```

先执行 1 epoch 冒烟测试：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --model yolo11n.pt \
  --epochs 1 \
  --imgsz 640 \
  --batch 1 \
  --device 0 \
  --name smoke-test
```

## Colab 训练

Google Colab 训练文件位于 `colab_train/train.py`，Notebook 位于 `colab_train/omni-coco-tooth-train.ipynb`。Colab 版本支持 `/content` 和 Google Drive 路径，长期保留的输出建议写入 Drive：

```python
from google.colab import drive
drive.mount('/content/drive')
```

下载并运行 Colab 启动器：

```python
!wget -q -O /content/train.py \\
  https://raw.githubusercontent.com/XiaoHG/MRG-05-Tooth-Seg/main/colab_train/train.py

!python /content/train.py \\
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \\
  --repo-ref main \\
  --dataset /content/drive/MyDrive/tooth-dataset/after \\
  --dataset-mode prepared \\
  --model yolo11n.pt \\
  --epochs 1 \\
  --imgsz 640 \\
  --batch 1 \\
  --device 0 \\
  --name smoke-test
```

冒烟测试通过后再改用正式训练参数。使用 `--output-dir /content/drive/MyDrive/toothseg-output` 可直接保存实验结果；也支持 `--weights`、`yolov8m.pt`、`yolo26m.pt` 和 `rtdetr-l.pt`。

正式训练只需将 `--epochs`、`--imgsz`、`--batch` 和 `--name` 改为正式配置。训练输出位于：

```text
/kaggle/working/toothseg-output/
  dataset-view/
  dataset.yaml
  train/<experiment-name>/
```

Kaggle 需要开启 Internet，以便 clone GitHub 仓库和安装训练依赖；`/kaggle/input` 是只读目录，输出必须写入 `/kaggle/working`。

## 权重与实验产物

公开预训练权重保存在 `weights/`，新增权重应记录模型名称、来源、版本、许可证和 SHA256。当前目录可能包含：

```text
weights/
  yolo11n.pt
  yolo26n.pt
```

训练输出应按数据版本、模型和实验名称区分，例如：

```text
output/train/
  omni-coco-before-yolo11n/
  omni-coco-after-yolo11n/
  omni-coco-after-rtdetr/
```

每个实验至少保留 `args.yaml`、`results.csv`、`weights/best.pt`、`weights/last.pt`、验证图和训练日志。不要覆盖已完成实验，也不要把模型权重、数据清单和验证结果分开保存。

## 测试

```bash
python -m pytest -q
```

测试重点覆盖数据准备、预测输出、标签导出和伪标签流程。真实模型训练、GPU 性能和临床效果不属于单元测试范围。

## 当前限制与后续计划

- 当前项目同时支持 Ultralytics、torchvision、Transformers 和 RF-DETR 原生后端；已适配 YOLO11、YOLOv8m、YOLO26m、RT-DETR-L、Faster R-CNN、RetinaNet、RT-DETRv2-L 和 RF-DETR-M。
- 四类非 Ultralytics 模型需要匹配的可选依赖和原生权重。缺少依赖时，CLI 会给出明确安装提示，不会静默回退到其他模型架构。
- Mask R-CNN 需要真实实例分割掩码；当前矩形框标签不足以支持正式分割实验。
- `test` 标签可用于最终评估，但不能在模型选择阶段反复使用。
- 伪标签、无标签图像和推理可视化不能替代固定人工真值验证集。
- 项目是科研检测基线，不是临床诊断系统。
