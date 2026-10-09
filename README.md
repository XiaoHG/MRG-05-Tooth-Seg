# MRG-05-Tooth-Seg

面向口腔图像的单类别牙齿检测项目。当前主任务是使用 YOLO 格式边界框训练牙齿检测模型，输出检测框、可视化图像、检测 JSON 和单颗牙齿裁剪图。

当前项目不是像素级牙齿分割系统，也不输出 FDI 牙位编号或氟斑牙临床分级结果。

## 项目现状

- 检测类别只有 `tooth`，类别 ID 为 `0`。
- 标签格式为 `class x_center y_center width height`，坐标归一化到 `[0, 1]`。
- 当前训练入口基于 Ultralytics，支持通过 `--model` 选择 Ultralytics 模型，例如 `yolo11n.pt`、`yolo11s.pt` 和 `rtdetr-l.pt`。
- `--weights` 用于加载已有权重初始化训练；当同时提供 `--model` 和 `--weights` 时，已有权重优先。
- Faster R-CNN、标准 DETR 和完整 Mask R-CNN 尚未接入当前 CLI。Mask R-CNN 还需要真实实例分割掩码，不能用矩形框伪造掩码作为正式分割实验。
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

使用默认 YOLO11n：

```bash
python cli/main.py train --data dataset/OMNI_COCO/before --model yolo11n.pt --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/train --name omni-coco-after-yolo11n
```

使用同系列更大模型：

```bash
python cli/main.py train --data dataset/OMNI_COCO/before --model weights/yolo11s.pt --epochs 1 --imgsz 1024 --batch 1 --device 0 --project output/train --name omni-coco-before-yolo11s
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
  --data-yaml dataset/data.yaml
```

模型选择和调参使用 `val`。最终模型确定后，使用包含 `test` 路径和标签的标准数据视图进行一次独立测试评估。当前 CLI 的 `val` 命令默认调用 Ultralytics 的 `val` split；如需评估 `test`，应准备将 `test` 设为评估 split 的 YAML，或直接使用 Ultralytics API/命令指定 `split=test`。

报告中至少保存 precision、recall、mAP50、mAP50-95、训练耗时和推理速度。before/after 或不同模型的比较必须使用相同的图像、标签和评估协议。

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

- 当前核心实现仍是 Ultralytics 训练/推理流程，非 YOLO 架构尚未统一接入。
- Faster R-CNN 和标准 DETR 需要独立的数据适配器、训练循环、权重加载和评估接口。
- Mask R-CNN 需要人工实例掩码；当前边界框标签不足以支持可信的分割实验。
- `test` 标签可用于最终评估，但不能在模型选择阶段反复使用。
- 伪标签、无标签图像和推理可视化不能替代固定人工真值验证集。
- 项目是科研检测基线，不是临床诊断系统。
