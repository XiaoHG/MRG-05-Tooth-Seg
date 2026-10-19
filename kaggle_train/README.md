# Kaggle 训练启动器

`train.py` 是 Kaggle 训练入口。它会从 GitHub 拉取指定版本的项目代码，安装训练依赖，然后调用项目的 `cli/main.py train`。

当前项目是单类别 `tooth` 目标检测，不是像素级分割，也不输出 FDI 牙位或临床分级结果。Kaggle 启动器支持 Ultralytics、torchvision、Transformers 和 RF-DETR 原生后端。

## 数据集结构

启动器支持两种 prepared 数据集结构。

标准 YOLO 结构：

```text
dataset/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml                 # 可选
```

当前 OMNI 数据集结构：

```text
before/
  train/                    # 训练图像
  val/                      # 验证图像
  test/                     # 可选，无标签图像
  labels/train/             # 训练标签
  labels/val/               # 验证标签
  labels/test/              # 测试标签（如有）
```

通过 `--dataset` 传入数据集根目录即可。启动器会自动识别上述结构。对于第二种结构，脚本会将图像和标签复制到：

```text
/kaggle/working/toothseg-output/dataset-view/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

复制是必要的，因为 Ultralytics 会解析符号链接并根据 `images/` 推导 `labels/` 路径。原始 Kaggle Dataset 只读，不会被修改。随后启动器将数据根目录传给仓库 CLI 的 `--data` 参数，由 CLI 在训练实验目录下生成运行时 YAML；不依赖数据集内已有的 `data.yaml`。

## Kaggle Notebook 用法

建议直接从 GitHub 下载启动器，不需要把 `train.py` 上传到 Kaggle Dataset：

```python
!wget -q -O /kaggle/working/train.py \
  https://raw.githubusercontent.com/XiaoHG/MRG-05-Tooth-Seg/main/kaggle_train/train.py
```

先运行 1 个 epoch 验证路径、标签和 GPU：

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

冒烟测试成功后运行正式训练：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --name omni-coco-before
```

使用 YOLOv8m 或 YOLO26m 时，只需替换 `--model` 和实验名称；YOLO26m 需要当前安装的 Ultralytics 版本提供该模型：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --model yolov8m.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --name omni-coco-before-yolov8m
```

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after/before \
  --dataset-mode prepared \
  --model yolo26m.pt \
  --epochs 50 \
  --imgsz 1024 \
  --batch -1 \
  --device 0 \
  --name omni-coco-before-yolo26m
```

## 新增模型

`train.py` 会根据 `--model` 自动安装所需后端：Faster R-CNN 和 RetinaNet 使用 Kaggle 已安装/项目 `.[train]` 中的 torchvision；RT-DETRv2-L 和 RF-DETR-M 会额外安装 `.[external]`（Transformers、RF-DETR 和 YAML 依赖）。Kaggle Notebook 必须开启 Internet，RT-DETRv2-L 需要从 Hugging Face 下载模型，RF-DETR-M 建议提供官方预训练权重。

### Faster R-CNN 冒烟与正式训练

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model fasterrcnn_resnet50_fpn \
  --epochs 1 --imgsz 1024 --batch 1 --device 0 \
  --name smoke-fasterrcnn

!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model fasterrcnn_resnet50_fpn \
  --epochs 50 --imgsz 1024 --batch 2 --device 0 \
  --name omni-coco-before-fasterrcnn
```

### RetinaNet 冒烟与正式训练

将上面命令中的模型和名称替换为：

```text
--model retinanet_resnet50_fpn --name smoke-retinanet
--model retinanet_resnet50_fpn --name omni-coco-before-retinanet
```

冒烟测试使用 `--epochs 1 --batch 1`，正式训练使用 `--epochs 50 --batch 2`。

### RT-DETRv2-L 冒烟与正式训练

RT-DETRv2-L 通过 Transformers 原生接口训练。首次运行会下载 Hugging Face 权重；如使用本地或 Kaggle Dataset 中的模型目录，将其传给 `--weights`：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model rtdetrv2-l --epochs 1 --imgsz 1024 --batch 1 --device 0 \
  --name smoke-rtdetrv2l

!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model rtdetrv2-l --epochs 50 --imgsz 1024 --batch 1 --device 0 \
  --name omni-coco-before-rtdetrv2l
```

### RF-DETR-M 冒烟与正式训练

RF-DETR-M 会在实验输出目录复制出其所需的 `train/images`、`train/labels`、`valid/images`、`valid/labels` 和 `data.yaml`，不会修改 Kaggle 输入数据。建议先将官方 `rf-detr-medium.pth` 上传为 Kaggle Dataset，再通过 `--weights` 传入：

```python
!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model rf-detr-m --epochs 1 --imgsz 1024 --batch 1 --device 0 \
  --weights /kaggle/input/rf-detr-weights/rf-detr-medium.pth \
  --name smoke-rfdetrm

!python /kaggle/working/train.py \
  --repo-url https://github.com/XiaoHG/MRG-05-Tooth-Seg.git \
  --repo-ref main \
  --dataset /kaggle/input/datasets/hgxiao/omni-coco-before-after-v4/before \
  --dataset-mode prepared \
  --model rf-detr-m --epochs 50 --imgsz 1024 --batch 1 --device 0 \
  --weights /kaggle/input/rf-detr-weights/rf-detr-medium.pth \
  --name omni-coco-before-rfdetrm
```

运行日志中应看到训练数据位于类似以下的运行时视图：

```text
/kaggle/working/toothseg-output/train/smoke-test/.runtime/dataset-view/images/train
```

如果日志仍显示直接读取 `/kaggle/input/.../before/train`，说明 Notebook 使用了旧版启动器，需要重新执行上面的 `wget`。

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--repo-url` | GitHub 仓库地址 |
| `--repo-ref` | branch、tag 或 commit；正式实验建议使用 commit |
| `--dataset` | Kaggle Dataset 中传入的实际数据根目录 |
| `--dataset-mode prepared` | 使用已有 train/val 划分，不重新划分数据 |
| `--model` | 模型名，例如 `yolo11n.pt`、`yolov8m.pt`、`yolo26m.pt`、`rtdetr-l.pt`、`fasterrcnn_resnet50_fpn`、`retinanet_resnet50_fpn`、`rtdetrv2-l`、`rf-detr-m` |
| `--epochs` | 训练轮数 |
| `--imgsz` | 输入尺寸；显存不足时可改为 640 |
| `--batch` | `-1` 自动选择，显存不足时可改为 1 或 2 |
| `--device` | Kaggle GPU 通常使用 `0` |
| `--weights` | 可选的初始 `.pt` 权重；不传则使用 YOLO11n 预训练权重 |
| `--name` | 输出实验名称，每次实验建议使用不同名称 |

## 输出位置

```text
/kaggle/working/toothseg-output/
  train/<experiment-name>/
    .runtime/
      dataset-view/
        images/train/
        images/val/
        labels/train/
        labels/val/
      dataset.yaml
    weights/best.pt
    weights/last.pt
    args.yaml
    results.csv
```

训练完成后，请从 Kaggle Output 下载整个实验目录，而不只是 `best.pt`，同时保留数据集版本、Git commit、训练参数和验证结果。

## 注意事项

- Kaggle Notebook 需要开启 Internet，脚本要从 GitHub clone 项目并按模型安装对应训练依赖。
- `/kaggle/input` 是只读目录，训练输出写入 `/kaggle/working`。
- 当前 `test` 图像配有 `labels/test`，应在模型和参数确定后作为独立最终评估集；不要在调参阶段反复使用 test 指标。
- `best.pt` 用于初始化新的训练实验；中断恢复应使用 Ultralytics 的 `last.pt` 恢复机制，不能混淆两者语义。
- RT-DETRv2-L 和 RF-DETR-M 依赖原生包及匹配权重；首次运行应先执行 1 epoch 冒烟测试，再开始正式训练。
- `--weights` 对 RF-DETR-M 应指向官方 `.pth` 权重；RT-DETRv2-L 可指向 Hugging Face 模型目录或权重来源。
